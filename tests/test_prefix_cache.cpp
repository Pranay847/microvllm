// Prefix caching: sharing KV cache between requests with a common prompt prefix.
//
// The claim that matters is not "blocks were refcounted" -- that is bookkeeping and would
// pass even if the backend recomputed every prompt. It is "prefill work was avoided". The
// mock engine counts tokens actually submitted to decode() versus tokens copied, so these
// tests assert the saving is real.
#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "microvllm/block_allocator.hpp"
#include "microvllm/mock_engine.hpp"
#include "microvllm/scheduler.hpp"

namespace microvllm {
namespace {

Request make_req(RequestId id, std::string prompt, std::uint32_t max_tokens = 8) {
    Request r;
    r.id              = id;
    r.spec.prompt     = std::move(prompt);
    r.spec.max_tokens = max_tokens;
    r.cancel          = std::make_shared<std::atomic<bool>>(false);
    return r;
}

// --- PrefixCache unit tests -------------------------------------------------------

TEST(PrefixCache, HashIsStableAndPrefixSensitive) {
    const std::vector<Token> a = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<Token> b = {1, 2, 3, 4, 9, 9, 9, 9};

    EXPECT_EQ(PrefixCache::hash_prefix(a, 4), PrefixCache::hash_prefix(b, 4))
        << "identical prefixes must hash identically";
    EXPECT_NE(PrefixCache::hash_prefix(a, 8), PrefixCache::hash_prefix(b, 8))
        << "diverging prefixes must not collide";
    EXPECT_NE(PrefixCache::hash_prefix(a, 4), PrefixCache::hash_prefix(a, 8))
        << "length must be part of the identity";
}

TEST(PrefixCache, FindsTheLongestBlockAlignedMatch) {
    PrefixCache cache;
    const std::vector<Token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    constexpr std::uint32_t  kBlock = 4;

    PrefixCache::Entry e4{.seq = 7, .n_tokens = 4, .blocks = {0}};
    cache.insert(PrefixCache::hash_prefix(tokens, 4), e4);
    PrefixCache::Entry e8{.seq = 9, .n_tokens = 8, .blocks = {0, 1}};
    cache.insert(PrefixCache::hash_prefix(tokens, 8), e8);

    const auto hit = cache.find_longest(tokens, kBlock);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->n_tokens, 8U) << "the longest match saves the most prefill";
    EXPECT_EQ(hit->seq, 9);
}

TEST(PrefixCache, PartialBlocksAreNeverShared) {
    // A sequence must be free to write the rest of its last block, so only whole blocks
    // may be handed to another sequence.
    PrefixCache cache;
    const std::vector<Token> tokens = {1, 2, 3, 4, 5};
    cache.insert(PrefixCache::hash_prefix(tokens, 5),
                 PrefixCache::Entry{.seq = 1, .n_tokens = 5, .blocks = {0}});

    EXPECT_FALSE(cache.find_longest(tokens, /*block_size=*/4).has_value())
        << "a 5-token entry is not block-aligned at block_size 4 and must not match";
}

TEST(PrefixCache, MissOnUnrelatedPrompt) {
    PrefixCache cache;
    const std::vector<Token> cached = {1, 2, 3, 4};
    cache.insert(PrefixCache::hash_prefix(cached, 4),
                 PrefixCache::Entry{.seq = 1, .n_tokens = 4, .blocks = {0}});

    const std::vector<Token> other = {9, 9, 9, 9};
    EXPECT_FALSE(cache.find_longest(other, 4).has_value());
}

TEST(PrefixCache, EvictingASequenceRemovesItsEntries) {
    // An entry names a live sequence; once that sequence's KV is gone the entry would
    // point at a recycled slot, so it must go with it.
    PrefixCache cache;
    const std::vector<Token> t1 = {1, 2, 3, 4};
    const std::vector<Token> t2 = {5, 6, 7, 8};
    cache.insert(PrefixCache::hash_prefix(t1, 4),
                 PrefixCache::Entry{.seq = 1, .n_tokens = 4, .blocks = {0}});
    cache.insert(PrefixCache::hash_prefix(t2, 4),
                 PrefixCache::Entry{.seq = 2, .n_tokens = 4, .blocks = {1}});
    EXPECT_EQ(cache.size(), 2U);

    cache.evict_sequence(1);
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_FALSE(cache.find_longest(t1, 4).has_value());
    EXPECT_TRUE(cache.find_longest(t2, 4).has_value());
}

// --- End-to-end through the scheduler ---------------------------------------------

SchedulerConfig cfg(bool prefix_caching, std::size_t batch = 2,
                    std::uint32_t block_size = 16) {
    return SchedulerConfig{.max_batch_size = batch,
                           .mode           = BatchingMode::kContinuous,
                           .kv_blocks      = 256,
                           .block_size     = block_size,
                           .prefix_caching = prefix_caching};
}

// Run N requests sharing a long system prompt; report what the engine actually prefilled.
struct SharedPromptRun {
    std::uint64_t prefilled = 0;
    std::uint64_t copied    = 0;
    std::uint64_t hits      = 0;
    std::uint64_t saved     = 0;
    std::vector<std::string> texts;
};

// More requests than batch slots, so later admissions happen after earlier sequences have
// finished prefilling and published their prefix. Sharing is between concurrently-live
// sequences -- a request admitted in the same step as its would-be donor has nothing to
// copy from yet, because that KV does not exist until the donor prefills.
SharedPromptRun run_shared_prompt(bool prefix_caching, int n_requests = 8) {
    const std::string system(96, 'S');  // long shared prefix, several whole blocks

    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, cfg(prefix_caching, /*batch=*/3));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < n_requests; ++i) {
        // Same long prefix, different suffix -- the shape of chat traffic. Suffix lengths
        // vary so sequences do NOT retire in lockstep: if every request finished on the
        // same step the pool would be empty at every admission and nothing could ever be
        // shared, which is an artefact of uniform test data rather than of the cache.
        const std::string suffix(static_cast<std::size_t>(1 + i * 3), 'q');
        Request r = make_req(static_cast<RequestId>(i), system + suffix,
                             static_cast<std::uint32_t>(4 + i));
        futures.push_back(r.result.get_future());
        EXPECT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    SharedPromptRun out;
    for (auto& f : futures) {
        out.texts.push_back(f.get().text);
    }
    out.prefilled = engine.tokens_prefilled();
    out.copied    = engine.tokens_copied();
    out.hits      = sched.stats().prefix_hits;
    out.saved     = sched.stats().prefix_tokens_saved;
    return out;
}

TEST(PrefixCacheScheduler, SharedSystemPromptAvoidsPrefillWork) {
    const SharedPromptRun off = run_shared_prompt(/*prefix_caching=*/false);
    const SharedPromptRun on  = run_shared_prompt(/*prefix_caching=*/true);

    EXPECT_GT(on.hits, 0U) << "no prefix hits: sharing never engaged";
    EXPECT_GT(on.copied, 0U) << "no KV was copied, so nothing was actually shared";
    EXPECT_LT(on.prefilled, off.prefilled)
        << "prefix caching must reduce tokens actually prefilled ("
        << on.prefilled << " vs " << off.prefilled << ")";
    EXPECT_EQ(on.saved, on.copied) << "reported savings must match the KV actually copied";
}

TEST(PrefixCacheScheduler, SharingDoesNotChangeResults) {
    // The decisive correctness property: sharing changes how much work is done, never
    // what is produced. If it did, the optimisation would be worthless.
    const SharedPromptRun off = run_shared_prompt(/*prefix_caching=*/false);
    const SharedPromptRun on  = run_shared_prompt(/*prefix_caching=*/true);
    EXPECT_EQ(on.texts, off.texts) << "prefix sharing corrupted generated output";
}

TEST(PrefixCacheScheduler, UnrelatedPromptsShareNothing) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, cfg(/*prefix_caching=*/true, /*batch=*/2));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 4; ++i) {
        // Entirely different leading bytes: no prefix in common.
        Request r = make_req(static_cast<RequestId>(i),
                             std::string(96, static_cast<char>('a' + i)));
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();
    for (auto& f : futures) {
        (void)f.get();
    }

    EXPECT_EQ(sched.stats().prefix_hits, 0U)
        << "unrelated prompts must not match -- a false hit would corrupt output";
}

TEST(PrefixCacheScheduler, BlocksAreStillFullyReclaimed) {
    // Sharing adds refcounts, which is exactly where a leak would hide: a block held by a
    // stale cache entry would never come back.
    const std::string system(96, 'S');

    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, cfg(/*prefix_caching=*/true, /*batch=*/2));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 10; ++i) {
        Request r = make_req(static_cast<RequestId>(i), system + std::to_string(i));
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();
    for (auto& f : futures) {
        (void)f.get();
    }

    EXPECT_EQ(sched.stats().kv_blocks_used, 0U)
        << "shared blocks leaked: " << sched.stats().kv_blocks_used << " still held";
}

}  // namespace
}  // namespace microvllm
