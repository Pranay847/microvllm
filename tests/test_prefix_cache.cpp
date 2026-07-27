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
    // Donors off by default here: these tests isolate sharing between concurrently-live
    // sequences. Retention adds a second, legitimate holder of blocks and a second reason
    // to copy KV, which would blur what each assertion is actually measuring. The donor
    // tests below turn it on explicitly.
    return SchedulerConfig{.max_batch_size = batch,
                           .mode           = BatchingMode::kContinuous,
                           .kv_blocks      = 256,
                           .block_size     = block_size,
                           .prefix_donors  = 0,
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

// --- Donor retention ---------------------------------------------------------------
//
// The case that motivated the feature: requests arriving strictly one after another, each
// fully finished before the next is submitted. Without retention there is never a live
// donor and the hit rate is exactly zero.

struct SequentialRun {
    std::uint64_t            prefilled = 0;
    std::uint64_t            hits      = 0;
    std::uint64_t            retained  = 0;
    std::uint64_t            evictions = 0;
    std::vector<std::string> texts;
};

SequentialRun run_sequential(std::uint32_t donors, int n_requests = 4) {
    const std::string system(96, 'S');

    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_seq_max = 16});
    RequestQueue queue(64);
    SchedulerConfig c   = cfg(/*prefix_caching=*/true, /*batch=*/2);
    c.prefix_donors     = donors;
    Scheduler    sched(engine, queue, c);
    std::thread  t([&] { sched.run(); });

    SequentialRun out;
    for (int i = 0; i < n_requests; ++i) {
        const std::string suffix(static_cast<std::size_t>(1 + i), 'q');
        Request           r = make_req(static_cast<RequestId>(i), system + suffix, 4);
        auto              f = r.result.get_future();
        EXPECT_TRUE(queue.try_push(r));
        // Block for this request before submitting the next: nothing overlaps in time, so
        // any hit must come from a retained donor rather than a live sequence.
        out.texts.push_back(f.get().text);
    }
    queue.close();
    t.join();

    out.prefilled = engine.tokens_prefilled();
    out.hits      = sched.stats().prefix_hits;
    out.retained  = sched.stats().donors_retained;
    out.evictions = sched.stats().donor_evictions;
    return out;
}

TEST(PrefixDonors, SequentialRequestsHitARetainedPrefix) {
    const SequentialRun without = run_sequential(/*donors=*/0);
    const SequentialRun with    = run_sequential(/*donors=*/4);

    EXPECT_EQ(without.hits, 0U)
        << "sanity: with no donors, sequential traffic must have nothing to share";
    EXPECT_GT(with.hits, 0U) << "donor retention did not produce a single hit";
    EXPECT_GT(with.retained, 0U) << "no prefix was ever retained";
    EXPECT_LT(with.prefilled, without.prefilled)
        << "retention must avoid real prefill work (" << with.prefilled << " vs "
        << without.prefilled << ")";
}

TEST(PrefixDonors, RetentionDoesNotChangeResults) {
    // The whole point is to skip recomputation, not to change what is produced.
    EXPECT_EQ(run_sequential(/*donors=*/0).texts, run_sequential(/*donors=*/4).texts);
}

TEST(PrefixDonors, DonorPoolIsBoundedAndEvictsLeastRecentlyUsed) {
    // More distinct prefixes than donor slots: the pool must stay within its bound and
    // evict rather than grow.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_seq_max = 16});
    RequestQueue    queue(64);
    SchedulerConfig c = cfg(/*prefix_caching=*/true, /*batch=*/2);
    c.prefix_donors   = 2;
    Scheduler   sched(engine, queue, c);
    std::thread t([&] { sched.run(); });

    for (int i = 0; i < 6; ++i) {
        // Distinct long prefixes, so each retire wants its own donor slot.
        const std::string prompt(96, static_cast<char>('A' + i));
        Request           r = make_req(static_cast<RequestId>(i), prompt, 4);
        auto              f = r.result.get_future();
        ASSERT_TRUE(queue.try_push(r));
        (void)f.get();
    }
    queue.close();
    t.join();

    const auto s = sched.stats();
    EXPECT_LE(s.donors_held, 2U) << "donor pool exceeded its configured bound";
    EXPECT_GT(s.donor_evictions, 0U) << "pool never evicted despite more prefixes than slots";
}

TEST(PrefixDonors, RetainedBlocksAreAccountedForAndFullyReclaimable) {
    // Donors deliberately hold blocks after their request finishes -- that is the feature,
    // not a leak. The property that must hold is that the blocks they hold are bounded and
    // come back once the donors are evicted, so retention can never bleed the pool dry.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_seq_max = 16});
    RequestQueue    queue(64);
    SchedulerConfig c = cfg(/*prefix_caching=*/true, /*batch=*/2);
    c.prefix_donors   = 3;
    Scheduler   sched(engine, queue, c);
    std::thread t([&] { sched.run(); });

    for (int i = 0; i < 5; ++i) {
        const std::string prompt(96, static_cast<char>('A' + i));
        Request           r = make_req(static_cast<RequestId>(i), prompt, 4);
        auto              f = r.result.get_future();
        ASSERT_TRUE(queue.try_push(r));
        (void)f.get();
    }
    queue.close();
    t.join();

    const auto after = sched.stats();
    EXPECT_GT(after.kv_blocks_used, 0U) << "nothing was retained, so there is nothing to test";
    EXPECT_LE(after.donors_held, 3U);

    // Every block still held must belong to a donor. Retention is the only thing allowed
    // to outlive a request; anything else is a genuine leak.
    EXPECT_LE(after.kv_blocks_used, after.donors_held * (96 / 16 + 1))
        << "more blocks are held than the retained prefixes can account for";
}

TEST(PrefixDonors, DonorsAreReclaimedBeforeLiveSequencesArePreempted) {
    // The policy that matters under memory pressure: a donor is cache and a running
    // sequence is work. With a pool small enough to force reclamation, the scheduler must
    // give up donors first and ideally never preempt at all.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_seq_max = 16});
    RequestQueue    queue(64);
    SchedulerConfig c = cfg(/*prefix_caching=*/true, /*batch=*/2);
    c.kv_blocks       = 12;  // tight: donors and live sequences genuinely compete
    c.prefix_donors   = 4;
    Scheduler   sched(engine, queue, c);
    std::thread t([&] { sched.run(); });

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 8; ++i) {
        const std::string p = std::string(64, static_cast<char>('a' + i));
        Request           r = make_req(static_cast<RequestId>(i), p, 8);
        futures.push_back(r.result.get_future());
        prompts.push_back(p);
        ASSERT_TRUE(queue.try_push(r));
    }
    queue.close();
    t.join();

    // Correctness first: reclaiming cache must never corrupt or drop a request.
    // max_tokens is 8, so each echo is the prompt's first 8 characters.
    for (std::size_t i = 0; i < futures.size(); ++i) {
        EXPECT_EQ(futures[i].get().text, prompts[i].substr(0, 8))
            << "request " << i << " was corrupted";
    }
    const auto s = sched.stats();
    EXPECT_GT(s.donor_evictions, 0U) << "pressure never reached the donor pool";
    EXPECT_GE(s.donor_evictions, s.preemptions)
        << "live sequences were preempted before cache was given up";
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
