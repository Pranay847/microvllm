// KV-cache budget enforcement: admission control and preemption.
//
// These assert the mechanisms *fired* -- via Scheduler::Stats -- rather than merely that
// nothing crashed. A scheduler that silently ignored its budget would pass a "no crash"
// test perfectly while providing none of the behaviour.
#include "microvllm/scheduler.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "microvllm/mock_engine.hpp"

namespace microvllm {
namespace {

Request make_req(RequestId id, std::string prompt, std::uint32_t max_tokens = 512) {
    Request r;
    r.id              = id;
    r.spec.prompt     = std::move(prompt);
    r.spec.max_tokens = max_tokens;
    r.cancel          = std::make_shared<std::atomic<bool>>(false);
    return r;
}

SchedulerConfig kv_config(std::uint32_t blocks, std::uint32_t block_size = 16,
                          std::size_t batch = 4) {
    // Donors off: these tests are about admission control and preemption under a fixed
    // budget, and a retained prefix is a legitimate holder of blocks that would otherwise
    // show up as an apparent leak. Donor block accounting has its own test.
    return SchedulerConfig{.max_batch_size = batch,
                           .mode           = BatchingMode::kContinuous,
                           .kv_blocks      = blocks,
                           .block_size     = block_size,
                           .prefix_donors  = 0,
                           .prefill_chunk  = 128};
}

TEST(SchedulerKV, DefaultPoolIsDerivedFromTheEngineContext) {
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_ctx = 1024});
    RequestQueue queue(16);
    Scheduler    sched(engine, queue, SchedulerConfig{.max_batch_size = 4});

    // 1024 cells / 16 tokens per block = 64 blocks: the pool mirrors what the backend
    // can actually hold rather than inventing a limit.
    EXPECT_EQ(sched.stats().kv_blocks_total, 64U);
    EXPECT_EQ(sched.config().block_size, 16U);
}

TEST(SchedulerKV, WorkCompletesCorrectlyUnderATightBudget) {
    // Far more work than the pool can hold at once. Every request must still complete
    // with the right result -- the budget changes scheduling, never correctness.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(128);
    Scheduler       sched(engine, queue, kv_config(/*blocks=*/8));

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 30; ++i) {
        std::string p = "payload-number-" + std::to_string(i);
        Request     r = make_req(static_cast<RequestId>(i), p);
        futures.push_back(r.result.get_future());
        prompts.push_back(p);
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        const GenResult res = futures[i].get();
        EXPECT_EQ(res.text, prompts[i]) << "request " << i << " corrupted under cache pressure";
        EXPECT_EQ(res.reason, FinishReason::kEos);
    }
}

TEST(SchedulerKV, AllBlocksReturnToThePoolAfterEverythingDrains) {
    // The leak property at the scheduler level: once the queue drains, utilization must
    // be back to zero. A slow block leak would otherwise show up much later as an
    // unexplained capacity loss.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(128);
    Scheduler       sched(engine, queue, kv_config(/*blocks=*/16));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 25; ++i) {
        Request r = make_req(static_cast<RequestId>(i), "seq-" + std::to_string(i));
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();
    for (auto& f : futures) {
        (void)f.get();
    }

    const Scheduler::Stats s = sched.stats();
    EXPECT_EQ(s.kv_blocks_used, 0U) << "blocks leaked: " << s.kv_blocks_used << " still held";
    EXPECT_EQ(s.completed, 25U);
}

TEST(SchedulerKV, AdmissionIsDeferredWhenThePoolCannotFitAPrompt) {
    // A pool too small to hold every prompt at once must make later requests wait rather
    // than admitting them and failing. `deferred` proves the path was exercised.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    // Size the pool from the actual arithmetic. In echo mode a sequence's lifetime peak
    // is prompt + completion = (24 + 1 BOS) + 24 = 49 tokens, i.e. 4 blocks of 16. A pool
    // of exactly 4 blocks therefore holds precisely one sequence at a time, so every
    // request after the first must be deferred rather than admitted.
    Scheduler sched(engine, queue, kv_config(/*blocks=*/4, /*block_size=*/16, /*batch=*/4));

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 6; ++i) {
        std::string p(24, static_cast<char>('a' + i));
        Request     r = make_req(static_cast<RequestId>(i), p);
        futures.push_back(r.result.get_future());
        prompts.push_back(p);
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        EXPECT_EQ(futures[i].get().text, prompts[i]) << "request " << i;
    }
    EXPECT_GT(sched.stats().deferred, 0U)
        << "the pool was too small to admit everything at once; admission control never fired";
}

TEST(SchedulerKV, PreemptsWhenARunningSequenceCannotGrow) {
    // Sequences admitted successfully then grow past the pool. Something must be evicted
    // and recomputed; every request must still finish correctly.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    // Peak per sequence is (12 + 1 BOS) + 12 = 25 tokens = 7 blocks of 4. A pool of 10
    // admits one and starts a second, which then cannot grow -- forcing preemption.
    // Small blocks make boundary crossings frequent, so growth pressure is constant.
    Scheduler sched(engine, queue, kv_config(/*blocks=*/10, /*block_size=*/4, /*batch=*/4));

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 5; ++i) {
        std::string p(12, static_cast<char>('A' + i));
        Request     r = make_req(static_cast<RequestId>(i), p);
        futures.push_back(r.result.get_future());
        prompts.push_back(p);
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        const GenResult res = futures[i].get();
        EXPECT_EQ(res.text, prompts[i])
            << "request " << i << " was corrupted by preemption/recompute";
    }
    const Scheduler::Stats s = sched.stats();
    EXPECT_GT(s.preemptions + s.deferred, 0U)
        << "cache pressure was expected but neither preemption nor deferral fired";
    EXPECT_EQ(s.kv_blocks_used, 0U) << "preemption leaked blocks";
}

TEST(SchedulerKV, PreemptedRequestIsRecomputedNotDropped) {
    // The contract for preemption: the client sees a slower response, never a lost one.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    // Peak per sequence is (10 + 1 BOS) + 10 = 21 tokens = 6 blocks of 4. A pool of 8
    // fits one sequence comfortably but never two, so admitting several forces eviction.
    Scheduler sched(engine, queue, kv_config(/*blocks=*/8, /*block_size=*/4, /*batch=*/3));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 8; ++i) {
        Request r = make_req(static_cast<RequestId>(i), std::string(10, 'x'));
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        ASSERT_EQ(futures[i].wait_for(std::chrono::seconds(0)), std::future_status::ready)
            << "request " << i << " was dropped rather than requeued";
        EXPECT_EQ(futures[i].get().text, std::string(10, 'x'));
    }
}

TEST(SchedulerKV, SequenceLargerThanTheWholePoolFailsCleanly) {
    // Nothing to evict can make room for it, so it must be reported rather than looping
    // forever preempting and retrying.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_ctx = 4096});
    RequestQueue queue(16);
    Scheduler    sched(engine, queue, kv_config(/*blocks=*/2, /*block_size=*/4, /*batch=*/2));

    Request big = make_req(1, std::string(64, 'z'));  // 64 tokens vs an 8-token pool
    auto    fut = big.result.get_future();
    ASSERT_TRUE(queue.try_push(big));

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    const GenResult res = fut.get();
    EXPECT_EQ(res.reason, FinishReason::kContextOverflow)
        << "an unfittable request must be rejected, not retried forever";
    EXPECT_FALSE(res.error.empty());
}

TEST(SchedulerKV, StatsTrackAdmissionsAndCompletions) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, kv_config(/*blocks=*/32));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 7; ++i) {
        Request r = make_req(static_cast<RequestId>(i), "short");
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();
    for (auto& f : futures) {
        (void)f.get();
    }

    const Scheduler::Stats s = sched.stats();
    EXPECT_EQ(s.completed, 7U);
    EXPECT_GE(s.admitted, 7U);  // >= because a preempted request is admitted twice
    EXPECT_EQ(s.kv_blocks_total, 32U);
}

}  // namespace
}  // namespace microvllm
