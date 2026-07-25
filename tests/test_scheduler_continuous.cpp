// Continuous batching: the properties that distinguish it from static batching.
//
// Throughput numbers live in the benchmarks; these tests prove *correctness* of the
// mechanism -- that sequences may be admitted and retired mid-flight without their
// results being crossed, truncated, or lost, and that a short request is not held hostage
// by a long one sharing the engine.
#include "microvllm/scheduler.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "microvllm/mock_engine.hpp"

namespace microvllm {
namespace {

SchedulerConfig continuous(std::size_t batch, std::size_t chunk = 128) {
    return SchedulerConfig{
        .max_batch_size = batch, .mode = BatchingMode::kContinuous, .prefill_chunk = chunk};
}

Request make_req(RequestId id, std::string prompt, std::uint32_t max_tokens = 512,
                 std::shared_ptr<std::atomic<bool>> cancel = nullptr) {
    Request r;
    r.id              = id;
    r.spec.prompt     = std::move(prompt);
    r.spec.max_tokens = max_tokens;
    r.cancel          = cancel ? std::move(cancel) : std::make_shared<std::atomic<bool>>(false);
    return r;
}

TEST(SchedulerContinuous, ShortRequestFinishesBeforeALongOneAdmittedFirst) {
    // The headline behavior. Under static batching both requests share a batch and the
    // short one cannot complete until the long one does. Under continuous batching the
    // short one retires as soon as it is done.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, continuous(8));

    Request long_req  = make_req(1, std::string(400, 'L'));
    Request short_req = make_req(2, "S");
    auto    long_fut  = long_req.result.get_future();
    auto    short_fut = short_req.result.get_future();

    ASSERT_TRUE(queue.try_push(long_req));   // queued first
    ASSERT_TRUE(queue.try_push(short_req));

    std::thread t([&] { sched.run(); });

    // The short request must be deliverable well before the long one completes.
    const auto status = short_fut.wait_for(std::chrono::seconds(5));
    ASSERT_EQ(status, std::future_status::ready) << "short request was blocked by the long one";
    const GenResult short_res = short_fut.get();
    EXPECT_EQ(short_res.text, "S");

    queue.close();
    t.join();

    const GenResult long_res = long_fut.get();
    EXPECT_EQ(long_res.text, std::string(400, 'L'));
    EXPECT_EQ(long_res.usage.completion_tokens, 400U);
}

TEST(SchedulerContinuous, AdmitsRequestsArrivingMidFlight) {
    // A request submitted while other sequences are already generating must be picked up
    // without waiting for them to finish.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, continuous(4));

    Request first = make_req(1, std::string(300, 'A'));
    auto    ffut  = first.result.get_future();
    ASSERT_TRUE(queue.try_push(first));

    std::thread t([&] { sched.run(); });

    // Submit more work after generation is underway.
    std::vector<std::future<GenResult>> later;
    std::vector<std::string>            later_prompts;
    for (int i = 0; i < 6; ++i) {
        std::string p = "late" + std::to_string(i);
        Request     r = make_req(static_cast<RequestId>(10 + i), p);
        later.push_back(r.result.get_future());
        later_prompts.push_back(p);
        while (!queue.try_push(r)) {
            std::this_thread::yield();
        }
    }

    queue.close();
    t.join();

    EXPECT_EQ(ffut.get().text, std::string(300, 'A'));
    for (std::size_t i = 0; i < later.size(); ++i) {
        EXPECT_EQ(later[i].get().text, later_prompts[i])
            << "mid-flight request " << i << " got the wrong result";
    }
}

TEST(SchedulerContinuous, ReusedSlotsDoNotLeakStateBetweenSequences) {
    // Slot ids are recycled as sequences retire. If a slot's KV cache or sampler state
    // were not released, a later request landing on it would produce corrupted output.
    // Far more requests than slots forces heavy reuse.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(128);
    Scheduler       sched(engine, queue, continuous(2));  // only two slots

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 40; ++i) {
        std::string p = "unique-payload-" + std::to_string(i);
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
        EXPECT_EQ(res.text, prompts[i]) << "slot reuse leaked state into request " << i;
        EXPECT_EQ(res.reason, FinishReason::kEos);
    }
}

TEST(SchedulerContinuous, ChunkedPrefillDoesNotChangeResults) {
    // A tiny chunk budget forces long prompts across many steps, interleaved with other
    // sequences' decodes. Output must be identical to prefilling whole.
    auto drive = [](std::size_t chunk) {
        MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
        RequestQueue    queue(64);
        Scheduler       sched(engine, queue, continuous(4, chunk));

        std::vector<std::future<GenResult>> futures;
        for (int i = 0; i < 6; ++i) {
            Request r = make_req(static_cast<RequestId>(i),
                                 std::string(static_cast<std::size_t>(20 + i * 30), 'x'));
            futures.push_back(r.result.get_future());
            EXPECT_TRUE(queue.try_push(r));
        }
        std::thread t([&] { sched.run(); });
        queue.close();
        t.join();

        std::vector<std::string> out;
        for (auto& f : futures) {
            out.push_back(f.get().text);
        }
        return out;
    };

    const std::vector<std::string> whole   = drive(4096);
    const std::vector<std::string> chunked = drive(8);
    EXPECT_EQ(whole, chunked) << "chunked prefill changed generated output";
}

TEST(SchedulerContinuous, MixedFinishReasonsRetireIndependently) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, continuous(8));

    Request eos    = make_req(1, "abcdef");
    Request capped = make_req(2, "abcdef", /*max_tokens=*/2);
    auto    cancel = std::make_shared<std::atomic<bool>>(true);
    Request killed = make_req(3, "abcdef", 512, cancel);

    auto f1 = eos.result.get_future();
    auto f2 = capped.result.get_future();
    auto f3 = killed.result.get_future();
    ASSERT_TRUE(queue.try_push(eos));
    ASSERT_TRUE(queue.try_push(capped));
    ASSERT_TRUE(queue.try_push(killed));

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    EXPECT_EQ(f1.get().reason, FinishReason::kEos);
    const GenResult capped_res = f2.get();
    EXPECT_EQ(capped_res.reason, FinishReason::kMaxTokens);
    EXPECT_EQ(capped_res.text, "ab");
    EXPECT_EQ(f3.get().reason, FinishReason::kCancelled);
}

TEST(SchedulerContinuous, MatchesStaticBatchingOutputExactly) {
    // Scheduling policy must not change what the model produces -- only when. Same
    // requests through both modes must yield identical text and accounting.
    auto drive = [](BatchingMode mode) {
        MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
        RequestQueue    queue(64);
        Scheduler       sched(engine, queue,
                              SchedulerConfig{.max_batch_size = 4, .mode = mode});

        std::vector<std::future<GenResult>> futures;
        for (int i = 0; i < 12; ++i) {
            Request r = make_req(static_cast<RequestId>(i), "payload-" + std::to_string(i),
                                 static_cast<std::uint32_t>(4 + i));
            futures.push_back(r.result.get_future());
            EXPECT_TRUE(queue.try_push(r));
        }
        std::thread t([&] { sched.run(); });
        queue.close();
        t.join();

        std::vector<std::pair<std::string, std::uint32_t>> out;
        for (auto& f : futures) {
            const GenResult r = f.get();
            out.emplace_back(r.text, r.usage.completion_tokens);
        }
        return out;
    };

    EXPECT_EQ(drive(BatchingMode::kStatic), drive(BatchingMode::kContinuous));
}

TEST(SchedulerContinuous, RejectsRequestsThatCannotFitTheContext) {
    // Regression: llama.cpp divides the KV pool across sequences, so a request is bounded
    // by n_ctx_seq, not n_ctx. Asking for more used to be admitted and then fail deep in
    // the backend with "failed to find a memory slot" -- after the prefill compute had
    // already been spent. It must be rejected at admission, and distinguishably so, since
    // it is a client error rather than a server fault.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_ctx = 64});
    RequestQueue queue(16);
    Scheduler    sched(engine, queue, continuous(4));

    Request too_big = make_req(1, "prompt", /*max_tokens=*/1000);
    Request ok      = make_req(2, "fine", /*max_tokens=*/8);
    auto    big_fut = too_big.result.get_future();
    auto    ok_fut  = ok.result.get_future();
    ASSERT_TRUE(queue.try_push(too_big));
    ASSERT_TRUE(queue.try_push(ok));

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    const GenResult big = big_fut.get();
    EXPECT_EQ(big.reason, FinishReason::kContextOverflow);
    EXPECT_NE(big.error.find("per-request context"), std::string::npos)
        << "the error must say what the limit is, not just that it failed";

    // An oversized request must not take its neighbours down with it.
    // ("fine" echoes 4 tokens, well under its 8-token budget, so it ends at EOS.)
    EXPECT_EQ(ok_fut.get().reason, FinishReason::kEos);
}

TEST(SchedulerStaticMode, RejectsRequestsThatCannotFitTheContext) {
    // Same admission rule in static mode: a pre-rejected job must never reach the engine.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_ctx = 64});
    RequestQueue queue(16);
    Scheduler    sched(engine, queue,
                       SchedulerConfig{.max_batch_size = 4, .mode = BatchingMode::kStatic});

    Request too_big = make_req(1, "prompt", 1000);
    Request ok      = make_req(2, "fine", 8);
    auto    big_fut = too_big.result.get_future();
    auto    ok_fut  = ok.result.get_future();
    ASSERT_TRUE(queue.try_push(too_big));
    ASSERT_TRUE(queue.try_push(ok));

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    EXPECT_EQ(big_fut.get().reason, FinishReason::kContextOverflow);
    EXPECT_EQ(ok_fut.get().reason, FinishReason::kEos);
}

TEST(SchedulerContinuous, DrainsEverythingBeforeExiting) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(256);
    Scheduler       sched(engine, queue, continuous(4));

    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 50; ++i) {
        Request r = make_req(static_cast<RequestId>(i), "d" + std::to_string(i));
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }
    queue.close();  // closed before the scheduler even starts

    std::thread t([&] { sched.run(); });
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        EXPECT_EQ(futures[i].wait_for(std::chrono::seconds(0)), std::future_status::ready)
            << "request " << i << " was abandoned at shutdown";
    }
}

}  // namespace
}  // namespace microvllm
