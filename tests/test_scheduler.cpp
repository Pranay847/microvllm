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

Request make_req(RequestId id, std::string prompt, std::uint32_t max_tokens = 128,
                 std::vector<std::string> stop = {}) {
    Request r;
    r.id              = id;
    r.spec.prompt     = std::move(prompt);
    r.spec.max_tokens = max_tokens;
    r.spec.stop       = std::move(stop);
    r.cancel          = std::make_shared<std::atomic<bool>>(false);
    return r;
}

// Run a batch directly through the scheduler and collect the results in submit order.
std::vector<GenResult> run_batch(MockModelEngine& engine, std::vector<Request> reqs,
                                 std::size_t max_batch = 8) {
    RequestQueue queue(64);  // unused by run_batch, but the scheduler holds a reference
    Scheduler    sched(engine, queue, SchedulerConfig{.max_batch_size = max_batch});

    std::vector<std::future<GenResult>> futures;
    futures.reserve(reqs.size());
    for (Request& r : reqs) {
        futures.push_back(r.result.get_future());
    }
    sched.run_batch(std::move(reqs));

    std::vector<GenResult> out;
    out.reserve(futures.size());
    for (auto& f : futures) {
        out.push_back(f.get());
    }
    return out;
}

TEST(Scheduler, BatchedSequencesEachGetTheirOwnResult) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    std::vector<Request> reqs;
    for (int i = 0; i < 5; ++i) {
        reqs.push_back(make_req(static_cast<RequestId>(i), "seq" + std::to_string(i)));
    }

    const std::vector<GenResult> res = run_batch(engine, std::move(reqs));
    ASSERT_EQ(res.size(), 5U);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(res[static_cast<std::size_t>(i)].text, "seq" + std::to_string(i))
            << "sequence " << i << " got another sequence's output";
        EXPECT_EQ(res[static_cast<std::size_t>(i)].reason, FinishReason::kEos);
    }
}

TEST(Scheduler, SequencesOfDifferentLengthsFinishIndependently) {
    // The core batching property: sequences leave the batch at different steps, and a
    // short one must not be truncated to match a long one (or vice versa).
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    std::vector<Request> reqs;
    reqs.push_back(make_req(1, "a"));                  // 1 token
    reqs.push_back(make_req(2, "abcdefghij"));         // 10 tokens
    reqs.push_back(make_req(3, "abcd"));               // 4 tokens

    const std::vector<GenResult> res = run_batch(engine, std::move(reqs));
    ASSERT_EQ(res.size(), 3U);
    EXPECT_EQ(res[0].text, "a");
    EXPECT_EQ(res[1].text, "abcdefghij");
    EXPECT_EQ(res[2].text, "abcd");
    EXPECT_EQ(res[0].usage.completion_tokens, 1U);
    EXPECT_EQ(res[1].usage.completion_tokens, 10U);
    EXPECT_EQ(res[2].usage.completion_tokens, 4U);
}

TEST(Scheduler, MixedFinishReasonsWithinOneBatch) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    std::vector<Request> reqs;
    reqs.push_back(make_req(1, "abcdef"));                      // runs to EOS
    reqs.push_back(make_req(2, "abcdef", /*max_tokens=*/2));     // hits the budget
    reqs.push_back(make_req(3, "abcdef", 128, /*stop=*/{"cd"})); // hits a stop string

    const std::vector<GenResult> res = run_batch(engine, std::move(reqs));
    ASSERT_EQ(res.size(), 3U);
    EXPECT_EQ(res[0].reason, FinishReason::kEos);
    EXPECT_EQ(res[0].text, "abcdef");
    EXPECT_EQ(res[1].reason, FinishReason::kMaxTokens);
    EXPECT_EQ(res[1].text, "ab");
    EXPECT_EQ(res[2].reason, FinishReason::kStopString);
    EXPECT_EQ(res[2].text, "ab") << "stop strings still truncate correctly inside a batch";
}

TEST(Scheduler, BatchOfOneMatchesSerialGeneration) {
    // Batching must not change what a request produces, only how fast.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    std::vector<Request> one;
    one.push_back(make_req(1, "hello world"));
    const std::vector<GenResult> serial = run_batch(engine, std::move(one), /*max_batch=*/1);

    MockModelEngine engine2(MockModelEngine::Config{.response = "", .echo_prompt = true});
    std::vector<Request> many;
    many.push_back(make_req(1, "hello world"));
    many.push_back(make_req(2, "other"));
    const std::vector<GenResult> batched = run_batch(engine2, std::move(many), /*max_batch=*/8);

    EXPECT_EQ(serial[0].text, batched[0].text);
    EXPECT_EQ(serial[0].usage.completion_tokens, batched[0].usage.completion_tokens);
}

TEST(Scheduler, CancelledSequenceLeavesBatchOthersContinue) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    auto    cancel = std::make_shared<std::atomic<bool>>(true);  // pre-cancelled
    Request a      = make_req(1, "abcdefghij");
    Request b      = make_req(2, "abcdefghij");
    b.cancel       = cancel;

    std::vector<Request> reqs;
    reqs.push_back(std::move(a));
    reqs.push_back(std::move(b));

    const std::vector<GenResult> res = run_batch(engine, std::move(reqs));
    ASSERT_EQ(res.size(), 2U);
    EXPECT_EQ(res[0].reason, FinishReason::kEos) << "an unrelated cancel must not affect this one";
    EXPECT_EQ(res[0].text, "abcdefghij");
    EXPECT_EQ(res[1].reason, FinishReason::kCancelled);
}

TEST(Scheduler, ClampsBatchSizeToEngineSequenceCapacity) {
    // Regression: a batch with more sequences than the context was built for makes
    // llama.cpp abort the process. The engine's capacity is authoritative, so no
    // --batch-size value may be allowed to exceed it.
    MockModelEngine engine(MockModelEngine::Config{
        .response = "", .echo_prompt = true, .n_seq_max = 4});
    RequestQueue queue(64);
    Scheduler    sched(engine, queue, SchedulerConfig{.max_batch_size = 32});

    std::vector<Request>                futures_owner;
    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 10; ++i) {
        Request r = make_req(static_cast<RequestId>(i), "s" + std::to_string(i));
        futures.push_back(r.result.get_future());
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    // All requests still complete correctly -- just in several smaller batches.
    for (std::size_t i = 0; i < futures.size(); ++i) {
        EXPECT_EQ(futures[i].get().text, "s" + std::to_string(i));
    }
}

TEST(Scheduler, ZeroBatchSizeIsTreatedAsOne) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(16);
    Scheduler       sched(engine, queue, SchedulerConfig{.max_batch_size = 0});

    Request r   = make_req(1, "ok");
    auto    fut = r.result.get_future();
    ASSERT_TRUE(queue.try_push(r));

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    EXPECT_EQ(fut.get().text, "ok") << "a degenerate batch size must not hang or crash";
}

TEST(Scheduler, EmptyBatchIsANoop) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(4);
    Scheduler       sched(engine, queue);
    sched.run_batch({});  // must not crash or hang
    SUCCEED();
}

TEST(Scheduler, RunDrainsQueueAndBatchesWhatIsAvailable) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, SchedulerConfig{.max_batch_size = 4});

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 10; ++i) {
        std::string p = "job" + std::to_string(i);
        Request     r = make_req(static_cast<RequestId>(i), p);
        futures.push_back(r.result.get_future());
        prompts.push_back(p);
        ASSERT_TRUE(queue.try_push(r));
    }

    std::thread t([&] { sched.run(); });
    queue.close();
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        EXPECT_EQ(futures[i].get().text, prompts[i]);
    }
}

// Engine that fails during decode, to prove no promise is left unfulfilled.
struct ThrowingEngine final : IModelEngine {
    [[nodiscard]] EngineCaps caps() const override { return EngineCaps{4096, 512, 16, -1}; }
    [[nodiscard]] std::vector<Token> tokenize(std::string_view, bool) override { return {1, 2}; }
    [[nodiscard]] std::string piece(Token) override { return {}; }
    void begin_sequence(SeqId, const SamplingParams&) override {}
    [[nodiscard]] std::vector<GenStep> decode(std::span<const BatchItem>) override {
        throw std::runtime_error("decode exploded");
    }
    void release_sequence(SeqId) override { ++released; }
    void copy_sequence(SeqId, SeqId, Pos) override {}
    int released = 0;
};

TEST(Scheduler, EngineFailureFulfillsEveryPromiseInTheBatch) {
    ThrowingEngine engine;
    RequestQueue   queue(16);
    Scheduler      sched(engine, queue);

    std::vector<Request>                reqs;
    std::vector<std::future<GenResult>> futures;
    for (int i = 0; i < 3; ++i) {
        Request r = make_req(static_cast<RequestId>(i), "p");
        futures.push_back(r.result.get_future());
        reqs.push_back(std::move(r));
    }
    sched.run_batch(std::move(reqs));

    for (auto& f : futures) {
        const GenResult res = f.get();  // must not hang
        EXPECT_EQ(res.reason, FinishReason::kError);
        EXPECT_NE(res.error.find("decode exploded"), std::string::npos);
    }
    EXPECT_GE(engine.released, 3) << "every sequence's KV slot must be released on failure";
}

}  // namespace
}  // namespace microvllm
