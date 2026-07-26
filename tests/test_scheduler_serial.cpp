// Scheduler with max_batch_size == 1: the serial path that Phase 2's EngineWorker used
// to own. These tests are kept as a distinct suite because batch-size-1 is the baseline
// every throughput measurement is compared against -- if batching ever silently broke the
// degenerate case, the benchmark's control arm would be measuring the wrong thing.
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

constexpr SchedulerConfig kSerial{.max_batch_size = 1};

Request make_req(RequestId id, std::string prompt,
                 std::shared_ptr<std::atomic<bool>> cancel = nullptr) {
    Request r;
    r.id              = id;
    r.spec.prompt     = std::move(prompt);
    r.spec.max_tokens = 128;
    r.cancel          = cancel ? std::move(cancel) : std::make_shared<std::atomic<bool>>(false);
    return r;
}

TEST(SchedulerSerial, FulfillsPromiseWithEchoedResult) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(16);
    Scheduler       sched(engine, queue, kSerial);
    std::thread     t([&] { sched.run(); });

    Request r   = make_req(1, "hello");
    auto    fut = r.result.get_future();
    ASSERT_TRUE(queue.try_push(r));

    queue.close();
    t.join();

    const GenResult res = fut.get();
    EXPECT_EQ(res.text, "hello");
    EXPECT_EQ(res.reason, FinishReason::kEos);
    EXPECT_EQ(res.usage.completion_tokens, 5U);
}

TEST(SchedulerSerial, ProcessesManyRequestsEachWithItsOwnResult) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    Scheduler       sched(engine, queue, kSerial);
    std::thread     t([&] { sched.run(); });

    std::vector<std::future<GenResult>> futures;
    std::vector<std::string>            prompts;
    for (int i = 0; i < 20; ++i) {
        std::string prompt = "request-" + std::to_string(i);
        Request     r      = make_req(static_cast<RequestId>(i), prompt);
        futures.push_back(r.result.get_future());
        prompts.push_back(prompt);
        ASSERT_TRUE(queue.try_push(r));
    }

    queue.close();
    t.join();

    for (std::size_t i = 0; i < futures.size(); ++i) {
        EXPECT_EQ(futures[i].get().text, prompts[i]) << "result " << i << " must match its request";
    }
}

struct ThrowingEngine final : IModelEngine {
    [[nodiscard]] EngineCaps caps() const override { return EngineCaps{4096, 512, 16, -1}; }
    [[nodiscard]] std::vector<Token> tokenize(std::string_view, bool) override {
        return {1, 2, 3};
    }
    [[nodiscard]] std::string piece(Token) override { return {}; }
    void begin_sequence(SeqId, const SamplingParams&) override {}
    [[nodiscard]] std::vector<GenStep> decode(std::span<const BatchItem>) override {
        throw std::runtime_error("boom");
    }
    void release_sequence(SeqId) override { ++released; }
    void copy_sequence(SeqId, SeqId, Pos) override {}
    int released = 0;
};

TEST(SchedulerSerial, FulfillsPromiseOnEngineError) {
    ThrowingEngine engine;
    RequestQueue   queue(4);
    Scheduler      sched(engine, queue, kSerial);
    std::thread    t([&] { sched.run(); });

    Request r   = make_req(1, "x");
    auto    fut = r.result.get_future();
    ASSERT_TRUE(queue.try_push(r));

    queue.close();
    t.join();

    const GenResult res = fut.get();
    EXPECT_EQ(res.reason, FinishReason::kError);
    EXPECT_NE(res.error.find("boom"), std::string::npos);
    EXPECT_GT(engine.released, 0) << "the dirty KV slot must be cleared after a failure";
}

TEST(SchedulerSerial, HonorsCancellation) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(4);
    Scheduler       sched(engine, queue, kSerial);

    auto    cancel = std::make_shared<std::atomic<bool>>(true);  // already cancelled
    Request r      = make_req(1, "abcdefgh", cancel);
    auto    fut    = r.result.get_future();

    std::thread t([&] { sched.run(); });
    ASSERT_TRUE(queue.try_push(r));
    queue.close();
    t.join();

    const GenResult res = fut.get();
    EXPECT_EQ(res.reason, FinishReason::kCancelled);
    EXPECT_LT(res.usage.completion_tokens, 8U);
}

}  // namespace
}  // namespace microvllm
