// The concurrency-correctness suite. Its whole reason to exist is to be run under
// ThreadSanitizer (the wsl-tsan preset, repeated in CI), so it hammers the exact shared
// surface of the server -- the RequestQueue and the per-request cancel flags -- from
// many threads against a single EngineWorker.
//
// It uses the mock engine, so a report here is necessarily a first-party data race.
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "microvllm/engine_worker.hpp"
#include "microvllm/mock_engine.hpp"

namespace microvllm {
namespace {

Request make_req(RequestId id, std::string prompt,
                 std::shared_ptr<std::atomic<bool>> cancel) {
    Request r;
    r.id              = id;
    r.spec.prompt     = std::move(prompt);
    r.spec.max_tokens = 128;
    r.cancel          = std::move(cancel);
    return r;
}

// Many producers push distinct requests through a deliberately small queue (forcing
// backpressure) to one worker; every request must come back with its own result.
TEST(Concurrency, ManyProducersSingleWorkerEveryResultMatches) {
    constexpr int kProducers    = 8;
    constexpr int kPerProducer  = 50;

    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(16);  // small on purpose: producers will be rejected and retry
    EngineWorker    worker(engine, queue);
    std::thread     worker_thread([&] { worker.run(); });

    struct Pending {
        std::string             prompt;
        std::future<GenResult>  fut;
    };
    std::mutex            pending_mu;
    std::vector<Pending>  pending;

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kPerProducer; ++i) {
                const std::string prompt =
                    "p" + std::to_string(p) + "-r" + std::to_string(i);
                Request r = make_req(static_cast<RequestId>(p * kPerProducer + i), prompt,
                                     std::make_shared<std::atomic<bool>>(false));
                auto    fut = r.result.get_future();

                // Reject-on-full backpressure: spin until accepted.
                while (!queue.try_push(r)) {
                    std::this_thread::yield();
                }
                const std::lock_guard<std::mutex> lock(pending_mu);
                pending.push_back({prompt, std::move(fut)});
            }
        });
    }
    for (std::thread& t : producers) {
        t.join();
    }
    queue.close();
    worker_thread.join();

    ASSERT_EQ(pending.size(), static_cast<std::size_t>(kProducers * kPerProducer));
    for (Pending& pd : pending) {
        const GenResult res = pd.fut.get();
        EXPECT_EQ(res.reason, FinishReason::kEos);
        EXPECT_EQ(res.text, pd.prompt) << "a result was delivered to the wrong request";
    }
}

// A separate thread trips cancel flags while the worker generates. Every request must
// still resolve exactly once to a valid terminal state, with no torn read of the flag.
TEST(Concurrency, ConcurrentCancellationIsRaceFree) {
    constexpr int kRequests = 200;

    MockModelEngine engine(MockModelEngine::Config{
        .response = "abcdefghijklmnop", .echo_prompt = false,
        .token_latency = std::chrono::microseconds(20)});  // a window for cancels to land
    RequestQueue queue(kRequests);
    EngineWorker worker(engine, queue);
    std::thread  worker_thread([&] { worker.run(); });

    std::vector<std::shared_ptr<std::atomic<bool>>> cancels;
    std::vector<std::future<GenResult>>             futures;
    cancels.reserve(kRequests);
    futures.reserve(kRequests);
    for (int i = 0; i < kRequests; ++i) {
        auto    cancel = std::make_shared<std::atomic<bool>>(false);
        Request r      = make_req(static_cast<RequestId>(i), "prompt", cancel);
        futures.push_back(r.result.get_future());
        cancels.push_back(cancel);
        ASSERT_TRUE(queue.try_push(r));
    }

    // Cancel every other request concurrently with processing.
    std::thread canceller([&] {
        for (int i = 0; i < kRequests; i += 2) {
            cancels[static_cast<std::size_t>(i)]->store(true, std::memory_order_relaxed);
        }
    });

    canceller.join();
    queue.close();
    worker_thread.join();

    int cancelled = 0;
    int completed = 0;
    for (std::future<GenResult>& f : futures) {
        const GenResult res = f.get();  // must not hang: promise always set
        if (res.reason == FinishReason::kCancelled) {
            ++cancelled;
        } else if (res.reason == FinishReason::kEos || res.reason == FinishReason::kMaxTokens) {
            ++completed;
        } else {
            ADD_FAILURE() << "unexpected terminal reason " << to_string(res.reason);
        }
    }
    EXPECT_EQ(cancelled + completed, kRequests) << "every request resolved exactly once";
}

// Closing the queue while producers are still pushing: accepted requests all complete;
// rejected ones are cleanly refused. Nothing is lost or double-delivered.
TEST(Concurrency, CloseDuringProductionLosesNothingAccepted) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(64);
    EngineWorker    worker(engine, queue);
    std::thread     worker_thread([&] { worker.run(); });

    std::mutex                          accepted_mu;
    std::vector<std::future<GenResult>> accepted;
    std::vector<std::string>            accepted_prompts;
    std::atomic<bool>                   stop{false};

    std::thread producer([&] {
        int i = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            const std::string prompt = "x" + std::to_string(i++);
            Request           r      = make_req(static_cast<RequestId>(i), prompt,
                                                std::make_shared<std::atomic<bool>>(false));
            auto              fut    = r.result.get_future();
            if (queue.try_push(r)) {
                const std::lock_guard<std::mutex> lock(accepted_mu);
                accepted.push_back(std::move(fut));
                accepted_prompts.push_back(prompt);
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    stop.store(true, std::memory_order_relaxed);
    producer.join();
    queue.close();
    worker_thread.join();

    for (std::size_t i = 0; i < accepted.size(); ++i) {
        const GenResult res = accepted[i].get();
        EXPECT_EQ(res.text, accepted_prompts[i]);
        EXPECT_EQ(res.reason, FinishReason::kEos);
    }
    SUCCEED() << accepted.size() << " requests accepted and all delivered correctly";
}

}  // namespace
}  // namespace microvllm
