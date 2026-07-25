#include "microvllm/server.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>

#include "microvllm/collecting_sink.hpp"
#include "microvllm/generator.hpp"
#include "microvllm/http_api.hpp"
#include "microvllm/request_queue.hpp"
#include "microvllm/scheduler.hpp"

namespace microvllm {
namespace {

constexpr char kJson[] = "application/json";

// Set from the signal handler only. A watcher thread turns this into a clean
// server.stop(); the handler itself does nothing that isn't async-signal-safe.
std::atomic<bool> g_signal_received{false};

extern "C" void on_signal(int) { g_signal_received.store(true, std::memory_order_relaxed); }

}  // namespace

bool serve(IModelEngine& engine, const ServerConfig& config) {
    RequestQueue queue(config.max_queue_depth);
    Scheduler    scheduler(engine, queue, SchedulerConfig{.max_batch_size = config.max_batch_size});
    std::thread  worker_thread([&] { scheduler.run(); });

    httplib::Server server;
    std::atomic<std::uint64_t> next_id{1};
    std::atomic<bool>          shutting_down{false};

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", kJson);
    });

    server.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        RequestSpec spec;
        try {
            spec = parse_generate_request(req.body);
        } catch (const BadRequest& e) {
            res.status = 400;
            res.set_content(make_error_response(e.message), kJson);
            return;
        }

        // Keep a copy of the cancel flag and the future: `r` is moved into the queue.
        auto    cancel = std::make_shared<std::atomic<bool>>(false);
        Request r;
        r.id     = next_id.fetch_add(1, std::memory_order_relaxed);
        r.spec   = std::move(spec);
        r.cancel = cancel;
        std::future<GenResult> fut = r.result.get_future();

        if (!queue.try_push(r)) {
            res.status = 503;
            res.set_content(make_error_response("server at capacity; retry later"), kJson);
            return;
        }

        // Block for the result, but wake periodically to notice a client disconnect or
        // a server shutdown and cancel through the shared flag either way.
        while (fut.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
            if (req.is_connection_closed() || shutting_down.load(std::memory_order_relaxed)) {
                cancel->store(true, std::memory_order_relaxed);
            }
        }

        const GenResult result = fut.get();
        if (result.reason == FinishReason::kError) {
            res.status = 500;
            res.set_content(make_error_response(result.error), kJson);
            return;
        }
        res.set_content(make_generate_response(result.text, result.reason, result.usage), kJson);
    });

    // Turn SIGINT/SIGTERM into a graceful stop. The watcher thread polls the flag the
    // handler set and calls server.stop(); handlers then see shutting_down, cancel their
    // in-flight work, and return, which lets listen() unwind.
    g_signal_received.store(false, std::memory_order_relaxed);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::atomic<bool> watcher_done{false};
    std::thread       watcher([&] {
        while (!watcher_done.load(std::memory_order_relaxed)) {
            if (g_signal_received.load(std::memory_order_relaxed)) {
                std::fprintf(stderr, "\nshutting down...\n");
                shutting_down.store(true, std::memory_order_relaxed);
                server.stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    std::printf("microvllm listening on http://%s:%d\n", config.host.c_str(), config.port);
    const bool ok = server.listen(config.host, config.port);

    // Unwind: stop the watcher, then drain and join the engine thread. By the time
    // listen() returns, every handler has returned, so every submitted request has been
    // answered; closing the queue lets the worker's pop() return and run() finish.
    watcher_done.store(true, std::memory_order_relaxed);
    watcher.join();
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);

    queue.close();
    worker_thread.join();
    return ok;
}

}  // namespace microvllm
