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
#include "microvllm/streaming_sink.hpp"

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
    Scheduler    scheduler(engine, queue,
                           SchedulerConfig{.max_batch_size = config.max_batch_size,
                                           .mode           = config.mode,
                                           .kv_blocks      = config.kv_blocks,
                                           .block_size     = config.block_size,
                                           .prefix_caching = config.prefix_caching,
                                           .prefill_chunk  = config.prefill_chunk});
    std::thread  worker_thread([&] { scheduler.run(); });

    httplib::Server server;
    std::atomic<std::uint64_t> next_id{1};
    std::atomic<bool>          shutting_down{false};

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", kJson);
    });

    // Prometheus text format. Served from an HTTP thread while the scheduler thread is
    // running, so everything here comes from atomic snapshots -- Scheduler::stats() and
    // RequestQueue's own locking -- never from reaching into scheduler internals.
    server.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        const Scheduler::Stats s = scheduler.stats();
        std::string            out;

        const auto counter = [&out](const char* name, const char* help, auto value) {
            out += "# HELP microvllm_";
            out += name;
            out += ' ';
            out += help;
            out += "\n# TYPE microvllm_";
            out += name;
            out += " counter\nmicrovllm_";
            out += name;
            out += ' ';
            out += std::to_string(value);
            out += '\n';
        };
        const auto gauge = [&out](const char* name, const char* help, auto value) {
            out += "# HELP microvllm_";
            out += name;
            out += ' ';
            out += help;
            out += "\n# TYPE microvllm_";
            out += name;
            out += " gauge\nmicrovllm_";
            out += name;
            out += ' ';
            out += std::to_string(value);
            out += '\n';
        };

        counter("requests_admitted_total", "Sequences admitted to the batch.", s.admitted);
        counter("requests_completed_total", "Sequences run to completion.", s.completed);
        counter("preemptions_total", "Sequences evicted to free KV cache and requeued.",
                s.preemptions);
        counter("admissions_deferred_total", "Admissions delayed because the KV pool was full.",
                s.deferred);
        counter("prefix_cache_hits_total", "Requests that inherited a cached prompt prefix.",
                s.prefix_hits);
        counter("prefix_tokens_saved_total", "Prompt tokens not prefilled thanks to sharing.",
                s.prefix_tokens_saved);

        counter("prompt_tokens_total", "Input tokens served.", s.prompt_tokens);
        counter("completion_tokens_total", "Output tokens generated.", s.completion_tokens);

        // Prometheus histograms are cumulative: each bucket counts observations <= its
        // bound, which is what lets a scraper interpolate percentiles.
        const auto histogram = [&out](const char* name, const char* help,
                                      const LatencyHistogram& h) {
            out += "# HELP microvllm_";
            out += name;
            out += ' ';
            out += help;
            out += "\n# TYPE microvllm_";
            out += name;
            out += " histogram\n";
            std::uint64_t cumulative = 0;
            for (std::size_t i = 0; i < LatencyHistogram::kBounds.size(); ++i) {
                cumulative += h.bucket(i);
                out += "microvllm_";
                out += name;
                out += "_bucket{le=\"";
                out += std::to_string(LatencyHistogram::kBounds[i] / 1000.0);
                out += "\"} ";
                out += std::to_string(cumulative);
                out += '\n';
            }
            out += "microvllm_";
            out += name;
            out += "_bucket{le=\"+Inf\"} ";
            out += std::to_string(h.count());
            out += "\nmicrovllm_";
            out += name;
            out += "_sum ";
            out += std::to_string(h.sum_seconds());
            out += "\nmicrovllm_";
            out += name;
            out += "_count ";
            out += std::to_string(h.count());
            out += '\n';
        };
        histogram("ttft_seconds", "Time from admission to first token.",
                  scheduler.ttft_histogram());
        histogram("request_duration_seconds", "End-to-end request latency including queueing.",
                  scheduler.e2e_histogram());

        gauge("queue_depth", "Requests waiting to be admitted.", queue.size());
        gauge("kv_blocks_total", "KV-cache blocks in the pool.", s.kv_blocks_total);
        gauge("kv_blocks_used", "KV-cache blocks currently allocated.", s.kv_blocks_used);
        gauge("kv_utilization",
              "Fraction of the KV pool in use.",
              s.kv_blocks_total == 0
                  ? 0.0
                  : static_cast<double>(s.kv_blocks_used) / s.kv_blocks_total);

        res.set_content(out, "text/plain; version=0.0.4");
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

        // Note: context-length admission control happens on the scheduler thread, which
        // owns the engine. Tokenizing here to pre-check would race with generation --
        // IModelEngine is single-threaded by contract.

        // Keep a copy of the cancel flag and the future: `r` is moved into the queue.
        auto      cancel = std::make_shared<std::atomic<bool>>(false);
        const auto r_id  = next_id.fetch_add(1, std::memory_order_relaxed);
        Request   r;
        r.id     = r_id;
        r.spec   = std::move(spec);
        r.cancel = cancel;
        r.timing.enqueued = Clock::now();
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
        if (config.log_requests) {
            // One line per request: cheap enough to leave on, and the token counts double
            // as usage metering.
            //
            // Flushed explicitly. stdout is block-buffered when it is not a terminal --
            // which is precisely how a server runs in production, redirected to a file or
            // a log collector -- so without this the lines sit in the buffer and are lost
            // entirely if the process is killed.
            std::string line = make_request_log(r_id, result);
            line += "\n";
            std::fputs(line.c_str(), stdout);
            std::fflush(stdout);
        }
        if (result.reason == FinishReason::kContextOverflow) {
            res.status = 400;  // the client asked for more than the context can hold
            res.set_content(make_error_response(result.error), kJson);
            return;
        }
        if (result.reason == FinishReason::kError) {
            res.status = 500;
            res.set_content(make_error_response(result.error), kJson);
            return;
        }
        res.set_content(make_generate_response(result.text, result.reason, result.usage), kJson);
    });

    // POST /generate/stream -- Server-Sent Events, one event per text delta.
    //
    // This is what the Phase 1 ITokenSink abstraction was for: the generator already
    // streams text deltas with stop strings correctly withheld, so streaming needed no
    // change to the scheduler, generator, or engine -- only a different sink.
    server.Post("/generate/stream", [&](const httplib::Request& req, httplib::Response& res) {
        RequestSpec spec;
        try {
            spec = parse_generate_request(req.body);
        } catch (const BadRequest& e) {
            res.status = 400;
            res.set_content(make_error_response(e.message), kJson);
            return;
        }

        auto cancel = std::make_shared<std::atomic<bool>>(false);
        auto sink   = std::make_shared<StreamingSink>(cancel);

        Request r;
        r.id     = next_id.fetch_add(1, std::memory_order_relaxed);
        r.spec   = std::move(spec);
        r.cancel = cancel;
        r.timing.enqueued = Clock::now();
        r.sink   = sink;
        std::future<GenResult> fut = r.result.get_future();

        if (!queue.try_push(r)) {
            res.status = 503;
            res.set_content(make_error_response("server at capacity; retry later"), kJson);
            return;
        }

        // Keep the future alive for the provider's lifetime so the scheduler always has
        // somewhere to put the result, even if the client vanishes mid-stream.
        auto shared_fut = std::make_shared<std::future<GenResult>>(std::move(fut));

        res.set_chunked_content_provider(
            "text/event-stream",
            [sink, cancel, shared_fut, &shutting_down](std::size_t, httplib::DataSink& out) {
                const StreamingSink::Event ev = sink->next();  // blocks for the next delta

                if (!ev.terminal) {
                    const std::string chunk = "data: " + make_stream_delta(ev.text) + "\n\n";
                    if (!out.write(chunk.data(), chunk.size())) {
                        cancel->store(true, std::memory_order_relaxed);  // client hung up
                        return false;
                    }
                    // A disconnect is often only visible on the *next* write, so also
                    // honour a server drain here rather than streaming into the void.
                    if (shutting_down.load(std::memory_order_relaxed)) {
                        cancel->store(true, std::memory_order_relaxed);
                    }
                    return true;
                }

                const std::string body = ev.reason == FinishReason::kError
                                             ? make_error_response(ev.error)
                                             : make_stream_done(ev.reason, ev.usage);
                const std::string tail = "data: " + body + "\n\ndata: [DONE]\n\n";
                out.write(tail.data(), tail.size());
                out.done();
                return false;  // stream complete
            });
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
