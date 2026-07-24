#include "microvllm/server.hpp"

#include <httplib.h>

#include <cstdio>
#include <mutex>

#include "microvllm/collecting_sink.hpp"
#include "microvllm/generator.hpp"
#include "microvllm/http_api.hpp"

namespace microvllm {
namespace {

constexpr char kJson[] = "application/json";

// Phase 1 runs every request on a single fixed KV-cache slot, under a lock. The
// scheduler in later phases assigns slots dynamically; here one is enough because the
// engine is used strictly serially.
constexpr SeqId kSeq = 0;

}  // namespace

bool serve(IModelEngine& engine, const ServerConfig& config) {
    httplib::Server server;
    std::mutex engine_mu;  // llama_context is not thread-safe; serialize all use.

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

        try {
            CollectingSink sink;
            {
                const std::lock_guard<std::mutex> lock(engine_mu);
                generate_one(engine, kSeq, spec, sink);
            }
            if (sink.reason() == FinishReason::kError) {
                res.status = 500;
                res.set_content(make_error_response(sink.error()), kJson);
                return;
            }
            res.set_content(make_generate_response(sink.text(), sink.reason(), sink.usage()), kJson);
        } catch (const std::exception& e) {
            // Backend failure mid-generation (e.g. llama_decode error). The engine's
            // KV state for this slot may be dirty; clear it so the next request is clean.
            {
                const std::lock_guard<std::mutex> lock(engine_mu);
                engine.release_sequence(kSeq);
            }
            res.status = 500;
            res.set_content(make_error_response(e.what()), kJson);
        }
    });

    std::printf("microvllm listening on http://%s:%d\n", config.host.c_str(), config.port);
    return server.listen(config.host, config.port);
}

}  // namespace microvllm
