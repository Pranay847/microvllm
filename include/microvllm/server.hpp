#pragma once

#include <string>

#include "microvllm/model_engine.hpp"

namespace microvllm {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int         port = 8080;
};

// Serve POST /generate and GET /health against `engine`, blocking until the process
// is signalled. Phase 1 processes requests serially: cpp-httplib dispatches on a
// thread pool, but a mutex guards the engine because llama_context is not thread-safe.
// The Phase 2 queue/worker split replaces that mutex.
//
// Returns true if the listener bound and ran, false if it failed to bind.
bool serve(IModelEngine& engine, const ServerConfig& config);

}  // namespace microvllm
