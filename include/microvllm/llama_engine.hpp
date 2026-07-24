#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "microvllm/model_engine.hpp"

// This header fronts llama.cpp but does NOT include llama.h: the backend's types
// stay out of every other translation unit (pimpl below), so only this library ever
// compiles against llama.cpp.

namespace microvllm {

// RAII owner of llama.cpp's process-global backend state. Exactly one must exist,
// constructed before any LlamaModelEngine and destroyed after all of them.
class LlamaBackend {
public:
    LlamaBackend();
    ~LlamaBackend();

    LlamaBackend(const LlamaBackend&)            = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;
    LlamaBackend(LlamaBackend&&)                 = delete;
    LlamaBackend& operator=(LlamaBackend&&)      = delete;

    // Which SIMD paths ggml compiled in (AVX2/FMA/F16C on this machine; no AVX-512).
    [[nodiscard]] static const char* system_info() noexcept;

    // Silence llama.cpp's info/debug logging (leaves warnings and errors). Useful so
    // the server's own logs are not drowned out by per-token backend chatter.
    static void quiet_logging() noexcept;
};

struct LlamaEngineConfig {
    std::string   model_path;
    std::uint32_t n_ctx        = 4096;
    std::uint32_t n_batch      = 2048;  // one prefill fits if prompt <= n_batch (Phase 4 chunks)
    std::uint32_t n_seq_max    = 16;
    std::int32_t  n_threads    = 8;     // the operating point chosen in Phase 0
    std::int32_t  n_gpu_layers = 0;     // CPU-only on this hardware
};

// IModelEngine backed by llama.cpp. Owns one model + one context; a single engine
// thread must own the instance (llama_context is not thread-safe). Assumes a
// LlamaBackend is alive.
class LlamaModelEngine final : public IModelEngine {
public:
    explicit LlamaModelEngine(const LlamaEngineConfig& config);
    ~LlamaModelEngine() override;

    LlamaModelEngine(const LlamaModelEngine&)            = delete;
    LlamaModelEngine& operator=(const LlamaModelEngine&) = delete;
    LlamaModelEngine(LlamaModelEngine&&)                 = delete;
    LlamaModelEngine& operator=(LlamaModelEngine&&)      = delete;

    [[nodiscard]] EngineCaps caps() const override;
    [[nodiscard]] std::vector<Token> tokenize(std::string_view text, bool add_special) override;
    [[nodiscard]] std::string piece(Token token) override;
    void begin_sequence(SeqId seq, const SamplingParams& params) override;
    [[nodiscard]] std::vector<GenStep> decode(std::span<const BatchItem> batch) override;
    void release_sequence(SeqId seq) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microvllm
