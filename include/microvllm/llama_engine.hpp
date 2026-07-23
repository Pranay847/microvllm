#pragma once

// The only first-party header that fronts llama.cpp. It deliberately does NOT
// include llama.h -- keeping the backend's types out of every other translation
// unit is what lets the scheduler be tested against a mock engine.

namespace microvllm {

// RAII owner of llama.cpp's process-global backend state. Exactly one of these
// should exist for the lifetime of the server.
class LlamaBackend {
public:
    LlamaBackend();
    ~LlamaBackend();

    LlamaBackend(const LlamaBackend&)            = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;
    LlamaBackend(LlamaBackend&&)                 = delete;
    LlamaBackend& operator=(LlamaBackend&&)      = delete;

    // Which SIMD paths ggml actually compiled in. On this machine expect AVX2
    // and FMA but no AVX-512 (Core Ultra 7 155U).
    [[nodiscard]] static const char* system_info() noexcept;
};

}  // namespace microvllm
