#include <cstdio>

#include "microvllm/types.hpp"
#include "microvllm/version.hpp"

#if MICROVLLM_HAS_LLAMA
#include "microvllm/llama_engine.hpp"
#endif

// Phase 0: a smoke test that the whole link graph works -- first-party code,
// llama.cpp, and the fetched dependencies. The HTTP server lands in Phase 1.
int main() {
    std::printf("microvllm %s\n", microvllm::kVersion);
    std::printf("finish-reason sanity: %s\n",
                microvllm::to_string(microvllm::FinishReason::kMaxTokens));

#if MICROVLLM_HAS_LLAMA
    const microvllm::LlamaBackend backend;
    std::printf("llama.cpp backend: %s\n", microvllm::LlamaBackend::system_info());
#else
    std::printf("llama.cpp backend: disabled at build time\n");
#endif

    return 0;
}
