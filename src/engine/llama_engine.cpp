#include "microvllm/llama_engine.hpp"

#include <llama.h>

namespace microvllm {

LlamaBackend::LlamaBackend() {
    llama_backend_init();
}

LlamaBackend::~LlamaBackend() {
    llama_backend_free();
}

const char* LlamaBackend::system_info() noexcept {
    return llama_print_system_info();
}

}  // namespace microvllm
