#include "microvllm/types.hpp"

namespace microvllm {

const char* to_string(FinishReason r) noexcept {
    switch (r) {
        case FinishReason::kUnset:      return "unset";
        case FinishReason::kEos:        return "eos";
        case FinishReason::kMaxTokens:  return "max_tokens";
        case FinishReason::kStopString: return "stop_string";
        case FinishReason::kCancelled:  return "cancelled";
        case FinishReason::kError:      return "error";
        case FinishReason::kContextOverflow: return "context_overflow";
    }
    return "unknown";
}

}  // namespace microvllm
