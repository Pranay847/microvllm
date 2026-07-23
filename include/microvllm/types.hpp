#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace microvllm {

// A token id as produced by the tokenizer. Matches llama_token, but first-party
// code does not include llama.h -- the whole point of IModelEngine is that the
// scheduler, queue, and allocator never see the backend.
using Token = std::int32_t;

// Position of a token within its own sequence. Matches llama_pos.
using Pos = std::int32_t;

// Identifies a slot in the model context's KV cache. Recycled when a sequence
// completes, so it is NOT stable across a request's lifetime the way RequestId is.
using SeqId = std::int32_t;

// Stable, monotonically increasing, unique for the process lifetime.
using RequestId = std::uint64_t;

// Index into the KV block pool. See BlockAllocator.
using BlockId = std::uint32_t;

enum class FinishReason : std::uint8_t {
    kUnset,      // still running
    kEos,        // model emitted an end-of-generation token
    kMaxTokens,  // hit the request's max_tokens
    kStopString, // matched one of the request's stop strings
    kCancelled,  // client disconnected or the server is draining
    kError,      // backend failure
};

const char* to_string(FinishReason) noexcept;

struct SamplingParams {
    float         temperature = 0.8F;
    float         top_p       = 0.95F;
    std::int32_t  top_k       = 40;
    float         repeat_penalty = 1.1F;
    std::uint32_t seed        = 0;  // 0 => nondeterministic
};

struct RequestSpec {
    std::string              prompt;
    std::vector<std::string> stop;
    std::uint32_t            max_tokens = 128;
    SamplingParams           sampling{};
};

// Per-request accounting. Doubles as the unit of usage metering.
struct Usage {
    std::uint32_t prompt_tokens     = 0;
    std::uint32_t completion_tokens = 0;
    // Tokens served from a shared prefix block instead of being prefilled.
    std::uint32_t cached_prompt_tokens = 0;

    [[nodiscard]] std::uint32_t total_tokens() const noexcept {
        return prompt_tokens + completion_tokens;
    }
};

}  // namespace microvllm
