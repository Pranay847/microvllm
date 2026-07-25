#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "microvllm/types.hpp"

namespace microvllm {

// Fixed properties of a loaded model, read once at startup.
struct EngineCaps {
    std::uint32_t n_ctx     = 0;  // total KV-cache positions across all sequences
    std::uint32_t n_batch   = 0;  // max tokens in one llama_decode call
    std::uint32_t n_seq_max = 0;  // max concurrent sequences
    Token         eos       = -1; // primary end-of-sequence token (informational)

    // Positions available to a SINGLE sequence -- the real limit on prompt + completion
    // length. llama.cpp divides the KV pool across n_seq_max, so this is typically
    // n_ctx / n_seq_max and NOT n_ctx. Mixing the two up means a request that looks like
    // it fits gets rejected deep inside the backend with an opaque decode failure.
    std::uint32_t n_ctx_seq = 0;
};

// One sequence's contribution to a single decode step.
//
// PREFILL:  tokens = the whole prompt (or a chunk of it), pos0 = position of tokens[0].
// DECODE:   tokens = the single previously-sampled token, pos0 = its position.
//
// `sample` marks the sequences whose next token should be produced from this step.
// A non-final chunked-prefill piece sets sample=false (Phase 4); a decode step and a
// final prefill piece set sample=true. Every item with sample=true yields one GenStep.
struct BatchItem {
    SeqId                  seq;
    std::span<const Token> tokens;
    Pos                    pos0;
    bool                   sample = true;
};

// The token an engine produced for one sequence in a decode step.
struct GenStep {
    SeqId seq;
    Token token;
    bool  is_eog;  // true if `token` ends generation (EOS / end-of-turn / etc.)
};

// The seam between the serving system and the tensor backend.
//
// Everything above this interface -- generator, scheduler, queue, block allocator --
// is backend-agnostic and links only microvllm_core. Two implementations exist:
// LlamaModelEngine (llama.cpp) and MockModelEngine (deterministic, no backend). That
// split is what lets the scheduler be unit-tested and ThreadSanitizer-checked with no
// model download.
//
// Sampling lives behind this interface on purpose: llama.cpp couples sampling to its
// context, and the mock can produce deterministic "sampled" tokens with no logits to
// fabricate. Each sequence owns its own sampler state, created by begin_sequence and
// destroyed by release_sequence.
//
// Threading: a single engine thread owns the implementation. No method is required to
// be thread-safe. llama.cpp's context is not.
class IModelEngine {
public:
    virtual ~IModelEngine() = default;

    [[nodiscard]] virtual EngineCaps caps() const = 0;

    // Text -> token ids. add_special prepends BOS etc. per the model's convention.
    [[nodiscard]] virtual std::vector<Token> tokenize(std::string_view text,
                                                       bool add_special) = 0;

    // One token id -> its text piece. May be empty (control tokens) or a partial
    // UTF-8 fragment for byte-fallback tokens; callers concatenate before display.
    [[nodiscard]] virtual std::string piece(Token token) = 0;

    // Register a sequence and its sampling parameters before its first decode.
    virtual void begin_sequence(SeqId seq, const SamplingParams& params) = 0;

    // Advance every sampling sequence by one token. Returns one GenStep per BatchItem
    // with sample=true, in the same order. Throws on backend failure.
    [[nodiscard]] virtual std::vector<GenStep> decode(std::span<const BatchItem> batch) = 0;

    // Release a sequence: free its sampler and its KV-cache slot. Idempotent.
    virtual void release_sequence(SeqId seq) = 0;
};

}  // namespace microvllm
