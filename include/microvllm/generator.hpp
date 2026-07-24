#pragma once

#include "microvllm/model_engine.hpp"
#include "microvllm/token_sink.hpp"
#include "microvllm/types.hpp"

namespace microvllm {

// Drive one request to completion against an engine, streaming output to a sink.
//
// This is the per-request state machine in its simplest, run-to-completion form:
// tokenize the prompt, prefill it, then loop decode -> emit -> check-stops until an
// end condition. Phase 1 calls it synchronously, one request at a time. The scheduler
// in later phases interleaves the same steps across many sequences, but the stop and
// accounting logic proven here is the same.
//
// `seq` is the KV-cache slot the caller has assigned to this request; the caller owns
// its lifetime (Phase 1 uses a single fixed slot). The generator calls begin_sequence
// and release_sequence around its work.
//
// Stop conditions, in priority order per step: engine end-of-generation token, a
// configured stop string, max_tokens, and sink cancellation. Stop strings can span
// token boundaries, so output is held back by up to (longest stop - 1) bytes until a
// match is ruled out; the held-back tail is flushed on a clean finish. This keeps the
// streamed deltas and the final text identical even when a stop is truncated mid-token.
void generate_one(IModelEngine& engine, SeqId seq, const RequestSpec& spec, ITokenSink& sink);

}  // namespace microvllm
