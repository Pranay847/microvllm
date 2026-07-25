#include "microvllm/generator.hpp"

#include <array>

#include "microvllm/sequence_state.hpp"

namespace microvllm {

// A batch of one. All the per-token rules -- stop strings, token budget, cancellation,
// usage accounting -- live in SequenceState, which the batched scheduler drives the same
// way over many sequences at once.
void generate_one(IModelEngine& engine, SeqId seq, const RequestSpec& spec, ITokenSink& sink) {
    SequenceState state(seq, spec, sink, engine.tokenize(spec.prompt, /*add_special=*/true));

    engine.begin_sequence(seq, spec.sampling);

    const std::array<BatchItem, 1> prefill{state.prefill_item()};
    GenStep step = engine.decode(prefill).at(0);

    while (state.accept(engine, step)) {
        const std::array<BatchItem, 1> next{state.decode_item()};
        step = engine.decode(next).at(0);
    }

    engine.release_sequence(seq);
    state.finish();
}

}  // namespace microvllm
