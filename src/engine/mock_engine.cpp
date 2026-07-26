#include "microvllm/mock_engine.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace microvllm {

MockModelEngine::MockModelEngine(Config config) : config_(std::move(config)) {
    response_tokens_.reserve(config_.response.size());
    for (const char c : config_.response) {
        response_tokens_.push_back(static_cast<Token>(static_cast<unsigned char>(c)));
    }
}

EngineCaps MockModelEngine::caps() const {
    return EngineCaps{
        .n_ctx     = config_.n_ctx,
        .n_batch   = config_.n_batch,
        .n_seq_max = config_.n_seq_max,
        .eos       = kEosToken,
        // The mock imposes no real memory limit, so every sequence may use the whole
        // context. Tests that need a constrained per-sequence budget set n_ctx directly.
        .n_ctx_seq = config_.n_ctx,
    };
}

std::vector<Token> MockModelEngine::tokenize(std::string_view text, bool add_special) {
    std::vector<Token> tokens;
    tokens.reserve(text.size() + (add_special ? 1 : 0));
    if (add_special) {
        tokens.push_back(kBosToken);
    }
    for (const char c : text) {
        tokens.push_back(static_cast<Token>(static_cast<unsigned char>(c)));
    }
    return tokens;
}

std::string MockModelEngine::piece(Token token) {
    // Control tokens render as nothing; byte tokens render as their single byte.
    if (token == kEosToken || token == kBosToken) {
        return {};
    }
    if (token >= 0 && token <= 0xFF) {
        return std::string(1, static_cast<char>(static_cast<unsigned char>(token)));
    }
    return {};
}

void MockModelEngine::begin_sequence(SeqId seq, const SamplingParams&) {
    sequences_[seq] = SeqState{};
}

std::vector<GenStep> MockModelEngine::decode(std::span<const BatchItem> batch) {
    std::vector<GenStep> steps;
    for (const BatchItem& item : batch) {
        if (config_.token_latency.count() > 0) {
            std::this_thread::sleep_for(config_.token_latency);
        }

        auto it = sequences_.find(item.seq);
        if (it == sequences_.end()) {
            throw std::logic_error("MockModelEngine::decode on a sequence without begin_sequence");
        }
        SeqState& state = it->second;

        if (state.gen_index == 0) {
            tokens_prefilled_ += item.tokens.size();  // work a shared prefix would avoid
        }

        // Echo mode: accumulate the prompt (minus BOS) as this sequence's response.
        //
        // This runs for EVERY item, before the sampling check, because that is what a
        // real engine does: all tokens in the batch are processed and populate the KV
        // cache, and only the flagged ones produce logits. Capturing solely on sampling
        // items would miss every intermediate chunk of a chunked prefill.
        if (config_.echo_prompt && state.gen_index == 0) {
            for (const Token t : item.tokens) {
                if (t != kBosToken) {
                    state.echo.push_back(t);
                }
            }
        }

        if (!item.sample) {
            continue;  // KV populated, but no token produced for this item
        }
        const std::vector<Token>& response = config_.echo_prompt ? state.echo : response_tokens_;

        if (state.gen_index < response.size()) {
            steps.push_back(
                GenStep{.seq = item.seq, .token = response[state.gen_index], .is_eog = false});
            ++state.gen_index;
        } else {
            steps.push_back(GenStep{.seq = item.seq, .token = kEosToken, .is_eog = true});
        }
    }
    return steps;
}

void MockModelEngine::release_sequence(SeqId seq) {
    sequences_.erase(seq);
}

void MockModelEngine::copy_sequence(SeqId src, SeqId dst, Pos n_tokens) {
    if (n_tokens <= 0 || src == dst) {
        return;
    }
    const auto s = sequences_.find(src);
    const auto d = sequences_.find(dst);
    if (s == sequences_.end() || d == sequences_.end()) {
        return;
    }
    // Model the real effect: `dst` inherits the prefix without prefilling it. In echo
    // mode that means adopting the copied prompt tokens, exactly as a real backend
    // inherits their KV cells.
    const auto take = std::min(static_cast<std::size_t>(n_tokens), s->second.echo.size());
    d->second.echo.assign(s->second.echo.begin(), s->second.echo.begin() +
                                                      static_cast<std::ptrdiff_t>(take));
    tokens_copied_ += take;
}

}  // namespace microvllm
