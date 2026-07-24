#include "microvllm/mock_engine.hpp"

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
        if (!item.sample) {
            continue;
        }

        auto it = sequences_.find(item.seq);
        if (it == sequences_.end()) {
            throw std::logic_error("MockModelEngine::decode on a sequence without begin_sequence");
        }
        SeqState& state = it->second;

        // Echo mode: the first decode of a sequence is its prefill, so capture the
        // prompt (minus the BOS token) as this sequence's response.
        if (config_.echo_prompt && state.gen_index == 0 && state.echo.empty()) {
            for (const Token t : item.tokens) {
                if (t != kBosToken) {
                    state.echo.push_back(t);
                }
            }
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

}  // namespace microvllm
