#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "microvllm/model_engine.hpp"

namespace microvllm {

// A deterministic IModelEngine with no tensor backend, used to test the generator,
// scheduler, queue, and block allocator without loading a model.
//
// Tokenization is byte-level and trivially reversible: token id == byte value, and
// piece(id) is that single byte. This round-trips exactly and makes stop-string tests
// precise, which is all a mock needs -- realism is the real engine's job.
//
// Generation ignores the prompt and replays a fixed `response` string, one byte-token
// per decode step, then emits an end-of-generation token. Per-token latency is
// configurable so later phases can build timing-sensitive tests on top of it.
class MockModelEngine final : public IModelEngine {
public:
    struct Config {
        std::string               response;              // text the model "generates"
        bool                      echo_prompt = false;   // if set, each seq replays its own prompt
        std::chrono::microseconds token_latency{0};      // sleep per decode step
        std::uint32_t             n_ctx     = 4096;
        std::uint32_t             n_batch   = 512;
        std::uint32_t             n_seq_max = 16;
    };

    explicit MockModelEngine(Config config);

    [[nodiscard]] EngineCaps caps() const override;
    [[nodiscard]] std::vector<Token> tokenize(std::string_view text, bool add_special) override;
    [[nodiscard]] std::string piece(Token token) override;
    void begin_sequence(SeqId seq, const SamplingParams& params) override;
    [[nodiscard]] std::vector<GenStep> decode(std::span<const BatchItem> batch) override;
    void release_sequence(SeqId seq) override;
    void copy_sequence(SeqId src, SeqId dst, Pos n_tokens) override;

    // How many prompt tokens were copied rather than prefilled, across all sequences.
    // Tests use this to prove prefix sharing actually avoided work instead of merely
    // bookkeeping refcounts.
    [[nodiscard]] std::uint64_t tokens_copied() const { return tokens_copied_; }
    // Prompt tokens actually submitted to decode(), i.e. genuinely prefilled.
    [[nodiscard]] std::uint64_t tokens_prefilled() const { return tokens_prefilled_; }

    // Token id reserved for end-of-generation (outside the 0..255 byte range).
    static constexpr Token kEosToken = 256;
    // Token id prepended by tokenize(add_special=true).
    static constexpr Token kBosToken = 257;

private:
    struct SeqState {
        std::size_t        gen_index = 0;  // how many response bytes emitted so far
        std::vector<Token> echo;           // captured prompt (echo mode only)
    };

    Config                              config_;
    std::vector<Token>                  response_tokens_;  // config_.response as byte-tokens
    std::unordered_map<SeqId, SeqState> sequences_;
    std::uint64_t                       tokens_copied_    = 0;
    std::uint64_t                       tokens_prefilled_ = 0;
};

}  // namespace microvllm
