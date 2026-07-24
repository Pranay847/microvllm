#include "microvllm/mock_engine.hpp"

#include <gtest/gtest.h>

#include <string>

namespace microvllm {
namespace {

MockModelEngine make(std::string response) {
    return MockModelEngine(MockModelEngine::Config{.response = std::move(response)});
}

// Collect the sampled token for a single decode of one sequence at a given position.
GenStep step_one(MockModelEngine& eng, SeqId seq, const Token* tok, Pos pos) {
    const BatchItem item{.seq = seq, .tokens = {tok, 1}, .pos0 = pos, .sample = true};
    const BatchItem batch[] = {item};
    auto steps = eng.decode(batch);
    EXPECT_EQ(steps.size(), 1U);
    return steps.at(0);
}

TEST(MockEngine, TokenizeIsByteLevelAndReversible) {
    auto eng = make("");
    const std::string text = "Hi!";
    const auto tokens = eng.tokenize(text, /*add_special=*/false);

    ASSERT_EQ(tokens.size(), 3U);
    EXPECT_EQ(tokens[0], static_cast<Token>('H'));
    EXPECT_EQ(tokens[1], static_cast<Token>('i'));
    EXPECT_EQ(tokens[2], static_cast<Token>('!'));

    std::string round;
    for (const Token t : tokens) {
        round += eng.piece(t);
    }
    EXPECT_EQ(round, text);
}

TEST(MockEngine, AddSpecialPrependsBos) {
    auto eng = make("");
    const auto with = eng.tokenize("x", /*add_special=*/true);
    const auto without = eng.tokenize("x", /*add_special=*/false);

    ASSERT_EQ(with.size(), without.size() + 1);
    EXPECT_EQ(with.front(), MockModelEngine::kBosToken);
    // The BOS piece is not rendered as visible text.
    EXPECT_TRUE(eng.piece(MockModelEngine::kBosToken).empty());
}

TEST(MockEngine, CapsReflectConfig) {
    MockModelEngine eng(MockModelEngine::Config{
        .response = "", .n_ctx = 2048, .n_batch = 256, .n_seq_max = 8});
    const auto c = eng.caps();
    EXPECT_EQ(c.n_ctx, 2048U);
    EXPECT_EQ(c.n_batch, 256U);
    EXPECT_EQ(c.n_seq_max, 8U);
    EXPECT_EQ(c.eos, MockModelEngine::kEosToken);
}

TEST(MockEngine, DecodeReplaysResponseThenEmitsEos) {
    auto eng = make("Hi");
    eng.begin_sequence(0, SamplingParams{});

    const auto prompt = eng.tokenize("prompt", /*add_special=*/true);
    // Prefill: the whole prompt, sampling the first generated token.
    const BatchItem prefill{
        .seq = 0, .tokens = prompt, .pos0 = 0, .sample = true};
    const BatchItem batch[] = {prefill};
    auto first = eng.decode(batch);
    ASSERT_EQ(first.size(), 1U);
    EXPECT_EQ(first[0].token, static_cast<Token>('H'));
    EXPECT_FALSE(first[0].is_eog);

    Token t = first[0].token;
    const GenStep s2 = step_one(eng, 0, &t, 10);
    EXPECT_EQ(s2.token, static_cast<Token>('i'));
    EXPECT_FALSE(s2.is_eog);

    t = s2.token;
    const GenStep s3 = step_one(eng, 0, &t, 11);
    EXPECT_TRUE(s3.is_eog);
    EXPECT_EQ(s3.token, MockModelEngine::kEosToken);
}

TEST(MockEngine, EmptyResponseEmitsEosImmediately) {
    auto eng = make("");
    eng.begin_sequence(0, SamplingParams{});
    const auto prompt = eng.tokenize("hello", true);
    const BatchItem prefill{.seq = 0, .tokens = prompt, .pos0 = 0, .sample = true};
    const BatchItem batch[] = {prefill};
    auto first = eng.decode(batch);
    ASSERT_EQ(first.size(), 1U);
    EXPECT_TRUE(first[0].is_eog);
}

TEST(MockEngine, SequencesAreIndependent) {
    auto eng = make("AB");
    eng.begin_sequence(1, SamplingParams{});
    eng.begin_sequence(2, SamplingParams{});

    const auto p = eng.tokenize("p", true);
    const BatchItem a{.seq = 1, .tokens = p, .pos0 = 0, .sample = true};
    const BatchItem b{.seq = 2, .tokens = p, .pos0 = 0, .sample = true};

    // Advance seq 1 twice; seq 2 should still be at the start of the response.
    {
        const BatchItem batch[] = {a};
        EXPECT_EQ(eng.decode(batch)[0].token, static_cast<Token>('A'));
    }
    Token t = static_cast<Token>('A');
    step_one(eng, 1, &t, 1);  // seq 1 now past 'B'

    {
        const BatchItem batch[] = {b};
        EXPECT_EQ(eng.decode(batch)[0].token, static_cast<Token>('A'))
            << "seq 2 must not be affected by seq 1's progress";
    }
}

TEST(MockEngine, DecodeReturnsOneStepPerSamplingItem) {
    auto eng = make("Z");
    eng.begin_sequence(1, SamplingParams{});
    eng.begin_sequence(2, SamplingParams{});
    const auto p = eng.tokenize("p", true);
    const BatchItem items[] = {
        {.seq = 1, .tokens = p, .pos0 = 0, .sample = true},
        {.seq = 2, .tokens = p, .pos0 = 0, .sample = false},
    };
    auto steps = eng.decode(items);
    ASSERT_EQ(steps.size(), 1U) << "only sample=true items yield a GenStep";
    EXPECT_EQ(steps[0].seq, 1);
}

TEST(MockEngine, ReleasedSequenceCanRestart) {
    auto eng = make("Q");
    eng.begin_sequence(0, SamplingParams{});
    const auto p = eng.tokenize("p", true);
    const BatchItem batch[] = {{.seq = 0, .tokens = p, .pos0 = 0, .sample = true}};
    EXPECT_EQ(eng.decode(batch)[0].token, static_cast<Token>('Q'));

    eng.release_sequence(0);
    eng.begin_sequence(0, SamplingParams{});  // reuse the same id
    EXPECT_EQ(eng.decode(batch)[0].token, static_cast<Token>('Q'))
        << "a reused sequence id starts the response over";
}

}  // namespace
}  // namespace microvllm
