// Chunked prefill: a long prompt split across several steps so it cannot stall every
// other sequence in a batch. The property that matters is that chunking is invisible in
// the output -- it changes when KV cache is populated, never what the model produces.
#include "microvllm/sequence_state.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "microvllm/collecting_sink.hpp"
#include "microvllm/mock_engine.hpp"

namespace microvllm {
namespace {

RequestSpec spec_with(std::string prompt, std::uint32_t max_tokens = 128) {
    RequestSpec s;
    s.prompt     = std::move(prompt);
    s.max_tokens = max_tokens;
    return s;
}

TEST(SequenceStatePrefill, WholePromptInOneChunkSamples) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    CollectingSink  sink;
    auto            prompt = engine.tokenize("hello", /*add_special=*/true);  // 6 tokens
    SequenceState   state(0, spec_with("hello"), sink, prompt);

    EXPECT_EQ(state.prefill_remaining(), 6U);
    EXPECT_TRUE(state.prefilling());

    const BatchItem item = state.take_prefill_chunk(64);
    EXPECT_EQ(item.tokens.size(), 6U);
    EXPECT_EQ(item.pos0, 0);
    EXPECT_TRUE(item.sample) << "the chunk that completes the prompt must sample";
    EXPECT_EQ(state.prefill_remaining(), 0U);
    EXPECT_FALSE(state.prefilling());
}

TEST(SequenceStatePrefill, ChunksCoverThePromptContiguouslyAndOnlyTheLastSamples) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    CollectingSink  sink;
    auto            prompt = engine.tokenize("abcdefghi", /*add_special=*/true);  // 10 tokens
    SequenceState   state(0, spec_with("abcdefghi"), sink, prompt);

    std::vector<Token> seen;
    Pos                expected_pos = 0;
    int                chunks       = 0;
    while (state.prefilling()) {
        const BatchItem item = state.take_prefill_chunk(4);
        EXPECT_EQ(item.pos0, expected_pos) << "chunks must be positionally contiguous";
        expected_pos += static_cast<Pos>(item.tokens.size());
        seen.insert(seen.end(), item.tokens.begin(), item.tokens.end());
        ++chunks;
        // Every chunk but the final one is pure KV population.
        EXPECT_EQ(item.sample, state.prefill_remaining() == 0);
    }

    EXPECT_EQ(chunks, 3) << "10 tokens at budget 4 -> 4 + 4 + 2";
    EXPECT_EQ(seen, prompt) << "chunks must reproduce the prompt exactly, in order";
}

TEST(SequenceStatePrefill, BudgetLargerThanPromptTakesWholePrompt) {
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    CollectingSink  sink;
    auto            prompt = engine.tokenize("hi", true);
    SequenceState   state(0, spec_with("hi"), sink, prompt);

    const BatchItem item = state.take_prefill_chunk(1000);
    EXPECT_EQ(item.tokens.size(), prompt.size());
    EXPECT_TRUE(item.sample);
}

TEST(SequenceStatePrefill, FirstDecodePositionFollowsThePromptRegardlessOfChunking) {
    // The decode position must depend only on prompt length, never on how the prefill
    // was split -- otherwise chunking would silently corrupt attention.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});

    CollectingSink sink_whole;
    auto           prompt = engine.tokenize("abcdefghi", true);
    SequenceState  whole(0, spec_with("abcdefghi"), sink_whole, prompt);
    (void)whole.take_prefill_chunk(1000);

    CollectingSink sink_chunked;
    SequenceState  chunked(1, spec_with("abcdefghi"), sink_chunked, prompt);
    while (chunked.prefilling()) {
        (void)chunked.take_prefill_chunk(3);
    }

    engine.begin_sequence(0, SamplingParams{});
    engine.begin_sequence(1, SamplingParams{});
    const GenStep step{.seq = 0, .token = static_cast<Token>('x'), .is_eog = false};
    EXPECT_TRUE(whole.accept(engine, step));
    EXPECT_TRUE(chunked.accept(engine, step));

    EXPECT_EQ(whole.decode_item().pos0, chunked.decode_item().pos0)
        << "chunking must not shift the decode position";
    EXPECT_EQ(whole.decode_item().pos0, static_cast<Pos>(prompt.size()));
}

TEST(SequenceStatePrefill, ChunkedGenerationProducesIdenticalOutput) {
    // End-to-end equivalence: drive the same request twice, once prefilled whole and once
    // in small chunks, and require byte-identical output and identical accounting.
    const std::string prompt_text = "the quick brown fox";

    auto drive = [&](std::size_t chunk_budget) {
        MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
        auto            sink   = std::make_unique<CollectingSink>();
        auto            prompt = engine.tokenize(prompt_text, true);
        SequenceState   state(0, spec_with(prompt_text), *sink, prompt);
        engine.begin_sequence(0, SamplingParams{});

        GenStep step{};
        while (state.prefilling()) {
            const BatchItem item = state.take_prefill_chunk(chunk_budget);
            const BatchItem batch[] = {item};
            const auto      steps   = engine.decode(batch);
            if (item.sample) {
                step = steps.at(0);
            }
        }
        while (state.accept(engine, step)) {
            const BatchItem batch[] = {state.decode_item()};
            step = engine.decode(batch).at(0);
        }
        state.finish();
        return sink;
    };

    const auto whole   = drive(1000);
    const auto chunked = drive(3);

    EXPECT_EQ(whole->text(), chunked->text()) << "chunking changed the generated text";
    EXPECT_EQ(whole->reason(), chunked->reason());
    EXPECT_EQ(whole->usage().completion_tokens, chunked->usage().completion_tokens);
    EXPECT_EQ(whole->usage().prompt_tokens, chunked->usage().prompt_tokens);
}

}  // namespace
}  // namespace microvllm
