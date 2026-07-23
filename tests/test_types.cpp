#include "microvllm/types.hpp"

#include <gtest/gtest.h>

#include <string_view>

namespace microvllm {
namespace {

TEST(FinishReason, EveryEnumeratorHasADistinctName) {
    const FinishReason all[] = {
        FinishReason::kUnset,      FinishReason::kEos,       FinishReason::kMaxTokens,
        FinishReason::kStopString, FinishReason::kCancelled, FinishReason::kError,
    };

    for (const FinishReason r : all) {
        const std::string_view name = to_string(r);
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name, "unknown") << "to_string() is missing a case";
    }

    // Distinctness: a copy-paste slip in the switch would otherwise go unnoticed.
    for (std::size_t i = 0; i < std::size(all); ++i) {
        for (std::size_t j = i + 1; j < std::size(all); ++j) {
            EXPECT_STRNE(to_string(all[i]), to_string(all[j]));
        }
    }
}

TEST(Usage, TotalIsPromptPlusCompletion) {
    Usage u;
    u.prompt_tokens     = 12;
    u.completion_tokens = 30;
    EXPECT_EQ(u.total_tokens(), 42U);
}

TEST(Usage, CachedPromptTokensDoNotDoubleCount) {
    // cached_prompt_tokens is a subset of prompt_tokens -- it records how many of
    // them came from a shared prefix block rather than being prefilled. It must
    // not inflate the billable total.
    Usage u;
    u.prompt_tokens         = 100;
    u.cached_prompt_tokens  = 64;
    u.completion_tokens     = 10;
    EXPECT_EQ(u.total_tokens(), 110U);
}

}  // namespace
}  // namespace microvllm
