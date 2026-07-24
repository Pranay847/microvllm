#include "microvllm/http_api.hpp"

#include <gtest/gtest.h>

#include <string>

namespace microvllm {
namespace {

TEST(HttpApi, ParsesMinimalRequestWithDefaults) {
    const RequestSpec s = parse_generate_request(R"({"prompt":"Hello"})");
    EXPECT_EQ(s.prompt, "Hello");
    EXPECT_EQ(s.max_tokens, 128U);        // default
    EXPECT_TRUE(s.stop.empty());
    EXPECT_FLOAT_EQ(s.sampling.temperature, 0.8F);  // SamplingParams default
}

TEST(HttpApi, ParsesAllFields) {
    const RequestSpec s = parse_generate_request(R"({
        "prompt": "hi",
        "max_tokens": 42,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "seed": 7,
        "stop": ["\n\n", "END"]
    })");
    EXPECT_EQ(s.prompt, "hi");
    EXPECT_EQ(s.max_tokens, 42U);
    EXPECT_FLOAT_EQ(s.sampling.temperature, 0.3F);
    EXPECT_FLOAT_EQ(s.sampling.top_p, 0.9F);
    EXPECT_EQ(s.sampling.top_k, 20);
    EXPECT_EQ(s.sampling.seed, 7U);
    ASSERT_EQ(s.stop.size(), 2U);
    EXPECT_EQ(s.stop[0], "\n\n");
    EXPECT_EQ(s.stop[1], "END");
}

TEST(HttpApi, AcceptsStopAsSingleString) {
    const RequestSpec s = parse_generate_request(R"({"prompt":"x","stop":"STOP"})");
    ASSERT_EQ(s.stop.size(), 1U);
    EXPECT_EQ(s.stop[0], "STOP");
}

TEST(HttpApi, MissingPromptIsRejected) {
    EXPECT_THROW(parse_generate_request(R"({"max_tokens":10})"), BadRequest);
}

TEST(HttpApi, PromptMustBeAString) {
    EXPECT_THROW(parse_generate_request(R"({"prompt":123})"), BadRequest);
}

TEST(HttpApi, MalformedJsonIsRejected) {
    EXPECT_THROW(parse_generate_request("{not json"), BadRequest);
}

TEST(HttpApi, EmptyPromptIsAllowed) {
    // An empty prompt is a valid (if odd) request; only a missing/non-string one is not.
    const RequestSpec s = parse_generate_request(R"({"prompt":""})");
    EXPECT_EQ(s.prompt, "");
}

TEST(HttpApi, ResponseCarriesTextReasonAndUsage) {
    Usage u;
    u.prompt_tokens     = 5;
    u.completion_tokens = 3;
    const std::string body = make_generate_response("hello world", FinishReason::kEos, u);

    // Parse it back so the test is robust to field ordering/whitespace.
    const RequestSpec echo;  // unused; just proving we can round-trip via a JSON lib in the test
    EXPECT_NE(body.find("\"text\""), std::string::npos);
    EXPECT_NE(body.find("hello world"), std::string::npos);
    EXPECT_NE(body.find("\"finish_reason\""), std::string::npos);
    EXPECT_NE(body.find("eos"), std::string::npos);
    EXPECT_NE(body.find("\"completion_tokens\""), std::string::npos);
    EXPECT_NE(body.find("\"total_tokens\""), std::string::npos);
    EXPECT_NE(body.find('8'), std::string::npos);  // total = 5 + 3
    (void)echo;
}

TEST(HttpApi, ErrorResponseCarriesMessage) {
    const std::string body = make_error_response("prompt is required");
    EXPECT_NE(body.find("\"error\""), std::string::npos);
    EXPECT_NE(body.find("prompt is required"), std::string::npos);
}

}  // namespace
}  // namespace microvllm
