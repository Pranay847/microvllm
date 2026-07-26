#pragma once

#include <string>
#include <string_view>

#include "microvllm/request.hpp"
#include "microvllm/types.hpp"

// The request/response contract for POST /generate, as pure string<->value functions.
// JSON is an implementation detail here (nlohmann/json lives in the .cpp), so this
// header stays dependency-free and the parsing logic is unit-testable without a socket.

namespace microvllm {

// Thrown by parse_generate_request on malformed input; the message is safe to return
// to the client as a 400 body.
struct BadRequest : std::exception {
    std::string message;
    explicit BadRequest(std::string m) : message(std::move(m)) {}
    [[nodiscard]] const char* what() const noexcept override { return message.c_str(); }
};

// Parse a POST /generate JSON body into a RequestSpec. Required: "prompt" (string).
// Optional: "max_tokens" (uint), "temperature"/"top_p"/"top_k"/"seed" (sampling),
// "stop" (string or array of strings). Throws BadRequest on anything malformed.
RequestSpec parse_generate_request(std::string_view body);

// Serialize a completed generation into the JSON response body.
std::string make_generate_response(std::string_view text, FinishReason reason, const Usage& usage);

// Serialize an error into a JSON body: {"error": message}.
std::string make_error_response(std::string_view message);

// One structured log line for a completed request: a single JSON object.
//
// JSON rather than prose so it can be piped straight into a log aggregator, and per
// request rather than aggregated so the tail is visible -- an average latency hides
// exactly the requests worth investigating. The token counts double as usage metering.
std::string make_request_log(RequestId id, const GenResult& result);

// One incremental chunk of a streamed response: {"text": delta}.
std::string make_stream_delta(std::string_view delta);

// The final event of a streamed response, carrying the reason and usage totals.
std::string make_stream_done(FinishReason reason, const Usage& usage);

}  // namespace microvllm
