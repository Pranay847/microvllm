#pragma once

#include <string_view>

#include "microvllm/types.hpp"

namespace microvllm {

// Where a request's generated output goes. Defined now (Phase 1) so the blocking
// endpoint and the Phase 4 SSE endpoint are the same code path with different sinks.
//
// The generator is the single writer. It streams output as *text deltas*, not as
// (token, piece) pairs: SSE consumers want text, and stop-string handling has to
// hold back a few characters (a stop can span a token boundary), so emitted deltas
// do not line up with token boundaries anyway. Token counts are reported separately
// via Usage. A sink therefore never has to reassemble pieces or know about tokens.
//
// Call sequence for one request, guaranteed by the generator:
//   on_text* ( on_finish | on_error )
// exactly one terminal call, and never after cancelled() has returned true and been
// honoured. All calls happen on the generating thread.
class ITokenSink {
public:
    virtual ~ITokenSink() = default;

    // A chunk of newly finalized output text. Concatenating every delta yields the
    // full response body. May be called with an empty view; consumers must tolerate that.
    virtual void on_text(std::string_view delta) = 0;

    // Terminal: generation stopped normally. reason says why; usage is final.
    virtual void on_finish(FinishReason reason, const Usage& usage) = 0;

    // Terminal: generation aborted before completing (backend failure, bad request).
    virtual void on_error(std::string_view message) = 0;

    // Polled by the generator between steps. Returning true asks it to stop as soon
    // as possible; it will then deliver on_finish with FinishReason::kCancelled.
    // Used for client disconnect and server drain. Must be cheap and thread-safe.
    [[nodiscard]] virtual bool cancelled() const = 0;
};

}  // namespace microvllm
