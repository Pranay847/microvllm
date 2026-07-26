#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace microvllm {

using Clock   = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Per-request timing, captured at the four points that matter for serving.
//
// These are the numbers a serving system is actually judged on, and they decompose
// end-to-end latency into parts with different causes:
//
//   queue wait -- how long the request sat before the scheduler had capacity. Rises with
//                 load and with a constrained KV pool; it is a scheduling signal, not a
//                 model one.
//   TTFT       -- admission to first token. Dominated by prefill, so it scales with prompt
//                 length and is what prefix caching and chunked prefill move.
//   TPOT       -- mean time per output token after the first. Dominated by decode, so it
//                 scales with batch occupancy and memory bandwidth.
//
// Splitting them is what makes a regression diagnosable: a jump in queue wait means the
// scheduler is starved, a jump in TTFT means prefill got more expensive, a jump in TPOT
// means decode did.
//
// The enqueue stamp is taken on an HTTP thread and the rest on the scheduler thread. The
// handoff is through RequestQueue's mutex, which establishes the happens-before edge, so
// no atomics are needed here.
struct RequestTiming {
    TimePoint                enqueued{};     // HTTP thread, before the request is queued
    std::optional<TimePoint> admitted;       // scheduler thread, when a slot was taken
    std::optional<TimePoint> first_token;    // scheduler thread, first sampled token
    std::optional<TimePoint> finished;       // scheduler thread, terminal state reached

    [[nodiscard]] static double ms(TimePoint a, TimePoint b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    // Time spent waiting for capacity. Includes any re-queue after a preemption, which is
    // deliberate: from the client's perspective that time was still spent waiting.
    [[nodiscard]] double queue_wait_ms() const {
        return admitted ? ms(enqueued, *admitted) : 0.0;
    }

    // Admission to first token. Measured from admission rather than enqueue so it reflects
    // prefill cost rather than queueing; queue_wait_ms covers the other part.
    [[nodiscard]] double ttft_ms() const {
        return (admitted && first_token) ? ms(*admitted, *first_token) : 0.0;
    }

    // Mean time per output token after the first. Zero for a single-token response, since
    // there is no inter-token interval to measure.
    [[nodiscard]] double tpot_ms(std::uint32_t completion_tokens) const {
        if (!first_token || !finished || completion_tokens < 2) {
            return 0.0;
        }
        return ms(*first_token, *finished) / static_cast<double>(completion_tokens - 1);
    }

    [[nodiscard]] double total_ms() const {
        return finished ? ms(enqueued, *finished) : 0.0;
    }

    // Output tokens per second over the whole request, the client-visible rate.
    [[nodiscard]] double output_tokens_per_sec(std::uint32_t completion_tokens) const {
        const double total = total_ms();
        return (total <= 0.0 || completion_tokens == 0)
                   ? 0.0
                   : static_cast<double>(completion_tokens) / (total / 1000.0);
    }
};

}  // namespace microvllm
