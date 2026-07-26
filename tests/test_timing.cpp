// Per-request timing arithmetic.
//
// Pure derivation from four timestamps, so it is tested directly with synthetic clock
// values rather than by measuring anything -- a test that sleeps and asserts on wall time
// is flaky by construction.
#include "microvllm/timing.hpp"

#include <gtest/gtest.h>

namespace microvllm {
namespace {

// Build a timing record from millisecond offsets off a fixed epoch.
RequestTiming at(double enqueue_ms, double admit_ms, double first_ms, double finish_ms) {
    const TimePoint base{};
    const auto      off = [&](double ms) {
        return base + std::chrono::microseconds(static_cast<std::int64_t>(ms * 1000.0));
    };
    RequestTiming t;
    t.enqueued    = off(enqueue_ms);
    t.admitted    = off(admit_ms);
    t.first_token = off(first_ms);
    t.finished    = off(finish_ms);
    return t;
}

TEST(RequestTiming, DecomposesEndToEndLatency) {
    // enqueued 0, admitted 50, first token 250, finished 1050.
    const RequestTiming t = at(0, 50, 250, 1050);

    EXPECT_NEAR(t.queue_wait_ms(), 50.0, 1e-6);
    EXPECT_NEAR(t.ttft_ms(), 200.0, 1e-6) << "TTFT is admission -> first token, not enqueue";
    EXPECT_NEAR(t.total_ms(), 1050.0, 1e-6);

    // The parts must account for the whole: queue wait + TTFT + generation == total.
    const double generation = t.total_ms() - t.queue_wait_ms() - t.ttft_ms();
    EXPECT_NEAR(generation, 800.0, 1e-6);
}

TEST(RequestTiming, TpotIsMeanInterTokenInterval) {
    // 800 ms spent producing tokens 2..5, i.e. 4 intervals of 200 ms.
    const RequestTiming t = at(0, 0, 250, 1050);
    EXPECT_NEAR(t.tpot_ms(/*completion_tokens=*/5), 200.0, 1e-6);
}

TEST(RequestTiming, TpotIsZeroWhenThereIsNoInterval) {
    const RequestTiming t = at(0, 0, 250, 1050);
    EXPECT_EQ(t.tpot_ms(1), 0.0) << "one token has no inter-token interval to average";
    EXPECT_EQ(t.tpot_ms(0), 0.0);
}

TEST(RequestTiming, OutputRateUsesEndToEndTime) {
    // 20 tokens over 2 s of wall clock, queueing included -- the client-visible rate.
    const RequestTiming t = at(0, 100, 300, 2000);
    EXPECT_NEAR(t.output_tokens_per_sec(20), 10.0, 1e-6);
}

TEST(RequestTiming, IncompleteRecordsReportZeroRatherThanGarbage) {
    // A request that never reached a stage must not produce a nonsense duration from an
    // unset optional; metrics derived from it would be silently wrong.
    RequestTiming t;
    t.enqueued = TimePoint{};
    EXPECT_EQ(t.queue_wait_ms(), 0.0);
    EXPECT_EQ(t.ttft_ms(), 0.0);
    EXPECT_EQ(t.total_ms(), 0.0);
    EXPECT_EQ(t.tpot_ms(10), 0.0);
    EXPECT_EQ(t.output_tokens_per_sec(10), 0.0);

    // Admitted but never produced a token: queue wait is known, TTFT is not.
    t.admitted = t.enqueued + std::chrono::milliseconds(25);
    EXPECT_NEAR(t.queue_wait_ms(), 25.0, 1e-6);
    EXPECT_EQ(t.ttft_ms(), 0.0);
}

TEST(RequestTiming, QueueWaitIncludesTimeSpentAfterPreemption) {
    // A preempted request is re-admitted later; `admitted` reflects the final admission,
    // so its queue wait covers the whole time the client was waiting, not just the first
    // stretch. That is the number a client actually experiences.
    const RequestTiming t = at(0, 900, 950, 1200);
    EXPECT_NEAR(t.queue_wait_ms(), 900.0, 1e-6);
    EXPECT_NEAR(t.ttft_ms(), 50.0, 1e-6);
}

}  // namespace
}  // namespace microvllm
