#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <span>

namespace microvllm {

// A fixed-bucket latency histogram, safe to write from the scheduler thread while an HTTP
// thread scrapes it.
//
// A histogram rather than a running mean, because a mean hides exactly the requests worth
// investigating: a server where most requests take 50 ms and a few take 30 s has a
// respectable average and a terrible tail. Prometheus histograms are cumulative
// ("less than or equal to"), which is what lets a scraper compute percentiles.
//
// Buckets are fixed and coarse rather than adaptive. That keeps writes to a single relaxed
// atomic increment on the hot path, which is what makes it safe to leave enabled.
class LatencyHistogram {
public:
    // Upper bounds in milliseconds, spanning "fast local decode" to "something is wrong".
    static constexpr std::array<double, 12> kBounds = {
        1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0};

    void observe(double value_ms) {
        for (std::size_t i = 0; i < kBounds.size(); ++i) {
            if (value_ms <= kBounds[i]) {
                counts_[i].fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        // Anything past the last bound still counts toward the total and the +Inf bucket,
        // which Prometheus derives from `count`.
        count_.fetch_add(1, std::memory_order_relaxed);
        // Sum is kept in microseconds as an integer: doubles cannot be added atomically,
        // and a float accumulator would drift over millions of observations anyway.
        sum_us_.fetch_add(static_cast<std::uint64_t>(value_ms * 1000.0),
                          std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t bucket(std::size_t i) const {
        return counts_[i].load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t count() const { return count_.load(std::memory_order_relaxed); }
    [[nodiscard]] double sum_seconds() const {
        return static_cast<double>(sum_us_.load(std::memory_order_relaxed)) / 1e6;
    }

private:
    std::array<std::atomic<std::uint64_t>, kBounds.size()> counts_{};
    std::atomic<std::uint64_t>                             count_{0};
    std::atomic<std::uint64_t>                             sum_us_{0};
};

}  // namespace microvllm
