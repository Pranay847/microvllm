#include "microvllm/latency_histogram.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace microvllm {
namespace {

std::uint64_t bucket_total(const LatencyHistogram& h) {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < LatencyHistogram::kBounds.size(); ++i) total += h.bucket(i);
    return total;
}

TEST(LatencyHistogram, BoundsAreStrictlyIncreasing) {
    // Prometheus rejects a histogram whose `le` labels are not ascending.
    for (std::size_t i = 1; i < LatencyHistogram::kBounds.size(); ++i) {
        EXPECT_LT(LatencyHistogram::kBounds[i - 1], LatencyHistogram::kBounds[i]);
    }
}

TEST(LatencyHistogram, StartsEmpty) {
    const LatencyHistogram h;
    EXPECT_EQ(h.count(), 0U);
    EXPECT_EQ(bucket_total(h), 0U);
    EXPECT_DOUBLE_EQ(h.sum_seconds(), 0.0);
}

TEST(LatencyHistogram, ValueOnABoundLandsInThatBucket) {
    // Buckets are "less than or equal to", so exactly 5 ms belongs to the 5 ms bucket.
    LatencyHistogram h;
    h.observe(5.0);
    EXPECT_EQ(h.bucket(0), 0U);
    EXPECT_EQ(h.bucket(1), 1U);
    EXPECT_EQ(h.bucket(2), 0U);
}

TEST(LatencyHistogram, ValueJustPastABoundLandsInTheNextBucket) {
    LatencyHistogram h;
    h.observe(5.001);
    EXPECT_EQ(h.bucket(1), 0U);
    EXPECT_EQ(h.bucket(2), 1U);
}

TEST(LatencyHistogram, EachObservationFillsExactlyOneBucket) {
    // Buckets are stored non-cumulatively; the exporter does the running sum.
    LatencyHistogram h;
    h.observe(0.5);
    h.observe(30.0);
    h.observe(10000.0);
    EXPECT_EQ(h.count(), 3U);
    EXPECT_EQ(bucket_total(h), 3U);
    EXPECT_EQ(h.bucket(0), 1U);
    EXPECT_EQ(h.bucket(4), 1U);
    EXPECT_EQ(h.bucket(LatencyHistogram::kBounds.size() - 1), 1U);
}

TEST(LatencyHistogram, OverflowCountsTowardTotalButNoBucket) {
    // Past the last bound is the +Inf bucket, which Prometheus derives from `count`.
    LatencyHistogram h;
    h.observe(30000.0);
    EXPECT_EQ(h.count(), 1U);
    EXPECT_EQ(bucket_total(h), 0U);
    EXPECT_DOUBLE_EQ(h.sum_seconds(), 30.0);
}

TEST(LatencyHistogram, SumIsReportedInSeconds) {
    LatencyHistogram h;
    h.observe(250.0);
    h.observe(1500.0);
    EXPECT_DOUBLE_EQ(h.sum_seconds(), 1.75);
}

TEST(LatencyHistogram, SumTruncatesBelowOneMicrosecond) {
    // The sum is an integer count of microseconds, so sub-microsecond parts are dropped.
    LatencyHistogram h;
    h.observe(0.0004);
    EXPECT_EQ(h.count(), 1U);
    EXPECT_DOUBLE_EQ(h.sum_seconds(), 0.0);
}

}  // namespace
}  // namespace microvllm
