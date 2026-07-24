#include "microvllm/request_queue.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>

namespace microvllm {
namespace {

Request make_req(RequestId id) {
    Request r;
    r.id     = id;
    r.cancel = std::make_shared<std::atomic<bool>>(false);
    return r;
}

TEST(RequestQueue, PushThenPopReturnsSameRequest) {
    RequestQueue q(4);
    Request r = make_req(42);
    ASSERT_TRUE(q.try_push(r));
    EXPECT_EQ(q.size(), 1U);

    auto got = q.pop();
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->id, 42U);
    EXPECT_EQ(q.size(), 0U);
}

TEST(RequestQueue, PreservesFifoOrder) {
    RequestQueue q(8);
    for (RequestId id : {1U, 2U, 3U}) {
        Request r = make_req(id);
        ASSERT_TRUE(q.try_push(r));
    }
    EXPECT_EQ(q.pop()->id, 1U);
    EXPECT_EQ(q.pop()->id, 2U);
    EXPECT_EQ(q.pop()->id, 3U);
}

TEST(RequestQueue, RejectsWhenFull) {
    RequestQueue q(2);
    Request a = make_req(1), b = make_req(2), c = make_req(3);
    EXPECT_TRUE(q.try_push(a));
    EXPECT_TRUE(q.try_push(b));
    EXPECT_FALSE(q.try_push(c)) << "third push exceeds capacity 2";
    EXPECT_EQ(q.size(), 2U);
}

TEST(RequestQueue, FailedPushLeavesRequestIntact) {
    RequestQueue q(1);
    Request a = make_req(1), b = make_req(2);
    ASSERT_TRUE(q.try_push(a));
    ASSERT_FALSE(q.try_push(b));
    EXPECT_EQ(b.id, 2U) << "a rejected request must still be usable";
    EXPECT_TRUE(b.cancel) << "its promise/cancel must not have been moved out";

    // Draining makes room, and the same request now enqueues.
    (void)q.pop();
    EXPECT_TRUE(q.try_push(b));
}

TEST(RequestQueue, RejectsWhenClosed) {
    RequestQueue q(4);
    q.close();
    Request r = make_req(1);
    EXPECT_FALSE(q.try_push(r));
    EXPECT_TRUE(q.closed());
}

TEST(RequestQueue, PopReturnsNulloptWhenClosedAndEmpty) {
    RequestQueue q(4);
    q.close();
    EXPECT_FALSE(q.pop().has_value());
}

TEST(RequestQueue, PopDrainsRemainingAfterClose) {
    RequestQueue q(4);
    Request a = make_req(1), b = make_req(2);
    ASSERT_TRUE(q.try_push(a));
    ASSERT_TRUE(q.try_push(b));
    q.close();  // closing must not discard already-queued work

    EXPECT_EQ(q.pop()->id, 1U);
    EXPECT_EQ(q.pop()->id, 2U);
    EXPECT_FALSE(q.pop().has_value()) << "nullopt only after the backlog is drained";
}

TEST(RequestQueue, DeliversResultThroughTheRequestPromise) {
    RequestQueue q(2);
    Request r = make_req(7);
    auto fut = r.result.get_future();
    ASSERT_TRUE(q.try_push(r));

    auto got = q.pop();
    ASSERT_TRUE(got.has_value());
    GenResult out;
    out.text   = "done";
    out.reason = FinishReason::kEos;
    got->result.set_value(std::move(out));

    const GenResult res = fut.get();
    EXPECT_EQ(res.text, "done");
    EXPECT_EQ(res.reason, FinishReason::kEos);
}

}  // namespace
}  // namespace microvllm
