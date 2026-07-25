// StreamingSink and the SSE payload helpers.
//
// The sink crosses a thread boundary -- the scheduler produces deltas, an HTTP thread
// consumes them -- so these tests cover the handoff as well as the framing.
#include "microvllm/streaming_sink.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "microvllm/http_api.hpp"
#include "microvllm/mock_engine.hpp"
#include "microvllm/scheduler.hpp"

namespace microvllm {
namespace {

TEST(StreamingSink, DeliversDeltasInOrderThenExactlyOneTerminal) {
    auto          cancel = std::make_shared<std::atomic<bool>>(false);
    StreamingSink sink(cancel);

    sink.on_text("Hello");
    sink.on_text(", ");
    sink.on_text("world");
    Usage u;
    u.prompt_tokens     = 3;
    u.completion_tokens = 5;
    sink.on_finish(FinishReason::kEos, u);

    EXPECT_EQ(sink.next().text, "Hello");
    EXPECT_EQ(sink.next().text, ", ");
    EXPECT_EQ(sink.next().text, "world");

    const StreamingSink::Event last = sink.next();
    EXPECT_TRUE(last.terminal);
    EXPECT_EQ(last.reason, FinishReason::kEos);
    EXPECT_EQ(last.usage.completion_tokens, 5U);
}

TEST(StreamingSink, EmptyDeltasAreNotEmitted) {
    auto          cancel = std::make_shared<std::atomic<bool>>(false);
    StreamingSink sink(cancel);
    sink.on_text("");       // must not become an event
    sink.on_text("real");
    sink.on_finish(FinishReason::kEos, Usage{});

    EXPECT_EQ(sink.next().text, "real") << "an empty delta would be a stray empty SSE frame";
    EXPECT_TRUE(sink.next().terminal);
}

TEST(StreamingSink, ErrorBecomesATerminalEvent) {
    auto          cancel = std::make_shared<std::atomic<bool>>(false);
    StreamingSink sink(cancel);
    sink.on_error("backend exploded");

    const StreamingSink::Event e = sink.next();
    EXPECT_TRUE(e.terminal);
    EXPECT_EQ(e.reason, FinishReason::kError);
    EXPECT_EQ(e.error, "backend exploded");
}

TEST(StreamingSink, ConsumerBlocksUntilProducerWrites) {
    auto          cancel = std::make_shared<std::atomic<bool>>(false);
    StreamingSink sink(cancel);

    std::thread producer([&] {
        sink.on_text("delayed");
        sink.on_finish(FinishReason::kEos, Usage{});
    });

    EXPECT_EQ(sink.next().text, "delayed");  // must block rather than spin or return empty
    EXPECT_TRUE(sink.next().terminal);
    producer.join();
}

TEST(StreamingSink, CancelFlagIsSharedWithTheRequest) {
    auto          cancel = std::make_shared<std::atomic<bool>>(false);
    StreamingSink sink(cancel);
    EXPECT_FALSE(sink.cancelled());

    // The HTTP thread trips the shared flag when the client disconnects.
    cancel->store(true, std::memory_order_relaxed);
    EXPECT_TRUE(sink.cancelled()) << "generation must observe a client disconnect";
}

TEST(StreamingSink, SchedulerStreamsThroughACallerSuppliedSink) {
    // End to end: a Request carrying a StreamingSink streams while generating, and the
    // concatenated deltas equal the response the buffered path would have returned.
    MockModelEngine engine(MockModelEngine::Config{.response = "", .echo_prompt = true});
    RequestQueue    queue(16);
    Scheduler       sched(engine, queue, SchedulerConfig{.max_batch_size = 4});

    auto cancel = std::make_shared<std::atomic<bool>>(false);
    auto sink   = std::make_shared<StreamingSink>(cancel);

    Request r;
    r.id              = 1;
    r.spec.prompt     = "streamed";
    r.spec.max_tokens = 128;
    r.cancel          = cancel;
    r.sink            = sink;
    auto fut          = r.result.get_future();
    ASSERT_TRUE(queue.try_push(r));

    std::thread t([&] { sched.run(); });
    queue.close();

    std::string streamed;
    while (true) {
        const StreamingSink::Event e = sink->next();
        if (e.terminal) {
            EXPECT_EQ(e.reason, FinishReason::kEos);
            break;
        }
        streamed += e.text;
    }
    t.join();

    EXPECT_EQ(streamed, "streamed");
    EXPECT_EQ(fut.get().text, streamed) << "streamed and buffered results must agree";
}

TEST(HttpApiStream, DeltaAndDoneFraming) {
    EXPECT_NE(make_stream_delta("hi").find("\"text\""), std::string::npos);
    EXPECT_NE(make_stream_delta("hi").find("hi"), std::string::npos);

    Usage u;
    u.prompt_tokens     = 2;
    u.completion_tokens = 3;
    const std::string done = make_stream_done(FinishReason::kMaxTokens, u);
    EXPECT_NE(done.find("max_tokens"), std::string::npos);
    EXPECT_NE(done.find("\"total_tokens\""), std::string::npos);
    EXPECT_NE(done.find('5'), std::string::npos);  // 2 + 3
}

TEST(HttpApiStream, DeltaEscapesJsonSpecialCharacters) {
    // Newlines and quotes are common in generated text and would corrupt the SSE frame
    // if they were not escaped.
    const std::string d = make_stream_delta("line\n\"quoted\"");
    EXPECT_EQ(d.find('\n'), std::string::npos) << "a raw newline would end the SSE event early";
    EXPECT_NE(d.find("\\n"), std::string::npos);
    EXPECT_NE(d.find("\\\""), std::string::npos);
}

}  // namespace
}  // namespace microvllm
