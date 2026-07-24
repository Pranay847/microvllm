#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

#include "microvllm/request.hpp"

namespace microvllm {

// A bounded, multi-producer / single-consumer queue between the HTTP threads and the
// engine thread. This narrow boundary is the entire shared surface of the server, and
// it is what test_concurrency exercises under ThreadSanitizer.
//
// Backpressure is by rejection: try_push returns false when full, and the caller
// answers 503. Blocking producers would tie up HTTP worker threads, so an inference
// server sheds load instead.
class RequestQueue {
public:
    explicit RequestQueue(std::size_t capacity);

    // Enqueue a request. Returns false (leaving `req` untouched) if the queue is full
    // or closed; true after moving `req` in. Safe to call from many threads.
    [[nodiscard]] bool try_push(Request& req);

    // Block until a request is available and return it, or return nullopt once the
    // queue is closed and drained. Intended for the single consumer thread.
    [[nodiscard]] std::optional<Request> pop();

    // Stop accepting new requests and wake the consumer. Already-queued requests remain
    // poppable so the consumer can drain them; pop() returns nullopt once empty.
    void close();

    [[nodiscard]] bool closed() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
    mutable std::mutex      mutex_;
    std::condition_variable not_empty_;
    std::deque<Request>     queue_;
    const std::size_t       capacity_;
    bool                    closed_ = false;
};

}  // namespace microvllm
