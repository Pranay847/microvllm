#include "microvllm/request_queue.hpp"

namespace microvllm {

RequestQueue::RequestQueue(std::size_t capacity) : capacity_(capacity) {}

bool RequestQueue::try_push(Request& req) {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= capacity_) {
            return false;  // leave req untouched; caller answers 503
        }
        queue_.push_back(std::move(req));
    }
    not_empty_.notify_one();
    return true;
}

std::optional<Request> RequestQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });
    if (queue_.empty()) {
        return std::nullopt;  // closed and fully drained
    }
    Request req = std::move(queue_.front());
    queue_.pop_front();
    return req;
}

void RequestQueue::close() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    // Wake every waiting consumer so they can drain and then observe the close.
    not_empty_.notify_all();
}

bool RequestQueue::closed() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

std::size_t RequestQueue::size() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

}  // namespace microvllm
