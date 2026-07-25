#include "microvllm/sequence_state.hpp"

#include <algorithm>
#include <utility>

namespace microvllm {
namespace {

// Earliest index at which any (non-empty) stop string occurs in `text`, or npos.
std::size_t earliest_stop(const std::string& text, const std::vector<std::string>& stops) {
    std::size_t best = std::string::npos;
    for (const std::string& s : stops) {
        if (s.empty()) {
            continue;
        }
        best = std::min(best, text.find(s));
    }
    return best;
}

// Longest stop string. A stop of length L can only begin L-1 bytes before the current
// end, so withholding (max_stop - 1) trailing bytes guarantees nothing is ever emitted
// into a region that later turns out to be inside a stop string.
std::size_t max_stop_len(const std::vector<std::string>& stops) {
    std::size_t m = 0;
    for (const std::string& s : stops) {
        m = std::max(m, s.size());
    }
    return m;
}

}  // namespace

SequenceState::SequenceState(SeqId seq, RequestSpec spec, ITokenSink& sink,
                             std::vector<Token> prompt)
    : seq_(seq),
      spec_(std::move(spec)),
      sink_(sink),
      prompt_(std::move(prompt)),
      holdback_(max_stop_len(spec_.stop)) {
    usage_.prompt_tokens = static_cast<std::uint32_t>(prompt_.size());
}

Pos SequenceState::feed_pos() const {
    // completion_tokens is incremented as each token is accepted, so by the time this is
    // read for the pending token it already counts it: k-th token -> n_prompt + k - 1.
    return static_cast<Pos>(prompt_.size()) + static_cast<Pos>(usage_.completion_tokens) - 1;
}

BatchItem SequenceState::prefill_item() const {
    return BatchItem{.seq = seq_, .tokens = prompt_, .pos0 = 0, .sample = true};
}

BatchItem SequenceState::decode_item() const {
    return BatchItem{
        .seq = seq_, .tokens = std::span<const Token>(&pending_, 1), .pos0 = feed_pos(),
        .sample = true};
}

std::size_t SequenceState::safe_prefix() const {
    if (holdback_ == 0) {
        return text_.size();
    }
    return text_.size() >= holdback_ - 1 ? text_.size() - (holdback_ - 1) : 0;
}

void SequenceState::flush_upto(std::size_t upto) {
    if (upto > emitted_) {
        sink_.on_text(std::string_view(text_).substr(emitted_, upto - emitted_));
        emitted_ = upto;
    }
}

void SequenceState::stop_with(FinishReason reason) {
    reason_   = reason;
    finished_ = true;
}

bool SequenceState::accept(IModelEngine& engine, const GenStep& step) {
    if (finished_) {
        return false;
    }
    if (step.is_eog) {
        stop_with(FinishReason::kEos);
        return false;
    }
    if (usage_.completion_tokens >= spec_.max_tokens) {
        stop_with(FinishReason::kMaxTokens);
        return false;
    }

    text_ += engine.piece(step.token);
    ++usage_.completion_tokens;

    const std::size_t stop_at = earliest_stop(text_, spec_.stop);
    if (stop_at != std::string::npos) {
        flush_upto(stop_at);  // emit up to the stop, never into or past it
        stop_with(FinishReason::kStopString);
        return false;
    }

    flush_upto(safe_prefix());

    if (sink_.cancelled()) {
        flush_upto(text_.size());
        stop_with(FinishReason::kCancelled);
        return false;
    }

    pending_ = step.token;
    return true;
}

void SequenceState::fail(std::string message) {
    error_ = std::move(message);
    stop_with(FinishReason::kError);
}

void SequenceState::finish() {
    if (notified_) {
        return;  // idempotent: an unwinding scheduler must not double-notify
    }
    notified_ = true;
    finished_ = true;

    if (reason_ == FinishReason::kError) {
        sink_.on_error(error_);
        return;
    }
    // A clean finish flushes the held-back tail; a stop-string finish must not, since
    // that tail is exactly the text that must be suppressed.
    if (reason_ == FinishReason::kEos || reason_ == FinishReason::kMaxTokens) {
        flush_upto(text_.size());
    }
    sink_.on_finish(reason_, usage_);
}

}  // namespace microvllm
