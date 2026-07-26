#include "microvllm/block_allocator.hpp"

#include <algorithm>
#include <cassert>

namespace microvllm {

BlockAllocator::BlockAllocator(std::uint32_t total_blocks, std::uint32_t block_size)
    : total_(total_blocks), block_size_(block_size == 0 ? 1 : block_size) {
    free_list_.reserve(total_);
    refcounts_.assign(total_, 0);
    // Hand out low ids first. The free list is a stack, so a just-freed block is reused
    // immediately -- friendlier to cache locality than round-robin over the whole pool.
    for (std::uint32_t i = total_; i > 0; --i) {
        free_list_.push_back(static_cast<BlockId>(i - 1));
    }
}

std::uint32_t BlockAllocator::blocks_for(std::uint32_t n_tokens) const {
    return (n_tokens + block_size_ - 1) / block_size_;
}

bool BlockAllocator::can_allocate(std::uint32_t n) const {
    return n <= free_blocks();
}

std::optional<std::vector<BlockId>> BlockAllocator::allocate(std::uint32_t n) {
    // All-or-nothing. A partial allocation would leave the caller holding blocks it must
    // unwind on a path that is easy to get wrong and hard to test.
    if (!can_allocate(n)) {
        return std::nullopt;
    }

    std::vector<BlockId> out;
    out.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const BlockId id = free_list_.back();
        free_list_.pop_back();
        assert(refcounts_[id] == 0 && "free-list block already had references");
        refcounts_[id] = 1;
        out.push_back(id);
    }
    return out;
}

void BlockAllocator::free(std::span<const BlockId> blocks) {
    for (const BlockId id : blocks) {
        if (id >= total_ || refcounts_[id] == 0) {
            continue;  // double free or out of range: ignore rather than corrupt the pool
        }
        if (--refcounts_[id] == 0) {
            free_list_.push_back(id);
        }
    }
}

void BlockAllocator::incref(BlockId block) {
    if (block < total_ && refcounts_[block] > 0) {
        ++refcounts_[block];
    }
}

std::uint32_t BlockAllocator::refcount(BlockId block) const {
    return block < total_ ? refcounts_[block] : 0;
}

// ---------------------------------------------------------------------------
// BlockTable
// ---------------------------------------------------------------------------
bool BlockTable::ensure_capacity(std::uint32_t n_tokens) {
    const std::uint32_t needed = allocator_->blocks_for(n_tokens);
    if (needed > n_blocks()) {
        // Only the shortfall is requested: growing by a token inside an existing block
        // costs nothing, which is the whole point of paging the cache.
        const auto extra = allocator_->allocate(needed - n_blocks());
        if (!extra) {
            return false;  // caller decides: queue, preempt, or reject
        }
        blocks_.insert(blocks_.end(), extra->begin(), extra->end());
    }
    n_tokens_ = n_tokens;
    return true;
}

void BlockTable::adopt_shared_prefix(std::span<const BlockId> blocks, std::uint32_t n_tokens) {
    for (const BlockId id : blocks) {
        allocator_->incref(id);
        blocks_.push_back(id);
    }
    shared_prefix_blocks_ = static_cast<std::uint32_t>(blocks.size());
    shared_prefix_tokens_ = n_tokens;
    n_tokens_             = std::max(n_tokens_, n_tokens);
}

void BlockTable::release() {
    // Shared blocks are freed the same way: free() is refcount-aware, so this drops only
    // this sequence's reference and leaves the block alive for anyone else holding it.
    allocator_->free(blocks_);
    blocks_.clear();
    n_tokens_             = 0;
    shared_prefix_blocks_ = 0;
    shared_prefix_tokens_ = 0;
}

}  // namespace microvllm
