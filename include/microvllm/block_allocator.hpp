#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "microvllm/types.hpp"

namespace microvllm {

// A fixed pool of KV-cache blocks, allocated and freed like pages.
//
// WHAT THIS OWNS, AND WHAT IT DOES NOT
//
// llama.cpp owns the actual KV bytes and decides which physical cells a sequence uses.
// This allocator owns the *policy*: how much cache a sequence may hold, who is admitted,
// who is evicted when the pool runs dry, and which blocks are shared between sequences.
// Every decision it makes is mirrored into the backend (release_sequence / copy_sequence).
//
// That makes it a governor over someone else's memory rather than a replacement for it --
// stated plainly because it is the first thing a careful reader should ask about. It is
// still a real allocator with real invariants, and the behaviour it produces (admission
// control, preemption, prefix sharing) is real and measurable.
//
// The reason it exists at all: KV cache, not compute, is the binding constraint in LLM
// serving. Cache grows linearly with sequence length times concurrency, so a server
// without a cache budget either over-commits and fails deep in the backend, or
// under-commits and wastes capacity.
class BlockAllocator {
public:
    // `block_size` is tokens per block. 16 matches vLLM's default: small enough to keep
    // internal fragmentation low (a sequence wastes at most 15 tokens of its last block),
    // large enough that the block table stays short.
    BlockAllocator(std::uint32_t total_blocks, std::uint32_t block_size);

    // Can `n` fresh blocks be handed out right now?
    [[nodiscard]] bool can_allocate(std::uint32_t n) const;

    // Take `n` blocks, each at refcount 1. All-or-nothing: returns nullopt and takes
    // nothing if the pool cannot satisfy the whole request, so a caller can never end up
    // holding a partial allocation it has to unwind.
    [[nodiscard]] std::optional<std::vector<BlockId>> allocate(std::uint32_t n);

    // Drop one reference to each block; those reaching zero return to the free list.
    // Refcount-aware because prefix sharing lets several sequences hold the same block.
    void free(std::span<const BlockId> blocks);

    // Take an additional reference, e.g. when a sequence adopts a cached prefix block.
    void incref(BlockId block);

    [[nodiscard]] std::uint32_t refcount(BlockId block) const;

    [[nodiscard]] std::uint32_t total_blocks() const { return total_; }
    [[nodiscard]] std::uint32_t free_blocks() const {
        return static_cast<std::uint32_t>(free_list_.size());
    }
    [[nodiscard]] std::uint32_t used_blocks() const { return total_ - free_blocks(); }
    [[nodiscard]] std::uint32_t block_size() const { return block_size_; }

    [[nodiscard]] float utilization() const {
        return total_ == 0 ? 0.0F
                           : static_cast<float>(used_blocks()) / static_cast<float>(total_);
    }

    // Blocks needed to hold `n_tokens`, i.e. ceil(n_tokens / block_size).
    [[nodiscard]] std::uint32_t blocks_for(std::uint32_t n_tokens) const;

private:
    std::uint32_t              total_;
    std::uint32_t              block_size_;
    std::vector<BlockId>       free_list_;
    std::vector<std::uint32_t> refcounts_;  // indexed by BlockId
};

// Maps a hashed prompt prefix to the sequence currently holding its KV cache.
//
// Prefix sharing exists because chat traffic is highly redundant: every request in a
// conversation, or every request behind the same system prompt, re-sends the same leading
// tokens. Prefilling them again is pure waste, and prefill is the compute-bound half of
// inference. A hit lets the new sequence copy that KV rather than recompute it.
//
// Keyed on a rolling hash of the token prefix, granular to whole blocks: a partial block
// cannot be shared, since the sequence must be free to write the rest of it.
//
// SCOPE, STATED PLAINLY: an entry names a *live* sequence, so sharing only happens between
// sequences that overlap in time. When the donor retires its KV is reclaimed and the entry
// is evicted, so a request arriving after every previous one has finished gets no benefit.
// That covers concurrent traffic behind a shared system prompt, but not the sequential
// case, which is where prefix caching is most often pitched.
//
// Retaining a donor past retirement would mean pinning its llama.cpp sequence and blocks
// (an LRU of "donor" slots that are evicted on demand). That is the natural next step and
// is deliberately not claimed here.
class PrefixCache {
public:
    struct Entry {
        SeqId                seq;       // sequence whose KV holds this prefix
        std::uint32_t        n_tokens;  // tokens covered
        std::vector<BlockId> blocks;
    };

    // Hash the first `n_tokens` of `tokens`. Whole-block granularity, so the returned
    // hash covers exactly floor(n_tokens / block_size) * block_size tokens.
    [[nodiscard]] static std::uint64_t hash_prefix(std::span<const Token> tokens,
                                                   std::uint32_t          n_tokens);

    // Longest cached prefix of `tokens`, or nullopt. Checks progressively shorter
    // block-aligned prefixes so a partial match still helps.
    [[nodiscard]] std::optional<Entry> find_longest(std::span<const Token> tokens,
                                                    std::uint32_t block_size) const;

    void insert(std::uint64_t hash, Entry entry);

    // Drop every entry owned by `seq`. Called when that sequence's cache goes away, since
    // the entry names a live sequence whose KV is about to stop existing.
    void evict_sequence(SeqId seq);

    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    void clear() { entries_.clear(); }

private:
    std::unordered_map<std::uint64_t, Entry> entries_;
};

// One sequence's page table: the blocks backing its KV cache, in logical order.
//
// A sequence grows one token at a time, but blocks are only taken when it crosses a block
// boundary -- so a 16-token block costs one allocation, not sixteen. The central
// invariant, asserted throughout the tests, is:
//
//     blocks.size() == ceil(n_tokens / block_size)
//
// Leading blocks may be *shared* with other sequences via the prefix cache. Those are
// counted separately so that releasing this sequence drops only its own reference and
// never pulls a block out from under a sequence still using it.
class BlockTable {
public:
    explicit BlockTable(BlockAllocator& allocator) : allocator_(&allocator) {}

    BlockTable(const BlockTable&)            = delete;
    BlockTable& operator=(const BlockTable&) = delete;
    BlockTable(BlockTable&&)                 = default;
    BlockTable& operator=(BlockTable&&)      = default;

    // Ensure capacity for `n_tokens` total. Returns false and changes nothing if the pool
    // cannot supply the blocks needed -- the caller then queues, preempts, or rejects.
    [[nodiscard]] bool ensure_capacity(std::uint32_t n_tokens);

    // Adopt `blocks` as this sequence's leading shared prefix, taking a reference to each.
    // `n_tokens` is how many tokens those blocks already cover.
    void adopt_shared_prefix(std::span<const BlockId> blocks, std::uint32_t n_tokens);

    // Drop every reference this sequence holds and reset it to empty. Idempotent.
    void release();

    [[nodiscard]] const std::vector<BlockId>& blocks() const { return blocks_; }
    [[nodiscard]] std::uint32_t n_tokens() const { return n_tokens_; }
    [[nodiscard]] std::uint32_t n_blocks() const {
        return static_cast<std::uint32_t>(blocks_.size());
    }
    [[nodiscard]] std::uint32_t shared_prefix_blocks() const { return shared_prefix_blocks_; }
    [[nodiscard]] std::uint32_t shared_prefix_tokens() const { return shared_prefix_tokens_; }

private:
    BlockAllocator*      allocator_;
    std::vector<BlockId> blocks_;
    std::uint32_t        n_tokens_             = 0;
    std::uint32_t        shared_prefix_blocks_ = 0;
    std::uint32_t        shared_prefix_tokens_ = 0;
};

}  // namespace microvllm
