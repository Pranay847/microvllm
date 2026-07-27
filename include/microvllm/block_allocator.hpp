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
// DONOR RETENTION: an entry normally names a *live* sequence, which limits sharing to
// requests that overlap in time -- useless for sequential traffic, which is the case prefix
// caching is most often pitched on. So when a sequence retires holding a cacheable prefix,
// its KV is copied into a reserved "donor" slot and kept, letting a request that arrives
// long afterwards still hit.
//
// Retention is not free: a donor pins a llama.cpp sequence id and holds real blocks that
// active requests could otherwise use. Two rules follow, and both matter:
//
//   * Donors are bounded (--prefix-donors) and evicted least-recently-used, so the cache
//     cannot grow without limit.
//   * Donors are reclaimed BEFORE live sequences are preempted. A donor is pure cache and
//     losing it costs a recompute later; a live sequence is work already in progress and
//     killing it throws that work away. Cache should always yield to work.
class PrefixCache {
public:
    struct Entry {
        SeqId                seq;       // sequence whose KV holds this prefix
        std::uint32_t        n_tokens;  // tokens covered
        std::vector<BlockId> blocks;
        // True once the originating request has retired and this entry has been moved to a
        // reserved donor slot. Retained entries are the ones eligible for LRU eviction.
        bool                 retained = false;
    };

    // Reserve `count` sequence ids starting at `first` for retained prefixes. These live
    // above the scheduler's batch slots, so the engine must be built with
    // n_seq_max >= max_batch_size + count.
    void configure_donors(SeqId first, std::uint32_t count);

    [[nodiscard]] std::uint32_t donor_capacity() const { return donor_capacity_; }
    [[nodiscard]] std::uint32_t donors_held() const {
        return donor_capacity_ - static_cast<std::uint32_t>(free_donor_slots_.size());
    }

    // Hash the first `n_tokens` of `tokens`. Whole-block granularity, so the returned
    // hash covers exactly floor(n_tokens / block_size) * block_size tokens.
    [[nodiscard]] static std::uint64_t hash_prefix(std::span<const Token> tokens,
                                                   std::uint32_t          n_tokens);

    // Longest cached prefix of `tokens`, or nullopt. Checks progressively shorter
    // block-aligned prefixes so a partial match still helps. Marks the hit as recently
    // used, which is what keeps a hot prefix from being evicted under churn.
    [[nodiscard]] std::optional<Entry> find_longest(std::span<const Token> tokens,
                                                    std::uint32_t block_size);

    void insert(std::uint64_t hash, Entry entry);

    // Take a free donor slot, or nullopt if all are occupied. The caller is responsible
    // for copying KV into it and calling retain().
    [[nodiscard]] std::optional<SeqId> take_donor_slot();

    // Record that `slot` now holds this prefix past its originating request's retirement.
    void retain(std::uint64_t hash, SeqId slot, std::uint32_t n_tokens,
                std::vector<BlockId> blocks);

    // Evict the least-recently-used retained entry and hand it back so the caller can free
    // its blocks and release its engine sequence. Returns nullopt if no donors are held.
    [[nodiscard]] std::optional<Entry> evict_lru_retained();

    // Drop every entry owned by `seq`, returning a donor slot to the free list if `seq` was
    // one. Called when that sequence's KV is about to stop existing.
    void evict_sequence(SeqId seq);

    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    void clear();

private:
    struct Record {
        Entry         entry;
        std::uint64_t last_used = 0;  // monotonic tick; smallest is the LRU victim
    };

    // A plain counter and a linear scan rather than an intrusive LRU list: donor counts are
    // single digits, so the scan is trivial and the simpler structure has fewer ways to
    // corrupt itself.
    std::unordered_map<std::uint64_t, Record> entries_;
    std::vector<SeqId>                        free_donor_slots_;
    SeqId                                     first_donor_slot_ = 0;
    std::uint32_t                             donor_capacity_   = 0;
    std::uint64_t                             tick_             = 0;
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
