// BlockAllocator invariants.
//
// The failure modes worth guarding are silent ones: a block handed to two sequences
// (corrupting both), or a block that never returns to the free list (a slow leak that
// ends as an unexplained capacity loss). These are tested as properties over sequences of
// operations, not just single calls.
#include "microvllm/block_allocator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

namespace microvllm {
namespace {

TEST(BlockAllocator, StartsFullyFree) {
    const BlockAllocator a(8, 16);
    EXPECT_EQ(a.total_blocks(), 8U);
    EXPECT_EQ(a.free_blocks(), 8U);
    EXPECT_EQ(a.used_blocks(), 0U);
    EXPECT_EQ(a.block_size(), 16U);
    EXPECT_FLOAT_EQ(a.utilization(), 0.0F);
}

TEST(BlockAllocator, BlocksForRoundsUp) {
    const BlockAllocator a(64, 16);
    EXPECT_EQ(a.blocks_for(0), 0U);
    EXPECT_EQ(a.blocks_for(1), 1U);
    EXPECT_EQ(a.blocks_for(15), 1U);
    EXPECT_EQ(a.blocks_for(16), 1U) << "an exact multiple must not waste a block";
    EXPECT_EQ(a.blocks_for(17), 2U);
    EXPECT_EQ(a.blocks_for(32), 2U);
}

TEST(BlockAllocator, AllocateTakesFromThePoolAndFreeReturnsIt) {
    BlockAllocator a(4, 16);
    const auto got = a.allocate(3);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->size(), 3U);
    EXPECT_EQ(a.free_blocks(), 1U);
    EXPECT_EQ(a.used_blocks(), 3U);
    EXPECT_FLOAT_EQ(a.utilization(), 0.75F);

    a.free(*got);
    EXPECT_EQ(a.free_blocks(), 4U);
    EXPECT_FLOAT_EQ(a.utilization(), 0.0F);
}

TEST(BlockAllocator, AllocatedBlocksAreDistinct) {
    BlockAllocator a(16, 16);
    const auto got = a.allocate(16);
    ASSERT_TRUE(got.has_value());
    const std::set<BlockId> unique(got->begin(), got->end());
    EXPECT_EQ(unique.size(), 16U) << "a block was handed out twice";
}

TEST(BlockAllocator, AllocationIsAllOrNothing) {
    BlockAllocator a(4, 16);
    EXPECT_TRUE(a.can_allocate(4));
    EXPECT_FALSE(a.can_allocate(5));

    EXPECT_FALSE(a.allocate(5).has_value()) << "over-large request must fail";
    EXPECT_EQ(a.free_blocks(), 4U)
        << "a failed allocation must take nothing -- a partial one would have to be unwound";

    // The pool is still fully usable after the failed attempt.
    EXPECT_TRUE(a.allocate(4).has_value());
    EXPECT_EQ(a.free_blocks(), 0U);
    EXPECT_FALSE(a.allocate(1).has_value());
}

TEST(BlockAllocator, AllocatingZeroBlocksSucceedsTrivially) {
    BlockAllocator a(2, 16);
    const auto got = a.allocate(0);
    ASSERT_TRUE(got.has_value());
    EXPECT_TRUE(got->empty());
    EXPECT_EQ(a.free_blocks(), 2U);
}

TEST(BlockAllocator, SharedBlockSurvivesUntilTheLastReferenceDrops) {
    // Prefix sharing: several sequences hold the same block, and it must not return to
    // the free list while any of them is still using it.
    BlockAllocator a(4, 16);
    const auto got = a.allocate(1);
    ASSERT_TRUE(got.has_value());
    const BlockId shared = got->at(0);
    EXPECT_EQ(a.refcount(shared), 1U);

    a.incref(shared);
    a.incref(shared);
    EXPECT_EQ(a.refcount(shared), 3U);
    EXPECT_EQ(a.free_blocks(), 3U);

    const BlockId one[] = {shared};
    a.free(one);
    EXPECT_EQ(a.refcount(shared), 2U);
    EXPECT_EQ(a.free_blocks(), 3U) << "still referenced; must not be reclaimed";

    a.free(one);
    EXPECT_EQ(a.free_blocks(), 3U);

    a.free(one);
    EXPECT_EQ(a.refcount(shared), 0U);
    EXPECT_EQ(a.free_blocks(), 4U) << "last reference dropped; now reclaimable";
}

TEST(BlockAllocator, ReclaimedBlockCanBeAllocatedAgain) {
    BlockAllocator a(2, 16);
    const auto first = a.allocate(2);
    ASSERT_TRUE(first.has_value());
    a.free(*first);

    const auto second = a.allocate(2);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(a.free_blocks(), 0U);
    const std::set<BlockId> unique(second->begin(), second->end());
    EXPECT_EQ(unique.size(), 2U);
}

TEST(BlockAllocator, EveryBlockReturnsExactlyOnceAcrossManyCycles) {
    // The leak property, exercised over a long mixed sequence of operations: after all
    // holders release, the pool must be exactly whole again -- no fewer blocks (leak) and
    // no more (double-free inflating the pool).
    constexpr std::uint32_t kTotal = 32;
    BlockAllocator          a(kTotal, 16);

    std::vector<std::vector<BlockId>> held;
    for (int round = 0; round < 50; ++round) {
        const auto want = static_cast<std::uint32_t>(1 + (round % 5));
        if (a.can_allocate(want)) {
            auto got = a.allocate(want);
            ASSERT_TRUE(got.has_value());
            held.push_back(std::move(*got));
        }
        // Release roughly every other holder to interleave allocation and reclamation.
        if (!held.empty() && round % 2 == 1) {
            a.free(held.front());
            held.erase(held.begin());
        }

        // Invariant that must hold at every point, not just at the end.
        EXPECT_LE(a.used_blocks(), kTotal);
        EXPECT_EQ(a.free_blocks() + a.used_blocks(), kTotal);
    }

    for (const auto& blocks : held) {
        a.free(blocks);
    }
    EXPECT_EQ(a.free_blocks(), kTotal) << "blocks leaked or were double-counted";

    // And the whole pool is usable again.
    EXPECT_TRUE(a.allocate(kTotal).has_value());
}

TEST(BlockAllocator, NoBlockIsEverHeldByTwoAllocationsAtOnce) {
    // Stronger than distinctness within one call: across concurrent live allocations, the
    // union of everything held must have no duplicates.
    BlockAllocator                   a(24, 16);
    std::vector<std::vector<BlockId>> held;
    for (int i = 0; i < 6; ++i) {
        auto got = a.allocate(4);
        ASSERT_TRUE(got.has_value()) << "allocation " << i << " unexpectedly failed";
        held.push_back(std::move(*got));
    }

    std::vector<BlockId> all;
    for (const auto& h : held) {
        all.insert(all.end(), h.begin(), h.end());
    }
    std::sort(all.begin(), all.end());
    EXPECT_TRUE(std::adjacent_find(all.begin(), all.end()) == all.end())
        << "the same block was handed to two live allocations";
    EXPECT_EQ(all.size(), 24U);
}

// --- BlockTable ------------------------------------------------------------------

TEST(BlockTable, MaintainsTheCeilingInvariantAsItGrows) {
    BlockAllocator a(16, 16);
    BlockTable     t(a);

    // The invariant that defines a page table: blocks == ceil(tokens / block_size).
    for (std::uint32_t tokens : {1U, 5U, 16U, 17U, 32U, 33U, 64U}) {
        ASSERT_TRUE(t.ensure_capacity(tokens)) << "at " << tokens << " tokens";
        EXPECT_EQ(t.n_blocks(), a.blocks_for(tokens))
            << "block count wrong at " << tokens << " tokens";
        EXPECT_EQ(t.n_tokens(), tokens);
    }
}

TEST(BlockTable, GrowingWithinABlockCostsNoAllocation) {
    BlockAllocator a(8, 16);
    BlockTable     t(a);

    ASSERT_TRUE(t.ensure_capacity(1));
    EXPECT_EQ(a.used_blocks(), 1U);

    // Tokens 2..16 all live in the block already held.
    for (std::uint32_t tokens = 2; tokens <= 16; ++tokens) {
        ASSERT_TRUE(t.ensure_capacity(tokens));
    }
    EXPECT_EQ(a.used_blocks(), 1U) << "paging is pointless if every token allocates";

    ASSERT_TRUE(t.ensure_capacity(17));  // crossing the boundary does allocate
    EXPECT_EQ(a.used_blocks(), 2U);
}

TEST(BlockTable, FailsCleanlyWhenThePoolIsExhausted) {
    BlockAllocator a(2, 16);
    BlockTable     t(a);
    ASSERT_TRUE(t.ensure_capacity(32));  // exactly 2 blocks
    EXPECT_EQ(a.free_blocks(), 0U);

    EXPECT_FALSE(t.ensure_capacity(33)) << "must refuse rather than over-commit";
    EXPECT_EQ(t.n_blocks(), 2U) << "a failed growth must leave the table untouched";
    EXPECT_EQ(t.n_tokens(), 32U);
}

TEST(BlockTable, ReleaseReturnsEveryBlockAndIsIdempotent) {
    BlockAllocator a(8, 16);
    BlockTable     t(a);
    ASSERT_TRUE(t.ensure_capacity(64));
    EXPECT_EQ(a.used_blocks(), 4U);

    t.release();
    EXPECT_EQ(a.free_blocks(), 8U);
    EXPECT_EQ(t.n_blocks(), 0U);
    EXPECT_EQ(t.n_tokens(), 0U);

    t.release();  // must not double-free and inflate the pool
    EXPECT_EQ(a.free_blocks(), 8U);
}

TEST(BlockTable, SharedPrefixBlocksSurviveOneSequenceReleasing) {
    // Prefix sharing: two sequences adopt the same leading blocks. One finishing must not
    // pull that cache out from under the other.
    BlockAllocator a(8, 16);

    BlockTable owner(a);
    ASSERT_TRUE(owner.ensure_capacity(32));  // 2 blocks of shared prefix
    const std::vector<BlockId> prefix = owner.blocks();

    BlockTable borrower(a);
    borrower.adopt_shared_prefix(prefix, 32);
    EXPECT_EQ(borrower.n_blocks(), 2U);
    EXPECT_EQ(borrower.shared_prefix_blocks(), 2U);
    EXPECT_EQ(borrower.shared_prefix_tokens(), 32U);
    EXPECT_EQ(a.used_blocks(), 2U) << "sharing must not consume extra blocks";
    EXPECT_EQ(a.refcount(prefix[0]), 2U);

    borrower.release();
    EXPECT_EQ(a.refcount(prefix[0]), 1U);
    EXPECT_EQ(a.used_blocks(), 2U) << "owner still holds the prefix";

    owner.release();
    EXPECT_EQ(a.free_blocks(), 8U) << "last holder released; now reclaimable";
}

TEST(BlockTable, SequenceGrowsBeyondAnAdoptedPrefix) {
    // The borrower diverges after the shared prefix and needs its own blocks for the rest.
    BlockAllocator a(8, 16);

    BlockTable owner(a);
    ASSERT_TRUE(owner.ensure_capacity(32));
    const std::vector<BlockId> prefix = owner.blocks();

    BlockTable borrower(a);
    borrower.adopt_shared_prefix(prefix, 32);
    ASSERT_TRUE(borrower.ensure_capacity(48));  // one more block, privately owned

    EXPECT_EQ(borrower.n_blocks(), 3U);
    EXPECT_EQ(a.used_blocks(), 3U) << "only the divergent block is newly allocated";
    EXPECT_EQ(a.refcount(prefix[0]), 2U);
    EXPECT_EQ(a.refcount(borrower.blocks().back()), 1U) << "the new tail is private";
}

TEST(BlockAllocator, ZeroSizedPoolNeverAllocates) {
    BlockAllocator a(0, 16);
    EXPECT_EQ(a.total_blocks(), 0U);
    EXPECT_FALSE(a.can_allocate(1));
    EXPECT_FALSE(a.allocate(1).has_value());
    EXPECT_FLOAT_EQ(a.utilization(), 0.0F) << "must not divide by zero";
}

}  // namespace
}  // namespace microvllm
