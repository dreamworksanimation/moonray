// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "TileWorkQueue.h"
#include "TileWorkQueueRuntimeVerify.h"

#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/util/StrUtil.h>

#ifdef RUNTIME_VERIFY_TILE_WORK_QUEUE // See RuntimeVerify.h
#define RUNTIME_VERIFY
#endif // end RUNTIME_VERIFY_TILE_WORK_QUEUE

namespace moonray {
namespace rndr {

namespace
{
constexpr unsigned roundUpDivision(unsigned dividend, unsigned divisor) noexcept
{
    return (dividend + (divisor - 1u)) / divisor;
}
}   // End of anon namespace.

//-----------------------------------------------------------------------------

TileWorkQueue::TileWorkQueue()
: mNumPasses(0)
, mNumTiles(0)
, mGroupClampIdx(0)
, mOffset(OffsetData{0, 0})
, mGlobalGroupIdx(0)
, mRuntimeDebug(false)
{
    // CPPCHECK -- Using memset() on struct which contains a floating point number.
    // cppcheck-suppress memsetClassFloat
    memset(mPassInfos, 0, sizeof(mPassInfos));

    parserConfigure();
}

TileWorkQueue::~TileWorkQueue() = default;

void
TileWorkQueue::init(RenderMode mode,
                    unsigned numTiles,
                    unsigned numPasses,
                    unsigned numRenderThreads,
                    const Pass *passes)
{
#ifdef RUNTIME_VERIFY
    if (TileGroupRuntimeVerify::get()) {
        TileGroupRuntimeVerify::get()->verify();
    }
    TileGroupRuntimeVerify::init(numTiles);
#endif // end RUNTIME_VERIFY

    MNRY_ASSERT(numPasses && numTiles && numRenderThreads);

    const unsigned minTilesPerGroup = 1;
    unsigned maxTilesPerGroup = 0;
    unsigned desiredSamplesPerGroup = 0;

    switch (mode) {
    case RenderMode::BATCH:
        maxTilesPerGroup       = 128;
        desiredSamplesPerGroup = 512;
        break;

    case RenderMode::PROGRESSIVE:
        maxTilesPerGroup       = 128;
        desiredSamplesPerGroup = 512;
        break;

    case RenderMode::PROGRESSIVE_FAST:
        maxTilesPerGroup       = 128;
        desiredSamplesPerGroup = 512;
        break;

    case RenderMode::REALTIME:
        maxTilesPerGroup       = 2;
        desiredSamplesPerGroup = 8;
        break;

    case RenderMode::PROGRESS_CHECKPOINT:
        if (passes[0].getNumSamplesPerTile() == 1) {
            // This is a very initial estimation stage
            // So this makes task as 10 tiles/thread for each thread.
            // See also RenderDriver.cc initPasses()
            maxTilesPerGroup       = 10;
            desiredSamplesPerGroup = 5;
        } else {
            // Otherwise back to same setting of BATCH and PROGRESSIVE
            maxTilesPerGroup       = 128;
            desiredSamplesPerGroup = 512;
        }
        break;

    default:
        MNRY_ASSERT(!"Should not get here");
    }

    MNRY_ASSERT(minTilesPerGroup > 0);
    MNRY_ASSERT(maxTilesPerGroup >= minTilesPerGroup);

    mNumPasses = numPasses;
    mNumTiles = numTiles;

    //
    // Compute tile groups within each pass.
    //

    unsigned groupIdx = 0;

    for (unsigned i = 0; i < mNumPasses; ++i) {

        const Pass &pass = passes[i];
        PassInfo *passInfo = &mPassInfos[i];

        unsigned samplesPerPixel = pass.mEndSampleIdx - pass.mStartSampleIdx;
        unsigned samplesPerTile = (pass.mEndPixelIdx - pass.mStartPixelIdx) * samplesPerPixel;
        unsigned tilesPerGroup = std::lround(static_cast<float>(desiredSamplesPerGroup) /
                                             static_cast<float>(samplesPerTile));
        tilesPerGroup = scene_rdl2::math::clamp<unsigned>(tilesPerGroup, minTilesPerGroup, maxTilesPerGroup);
        MNRY_ASSERT(tilesPerGroup);

        unsigned numGroups = mNumTiles / tilesPerGroup;
        if (numGroups * tilesPerGroup < mNumTiles) {
            ++numGroups;
            MNRY_ASSERT(numGroups * tilesPerGroup >= mNumTiles);
        }

        passInfo->mPass = pass;

        passInfo->mStartGroupIdx = groupIdx;
        passInfo->mEndGroupIdx = groupIdx + numGroups;
        passInfo->mTilesPerGroup = tilesPerGroup;

        groupIdx = passInfo->mEndGroupIdx;
    }

    reset();

    if (mRuntimeDebug) {
        std::cerr << show() << '\n';
    }
}

void
TileWorkQueue::reset()
{
    if (mNumPasses > 0) {
        unclampPasses();
    }

    mGlobalGroupIdx.store(0);
    mOffset.store(OffsetData{0, 0});
}

// Executes the up until and including this pass and then stops.
void
TileWorkQueue::clampToPass(unsigned passIdx)
{
    MNRY_ASSERT(mNumPasses);
    passIdx = std::min(passIdx, mNumPasses - 1u);
    mGroupClampIdx = mPassInfos[passIdx].mEndGroupIdx;
}

void
TileWorkQueue::unclampPasses()
{
    MNRY_ASSERT(mNumPasses);
    mGroupClampIdx = mPassInfos[mNumPasses - 1].mEndGroupIdx;
}

TileGroup
TileWorkQueue::reserveNextTileGroup()
{
    // TODO: C++17 change to is_always_lock_free and do a static assert, although this is desired behavior, we're
    // probably okay even if it's not true.
    MNRY_ASSERT(mOffset.is_lock_free());

    constexpr unsigned minTilesPerGroup = 1u;
    TileGroup group;

    OffsetData currentOffset = mOffset;
    while (true) {
        // Get pass information
        const PassInfo& passInfo = mPassInfos[currentOffset.mPass];
        const unsigned numGroupTiles = std::max(minTilesPerGroup, passInfo.mTilesPerGroup);

        // This will load-balance in that each tile group will only differ by
        // +/-1 from the (number of tiles)/(number of group tiles).
        //
        // groupsLeftInPass will eventually be 1, which leaves that group to
        // get the remainder of the tiles.
        const auto tilesLeftInPass = mNumTiles - currentOffset.mTile;
        const auto groupsLeftInPass = roundUpDivision(tilesLeftInPass, numGroupTiles);

        OffsetData nextOffset(currentOffset);
        if (groupsLeftInPass <= 1) {
            // We need to start a new pass next time
            nextOffset.mTile = 0;
            ++nextOffset.mPass;
        } else {
            const auto numTiles = tilesLeftInPass/groupsLeftInPass;
            nextOffset.mTile = currentOffset.mTile + numTiles;
            MNRY_ASSERT(nextOffset.mTile < mNumTiles);
        }

        // Check to see if we can actually claim this offset though CAS.
        if (mOffset.compare_exchange_strong(currentOffset, nextOffset)) {
            // We have successfully claimed this offset. Update our group data for returning.
            group.mPassIdx      = currentOffset.mPass;
            group.mStartTileIdx = currentOffset.mTile;
            group.mEndTileIdx   = (nextOffset.mTile == 0) ? mNumTiles : (nextOffset.mTile);

            MNRY_ASSERT(group.mEndTileIdx > group.mStartTileIdx);
            MNRY_ASSERT(group.mStartTileIdx < mNumTiles);
            MNRY_ASSERT(group.mEndTileIdx <= mNumTiles);
            return group;
        }
    }
}

bool
TileWorkQueue::getNextTileGroup(unsigned threadIdx, TileGroup *group, unsigned lastCoursePassIdx)
{
    MNRY_ASSERT(mNumPasses);

    // mGroupClampIdx is set by the main thread.
    auto groupIdx = mGlobalGroupIdx.load();
    do {
        if (groupIdx >= mGroupClampIdx) {
            return false;
        }
    } while (!mGlobalGroupIdx.compare_exchange_weak(groupIdx, groupIdx + 1u));
    *group = reserveNextTileGroup();

#ifdef RUNTIME_VERIFY
    TileGroupRuntimeVerify::get()->push(*group);
#endif // end RUNTIME_VERIFY

    /*
    // A debug variable to see how many groups we let through: this seems
    // consistent on the intial render, but varies on the resume render.
    // (Of course, they get printed out of order).
    static std::atomic<int> count(0);
    PRINT(++count);
    */

    const bool startOfFinePass = group->mStartTileIdx == 0 && group->mPassIdx == lastCoursePassIdx;
    group->mFirstFinePass = startOfFinePass;

    return true;
}

unsigned
TileWorkQueue::getTotalTileSamples() const
{
    unsigned totalTileSamples = 0;
    for (size_t i = 0; i < mNumPasses; ++i) {
        totalTileSamples += mPassInfos[i].mPass.getNumSamplesPerTile();
    }
    return totalTileSamples;
}

std::string
TileWorkQueue::show() const
{
    auto showPass = [&](const Pass &pass) -> std::string {
        std::ostringstream ostr;
        ostr
        << "Pass {\n"
        << "  pix(" << pass.mStartPixelIdx << "~" << pass.mEndPixelIdx << ")\n"
        << "  smp(" << pass.mStartSampleIdx << "~" << pass.mEndSampleIdx << ")\n"
        << "}";
        return ostr.str();
    };
    auto showPassInfo = [&](const PassInfo &passInfo) -> std::string {
        std::ostringstream ostr;
        ostr
        << "PassInfo {\n"
        << scene_rdl2::str_util::addIndent(showPass(passInfo.mPass)) << '\n'
        << "  TileGroup(" << passInfo.mStartGroupIdx << "~" << passInfo.mEndGroupIdx << ")\n"
        << "  mTilesPerGroup:" << passInfo.mTilesPerGroup << '\n'
        << "}";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "TileWorkQueue {\n";
    ostr << "  mNumTiles:" << mNumTiles << '\n'
         << "  mGroupClampIdx:" << mGroupClampIdx << '\n';
    ostr << "  mNumPasses:" << mNumPasses << " {\n";
    for (unsigned i = 0; i < mNumPasses; ++i) {
        ostr << "    i:" << i << '\n'
             << scene_rdl2::str_util::addIndent(showPassInfo(mPassInfos[i]), 2) << '\n';
    }
    ostr << "  }\n";
    ostr << "  mRuntimeDebug:" << scene_rdl2::str_util::boolStr(mRuntimeDebug) << '\n';
    ostr << "}";
    return ostr.str();
}

//-----------------------------------------------------------------------------

void
TileWorkQueue::parserConfigure()
{
    mParser.description("tileWorkQueue command");
    mParser.opt("show", "", "show all tileWorkQueue info",
                [&](Arg& arg) -> bool { return arg.msg(show() + '\n'); });
    mParser.opt("debug", "<on|off>", "set runtime debug condition",
                [&](Arg& arg) -> bool {
                    mRuntimeDebug = (arg++).as<bool>(0);
                    return arg.msg(scene_rdl2::str_util::boolStr(mRuntimeDebug) + '\n');
                });
}

} // namespace rndr
} // namespace moonray

