// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include "Types.h"
#include <moonray/common/mcrt_util/Atomic.h>
#include <moonray/rendering/mcrt_common/Types.h> // for MAX_RENDER_PASSES
#include <scene_rdl2/common/grid_util/Arg.h>
#include <scene_rdl2/common/grid_util/Parser.h>

#include <atomic>

namespace moonray {
namespace rndr {

// This is the granularity which work is handled. A single TileGroup represents
// a single unit of work for a single thread.
struct TileGroup
{
    unsigned    mPassIdx;
    unsigned    mStartTileIdx;
    unsigned    mEndTileIdx;        // One past the end.
    bool        mFirstFinePass;     // If this is the first tile group of the fine pass this will be set to true.
};

//
// This is a shared queue which hands out work for all the render threads on request.
// Along with queue draining, it is the mechanism we use to load balance the work
// amongst threads. It's a virtual queue in that it looks like a queue to the outside
// world but doesn't explicitly store any elements in a queue internally. Instead,
// these are synthesized given an atomic incrementing counter and the current list
// of passes and tiles for the frame.
//
class TileWorkQueue
{
public:
    using Arg = scene_rdl2::grid_util::Arg;
    using Parser = scene_rdl2::grid_util::Parser;

                TileWorkQueue();
                ~TileWorkQueue();

    void        init(RenderMode mode,
                     unsigned numTiles,
                     unsigned numPasses,
                     unsigned numRenderThreads,
                     const Pass *passes);

    void        reset();

    unsigned    getNumPasses() const noexcept { return mNumPasses; }
    const Pass &getPass(unsigned idx) const { MNRY_ASSERT(idx < mNumPasses); return mPassInfos[idx].mPass; }

    // Pass out work up until and including this pass and then stop handing new work out.
    void        clampToPass(unsigned passIdx);

    // Keep rendering until there is no more work to do.
    void        unclampPasses();

    //
    // For each thread to query the next chunk of work. Thread-safe.
    //
    // Tiles may be assigned to multiple threads.
    //
    // getNextTileGroup returns false if there were no more tile groups left.
    //
    bool        getNextTileGroup(unsigned threadIdx, TileGroup* group, unsigned lastCoarsePassIdx);

    unsigned    getTotalTileSamples() const;

    unsigned getNumTiles() const { return mNumTiles; }

    std::string show() const;

    Parser& getParser() { return mParser; }

private:
    TileGroup reserveNextTileGroup();

    void parserConfigure();

    static constexpr std::size_t OffsetDataAlignment = alignof(std::atomic<std::uint64_t>);

#pragma pack(push, 1)
        struct alignas(OffsetDataAlignment) OffsetData
        {
            std::uint32_t mPass;
            std::uint32_t mTile;
        };
#pragma pack(pop)

    // TODO: C++17 change to always_lock_free
    static_assert(sizeof(OffsetData) <= 64, "We expect to be able to do atomic operations on this struct");

    // Each input pass has an associated PassInfo structure created for it.
    struct PassInfo
    {
        Pass        mPass;

        unsigned    mStartGroupIdx;     // TileGroup index.
        unsigned    mEndGroupIdx;       // TileGroup index.
        unsigned    mTilesPerGroup;
    };

    unsigned    mNumPasses{0};
    PassInfo    mPassInfos[MAX_RENDER_PASSES];

    unsigned                       mNumTiles{0};
    unsigned                       mGroupClampIdx{0};

    std::atomic<OffsetData>        mOffset{OffsetData{0, 0}};
    std::atomic<std::uint32_t>     mGlobalGroupIdx{0};

    Parser mParser;
    bool mRuntimeDebug;
};

//-----------------------------------------------------------------------------

} // namespace rndr
} // namespace moonray

