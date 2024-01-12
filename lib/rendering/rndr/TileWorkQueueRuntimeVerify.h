// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "RunTimeVerify.h"

#ifdef RUNTIME_VERIFY_TILE_WORK_QUEUE

#include "TileWorkQueue.h"

#include <mutex>
#include <vector>

namespace moonray {
namespace rndr {

class TileGroupRuntimeVerify
//
// This class is designed for runtime verification of properly assignment tileGroup to each MCRT thread.
// The verification has an impact on the execution performance and this class should not be used for the
// release version of moonray. This is only used for debugging purposes.
// Also verification logic itself is pretty depending on the current tileWorkQueue implementation.
// And also verification logic is designed in a pretty naive way and does not have much attention for
// execution performance. It would be useful to keep this verification code for debug purposes but should
// be reworked if tileWorkQueue is redesigned its internal logic.
//
{
public:
    using TileGroupArray = std::vector<TileGroup>;

    static void init(unsigned numTiles);
    static TileGroupRuntimeVerify *get();

    explicit TileGroupRuntimeVerify(unsigned numTiles) : mNumTiles(numTiles) {};
    ~TileGroupRuntimeVerify() = default;

    void reset(unsigned numTiles) {
        mNumTiles = numTiles;
        std::lock_guard<std::mutex> lock(mMutex);
        mTileGroupArray.clear();
    }

    void push(const TileGroup &group) {
        std::lock_guard<std::mutex> lock(mMutex);
        mTileGroupArray.push_back(group);
    }

    bool verify();

    std::string show() const;

protected:
    bool verifyTileGroupArray() const;

    static unsigned getMinPassId(const TileGroupArray &srcTileGroupArray);
    static TileGroupArray makeTileGroupArray(const TileGroupArray &srcTileGroupArray, unsigned passIdx);
    static TileGroupArray subTileGroupArray(const TileGroupArray &A, const TileGroupArray &B); // return A - B
    static unsigned verifyStartTileIdxOrder(const TileGroupArray &srcTileGroupArray);
    static bool findTileGroup(const TileGroupArray &srcTileGroupArray, const TileGroup &tileGroup);
    static std::string showTileGroupArray(const TileGroupArray &srcTileGroupArray);
    static std::string showTileGroup(const TileGroup &srcTileGroup);
    static bool isSameTileGroup(const TileGroup &A, const TileGroup &B);

    //------------------------------

    unsigned mNumTiles;

    mutable std::mutex mMutex;;
    TileGroupArray mTileGroupArray;
};

} // namespace rndr
} // namespace moonray

#endif // end RUNTIME_VERIFY_TILE_WORK_QUEUE

