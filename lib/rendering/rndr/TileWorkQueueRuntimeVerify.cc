// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "TileWorkQueueRuntimeVerify.h"

#ifdef RUNTIME_VERIFY_TILE_WORK_QUEUE

#include <iomanip>
#include <iostream>
#include <memory>

namespace moonray {
namespace rndr {

bool
TileGroupRuntimeVerify::verify()
{
    std::cerr << ">> TileWorkQueueRuntimeVerify.cc " << show() << '\n';
    return verifyTileGroupArray();
}

std::string
TileGroupRuntimeVerify::show() const
{
    std::ostringstream ostr;
    ostr << "TileGroupArray (total:" << mTileGroupArray.size() << ") {\n";
    for (size_t i = 0; i < mTileGroupArray.size(); ++i) {
        ostr << "  " << showTileGroup(mTileGroupArray[i]) << '\n';
    }
    ostr << "}";
    return ostr.str();
}
                                   
//------------------------------------------------------------------------------------------

bool operator < (const TileGroup &A, const TileGroup &B) 
{
    return A.mStartTileIdx < B.mStartTileIdx;
}

bool
TileGroupRuntimeVerify::verifyTileGroupArray() const
{
    TileGroupArray array = mTileGroupArray;

    bool result = true;

    std::cerr << "verifyTileGroupArray start\n";
    while (1) {
        if (array.size() == 0) break;

        unsigned passId = getMinPassId(array);
        TileGroupArray A = makeTileGroupArray(array, passId); // pickup tileGroups which has passId
        TileGroupArray B = subTileGroupArray(array, A); // B = array - A

        std::sort(A.begin(), A.end());
        unsigned lastTileIdx = verifyStartTileIdxOrder(A);
        if (lastTileIdx == mNumTiles) {
            std::cerr << "passId:" << passId << " verify:OK\n";
        } else {
            std::cerr << "passId:" << passId << " verify:NG" << showTileGroupArray(A) << '\n';
            result = false;
        }

        array = B;
    }
    std::cerr << "verifyTileGroupArray end : VERIFY-" << (result ? "OK" : "NG") << '\n';

    return result;
}

// static function
unsigned
TileGroupRuntimeVerify::getMinPassId(const TileGroupArray &srcTileGroupArray)
{
    unsigned minPassId = 0;
    for (size_t i = 0; i < srcTileGroupArray.size(); ++i) {
        unsigned currPassId = srcTileGroupArray[i].mPassIdx;
        if (i == 0) {
            minPassId = currPassId;
        } else {
            if (currPassId < minPassId) minPassId = currPassId;
        }
    }
    return minPassId;
}

// static function
TileGroupRuntimeVerify::TileGroupArray
TileGroupRuntimeVerify::makeTileGroupArray(const TileGroupArray &srcTileGroupArray, unsigned passIdx)
{
    TileGroupArray array;
    for (size_t i = 0; i < srcTileGroupArray.size(); ++i) {
        if (srcTileGroupArray[i].mPassIdx == passIdx) {
            array.push_back(srcTileGroupArray[i]);
        }
    }
    return array;
}
                                   
// static function
TileGroupRuntimeVerify::TileGroupArray
TileGroupRuntimeVerify::subTileGroupArray(const TileGroupArray &A, const TileGroupArray &B)
// return A - B
{
    TileGroupArray array;
    for (size_t i = 0; i < A.size(); ++i) {
        if (!findTileGroup(B, A[i])) array.push_back(A[i]);
    }
    return array;
}

// static function
unsigned
TileGroupRuntimeVerify::verifyStartTileIdxOrder(const TileGroupArray &srcTileGroupArray)
// return last tileGoup's endTileIdx
{
    if (!srcTileGroupArray.size()) return 0;

    unsigned currId = srcTileGroupArray[0].mEndTileIdx;
    for (size_t i = 1; i < srcTileGroupArray.size(); ++i) {
        const TileGroup &currTileGroup = srcTileGroupArray[i];
        if (currId != currTileGroup.mStartTileIdx) break;
        currId = currTileGroup.mEndTileIdx;
    }
    return currId;
}

// static function
bool
TileGroupRuntimeVerify::findTileGroup(const TileGroupArray &srcTileGroupArray, const TileGroup &tileGroup)
{
    for (size_t i = 0; i < srcTileGroupArray.size(); ++i) {
        if (isSameTileGroup(srcTileGroupArray[i], tileGroup)) return true;
    }
    return false;
}

// static function
std::string
TileGroupRuntimeVerify::showTileGroupArray(const TileGroupArray &srcTileGroupArray)
{
    std::ostringstream ostr;
    ostr << "TileGroupArray (size:" << srcTileGroupArray.size() << ") {\n";
    int w = std::to_string(srcTileGroupArray.size()).size();
    for (size_t i = 0; i < srcTileGroupArray.size(); ++i) {
        ostr << "  i:" << std::setw(w) << i << " " << showTileGroup(srcTileGroupArray[i]) << '\n';
    }
    ostr << "}";
    return ostr.str();
}

// static function
std::string
TileGroupRuntimeVerify::showTileGroup(const TileGroup &srcTileGroup)
{
    std::ostringstream ostr;
    ostr << "TileGroup { mPassIdx:" << std::setw(1) << srcTileGroup.mPassIdx
         << " startTileIdx:" << std::setw(4) << srcTileGroup.mStartTileIdx
         << " endTileIdx:" << std::setw(4) << srcTileGroup.mEndTileIdx
         << " finePass:" << ((srcTileGroup.mFirstFinePass) ? "T" : "F") << "}";
    return ostr.str();
}
    
// static function
bool
TileGroupRuntimeVerify::isSameTileGroup(const TileGroup &A, const TileGroup &B)
{
    return (A.mPassIdx == B.mPassIdx &&
            A.mStartTileIdx == B.mStartTileIdx &&
            A.mEndTileIdx == B.mEndTileIdx &&
            A.mFirstFinePass == B.mFirstFinePass);
}

//------------------------------------------------------------------------------------------

static std::unique_ptr<TileGroupRuntimeVerify> sTileGroupRuntimeVerify;

// static function
void
TileGroupRuntimeVerify::init(unsigned numTiles)
{
    sTileGroupRuntimeVerify.reset(new TileGroupRuntimeVerify(numTiles));
}

// static function
TileGroupRuntimeVerify *
TileGroupRuntimeVerify::get()
{
    return sTileGroupRuntimeVerify.get();
}

} // namespace rndr
} // namespace moonray

#endif // end RUNTIME_VERIFY_TILE_WORK_QUEUE

