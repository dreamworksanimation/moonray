// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeRegions.h
///

#pragma once

#include <scene_rdl2/render/util/BitUtils.h>

namespace moonray {
namespace geom {
namespace internal {

class Primitive;

/// This class serves the functionality of tracking status of volume in 3d space
/// (ie. which volume regions a point or an interval is belong to)
/// we need to consider the scenario where a point or an interval can belong to
/// multiple volume regions (fire inside smoke, for example)
class VolumeRegions
{
public:
    VolumeRegions() : mBitMask({0}) {}

    size_t getMaxRegionsCount() const { return 32 * mBitMask.size(); }

    void reset() {
        memset(mBitMask.data(), 0, sizeof(uint32_t) * mBitMask.size());
    }

    bool isOn(int volumeId) const {
        MNRY_ASSERT(volumeId < (int)getMaxRegionsCount());
        return (mBitMask[volumeId >> 5]) & (1 << (volumeId & 31));
    }

    bool isVacuum() const {
        for (uint32_t i = 0; i < mBitMask.size(); ++i) {
            if (mBitMask[i] != 0) {
                return false;
            }
        }
        return true;
    }

    void turnOn(int volumeId, const Primitive* primitive) {
        MNRY_ASSERT(volumeId < (int)getMaxRegionsCount());
        mBitMask[volumeId >> 5] |= 1 << (volumeId & 31);
        mPrimitives[volumeId] = primitive;
    }

    void turnOff(int volumeId) {
        MNRY_ASSERT(volumeId < (int)getMaxRegionsCount());
        mBitMask[volumeId >> 5] &= ~(1 << (volumeId & 31));
    }

    int getRegionsCount() const {
        int regionsCount = 0;
        for (uint32_t i = 0; i < mBitMask.size(); ++i) {
            regionsCount += scene_rdl2::util::countOnBits(mBitMask[i]);
        }
        return regionsCount;
    }

    int getVolumeIds(int* volumeIds) const {
        // locate all 1 bits in bit mask
        int regionsCount = 0;
        for (uint32_t i = 0; i < mBitMask.size(); ++i) {
            uint32_t v = mBitMask[i];
            uint32_t baseIdx = (i << 5);
            while (v != 0) {
                // Note: __builtin_ctz(0) is undefined but that's ok since we're
                //       handling that case.
                uint32_t idx = __builtin_ctz(v);    // index of lowest 1-bit
                volumeIds[regionsCount++] = baseIdx + idx;
                v &= v - 1;                         // set lowest 1-bit to zero
            }
        }
        return regionsCount;
    }

    const Primitive* getPrimitive(int volumeId) const {
        return mPrimitives[volumeId];
    }

private:
    // TODO 32 * 16 = 512 volume regions are probably not sufficient for extreme
    // production case, we can extend it in the future.
    std::array<uint32_t, 16> mBitMask;
    const Primitive* mPrimitives[512]; // per volumeID
    // We don't need to initialize this in the constructor because the values are
    //  set when we turnOn() and aren't read back unless the region is on.
};

} // namespace internal
} // namespace geom
} // namespace moonray

