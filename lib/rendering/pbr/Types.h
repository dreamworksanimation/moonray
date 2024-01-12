// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once
#include "Types.hh"
#include <moonray/rendering/mcrt_common/Types.h> // for MAX_RENDER_PASSES
#include <moonray/rendering/shading/bsdf/BsdfSlice.h> // FrameState::ShadowTerminatorFix
#include <moonray/rendering/shading/Types.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Vec4.h>

#include <tbb/atomic.h>
#include <vector>


namespace scene_rdl2 {
namespace rdl2 {
class Layer;
}
}

namespace moonray {

namespace geom {
class Primitive;
}

namespace mcrt_common {
class ThreadLocalState;
}

namespace rt {
class EmbreeAccelerator;
class GPUAccelerator;
}

namespace pbr {

class AovSchema;
class Light;
class LightAovs;
class MaterialAovs;
class PathIntegrator;
class PixelFilter;
struct RayState;
struct RayStatev;
struct Sample2D;
class Scene;
class TLState;
class CryptomatteBuffer;

typedef uint32_t EmbreeId;

typedef scene_rdl2::math::Vec4f RenderColor; // Must match rndr::RenderColor!

finline uint32_t
makeTilePass(uint32_t tile, uint32_t pass)
{
    // 22 bits for the tile
    // 7 bits for the pass
    // 3 bits are unused and can be used for future development
    MNRY_ASSERT(tile <= 0x3fffff && pass < MAX_RENDER_PASSES);
    return (tile << 10) | (pass << 3);
}

finline uint32_t
getTile(uint32_t tilePass)
{
    return (tilePass >> 10) & 0x3fffff;
}

finline uint32_t
getPass(uint32_t tilePass)
{
    return (tilePass >> 3) & 0x7f;
}

// All frame state which need to be access inside of the pbr library should go
// in here.
struct FrameState
{
    FRAME_STATE_MEMBERS;

    // HUD validation.
    static uint32_t hudValidation(bool verbose) { FRAME_STATE_VALIDATION; }
};

enum BundledOcclRayDataFlags
{
    // As a future potential optimization, we could save a pointer de-reference
    //  if we stored these flags in the upper bits of BundledOcclRay::mDataPtrIdx.
    LPE = 1 << 0,
    LIGHT_SAMPLE = 1 << 1
};

struct BundledOcclRayData
{
    BUNDLED_OCCL_RAY_DATA_MEMBERS;

    static uint32_t hudValidation(bool verbose) { BUNDLED_OCCL_RAY_DATA_VALIDATION; }
};

MNRY_STATIC_ASSERT(sizeof(BundledOcclRayData) == 32);

struct CACHE_ALIGN BundledOcclRay
{
    // Inline helpers to make a BundledOcclRay look like a pbr::Ray from an
    // interface point of view. Useful when using templates.
    finline const scene_rdl2::math::Vec3f &getOrigin() const      { return mOrigin; }
    finline const scene_rdl2::math::Vec3f &getDirection() const   { return mDir; }
    finline float getStart() const                    { return mMinT; }
    finline float getEnd() const                      { return mMaxT; }
    finline float getTime() const                     { return mTime; }

    bool isValid() const;

    BUNDLED_OCCL_RAY_MEMBERS;

    // HUD validation.
    static uint32_t hvdValidation(bool verbose) { BUNDLED_OCCL_RAY_VALIDATION(VLEN); }
};

struct CACHE_ALIGN BundledOcclRayv
{
    uint8_t mPlaceholder[sizeof(BundledOcclRay) * VLEN];
};

MNRY_STATIC_ASSERT(sizeof(BundledOcclRay) == 128);
MNRY_STATIC_ASSERT(sizeof(BundledOcclRay) * VLEN == sizeof(BundledOcclRayv));

#ifdef __AVX512F__
struct ALIGN(64) BundledRadiance
#else
struct ALIGN(32) BundledRadiance
#endif
{
    BUNDLED_RADIANCE_MEMBERS;

    // HUD validation.
    static uint32_t hvdValidation(bool verbose) { BUNDLED_RADIANCE_VALIDATION(VLEN); }
};

struct CACHE_ALIGN BundledRadiancev
{
    uint8_t mPlaceholder[sizeof(BundledRadiance) * VLEN];
};

#ifdef __AVX512F__
MNRY_STATIC_ASSERT(sizeof(BundledRadiance) == 96);
#else
MNRY_STATIC_ASSERT(sizeof(BundledRadiance) == 96);
#endif

MNRY_STATIC_ASSERT(sizeof(BundledRadiance) * VLEN == sizeof(BundledRadiancev));

struct ALIGN(32) BundledAov
{
    static const unsigned MAX_AOV = 5;         // up to 5 aovs in a bunedled aov entry
    static const unsigned MAX_AOV_IDX = 0xffe; // 0xfff indicates an unused slot
    static const unsigned AOV_NUM_BITS = 12;
    static const uint64_t AOV_ALL_BITS = 0xfff;
    static const uint64_t AOV_ALL_UNUSED = 0xfffffffffffffff;

    finline BundledAov(uint32_t pixel, uint32_t deepDataHandle)
    {
        init(pixel, deepDataHandle);
    }
    finline void init(uint32_t pixel, uint32_t deepDataHandle)
    {
        mPixel = pixel;
        mDeepDataHandle = deepDataHandle;
        mIndices = AOV_ALL_UNUSED;
    }
    finline uint32_t aovIdx(unsigned aov) const
    {
        MNRY_ASSERT(aov < MAX_AOV);
        return uint32_t((mIndices >> (aov * AOV_NUM_BITS)) & AOV_ALL_BITS);
    }
    finline void setAov(unsigned aov, float val, uint32_t aovIdx)
    {
        MNRY_ASSERT(aovIdx <= MAX_AOV_IDX);
        mAovs[aov] = val;
        mIndices &= ~(AOV_ALL_BITS << (aov * AOV_NUM_BITS)); // clear
        mIndices |= ((aovIdx & AOV_ALL_BITS) << (aov * AOV_NUM_BITS)); // set
    }
    finline void setAov(unsigned aov, float val)
    {
        // Use this function to set a value in mAovs that does not have
        // an aovIdx. Normally, any value in mAovs should correspond to a valid aov index.
        // For closest filtering, we store depth in the aov slot just past
        // the aov value.  The index is set to 0xfff to indicate this.
        mAovs[aov] = val;
        mIndices |= ((AOV_ALL_BITS) << (aov * AOV_NUM_BITS));
    }

    float    mAovs[MAX_AOV]; // up to MAX_AOV values
    uint32_t mPixel;         // Screen coordinates of pixel.
    uint32_t mDeepDataHandle;
    uint64_t mIndices;       // 4 bit film indx + 5 * 12 bits for aov indices
};

MNRY_STATIC_ASSERT(sizeof(BundledAov) == 64);

struct ALIGN(16) BundledHeatMapSample
{
    int64_t  mTicks;
    uint32_t mPixel;     // Screen coordinates of pixel 

    // helper function to initialize an array of bundled heat map samples
    finline static void initArray(BundledHeatMapSample *b, unsigned &bIndx,
                                  uint32_t pixel)
    {
        bIndx = 0;
        b[bIndx].mTicks = 0;
        b[bIndx].mPixel = pixel;
    }

    // helper function for adding ticks to an array of bundled heat map samples
    finline static void addTicksToArray(BundledHeatMapSample *b, unsigned &bIndx,
                                        int64_t ticks, uint32_t pixel)
    {
        if (b[bIndx].mPixel != pixel) {
            bIndx++;
            b[bIndx].mTicks = 0;
            b[bIndx].mPixel = pixel;
        }
        b[bIndx].mTicks += ticks;
    }
};

MNRY_STATIC_ASSERT(sizeof(BundledHeatMapSample) == 16);


struct DeepData
{
    DEEP_DATA_MEMBERS;

    static uint32_t hudValidation(bool verbose) { DEEP_DATA_VALIDATION; }
};

MNRY_STATIC_ASSERT(sizeof(DeepData) == 64);


struct CryptomatteData
{
    CRYPTOMATTE_DATA_MEMBERS;

    static uint32_t hudValidation(bool verbose) { CRYPTOMATTE_DATA_VALIDATION; }
    void init(CryptomatteBuffer* buffer) {
        mRefCount = 1;                              // number of "owners" of this data; when 0, data is released
        mCryptomatteBuffer = buffer;                // cryptomatte buffer
        mHit = 0;                                   // if zero, hit a presence surface or terminated early
        mPrevPresence = 0;                          // whether previously on presence path (0 = false, 1 = true)
        mId = 0;                                    // id for hit
        mPosition = scene_rdl2::math::Vec3f(0.f);   // position at hit
        mNormal = scene_rdl2::math::Vec3f(0.f);     // shading normal at hit
        mPresenceDepth = -1;                        // presence depth, -1 indicates that this isn't a presence path
        mPathPixelWeight = 1.f;                     // accumulated presence (used to remove presence from beauty) 
        mIsFirstSample = 1;                         // is this the first sample added for this path? (0 = false, 1 = true)
    }
};

finline void
uint32ToPixelLocation(uint32_t val, unsigned *px, unsigned *py)
{
    *px = unsigned(val & 0xffff);
    *py = unsigned(val >> 16);
}

finline uint32_t
uint32ToPixelX(uint32_t val)
{
    return unsigned(val & 0xffff);
}

finline uint32_t
uint32ToPixelY(uint32_t val)
{
    return unsigned(val >> 16);
}

finline uint32_t
pixelLocationToUint32(unsigned px, unsigned py)
{
    MNRY_ASSERT(px <= 0xffff);
    MNRY_ASSERT(py <= 0xffff);
    return uint32_t((py << 16) | px);
}

enum class LightSamplingMode
{
    UNIFORM = 0,
    ADAPTIVE = 1
};

} // namespace pbr
} // namespace moonray

