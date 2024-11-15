// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "PbrTLState.h"

#include <scene_rdl2/common/fb_util/PixelBuffer.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>

#include <list>
#include <tbb/mutex.h>
#include <vector>

namespace moonray {
namespace pbr {

/*

A note on terminology
---------------------

In the code, we use a mixture of terms derived from the Cryptomatte Specification and Arnold, supplemented by
our own terms.

We use the following terms of our own:

* fragment: the part of a pixel covered by a surface all having a single id

The Cryptomatte spec (see  https://github.com/Psyop/Cryptomatte/blob/master/specification/cryptomatte_specification.pdf)
uses the following terms:

* ID: the 32-bit id values which correspond to our deep ids
* coverage: the fraction of a pixel covered by a given fragment
* layer: a set of 4 output channels (RGBA) containing two ID/coverage data sets

Arnold uses the following terms:

* depth: the number of ID/coverage data sets to be output

The depth is user-specified and defaults to 6. The number of layers output to the EXR file will be (depth + 1) / 2,
and if an odd depth is requested then the final blue/alpha channel pair will contain zeroes as padding (which compress
to almost no size in the EXR).

*/

/*
    We can output multiple sets of Cryptomatte data: one for primary/camera rays
    and potentially two more for surfaces seen through reflections or transmission (refraction).

    To enable recording of the reflected/refracted data you must tag the materials
    by setting the "record_reflected_cryptomatte" and/or "record_refracted_cryptomatte" 
    attributes to true on the  material.

    The reflected/refracted cryptomatte output is written to a separate set of render output channels
    that are named the same as the regular cryptomatte channels except with "Reflect" or "Refract" appended.
*/
enum CryptomatteType {
    CRYPTOMATTE_TYPE_REGULAR,
    CRYPTOMATTE_TYPE_REFLECTED,
    CRYPTOMATTE_TYPE_REFRACTED,
    NUM_CRYPTOMATTE_TYPES
};


class CryptomatteBuffer
{
private:

    struct Fragment
    {
        float mId;
        float mCoverage;
        scene_rdl2::math::Vec3f mPosition;
        scene_rdl2::math::Vec3f mP0;
        scene_rdl2::math::Vec3f mNormal;
        scene_rdl2::math::Color4 mBeauty;
        scene_rdl2::math::Vec3f mRefP;
        scene_rdl2::math::Vec3f mRefN;
        scene_rdl2::math::Vec2f mUV;
        unsigned mPresenceDepth;
        unsigned mNumSamples;    // num pixel samples that hit this id -- used to average position/normal data

        Fragment(float id, float coverage,
                 const scene_rdl2::math::Vec3f& position,
                 const scene_rdl2::math::Vec3f& p0,
                 const scene_rdl2::math::Vec3f& normal,
                 const scene_rdl2::math::Color4& beauty,
                 const scene_rdl2::math::Vec3f refP,
                 const scene_rdl2::math::Vec3f refN,
                 const scene_rdl2::math::Vec2f uv,
                 unsigned presenceDepth, unsigned numSamples = 1)
        : mId(id),
          mCoverage(coverage),
          mPosition(position),
          mP0(p0),
          mNormal(normal),
          mBeauty(beauty),
          mRefP(refP),
          mRefN(refN),
          mUV(uv),
          mPresenceDepth(presenceDepth),
          mNumSamples(numSamples)
        {}
    };

    struct PixelEntry
    {
        std::list<Fragment> mFragments;
    };


public:
    CryptomatteBuffer();
    ~CryptomatteBuffer();

    void init(unsigned width, unsigned height, unsigned numIdChannels, bool multiPresenceOn);

    void clear();

    // ---------------------------------------- Setters ----------------------------------------------------------------
    void setFinalized(bool finalized) { mFinalized = finalized; }
    void setMultiPresenceOn(bool multiPresenceOn) { mMultiPresenceOn = multiPresenceOn; }

    // ---------------------------------------- Getters ----------------------------------------------------------------
    unsigned getWidth()     const { return mWidth; }
    unsigned getHeight()    const { return mHeight; }
    bool getMultiPresenceOn() const { return mMultiPresenceOn; }

    // -----------------------------------------------------------------------------------------------------------------

    void addSampleScalar(unsigned x, unsigned y, float id, float weight,
                         const scene_rdl2::math::Vec3f& position,
                         const scene_rdl2::math::Vec3f& p0,
                         const scene_rdl2::math::Vec3f& normal,
                         const scene_rdl2::math::Color4& beauty,
                         const scene_rdl2::math::Vec3f refP,
                         const scene_rdl2::math::Vec3f refN,
                         const scene_rdl2::math::Vec2f uv,
                         unsigned presenceDepth,
                         int cryptoType);

    // For details on why we have the incrementSamples parameter, see CryptomatteBuffer.cc::addBeautySampleVector
    void addSampleVector(unsigned x, unsigned y, float id, float weight,
                         const scene_rdl2::math::Vec3f& position,
                         const scene_rdl2::math::Vec3f& normal,
                         const scene_rdl2::math::Color4& beauty,
                         const scene_rdl2::math::Vec3f refP,
                         const scene_rdl2::math::Vec3f& p0,
                         const scene_rdl2::math::Vec3f refN,
                         const scene_rdl2::math::Vec2f uv,
                         unsigned presenceDepth,
                         bool incrementSamples = true);

    // see CryptomatteBuffer.cc::addBeautySampleVector for info on why this function exists only in vector mode
    void addBeautySampleVector(unsigned x, unsigned y, float id, const scene_rdl2::math::Color4& beauty, unsigned depth);

    void finalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount);
    void outputFragments(unsigned x, unsigned y, int numLayers, float *dest, const scene_rdl2::rdl2::RenderOutput& ro) const;

    void unfinalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount);
    void addFragments(unsigned x, unsigned y,
                      const scene_rdl2::rdl2::RenderOutput& ro,
                      const float *idAndCoverageData[NUM_CRYPTOMATTE_TYPES],
                      const float *positionData[NUM_CRYPTOMATTE_TYPES],
                      const float *p0Data[NUM_CRYPTOMATTE_TYPES],
                      const float *normalData[NUM_CRYPTOMATTE_TYPES],
                      const float *beautyData[NUM_CRYPTOMATTE_TYPES],
                      const float *refPData[NUM_CRYPTOMATTE_TYPES],
                      const float *refNData[NUM_CRYPTOMATTE_TYPES],
                      const float *uvData[NUM_CRYPTOMATTE_TYPES],
                      const float *resumeRenderSupportData[NUM_CRYPTOMATTE_TYPES]);


    // --------------------------------------------- Print -------------------------------------------------------------
    void printAllPixelEntries() const;
    void printFragments(unsigned x, unsigned y, int cryptoType) const;

private:
    // Three sets of pixel entries: one for the regular cryptomatte data, and one each for reflected
    // and refracted cryptomatte data.
    std::vector<PixelEntry> mPixelEntries[NUM_CRYPTOMATTE_TYPES];

    // buffer dimensions
    unsigned mWidth;
    unsigned mHeight;

    bool mFinalized;

    bool mMultiPresenceOn;

/* The following notes are adapted from the DeepBuffer mutex description.
 *
 * Cryptomatte pixels are independent of each other, so there is no threading hazard
 * when different threads are writing to different pixels.  It is possible
 * (although uncommon) that multiple threads might write to the same pixel,
 * so that needs to be protected with a mutex.  The simple solution is to
 * have one mutex per pixel.  This would consume a lot of memory, so instead
 * there is an array of 225 mutexes for the entire image that are shared
 * between the pixels.  Some pixels will share the same mutex, which results
 * in some extra locking... however testing has shown the overhead is negligible
 * once there are >= 64 mutexes for ~20 rendering threads.  This may need to be
 * increased as the number of threads increases though.
 * The mutexes form a 15x15 tile that repeats across the image, and the
 * getMutexIdx() logic maps a screen coordinate to a [0, 225] tile index.
 * Note that the tile size is not a multiple of the rendering 8x8 tile size.
 * This reduces mutex sharing during rendering.
 */
    static const int mMutexTileSize = 15;
    // force mutex to be cache-line aligned for speed
    struct CACHE_ALIGN AlignedMutex : public tbb::mutex {};
    AlignedMutex *mPixelMutexes;
    int getMutexIdx(unsigned x, unsigned y) const {
        return (y % mMutexTileSize) * mMutexTileSize + (x % mMutexTileSize);
    }
};

}
}

