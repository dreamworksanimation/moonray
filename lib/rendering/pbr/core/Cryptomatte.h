// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "PbrTLState.h"

#include <scene_rdl2/common/fb_util/PixelBuffer.h>

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

class CryptomatteBuffer
{

private:

    struct Fragment
    {
        float mId;
        float mCoverage;

        Fragment(float id, float coverage)
        : mId(id), mCoverage(coverage)
        {}
    };

    struct PixelEntry
    {
        std::list<Fragment> mFragments;
    };


public:
    CryptomatteBuffer();
    ~CryptomatteBuffer();

    void init(unsigned width, unsigned height, unsigned numIdChannels);

    void clear();

    unsigned getWidth()     const { return mWidth; }
    unsigned getHeight()    const { return mHeight; }

    void addSampleScalar(unsigned x, unsigned y, float id, float weight);
    void addSampleVector(unsigned x, unsigned y, float id, float weight);
    void addFragments(unsigned x, unsigned y, const float *data, int numFragments);

    void finalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount);
    void unfinalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount);
    void setFinalized(bool finalized) { mFinalized = finalized; }

    void outputFragments(unsigned x, unsigned y, int numLayers, float *dest) const;

    void printAllPixelEntries() const;
    void printFragments(unsigned x, unsigned y) const;

private:
    std::vector<PixelEntry> mPixelEntries;

    // buffer dimensions
    unsigned mWidth;
    unsigned mHeight;

    bool mFinalized;

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

