// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "Film.h"
#include "AdaptiveRenderTileInfo.h"
#include "AdaptiveRenderTilesTable.h"
#include "PixelBufferUtils.h"
#include "Util.h"

#include <moonray/common/mcrt_util/Atomic.h>

#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>

#include <scene_rdl2/common/fb_util/TileExtrapolation.h>
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/common/math/Viewport.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

// Enable debugSamplesRecArray logic (record/playback all computeRadiance() result for beauty AOV)
// In order to activate debugSamplesRecArray mode, you should modify code for save or load.
//#define DEBUG_SAMPLE_REC_MODE

namespace moonray {
namespace rndr {

namespace
{

// Atomic free version of mathFilter, suitable for vector execution.
inline void
atomicMathFilter(float *dest, const float *src, const size_t sz, float depth, const pbr::AovFilter filter)
{
    if (filter == pbr::AOV_FILTER_CLOSEST) {

        // entire aov needs to be handled atomically - can't do it one component at a time.
        if (scene_rdl2::math::isfinite(depth)) {
            for (size_t i = 0; i < sz; ++i) {
                if (!scene_rdl2::math::isfinite(src[i])) return;
            }

            alignas(util::kDoubleQuadWordAtomicAlignment) float newVal[4];
            newVal[3] = depth;
            switch (sz) {
            case 1:
                newVal[0] = src[0];
                break;
            case 2:
                newVal[0] = src[0];
                newVal[1] = src[1];
                break;
            case 3:
                newVal[0] = src[0];
                newVal[1] = src[1];
                newVal[2] = src[2];
                break;
            default:
                MNRY_ASSERT(0 && "unexpected size");
            }
            util::atomicAssignIfClosest(dest, newVal);
        }

        return;
    }

    for (size_t i = 0; i < sz; ++i) {
        // only execute the atomic adds, mins and maxs if we absolutely need to.
        // even though some aov values might be +inf or -inf, we assume that
        // the aov is set to min or max filtering in those cases, and that the
        // buffer is already cleared with the appropriate +inf or -inf.  So it
        // is safe to skip all cases when the aov value is +inf or -inf
        if (scene_rdl2::math::isfinite(*(src+i))) {
            switch (filter) {
            case pbr::AOV_FILTER_AVG:
            case pbr::AOV_FILTER_SUM:
                if (*(src+i) != 0.f) {
                    util::atomicAdd(dest + i, *(src+i));
                }
                break;
            case pbr::AOV_FILTER_MIN:
                util::atomicMin(dest + i, *(src+i));
                break;
            case pbr::AOV_FILTER_MAX:
                util::atomicMax(dest + i, *(src+i));
                break;
            case pbr::AOV_FILTER_CLOSEST:
            default:
                MNRY_ASSERT(0 && "unexpected aov scene_rdl2::math filter");
            }
        }
    }
}

#ifdef DEBUG
bool
areBundledEntriesValid(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                       pbr::BundledRadiance **entries)
{
    MNRY_ASSERT(numEntries);

    uint32_t prevPixel = 0;

    for (unsigned i = 0; i < numEntries; ++i) {
        pbr::BundledRadiance *br = entries[i];

        MNRY_ASSERT(scene_rdl2::math::isFinite(br->mRadiance));
        MNRY_ASSERT(br->mPathPixelWeight >= 0.f);

        uint32_t pixel = br->mPixel;
        MNRY_ASSERT(prevPixel <= pixel);
        prevPixel = pixel;

        MNRY_ASSERT(pbr::getPass(br->mTilePass) < MAX_RENDER_PASSES);
    }

    return true;
}
#endif

#ifdef DEBUG
bool
areBundledAovEntriesValid(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                          pbr::BundledAov **entries)
{
    MNRY_ASSERT(numEntries);

    uint32_t prevPixel = 0;

    for (unsigned i = 0; i < numEntries; ++i) {
        pbr::BundledAov *ba = entries[i];

        // handle each aov in the bundle
        for (unsigned aov = 0; aov < pbr::BundledAov::MAX_AOV; ++aov) {
            unsigned aovIdx = ba->aovIdx(aov);
            if (aovIdx <= pbr::BundledAov::MAX_AOV_IDX) {
                // +inf, -inf are valid for some aovs
                MNRY_ASSERT(!std::isnan(ba->mAovs[aov]));

                uint32_t pixel = ba->mPixel;
                MNRY_ASSERT(prevPixel <= pixel);
                prevPixel = pixel;
            }
        }
    }

    return true;
}
#endif

scene_rdl2::fb_util::VariablePixelBuffer::Format
aovBufferFormat(pbr::AovStorageType type)
{
    switch (type) {
    case pbr::AovStorageType::FLOAT:          return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT;
    case pbr::AovStorageType::VEC2:           return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2;
    case pbr::AovStorageType::VEC3:           return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3;
    case pbr::AovStorageType::VEC4:           return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4;
    case pbr::AovStorageType::RGB:            return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3;
    case pbr::AovStorageType::RGB4:           return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4;
    case pbr::AovStorageType::VISIBILITY:     return scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2;
    default:
        MNRY_ASSERT(0 && "unhandled type");
    }
    return scene_rdl2::fb_util::VariablePixelBuffer::UNINITIALIZED;
}

inline void
extrapolatePartialTileV2(scene_rdl2::fb_util::RenderColor *__restrict dst,
                         const scene_rdl2::fb_util::RenderColor *__restrict srcColor,
                         const float *__restrict srcWeight,
                         unsigned minX, unsigned maxX, unsigned minY, unsigned maxY,
                         const scene_rdl2::fb_util::TileExtrapolation *tileExtrapolation)
{
    //
    // Version 2 of extrapolation code.
    //
    scene_rdl2::fb_util::RenderColor workPixel[64];
    uint64_t activePixelMask = (uint64_t)0x0;
    for (unsigned y = minY; y < maxY; ++y) {
        for (unsigned x = minX; x < maxX; ++x) {
            unsigned offset = (y << 3) + x;
            if (srcWeight[offset] > 0.f) {
                workPixel[offset] = srcColor[offset] * (1.f / srcWeight[offset]);
                activePixelMask |= ((uint64_t)0x1 << offset);
            }
        } // loop x
    } // loop y

    int extrapolatePixIdArray[64];
    tileExtrapolation->searchActiveNearestPixel(activePixelMask, extrapolatePixIdArray,
                                                minX, maxX, minY, maxY);

    for (unsigned y = minY; y < maxY; ++y) {
        for (unsigned x = minX; x < maxX; ++x) {
            unsigned offset = (y << 3) + x;
            dst[offset] = workPixel[extrapolatePixIdArray[offset]];
        }
    }
}

void
copyPixIdTable(const std::vector<uint8_t> &src, uint8_t dst[64])
{
    for (size_t i = 0; i < 64; ++i) {
        dst[i] = src[i];
    }
}

}   // End of anon namespace.

//-----------------------------------------------------------------------------

//
// Each element in the gPixelFillOrder array represents an xy pair in an 8x8 tile.
// To decode, do the following:
//
//    x = element & 7;
//    y = element >> 3;
//
// The pattern is designed to be used in conjunction with the extrapolateQuads
// function (only V1 logic. V2 does not have any strict pixel order requirement).
// (for more detail about V2, see lib/common/fb_util/TileExtrapolation.cc).
//
// >> V1 logic detail <<
// The ordering ensures that we can fill the 8x8 tile with the nearest
// color sample with only 4 lookups per pixel. Conceptually we are first
// searching in the current 1x1 quad, if a sample isn't found then we search
// within a 2x2, and so on for 4x4 and 8x8, thus giving the 4 lookups.
//
// This results in the following pixel order:
//
//    04 36  12 44    06 38  14 46
//    52 20  60 28    54 22  62 30
//
//    08 40  00 32    10 42  02 34
//    56 24  48 16    58 26  50 18
//
//
//    07 39  15 47    05 37  13 45
//    55 23  63 31    53 21  61 29
//
//    11 43  03 35    09 41  01 33
//    59 27  51 19    57 25  49 17
//
// Note the nested ordering patterns:
//
//    For 4x4 blocks within 8x8 blocks we have: A C
//                                              D B
//
//    For 2x2 blocks within 4x4 blocks we have: B D
//                                              C A
//
//    For 1x1 blocks within 2x2 blocks we have: A C
//                                              D B
//

const uint8_t gPixelFillOrder[64] =
{
    18, 54, 22, 50,  0, 36,  4, 32,
    16, 52, 20, 48,  2, 38,  6, 34,
    27, 63, 31, 59,  9, 45, 13, 41,
    25, 61, 29, 57, 11, 47, 15, 43,
    19, 55, 23, 51,  1, 37,  5, 33,
    17, 53, 21, 49,  3, 39,  7, 35,
    26, 62, 30, 58,  8, 44, 12, 40,
    24, 60, 28, 56, 10, 46, 14, 42,
};

// Scrambled pixel fill order table for Multiplex Pixel Distribution (tileId 0 ~ 1023)
// This information is used by distributed multi-MCRT computations only.
// We create pre-scrambled pixel fill order for first 1024 tileId entry.
// One tile has 8x8 = 64 pixels and this table keeps pixelId order by pixId.
//
// Under multi-MCRT computation context with multiplex pixel distribution mode, if each MCRT computation uses
// same pixel order to compute inside tile, as result very early stage of image generation can not utilize
// advantage of multi-machine well. Because every MCRT computation follows exactly same pixel order to compute.
// This means all MCRT computations are focused on increase quality of same pixels.
// If possible, at early stage of rendering, we would like to see low resolution samples for many different
// pixels instead of focused on same pixels with increase quality.
// If we use randomly shuffled different pixel Id order pattern for different MCRT computation, this is much
// better especially early stage of image generation because each MCRT computation try to get different pixel
// to compute and we can get early stage image faster convergence speed.
// This is a reason why using pre-shuffled scrambled pixel fill order for multiplex pixel distribution mode.
//
uint8_t gScrambledPixelFillOrder[1024][64];

Film::Film() :
    mFilmActivity(0),
    mPixelInfoBufActivity(0),
    mUseAdaptiveSampling(false),
    mRenderBufOdd(nullptr),
    mDeepBuf(nullptr),
    mCryptomatteBuf(nullptr),
    mAovBufNumFloats(0),
    mAovHasClosestFilter(false),
    mHeatMapBuf(nullptr),
    mTileExtrapolation(nullptr),
    mResumedFromFileCondition(false)
{
}

Film::~Film()
{
    delete mHeatMapBuf;
    delete mPixelInfoBuf;
    delete mDeepBuf;
    delete mCryptomatteBuf;
    delete mRenderBufOdd;
}

void
Film::init(unsigned w, unsigned h,
           const scene_rdl2::math::Viewport &viewport,
           uint32_t flags,
           int deepFormat,
           float deepCurvatureTolerance,
           float deepZTolerance,
           unsigned deepVolCompressionRes,
           const std::vector<std::string>& deepIDChannelNames,
           int deepMaxLayers,
           unsigned numRenderThreads,
           const pbr::AovSchema &aovSchema,
           const int numDisplayFilters,
           const scene_rdl2::fb_util::TileExtrapolation *tileExtrapolation,
           const unsigned maxSamplesPerPixel, // for construction of AdaptiveRenderTileTable
           float targetAdaptiveError,
           bool multiPresenceOn
           )
{
    mTileExtrapolation = tileExtrapolation;

    MNRY_ASSERT(w * h > 0);
    mTiler = scene_rdl2::fb_util::Tiler(w, h);

    unsigned alignedW = mTiler.mAlignedW;
    unsigned alignedH = mTiler.mAlignedH;

    // All buffers which we can run extrapolation on must be aligned to tile
    // boundaries.
    mRenderBuf.init(alignedW, alignedH);
    mWeightBuf.init(alignedW, alignedH);

    // aov buffer
    mAovBuf.clear();
    mAovBuf.resize(aovSchema.size());
    mAovEntries.clear();
    mAovEntries.reserve(aovSchema.size());
    mAovBufNumFloats = aovSchema.numChannels();
    mAovBeautyBufIdx.clear();
    mAovBeautyBufIdx.reserve(aovSchema.size());
    mAovAlphaBufIdx.clear();
    mAovAlphaBufIdx.reserve(aovSchema.size());

    std::for_each(aovSchema.begin(), aovSchema.end(),
        [count = 0, this, alignedW, alignedH](const pbr::AovSchema::Entry& entry) mutable
    {
        mAovBuf[count].init(aovBufferFormat(entry.storageType()), alignedW, alignedH);
        mAovEntries.push_back(entry);
        if (entry.id() == pbr::AOV_SCHEMA_ID_BEAUTY) {
            mAovBeautyBufIdx.push_back(count);
        }
        if (entry.id() == pbr::AOV_SCHEMA_ID_ALPHA) {
            mAovAlphaBufIdx.push_back(count);
        }
        ++count;
    });
    mAovHasClosestFilter = aovSchema.hasClosestFilter();

    mAovBeautyBufIdx.shrink_to_fit();
    mAovAlphaBufIdx.shrink_to_fit();

    // aov buffer indices
    // This structure is used inside addAovSampleBufferHandler when we need to
    // know which buffer corresponds to which aov index.
    mAovIdxToBufIdx.clear();
    mAovIdxToBufIdx.resize(mAovBufNumFloats);
    unsigned int idx = 0;
    for (unsigned int i = 0; i < aovSchema.size(); ++i) {
        for (unsigned int j = 0; j < aovSchema[i].numChannels(); ++j) {
            mAovIdxToBufIdx[idx++] = i;
        }
    }

    if (flags & USE_ADAPTIVE_SAMPLING) {
        mUseAdaptiveSampling = true;

        if (!mRenderBufOdd) {
            mRenderBufOdd = new scene_rdl2::fb_util::RenderBuffer;
        }
        mRenderBufOdd->init(alignedW, alignedH);

        mAdaptiveRenderTilesTable.reset(new AdaptiveRenderTilesTable(w, h, maxSamplesPerPixel));

#       ifdef DEBUG_SAMPLE_REC_MODE
        // Useful debug code for DebugSamplesRecArray activation.
        // mDebugSamplesRecArray.reset(new DebugSamplesRecArray(w, h)); // for debug : save mode
        mDebugSamplesRecArray.reset(new DebugSamplesRecArray("./rec0.samples")); // for debug : load mode
        if (mDebugSamplesRecArray) {
            std::cerr << ">> Film.cc DebugSamplesRecArray construction "
                      << mDebugSamplesRecArray->show("") << std::endl;
        }
#       endif // end DEBUG_SAMPLE_REC_MODE

        const bool vectorized = (flags & VECTORIZED_CPU) || (flags & VECTORIZED_XPU);
        initAdaptiveRegions(viewport, targetAdaptiveError, vectorized);
    } else {
        mUseAdaptiveSampling = false;

        if (flags & RESUMABLE_OUTPUT) {
            if (!mRenderBufOdd) {
                mRenderBufOdd = new scene_rdl2::fb_util::RenderBuffer;
            }
            mRenderBufOdd->init(alignedW, alignedH);
        } else {
            delete mRenderBufOdd;
            mRenderBufOdd = nullptr;
        }

        mAdaptiveRenderTilesTable.reset();
    }

    if (flags & ALLOC_DEEP_BUFFER) {
        if (!mDeepBuf) {
            mDeepBuf = new pbr::DeepBuffer;
        }
        mDeepBuf->initDeep(w, h,
                           static_cast<pbr::DeepFormat>(deepFormat),
                           deepCurvatureTolerance,
                           deepZTolerance,
                           deepVolCompressionRes,
                           numRenderThreads,
                           aovSchema,
                           deepIDChannelNames,
                           deepMaxLayers);
    } else {
        delete mDeepBuf;
        mDeepBuf = nullptr;
    }

    if (flags & ALLOC_CRYPTOMATTE_BUFFER) {
        if (!mCryptomatteBuf) {
            mCryptomatteBuf = new pbr::CryptomatteBuffer;
        }
        mCryptomatteBuf->init(w, h, deepIDChannelNames.size(), multiPresenceOn);
    } else {
        delete mCryptomatteBuf;
        mCryptomatteBuf = nullptr;
    }

    // setup DisplayFilters
    mDisplayFilterBufs.clear();
    mDisplayFilterBufs.resize(numDisplayFilters);

    for (int i = 0; i < numDisplayFilters; i++) {
        // TODO: allow for more pixel buffer types
        mDisplayFilterBufs[i].init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3, alignedW, alignedH);
    }

    delete mPixelInfoBuf;

    if (flags & ALLOC_PIXEL_INFO_BUFFER) {
        mPixelInfoBuf = new scene_rdl2::fb_util::PixelInfoBuffer;
        mPixelInfoBuf->init(alignedW, alignedH);
    }

    if (flags & ALLOC_HEAT_MAP_BUFFER) {
        if (!mHeatMapBuf) {
            mHeatMapBuf = new scene_rdl2::fb_util::HeatMapBuffer;
        }
        mHeatMapBuf->init(alignedW, alignedH);
    } else {
        delete mHeatMapBuf;
        mHeatMapBuf = nullptr;
    }

    clearAllBuffers();
}

void
Film::cleanUp()
{
    mWeightBuf.cleanUp();
    mRenderBuf.cleanUp();
    for (auto &buf: mAovBuf) buf.cleanUp();

    delete mPixelInfoBuf;
    mPixelInfoBuf = nullptr;

    delete mHeatMapBuf;
    mHeatMapBuf = nullptr;

    delete mRenderBufOdd;
    mRenderBufOdd = nullptr;

    mFilmActivity = 0;
    mPixelInfoBufActivity = 0;
}

void
Film::clearAllBuffers()
{
    mRenderBuf.clear();
    mWeightBuf.clear();

    if (mRenderBufOdd) {
        mRenderBufOdd->clear();
    }

    if (mAdaptiveRenderTilesTable) {
        mAdaptiveRenderTilesTable->reset();
    }

    // set aov buffers
    for (size_t b = 0; b < mAovBuf.size(); ++b) {
        scene_rdl2::fb_util::VariablePixelBuffer &buf = mAovBuf[b];
        buf.clear(mAovEntries[b].defaultValue());
    }

    if (mDeepBuf) {
        mDeepBuf->clear();
    }

    if (mCryptomatteBuf) {
        mCryptomatteBuf->clear();
    }

    if (mPixelInfoBuf) {
        mPixelInfoBuf->clear(scene_rdl2::fb_util::PixelInfo(FLT_MAX));
    }

    if (mHeatMapBuf) {
        mHeatMapBuf->clear();
    }

    mFilmActivity = 0;
    mPixelInfoBufActivity = 0;
}

void
Film::initAdaptiveRegions(const scene_rdl2::math::Viewport &viewport, float targetAdaptiveError, bool vectorized)
{
    // The viewport is a closed interval in [min, max]. The bounding boxes are closed-open.
    mAdaptiveRegions.init(
        scene_rdl2::math::BBox2i(
            scene_rdl2::math::Vec2i(viewport.mMinX, viewport.mMinY),
            scene_rdl2::math::Vec2i(viewport.mMaxX + 1, viewport.mMaxY + 1)),
        targetAdaptiveError, vectorized);
}

void
Film::addSamplesToAovBuffer(unsigned px, unsigned py, float depth,
        const float *accAovs)
{
    mTiler.linearToTiledCoords(px, py, &px, &py);
    addAovSamplesToBuffer(mAovBuf, mAovEntries, px, py, depth, accAovs);

    updateFilmActivity();
}

void
Film::addAovSamplesToBuffer(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuf,
                            const std::vector<pbr::AovSchema::Entry> &aovEntries,
                            unsigned px, unsigned py, const float depth, const float *aovs)
{
    for (size_t b = 0; b < aovBuf.size(); ++b) {
        scene_rdl2::fb_util::VariablePixelBuffer &buf = aovBuf[b];
        const pbr::AovFilter &filter = aovEntries[b].filter();
        const size_t numFloats = aovEntries[b].numChannels();

        MNRY_ASSERT(filter != pbr::AOV_FILTER_CLOSEST ||
                   buf.getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4);

        switch (buf.getFormat()) {
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
            {
                float *val = &buf.getFloatBuffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 1);
                atomicMathFilter(val, aovs, numFloats, depth, filter);
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
            {
                scene_rdl2::math::Vec2f &val = buf.getFloat2Buffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 2);
                atomicMathFilter(&val.x, aovs, numFloats, depth, filter);
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
            {
                scene_rdl2::math::Vec3f &val = buf.getFloat3Buffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 3);
                atomicMathFilter(&val.x, aovs, numFloats, depth, filter);
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4:
            {
                scene_rdl2::math::Vec4f &val = buf.getFloat4Buffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 1 || numFloats == 2 || numFloats == 3);
                atomicMathFilter(&val.x, aovs, numFloats, depth, filter);
            }
            break;
        default:
            MNRY_ASSERT(0 && "unexpected aov buffer format");
        }

        // onto the next aov
        aovs += numFloats;
    }
}

// Atomic version of addAovSamplesToBuffer, suitable for vector execution.
// The one major difference is that depths has a value for each aov entry.  Unlike
// the scalar case where we can assume that any given camera was only used once
// to create the aovs array.
void
Film::addAovSamplesToBufferSafe(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuf,
                                const std::vector<pbr::AovSchema::Entry> &aovEntries,
                                unsigned px, unsigned py, const float *depths, const float *aovs)
{
    for (size_t b = 0; b < aovBuf.size(); ++b) {
        scene_rdl2::fb_util::VariablePixelBuffer &buf = aovBuf[b];
        const pbr::AovFilter &filter = aovEntries[b].filter();
        const size_t numFloats = aovEntries[b].numChannels();
        const float depth = depths ? depths[b] : scene_rdl2::math::inf;

        MNRY_ASSERT(filter != pbr::AOV_FILTER_CLOSEST ||
                   buf.getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4);

        switch (buf.getFormat()) {
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
            {
                float *val = &buf.getFloatBuffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 1);
                atomicMathFilter(val, aovs, numFloats, depth, filter);
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
            {
                scene_rdl2::math::Vec2f &val = buf.getFloat2Buffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 2);
                atomicMathFilter(&val.x, aovs, numFloats, depth, filter);
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
            {
                scene_rdl2::math::Vec3f &val = buf.getFloat3Buffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 3);
                atomicMathFilter(&val.x, aovs, numFloats, depth, filter);
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4:
            {
                scene_rdl2::math::Vec4f &val = buf.getFloat4Buffer().getPixel(px, py);
                MNRY_ASSERT(numFloats == 1 || numFloats == 2 || numFloats == 3);
                atomicMathFilter(&val.x, aovs, numFloats, depth, filter);
            }
            break;
        default:
            MNRY_ASSERT(0 && "unexpected aov buffer format");
        }

        // onto the next aov
        aovs += numFloats;
    }
}

void
Film::addTileSamplesToDisplayFilterBuffer(unsigned bufferIdx,
                                          unsigned startX, unsigned startY,
                                          unsigned length,
                                          const uint8_t *values)
{
    mTiler.linearToTiledCoords(startX, startY, &startX, &startY);
    addTileSamplesToDisplayFilterBuffer(mDisplayFilterBufs[bufferIdx], startX, startY, length, values);

    updateFilmActivity();
}

void
Film::addTileSamplesToDisplayFilterBuffer(scene_rdl2::fb_util::VariablePixelBuffer &buf,
                                          unsigned startX, unsigned startY,
                                          unsigned length,
                                          const uint8_t *values)
{
    MNRY_ASSERT(startX < buf.getWidth() && startY < buf.getHeight());

    size_t pixelSize = buf.getSizeOfPixel();
    uint8_t * data = buf.getData() + (startY * buf.getWidth() + startX) * pixelSize;
    memcpy(data, values, length * pixelSize);
}

// Corresponds to Aov.cc::aovSetBeautyAndAlpha()
inline void
Film::addBeautyAndAlphaSamplesToBuffer(unsigned px, unsigned py, const scene_rdl2::fb_util::RenderColor& color)
{
    // Multi-camera beauty output in vectorized mode doesn't work because
    // the color comes from the Film's render buffer which is not multi-camera aware.
    // (See addSampleBundleHandler()).
    // It works in scalar mode because Aov.cc::aovSetBeautyAndAlpha() takes the
    // radiance directly from the integrator to populate the AOVs, bypassing the
    // Film's render buffer.
    for (size_t b = 0; b < mAovBeautyBufIdx.size(); ++b) {
        scene_rdl2::fb_util::VariablePixelBuffer &buf = mAovBuf[mAovBeautyBufIdx[b]];
        MNRY_ASSERT(buf.getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3);
        scene_rdl2::math::Vec3f &val = buf.getFloat3Buffer().getPixel(px, py);
        util::atomicAssignFloat(&val.x, color[0]);
        util::atomicAssignFloat(&val.y, color[1]);
        util::atomicAssignFloat(&val.z, color[2]);
    }
    for (size_t b = 0; b < mAovAlphaBufIdx.size(); ++b) {
        scene_rdl2::fb_util::VariablePixelBuffer &buf = mAovBuf[mAovAlphaBufIdx[b]];
        MNRY_ASSERT(buf.getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::FLOAT);
        float &val = buf.getFloatBuffer().getPixel(px, py);
        util::atomicAssignFloat(&val, color[3]);
    }

    updateFilmActivity();
}

void
Film::addSamplesToRenderBuffer(unsigned px, unsigned py,
                               const scene_rdl2::fb_util::RenderColor &accSamples,
                               float numSamples,
                               const scene_rdl2::fb_util::RenderColor *accSamplesOdd)
{
    mTiler.linearToTiledCoords(px, py, &px, &py);

    auto& renderColor = mRenderBuf.getPixel(px, py);
    util::atomicAdd(&renderColor.x, accSamples.x);
    util::atomicAdd(&renderColor.y, accSamples.y);
    util::atomicAdd(&renderColor.z, accSamples.z);
    util::atomicAdd(&renderColor.w, accSamples.w);

    auto& weight = mWeightBuf.getPixel(px, py);
    util::atomicAdd(&weight, numSamples);

    if (mRenderBufOdd) {
        MNRY_ASSERT(accSamplesOdd);
        auto& renderColorOdd = mRenderBufOdd->getPixel(px, py);
        util::atomicAdd(&renderColorOdd.x, accSamplesOdd->x);
        util::atomicAdd(&renderColorOdd.y, accSamplesOdd->y);
        util::atomicAdd(&renderColorOdd.z, accSamplesOdd->z);
        util::atomicAdd(&renderColorOdd.w, accSamplesOdd->w);
    }

    updateFilmActivity();
}

void
Film::normalizeRenderBuffer(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                            scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer, bool parallel) const
{
    const unsigned w = std::min(dstRenderBuffer->getWidth(), srcRenderBuffer->getWidth());
    const unsigned h = std::min(dstRenderBuffer->getHeight(), srcRenderBuffer->getHeight());

    simpleLoop(parallel, 0u, h, [&](unsigned y) {

            // The use of __restrict here tells the compiler that these arrays
            // don't alias each so it's free to generate code with that assumption.
            scene_rdl2::fb_util::RenderColor *__restrict dstRow = dstRenderBuffer->getRow(y);
            const scene_rdl2::fb_util::RenderColor *__restrict srcColor = srcRenderBuffer->getRow(y);
            const float *__restrict srcWeight = mWeightBuf.getRow(y);

            for (unsigned x = 0; x < w; ++x) {

                if (*srcWeight > 0.f) {
                    *dstRow = *srcColor * (1.f / *srcWeight);
                } else {
                    *dstRow = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::zero);
                }

                ++dstRow;
                ++srcColor;
                ++srcWeight;
            }
        });
}

void
Film::extrapolateRenderBufferFastPath(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                                      scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer, bool parallel) const
{
    // All buffer dimensions are assumed to be multiples to tile boundaries.
    MNRY_ASSERT(dstRenderBuffer->getWidth() == srcRenderBuffer->getWidth());
    MNRY_ASSERT(dstRenderBuffer->getHeight() == srcRenderBuffer->getHeight());
    MNRY_ASSERT((dstRenderBuffer->getWidth() & 7) == 0);
    MNRY_ASSERT((dstRenderBuffer->getHeight() & 7) == 0);

    const unsigned numTiles = mTiler.mNumTiles;
    //
    // Version 2 of extrapolation code.
    //
    simpleLoop(parallel, 0u, numTiles, [&](unsigned i) {

            scene_rdl2::fb_util::RenderColor *__restrict dst = dstRenderBuffer->getData() + (i << 6);
            const scene_rdl2::fb_util::RenderColor *__restrict srcColor = srcRenderBuffer->getData() + (i << 6);
            const float *__restrict srcWeight = mWeightBuf.getData() + (i << 6);

            extrapolatePartialTileV2(dst, srcColor, srcWeight, 0, 8, 0, 8, mTileExtrapolation);
        });
}

void
Film::extrapolateRenderBufferWithViewport(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                                          scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer,
                                          const scene_rdl2::math::Viewport &viewport,
                                          bool parallel) const
{
    // All buffer dimensions are assumed to be multiples to tile boundaries.
    MNRY_ASSERT(dstRenderBuffer);
    MNRY_ASSERT(dstRenderBuffer->getWidth() == srcRenderBuffer->getWidth());
    MNRY_ASSERT(dstRenderBuffer->getHeight() == srcRenderBuffer->getHeight());
    MNRY_ASSERT((dstRenderBuffer->getWidth() & 7) == 0);
    MNRY_ASSERT((dstRenderBuffer->getHeight() & 7) == 0);

    const unsigned numTilesX = ((viewport.mMaxX) >> 3) - ((viewport.mMinX) >> 3) + 1;
    const unsigned numTilesY = ((viewport.mMaxY) >> 3) - ((viewport.mMinY) >> 3) + 1;
    const unsigned baseTileX = viewport.mMinX & ~0x07;
    const unsigned baseTileY = viewport.mMinY & ~0x07;

    //
    // Version 2 of extrapolation code.
    //

    simpleLoop(parallel, 0u, numTilesY, [&](unsigned tileY) {

        unsigned baseY = baseTileY + (tileY << 3);
        unsigned minY = std::max<unsigned>(baseY, viewport.mMinY) - baseY;
        unsigned maxY = std::min<unsigned>(baseY + 8, viewport.mMaxY + 1) - baseY;
        MNRY_ASSERT(minY < maxY && minY < 8 && maxY <= 8);

        for (unsigned tileX = 0; tileX < numTilesX; ++tileX) {

            unsigned baseX = baseTileX + (tileX << 3);
            unsigned minX = std::max<unsigned>(baseX, viewport.mMinX) - baseX;
            unsigned maxX = std::min<unsigned>(baseX + 8, viewport.mMaxX + 1) - baseX;
            MNRY_ASSERT(minX < maxX && minX < 8 && maxX <= 8);

            unsigned tileOfs = mTiler.linearCoordsToCoarseTileOffset(baseX, baseY);
            MNRY_ASSERT((tileOfs & 63) == 0);

            scene_rdl2::fb_util::RenderColor *__restrict dst = dstRenderBuffer->getData() + tileOfs;
            const scene_rdl2::fb_util::RenderColor *__restrict srcColor = srcRenderBuffer->getData() + tileOfs;
            const float *__restrict srcWeight = mWeightBuf.getData() + tileOfs;

            extrapolatePartialTileV2(dst, srcColor, srcWeight, minX, maxX, minY, maxY, mTileExtrapolation);
        }
    });
}

void
Film::extrapolateRenderBufferWithTileList(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                                          scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer,
                                          const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                          bool parallel) const
{
    // All buffer dimensions are assumed to be multiples to tile boundaries.
    MNRY_ASSERT(dstRenderBuffer);
    MNRY_ASSERT(dstRenderBuffer->getWidth() == srcRenderBuffer->getWidth());
    MNRY_ASSERT(dstRenderBuffer->getHeight() == srcRenderBuffer->getHeight());
    MNRY_ASSERT((dstRenderBuffer->getWidth() & 7) == 0);
    MNRY_ASSERT((dstRenderBuffer->getHeight() & 7) == 0);

    const unsigned numTiles = unsigned(tiles.size());
    //
    // Version 2 of extrapolation code.
    //

    simpleLoop(parallel, 0u, numTiles, [&](unsigned i) {

        const scene_rdl2::fb_util::Tile &tile = tiles[i];
        const unsigned baseX = (tile.mMinX & ~0x07);
        const unsigned baseY = (tile.mMinY & ~0x07);
        const unsigned minX = tile.mMinX - baseX;
        const unsigned maxX = tile.mMaxX - baseX;
        const unsigned minY = tile.mMinY - baseY;
        const unsigned maxY = tile.mMaxY - baseY;
        MNRY_ASSERT(minX < maxX && minX < 8 && maxX <= 8);
        MNRY_ASSERT(minY < maxY && minY < 8 && maxY <= 8);

        const unsigned tileOfs = mTiler.linearCoordsToCoarseTileOffset(tile.mMinX, tile.mMinY);
        MNRY_ASSERT((tileOfs & 63) == 0);

        scene_rdl2::fb_util::RenderColor *__restrict dst = dstRenderBuffer->getData() + tileOfs;
        const scene_rdl2::fb_util::RenderColor *__restrict srcColor = srcRenderBuffer->getData() + tileOfs;
        const float *__restrict srcWeight = mWeightBuf.getData() + tileOfs;

        extrapolatePartialTileV2(dst, srcColor, srcWeight, minX, maxX, minY, maxY, mTileExtrapolation);
    });
}

// static function
void
Film::constructPixelFillOrderTable(const unsigned nodeId, const unsigned nodeTotal)
{
    // setup initial scrambled pixId table
    std::vector<uint8_t> data;
    data.resize(64);
    for (unsigned i = 0; i < 64; ++i) {
        data[i] = gPixelFillOrder[i];
    }

    // construct scrambled pixelId table for tileId = 0 ~ 1023
    if (nodeTotal == 1) {
        // Single node is using regular gPixelFillOrder which is used by moonray/moonray_gui.
        for (unsigned i = 0; i < 1024; ++i) {
            copyPixIdTable(data, gScrambledPixelFillOrder[i]);
        }

    } else {
        // Multiple nodes use randomly shuffled pixelId table which support
        // TileId from 0 to 1023. Random seed is different every node and every run.
        // Multi nodes case, all extrapolation is done by client side by scene_rdl2::fb_util::TileExtrapolation
        // which supports random pixelId order.
        std::random_device seed_gen;
        scene_rdl2::util::Random rnd(seed_gen());
        for (unsigned i = 0; i < 1024; ++i) {
            std::shuffle(data.begin(), data.end(), rnd);
            copyPixIdTable(data, gScrambledPixelFillOrder[i]);
        }
    }
}

template <bool adaptive>
void
Film::addSampleBundleHandlerHelper(mcrt_common::ThreadLocalState *tls,
                                   unsigned numEntries,
                                   pbr::BundledRadiance **entries,
                                   Film &film)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_ADD_SAMPLE_HANDLER);

    MNRY_ASSERT(areBundledEntriesValid(tls, numEntries, entries));

    pbr::TLState *pbrTls = MNRY_VERIFY(tls->mPbrTls.get());
    scene_rdl2::util::BitArray &tilesRenderedTo = pbrTls->mTilesRenderedTo;

    unsigned entryIdx = 0;
    unsigned entriesRemaining = numEntries;

    do {
        scene_rdl2::fb_util::RenderColor accSamples     = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::zero);
        scene_rdl2::fb_util::RenderColor accSamplesHalf = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::zero);
        float accWeight = 0.f;
        unsigned numLocalSamples = 0;

        uint32_t currPixel = entries[entryIdx]->mPixel;
        unsigned px, py;
        pbr::uint32ToPixelLocation(currPixel, &px, &py);

        unsigned tile = pbr::getTile(entries[entryIdx]->mTilePass);
        tilesRenderedTo.setBit(tile);

        // Loop over all entries for this pixel (we've pre-sorted entries by
        // pixel at this stage).
        do {
            const pbr::BundledRadiance *br = entries[entryIdx];

            MNRY_ASSERT(br->mPixel == currPixel);

            accSamples += br->mRadiance;
            accWeight += br->mPathPixelWeight;
            if /*constexpr*/ (adaptive) {
                if (br->mSubPixelIndex & 1) {
                    accSamplesHalf += br->mRadiance;
                }
            }

            ++numLocalSamples;
            ++entryIdx;

            if (br->mDeepDataHandle != pbr::nullHandle) {
                pbr::DeepData *deepData = static_cast<pbr::DeepData*>(pbrTls->getListItem(br->mDeepDataHandle, 0));
                if (deepData->mHitDeep) {
                    constexpr int channels[3] = { 0, 1, 2 };
                    const float vals[3] = { br->mRadiance[0], br->mRadiance[1], br->mRadiance[2] };
                    film.mDeepBuf->addSample(pbrTls, px, py,
                                             deepData->mSubpixelX, deepData->mSubpixelY, deepData->mLayer,
                                             deepData->mDeepIDs, deepData->mDeepT, deepData->mRayZ,
                                             deepData->mDeepNormal, br->mRadiance[3],
                                             channels, 3, vals,
                                             1.f, br->mPathPixelWeight);
                }
                pbrTls->releaseDeepData(br->mDeepDataHandle);
            }

            if (br->mCryptomatteDataHandle != pbr::nullHandle) {
                pbr::CryptomatteData *cryptomatteData =
                            static_cast<pbr::CryptomatteData*>(pbrTls->getListItem(br->mCryptomatteDataHandle, 0));

                if (film.mCryptomatteBuf != nullptr) {
                    float id = cryptomatteData->mId;
                    unsigned int depth = cryptomatteData->mPresenceDepth;
                    scene_rdl2::math::Vec3f position = cryptomatteData->mPosition;
                    scene_rdl2::math::Vec3f normal = cryptomatteData->mNormal;
                    scene_rdl2::math::Color4 beauty(br->mRadiance[0], 
                                                    br->mRadiance[1], 
                                                    br->mRadiance[2], 
                                                    br->mRadiance[3]);
                    scene_rdl2::math::Vec3f refP = br->mCryptoRefP;
                    scene_rdl2::math::Vec3f refN = br->mCryptoRefN;
                    scene_rdl2::math::Vec2f uv = br->mCryptoUV;

                    float presenceInv = cryptomatteData->mPathPixelWeight == 0.f ? 0.f 
                                                                             : (1.f / cryptomatteData->mPathPixelWeight);
                    beauty *= presenceInv; // multiply by the reciprocal of presence to remove baked-in presence

                    if (cryptomatteData->mHit && cryptomatteData->mPresenceDepth < 0) {
                        // non-presence path
                        if (cryptomatteData->mIsFirstSample) {
                            // we only want to increment coverage, position, normal, and the normalization factor
                            // numFragSamples if we're dealing with the first sample for this path. We don't want 
                            // any data from the subsequent bounces except for the beauty (for GI)
                            film.mCryptomatteBuf->addSampleVector(px, py, id, 1.f, position, normal, beauty,
                                                                  refP, refN, uv, depth);                    
                            cryptomatteData->mIsFirstSample = 0;
                        } else {
                            film.mCryptomatteBuf->addBeautySampleVector(px, py, id, beauty, depth);
                        }
                    } else if (cryptomatteData->mPresenceDepth >= 0 && cryptomatteData->mPathPixelWeight > 0.01f) {
                        // We divide by pathPixelWeight to compute Cryptomatte beauty.  This can cause fireflies if
                        // the value is small, so we clamp at 0.01.
                        beauty.a = 0.f;
                        // presence path: only add beauty -- the rest of the data is populated in the shadeBundleHandler 
                        film.mCryptomatteBuf->addBeautySampleVector(px, py, id, beauty, depth);
                    }
                }
                pbrTls->releaseCryptomatteData(br->mCryptomatteDataHandle);
            }

        } while (entryIdx != numEntries && currPixel == entries[entryIdx]->mPixel);

        MNRY_ASSERT(numLocalSamples);

        // Update render buffer - only execute the atomic adds if we absolutely
        // need to.
        film.mTiler.linearToTiledCoords(px, py, &px, &py);

        scene_rdl2::fb_util::RenderColor* const dstColor = &film.mRenderBuf.getPixel(px, py);
        if (accSamples.x != 0.f) {
            util::atomicAdd(&dstColor->x, accSamples.x);
        }
        if (accSamples.y != 0.f) {
            util::atomicAdd(&dstColor->y, accSamples.y);
        }
        if (accSamples.z != 0.f) {
            util::atomicAdd(&dstColor->z, accSamples.z);
        }
        if (accSamples.w != 0.f) {
            util::atomicAdd(&dstColor->w, accSamples.w);
        }
        if (accWeight != 0.f) {
            util::atomicAdd(&film.mWeightBuf.getPixel(px, py), accWeight);
        }

        if /*constexpr*/ (adaptive) {
            scene_rdl2::fb_util::RenderColor* const dstColorHalf = &film.mRenderBufOdd->getPixel(px, py);
            if (accSamplesHalf.x != 0.f) {
                util::atomicAdd(&dstColorHalf->x, accSamplesHalf.x);
            }
            if (accSamplesHalf.y != 0.f) {
                util::atomicAdd(&dstColorHalf->y, accSamplesHalf.y);
            }
            if (accSamplesHalf.z != 0.f) {
                util::atomicAdd(&dstColorHalf->z, accSamplesHalf.z);
            }
            if (accSamplesHalf.w != 0.f) {
                util::atomicAdd(&dstColorHalf->w, accSamplesHalf.w);
            }
        }
        film.addBeautyAndAlphaSamplesToBuffer(px, py, *dstColor);

        MNRY_ASSERT(entriesRemaining >= numLocalSamples);
        entriesRemaining -= numLocalSamples;

    } while (entriesRemaining);

    film.updateFilmActivity();
}

void
Film::addSampleBundleHandler(mcrt_common::ThreadLocalState *tls,
                             unsigned numEntries,
                             pbr::BundledRadiance **entries,
                             void *userData)
{
    Film* const film = static_cast<Film*>(userData);
    if (film->mRenderBufOdd) {
        addSampleBundleHandlerHelper<true>(tls, numEntries, entries, *film);
    } else {
        addSampleBundleHandlerHelper<false>(tls, numEntries, entries, *film);
    }
}

void
Film::addAovSampleBundleHandler(mcrt_common::ThreadLocalState *tls,
                                unsigned numEntries,
                                pbr::BundledAov **entries,
                                void *userData)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_AOVS);

    MNRY_ASSERT(areBundledAovEntriesValid(tls, numEntries, entries));

    Film *film = (Film *)userData;
    pbr::TLState *pbrTls = MNRY_VERIFY(tls->mPbrTls.get());

    unsigned entryIdx = 0;
    unsigned entriesRemaining = numEntries;

    MNRY_ASSERT(!film->mAovBuf.empty());

    SCOPED_MEM(&tls->mArena);
    float *accAovs = tls->mArena.allocArray<float>(film->mAovBufNumFloats);

    do {
        memset(accAovs, 0, sizeof(float) * film->mAovBufNumFloats);
        unsigned numLocalSamples = 0;

        uint32_t currPixel = entries[entryIdx]->mPixel;
        unsigned px, py;
        pbr::uint32ToPixelLocation(currPixel, &px, &py);

        // Loop over all entries for this pixel (we've pre-sorted entries by
        // pixel at this stage).
        do {
            const pbr::BundledAov *ba = entries[entryIdx];

            MNRY_ASSERT(ba->mPixel == currPixel);

            for (unsigned aov = 0; aov < pbr::BundledAov::MAX_AOV; ++aov) {
                const uint32_t aovIdx = ba->aovIdx(aov);
                if (aovIdx <= pbr::BundledAov::MAX_AOV_IDX) {
                    accAovs[aovIdx] += ba->mAovs[aov];

                    if (ba->mDeepDataHandle != pbr::nullHandle) {
                        pbr::DeepData *deepData = static_cast<pbr::DeepData*>(pbrTls->getListItem(ba->mDeepDataHandle, 0));
                        if (deepData->mHitDeep) {
                            int channels[1] = { (int)aovIdx + 3 };
                            float vals[1] = { ba->mAovs[aov] };
                            film->mDeepBuf->addSample(pbrTls, px, py,
                                                      deepData->mSubpixelX, deepData->mSubpixelY, deepData->mLayer,
                                                      deepData->mDeepIDs, deepData->mDeepT, deepData->mRayZ,
                                                      deepData->mDeepNormal, 0.f,
                                                      channels, 1, vals,
                                                      1.f, 0.f);
                        }
                    }
                }
            }

            pbrTls->releaseDeepData(ba->mDeepDataHandle);

            ++numLocalSamples;
            ++entryIdx;

        } while (entryIdx != numEntries && currPixel == entries[entryIdx]->mPixel);

        MNRY_ASSERT(numLocalSamples);

        // Update aov buffer - only execute the atomic adds if we absolutely
        // need to.
        film->mTiler.linearToTiledCoords(px, py, &px, &py);

        const float *f = accAovs;
        for (auto &buf: film->mAovBuf) {
            switch (buf.getFormat()) {
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
                if (*f != 0.f) {
                    util::atomicAdd(&buf.getFloatBuffer().getPixel(px, py), *f);
                }
                ++f;
                break;
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
                {
                    scene_rdl2::math::Vec2f &val = buf.getFloat2Buffer().getPixel(px, py);
                    if (*f != 0.f) {
                        util::atomicAdd(&val.x, *f);
                    }
                    ++f;
                    if (*f != 0.f) {
                        util::atomicAdd(&val.y, *f);
                    }
                    ++f;
                }
                break;
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
                {
                    scene_rdl2::math::Vec3f &val = buf.getFloat3Buffer().getPixel(px, py);
                    if (*f != 0.f) {
                        util::atomicAdd(&val.x, *f);
                    }
                    ++f;
                    if (*f != 0.f) {
                        util::atomicAdd(&val.y, *f);
                    }
                    ++f;
                    if (*f != 0.f) {
                        util::atomicAdd(&val.z, *f);
                    }
                    ++f;
                }
                break;
            default:
                MNRY_ASSERT(0 && "unexpected aov buffer format");
            }
        }

        MNRY_ASSERT(entriesRemaining >= numLocalSamples);
        entriesRemaining -= numLocalSamples;

    } while (entriesRemaining);

    film->updateFilmActivity();
}

void
Film::addFilteredAovSampleBundleHandler(mcrt_common::ThreadLocalState *tls,
                                        unsigned numEntries,
                                        pbr::BundledAov **entries,
                                        void *userData)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_AOVS);

    MNRY_ASSERT(areBundledAovEntriesValid(tls, numEntries, entries));

    Film *film = (Film *)userData;
    pbr::TLState *pbrTls = MNRY_VERIFY(tls->mPbrTls.get());

    unsigned entryIdx = 0;
    unsigned entriesRemaining = numEntries;

    MNRY_ASSERT(!film->mAovBuf.empty());

    SCOPED_MEM(&tls->mArena);
    float *localAovs = tls->mArena.allocArray<float>(film->mAovBufNumFloats);
    // we only need this if we have closest filters
    float *localDepths = film->mAovHasClosestFilter ?
        tls->mArena.allocArray<float>(film->mAovEntries.size()) : nullptr;

    do {
        // Initialize localAovs with nan to indicate that the value for that
        // aov has not yet been set.
        std::fill(localAovs, localAovs + film->mAovBufNumFloats, scene_rdl2::math::nan);
        // Initialize localDepths to +inf
        if (localDepths) std::fill(localDepths, localDepths + film->mAovEntries.size(), scene_rdl2::math::inf);
        unsigned numLocalSamples = 0;
        uint32_t currPixel = entries[entryIdx]->mPixel;
        unsigned px, py;
        pbr::uint32ToPixelLocation(currPixel, &px, &py);

        // Loop over all entries for this pixel (we've pre-sorted entries by
        // pixel at this stage).
        do {
            const pbr::BundledAov *ba = entries[entryIdx];

            MNRY_ASSERT(ba->mPixel == currPixel);

            for (unsigned aov = 0; aov < pbr::BundledAov::MAX_AOV; ++aov) {
                const uint32_t aovIdx = ba->aovIdx(aov);
                if (aovIdx <= pbr::BundledAov::MAX_AOV_IDX) { // is the slot in use?
                    const uint32_t bufIdx = film->mAovIdxToBufIdx[aovIdx];
                    const pbr::AovSchema::Entry &entry = film->mAovEntries[bufIdx];

                    // closest filtering guarantees the entry is not split across BundledAov objects
                    if (entry.filter() == pbr::AOV_FILTER_CLOSEST) {
                        const unsigned int numEntryFloats = entry.numChannels();
                        MNRY_ASSERT(aov + numEntryFloats + 1 < pbr::BundledAov::MAX_AOV);
                        for (unsigned int i = 0; i < numEntryFloats; ++i) {
                            MNRY_ASSERT(ba->aovIdx(aov + i) == aovIdx + i);
                        }
                        MNRY_ASSERT(ba->aovIdx(aov + numEntryFloats) == pbr::BundledAov::AOV_ALL_BITS);

                        MNRY_ASSERT(localDepths);
                        const float depth = ba->mAovs[aov + numEntryFloats];
                        if (depth < localDepths[bufIdx]) {
                            // update the local value
                            for (unsigned int i = 0; i < numEntryFloats; ++i) {
                                localAovs[aovIdx + i] = ba->mAovs[aov + i];
                            }
                            localDepths[bufIdx] = depth;
                        }
                        aov += numEntryFloats;
                        continue;
                    }

                    if (scene_rdl2::math::isfinite(localAovs[aovIdx])) {
                        // there is already a value for this aov so we must apply
                        // the math filter here
                        switch (entry.filter()) {
                        case pbr::AOV_FILTER_AVG:
                        case pbr::AOV_FILTER_SUM:
                            localAovs[aovIdx] += ba->mAovs[aov];
                            break;
                        case pbr::AOV_FILTER_MIN:
                            if (localAovs[aovIdx] > ba->mAovs[aov]) {
                                localAovs[aovIdx] = ba->mAovs[aov];
                            }
                            break;
                        case pbr::AOV_FILTER_MAX:
                            if (localAovs[aovIdx] < ba->mAovs[aov]) {
                                localAovs[aovIdx] = ba->mAovs[aov];
                            }
                            break;
                        case pbr::AOV_FILTER_CLOSEST:
                            // Intentional fallthrough.  This case already
                            // handled above.
                        default:
                            MNRY_ASSERT(0 && "unexpected aov math filter");
                        }
                    } else {
                        // there is no value for this aov, so we must initialize
                        // the value here
                        MNRY_ASSERT(entry.filter() != pbr::AOV_FILTER_CLOSEST);
                        localAovs[aovIdx] = ba->mAovs[aov];
                    }
                    if (ba->mDeepDataHandle != pbr::nullHandle) {
                        pbr::DeepData *deepData = static_cast<pbr::DeepData*>(pbrTls->getListItem(ba->mDeepDataHandle, 0));
                        if (deepData->mHitDeep) {
                            int channels[1] = { (int)aovIdx + 3 };
                            float vals[1] = { ba->mAovs[aov] };
                            film->mDeepBuf->addSample(pbrTls, px, py,
                                                      deepData->mSubpixelX, deepData->mSubpixelY, deepData->mLayer,
                                                      deepData->mDeepIDs, deepData->mDeepT, deepData->mRayZ,
                                                      deepData->mDeepNormal, 0.f,
                                                      channels, 1, vals,
                                                      1.f, 0.f);
                        }
                    }
                }
            }

            pbrTls->releaseDeepData(ba->mDeepDataHandle);

            ++numLocalSamples;
            ++entryIdx;

        } while (entryIdx != numEntries && currPixel == entries[entryIdx]->mPixel);

        MNRY_ASSERT(numLocalSamples);

        // Update aov buffer
        film->mTiler.linearToTiledCoords(px, py, &px, &py);
        film->addAovSamplesToBufferSafe(film->mAovBuf, film->mAovEntries, px, py, localDepths, localAovs);

        MNRY_ASSERT(entriesRemaining >= numLocalSamples);
        entriesRemaining -= numLocalSamples;

    } while (entriesRemaining);

    film->updateFilmActivity();
}

void
Film::addHeatMapBundleHandler(mcrt_common::ThreadLocalState *tls,
                              unsigned numEntries,
                              pbr::BundledHeatMapSample **entries,
                              void *userData)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_ADD_HEAT_MAP_HANDLER);

    Film *film = (Film *)userData;

    unsigned entryIdx = 0;
    unsigned entriesRemaining = numEntries;

    MNRY_ASSERT(film->mHeatMapBuf);

    do {
        int64_t ticks = 0;
        unsigned numLocalSamples = 0;

        uint32_t currPixel = entries[entryIdx]->mPixel;
        unsigned px, py;
        pbr::uint32ToPixelLocation(currPixel, &px, &py);

        // Loop over all entries for this pixel (we've pre-sorted entries by
        // pixel at this stage).
        do {
            const pbr::BundledHeatMapSample *b = entries[entryIdx];

            MNRY_ASSERT(b->mPixel == currPixel);

            ticks += b->mTicks;

            ++numLocalSamples;
            ++entryIdx;

        } while (entryIdx != numEntries && currPixel == entries[entryIdx]->mPixel);


        MNRY_ASSERT(numLocalSamples);

        if (ticks) {
            // Update pixel info buffer - only execute the atomic adds if we absolutely
            // need to.
            film->mTiler.linearToTiledCoords(px, py, &px, &py);
            util::atomicAdd(&film->mHeatMapBuf->getPixel(px, py), ticks);
        }

        MNRY_ASSERT(entriesRemaining >= numLocalSamples);
        entriesRemaining -= numLocalSamples;

    } while (entriesRemaining);

    film->updatePixelInfoBufferActivity();
}

void
Film::updateAdaptiveError(const scene_rdl2::fb_util::Tile& tile,
                          const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                          const scene_rdl2::fb_util::RenderBuffer& renderBufOdd,
                          mcrt_common::ThreadLocalState* tls,
                          const unsigned endSampleIdx)
{
    const scene_rdl2::math::BBox2i bounds(scene_rdl2::math::Vec2i(tile.mMinX, tile.mMinY),
                                          scene_rdl2::math::Vec2i(tile.mMaxX, tile.mMaxY));
    mAdaptiveRegions.update(bounds, mTiler, renderBuf, mWeightBuf, renderBufOdd, tls, endSampleIdx);
    if (AdaptiveRenderTilesTable* const table = getAdaptiveRenderTilesTable()) {
        // We're not all that concerned if these are slightly out-of-sync.
        // It's only used for the progress bar.
        const float currentError = mAdaptiveRegions.getError();
        table->setCurrentError(currentError);
    }
}

void
Film::updateAdaptiveErrorAll(const scene_rdl2::fb_util::RenderBuffer &renderBuf,
                             const scene_rdl2::fb_util::RenderBuffer &renderBufOdd)
{
    mAdaptiveRegions.updateAll(mTiler, renderBuf, mWeightBuf, renderBufOdd);
}

void
Film::disableAdjustAdaptiveTreeUpdateTiming()
//
// Set disable codnition about adjust adaptiveTree update timing logic
//
{
    mAdaptiveRegions.disableAdjustUpdateTiming();
}

void
Film::enableAdjustAdaptiveTreeUpdateTiming(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl)
//
// Set enable condition about adjust adaptiveTree update timing logic.
// we need to set adaptiveIterationPixSampleIdTable.
//
{
    mAdaptiveRegions.enableAdjustUpdateTiming(adaptiveIterationPixSampleIdTbl);
}

bool
Film::getAdaptiveDone() const
{
    return getAdaptiveRenderTilesTable()->getMinimumDone() && mAdaptiveRegions.done();
}

float
Film::getProgressFraction(bool activeRendering, size_t *submitted, size_t *total) const
{
    MNRY_ASSERT_REQUIRE(mAdaptiveRenderTilesTable);
    return mAdaptiveRenderTilesTable->getCompletedFraction(activeRendering, submitted, total);
}

const scene_rdl2::fb_util::VariablePixelBuffer *
Film::getBeautyAovBuff() const
{
    if (mAovBeautyBufIdx.empty()) return nullptr;
    return &getAovBuffer(mAovBeautyBufIdx[0]);
}

const scene_rdl2::fb_util::VariablePixelBuffer *
Film::getAlphaAovBuff() const
{
    if (mAovAlphaBufIdx.empty()) return nullptr;
    return &getAovBuffer(mAovAlphaBufIdx[0]);
}

} // namespace rndr
} // namespace moonray

