// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// The render buffer here stores summed weighted radiances in the rgb channels
// and the accumulated alpha in the alpha channel. On lookup, we divide across
// by the corresponding weight in the weight buffer to get the final radiance
// and alpha which we display/save.
//
#pragma once
#include "DebugSamplesRecArray.h"
#include "SampleIdBuff.h"
#include "Util.h"
#include "adaptive/ActivePixelMask.h"
#include "adaptive/AdaptiveRegions.h"

#include <moonray/common/mcrt_util/Atomic.h>
#include <moonray/common/mcrt_util/MutexPool2D.h>
#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/DeepBuffer.h>
#include <moonray/rendering/pbr/core/Cryptomatte.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/fb_util/PixelBuffer.h>
#include <scene_rdl2/common/fb_util/Tiler.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/render/util/MiscUtils.h>
#include <vector>

namespace scene_rdl2 {
namespace math { class Viewport; }
namespace fb_util {
    class ActivePixels;
    class TileExtrapolation;
}
}

namespace moonray {
namespace pbr { struct BundledRadiance; }
namespace mcrt_common { class ThreadLocalState; }

namespace rndr {

extern const uint8_t gPixelFillOrder[64];
extern uint8_t gScrambledPixelFillOrder[1024][64];

 class AdaptiveRenderTilesTable;

//-------------------------------------------------------------------------------------------------------------

class CACHE_ALIGN Film
{
public:
    enum
    {
        USE_ADAPTIVE_SAMPLING       = 0x0001,
        ALLOC_PIXEL_INFO_BUFFER     = 0x0002,
        ALLOC_HEAT_MAP_BUFFER       = 0x0004,
        ALLOC_DEEP_BUFFER           = 0x0008,
        ALLOC_CRYPTOMATTE_BUFFER    = 0x0010,
        RESUMABLE_OUTPUT            = 0x0020,
        VECTORIZED_CPU              = 0x0040,
        VECTORIZED_XPU              = 0x0080,
    };

    Film();
    ~Film();

    // aovChannels is an array where each entry contains the number
    // of floats stored for that particular aov.
    void init(unsigned w, unsigned h,
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
              const unsigned maxSamplesPerPixel,
              float targetAdaptiveError,
              bool multiPresenceOn);

    void cleanUp();

    void clearAllBuffers();
    void initAdaptiveRegions(const scene_rdl2::math::Viewport &viewport, float targetAdativeError, bool vectorized);

    //
    // General const query APIs:
    //

    // Returns unaligned width and height.
    unsigned getWidth() const           { return mTiler.mOriginalW; }
    unsigned getHeight() const          { return mTiler.mOriginalH; }

    // Returns width and height aligned to tile boundaries.
    unsigned getAlignedWidth() const    { return mTiler.mAlignedW; }
    unsigned getAlignedHeight() const   { return mTiler.mAlignedH; }

    const scene_rdl2::fb_util::Tiler &getTiler() const { return mTiler; }

    // These exist to see if there has been any activity within buffers between
    // 2 points in time. Just query the values, and compare for equality.
    unsigned getFilmActivity() const            { return mFilmActivity.load(std::memory_order_acquire); }
    unsigned getPixelInfoBufferActivity() const { return mPixelInfoBufActivity.load(std::memory_order_acquire); }
    void updateFilmActivity()                   { mFilmActivity.fetch_add(1u, std::memory_order_release); }
    void updatePixelInfoBufferActivity()        { mPixelInfoBufActivity.fetch_add(1u, std::memory_order_release); }

    // This is only approximated since it's derived from the accumulated weight.
    // It should be exact for non-bundled rendering but only an approximation
    // for bundled rendering.
    unsigned getNumRenderBufferPixelSamples(unsigned px, unsigned py) const;

    // Initializes and fills the given PixelBuffer with the number of samples per
    // pixel - as if you had called getNumRenderBufferPixelSamples() for each
    // pixel in the buffer.
    void fillPixelSampleCountBuffer(scene_rdl2::fb_util::PixelBuffer<unsigned>& buf) const;

    //
    // APIs to update individual pixels in various buffers:
    //
    // Coordinate parameters are alway linear, tiled coordinates are an internal
    // implementation detail.
    //

    // accSamples is the sum of all the actual radiance values, each scaled by
    // its associated weight.
    // accSamples.a should contains the alpha channel value, it can go outside
    //              of [0, 1] if numSamples > 1.
    // accSamplesOdd need only be supplied if Film::USE_ADAPTIVE_SAMPLING was set,
    // otherwise nullptr should be passed instead. When adaptive sampling is on,
    // it should contain the accumulation of all oddly indexed color samples.
    void addSamplesToRenderBuffer(unsigned px, unsigned py,
                                  const scene_rdl2::fb_util::RenderColor &accSamples,
                                  float numSamples,
                                  const scene_rdl2::fb_util::RenderColor *accSamplesOdd);

    // This adds aovs to the Aov Buffers.
    void addSamplesToAovBuffer(unsigned px, unsigned py, float depth, const float *accAovs);

    void addTileSamplesToDisplayFilterBuffer(unsigned bufferIdx,
                                             unsigned startX, unsigned startY,
                                             unsigned length,
                                             const uint8_t *values);

private:
    // Adds the aov samples to the aov buffers
    // aovBuf are the aov pixel buffers
    // aovEntries provide metadata about the aov buffers
    // px, py is the pixel
    // depth
    // aovs is the float array of aov values
    static void addAovSamplesToBuffer(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuf,
                                      const std::vector<pbr::AovSchema::Entry> &aovEntries,
                                      unsigned px, unsigned py, const float depth, const float *aovs);

    // Same as the above version but does accumulation internally using atomics
    // so it's safe to call in vector mode.
    // The depths array contains a depth value, one per aovBuf.
    // When bundling, we can't be sure that all values from a camera, came from the same
    // camera ray.
    static void addAovSamplesToBufferSafe(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuf,
                                          const std::vector<pbr::AovSchema::Entry> &aovEntries,
                                          unsigned px, unsigned py, const float *depths, const float *aovs);

    static void addTileSamplesToDisplayFilterBuffer(scene_rdl2::fb_util::VariablePixelBuffer &buf,
                                                    unsigned px, unsigned py,
                                                    unsigned length,
                                                    const uint8_t *values);

public:
    void addBeautyAndAlphaSamplesToBuffer(unsigned px, unsigned py, const scene_rdl2::fb_util::RenderColor& color);

    inline void setPixelInfo(unsigned px, unsigned py, const scene_rdl2::fb_util::PixelInfo &data);

    inline unsigned getNumAovs() const { return mAovBuf.size(); }

    //
    // APIs to query data in various buffers:
    //
    // Coordinate parameters are always linear, tiled coordinates are an internal
    // implementation detail.
    //

    inline float getWeight(unsigned px, unsigned py) const;

    inline scene_rdl2::fb_util::PixelInfo &getPixelInfo(unsigned px, unsigned py);
    inline const scene_rdl2::fb_util::PixelInfo &getPixelInfo(unsigned px, unsigned py) const;

    inline int64_t &getHeatMap(unsigned px, unsigned py);
    inline const int64_t &getHeatMap(unsigned px, unsigned py) const;

    // Be careful when accessing buffer contents directly, the calling code is
    // responsible for any tiled to linear pixel coordinate conversions.
    // (Note: The entire buffer can be untiled using the untile() call.)
    scene_rdl2::fb_util::RenderBuffer       &getRenderBuffer()       { return mRenderBuf; }
    const scene_rdl2::fb_util::RenderBuffer &getRenderBuffer() const { return mRenderBuf; }

    scene_rdl2::fb_util::RenderBuffer       *getRenderBufferOdd()       { return mRenderBufOdd; }
    const scene_rdl2::fb_util::RenderBuffer *getRenderBufferOdd() const { return mRenderBufOdd; }

    scene_rdl2::fb_util::FloatBuffer       &getWeightBuffer()       { return mWeightBuf; }
    const scene_rdl2::fb_util::FloatBuffer &getWeightBuffer() const { return mWeightBuf; }

    pbr::DeepBuffer       *getDeepBuffer()       { return mDeepBuf; }
    const pbr::DeepBuffer *getDeepBuffer() const { return mDeepBuf; }

    pbr::CryptomatteBuffer       *getCryptomatteBuffer()       { return mCryptomatteBuf; }
    const pbr::CryptomatteBuffer *getCryptomatteBuffer() const { return mCryptomatteBuf; }

    scene_rdl2::fb_util::VariablePixelBuffer       &getAovBuffer(unsigned aov)       { return mAovBuf[aov]; }
    const scene_rdl2::fb_util::VariablePixelBuffer &getAovBuffer(unsigned aov) const { return mAovBuf[aov]; }

    pbr::AovFilter getAovBufferFilter(unsigned aov) const { return mAovEntries[aov].filter(); }

    // This is the number of floats in the aov value, which may be different
    // than number of floats actually stored in the aov buffer.  For example,
    // an ST aov will return a value of '2' floats for this function, but under
    // some circumstances (e.g. closest filtering) might be stored in a float4 buffer.
    unsigned int getAovNumFloats(unsigned aov) const { return mAovEntries[aov].numChannels(); }

    scene_rdl2::fb_util::VariablePixelBuffer       &getDisplayFilterBuffer(unsigned idx)       { return mDisplayFilterBufs[idx]; }
    const scene_rdl2::fb_util::VariablePixelBuffer &getDisplayFilterBuffer(unsigned idx) const { return mDisplayFilterBufs[idx]; }
    unsigned int getDisplayFilterCount() const { return mDisplayFilterBufs.size(); }

    bool                           hasPixelInfoBuffer() const              { return mPixelInfoBuf != nullptr; }
    scene_rdl2::fb_util::PixelInfoBuffer       *getPixelInfoBuffer()       { return mPixelInfoBuf; }
    const scene_rdl2::fb_util::PixelInfoBuffer *getPixelInfoBuffer() const { return mPixelInfoBuf; }

    scene_rdl2::fb_util::HeatMapBuffer       *getHeatMapBuffer()       { return mHeatMapBuf; }
    const scene_rdl2::fb_util::HeatMapBuffer *getHeatMapBuffer() const { return mHeatMapBuf; }

    SampleIdBuff       &getResumeStartSampleIdBuff()       { return mResumeStartSampleId; }
    const SampleIdBuff &getResumeStartSampleIdBuff() const { return mResumeStartSampleId; }
    SampleIdBuff       &getCurrSampleIdBuff()       { return mCurrSampleId; }
    const SampleIdBuff &getCurrSampleIdBuff() const { return mCurrSampleId; }

    // Normalizes pixel data using the corresponding existing weight.
    void normalizeRenderBuffer(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                               scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer, bool parallel) const;

    //
    // There is an implicit assumption here that the sparse pixel data was
    // rendered according to the pattern returned by getPixelFillOrder, otherwise
    // don't bother trying to use these extrapolation functions. All input and
    // output buffers must be tiled aligned.
    //

    // Render buffer specific extrapolation functions in that the also lookup
    // the weight buffer for each pixel and take care of pixel normalization.
    // The result is a tiled extrapolated normalized render buffer containing
    // radiance values with the scene alpha in the alpha channel.
    void extrapolateRenderBufferFastPath(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                                         scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer,
                                         bool parallel) const;

    void extrapolateRenderBufferWithViewport(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                                             scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer,
                                             const scene_rdl2::math::Viewport &viewport,
                                             bool parallel) const;

    void extrapolateRenderBufferWithTileList(const scene_rdl2::fb_util::RenderBuffer *srcRenderBuffer,
                                             scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer,
                                             const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                             bool parallel) const;

    finline static unsigned getPixelFillOrder(unsigned pixelIdx)
    {
        MNRY_ASSERT(pixelIdx < 64);
        return (unsigned)gPixelFillOrder[pixelIdx];
    }

    finline static unsigned getPixelFillOrder(unsigned tileId, unsigned pixelId)
    {
        return (unsigned)gScrambledPixelFillOrder[tileId & 1023][pixelId & 63];
    }

    static void constructPixelFillOrderTable(const unsigned nodeId, const unsigned nodeTotal);

    //
    // Extensions to support bundled operation:
    //
    template <bool adaptive>
    static void addSampleBundleHandlerHelper(mcrt_common::ThreadLocalState *tls,
                                             unsigned numEntries,
                                             pbr::BundledRadiance **entries,
                                             Film &film);
    static void addSampleBundleHandler(mcrt_common::ThreadLocalState *tls,
                                       unsigned numEntries,
                                       pbr::BundledRadiance **entries,
                                       void *userData);
    static void addAovSampleBundleHandler(mcrt_common::ThreadLocalState *tls,
                                          unsigned numEntries,
                                          pbr::BundledAov **entries,
                                          void *userData);
    static void addFilteredAovSampleBundleHandler(mcrt_common::ThreadLocalState *tls,
                                                  unsigned numEntries,
                                                  pbr::BundledAov **entries,
                                                  void *userData);
    static void addHeatMapBundleHandler(mcrt_common::ThreadLocalState *tls,
                                        unsigned numEntries,
                                        pbr::BundledHeatMapSample **entries,
                                        void *userData);

    //
    // Adaptive sampling functionality:
    //
    bool isAdaptive() const { return mUseAdaptiveSampling; }

    ActivePixelMask getAdaptiveSampleArea(const scene_rdl2::fb_util::Tile& tile, mcrt_common::ThreadLocalState* tls) const;
    void updateAdaptiveError(const scene_rdl2::fb_util::Tile& tile,
                             const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                             const scene_rdl2::fb_util::RenderBuffer& renderBufOdd,
                             mcrt_common::ThreadLocalState* tls,
                             const unsigned endSampleIdx);
    void updateAdaptiveErrorAll(const scene_rdl2::fb_util::RenderBuffer &renderBuf,
                                const scene_rdl2::fb_util::RenderBuffer &renderBufOdd);

    void disableAdjustAdaptiveTreeUpdateTiming();
    void enableAdjustAdaptiveTreeUpdateTiming(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl);

    bool getAdaptiveDone() const;

    AdaptiveRenderTilesTable *getAdaptiveRenderTilesTable() { return mAdaptiveRenderTilesTable.get(); }
    const AdaptiveRenderTilesTable *getAdaptiveRenderTilesTable() const { return mAdaptiveRenderTilesTable.get(); }

    // This reports the progress fraction value which inherently takes into
    // account any adaptive sampling settings.
    float getProgressFraction(bool activeRendering, size_t *submitted, size_t *total) const;

    bool getResumedFromFileCondition() const { return mResumedFromFileCondition; }
    void setResumedFromFileCondition() { mResumedFromFileCondition = true; }

    const scene_rdl2::fb_util::VariablePixelBuffer *getBeautyAovBuff() const;
    const scene_rdl2::fb_util::VariablePixelBuffer *getAlphaAovBuff() const;

    DebugSamplesRecArray *getDebugSamplesRecArray() { return mDebugSamplesRecArray.get(); }

protected:
    alignas(CACHE_LINE_SIZE) std::atomic<unsigned>   mFilmActivity;
    alignas(CACHE_LINE_SIZE) std::atomic<unsigned>   mPixelInfoBufActivity;

    // 4th channel contains accumulated alpha value for pixel.
    scene_rdl2::fb_util::RenderBuffer mRenderBuf;

    // Accumulated weights for this pixel for corresponding pixel in mRenderBuf.
    scene_rdl2::fb_util::FloatBuffer mWeightBuf;

    // This is used for adaptive sampling and contains the accumulation of all
    // the oddly indexed samples in mRenderBuf. The number of samples can be
    // derived from mWeightBuf, so there is no need to keep a secondary number
    // of samples buffer around also.
    bool mUseAdaptiveSampling;
    scene_rdl2::fb_util::RenderBuffer *mRenderBufOdd;

    // Optional, contains deep samples.  Does not currently support multi-camera.
    pbr::DeepBuffer *mDeepBuf;

    // Optional, contains Cryptomatte samples
    pbr::CryptomatteBuffer     *mCryptomatteBuf;

    // Aov buffers for all cameras
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer> mAovBuf;
    std::vector<pbr::AovSchema::Entry>        mAovEntries;
    unsigned int                              mAovBufNumFloats;
    bool                                      mAovHasClosestFilter;
    // this is used in addAovSampleBundleHandler
    // to efficiently find the buffer associated with an aov index
    std::vector<unsigned>            mAovIdxToBufIdx;
    std::vector<unsigned>            mAovBeautyBufIdx;
    std::vector<unsigned>            mAovAlphaBufIdx;

    // DisplayFilter buffers
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer> mDisplayFilterBufs;

    // Optional, contains linear depth.
    scene_rdl2::fb_util::PixelInfoBuffer* mPixelInfoBuf;

    // Optional, contains heat map samples.
    scene_rdl2::fb_util::HeatMapBuffer * mHeatMapBuf;

    // The start condition of sampleId val at resume render situation. It's empty if non resume render case.
    // After finishing construction by the resume file read stage, this buffer is accessed read-only.
    SampleIdBuff mResumeStartSampleId;

    // This sampleId buffer is used for tracking sampleId for each pixel by adaptive sampling situation.
    // It is empty if uniform sampling case.
    SampleIdBuff mCurrSampleId;

    scene_rdl2::fb_util::Tiler mTiler;

    // tile extrapolation main logic for vectorized mode
    const scene_rdl2::fb_util::TileExtrapolation *mTileExtrapolation;

    // tile render information for adaptive sampling mode
    std::unique_ptr<AdaptiveRenderTilesTable> mAdaptiveRenderTilesTable;

    // Creates a pools of 2^7 mutexes. Only needed for vector mode.
    MutexPool2D<7> mStatsMutex;

    AdaptiveRegions mAdaptiveRegions;

    // In order to track Film is already resumed from file or not.
    bool mResumedFromFileCondition;

    // for debug adaptive resume render
    std::unique_ptr<DebugSamplesRecArray> mDebugSamplesRecArray;
};

//-------------------------------------------------------------------------------------------------------------

inline unsigned
Film::getNumRenderBufferPixelSamples(unsigned px, unsigned py) const
{
    mTiler.linearToTiledCoords(px, py, &px, &py);

    const float weightValue = util::atomicLoad(&mWeightBuf.getPixel(px, py), std::memory_order_relaxed);

    // Round to nearest as an approximation of the number of samples.
    return static_cast<unsigned>(weightValue + 0.5f);
}

inline void
Film::fillPixelSampleCountBuffer(scene_rdl2::fb_util::PixelBuffer<unsigned>& buf) const
{
    const unsigned w = getWidth();
    const unsigned h = getHeight();
    buf.init(w, h);

    for (unsigned x = 0; x < w; ++x) {
        for (unsigned y = 0; y < h; ++y) {
            buf.setPixel(x, y, getNumRenderBufferPixelSamples(x, y));
        }
    }
}

// It is up to the client to check a pixel data buffer has been allocated before
// calling this function.
inline void
Film::setPixelInfo(unsigned px, unsigned py, const scene_rdl2::fb_util::PixelInfo &data)
{
    MNRY_ASSERT(mPixelInfoBuf);

    mTiler.linearToTiledCoords(px, py, &px, &py);

    mPixelInfoBuf->setPixel(px, py, data);

    updatePixelInfoBufferActivity();
}

inline float
Film::getWeight(unsigned px, unsigned py) const
{
    mTiler.linearToTiledCoords(px, py, &px, &py);
    return mWeightBuf.getPixel(px, py);
}

inline scene_rdl2::fb_util::PixelInfo &
Film::getPixelInfo(unsigned px, unsigned py)
{
    MNRY_ASSERT(mPixelInfoBuf);

    mTiler.linearToTiledCoords(px, py, &px, &py);

    return mPixelInfoBuf->getPixel(px, py);
}

inline const scene_rdl2::fb_util::PixelInfo &
Film::getPixelInfo(unsigned px, unsigned py) const
{
    return const_cast<Film *>(this)->getPixelInfo(px, py);
}

inline int64_t &
Film::getHeatMap(unsigned px, unsigned py)
{
    MNRY_ASSERT(mHeatMapBuf);

    mTiler.linearToTiledCoords(px, py, &px, &py);

    return mHeatMapBuf->getPixel(px, py);
}

inline const int64_t &
Film::getHeatMap(unsigned px, unsigned py) const
{
    return const_cast<Film *>(this)->getHeatMap(px, py);
}

inline ActivePixelMask
Film::getAdaptiveSampleArea(const scene_rdl2::fb_util::Tile& tile, mcrt_common::ThreadLocalState* tls) const
{
    const scene_rdl2::math::BBox2i bounds(
        scene_rdl2::math::Vec2i(tile.mMinX, tile.mMinY),
        scene_rdl2::math::Vec2i(tile.mMaxX, tile.mMaxY));
    return mAdaptiveRegions.getSampleArea(bounds, tls);
}

//-------------------------------------------------------------------------------------------------------------

// General purpose extrapolation functions. Can be used to extrapolate any
// type of buffers.
// The HAS_DATA_FUNC functor should return 0xffffffff if valid data is
// contained in the pixel or 0 if not. The 0xffffffff is important!
// The HAS_DATA_FUNC functor should expect two parameters.  The first
// is the value of the buffer (or a const reference to the value).  The
// second is the offset to the pixel data in the buffer.

//
// Fast path when we are extrapolating without a viewport and we aren't doing
// distributed rendering.
//
// Warning: If the buffer isn't aligned to tile boundaries, pixels along the
// viewport edges may not get extrapolated during early progressive passes
// since the pixels they should have been extrapolated from may have been
// clipped out.
//

template<typename PIXEL_TYPE, typename HAS_DATA_FUNC>
inline void
extrapolateBufferFastPath(scene_rdl2::fb_util::PixelBuffer<PIXEL_TYPE> *dstTiledBuffer,
                          const scene_rdl2::fb_util::PixelBuffer<PIXEL_TYPE> &srcTiledBuffer,
                          const scene_rdl2::fb_util::Tiler &tiler,
                          HAS_DATA_FUNC hasData,
                          bool parallel)
{
    MNRY_ASSERT(dstTiledBuffer->getWidth() == srcTiledBuffer.getWidth());
    MNRY_ASSERT(dstTiledBuffer->getHeight() == srcTiledBuffer.getHeight());
    MNRY_ASSERT((dstTiledBuffer->getWidth() & 7) == 0);
    MNRY_ASSERT((dstTiledBuffer->getHeight() & 7) == 0);

    const unsigned numTiles = tiler.mNumTiles;

    simpleLoop(parallel, 0u, numTiles, [&](unsigned i)
    {
        const unsigned tileOfs = i << 6;
        PIXEL_TYPE *__restrict dst = dstTiledBuffer->getData() + tileOfs;
        const PIXEL_TYPE *__restrict src = srcTiledBuffer.getData() + tileOfs;

        for (unsigned y = 0; y < 8; ++y) {

            for (unsigned x = 0; x < 8; ++x) {

                unsigned ofs0 = ((y & ~0) << 3) + (x & ~0) +  0;
                unsigned ofs1 = ((y & ~1) << 3) + (x & ~1) +  0;
                unsigned ofs2 = ((y & ~3) << 3) + (x & ~3) + 18;
                unsigned ofs3 = ((y & ~7) << 3) + (x & ~7) + 18;

                unsigned resultOfs = 0xffffffff;

                uint32_t mask3 = hasData(src[ofs3], tileOfs + ofs3);
                uint32_t mask2 = hasData(src[ofs2], tileOfs + ofs2);
                uint32_t mask1 = hasData(src[ofs1], tileOfs + ofs1);
                uint32_t mask0 = hasData(src[ofs0], tileOfs + ofs0);

                resultOfs = (mask3 & ofs3) | (~mask3 & resultOfs);
                resultOfs = (mask2 & ofs2) | (~mask2 & resultOfs);
                resultOfs = (mask1 & ofs1) | (~mask1 & resultOfs);
                resultOfs = (mask0 & ofs0) | (~mask0 & resultOfs);

                // Reasons for not filling in a pixel are:
                // a) This function was called without first rendering at least
                //    a single sample per tile.
                // b) The pixel landed on a portion of the tile which was clipped
                //    out either by viewport clipping or because the buffer
                //    wasn't aligned to tile boundaries.
                if (resultOfs != 0xffffffff) {
                    *dst = src[resultOfs];
                }

                ++dst;
            }
        }
    });
}

template<typename PIXEL_TYPE, typename HAS_DATA_FUNC>
inline void
extrapolateBufferWithViewport(scene_rdl2::fb_util::PixelBuffer<PIXEL_TYPE> *dstTiledBuffer,
                              const scene_rdl2::fb_util::PixelBuffer<PIXEL_TYPE> &srcTiledBuffer,
                              const scene_rdl2::fb_util::Tiler &tiler,
                              const scene_rdl2::math::Viewport &viewport,
                              HAS_DATA_FUNC hasData,
                              bool parallel)
{
    MNRY_ASSERT(dstTiledBuffer->getWidth() == srcTiledBuffer.getWidth());
    MNRY_ASSERT(dstTiledBuffer->getHeight() == srcTiledBuffer.getHeight());
    MNRY_ASSERT((dstTiledBuffer->getWidth() & 7) == 0);
    MNRY_ASSERT((dstTiledBuffer->getHeight() & 7) == 0);

    const unsigned numTilesX = ((viewport.mMaxX) >> 3) - ((viewport.mMinX) >> 3) + 1;
    const unsigned numTilesY = ((viewport.mMaxY) >> 3) - ((viewport.mMinY) >> 3) + 1;
    const unsigned baseTileX = viewport.mMinX & ~0x07;
    const unsigned baseTileY = viewport.mMinY & ~0x07;

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

            unsigned tileOfs = tiler.linearCoordsToCoarseTileOffset(baseX, baseY);
            MNRY_ASSERT((tileOfs & 63) == 0);

            PIXEL_TYPE *__restrict dst = dstTiledBuffer->getData() + tileOfs;
            const PIXEL_TYPE *__restrict src = srcTiledBuffer.getData() + tileOfs;

            for (unsigned y = minY; y < maxY; ++y) {

                for (unsigned x = minX; x < maxX; ++x) {

                    unsigned ofs0 = ((y & ~0) << 3) + (x & ~0) +  0;
                    unsigned ofs1 = ((y & ~1) << 3) + (x & ~1) +  0;
                    unsigned ofs2 = ((y & ~3) << 3) + (x & ~3) + 18;
                    unsigned ofs3 = ((y & ~7) << 3) + (x & ~7) + 18;

                    unsigned resultOfs = 0xffffffff;

                    uint32_t mask3 = hasData(src[ofs3], tileOfs + ofs3);
                    uint32_t mask2 = hasData(src[ofs2], tileOfs + ofs2);
                    uint32_t mask1 = hasData(src[ofs1], tileOfs + ofs1);
                    uint32_t mask0 = hasData(src[ofs0], tileOfs + ofs0);

                    resultOfs = (mask3 & ofs3) | (~mask3 & resultOfs);
                    resultOfs = (mask2 & ofs2) | (~mask2 & resultOfs);
                    resultOfs = (mask1 & ofs1) | (~mask1 & resultOfs);
                    resultOfs = (mask0 & ofs0) | (~mask0 & resultOfs);

                    if (resultOfs != 0xffffffff) {
                        dst[(y << 3) + x] = src[resultOfs];
                    }
                }
            }
        }
    });
}

// Slow path when we have a viewport assigned.
template<typename PIXEL_TYPE, typename HAS_DATA_FUNC>
inline void
extrapolateBufferWithTileList(scene_rdl2::fb_util::PixelBuffer<PIXEL_TYPE> *dstTiledBuffer,
                              const scene_rdl2::fb_util::PixelBuffer<PIXEL_TYPE> &srcTiledBuffer,
                              const scene_rdl2::fb_util::Tiler &tiler,
                              const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                              HAS_DATA_FUNC hasData,
                              bool parallel)
{
    MNRY_ASSERT(dstTiledBuffer->getWidth() == srcTiledBuffer.getWidth());
    MNRY_ASSERT(dstTiledBuffer->getHeight() == srcTiledBuffer.getHeight());
    MNRY_ASSERT((dstTiledBuffer->getWidth() & 7) == 0);
    MNRY_ASSERT((dstTiledBuffer->getHeight() & 7) == 0);

    const unsigned numTiles = unsigned(tiles.size());

    simpleLoop(parallel, 0u, numTiles, [&](unsigned i) {

        const scene_rdl2::fb_util::Tile &tile = tiles[i];
        const unsigned tileOfs = tiler.linearCoordsToCoarseTileOffset(tile.mMinX, tile.mMinY);
        const unsigned baseX = (tile.mMinX & ~0x07);
        const unsigned baseY = (tile.mMinY & ~0x07);
        const unsigned minX = tile.mMinX - baseX;
        const unsigned maxX = tile.mMaxX - baseX;
        const unsigned minY = tile.mMinY - baseY;
        const unsigned maxY = tile.mMaxY - baseY;
        MNRY_ASSERT(minX < maxX && minX < 8 && maxX <= 8);

        PIXEL_TYPE *__restrict dst = dstTiledBuffer->getData() + tileOfs;
        const PIXEL_TYPE *__restrict src = srcTiledBuffer.getData() + tileOfs;

        for (unsigned y = minY; y < maxY; ++y) {

            for (unsigned x = minX; x < maxX; ++x) {

                unsigned ofs0 = ((y & ~0) << 3) + (x & ~0) +  0;
                unsigned ofs1 = ((y & ~1) << 3) + (x & ~1) +  0;
                unsigned ofs2 = ((y & ~3) << 3) + (x & ~3) + 18;
                unsigned ofs3 = ((y & ~7) << 3) + (x & ~7) + 18;

                unsigned resultOfs = 0xffffffff;

                uint32_t mask3 = hasData(src[ofs3], tileOfs + ofs3);
                uint32_t mask2 = hasData(src[ofs2], tileOfs + ofs2);
                uint32_t mask1 = hasData(src[ofs1], tileOfs + ofs1);
                uint32_t mask0 = hasData(src[ofs0], tileOfs + ofs0);

                resultOfs = (mask3 & ofs3) | (~mask3 & resultOfs);
                resultOfs = (mask2 & ofs2) | (~mask2 & resultOfs);
                resultOfs = (mask1 & ofs1) | (~mask1 & resultOfs);
                resultOfs = (mask0 & ofs0) | (~mask0 & resultOfs);

                if (resultOfs != 0xffffffff) {
                    dst[(y << 3) + x] = src[resultOfs];
                }
            }
        }
    });
}

//-------------------------------------------------------------------------------------------------------------

} // namespace rndr
} // namespace moonray

