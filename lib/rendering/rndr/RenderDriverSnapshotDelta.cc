// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#include "RenderDriver.h"
#include "Util.h"

#include <moonray/rendering/mcrt_common/Clock.h>
#include <scene_rdl2/common/fb_util/ActivePixels.h>
#include <scene_rdl2/common/fb_util/SnapshotUtil.h>

//#define SNAPSHOT_DELTA_TIMING_TEST
//#define SNAPSHOT_DELTA_PIXINFO_TIMING_TEST
//#define SNAPSHOT_DELTA_HEATMAP_TIMING_TEST
//#define SNAPSHOT_DELTA_WEIGHTBUFFER_TIMING_TEST
//#define SNAPSHOT_DELTA_RENDERBUFFERODD_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT2_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT3_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT4_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT_VARIANCE_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT2_VARIANCE_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_FLOAT3_VARIANCE_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_RGB_VARIANCE_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_VISIBILITY_TIMING_TEST
//#define SNAPSHOT_DELTA_AOV_VARIANCE_VISIBILITY_TIMING_TEST

#if defined(SNAPSHOT_DELTA_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_PIXINFO_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_HEATMAP_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_WEIGHTBUFFER_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_RENDERBUFFERODD_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_FLOAT_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_FLOAT2_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_FLOAT3_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_FLOAT_VARIANCE_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_FLOAT2_VARIANCE_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_FLOAT3_VARIANCE_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_RGB_VARIANCE_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_VISIBILITY_TIMING_TEST) || \
    defined(SNAPSHOT_DELTA_AOV_VARIANCE_VISIBILITY_TIMING_TEST)
#include <scene_rdl2/common/rec_time/RecTime.h>
#endif

namespace moonray {
namespace rndr {

//------------------------------------------------------------------------------

template <int dimension>
struct SnapshotDeltaAovFloatN;

template<>
struct SnapshotDeltaAovFloatN<1>
{
    static scene_rdl2::fb_util::FloatBuffer &get(scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloatBuffer();
    }
    static const scene_rdl2::fb_util::FloatBuffer &get(const scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloatBuffer();
    }
    static uint64_t snapshotTileValWeight(float *dst, float *dstWeight, const float *src, const float *srcWeight) {
        return scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloatWeight((uint32_t *)dst,
                                                              (uint32_t *)dstWeight,
                                                              (const uint32_t *)src,
                                                              (const uint32_t *)srcWeight);
    }
};

template<>
struct SnapshotDeltaAovFloatN<2>
{
    static scene_rdl2::fb_util::Float2Buffer &get(scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloat2Buffer();
    }
    static const scene_rdl2::fb_util::Float2Buffer &get(const scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloat2Buffer();
    }
    static uint64_t snapshotTileValWeight(scene_rdl2::math::Vec2f *dst, float *dstWeight, const scene_rdl2::math::Vec2f *src, const float *srcWeight) {
        return scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloat2Weight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)src,
                                                               (const uint32_t *)srcWeight);
    }
};

template<>
struct SnapshotDeltaAovFloatN<3>
{
    static scene_rdl2::fb_util::Float3Buffer &get(scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloat3Buffer();
    }
    static const scene_rdl2::fb_util::Float3Buffer &get(const scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloat3Buffer();
    }
    static uint64_t snapshotTileValWeight(scene_rdl2::math::Vec3f *dst, float *dstWeight, const scene_rdl2::math::Vec3f *src, const float *srcWeight) {
        return scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloat3Weight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)src,
                                                               (const uint32_t *)srcWeight);
    }
};

template<>
struct SnapshotDeltaAovFloatN<4>
{
    static scene_rdl2::fb_util::Float4Buffer &get(scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloat4Buffer();
    }
    static const scene_rdl2::fb_util::Float4Buffer &get(const scene_rdl2::fb_util::VariablePixelBuffer &vPixBuffer) {
        return vPixBuffer.getFloat4Buffer();
    }
    static uint64_t snapshotTileValWeight(scene_rdl2::math::Vec4f *dst, float *dstWeight, const scene_rdl2::math::Vec4f *src, const float *srcWeight) {
        return scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloat4Weight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)src,
                                                               (const uint32_t *)srcWeight);
    }
};

template<int dimension>
static void
snapshotDeltaAovFloatN(const Film &film,
                       const unsigned aovIdx,
                       scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                       scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                       scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                       bool parallel)
{
    const unsigned numTiles = film.getTiler().mNumTiles;
    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            auto &dstBuffer = SnapshotDeltaAovFloatN<dimension>::get(*dstRenderOutputBuffer);
            const auto &srcBuffer = SnapshotDeltaAovFloatN<dimension>::get(film.getAovBuffer(aovIdx));

            auto *dst = dstBuffer.getData() + pixId;
            float *dstWeight = dstRenderOutputWeightBuffer->getData() + pixId;
            const auto *src = srcBuffer.getData() + pixId;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask =
                SnapshotDeltaAovFloatN<dimension>::snapshotTileValWeight(dst, dstWeight, src, srcWeight);

            activePixelsRenderOutput.setTileMask(tileId, activePixelMask);
        });
}

//------------------------------------------------------------------------------

namespace {
float
reduce_max(float f)
{
    return f;
}
} // anonymous namespace

template<typename SRC_BUFF_TYPE>
static void
snapshotDeltaAovVariance(SRC_BUFF_TYPE srcVarianceBuffer,
                         const Film &film,
                         scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                         scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                         scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                         bool parallel)
{
    const unsigned numTiles = film.getTiler().mNumTiles;

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            scene_rdl2::fb_util::FloatBuffer &dstFloatBuffer = dstRenderOutputBuffer->getFloatBuffer();

            // Compute variance value for current tile
            //
            //   varianceAOV buffer has own pixel data format and we need to convert to single variance value
            //   before creating delta information by snapshotTile.
            //   Current logic is very naive and always compute variance value regardless of
            //   values are changed or not. This visibility value computation is relatively
            //   ultralight, however, we might be considered compare source variable information
            //   instead of computed variance when we need more speed up here. Toshi (Sep/10/2018)
            //
            const auto *srcTileStartAddr = &(srcVarianceBuffer.getData()[pixId]);
            float varianceTileInfo[64];
            for (int currPixId = 0; currPixId < 64; ++currPixId) {
                auto &currPix = srcTileStartAddr[currPixId];
                varianceTileInfo[currPixId] = reduce_max(currPix.variance());
            }

            // do snapshot against computed variance
            float *dst = dstFloatBuffer.getData() + pixId;
            float *dstWeight = dstRenderOutputWeightBuffer->getData() + pixId;
            const float *src = varianceTileInfo;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;
            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloatWeight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)src,
                                                               (const uint32_t *)srcWeight);
            activePixelsRenderOutput.setTileMask(tileId, activePixelMask);
        });
}

//------------------------------------------------------------------------------

void
RenderDriver::snapshotDelta(unsigned filmIdx,
                            scene_rdl2::fb_util::RenderBuffer *dstRenderBuffer,
                            scene_rdl2::fb_util::FloatBuffer *dstWeightBuffer,
                            scene_rdl2::fb_util::ActivePixels &activePixels,
                            bool parallel) const
//
// Creates snapshot renderBuffer/weightBuffer data w/ activePixels information
// for ProgressiveFrame message related logic.
// No resize, no extrapolation, no untiling logic is handled internally.
// Simply just create snapshot data with properly constructed activePixels data based on
// difference between current and previous renderBuffer and weightBuffer.
// renderBuffer/weightBuffer is tiled format and renderBuffer is not normalized by weight
//
{
    const Film &film = mFilms[filmIdx];
    const unsigned numTiles = film.getTiler().mNumTiles;

#ifdef SNAPSHOT_DELTA_TIMING_TEST
    static rec_time::RecTimeLog recTimeLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            scene_rdl2::fb_util::RenderColor *dst = dstRenderBuffer->getData() + pixId;
            float *dstWeight = dstWeightBuffer->getData() + pixId;
            const scene_rdl2::fb_util::RenderColor *srcColor = film.getRenderBuffer().getData() + pixId;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileColorWeight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)srcColor,
                                                               (const uint32_t *)srcWeight);

            activePixels.setTileMask(tileId, activePixelMask);
        });

#ifdef SNAPSHOT_DELTA_TIMING_TEST
    recTimeLog.add(recTime.end());
    if (recTimeLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot ave:"
                  << recTimeLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeLog.reset();
    }
#endif // end SNAPSHOT_DELTA_TIMING_TEST
}

void
RenderDriver::snapshotDeltaRenderBufferOdd(unsigned filmIdx,
                                           scene_rdl2::fb_util::RenderBuffer *dstRenderBufferOdd,
                                           scene_rdl2::fb_util::FloatBuffer *dstWeightRenderBufferOdd,
                                           scene_rdl2::fb_util::ActivePixels &activePixelsRenderBufferOdd,
                                           bool parallel) const
//
// Creates snapshot renderBufferOdd/weightRenderBufferOdd data w/ activePixelsRenderBufferOdd information
// for ProgressiveFrame message related logic.
// No resize, no extrapolation, no untiling logic is handled internally.
// Simply just create snapshot data with properly constructed activePixelsRenderBufferOdd data based on
// difference between current and previous renderBufferOdd and weightRenderBufferOdd.
// renderBufferOdd/weightRenderBufferOdd is tiled format and renderBufferOdd is not normalized by weight
//
{
    const Film &film = mFilms[filmIdx];
    const unsigned numTiles = film.getTiler().mNumTiles;

    if (!film.getRenderBufferOdd()) {
        return;                 // skip snapshot when could not access renderBufferOdd
    }

#ifdef SNAPSHOT_DELTA_RENDERBUFFERODD_TIMING_TEST
    static rec_time::RecTimeLog recTimeRenderBufferOddLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_RENDERBUFFERODD_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            scene_rdl2::fb_util::RenderColor *dst = dstRenderBufferOdd->getData() + pixId;
            float *dstWeight = dstWeightRenderBufferOdd->getData() + pixId;
            const scene_rdl2::fb_util::RenderColor *srcColor = film.getRenderBufferOdd()->getData() + pixId;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileColorWeight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)srcColor,
                                                               (const uint32_t *)srcWeight);

            activePixelsRenderBufferOdd.setTileMask(tileId, activePixelMask);
        });

#ifdef SNAPSHOT_DELTA_RENDERBUFFERODD_TIMING_TEST
    recTimeRenderBufferOddLog.add(recTime.end());
    if (recTimeRenderBufferOddLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot renderBufferOdd ave:"
                  << recTimeRenderBufferOddLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeRenderBufferOddLog.reset();
    }
#endif // end SNAPSHOT_DELTA_RENDERBUFFERODD_TIMING_TEST
}

void
RenderDriver::snapshotDeltaPixelInfo(unsigned filmIdx,
                                     scene_rdl2::fb_util::PixelInfoBuffer *dstPixelInfoBuffer,
                                     scene_rdl2::fb_util::FloatBuffer *dstPixelInfoWeightBuffer,
                                     scene_rdl2::fb_util::ActivePixels &activePixelsPixelInfo,
                                     bool parallel) const
//
// Creates snapshot pixelInfoBuffer/pixelInfoWeightBuffer data w/ activePixelsPixelInfo information
// for ProgressiveFrame message related logic.
// Simply just create snapshot data with properly constructed activePixels data based on
// difference between current and previous pixelInfoBuffer and pixelInfoWeightBuffer.
// pixelInfoBuffer/pixelInfoWeightBuffer is tiled format.
//
{
    const Film &film = mFilms[filmIdx];
    const unsigned numTiles = film.getTiler().mNumTiles;

#ifdef SNAPSHOT_DELTA_PIXINFO_TIMING_TEST
    static rec_time::RecTimeLog recTimePixInfoLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_PIXINFO_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            scene_rdl2::fb_util::PixelInfo *dst = dstPixelInfoBuffer->getData() + pixId;
            float *dstW = dstPixelInfoWeightBuffer->getData() + pixId;
            // primary camera only for now
            const scene_rdl2::fb_util::PixelInfo *src = film.getPixelInfoBuffer(0)->getData() + pixId;
            const float *srcW = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask = scene_rdl2::fb_util::SnapshotUtil::snapshotTilePixelInfoWeight((uint32_t *)dst,
                                                                                          (uint32_t *)dstW,
                                                                                          (const uint32_t *)src,
                                                                                          (const uint32_t *)srcW);

            activePixelsPixelInfo.setTileMask(tileId, activePixelMask);
        });

#ifdef SNAPSHOT_DELTA_PIXINFO_TIMING_TEST
    recTimePixInfoLog.add(recTime.end());
    if (recTimePixInfoLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot PixelInfo ave:"
                  << recTimePixInfoLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimePixInfoLog.reset();
    }
#endif // end SNAPSHOT_DELTA_PIXINFO_TIMING_TEST
}

void
RenderDriver::snapshotDeltaHeatMap(unsigned filmIdx,
                                   scene_rdl2::fb_util::HeatMapBuffer *dstHeatMapBuffer,
                                   scene_rdl2::fb_util::FloatBuffer *dstHeatMapWeightBuffer,
                                   scene_rdl2::fb_util::ActivePixels &activePixelsHeatMap,
                                   scene_rdl2::fb_util::FloatBuffer *dstHeatMapSecBuffer,
                                   bool parallel) const
//
// Creates snapshot heatMapBuffer/heatMapWeightBuffer data w/ activePixelsHeatMap information
// for ProgressiveFrame message related logic.
// Simply just create snapshot data with properly constructed activePixels data based on
// difference between current and previous heatMapBuffer/heatMapWeightBuffer.
// Also create heatMapSecBuffer just for active pixels only.
// heatMapBuffer/heatMapWeightBuffer/heatMapSecBuffer are tiled format.
//
{
    const Film &film = mFilms[filmIdx];
    const unsigned numTiles = film.getTiler().mNumTiles;

#ifdef SNAPSHOT_DELTA_HEATMAP_TIMING_TEST
    static rec_time::RecTimeLog recTimeHeatMapLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_HEATMAP_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels

            int64_t *dst = dstHeatMapBuffer->getData() + pixId;
            float *dstW = dstHeatMapWeightBuffer->getData() + pixId;
            const int64_t *src = film.getHeatMapBuffer()->getData() + pixId;
            const float *srcW = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileHeatMapWeight((uint64_t *)dst,
                                                                             (uint32_t *)dstW,
                                                                             (const uint64_t *)src,
                                                                             (const uint32_t *)srcW);
            activePixelsHeatMap.setTileMask(tileId, activePixelMask);

            //
            // convert tick to sec only for active pixels
            //
            float *dstSec = dstHeatMapSecBuffer->getData() + pixId;
            for (unsigned currPixId = 0; currPixId < 64; ++currPixId) {
                if (!activePixelMask) break; // early exit
                if (activePixelMask & static_cast<uint64_t>(0x1)) {
                    dstSec[currPixId] = mcrt_common::Clock::seconds(dst[currPixId]); // convert to seconds : not normalized
                }
                activePixelMask >>= 1;
            }
        });

#ifdef SNAPSHOT_DELTA_HEATMAP_TIMING_TEST
    recTimeHeatMapLog.add(recTime.end());
    if (recTimeHeatMapLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot HeatMap ave:"
                  << recTimeHeatMapLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeHeatMapLog.reset();
    }
#endif // end SNAPSHOT_DELTA_HEATMAP_TIMING_TEST
}

void
RenderDriver::snapshotDeltaWeightBuffer(unsigned filmIdx,
                                        scene_rdl2::fb_util::FloatBuffer *dstWeightBuffer,
                                        scene_rdl2::fb_util::ActivePixels &activePixelsWeightBuffer,
                                        bool parallel) const
//
// Creates snapshot weightBuffer data w/ activePixelsWeightBuffer information
// for ProgressiveFrame message related logic.
// Simply just create snapshot data with properly constructed activePixels data based on
// difference between current and previous weightBuffer.
// weightBuffer is tiled format.
//
{
    const Film &film = mFilms[filmIdx];
    const unsigned numTiles = film.getTiler().mNumTiles;

#ifdef SNAPSHOT_DELTA_WEIGHTBUFFER_TIMING_TEST
    static rec_time::RecTimeLog recTimeWeightBufferLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_WEIGHTBUFFER_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels

            float *dst = dstWeightBuffer->getData() + pixId;
            const float *src = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask = scene_rdl2::fb_util::SnapshotUtil::snapshotTileWeightBuffer((uint32_t *)dst, (const uint32_t *)src);
            activePixelsWeightBuffer.setTileMask(tileId, activePixelMask);
        });

#ifdef SNAPSHOT_DELTA_WEIGHTBUFFER_TIMING_TEST
    recTimeWeightBufferLog.add(recTime.end());
    if (recTimeWeightBufferLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot WeightBuffer ave:"
                  << recTimeWeightBufferLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeWeightBufferLog.reset();
    }
#endif // end SNAPSHOT_DELTA_WEIGHTBUFFER_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAov(unsigned filmIdx,
                               unsigned aovIdx,
                               scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                               scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                               scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                               bool parallel) const
//
// Creates snapshot renderOutputBuffer(aovIndex)/renderOutputWeightBuffer(aovIndex) data w/
// activePixelsRenderOutput information for ProgressiveFrame message related logic.
// Simply just create snapshot data with properly constructed activePixels data based on
// difference between current and previous renderOutputBuffer(aovIndex) and renderOutputWeightBuffer(aovIndex).
// renderOutputBuffer(aovIndex)/renderOutputWeightBuffer(aovIndex) are tiled format.
//
{
    const Film &film = getFilm(filmIdx);

    switch (film.getAovBuffer(aovIdx).getFormat()) {
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT :
        snapshotDeltaAovFloat(filmIdx,
                              aovIdx,
                              dstRenderOutputBuffer,
                              dstRenderOutputWeightBuffer,
                              activePixelsRenderOutput,
                              parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2 :
        snapshotDeltaAovFloat2(filmIdx,
                               aovIdx,
                               dstRenderOutputBuffer,
                               dstRenderOutputWeightBuffer,
                               activePixelsRenderOutput,
                               parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3 :
        snapshotDeltaAovFloat3(filmIdx,
                               aovIdx,
                               dstRenderOutputBuffer,
                               dstRenderOutputWeightBuffer,
                               activePixelsRenderOutput,
                               parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4 :
        // currently these should only be closest filter aovs
        // remove this check when/if this is no longer the case
        MNRY_ASSERT(film.getAovBufferFilter(aovIdx) == pbr::AOV_FILTER_CLOSEST);
        snapshotDeltaAovFloat4(filmIdx,
                               aovIdx,
                               dstRenderOutputBuffer,
                               dstRenderOutputWeightBuffer,
                               activePixelsRenderOutput,
                               parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT_VARIANCE:
        snapshotDeltaAovFloatVariance(filmIdx,
                                      aovIdx,
                                      dstRenderOutputBuffer,
                                      dstRenderOutputWeightBuffer,
                                      activePixelsRenderOutput,
                                      parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2_VARIANCE:
        snapshotDeltaAovFloat2Variance(filmIdx,
                                       aovIdx,
                                       dstRenderOutputBuffer,
                                       dstRenderOutputWeightBuffer,
                                       activePixelsRenderOutput,
                                       parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3_VARIANCE:
        snapshotDeltaAovFloat3Variance(filmIdx,
                                       aovIdx,
                                       dstRenderOutputBuffer,
                                       dstRenderOutputWeightBuffer,
                                       activePixelsRenderOutput,
                                       parallel);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::RGB_VARIANCE:
        snapshotDeltaAovRgbVariance(filmIdx,
                                    aovIdx,
                                    dstRenderOutputBuffer,
                                    dstRenderOutputWeightBuffer,
                                    activePixelsRenderOutput,
                                    parallel);
        break;
    }
}

void
RenderDriver::snapshotDeltaAovVisibility(unsigned filmIdx,
                                         unsigned aovIdx,
                                         scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                         scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                         scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                         bool parallel) const
{
    const Film &film = getFilm(filmIdx);
    const unsigned numTiles = film.getTiler().mNumTiles;

#ifdef SNAPSHOT_DELTA_AOV_VISIBILITY_TIMING_TEST
    static rec_time::RecTimeLog recTimeVisibilityLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_VISIBILITY_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            scene_rdl2::fb_util::FloatBuffer &dstFloatBuffer = dstRenderOutputBuffer->getFloatBuffer();

            // VisibilityAOV is always float2 buffer
            const scene_rdl2::fb_util::Float2Buffer &srcFloat2Buffer = film.getAovBuffer(aovIdx).getFloat2Buffer();

            // Compute visibility value for current tile.
            //
            //   visibilityAOV buffer has 2 float per pixel and we need to convert this 2 floats
            //   into single visibility value before creating delta information by snapshotTile.
            //   Current logic is very naive and always compute visiblity value regardless of
            //   values are changed or not. This visibility value computation is relatively
            //   ultralight, however, we might be considered compare source 2 floats value
            //   instead of computed visibility when we need more speed up here. Toshi (Sep/07/2018)
            //
            const scene_rdl2::math::Vec2f *srcTileStartAddr = srcFloat2Buffer.getData() + pixId;
            float visibilityTileInfo[64];
            for (int currPixId = 0; currPixId < 64; ++currPixId) {
                const scene_rdl2::math::Vec2f &currPix = srcTileStartAddr[currPixId];
                visibilityTileInfo[currPixId] = (currPix.y > 0.0f)? (currPix.x / currPix.y): 0.0f;
            }

            // do snapshot against computed visibility
            float *dst = dstFloatBuffer.getData() + pixId;
            float *dstWeight = dstRenderOutputWeightBuffer->getData() + pixId;
            const float *src = visibilityTileInfo;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;
            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloatWeight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)src,
                                                               (const uint32_t *)srcWeight);
            activePixelsRenderOutput.setTileMask(tileId, activePixelMask);
        });

#ifdef SNAPSHOT_DELTA_AOV_VISIBILITY_TIMING_TEST
    recTimeVisibilityLog.add(recTime.end());
    if (recTimeVisibilityLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot visibility ave:"
                  << recTimeVisibilityLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeVisibilityLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_VISIBILITY_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovVarianceVisibility(unsigned filmIdx,
                                                 unsigned aovIdx,
                                                 scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                                 scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                                 scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                                 bool parallel) const
{
    const Film &film = getFilm(filmIdx);
    const unsigned numTiles = film.getTiler().mNumTiles;

#ifdef SNAPSHOT_DELTA_AOV_VARIANCE_VISIBILITY_TIMING_TEST
    static rec_time::RecTimeLog recTimeVarianceVisibilityLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_VARIANCE_VISIBILITY_TIMING_TEST

    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            scene_rdl2::fb_util::FloatBuffer &dstFloatBuffer = dstRenderOutputBuffer->getFloatBuffer();

            // VisibilityAOV is always float2 buffer
            const scene_rdl2::fb_util::Float2Buffer &srcFloat2Buffer = film.getAovBuffer(aovIdx).getFloat2Buffer();

            // Compute variance visibility value for current tile.
            // See also RenderDriver::snapshotVisibilityVarianceBuffer()
            const scene_rdl2::math::Vec2f *srcTileStartAddr = srcFloat2Buffer.getData() + pixId;
            float visibilityTileInfo[64];
            for (int currPixId = 0; currPixId < 64; ++currPixId) {
                const scene_rdl2::math::Vec2f &currPix = srcTileStartAddr[currPixId];

                float s2 = 0.0f;
                if (currPix.y > 0.0f) {
                    const float p = currPix.x / currPix.y;
                    s2 = p * (1.0f - p);
                }
                visibilityTileInfo[currPixId] = s2;
            }

            // do snapshot against computed visibility
            float *dst = dstFloatBuffer.getData() + pixId;
            float *dstWeight = dstRenderOutputWeightBuffer->getData() + pixId;
            const float *src = visibilityTileInfo;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;
            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloatWeight((uint32_t *)dst,
                                                               (uint32_t *)dstWeight,
                                                               (const uint32_t *)src,
                                                               (const uint32_t *)srcWeight);
            activePixelsRenderOutput.setTileMask(tileId, activePixelMask);
        });

#ifdef SNAPSHOT_DELTA_AOV_VARIANCE_VISIBILITY_TIMING_TEST
    recTimeVarianceVisibilityLog.add(recTime.end());
    if (recTimeVarianceVisibilityLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot variance visibility ave:"
                  << recTimeVarianceVisibilityLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeVarianceVisibilityLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_VARIANCE_VISIBILITY_TIMING_TEST
}

void
RenderDriver::snapshotDeltaDisplayFilter(unsigned filmIdx,
                                         unsigned dfIdx,
                                         scene_rdl2::fb_util::VariablePixelBuffer *renderOutputBuffer,
                                         scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                         scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                         bool parallel) const
{
    // Only supports FLOAT3 DisplayFilters currently
    MNRY_ASSERT(renderOutputBuffer->getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3);
    const Film &film = getFilm(filmIdx);
    const unsigned numTiles = film.getTiler().mNumTiles;
    simpleLoop(parallel, 0u, numTiles, [&](unsigned tileId) {
            unsigned pixId = tileId << 6; // tile is 8x8 = 64pixels
            auto &dstBuffer = renderOutputBuffer->getFloat3Buffer();
            const auto &srcBuffer = film.getDisplayFilterBuffer(dfIdx).getFloat3Buffer();

            auto *dst = dstBuffer.getData() + pixId;
            float *dstWeight = dstRenderOutputWeightBuffer->getData() + pixId;
            const auto *src = srcBuffer.getData() + pixId;
            const float *srcWeight = film.getWeightBuffer().getData() + pixId;

            uint64_t activePixelMask =
                scene_rdl2::fb_util::SnapshotUtil::snapshotTileFloat3Weight((uint32_t *)dst,
                                                                (uint32_t *)dstWeight,
                                                                (const uint32_t *)src,
                                                                (const uint32_t *)srcWeight);

            activePixelsRenderOutput.setTileMask(tileId, activePixelMask);
        });
}

//------------------------------------------------------------------------------

void
RenderDriver::snapshotDeltaAovFloat(unsigned filmIdx,
                                    unsigned aovIdx,
                                    scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                    scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                    scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                    bool parallel) const
{
#ifdef SNAPSHOT_DELTA_AOV_FLOAT_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloatLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT_TIMING_TEST

    snapshotDeltaAovFloatN<1>(getFilm(filmIdx),
                              aovIdx,
                              dstRenderOutputBuffer,
                              dstRenderOutputWeightBuffer,
                              activePixelsRenderOutput,
                              parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT_TIMING_TEST
    recTimeFloatLog.add(recTime.end());
    if (recTimeFloatLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float ave:"
                  << recTimeFloatLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloatLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovFloat2(unsigned filmIdx,
                                     unsigned aovIdx,
                                     scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                     scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                     scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                     bool parallel) const
{
#ifdef SNAPSHOT_DELTA_AOV_FLOAT2_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloat2Log;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT2_TIMING_TEST

    snapshotDeltaAovFloatN<2>(getFilm(filmIdx),
                              aovIdx,
                              dstRenderOutputBuffer,
                              dstRenderOutputWeightBuffer,
                              activePixelsRenderOutput,
                              parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT2_TIMING_TEST
    recTimeFloat2Log.add(recTime.end());
    if (recTimeFloat2Log.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float2 ave:"
                  << recTimeFloat2Log.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloat2Log.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT2_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovFloat3(unsigned filmIdx,
                                     unsigned aovIdx,
                                     scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                     scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                     scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                     bool parallel) const
{


#ifdef SNAPSHOT_DELTA_AOV_FLOAT3_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloat3Log;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT3_TIMING_TEST

    snapshotDeltaAovFloatN<3>(getFilm(filmIdx),
                              aovIdx,
                              dstRenderOutputBuffer,
                              dstRenderOutputWeightBuffer,
                              activePixelsRenderOutput,
                              parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT3_TIMING_TEST
    recTimeFloat3Log.add(recTime.end());
    if (recTimeFloat3Log.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float3 ave:"
                  << recTimeFloat3Log.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloat3Log.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT3_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovFloat4(unsigned filmIdx,
                                     unsigned aovIdx,
                                     scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                     scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                     scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                     bool parallel) const
{


#ifdef SNAPSHOT_DELTA_AOV_FLOAT4_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloat4Log;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT4_TIMING_TEST

    snapshotDeltaAovFloatN<4>(getFilm(filmIdx),
                              aovIdx,
                              dstRenderOutputBuffer,
                              dstRenderOutputWeightBuffer,
                              activePixelsRenderOutput,
                              parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT4_TIMING_TEST
    recTimeFloat4Log.add(recTime.end());
    if (recTimeFloat4Log.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float4 ave:"
                  << recTimeFloat4Log.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloat4Log.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT4_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovFloatVariance(unsigned filmIdx,
                                            unsigned aovIdx,
                                            scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                            scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                            scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                            bool parallel) const
{
#ifdef SNAPSHOT_DELTA_AOV_FLOAT_VARIANCE_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloatVarianceLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT_VARIANCE_TIMING_TEST

    const Film &film = getFilm(filmIdx);
    const auto &srcVarianceBuffer = film.getAovBuffer(aovIdx).getFloatVarianceBuffer();
    snapshotDeltaAovVariance(srcVarianceBuffer,
                             film,
                             dstRenderOutputBuffer,
                             dstRenderOutputWeightBuffer,
                             activePixelsRenderOutput,
                             parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT_VARIANCE_TIMING_TEST
    recTimeFloatVarianceLog.add(recTime.end());
    if (recTimeFloatVarianceLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float Variance ave:"
                  << recTimeFloatVarianceLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloatVarianceLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT_VARIANCE_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovFloat2Variance(unsigned filmIdx,
                                             unsigned aovIdx,
                                             scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                             scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                             scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                             bool parallel) const
{
#ifdef SNAPSHOT_DELTA_AOV_FLOAT2_VARIANCE_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloat2VarianceLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT2_VARIANCE_TIMING_TEST

    const Film &film = getFilm(filmIdx);
    const auto &srcVarianceBuffer = film.getAovBuffer(aovIdx).getFloat2VarianceBuffer();
    snapshotDeltaAovVariance(srcVarianceBuffer,
                             film,
                             dstRenderOutputBuffer,
                             dstRenderOutputWeightBuffer,
                             activePixelsRenderOutput,
                             parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT2_VARIANCE_TIMING_TEST
    recTimeFloat2VarianceLog.add(recTime.end());
    if (recTimeFloat2VarianceLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float2 Variance ave:"
                  << recTimeFloat2VarianceLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloat2VarianceLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT2_VARIANCE_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovFloat3Variance(unsigned filmIdx,
                                             unsigned aovIdx,
                                             scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                             scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                             scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                             bool parallel) const
{
#ifdef SNAPSHOT_DELTA_AOV_FLOAT3_VARIANCE_TIMING_TEST
    static rec_time::RecTimeLog recTimeFloat3VarianceLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_FLOAT3_VARIANCE_TIMING_TEST

    const Film &film = getFilm(filmIdx);
    const auto &srcVarianceBuffer = film.getAovBuffer(aovIdx).getFloat3VarianceBuffer();
    snapshotDeltaAovVariance(srcVarianceBuffer,
                             film,
                             dstRenderOutputBuffer,
                             dstRenderOutputWeightBuffer,
                             activePixelsRenderOutput,
                             parallel);

#ifdef SNAPSHOT_DELTA_AOV_FLOAT3_VARIANCE_TIMING_TEST
    recTimeFloat3VarianceLog.add(recTime.end());
    if (recTimeFloat3VarianceLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Float3 Variance ave:"
                  << recTimeFloat3VarianceLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeFloat3VarianceLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_FLOAT3_VARIANCE_TIMING_TEST
}

void
RenderDriver::snapshotDeltaAovRgbVariance(unsigned filmIdx,
                                          unsigned aovIdx,
                                          scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                          scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                          scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                          bool parallel) const
{
#ifdef SNAPSHOT_DELTA_AOV_RGB_VARIANCE_TIMING_TEST
    static rec_time::RecTimeLog recTimeRgbVarianceLog;
    rec_time::RecTime recTime;
    recTime.start();
#endif // end SNAPSHOT_DELTA_AOV_RGB_VARIANCE_TIMING_TEST

    const Film &film = getFilm(filmIdx);
    const auto &srcVarianceBuffer = film.getAovBuffer(aovIdx).getRgbVarianceBuffer();
    snapshotDeltaAovVariance(srcVarianceBuffer,
                             film,
                             dstRenderOutputBuffer,
                             dstRenderOutputWeightBuffer,
                             activePixelsRenderOutput,
                             parallel);

#ifdef SNAPSHOT_DELTA_AOV_RGB_VARIANCE_TIMING_TEST
    recTimeRgbVarianceLog.add(recTime.end());
    if (recTimeRgbVarianceLog.getTotal() == 24) {
        std::cerr << ">> Film.cc snapshot Rgb Variance ave:"
                  << recTimeRgbVarianceLog.getAverage() * 1000.0f << " ms" << std::endl;
        recTimeRgbVarianceLog.reset();
    }
#endif // end SNAPSHOT_DELTA_AOV_RGB_VARIANCE_TIMING_TEST
}

} // namespace rndr
} // namespace moonray
