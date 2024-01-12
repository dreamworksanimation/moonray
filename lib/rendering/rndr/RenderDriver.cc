// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include <scene_rdl2/render/util/AtomicFloat.h> // Needs to be included before any OpenImageIO file
#include <moonray/rendering/pbr/core/Scene.h>

#include "AdaptiveRenderTilesTable.h"
#include "RenderDriver.h"
#include "RenderContext.h"
#include "PixelBufferUtils.h"
#include "RenderOutputDriver.h"
#include "RenderOutputHelper.h"
#include "TileScheduler.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/Statistics.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/DebugRay.h>
#include <moonray/rendering/pbr/handlers/XPURayHandlers.h>
#include <moonray/rendering/rt/gpu/GPUAccelerator.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>

#include <scene_rdl2/render/util/Memory.h>
#include <random>

// Quick way to force a single sample per pixel. For debugging.
// This define doesn't affect realtime mode.
//#define FORCE_SINGLE_SAMPLE_PER_PIXEL

// Quick way to dump out detailed statistics related to the realtime render mode.
// For debugging. This is not meant as a user facing feature, but more of a way
// for developers to diagnose realtime rendering performance issues.
//#define LOG_REALTIME_FRAME_STATS_TO_FILE

#define MAX_REALTIME_FRAME_STATS_TO_RECORD  1024
#define REALTIME_FRAME_STATS_LOGFILE        "./RealtimeFrameStats_Machine%03d_Save%02d.csv"

// This is a directive to switch multi-machine tile schedule. We have to select one of them.
// DETERMINISTIC_SHIFT is better at this moment but we also keep RANDOM_SHIFT version for
// future enhancement/comparison reasons
//#define MULTI_MACHINE_TILE_SCHEDULE_RANDOM_SHIFT
#define MULTI_MACHINE_TILE_SCHEDULE_DETERMINISTIC_SHIFT

using namespace scene_rdl2::util;
// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.
using scene_rdl2::logging::Logger;

// Can be called from C++ and ISPC
extern "C" bool isRenderCanceled()
{
    const bool canceled = moonray::rndr::gCancelFlag.isCanceled();
    return canceled;
}

namespace moonray {

using namespace mcrt_common;

namespace rndr {

CancelFlag gCancelFlag;

bool
hasData(float pixel) {
    return scene_rdl2::math::isfinite(pixel);
}

bool
hasData(const scene_rdl2::math::Vec2f &pixel) {
    return scene_rdl2::math::isFinite(pixel);
}

bool
hasData(const scene_rdl2::math::Vec3f &pixel) {
    return scene_rdl2::math::isFinite(pixel);
}

bool
hasData(const scene_rdl2::math::Vec4f &pixel) {
    return scene_rdl2::math::isFinite(pixel);
}

namespace
{

// Passes must follow a strict convention - render a single sample for each pixel
// first, and then start adding more samples to each pixel (if there is more
// than one pass).
#ifdef DEBUG
bool
arePassesValid(const std::vector<Pass> &passes)
{
    MNRY_ASSERT(!passes.empty());
    MNRY_ASSERT(passes[0].isValid());

    size_t passIdx = 1;

    while (passIdx < passes.size()) {

        const Pass &prevPass = passes[passIdx - 1];
        const Pass &currPass = passes[passIdx];

        MNRY_ASSERT(currPass.isValid());

        unsigned state = prevPass.isFinePass() ? 2 : 0;
        state |= currPass.isFinePass() ? 1 : 0;

        switch (state) {
        case 0:
            // prevPass = coarse, currPass = coarse:
            MNRY_ASSERT(currPass.mStartPixelIdx == prevPass.mEndPixelIdx);
            MNRY_ASSERT(prevPass.mStartSampleIdx == 0);
            MNRY_ASSERT(prevPass.mEndSampleIdx == 1);
            MNRY_ASSERT(currPass.mStartSampleIdx == 0);
            MNRY_ASSERT(currPass.mEndSampleIdx == 1);
            break;

        case 1:
            // prevPass = coarse, currPass = fine:
            MNRY_ASSERT(prevPass.mEndPixelIdx == 64);
            MNRY_ASSERT(prevPass.mStartSampleIdx == 0);
            MNRY_ASSERT(prevPass.mEndSampleIdx == 1);
            MNRY_ASSERT(currPass.mStartSampleIdx == 1);
            break;

        case 2:
            // prevPass = fine, currPass = coarse:
            MNRY_ASSERT(0);  // invalid state
            break;

        case 3:
            // prevPass = fine, currPass = fine:
            MNRY_ASSERT(currPass.mStartSampleIdx == prevPass.mEndSampleIdx);
            break;

        default:
            MNRY_ASSERT(0);
        }

        ++passIdx;
    }

    return true;
}
#endif

//
// Returns either:
// - MAX_RENDER_PASSES if all passes are fine passes, or
// - the index of the last coarse pass.
//
unsigned
computeLastCoarsePassIdx(const std::vector<Pass> &passes)
{
    MNRY_ASSERT(arePassesValid(passes));

    // Check if all passes are fine passes.
    if (passes[0].isFinePass()) {
        return MAX_RENDER_PASSES;
    }

    unsigned lastCoarsePassIdx = unsigned(passes.size() - 1);
    for (unsigned i = 1; i < unsigned(passes.size()); ++i) {
        if (passes[i].isFinePass()) {
            lastCoarsePassIdx = i - 1;
            break;
        }
    }

#ifdef DEBUG
    MNRY_ASSERT(passes[lastCoarsePassIdx].isCoarsePass());
    if (lastCoarsePassIdx + 1 < passes.size())
        MNRY_ASSERT(passes[lastCoarsePassIdx + 1].isFinePass());
#endif

    return lastCoarsePassIdx;
}

// Fills in the samplesPerPass array and returns the total samples over all
// passes.
size_t
computeSamplesPerPass(size_t *samplesPerPass,
                      const std::vector<Pass> &passes,
                      const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                      const scene_rdl2::math::Viewport &viewport,
                      bool distributed)
{
    MNRY_ASSERT(!passes.empty());
    MNRY_ASSERT(!tiles.empty());
    MNRY_ASSERT(viewport.mMinX >= 0);
    MNRY_ASSERT(viewport.mMaxX >= viewport.mMinX);
    MNRY_ASSERT(viewport.mMinY >= 0);
    MNRY_ASSERT(viewport.mMaxY >= viewport.mMinY);

    size_t passIdx = 0;
    size_t totalSamples = 0;

    const size_t vpArea = viewport.width() * viewport.height();
    bool vpIsTileAligned = (unsigned(viewport.mMinX)     & 0x07) == 0 &&
                           (unsigned(viewport.mMaxX + 1) & 0x07) == 0 &&
                           (unsigned(viewport.mMinY)     & 0x07) == 0 &&
                           (unsigned(viewport.mMaxY + 1) & 0x07) == 0;

    // First compute samples per pass for all coarse passes.
    while (passIdx < passes.size()) {

        const Pass &pass = passes[passIdx];
        if (pass.isFinePass()) {
            break;
        }

        MNRY_ASSERT(pass.mStartSampleIdx == 0 && pass.mEndSampleIdx == 1);

        if (vpIsTileAligned) {
            // Trivial case, no partial tiles to deal with.
            samplesPerPass[passIdx] = (pass.mEndPixelIdx - pass.mStartPixelIdx) * tiles.size();
        } else {

            // Computing samples per pass can get complicated when the viewport
            // isn't tile aligned since the pixels which get rendered in each
            // pass depend on the pixel fill order defined in the Film class.
            for (size_t itile = 0; itile != tiles.size(); ++itile) {

                const scene_rdl2::fb_util::Tile &tile = tiles[itile];
                if (tile.getArea() == 64) {
                    samplesPerPass[passIdx] += pass.mEndPixelIdx - pass.mStartPixelIdx;
                } else {

                    for (unsigned ipix = pass.mStartPixelIdx; ipix != pass.mEndPixelIdx; ++ipix) {
                        unsigned pixLoc = Film::getPixelFillOrder(itile, ipix);
                        unsigned pixLocX = pixLoc & 7;
                        unsigned pixLocY = pixLoc >> 3;

                        // These are linear coordinates.
                        unsigned px = (tile.mMinX & ~0x07) + pixLocX;
                        unsigned py = (tile.mMinY & ~0x07) + pixLocY;

                        // Check this pixel is inside the viewport.
                        if (viewport.contains(px, py)) {
                            ++samplesPerPass[passIdx];
                        }
                    }
                }
            }
        }

        totalSamples += samplesPerPass[passIdx];

        ++passIdx;
    }

    if (!distributed && passIdx != 0 && passIdx != passes.size()) {
        MNRY_ASSERT(totalSamples == vpArea);
    }

    // Now compute samples per pass for all fine passes.
    if (passIdx < passes.size()) {

        size_t pixelsPerFinePass = vpArea;

        if (distributed) {
            if (vpIsTileAligned) {
                // Fast path.
                pixelsPerFinePass = 64u * tiles.size();
            } else {
                pixelsPerFinePass = 0;
                for (size_t itile = 0; itile != tiles.size(); ++itile) {
                    pixelsPerFinePass += tiles[itile].getArea();
                }
            }
        }

        while (passIdx < passes.size()) {
            const Pass &pass = passes[passIdx];
            MNRY_ASSERT(pass.isFinePass());

            samplesPerPass[passIdx] = (pass.mEndSampleIdx - pass.mStartSampleIdx) * pixelsPerFinePass;
            totalSamples += samplesPerPass[passIdx];

            ++passIdx;
        }
    }

    MNRY_ASSERT(passIdx == passes.size());

    if (!distributed && passes.back().mEndSampleIdx == 64) {
        MNRY_ASSERT(totalSamples == vpArea * passes.back().mEndSampleIdx);
    }

    return totalSamples;
}

// passes[0] is expected to be filled in. This function updates the remaining
// passes and returns the index of the last coarse pass.
unsigned
refineProgressivePasses(std::vector<Pass> *passes, unsigned samplesPerPixel, SamplingMode samplingMode)
{
    MNRY_ASSERT(passes && passes->size() == 1);

    const Pass &initialPass = (*passes)[0];
    MNRY_ASSERT(initialPass.isValid());
    MNRY_ASSERT(initialPass.mStartSampleIdx == 0);
    MNRY_ASSERT(initialPass.mEndSampleIdx <= samplesPerPixel);

    if (initialPass.mEndPixelIdx == 64 &&
        initialPass.mEndSampleIdx == samplesPerPixel) {
        return computeLastCoarsePassIdx(*passes);
    }

    Pass pass;
    pass.mStartPixelIdx = 0;
    pass.mEndPixelIdx = 1;
    pass.mStartSampleIdx = 0;
    pass.mEndSampleIdx = initialPass.mEndSampleIdx;

    if (initialPass.mEndPixelIdx == 1) {
        // coarse, 3 samples per tile
        pass.mStartPixelIdx = 1;
        pass.mEndPixelIdx = 4;
        passes->push_back(pass);
    }

    if (initialPass.mEndPixelIdx <= 4) {
        pass.mStartPixelIdx = passes->back().mEndPixelIdx;
        pass.mEndPixelIdx = 16;
        passes->push_back(pass);
    }

    if (initialPass.mEndPixelIdx <= 16) {
        pass.mStartPixelIdx = passes->back().mEndPixelIdx;
        pass.mEndPixelIdx = 32;
        passes->push_back(pass);
    }

    if (initialPass.mEndPixelIdx != 64) {
        pass.mStartPixelIdx = passes->back().mEndPixelIdx;
        pass.mEndPixelIdx = 64;
        passes->push_back(pass);
    }

    MNRY_ASSERT(passes->back().mEndPixelIdx == 64);

    unsigned currSample = passes->back().mEndSampleIdx;
    MNRY_ASSERT(currSample <= samplesPerPixel);

    unsigned remainingSamples = samplesPerPixel - currSample;
    if (remainingSamples == 0) {
        return computeLastCoarsePassIdx(*passes);
    }

    //
    // Now we've rendered at least a single sample for all pixels,
    // all further passes deal with full tiles of pixels
    //
    pass.mStartPixelIdx = 0;
    pass.mEndPixelIdx = 64;

    if (samplingMode == SamplingMode::ADAPTIVE) {
        unsigned size = 4;
        while (remainingSamples > size && passes->size() < MAX_RENDER_PASSES - 1) {
            pass.mStartSampleIdx = currSample;
            pass.mEndSampleIdx = currSample + size;
            passes->push_back(pass);
            remainingSamples -= size;
            currSample += size;
            size += 2;
        }
    } else {
        // Align currSample to next power of 2.
        unsigned nextPow2Sample = scene_rdl2::util::roundUpToPowerOfTwo(currSample);
        if (nextPow2Sample != currSample) {
            unsigned samplesToAdd = std::min(nextPow2Sample, samplesPerPixel) - currSample;
            MNRY_ASSERT(samplesToAdd);

            pass.mStartSampleIdx = currSample;
            pass.mEndSampleIdx = currSample + samplesToAdd;
            passes->push_back(pass);

            remainingSamples -= samplesToAdd;
            currSample += samplesToAdd;
        }

        // Iterate, adding new passes in powers of 2...
        while (remainingSamples >= currSample && passes->size() < MAX_RENDER_PASSES - 1) {
            pass.mStartSampleIdx = currSample;
            pass.mEndSampleIdx = currSample * 2;
            passes->push_back(pass);
            remainingSamples -= currSample;
            currSample *= 2;
        }
    }

    // ...and handle any remaining samples.
    if (remainingSamples) {
        pass.mStartSampleIdx = currSample;
        pass.mEndSampleIdx = currSample + remainingSamples;
        passes->push_back(pass);
    }

    MNRY_ASSERT(passes->size() <= MAX_RENDER_PASSES);
    MNRY_ASSERT(passes->back().mEndSampleIdx == samplesPerPixel);

    return computeLastCoarsePassIdx(*passes);
}

// Returns the index of the last coarse pass.
unsigned
initPasses(std::vector<Pass> *passes, unsigned samplesPerPixel, // This is maximum samples per pixel over whole image
           RenderMode renderMode, SamplingMode samplingMode)
{
    passes->clear();
    passes->reserve(MAX_RENDER_PASSES);

    Pass pass;
    memset(&pass, 0, sizeof(Pass));

#ifdef FORCE_SINGLE_SAMPLE_PER_PIXEL
    samplesPerPixel = 1;
#endif

    switch (renderMode) {
    case RenderMode::BATCH:
        pass.mStartPixelIdx = 0;
        pass.mEndPixelIdx = 64;
        pass.mStartSampleIdx = 0;
        pass.mEndSampleIdx = samplesPerPixel;
        passes->push_back(pass);
        /* useful debug dump message for debug
        {
            std::cerr << ">> RenderDriver.cc renderMode:BATCH"
                      << " pix(" << pass.mStartPixelIdx << "~" << pass.mEndPixelIdx << ")"
                      << " smp(" << pass.mStartSampleIdx << "~" << pass.mEndSampleIdx << ")" << std::endl;

        }
        */
        break;

    case RenderMode::PROGRESSIVE:
        // The initial configuration just renders a single sample per tile.
        // It gets dynamically refined however based on how many samples we
        // can render in the given frame budget.
        pass.mStartPixelIdx = 0;
        pass.mEndPixelIdx = 1;
        pass.mStartSampleIdx = 0;
        pass.mEndSampleIdx = 1;
        passes->push_back(pass);
        refineProgressivePasses(passes, samplesPerPixel, samplingMode);
        /* useful debug dump message for debug
        {
            for (size_t i = 0; i < passes->size(); ++i) {
                std::cerr << ">> RenderDriver.cc renderMode:PROGRESSIVE i:" << i
                          << " pix(" << (*passes)[i].mStartPixelIdx << '~' << (*passes)[i].mEndPixelIdx
                          << ") smp(" << (*passes)[i].mStartSampleIdx << '~' << (*passes)[i].mEndSampleIdx
                          << ')' << std::endl;
            }
        }
        */
        break;
    case RenderMode::PROGRESSIVE_FAST:
        // For now, do the same things as PROGRESSIVE
        // The initial configuration just renders a single sample per tile.
        // It gets dynamically refined however based on how many samples we
        // can render in the given frame budget.
        pass.mStartPixelIdx = 0;
        pass.mEndPixelIdx = 1;
        pass.mStartSampleIdx = 0;
        pass.mEndSampleIdx = 1;
        passes->push_back(pass);
        refineProgressivePasses(passes, samplesPerPixel, samplingMode);
        break;

    case RenderMode::REALTIME:
    case RenderMode::PROGRESS_CHECKPOINT:
        // The initial configuration just renders a single sample per tile.
        // It gets dynamically refined however based on how many samples we
        // can render in the given frame budget.
        pass.mStartPixelIdx = 0;
        pass.mEndPixelIdx = 1;
        pass.mStartSampleIdx = 0;
        pass.mEndSampleIdx = 1;
        passes->push_back(pass);
        break;

    default:
        MNRY_ASSERT(0);
    }

    MNRY_ASSERT(passes->size() <= MAX_RENDER_PASSES);

    return computeLastCoarsePassIdx(*passes);
}

}   // End of anon namespace.

//-----------------------------------------------------------------------------

RenderDriver::RenderDriver(const TLSInitParams &initParams) :
    mUnalignedW(0),
    mUnalignedH(0),
    mCachedSamplesPerPixel(0),
    mCachedRenderMode(RenderMode::NUM_MODES),
    mCachedRequiresDeepBuffer(false),
    mCachedRequiresCryptomatteBuffer(false),
    mCachedGeneratePixelInfo(false),
    mCachedRequiresHeatMap(false),
    mCachedDeepFormat(0),
    mCachedDeepCurvatureTolerance(0.0),
    mCachedDeepZTolerance(0.0),
    mCachedTargetAdaptiveError(0.0f),
    mCachedDeepVolCompressionRes(0),
    mCachedDeepMaxLayers(0),
    mCachedViewport(scene_rdl2::math::Viewport(0, 0, 0, 0)),
    mCachedSamplingMode(SamplingMode::UNIFORM),
    mCachedDisplayFilterCount(0),
    mFilm(nullptr),
    mLastCoarsePassIdx(0),
    mTileScheduler(nullptr),
    mTileSchedulerCheckpointInitEstimation(nullptr),
    mTaskScheduler(nullptr),
    mPrimaryRaysSubmitted(MAX_RENDER_PASSES, 0),
    mReadyForDisplay(false),
    mCoarsePassesComplete(false),
    mFrameComplete(false),
    mFrameCompleteAtPassBoundary(false),
    mUpdateDurationOffset(0.0),
    mMcrtStartTime(-1.0),
    mMcrtDuration(-1.0),
    mMcrtUtilization(-1.0),
    mFrameEndTime(-1.0),
    mStopAtFrameReadyForDisplay(false),
    mRenderThreadState(),
    mDebugRayState(READY),
    mAdaptiveTileSampleCap(0),
    mLastCheckpointFileEndSampleId(-1),
    mRenderStopAtPassBoundary(false),
    mXPUOcclusionRayQueue(nullptr),
    mXPURayQueue(nullptr),
    mParallelInitFrameUpdate(true),
    mParallelInitFrameUpdateMcrtCount(0),
    mCheckpointEstimationStage(false),
    mRenderPrepTime(10),
    mCheckpointEstimationTime(10)
{
    // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
    memset(&mFs, 0, sizeof(FrameState));

    // Setup TLS parameters.
    TLSInitParams tlsInitParams = initParams;
    MNRY_ASSERT(tlsInitParams.mArenaBlockPool);

    if (tlsInitParams.mDesiredNumTBBThreads == 0) {
        tlsInitParams.mDesiredNumTBBThreads = tbb::task_scheduler_init::default_num_threads();
    }

#ifdef FORCE_SINGLE_THREADED_RENDERING
    tlsInitParams.mDesiredNumTBBThreads = 1;
#endif

    tlsInitParams.initGeomTls = geom::internal::TLState::allocTls;
    tlsInitParams.initPbrTls = pbr::TLState::allocTls;
    tlsInitParams.initShadingTls = shading::TLState::allocTls;
    tlsInitParams.initTLSTextureSupport = shading::initTexturingSupport;

    // There are 2 task_scheduler_init objects created in this class. Both are
    // essential. This first call sets the number of threads for the frame
    // building phase to that specified in the TLSInitParams. The second allows
    // the render thread to take part in the rendering phase and ensures a TLS
    // is created for it.
    mTaskScheduler = new tbb::task_scheduler_init(int(tlsInitParams.mDesiredNumTBBThreads));

    mFilm = alignedMallocCtor<Film>(CACHE_LINE_SIZE);

    if (mRenderThreadState.get() == UNINITIALIZED) {
        MNRY_ASSERT(!mRenderThread.joinable());

        // TLS initialization happens on the renderThread.
        mRenderThread = std::thread(renderThread, this, tlsInitParams);
    }

    parserConfigure();

    // We must wait for the render thread to get initialized before much of this
    // class becomes functional.
    while (mRenderThreadState.get() == UNINITIALIZED) {
        mcrt_common::threadSleep();
    }
}

RenderDriver::~RenderDriver()
{
    stopFrame();

    MNRY_ASSERT(mRenderThreadState.get() == UNINITIALIZED ||
               mRenderThreadState.get() == READY_TO_RENDER);

    // We could probably do acquire/release memory semantics here, but we're in the destructor. Who cares about
    // efficiency?
    mRenderThreadState.set(mRenderThreadState.get(), KILL_RENDER_THREAD);
    if (mRenderThread.joinable()) {
        mRenderThread.join();
    }

    MNRY_ASSERT(mRenderThreadState.get() == UNINITIALIZED || mRenderThreadState.get() == DEAD);

    saveRealtimeStats();

    alignedFreeDtor(mFilm);

    RenderThreadState state = mRenderThreadState.get();
    MNRY_ASSERT_REQUIRE(state == UNINITIALIZED || state == DEAD);

    freeXPUQueues();

    // Terminate task scheduler and wait for tbb worker threads to finish.
    // We need worker threads to exit before the TLS cleanup which is where
    // we cleanup OpenImageIO. Otherwise we get a crash in that library when
    // worker threads attempt to access the static mutex
    // ImageCacheImpl::m_perthread_info_mutex upon cleanup
    // (in ImageCacheImpl::cleanup_perthread_info()).
    // This requires using -D__TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
    // when building this library, and also to make sure that no other
    // tbb::task_scheduler_init or other higher-level task scheduler objects
    // (i.e. tbb::task_group, etc.) are active in the process.
    MNRY_VERIFY(mTaskScheduler)->terminate();
    delete mTaskScheduler;

    cleanUpTLS();
}

const std::vector<scene_rdl2::fb_util::Tile> *
RenderDriver::getTiles() const
{
    if (!mCheckpointEstimationStage) {
        return mTileScheduler ? &mTileScheduler->getTiles() : nullptr;
    } else {
        // returns based on the special tile scheduler for checkpoint estimation stage.
        // This is related to the multi-machine interactive rendering.
        // We are using a time-based checkpoint mode for the multi-machine situation.
        return &mTileSchedulerCheckpointInitEstimation->getTiles();
    }
}

void
RenderDriver::startFrame(const FrameState &fs)
{
    MNRY_ASSERT(mRenderThreadState.get() == READY_TO_RENDER);

    //
    // Update RenderDriver state:
    //

    mFs = fs;

    // This is set to true if either the tiles or passes were updated.
    bool updated = false;

    unsigned w = mFs.mWidth;
    unsigned h = mFs.mHeight;
    std::vector<unsigned int> aovChannels;
    for (const auto &entry: *mFs.mAovSchema) {
        aovChannels.push_back(entry.numChannels());
    }

    const bool needToUpdateBasedOnAdaptiveError = mFs.mSamplingMode == SamplingMode::ADAPTIVE &&
                                                 (mFs.mTargetAdaptiveError != mCachedTargetAdaptiveError);

    if (mUnalignedW != w || mUnalignedH != h ||
        mCachedViewport != mFs.mViewport ||
        mFs.mRequiresDeepBuffer != mCachedRequiresDeepBuffer ||
        mFs.mRequiresCryptomatteBuffer != mCachedRequiresCryptomatteBuffer ||
        mFs.mGeneratePixelInfo != mCachedGeneratePixelInfo ||
        aovChannels != mCachedAovChannels ||
        mFs.mRequiresHeatMap != mCachedRequiresHeatMap ||
        mFs.mDeepFormat != mCachedDeepFormat ||
        mFs.mDeepCurvatureTolerance != mCachedDeepCurvatureTolerance ||
        mFs.mDeepZTolerance != mCachedDeepZTolerance ||
        needToUpdateBasedOnAdaptiveError ||
        mFs.mDeepVolCompressionRes != mCachedDeepVolCompressionRes ||
        *(mFs.mDeepIDChannelNames) != mCachedDeepIDChannelNames ||
        mFs.mDeepMaxLayers != mCachedDeepMaxLayers ||
        mFs.mDeepLayerBias != mCachedDeepLayerBias ||
        mFs.mSamplingMode != mCachedSamplingMode ||
        mFs.mDisplayFilterCount != mCachedDisplayFilterCount) {

        unsigned alignedW = scene_rdl2::util::alignUp(w, COARSE_TILE_SIZE);
        unsigned alignedH = scene_rdl2::util::alignUp(h, COARSE_TILE_SIZE);

        if (alignedW != scene_rdl2::util::alignUp(mExtrapolationBuffer.getWidth(), COARSE_TILE_SIZE) ||
            alignedH != scene_rdl2::util::alignUp(mExtrapolationBuffer.getHeight(), COARSE_TILE_SIZE)) {
            mExtrapolationBuffer.init(alignedW, alignedH);
        }

        MNRY_ASSERT(mFilm);
        uint32_t filmFlags = 0;
        if (mFs.mSamplingMode != SamplingMode::UNIFORM) filmFlags |= Film::USE_ADAPTIVE_SAMPLING;
        if (mFs.mGeneratePixelInfo) filmFlags |= Film::ALLOC_PIXEL_INFO_BUFFER;
        if (mFs.mRequiresHeatMap) filmFlags |= Film::ALLOC_HEAT_MAP_BUFFER;
        if (mFs.mExecutionMode == mcrt_common::ExecutionMode::XPU) filmFlags |= Film::VECTORIZED_XPU;
        else if (mFs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED) filmFlags |= Film::VECTORIZED_CPU;
        if (mFs.mRequiresDeepBuffer) {
            filmFlags |= Film::ALLOC_DEEP_BUFFER;
        }
        if (mFs.mRequiresCryptomatteBuffer) filmFlags |= Film::ALLOC_CRYPTOMATTE_BUFFER;
        if (mFs.mRenderContext->getSceneContext().getResumableOutput()) {
            filmFlags |= Film::RESUMABLE_OUTPUT;
        }
        bool cryptomatteMultiPresence = mFs.mRenderContext->getSceneContext()
                                                  .getSceneVariables()
                                                  .get(scene_rdl2::rdl2::SceneVariables::sCryptomatteMultiPresence);

        mFilm->init(w, h,
                    mFs.mViewport,
                    filmFlags,
                    mFs.mDeepFormat,
                    mFs.mDeepCurvatureTolerance,
                    mFs.mDeepZTolerance,
                    mFs.mDeepVolCompressionRes,
                    *(mFs.mDeepIDChannelNames),
                    mFs.mDeepMaxLayers,
                    mFs.mNumRenderThreads,
                    *mFs.mAovSchema,
                    mFs.mDisplayFilterCount,
                    &mTileExtrapolation,
                    mFs.mMaxSamplesPerPixel,
                    mFs.mTargetAdaptiveError,
                    cryptomatteMultiPresence);

        updated = true;
    }

    new(&mProgressEstimation) RenderProgressEstimation; // for interactive session, we need reset

    // We need to update mProgressEstimation and initialize adaptiveRegions data when
    // we are adaptive sampling.
    if (mFs.mSamplingMode != SamplingMode::UNIFORM) {
        mProgressEstimation.setAdaptiveSampling(true);

        mFilm->getAdaptiveRenderTilesTable()->setTargetError(mFs.mTargetAdaptiveError);

        if (!updated) {
            // We need to reset adaptiveRegions because adaptiveRegions is not initialized under
            // re-render condition. We skip updated = true case because adaptiveRegions already
            // initialized.
            mFilm->initAdaptiveRegions(mFs.mViewport,
                                       mFs.mTargetAdaptiveError,
                                       (mFs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED ||
                                        mFs.mExecutionMode == mcrt_common::ExecutionMode::XPU));
        }
    }

    // Update the tile scheduler if necessary.
    if (!mTileScheduler || (mTileScheduler->getType() != fs.mTileSchedulerType)) {
        mTileScheduler = TileScheduler::create((TileScheduler::Type)fs.mTileSchedulerType);
        if (fs.mTileSchedulerType == TileScheduler::MORTON_SHIFTFLIP) {
            if (fs.mNumRenderNodes == 1) {
                MortonShiftFlipTileScheduler* tileScheduler = (MortonShiftFlipTileScheduler*)(mTileScheduler.get());
                tileScheduler->set(0, 0, false, false); // same as normal MortonTileScheduler
            } else {
                setupMultiMachineTileScheduler(w, h);
            }
        }
        updated = true;
    }

    if (mFs.mRenderMode == RenderMode::PROGRESS_CHECKPOINT &&
        mTileSchedulerCheckpointInitEstimation == nullptr) {
        // We have to generate special tileScheduler for checkpoint mode
        mTileSchedulerCheckpointInitEstimation = TileScheduler::create(TileScheduler::Type::RANDOM);
    }

    // Update tiles as needed.
    if (updated ||
        mCachedViewport != mFs.mViewport ||
        mTileScheduler->taskDistribType() != mFs.mTaskDistributionType) {
        MNRY_ASSERT(mTileScheduler);
        mTileScheduler->generateTiles(&getGuiTLS()->mArena, w, h, mFs.mViewport,
                                      mFs.mRenderNodeIdx, mFs.mNumRenderNodes,
                                      mFs.mTaskDistributionType);

#ifdef DEBUG
        if (!mTileScheduler->isDistributed()) {
            unsigned numTilesX = ((mFs.mViewport.mMaxX) >> 3) - ((mFs.mViewport.mMinX) >> 3) + 1;
            unsigned numTilesY = ((mFs.mViewport.mMaxY) >> 3) - ((mFs.mViewport.mMinY) >> 3) + 1;
            unsigned numTiles = numTilesX * numTilesY;
            MNRY_ASSERT(numTiles == unsigned(mTileScheduler->getTiles().size()));
        }
#endif

        if (mTileSchedulerCheckpointInitEstimation != nullptr) {
            mTileSchedulerCheckpointInitEstimation->
                generateTiles(&getGuiTLS()->mArena, w, h, mFs.mViewport,
                              mFs.mRenderNodeIdx, mFs.mNumRenderNodes, mFs.mTaskDistributionType);
        }

        // Each pbr::TLState keeps track of which tiles were rendered to so make
        // sure they have sufficient buffers allocated for this.
        unsigned numTiles = mTileScheduler->getTiles().size();
        pbr::forEachTLS([&](pbr::TLState *tls) {
            tls->mTilesRenderedTo.init(numTiles);
        });

        updated = true;
    }

    // Initialize DisplayFilterDriver after Film, RenderOutputDriver, and TileScheduler have been initialized.
    mDisplayFilterDriver.init(mFilm, mFs.mRenderContext->getRenderOutputDriver(),
                              getTiles(), mTileScheduler->getTileIndices(), mFs.mViewport, getNumTBBThreads());


    if (mFs.mNumRenderNodes <= 1) {
        mParallelInitFrameUpdate = false; // disable parallel init frame update when single moonray context
        mMultiMachineCheckpointMainLoop = false;
    } else {
        mParallelInitFrameUpdate = true; // enable parallel init frame update for multiple moonray context
        if (!mParallelInitFrameUpdateMcrtCount) {
            // Initialize parameters for parallel-initial-frame update.
            // Regarding the phase of 1 sample per tile, this number of hosts divides the entire tiles and
            // computes them in parallel.
            mParallelInitFrameUpdateMcrtCount = mFs.mNumRenderNodes;
        }
        mMultiMachineCheckpointMainLoop = true;
    }

    std::vector<Pass> passes;
    mLastCoarsePassIdx = initPasses(&passes, mFs.mMaxSamplesPerPixel, mFs.mRenderMode, mFs.mSamplingMode);

    // Reset progress state.
    mReadyForDisplay = false;
    mCoarsePassesComplete = false;
    mFrameComplete = false;
    mFrameCompleteAtPassBoundary = false;
    std::fill(mPrimaryRaysSubmitted.begin(), mPrimaryRaysSubmitted.end(), 0);

    // Even if the frame end time already exceeds frame start time, we still
    // should render the coarsest possible pass regardless of render mode.
    const float frameBudget = 1.f / fs.mFps;
    mFrameEndTime = fs.mFrameStartTime + double(frameBudget);
    gCancelFlag.set(false);
    mStopAtFrameReadyForDisplay = false;
    mRenderStopAtPassBoundary = false;

    if (updated ||
        mCachedSamplesPerPixel != mFs.mMaxSamplesPerPixel ||
        mCachedRenderMode != mFs.mRenderMode ||
        mFs.mRenderMode == RenderMode::REALTIME || mFs.mRenderMode == RenderMode::PROGRESS_CHECKPOINT) {

        MNRY_ASSERT(arePassesValid(passes));

        // Compute the relative amount of samples each pass contains, normalized
        // to sum to 1.
        memset(mSamplesPerPass, 0, sizeof(mSamplesPerPass));

        if (mFs.mRenderMode != RenderMode::REALTIME && mFs.mRenderMode != RenderMode::PROGRESS_CHECKPOINT) {
            computeSamplesPerPass(mSamplesPerPass, passes,
                                  mTileScheduler->getTiles(),
                                  mFs.mViewport,
                                  mTileScheduler->isDistributed());
        }

        // Initialize the work queue. This will get dynamically refined later
        // in the frame for the realtime/progressCheckpoint render mode.
        mTileWorkQueue.init(mFs.mRenderMode,
                            unsigned(mTileScheduler->getTiles().size()),
                            unsigned(passes.size()),
                            getNumTBBThreads(),
                            &passes.front());
    }

    // Reset realtime stats.
    RealtimeFrameStats &rfs = getCurrentRealtimeFrameStats();

    // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
    memset(&rfs, 0, sizeof(RealtimeFrameStats));

    { // setup post checkpoint lua script name and checkpoint parameters
        using SceneVariables = scene_rdl2::rdl2::SceneVariables;
        const SceneVariables &vars = mFs.mRenderContext->getSceneContext().getSceneVariables();
        mCheckpointPostScript = vars.get(SceneVariables::sCheckpointPostScript);
        mCheckpointController.set(vars.get(SceneVariables::sCheckpointSnapshotInterval),
                                  vars.get(SceneVariables::sCheckpointMaxSnapshotOverhead));
    }

    // Update all cached data.
    mUnalignedW = w;
    mUnalignedH = h;
    mCachedSamplesPerPixel = mFs.mMaxSamplesPerPixel;
    mCachedRenderMode = mFs.mRenderMode;
    mCachedRequiresDeepBuffer = mFs.mRequiresDeepBuffer;
    mCachedRequiresCryptomatteBuffer = mFs.mRequiresCryptomatteBuffer;
    mCachedGeneratePixelInfo = mFs.mGeneratePixelInfo;
    mCachedAovChannels = std::move(aovChannels);
    mCachedRequiresHeatMap = mFs.mRequiresHeatMap;
    mCachedDeepFormat = mFs.mDeepFormat;
    mCachedDeepCurvatureTolerance = mFs.mDeepCurvatureTolerance;
    mCachedDeepZTolerance = mFs.mDeepZTolerance;
    mCachedTargetAdaptiveError = mFs.mTargetAdaptiveError;
    mCachedDeepVolCompressionRes = mFs.mDeepVolCompressionRes;
    mCachedDeepIDChannelNames = *(mFs.mDeepIDChannelNames);
    mCachedDeepMaxLayers = mFs.mDeepMaxLayers;
    mCachedDeepLayerBias = mFs.mDeepLayerBias;
    mCachedViewport = mFs.mViewport;
    mCachedSamplingMode = mFs.mSamplingMode;
    mCachedDisplayFilterCount = mFs.mDisplayFilterCount;

    // Kick off the frame.
    mRenderThreadState.set(READY_TO_RENDER, REQUEST_RENDER, std::memory_order_release);
}

#ifdef MULTI_MACHINE_TILE_SCHEDULE_RANDOM_SHIFT
void
RenderDriver::setupMultiMachineTileScheduler(unsigned pixW, unsigned pixH)
{
    int flipId = mFs.mRenderNodeIdx % 4;

    // flipId flipX flipY
    //    0   false false
    //    1   true  false
    //    2   false true
    //    3   true  true
    bool flipX = (flipId & 0x1) != 0x0;
    bool flipY = (flipId & 0x2) != 0x0;
                
    auto randIntRange = [](int maxVal) {
        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::uniform_int_distribution<> randRange(0, maxVal);
        return randRange(mt);
    };

    int numTilesX = ((pixW + 7) & ~7) >> 3;
    int numTilesY = ((pixH + 7) & ~7) >> 3;
    unsigned shiftX = (unsigned)randIntRange(numTilesX - 1);
    unsigned shiftY = (unsigned)randIntRange(numTilesY - 1);

    MortonShiftFlipTileScheduler* tileScheduler = (MortonShiftFlipTileScheduler*)(mTileScheduler.get());
    tileScheduler->set(shiftX, shiftY, flipX, flipY);

    std::cerr << ">> RenderDriver.cc setupMultiMachineTileScheduler() random_shift"
              << " shiftX:" << shiftX
              << " shiftY:" << shiftY
              << " flipX:" << scene_rdl2::str_util::boolStr(flipX)
              << " flipY:" << scene_rdl2::str_util::boolStr(flipY) << '\n';
}
#endif // end MULTI_MACHINE_TILE_SCHEDULE_RANDOM_SHIFT

#ifdef MULTI_MACHINE_TILE_SCHEDULE_DETERMINISTIC_SHIFT
void
RenderDriver::setupMultiMachineTileScheduler(unsigned pixW, unsigned pixH)
{
    int flipId = mFs.mRenderNodeIdx % 4;

    // flipId flipX flipY
    //    0   false false
    //    1   true  false
    //    2   false true
    //    3   true  true
    bool flipX = (flipId & 0x1) != 0x0;
    bool flipY = (flipId & 0x2) != 0x0;
                
    unsigned numRenderNodes = mFs.mNumRenderNodes;
    unsigned halfNumRenderNodes = numRenderNodes / 2;

    unsigned shiftX {0};
    unsigned shiftY {0};
    if (mFs.mRenderNodeIdx <= halfNumRenderNodes - 1) {
        // shift x direction
        unsigned shiftId = mFs.mRenderNodeIdx;
        unsigned numTilesX = ((pixW + 7) & ~7) >> 3;
        unsigned stepShift = numTilesX / halfNumRenderNodes;

        shiftX = stepShift * shiftId;
    } else {
        // shift y direction
        unsigned shiftId = mFs.mRenderNodeIdx - halfNumRenderNodes;
        unsigned numTilesY = ((pixH + 7) & ~7) >> 3;
        unsigned stepShift = numTilesY / (numRenderNodes - halfNumRenderNodes);

        shiftY = stepShift * shiftId;
    }

    MortonShiftFlipTileScheduler* tileScheduler = (MortonShiftFlipTileScheduler*)(mTileScheduler.get());
    tileScheduler->set(shiftX, shiftY, flipX, flipY);

    std::cerr << ">> RenderDriver.cc setupMultiMachineTileScheduler() deterministic_shift"
              << " shiftX:" << shiftX
              << " shiftY:" << shiftY
              << " flipX:" << scene_rdl2::str_util::boolStr(flipX)
              << " flipY:" << scene_rdl2::str_util::boolStr(flipY) << '\n';
}
#endif // end MULTI_MACHINE_TILE_SCHEDULE_DETERMINISTIC_SHIFT

void
RenderDriver::requestStop()
{
    MNRY_ASSERT(mRenderThreadState.get() == REQUEST_RENDER ||
               mRenderThreadState.get() == RENDERING ||
               mRenderThreadState.get() == RENDERING_DONE);

    mFrameEndTime = scene_rdl2::util::getSeconds();
    requestStopAsyncSignalSafe();
}

void
RenderDriver::requestStopAsyncSignalSafe()
{
    gCancelFlag.set(true);
}

void
RenderDriver::requestStopAtFrameReadyForDisplay()
{
    // If this condition is true, progressive/checkpointResume rendering logic stops after
    // initial render pass and avoids useless execution of following render pass.
    // This is important to achieve fast interactive response under arras context.
    mStopAtFrameReadyForDisplay = true;
}

void
RenderDriver::stopFrame()
{
    if (mRenderThreadState.get() != REQUEST_RENDER &&
        mRenderThreadState.get() != RENDERING &&
        mRenderThreadState.get() != RENDERING_DONE) {
        return;
    }

    mFrameEndTime = scene_rdl2::util::getSeconds();
    gCancelFlag.set(true);

    // Wait on the cancellation to propagate.
    mRenderThreadState.wait(RENDERING_DONE);

    MNRY_ASSERT(mMcrtStartTime > 0.0);
    MNRY_ASSERT(mMcrtDuration > 0.0);

    // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
    memset(&mFs, 0, sizeof(FrameState));

    mRenderThreadState.set(RENDERING_DONE, READY_TO_RENDER, std::memory_order_release);
}

float
RenderDriver::getPassProgressPercentage(unsigned passIdx,
                                        size_t *submitted,
                                        size_t *total) const
{
    MNRY_ASSERT(passIdx < mTileWorkQueue.getNumPasses());

    size_t samplesSubmitted = mPrimaryRaysSubmitted[passIdx];
    size_t samplesPerPass = mSamplesPerPass[passIdx];

    if (samplesPerPass < samplesSubmitted) {
        samplesSubmitted = samplesPerPass;
    }

    if (submitted) {
        *submitted = samplesSubmitted;
    }

    if (total) {
        *total = samplesPerPass;
    }

    return samplesPerPass ? float((double(samplesSubmitted) / double(samplesPerPass)) * 100.0) : 0.f;
}

float
RenderDriver::getOverallProgressFraction(bool activeRendering, size_t *submitted, size_t *total) const
{
    // For the adaptive case, report the number of tiles complete, which
    // inherently takes into account any adaptive sampling settings.
    if (mFilm) {
        if (mFilm->isAdaptive()) {
            // adaptive sampling (checkpoint, non-checkpoint, resume) case
            return mFilm->getProgressFraction(activeRendering, submitted, total);
        }
    }

    size_t totalSubmitted = 0;
    size_t totalForAllPasses = 0;

    float progressFraction = 0.0f;
    if (mProgressEstimation.getSamplesTotal() > 0) {
        // checkpoint/resume render uniform sampling case
        size_t startSamples = mProgressEstimation.getStartUniformSamples();
        totalSubmitted = mProgressEstimation.getSamplesTotal();
        totalForAllPasses = mUnalignedW * mUnalignedH * mCachedSamplesPerPixel;
        if (totalForAllPasses < (totalSubmitted + startSamples)) { // just in caase
            size_t overflow = totalSubmitted + startSamples - totalForAllPasses;
            totalSubmitted = totalForAllPasses - startSamples - overflow;
        }
        if (totalForAllPasses > startSamples) {
            progressFraction = (float)totalSubmitted / (float)(totalForAllPasses - startSamples);
        }

    } else {
        // non checkpoint with non adaptive sampling case
        for (unsigned i = 0; i < mTileWorkQueue.getNumPasses(); ++i) {
            size_t submittedForThisPass, totalForThisPass;
            getPassProgressPercentage(i, &submittedForThisPass, &totalForThisPass);
            totalSubmitted += submittedForThisPass;
            totalForAllPasses += totalForThisPass;
        }
        if (totalForAllPasses < totalSubmitted) {
            totalSubmitted = totalForAllPasses;
        }
        progressFraction =
            totalForAllPasses ? float((double(totalSubmitted) / double(totalForAllPasses))) : 0.f;
    }

    if (submitted) {
        *submitted = totalSubmitted;
    }

    if (total) {
        *total = totalForAllPasses;
    }

    return progressFraction;
}

void
RenderDriver::setMultiMachineGlobalProgressFraction(float fraction)
{
    mMultiMachineGlobalProgressFraction = fraction;
}

RealtimeFrameStats &
RenderDriver::getCurrentRealtimeFrameStats()
{
    static RealtimeFrameStats rfs;
    return rfs;
}

void
RenderDriver::commitCurrentRealtimeStats()
{
#ifdef LOG_REALTIME_FRAME_STATS_TO_FILE

    RealtimeFrameStats &rfs = getCurrentRealtimeFrameStats();

    MNRY_ASSERT_REQUIRE(rfs.mRenderFrameStartTime <= rfs.mPredictedEndTime);
    MNRY_ASSERT_REQUIRE(rfs.mRenderFrameStartTime < rfs.mActualEndTime);
    MNRY_ASSERT_REQUIRE(rfs.mNumRenderPasses);
    MNRY_ASSERT_REQUIRE(rfs.mSamplesPerTile);
    MNRY_ASSERT_REQUIRE(rfs.mSamplesAll);

    if (mRealtimeStats.empty()) {
        mRealtimeStats.reserve(MAX_REALTIME_FRAME_STATS_TO_RECORD);
    }

    if (mRealtimeStats.size() < MAX_REALTIME_FRAME_STATS_TO_RECORD) {
        mRealtimeStats.push_back(rfs);
    }

    // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
    memset(&rfs, 0, sizeof(RealtimeFrameStats));

#endif  // LOG_REALTIME_FRAME_STATS_TO_FILE
}

void
RenderDriver::resetRealtimeStats()
{
    mRealtimeStats.clear();
}

void
RenderDriver::saveRealtimeStats()
{
    if (mCachedRenderMode == RenderMode::REALTIME && !mRealtimeStats.empty()) {

        // So new multiple saves from the same session don't overwrite each other.
        static int saveIdx = 0;

        unsigned renderNodeIdx = mTileScheduler->getRenderNodeIdx();

        static char filename[PATH_MAX];
        sprintf(filename, REALTIME_FRAME_STATS_LOGFILE, (int)renderNodeIdx, saveIdx);

        MOONRAY_THREADSAFE_STATIC_WRITE(++saveIdx);

        FILE *file = fopen(filename, "w");

        if (file) {

            printf("Writing realtime frame stats to file \"%s\".\n", filename);

            const char *header = "Update duration (ms),"
                                 "duration offset (ms),"
                                 "Render budget (ms),"
                                 "Render frame start,"
                                 "First pass start,"
                                 "First pass end,"
                                 "Predicted sample cost,"
                                 "Actual sample cost (ms),"
                                 "Predicted end,"
                                 "Actual end,"
                                 "Overhead duration (ms),"
                                 "Over run (ms),"
                                 "Num render passes,"
                                 "Samples per tile,"
                                 "Samples All\n";

            fwrite(header, 1, strlen(header), file);

            const double baseTime = mRealtimeStats[0].mRenderFrameStartTime;

            char workBuf[512];
            for (size_t i = 0; i < mRealtimeStats.size(); ++i) {

                const RealtimeFrameStats &stats = mRealtimeStats[i];

                sprintf(workBuf, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d, %d %d\n",
                        stats.mUpdateDuration * 1000.0,
                        stats.mUpdateDurationOffset * 1000.0,
                        stats.mRenderBudget * 1000.0,
                        stats.mRenderFrameStartTime - baseTime,
                        stats.mFirstPassStartTime - baseTime,
                        stats.mFirstPassEndTime - baseTime,
                        stats.mPredictedSampleCost * 1000.0,
                        stats.mActualSampleCost,
                        stats.mPredictedEndTime - baseTime,
                        stats.mActualEndTime - baseTime,
                        stats.mOverheadDuration * 1000.0,
                        (stats.mActualEndTime - stats.mPredictedEndTime) * 1000.0,
                        (int)stats.mNumRenderPasses,
                        (int)stats.mSamplesPerTile,
                        (int)stats.mSamplesAll);

                fwrite(workBuf, 1, strlen(workBuf), file);
            }

            fclose(file);

            printf("Done!\n");

        } else {
            printf("Error opening \"%s\" for writing!\n", filename);
        }

        resetRealtimeStats();
    }
}

void
RenderDriver::switchDebugRayState(DebugRayState oldState, DebugRayState newState)
{
    // If this assert triggers we either have:
    //  (a) a code bug where the caller is passing in the wrong state, or
    //  (b) multiple threads are trying to set the same value for newState
    //      which we disallow
    const auto old [[gnu::unused]] = mDebugRayState.exchange(newState, std::memory_order_relaxed);
    MNRY_ASSERT(old == oldState);
}

void
RenderDriver::snapshotRenderBuffer(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                                   bool untile, bool parallel) const
{
    snapshotRenderBufferSub(outputBuffer, untile, parallel, false);
}

void
RenderDriver::snapshotRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                                      bool untile, bool parallel) const
{
    snapshotRenderBufferSub(outputBuffer, untile, parallel, true);
}

void
RenderDriver::snapshotRenderBufferSub(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                                      bool untile, bool parallel, bool oddBuffer) const
{
    std::lock_guard<std::mutex> lock(mExtrapolationBufferMutex);

    MNRY_ASSERT(outputBuffer);

    if (untile) {
        outputBuffer->init(mUnalignedW, mUnalignedH);
    } else {
        unsigned alignedW = scene_rdl2::util::alignUp(mUnalignedW, COARSE_TILE_SIZE);
        unsigned alignedH = scene_rdl2::util::alignUp(mUnalignedH, COARSE_TILE_SIZE);
        outputBuffer->init(alignedW, alignedH);
    }

    const bool extrapolate = !mCoarsePassesComplete;

    const scene_rdl2::fb_util::RenderBuffer *srcBuffer = nullptr;
    if (!oddBuffer) {
        srcBuffer = &(mFilm->getRenderBuffer());
    } else {
        if (!(srcBuffer = mFilm->getRenderBufferOdd())) {
            return; // We don't have renderBufferOdd when it's non adaptive sampling situation -> skip snapshot
        }
    }
    scene_rdl2::fb_util::RenderBuffer *normalizedBuffer = untile ? &mExtrapolationBuffer : outputBuffer;

    // Check if we still require an extrapolation pass. If we do go through the
    // extrapolation code path, the resultant buffer will be normalized also.
    if (extrapolate) {

        const bool viewportActive = (mCachedViewport.mMinX != 0 ||
                                     mCachedViewport.mMaxX != mUnalignedW - 1 ||
                                     mCachedViewport.mMinY != 0 ||
                                     mCachedViewport.mMaxY != mUnalignedH - 1);

        const bool distributed = mTileScheduler->isDistributed();

        // If we have a viewport set which is smaller than the render buffer
        // then clear the extrapolation buffer beforehand since not all of it
        // may get written to in the ExtrapolateBuffer call.
        if (viewportActive || distributed) {

            normalizedBuffer->clear();   // mlee: Is this necessary?

            if (distributed) {
                // Only extrapolate tiles in the list.
                mFilm->extrapolateRenderBufferWithTileList(srcBuffer,
                                                          normalizedBuffer,
                                                          mTileScheduler->getTiles(),
                                                          parallel);
            } else {
                MNRY_ASSERT(viewportActive);

                // Synthesize each tile and clip them to the viewport.
                mFilm->extrapolateRenderBufferWithViewport(srcBuffer,
                                                          normalizedBuffer,
                                                          mCachedViewport,
                                                          parallel);
            }
        } else {
            // Fast path.
            mFilm->extrapolateRenderBufferFastPath(srcBuffer, normalizedBuffer, parallel);
        }
    } else {
        // No extrapolation needed, just normalize the render buffer.
        // If untile is true, use the intermediate buffer so we can
        // later untile directly into the output buffer.
        mFilm->normalizeRenderBuffer(srcBuffer, normalizedBuffer, parallel);
    }

    if (untile) {
#if 1
        scene_rdl2::fb_util::untile(outputBuffer, mExtrapolationBuffer, mFilm->getTiler(), parallel,
                        [](const scene_rdl2::fb_util::RenderColor &pixel, unsigned) -> const scene_rdl2::fb_util::RenderColor & {
            return pixel;
        });
#else
        // Hack to debug alpha channel.
        scene_rdl2::fb_util::untile(outputBuffer, mExtrapolationBuffer, film.getTiler(), parallel,
                        [](const scene_rdl2::fb_util::RenderColor &pixel, unsigned) -> scene_rdl2::fb_util::RenderColor {
            return RenderColor(pixel.w);
        });
#endif
    }
}

void
RenderDriver::snapshotWeightBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                                   bool untile,
                                   bool parallel) const
{
    MNRY_ASSERT(outputBuffer);

    const scene_rdl2::fb_util::FloatBuffer *weightBuffer = &getFilm().getWeightBuffer();

    outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT, weightBuffer->getWidth(), weightBuffer->getHeight());

    snapshotBuffer(&outputBuffer->getFloatBuffer(),
                   *weightBuffer,
                   (scene_rdl2::fb_util::FloatBuffer *)nullptr,
                   false,
                   untile,
                   parallel,
                   [](const float p, unsigned ofs) -> uint32_t { return 0xffffffffu; },
                   [](const float p, unsigned ofs) -> float { return p; });
}

void
RenderDriver::snapshotWeightBuffer(scene_rdl2::fb_util::FloatBuffer *outputBuffer,
                                   bool untile,
                                   bool parallel) const
{
    MNRY_ASSERT(outputBuffer);

    const scene_rdl2::fb_util::FloatBuffer *weightBuffer = &getFilm().getWeightBuffer();

    outputBuffer->init(weightBuffer->getWidth(), weightBuffer->getHeight());

    snapshotBuffer(outputBuffer,
                   *weightBuffer,
                   (scene_rdl2::fb_util::FloatBuffer *)nullptr,
                   false,
                   untile,
                   parallel,
                   [](const float p, unsigned ofs) -> uint32_t { return 0xffffffffu; },
                   [](const float p, unsigned ofs) -> float { return p; });
}

const pbr::DeepBuffer*
RenderDriver::getDeepBuffer() const
{
    return getFilm().getDeepBuffer();
}

pbr::CryptomatteBuffer*
RenderDriver::getCryptomatteBuffer()
{
    return getFilm().getCryptomatteBuffer();
}

const pbr::CryptomatteBuffer*
RenderDriver::getCryptomatteBuffer() const
{
    return getFilm().getCryptomatteBuffer();
}

bool
RenderDriver::snapshotPixelInfoBuffer(scene_rdl2::fb_util::PixelInfoBuffer *outputBuffer,
                                      bool untile,
                                      bool parallel) const
{
    // can only snapshot primary camera for now
    const scene_rdl2::fb_util::PixelInfoBuffer *pixelInfoBuffer = getFilm().getPixelInfoBuffer();

    if (pixelInfoBuffer) {
        std::lock_guard<std::mutex> lock(mExtrapolationBufferMutex);

        snapshotBuffer(outputBuffer,
                       *pixelInfoBuffer,
                       reinterpret_cast<scene_rdl2::fb_util::PixelInfoBuffer *>(&mExtrapolationBuffer),
                       areCoarsePassesComplete() ? false : true,
                       untile,
                       parallel,
                       [](const scene_rdl2::fb_util::PixelInfo &p, unsigned) {
                           return p.depth == FLT_MAX ? 0u : 0xffffffffu;
                       },
                       [](const scene_rdl2::fb_util::PixelInfo &p, unsigned) { return p; });
        return true;
    }
    return false;
}

bool
RenderDriver::snapshotHeatMapBuffer(scene_rdl2::fb_util::HeatMapBuffer *outputBuffer,
                                    bool untile,
                                    bool parallel) const
{
    const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer = getFilm().getHeatMapBuffer();

    if (heatMapBuffer) {
        snapshotBuffer(outputBuffer,
                       *heatMapBuffer,
                       static_cast<scene_rdl2::fb_util::HeatMapBuffer *>(nullptr),
                       false, /* don't extrapolate per-pixel stat */
                       untile,
                       parallel,
                       [](const int64_t &p, unsigned ofs) -> uint32_t { return 0xffffffff; },
                       [](const int64_t &p, unsigned ofs) -> const int64_t & { return p; });
        return true;
    }
    return false;
}

// Not a general purpose snapshot for variable pixel buffers.
// This has some AOV specific logic internally.
template<typename DST_PIXEL_TYPE, typename SRC_PIXEL_TYPE>
static void
snapshotVariablePixelBuffer(const RenderDriver *renderDriver,
                            scene_rdl2::fb_util::PixelBuffer<DST_PIXEL_TYPE> *dst,
                            const scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> &src,
                            scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> *scratchBuffer,
                            bool extrapolate,
                            bool untile,
                            bool parallel,
                            const float *weights,
                            int numConsistentSamples,
                            pbr::AovFilter filter)
{
    // closest filtering requires a special case.
    // see snapshotAovBuffer
    MNRY_ASSERT(filter != pbr::AovFilter::AOV_FILTER_CLOSEST);

    if (filter == pbr::AovFilter::AOV_FILTER_AVG) {

        // Final value is scaled by weight here.
        renderDriver->snapshotBuffer(dst, src, scratchBuffer, extrapolate, untile, parallel,
                                     // Assume that the weight buffer and the aov
                                     // buffers are the same size and use the same
                                     // tiling.  So even though ofs is an offset into
                                     // the src aov buffer, it is safe and correct to
                                     // apply ofs to the weight buffer.
                                     [weights](const SRC_PIXEL_TYPE &, unsigned ofs) -> uint32_t {
                                         const float w = weights[ofs];
                                         return w > 0.f? 0xffffffffu : 0u;
                                     },
                                     [weights](const SRC_PIXEL_TYPE &pixel, unsigned ofs) {
                                         const float w = weights[ofs];
                                         // in case of +inf or -inf, we want to return +inf or -inf
                                         return DST_PIXEL_TYPE((w > 0.f && hasData(pixel)) ?
                                                               pixel / w : pixel);
                                     });

    } else if (filter == pbr::AovFilter::AOV_FILTER_FORCE_CONSISTENT_SAMPLING) {

        // Code path for AOV_FILTER_FORCE_CONSISTENT_SAMPLING
        // The weight used is the number of consistent samples.
        float weight = (float)numConsistentSamples;
        renderDriver->snapshotBuffer(dst, src, scratchBuffer, extrapolate, untile, parallel,
                                     [](const SRC_PIXEL_TYPE &, unsigned ofs) -> uint32_t {
                                         return 0xffffffffu;
                                     },
                                     [weight](const SRC_PIXEL_TYPE &pixel, unsigned ofs) {
                                         // in case of +inf or -inf, we want to return +inf or -inf
                                         return DST_PIXEL_TYPE(hasData(pixel) ?
                                                               pixel / weight : pixel);
                                     });

    } else {

        // Code path for AOV_FILTER_SUM / AOV_FILTER_MIN / AOV_FILTER_MAX.
        // Final value isn't scaled by weight here. We still lookup weight to
        // check pixel validity however.
        renderDriver->snapshotBuffer(dst, src, scratchBuffer, extrapolate, untile, parallel,
                                     [weights](const SRC_PIXEL_TYPE &, unsigned ofs) -> uint32_t {
                                         const float w = weights[ofs];
                                         return w > 0.f? 0xffffffffu : 0u;
                                     },
                                     [](const SRC_PIXEL_TYPE &pixel, unsigned ofs) {
                                         return DST_PIXEL_TYPE(pixel);
                                     });
    }
}

template<typename DST_PIXEL_TYPE, typename SRC_PIXEL_TYPE>
static void
snapshotDisplayFilterVariablePixelBuffer(const RenderDriver *renderDriver,
                                         scene_rdl2::fb_util::PixelBuffer<DST_PIXEL_TYPE> *dst,
                                         const scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> &src,
                                         scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> *scratchBuffer,
                                         bool extrapolate,
                                         bool untile,
                                         bool parallel)
{
    renderDriver->snapshotBuffer(dst, src, scratchBuffer, extrapolate, untile, parallel,
                                    [](const SRC_PIXEL_TYPE &, unsigned ofs) -> uint32_t {
                                        return 0xffffffffu;
                                    },
                                    [](const SRC_PIXEL_TYPE &pixel, unsigned ofs) {
                                        return DST_PIXEL_TYPE(pixel);
                                    });
}

template <typename SRC_PIXEL_TYPE>
static void
denormalizeVariablePixelBuffer(const RenderDriver *renderDriver,
                               scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> &buff,
                               const float *weightBuffer,
                               int numConsistentSamples,
                               pbr::AovFilter filter)
{
    if (filter == pbr::AovFilter::AOV_FILTER_AVG) {
        // average AOV filter : We should do denormalize by weight
        renderDriver->crawlAllTiledPixels([&](unsigned pixOffset) {
                SRC_PIXEL_TYPE &pix = *(buff.getData() + pixOffset);
                if (hasData(pix)) { pix *= weightBuffer[pixOffset]; }
            });

    } else if (filter == pbr::AovFilter::AOV_FILTER_FORCE_CONSISTENT_SAMPLING) {
        // Code path for AOV_FILTER_FORCE_CONSISTENT_SAMPLING
        float weight = (float)numConsistentSamples;
        renderDriver->crawlAllTiledPixels([&](unsigned pixOffset) {
                SRC_PIXEL_TYPE &pix = *(buff.getData() + pixOffset);
                if (hasData(pix)) { pix *= weight; }
            });
    } else {
        // Code path for AOV_FILTER_SUM / AOV_FILTER_MIN / AOV_FILTER_MAX
        // We don't need any denormalize operation for this AOV filter type
    }
}

template <typename SRC_PIXEL_TYPE>
static void
zeroWeightMaskVariablePixelBuffer(const RenderDriver *renderDriver,
                                  scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> &buff,
                                  const float *weightBuff)
{
    renderDriver->crawlAllTiledPixels([&](unsigned pixOffset) {
            if (weightBuff[pixOffset] == 0.0f) {
                SRC_PIXEL_TYPE *pix = buff.getData() + pixOffset;
                memset((void *)pix, 0x0, sizeof(SRC_PIXEL_TYPE));
            }
        });
}

void
RenderDriver::snapshotDisplayFilterBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                                          unsigned int dfIdx,
                                          bool untile,
                                          bool parallel) const
{
    std::lock_guard<std::mutex> lock(mExtrapolationBufferMutex);

    const Film &film                                        = getFilm();
    const scene_rdl2::fb_util::VariablePixelBuffer &displayFilterBuffer = film.getDisplayFilterBuffer(dfIdx);
    const scene_rdl2::fb_util::VariablePixelBuffer::Format format       = displayFilterBuffer.getFormat();
    const bool extrapolate                                  = false; // There is no coarse pass for display filters yet.

    // TODO: Accommodate other pixel buffer formats for DisplayFilters.
    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
        outputBuffer->init(format, displayFilterBuffer.getWidth(), displayFilterBuffer.getHeight());
        snapshotDisplayFilterVariablePixelBuffer(this,
                                                 &outputBuffer->getFloat3Buffer(),
                                                 displayFilterBuffer.getFloat3Buffer(),
                                                 reinterpret_cast<scene_rdl2::fb_util::Float3Buffer *>(&mExtrapolationBuffer),
                                                 extrapolate,
                                                 untile,
                                                 parallel);
        break;
    default:
        MNRY_ASSERT(0 && "unexpected DisplayFilter buffer format");
    }
}

void
RenderDriver::snapshotAovBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                                int numConsistentSamples,
                                unsigned int aov,
                                bool untile,
                                bool parallel,
                                bool fulldump) const
{
    std::lock_guard<std::mutex> lock(mExtrapolationBufferMutex);

    const Film &film                     = getFilm();
    const float *weights                 = film.getWeightBuffer().getData();
    const scene_rdl2::fb_util::VariablePixelBuffer &aovBuffer = film.getAovBuffer(aov);
    const pbr::AovFilter filter          = film.getAovBufferFilter(aov);
    const auto format                    = aovBuffer.getFormat();
    const bool extrapolate               = !areCoarsePassesComplete();

    // aovBuffer is not variance of visibility AOV buffer. (See RenderContext::snapshotAovBuffers())

    // closest filter aovs require special handling because the source film buffer
    // has a different number of channels (4) than the snapshot destination, and requires
    // a non-default source pixel to dest pixel type conversion
    if (filter == pbr::AOV_FILTER_CLOSEST) {
        // all closest filter aovs are stored with 4 float channels
        MNRY_ASSERT(format == scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4);
        unsigned int numFloats = film.getAovNumFloats(aov);
        switch (numFloats) {
        case 1:
            if (fulldump) {
                // float4 -> float2 (float + depth) : we have to dump with depth info
                outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2, aovBuffer.getWidth(), aovBuffer.getHeight());
                snapshotBuffer(&outputBuffer->getFloat2Buffer(),
                               aovBuffer.getFloat4Buffer(),
                               reinterpret_cast<scene_rdl2::fb_util::Float4Buffer *>(&mExtrapolationBuffer),
                               extrapolate, untile, parallel,
                               [weights](const scene_rdl2::math::Vec4f &, unsigned ofs) -> uint32_t {
                                   const float w = weights[ofs];
                                   return w > 0.f? 0xffffffffu : 0u;
                               },
                               [](const scene_rdl2::math::Vec4f &pixel, unsigned) {
                                   return scene_rdl2::math::Vec2f(pixel.x, pixel.w); // pixel.w is depth
                               });
            } else {
                // float4 -> float
                outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT, aovBuffer.getWidth(), aovBuffer.getHeight());
                snapshotBuffer(&outputBuffer->getFloatBuffer(),
                               aovBuffer.getFloat4Buffer(),
                               reinterpret_cast<scene_rdl2::fb_util::Float4Buffer *>(&mExtrapolationBuffer),
                               extrapolate, untile, parallel,
                               [weights](const scene_rdl2::math::Vec4f &, unsigned ofs) -> uint32_t {
                                   const float w = weights[ofs];
                                   return w > 0.f? 0xffffffffu : 0u;
                               },
                               [](const scene_rdl2::math::Vec4f &pixel, unsigned) {
                                   return pixel.x;
                               });
            }
            break;
        case 2:
            if (fulldump) {
                // float4 -> float3 (float2 + depth) : we have to dump with depth info
                outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3, aovBuffer.getWidth(), aovBuffer.getHeight());
                snapshotBuffer(&outputBuffer->getFloat3Buffer(),
                               aovBuffer.getFloat4Buffer(),
                               reinterpret_cast<scene_rdl2::fb_util::Float4Buffer *>(&mExtrapolationBuffer),
                               extrapolate, untile, parallel,
                               [weights](const scene_rdl2::math::Vec4f &, unsigned ofs) -> uint32_t {
                                   const float w = weights[ofs];
                                   return w > 0.f? 0xffffffffu : 0u;
                               },
                               [](const scene_rdl2::math::Vec4f &pixel, unsigned) {
                                   return scene_rdl2::math::Vec3f(pixel.x, pixel.y, pixel.w); // pixel.w is depth
                               });
            } else {
                // float4 -> float2
                outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2, aovBuffer.getWidth(), aovBuffer.getHeight());
                snapshotBuffer(&outputBuffer->getFloat2Buffer(),
                               aovBuffer.getFloat4Buffer(),
                               reinterpret_cast<scene_rdl2::fb_util::Float4Buffer *>(&mExtrapolationBuffer),
                               extrapolate, untile, parallel,
                               [weights](const scene_rdl2::math::Vec4f &, unsigned ofs) -> uint32_t {
                                   const float w = weights[ofs];
                                   return w > 0.f? 0xffffffffu : 0u;
                               },
                               [](const scene_rdl2::math::Vec4f &pixel, unsigned) {
                                   return scene_rdl2::math::Vec2f(pixel.x, pixel.y);
                               });
            }
            break;
        case 3:
            if (fulldump) {
                // float4 -> float4 (float3 + depth) : we have to dump with depth info
                outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4, aovBuffer.getWidth(), aovBuffer.getHeight());
                snapshotBuffer(&outputBuffer->getFloat4Buffer(),
                               aovBuffer.getFloat4Buffer(),
                               reinterpret_cast<scene_rdl2::fb_util::Float4Buffer *>(&mExtrapolationBuffer),
                               extrapolate, untile, parallel,
                               [weights](const scene_rdl2::math::Vec4f &, unsigned ofs) -> uint32_t {
                                   const float w = weights[ofs];
                                   return w > 0.f? 0xffffffffu : 0u;
                               },
                               [](const scene_rdl2::math::Vec4f &pixel, unsigned) {
                                   return pixel; // pixel.w is depth
                               });
            } else {
                // float4 -> float3
                outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3, aovBuffer.getWidth(), aovBuffer.getHeight());
                snapshotBuffer(&outputBuffer->getFloat3Buffer(),
                               aovBuffer.getFloat4Buffer(),
                               reinterpret_cast<scene_rdl2::fb_util::Float4Buffer *>(&mExtrapolationBuffer),
                               extrapolate, untile, parallel,
                               [weights](const scene_rdl2::math::Vec4f &, unsigned ofs) -> uint32_t {
                                   const float w = weights[ofs];
                                   return w > 0.f? 0xffffffffu : 0u;
                               },
                               [](const scene_rdl2::math::Vec4f &pixel, unsigned) {
                                   return scene_rdl2::math::Vec3f(pixel.x, pixel.y, pixel.z);
                               });
            }
            break;
        default:
            MNRY_ASSERT(0 && "unexpected aov size");
        }

        return;
    }

    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
        outputBuffer->init(format, aovBuffer.getWidth(), aovBuffer.getHeight());
        snapshotVariablePixelBuffer(this,
                                    &outputBuffer->getFloatBuffer(),
                                    aovBuffer.getFloatBuffer(),
                                    reinterpret_cast<scene_rdl2::fb_util::FloatBuffer *>(&mExtrapolationBuffer),
                                    extrapolate,
                                    untile,
                                    parallel,
                                    weights,
                                    numConsistentSamples,
                                    filter);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
        outputBuffer->init(format, aovBuffer.getWidth(), aovBuffer.getHeight());
        snapshotVariablePixelBuffer(this,
                                    &outputBuffer->getFloat2Buffer(),
                                    aovBuffer.getFloat2Buffer(),
                                    reinterpret_cast<scene_rdl2::fb_util::Float2Buffer *>(&mExtrapolationBuffer),
                                    extrapolate,
                                    untile,
                                    parallel,
                                    weights,
                                    numConsistentSamples,
                                    filter);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
        outputBuffer->init(format, aovBuffer.getWidth(), aovBuffer.getHeight());
        snapshotVariablePixelBuffer(this,
                                    &outputBuffer->getFloat3Buffer(),
                                    aovBuffer.getFloat3Buffer(),
                                    reinterpret_cast<scene_rdl2::fb_util::Float3Buffer *>(&mExtrapolationBuffer),
                                    extrapolate,
                                    untile,
                                    parallel,
                                    weights,
                                    numConsistentSamples,
                                    filter);
        break;

    default:
        MNRY_ASSERT(0 && "unexpected aov buffer format");
    }
}

void
RenderDriver::snapshotAovBuffer(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                                int numConsistentSamples,
                                unsigned int aov,
                                bool untile,
                                bool parallel) const
//
// Snapshot the contents of an aov into a 4 channel RenderBuffer. (for optix denoise)
//
{
    const Film &film                              = getFilm();
    const float *weights                          = film.getWeightBuffer().getData();
    const scene_rdl2::fb_util::VariablePixelBuffer &aovBuffer = film.getAovBuffer(aov);
    const pbr::AovFilter filter                   = film.getAovBufferFilter(aov);
    const auto format                             = aovBuffer.getFormat();
    const bool extrapolate                        = !areCoarsePassesComplete();

    outputBuffer->init(aovBuffer.getWidth(), aovBuffer.getHeight());

    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
        snapshotVariablePixelBuffer(this,
                                    outputBuffer,
                                    aovBuffer.getFloatBuffer(),
                                    reinterpret_cast<scene_rdl2::fb_util::FloatBuffer *>(&mExtrapolationBuffer),
                                    extrapolate,
                                    untile,
                                    parallel,
                                    weights,
                                    numConsistentSamples,
                                    filter);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
        snapshotVariablePixelBuffer(this,
                                    outputBuffer,
                                    aovBuffer.getFloat2Buffer(),
                                    reinterpret_cast<scene_rdl2::fb_util::Float2Buffer *>(&mExtrapolationBuffer),
                                    extrapolate,
                                    untile,
                                    parallel,
                                    weights,
                                    numConsistentSamples,
                                    filter);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
        snapshotVariablePixelBuffer(this,
                                    outputBuffer,
                                    aovBuffer.getFloat3Buffer(),
                                    reinterpret_cast<scene_rdl2::fb_util::Float3Buffer *>(&mExtrapolationBuffer),
                                    extrapolate,
                                    untile,
                                    parallel,
                                    weights,
                                    numConsistentSamples,
                                    filter);
        break;

    default:
        MNRY_ASSERT(0 && "unexpected aov buffer format");
    }
}

void
RenderDriver::snapshotVisibilityBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                                       unsigned int aov,
                                       bool untile,
                                       bool parallel,
                                       bool fulldumpVisibility) const
{
    std::lock_guard<std::mutex> lock(mExtrapolationBufferMutex);

    const Film &film = getFilm();
    const scene_rdl2::fb_util::VariablePixelBuffer &aovBuffer = film.getAovBuffer(aov);
    const bool extrapolate = !areCoarsePassesComplete();

    if (!fulldumpVisibility) {
        //
        // non fulldump mode : single float = visibility
        //
        outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT, aovBuffer.getWidth(), aovBuffer.getHeight());

        snapshotBuffer(&outputBuffer->getFloatBuffer(),
                       aovBuffer.getFloat2Buffer(),
                       reinterpret_cast<scene_rdl2::fb_util::Float2Buffer *>(&mExtrapolationBuffer),
                       extrapolate,
                       untile,
                       parallel,
                       [](const scene_rdl2::math::Vec2f &, unsigned ofs) -> uint32_t {
                           return 0xffffffffu;
                       },
                       [](const scene_rdl2::math::Vec2f &pixel, unsigned ofs) {
                           // we need to find the ratio of # hits / # attempts
                           float p = 0.0f;
                           if (pixel.y > 0.0f) {
                               p = pixel.x / pixel.y;
                           }
                           return p;
                       });
    } else {
        //
        // fulldump mode : scene_rdl2::math::Vec3f : [0]=pixel.x [1]=pixel.y [2]=visibility
        //
        outputBuffer->init(scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3, aovBuffer.getWidth(), aovBuffer.getHeight());

        snapshotBuffer(&outputBuffer->getFloat3Buffer(),
                       aovBuffer.getFloat2Buffer(),
                       reinterpret_cast<scene_rdl2::fb_util::Float2Buffer *>(&mExtrapolationBuffer),
                       extrapolate,
                       untile,
                       parallel,
                       [](const scene_rdl2::math::Vec2f &, unsigned ofs) -> uint32_t {
                           return 0xffffffffu;
                       },
                       [](const scene_rdl2::math::Vec2f &pixel, unsigned ofs) {
                           // we need to find the ratio of # hits / # attempts
                           float p = 0.0f;
                           if (pixel.y > 0.0f) {
                               p = pixel.x / pixel.y;
                           }
                           return scene_rdl2::math::Vec3f(pixel.x, pixel.y, p);
                       });
    }
}

bool
RenderDriver::revertFilmData(RenderOutputDriver *renderOutputDriver,
                             const FrameState &fs,
                             unsigned &resumeTileSamples)
//
// unsigned minAdaptiveSamples
//   This value is used when resume file does not have minAdaptiveSamples info otherwise reverted file value
//   is used.
//
// int &resumeTileSample
//   Resume render start tile samples number which retrieved from resume file.
//
{
    //
    // Step 1 : Revert film object data from file : This is done by multi-threaded internally.
    //
    int resumeNumConsistentSamples;
    bool zeroWeightMask;
    bool adaptiveSampling;
    float adaptiveSampleParam[3];
    if (!renderOutputDriver->revertFilmData(*mFilm,
                                            resumeTileSamples, resumeNumConsistentSamples, zeroWeightMask,
                                            adaptiveSampling, adaptiveSampleParam)) {
        for (const auto &e: renderOutputDriver->getErrors()) Logger::error(e);
        renderOutputDriver->resetErrors(); // clean up error buffer
        Logger::error("Revert film data failed and could not run resume render any more.");
        return false;
    }

    //
    // Sampling mode condition check
    //
    if (adaptiveSampling) {
        // Resume file is adaptive sampling image.
        // This case, we can resume both of adaptive and uniform sampling render
        if (fs.mSamplingMode != SamplingMode::ADAPTIVE) {
            // Actually, main adaptive sampling resume render logic can support uniform resume file.
            // However, in this case progress report does not work properly. Because current uniform
            // sampling progress report can not handle complex situation of resume render from
            // adaptive resume file. If we can re-design progress percentage computation logic for
            // adaptive resume file, we can remove this restriction and adaptive resume can work with
            // both of uniform/adaptive resume render.
            Logger::error("ADAPTIVE sampling resume file can only used for ADAPTIVE resume render.");
            return false;
        }
    } else {
        // Resume file is uniform sampling image.
        // This case, technically we can resume by uniform and adaptive sample render.
    }
    if (fs.mSamplingMode == SamplingMode::ADAPTIVE) {
        // for adaptive resume, we need "Beauty Odd"
        if (mFilm->getRenderBufferOdd()) {
            Logger::error("ADAPTIVE sampling resume file required \"Beauty Odd\" AOV");
            return false;
        }
    }

    //
    // Step 2 : Denormalize/zeroWeightMask operation : This is done by multi-threaded internally.
    //
    crawlAllRenderOutput(*renderOutputDriver,
                         [&](const scene_rdl2::rdl2::RenderOutput *ro) { // non active AOV
                             if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX) {
                                 denormalizeBeautyOdd();
                             }
                             if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX) {
                                 denormalizeAlphaOdd();
                             }
                             if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE) {
                                 // naively create sample count array first.
                                 scene_rdl2::fb_util::PixelBuffer<unsigned> samplesCount;
                                 mFilm->fillPixelSampleCountBuffer(samplesCount);
                                 mFilm->getCryptomatteBuffer()->unfinalize(samplesCount);
                             }
                         },
                         [&](const int aovIdx) { // Visibility AOV
                             if (!zeroWeightMask) return;
                             // We have to set zero at zero weight pixel in this case.
                             zeroWeightMaskVisibilityBuffer(aovIdx);
                         },
                         [&](const int aovIdx) { // regular AOV
                             unsigned currNumConsistentSamples = fs.mRenderContext->getNumConsistentSamples();
                             if (resumeNumConsistentSamples != -1) {
                                 // If resume file has numConsistentSamples info, pick resume file info
                                 currNumConsistentSamples = (unsigned)resumeNumConsistentSamples;
                             }
                             denormalizeAovBuffer(currNumConsistentSamples, aovIdx);
                         });

    //
    // Step 3 : Copy beauty buffer from aov (BEAUTY, ALPHA) : This is done by multi-threaded internally.
    //
    if (!copyBeautyBuffer()) {
        return false;
    }

    return true;
}

void
RenderDriver::createXPUQueues()
{
    unsigned numCPUThreads = mcrt_common::getNumTBBThreads();
    // This queue size was determined empirically through performance testing.
    const unsigned cpuThreadQueueSize = 65536; // number of rays
    uint32_t rayHandlerFlags = 0;

    mXPUOcclusionRayQueue = new pbr::XPUOcclusionRayQueue(numCPUThreads,
                                                          cpuThreadQueueSize,
                                                          pbr::occlusionQueryBundleHandler,
                                                          pbr::xpuOcclusionQueryBundleHandlerGPU,
                                                          (void *)((uint64_t)rayHandlerFlags));

    // TODO: mXPURayQueue

    pbr::forEachTLS([&](pbr::TLState *tls) {
        tls->setXPUOcclusionRayQueue(mXPUOcclusionRayQueue);
        // TODO: mXPURayQueue
    });
}

unsigned
RenderDriver::flushXPUQueues(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena)
{
    unsigned numFlushed = 0;

    if (mXPUOcclusionRayQueue) {
        numFlushed += mXPUOcclusionRayQueue->flush(tls, arena);
    }
    if (mXPURayQueue) {
        numFlushed += mXPURayQueue->flush(tls, arena);
    }

    return numFlushed;
}

void
RenderDriver::freeXPUQueues()
{
    delete mXPUOcclusionRayQueue;
    mXPUOcclusionRayQueue = nullptr;

    pbr::forEachTLS([&](pbr::TLState *tls) {
        tls->setXPUOcclusionRayQueue(nullptr);
    });

    delete mXPURayQueue;
    mXPURayQueue = nullptr;

    pbr::forEachTLS([&](pbr::TLState *tls) {
        tls->setXPURayQueue(nullptr);
    });
}

void
RenderDriver::transferAllProgressFromSingleTLS(pbr::TLState *tls)
{
    MNRY_ASSERT(tls);

    // Transfer progress to RenderDriver.
    for (size_t ipass = 0; ipass < mTileWorkQueue.getNumPasses(); ++ipass) {
        if (tls->mPrimaryRaysSubmitted[ipass])
        {
            if (mFs.mRenderMode != RenderMode::PROGRESS_CHECKPOINT) {
                MNRY_ASSERT(mPrimaryRaysSubmitted[ipass] <= mSamplesPerPass[ipass]);
                mPrimaryRaysSubmitted[ipass] += tls->mPrimaryRaysSubmitted[ipass];
                MNRY_ASSERT(mPrimaryRaysSubmitted[ipass] <= mSamplesPerPass[ipass]);
            }

            tls->mPrimaryRaysSubmitted[ipass] = 0;
        }
    }

    // Check if all coarse passes have completed.
    // This is accurate for the non-bundled case but due to threading can
    // theoretically but rarely lead to false positives for the bundled case.
    // TODO: fix for bundled case.
    if (!mCoarsePassesComplete) {
        MNRY_ASSERT(mLastCoarsePassIdx != MAX_RENDER_PASSES);

        bool hasFinePasses = mTileWorkQueue.getNumPasses() > (mLastCoarsePassIdx + 1);
        if (hasFinePasses &&
            mSamplesPerPass[mLastCoarsePassIdx] &&
            mPrimaryRaysSubmitted[mLastCoarsePassIdx] >= mSamplesPerPass[mLastCoarsePassIdx]) {
            mCoarsePassesComplete = true;
        }
    }
}

void
RenderDriver::setReadyForDisplay()
{
    mReadyForDisplay = true;
}

void
RenderDriver::setCoarsePassesComplete()
{
    mCoarsePassesComplete = true;
}

void
RenderDriver::setFrameComplete()
{
    mFrameComplete = true;
}

void
RenderDriver::denormalizeAovBuffer(int numConsistentSamples, unsigned int aov)
{
    const float *weightBuff = mFilm->getWeightBuffer().getData();
    scene_rdl2::fb_util::VariablePixelBuffer &aovBuffer = mFilm->getAovBuffer(aov);
    const pbr::AovFilter filter = mFilm->getAovBufferFilter(aov);
    const auto format = aovBuffer.getFormat();

    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT :
        denormalizeVariablePixelBuffer(this,
                                       aovBuffer.getFloatBuffer(),
                                       weightBuff,
                                       numConsistentSamples,
                                       filter);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2 :
        denormalizeVariablePixelBuffer(this,
                                       aovBuffer.getFloat2Buffer(),
                                       weightBuff,
                                       numConsistentSamples,
                                       filter);
        break;
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3 :
        denormalizeVariablePixelBuffer(this,
                                       aovBuffer.getFloat3Buffer(),
                                       weightBuff,
                                       numConsistentSamples,
                                       filter);
        break;

    default:
        MNRY_ASSERT(0 && "unexpected aov buffer format");
    }
}

void
RenderDriver::denormalizeBeautyOdd()
{
    //
    // Essentially beautyOdd buffer should be normalized and denormalized by half weight
    // because this buffer contains half samples instead of full. However we are using full weight
    // for normalize/denormalize operation for beautyOdd buffer at this moment.
    //
    // Basically renderBuffer and renderBufferOdd is using same shared snapshot code.
    // (See also RenderDriver::snapshotRenderBufferSub())
    // This means extrapolation and normalization for standard and distributed rendering cases
    // are no difference between above 2 buffers. If we need to normalize by half weight for renderBufferOdd,
    // we have to update several different places to modify the code and we don't like that.
    // This is a reason why we are using full weight for normalize/denormalize operation for renderBufferOdd.
    //

    const float *weightBuff = mFilm->getWeightBuffer().getData();
    scene_rdl2::fb_util::RenderBuffer *renderBufferOdd = mFilm->getRenderBufferOdd();
    if (!renderBufferOdd) return; // just in case

    crawlAllTiledPixels([&](unsigned pixOffset) {
            // We only denormalize for RGB and not update A(lpha) in this function.
            scene_rdl2::math::Vec3f &pix = *(reinterpret_cast<scene_rdl2::math::Vec3f *>(renderBufferOdd->getData() + pixOffset));
            if (hasData(pix)) { pix *= weightBuff[pixOffset]; }
        });
}

void
RenderDriver::denormalizeAlphaOdd()
{
    //
    // Same as denormalizeBeautyOdd() function, we used standard weight value to normalize alphaOdd value.
    // So use standard weight value is used for denormalized as well.
    //
    const float *weightBuff = mFilm->getWeightBuffer().getData();
    scene_rdl2::fb_util::RenderBuffer *renderBufferOdd = mFilm->getRenderBufferOdd();
    if (!renderBufferOdd) return; // just in case

    crawlAllTiledPixels([&](unsigned pixOffset) {
            // We only denormalize for A and not for RGB in this function.
            scene_rdl2::math::Vec4f &pix = *(reinterpret_cast<scene_rdl2::math::Vec4f *>(renderBufferOdd->getData() + pixOffset));
            if (hasData(pix[3])) { pix[3] *= weightBuff[pixOffset]; }
        });
}

void
RenderDriver::zeroWeightMaskAovBuffer(unsigned int aov)
{
    const float *weightBuff = mFilm->getWeightBuffer().getData();
    scene_rdl2::fb_util::VariablePixelBuffer &aovBuffer = mFilm->getAovBuffer(aov);
    const auto format = aovBuffer.getFormat();

    switch (format) {
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT :
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2 :
    case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3 :
        // We don't need any zeroWeightMask operation for this type
        break;
    }
}

void
RenderDriver::zeroWeightMaskVisibilityBuffer(unsigned int aov)
{
    const float *weightBuff = mFilm->getWeightBuffer().getData();
    scene_rdl2::fb_util::VariablePixelBuffer &aovBuffer = mFilm->getAovBuffer(aov);

    zeroWeightMaskVariablePixelBuffer(this,
                                      aovBuffer.getFloat2Buffer(), // visibility is always float2 buffer
                                      weightBuff);
}

bool
RenderDriver::copyBeautyBuffer()
{
    const scene_rdl2::fb_util::VariablePixelBuffer *beautyAovBuff = mFilm->getBeautyAovBuff();
    const scene_rdl2::fb_util::VariablePixelBuffer *alphaAovBuff = mFilm->getAlphaAovBuff();
    if (!beautyAovBuff || !alphaAovBuff) {
        std::ostringstream ostr;
        ostr << "copy beauty buffer from AOV failed. Could not find beausy AOV or Alpha AOV";
        Logger::error(ostr.str());
        return false;           // Could not find beauty/alpha AOV buffer
    }
    const scene_rdl2::fb_util::Float3Buffer *beautyBuff = &(beautyAovBuff->getFloat3Buffer());
    const scene_rdl2::fb_util::FloatBuffer *alphaBuff = &(alphaAovBuff->getFloatBuffer());

    scene_rdl2::fb_util::RenderBuffer &renderBuff = mFilm->getRenderBuffer();
    crawlAllTiledPixels([&](unsigned pixOffset) {
            scene_rdl2::fb_util::RenderColor &currPix = *(renderBuff.getData() + pixOffset);
            const scene_rdl2::math::Vec3f &currRgb = *(beautyBuff->getData() + pixOffset);
            const float &currAlpha = *(alphaBuff->getData() + pixOffset);

            currPix.x = currRgb.x;
            currPix.y = currRgb.y;
            currPix.z = currRgb.z;
            currPix.w = currAlpha;
        });

    return true;
}

void
RenderDriver::renderThread(RenderDriver *driver,
                           const mcrt_common::TLSInitParams &initParams)
{
    // We are now running in the context of the render thread. This sets up the
    // task scheduler used for rendering. By creating a new task scheduler
    // instance here, we are allowing this thread to take part in rendering work
    // (tbb will spawn tasks on this thread when invoked from this thread.)
    tbb::task_scheduler_init scheduler(int(initParams.mDesiredNumTBBThreads));

    // TLS initialization.
    initTLS(initParams);

    driver->mRenderThreadState.set(UNINITIALIZED, READY_TO_RENDER);

    bool quit = false;
    while (!quit) {
        RenderThreadState state = driver->mRenderThreadState.get();

        switch(state) {
        case UNINITIALIZED:
        case READY_TO_RENDER:
            mcrt_common::threadSleep();
            break;

        case REQUEST_RENDER:
            driver->mRenderThreadState.set(REQUEST_RENDER, RENDERING, std::memory_order_acquire);

            // Turn on scoped accumulator style profiling.
            setAccumulatorActiveState(true);

            startRenderPhaseOfFrame();
            renderFrame(driver, driver->mFs);
            startUpdatePhaseOfFrame();

            // Turn off scoped accumulator profiling until we start rendering again.
            setAccumulatorActiveState(false);

            driver->mRenderThreadState.set(RENDERING, RENDERING_DONE, std::memory_order_release);
            break;

        case RENDERING_DONE:
            mcrt_common::threadYield();
            break;

        case KILL_RENDER_THREAD:
            // This is a sub-tbb scheduler init, we still have the main one to
            // clean up later, which happens in the RenderDriver destructor.
            scheduler.terminate();
            driver->mRenderThreadState.set(KILL_RENDER_THREAD, DEAD);
            quit = true;
            break;

        default:
            MNRY_ASSERT(0);
        }
    }
}

//------------------------------------------------------------------------------------------

void
RenderDriver::parserConfigure()
{
    mParser.description("RenderDriver command");
    mParser.opt("initFrame", "...command...", "initial frame control command",
                [&](Arg& arg) { return mParserInitFrameControl.main(arg.childArg()); });
    mParser.opt("tileWorkQueue", "...command...", "tileWorkQueue command",
                [&](Arg& arg) { return mTileWorkQueue.getParser().main(arg.childArg()); });
    mParser.opt("multiMachine", "...command...", "multi-machine related renderDriver command",
                [&](Arg& arg) { return mParserMultiMachineControl.main(arg.childArg()); });

    //------------------------------

    Parser& iniParser = mParserInitFrameControl;
    iniParser.description("initial frame control command under arras context");
    iniParser.opt("parallel", "<on|off>", "set special parallel initial frame update mode",
                  [&](Arg& arg) {
                      mParallelInitFrameUpdate = (arg++).as<bool>(0);
                      mCheckpointEstimationTime.reset();
                      return arg.msg(scene_rdl2::str_util::boolStr(mParallelInitFrameUpdate) + '\n');
                  });
    iniParser.opt("max", "<total>", "set parallel execution host count",
                  [&](Arg& arg) {
                      mParallelInitFrameUpdateMcrtCount = (arg++).as<unsigned>(0);
                      return arg.fmtMsg("max:%d\n", mParallelInitFrameUpdateMcrtCount);
                  });
    iniParser.opt("show", "", "show current information",
                  [&](Arg& arg) { return arg.msg(showInitFrameControl() + '\n'); });

    //------------------------------

    Parser& parserMm = mParserMultiMachineControl;
    parserMm.description("multi-machine related renderDriver command");
    parserMm.opt("active", "<on|off|show>", "set multi-machine mode for renderDriver",
                 [&](Arg& arg) {
                     if (arg() == "show") arg++;
                     else mMultiMachineCheckpointMainLoop = (arg++).as<bool>(0);
                     return arg.msg(scene_rdl2::str_util::boolStr(mMultiMachineCheckpointMainLoop) + '\n');
                 });
    parserMm.opt("budgetShort", "<sec>", "set multi-machine frame budget for initial short stint",
                 [&](Arg& arg) {
                     mMultiMachineFrameBudgetSecShort = (arg++).as<float>(0);
                     return arg.msg(showMultiMachineCheckpointMainLoopInfo() + '\n');
                 });
    parserMm.opt("quickPhaseLength", "<sec>", "set multi-machine quick phase length",
                 [&](Arg& arg) {
                     mMultiMachineQuickPhaseLengthSec = (arg++).as<float>(0);
                     return arg.msg(showMultiMachineCheckpointMainLoopInfo() + '\n');
                 });
    parserMm.opt("budgetLong", "<sec>", "set multi-machine frame budget for main long stint",
                 [&](Arg& arg) {
                     mMultiMachineFrameBudgetSecLong = (arg++).as<float>(0);
                     return arg.msg(showMultiMachineCheckpointMainLoopInfo() + '\n');
                 });
    parserMm.opt("show", "", "show multi-machine renderDriver setup",
                 [&](Arg& arg) { return arg.msg(showMultiMachineCheckpointMainLoopInfo() + '\n'); });
}

std::string
RenderDriver::showInitFrameControl() const
{
    using scene_rdl2::str_util::boolStr;
    using scene_rdl2::str_util::secStr;

    std::ostringstream ostr;
    ostr << "initFrameControl {\n"
         << "  mParallelInitFrameUpdate:" << boolStr(mParallelInitFrameUpdate) << '\n'
         << "  mParallelInitFrameUpdateMcrtCount:" << mParallelInitFrameUpdateMcrtCount << '\n'
         << "  mCheckpointEstimationStage:" << boolStr(mCheckpointEstimationStage) << '\n'
         << "  mRenderPrepTime.getAvg():" << secStr(mRenderPrepTime.getAvg()) << '\n'
         << "  mCheckpointEstimationTime.getAvg():" << secStr(mCheckpointEstimationTime.getAvg()) << '\n'
         << "}";
    return ostr.str();
}

std::string
RenderDriver::showMultiMachineCheckpointMainLoopInfo() const
{
    using scene_rdl2::str_util::boolStr;
    using scene_rdl2::str_util::secStr;

    std::ostringstream ostr;
    ostr << "multiMachine checkpointMainLoop info {\n"
         << "  mMultiMachineCheckpointMainLoop:" << boolStr(mMultiMachineCheckpointMainLoop) << '\n'
         << "  mMultiMachineFrameBudgetSecShort:" << secStr(mMultiMachineFrameBudgetSecShort) << '\n'
         << "  mMultiMachineQuickPhaseLengthSec:" << secStr(mMultiMachineQuickPhaseLengthSec) << '\n'
         << "  mMultiMachineFrameBudgetSecLong:" << secStr(mMultiMachineFrameBudgetSecLong) << '\n'
         << "  mMultiMachineGlobalProgressFraction:" << mMultiMachineGlobalProgressFraction << '\n'
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

std::shared_ptr<RenderDriver> gRenderDriver;

void
initRenderDriver(const TLSInitParams &initParams)
{
    // If this assertion triggers it means we've already called initRenderDriver
    // in this application. It should only ever be called once.
    MNRY_ASSERT_REQUIRE(gRenderDriver == nullptr);
    MOONRAY_THREADSAFE_STATIC_WRITE(gRenderDriver.reset(new RenderDriver(initParams)));
}

void
cleanUpRenderDriver()
{
    gRenderDriver = nullptr;
}

std::shared_ptr<RenderDriver>
getRenderDriver()
{
    return gRenderDriver;
}

} // namespace rndr
} // namespace moonray

