// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// This file includes logic which perform rendering single render frame by multi-thread
// based on the already constructed multi-passes workQueue information.
// Boot and shutdown multi-threads for MCRT computation is also placed here.
//
#include <scene_rdl2/render/util/AtomicFloat.h> // Needs to be included before any OpenImageIO file
#include <moonray/rendering/pbr/core/Scene.h>

#include "RenderDriver.h"
#include "RenderContext.h"
#include "AdaptiveRenderTilesTable.h"
#include "DisplayFilterDriver.h"
#include "PixSampleRuntimeVerify.h"

#include <moonray/rendering/rndr/adaptive/ActivePixelMask.h>

#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/mcrt_common/Clock.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>
#include <moonray/rendering/pbr/integrator/Picking.h>
#include <moonray/rendering/pbr/sampler/PixelScramble.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/render/util/ThreadPoolExecutor.h>

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_COUNT // See RuntimeVerify.h
#define RUNTIME_VERIFY0
#endif // end RUNTIME_VERIFY_PIX_SAMPLE_COUNT

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_SPAN // See RuntimeVerify.h
#define RUNTIME_VERIFY1
#endif // end RUNTIME_VERIFY_PIX_SAMPLE_SPAN

// This directive is used to fall back to the original TBB version of MCRT thread pool for emergency purposes.
//#define TBB_MCRT_THREADPOOL

#ifdef TBB_MCRT_THREADPOOL
#include <tbb/task_group.h>
#endif // end TBB_MCRT_THREADPOOL

// Debug message display for adaptive sampling stage rendering
//#define PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE

// Enable debugSamplesRecArray logic (record/playback all computeRadiance() result for beauty AOV)
// In order to activate debugSamplesRecArray mode, you should check Film.cc and
// do "grep mDebugSamplesRecArray.reset()"
//#define DEBUG_SAMPLE_REC_MODE

namespace moonray {
namespace rndr {

namespace {

template<typename F>
inline void
callLambda(F f)
{
    f();
}

#ifdef DEBUG
// Can only be called on master render thread.
bool
verifyNoOutstandingWork(const FrameState &fs)
{
    // Verify that all queues have been completely drained by now.
    if (fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED) {
        pbr::forEachTLS([&](pbr::TLState *tls) {
            MNRY_ASSERT(tls->areAllLocalQueuesEmpty());
        });
        MNRY_ASSERT(shading::Material::areAllShadeQueuesEmpty());
    }

    return true;
}
#endif // end DEBUG

} // namespace

//---------------------------------------------------------------------------------------------------------------

// static function
RenderDriver::RenderPassesResult
RenderDriver::renderPasses(RenderDriver *driver, const FrameState &fs,
                           bool allowCancelation)
{
#   ifdef RUNTIME_VERIFY1
    if (!PixSampleSpanRuntimeVerify::get()) {
        PixSampleSpanRuntimeVerify::init(fs.mWidth, fs.mHeight);
    }
#   endif // end RUNTIME_VERIFY1    

    MNRY_ASSERT(fs.mNumRenderThreads <= mcrt_common::getNumTBBThreads());

    // This is a part of the termination logic of multi-machine mcrt-computation
    if (driver->mRenderStopAtPassBoundary) {
        return RenderPassesResult::STOP_AT_PASS_BOUNDARY; // render completed at pass boundary
    }
    bool stopAtPassBoundary = false;

#   ifndef FORCE_SINGLE_THREADED_RENDERING
#   ifdef TBB_MCRT_THREADPOOL
    tbb::task_group taskGroup;
    std::string msg = "TBB MCRT thread pool";
    scene_rdl2::logging::Logger::info(msg);
    std::cerr << msg << '\n';
#   else // else TBB_MCRT_THREADPOOL
    auto calcCpuIdSequential = [&](size_t threadId) -> size_t { return threadId; };
    scene_rdl2::ThreadPoolExecutor::CalcCpuIdFunc calcCpuIdFunc = nullptr;

    std::ostringstream ostr;
    ostr << "MOONRAY MCRT thread pool";
    if (fs.mNumRenderThreads == std::thread::hardware_concurrency()) {
        // We want to use all cores. We activate CPU-affinity control and
        // all MCRT threads are individually attached to the core.
        calcCpuIdFunc = calcCpuIdSequential;
        ostr << " : enable MCRT-CPU-affinity";
    }
    std::string msg = ostr.str();
    scene_rdl2::logging::Logger::info(msg);
    std::cerr << msg << '\n';
    scene_rdl2::ThreadPoolExecutor taskGroup(fs.mNumRenderThreads, calcCpuIdFunc);
#   endif // end else TBB_MCRT_THREADPOOL
#   endif // end ifndef FORCE_SINGLE_THREADED_RENDERING

    // This counter verifies that we don't leave this function until all threads
    // have started working.
    CACHE_ALIGN tbb::atomic<unsigned> numTBBThreads;
    CACHE_ALIGN tbb::atomic<bool> canceled;

    numTBBThreads = 0;
    canceled = false;

    TileWorkQueue *workQueue = &driver->mTileWorkQueue;
    RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();

    // Setup samplesPerTile value
    timingRec.setRenderPassesSamplesPerTile(driver->mTileWorkQueue.getTotalTileSamples());

    // Record start time
    RenderPassesTimingRecord &renderPassesTimingRec = timingRec.startRenderPasses(fs.mNumRenderThreads);

    // Hand out one TLS per TBB thread from this list.
    mcrt_common::ThreadLocalState *topLevelTlsList = MNRY_VERIFY(mcrt_common::getTLSList());

    std::mutex checkpointEstimationTimeMutex;
    std::atomic<size_t> finishedTilesCount;
    finishedTilesCount = 0;
    size_t parallelInitFrameUpdateMinTileTotal = 0;
    if (driver->mParallelInitFrameUpdate) {
        parallelInitFrameUpdateMinTileTotal =
            std::max(1UL,
                     static_cast<size_t>(workQueue->getNumTiles()) / driver->mParallelInitFrameUpdateMcrtCount);
        /* useful debug message
        std::cerr << ">> RenderFramePasses.cc init minTileTotal:" << parallelInitFrameUpdateMinTileTotal
                  << " mParallelInitFrameUpdateMcrtCount:" << driver->mParallelInitFrameUpdateMcrtCount << '\n';
        */
    }

    // Spawn one task for each tbb thread.
    for (unsigned ithread = 0; ithread < fs.mNumRenderThreads; ++ithread) {

        mcrt_common::ThreadLocalState *topLevelTls = topLevelTlsList + ithread;

        // The topLevelTls pointer must be captured by value to prevent it changing out
        // from under us as other threads are subsequently spawned.

#ifdef FORCE_SINGLE_THREADED_RENDERING
        callLambda([&, topLevelTls]()
#else
        taskGroup.run([&, topLevelTls]()
#endif
        {
            double timeStart = scene_rdl2::util::getSeconds(); // get current time

            ++numTBBThreads;

            pbr::TLState *tls = MNRY_VERIFY(topLevelTls->mPbrTls.get());

            EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_RENDER_DRIVER_OVERHEAD);
            ACCUMULATOR_PROFILE(tls, ACCUM_RENDER_DRIVER_PARALLEL);

            // Wait barrier until all threads are wake up and ready to go.
            // This is a busy loop and cost is big concern for REALTIME renderMode case.
            // (It's negligible when BATCH and PROGRESSIVE case).
            // Why all threads are sync to start ? This is a fastest and reliable solution to boot
            // all threads. If we boot threads without this sync barrier logic, booted thread is start working
            // and increase CPU load then we need longer time to complete boot for all render threads.
            // As a result, we need longer duration to get full speed.
            while (numTBBThreads < fs.mNumRenderThreads) {
                if (isRenderCanceled()) {
                    // There is some risk to get deadlock situation on this loop.
                    // This is a safety logic to exit by cancel.
                    canceled = true;
                    return;
                }

                // Actually this loop can not get constant performance even carefully controlled nano sleep value.
                // Because render has lots of sleeped (not active) threads actually and kernel need to check
                // about them constantly. Shortest context switching interval is around 10ms (I guess).
                // So sometimes some threads need to wait extra 10ms or so to wake up from suspended condition.
                // Obviously 10ms loss is big for realtime render case but this solution is a best result so far.
                struct timespec tm;
                tm.tv_sec = 0;
                tm.tv_nsec = 1000; // 0.001ms : tested several different range but could not find any difference.
                nanosleep(&tm, NULL); // yield CPU resource
            }

            double timeReady = scene_rdl2::util::getSeconds(); // get current time


            // Embree wants these modes set on each thread it uses.
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

            // Initialize for texturing, this needs to be called once per
            // render frame.
            shading::TLState *shadingTls = topLevelTls->mShadingTls.get();
            if (shadingTls) {
                shadingTls->initTexturingSupport();
            }

            if (allowCancelation) {
                tls->enableCancellation((fs.mRenderMode == RenderMode::BATCH) ? false : true);
            } else {
                tls->disableCancellation();
            }

            const unsigned threadIdx = getThreadIdx(tls);

            double timeInitTexturing = scene_rdl2::util::getSeconds(); // get current time

            TileGroup group;

            unsigned long long processedTilesTotal = 0ULL;
            unsigned long long processedSampleTotal = 0ULL;
                            
            // We have to track the condition of stopAtPassBoundary each thread independently
            // in order to properly flush the radiance queue under vector mode.
            bool stopAtPassBoundaryThreadLocal = false;
            while (!tls->isCanceled() && workQueue->getNextTileGroup(threadIdx, &group, driver->getLastCoarsePassIdx())) {
                if (group.mFirstFinePass && (fs.mRenderMode == RenderMode::PROGRESSIVE ||
                                             fs.mRenderMode == RenderMode::PROGRESSIVE_FAST ||
                                             fs.mRenderMode == RenderMode::PROGRESS_CHECKPOINT)) {
                    // Increase all queue sizes in the system if we've finished coarse pass
                    pbr::forEachTLS([](pbr::TLState *tls) {
                        tls->setAllQueueSizes(1.0f);
                    });
                }

                // Record tiles currently being rendered.
                tls->mCurrentPassIdx = group.mPassIdx;
                if (fs.mRenderMode == RenderMode::PROGRESS_CHECKPOINT) {
                    // If tls->mCurrentPassIdx is 0, we can not cancel this pass (see pbr::TLState::isCanceled()).
                    // So we set passId + 1 for tls when PROGRESS_CHECKPOINT mode.
                    tls->mCurrentPassIdx++;
                }
                processedSampleTotal += static_cast<unsigned long long>(renderTiles(driver, topLevelTls, group));
                ++processedTilesTotal;

                if (driver->mParallelInitFrameUpdate) {
                    // Under parallel init frame update mode, we have to check the total number of processed
                    // tiles exceeds threshold. If we processed tile more than the threshold, we turn on the
                    // snapshot ready flag and set cancellation available enable in order to enhance
                    // interactive performance under multi-machine context.
                    finishedTilesCount += (group.mEndTileIdx - group.mStartTileIdx);
                    if (finishedTilesCount >= parallelInitFrameUpdateMinTileTotal) {
                        if (!driver->isReadyForDisplay()) {
                            driver->setReadyForDisplay();
                            tls->enableCancellation(true);
                            { // statistical information for debugging purposes
                                std::lock_guard<std::mutex> lock(checkpointEstimationTimeMutex);
                                float time = driver->mParallelInitFrameUpdateTime.end();
                                driver->mCheckpointEstimationTime.set(time);
                                /* useful debug message
                                std::ostringstream ostr;
                                ostr << ">> RenderFramePasses.cc READY-FOR-DISPLAY parallel"
                                     << " time:" << time << " sec"
                                     << " parallelInitFrameUpdateMinTileTotal:" << parallelInitFrameUpdateMinTileTotal
                                     << " finishedTilesCount:" << finishedTilesCount;
                                std::cerr << ostr.str() << '\n';
                                */
                            }
                        }
                    }
                }

                // Update progress.
                driver->transferAllProgressFromSingleTLS(tls);

                if (fs.mSamplingMode == SamplingMode::ADAPTIVE) {
                    if (driver->mFilm->getAdaptiveDone()) {
                        break;
                    }
                }
            }

            double timeRenderTiles = scene_rdl2::util::getSeconds(); // get current time

            //
            // Queue draining phase for bundled mode.
            //
            // TODO: track progress for this phase also if it starts taking
            //       significant time.
            //
            if (fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED ||
                fs.mExecutionMode == mcrt_common::ExecutionMode::XPU) {

                ACCUMULATOR_PROFILE(tls, ACCUM_DRAIN_QUEUES);

                if (tls->isCanceled() || stopAtPassBoundaryThreadLocal) {

                    ACCUMULATOR_PROFILE(tls, ACCUM_NON_RENDER_DRIVER);

                    // Radiance values are already computed so just write them into
                    // the frame buffer. This gives a more complete image even if
                    // a cancellation was requested (useful for progressive mode).
                    tls->flushRadianceQueue();

                } else {

                    ACCUMULATOR_PROFILE(tls, ACCUM_NON_RENDER_DRIVER);

                    unsigned flushed;

                    // Note, this flushing logic depends on one thread not being able
                    // to add work to another threads queues.
                    do {
                        flushed = shading::Material::flushNonEmptyShadeQueue(topLevelTls);
                        CHECK_CANCELLATION(tls, break);

                        flushed += driver->flushXPUQueues(topLevelTls, &topLevelTls->mArena);

                        flushed += tls->flushLocalQueues();
                        CHECK_CANCELLATION(tls, break);

                    } while (flushed || numTBBThreads < fs.mNumRenderThreads);
                }
            }

            double timeQueueDraining = scene_rdl2::util::getSeconds(); // get current time

            if (tls->isCanceled()) {
                canceled = true;
            }

            tls->disableCancellation();

            // Update progress.
            driver->transferAllProgressFromSingleTLS(tls);

            double timeEnd = scene_rdl2::util::getSeconds(); // get current time

            // Update duration record information
            RenderEngineTimingRecord &rec = renderPassesTimingRec.getEngineTimingRecord(tls->mThreadIdx);
            rec.setId(tls->mThreadIdx);
            rec.set(RenderEngineTimingRecord::TagDBL::START, timeStart - renderPassesTimingRec.getStartTime());
            rec.set(RenderEngineTimingRecord::TagDBL::READY, timeReady - renderPassesTimingRec.getStartTime());
            rec.set(RenderEngineTimingRecord::TagDBL::INITTEXTURING, timeInitTexturing - timeReady);
            rec.set(RenderEngineTimingRecord::TagDBL::RENDERTILES, timeRenderTiles - timeInitTexturing);
            rec.set(RenderEngineTimingRecord::TagDBL::QUEUEDRAINING, timeQueueDraining - timeRenderTiles);
            rec.set(RenderEngineTimingRecord::TagDBL::FINISHUP, timeEnd - timeQueueDraining);
            rec.set(RenderEngineTimingRecord::TagDBL::ENDGAP, timeEnd); // set timeEnd at this moment. update later.
            rec.set(RenderEngineTimingRecord::TagDBL::ACTIVE, timeEnd - timeReady);
            rec.set(RenderEngineTimingRecord::TagULL::PROCESSEDTILESTOTAL, processedTilesTotal);
            rec.set(RenderEngineTimingRecord::TagULL::PROCESSEDSAMPLETOTAL, processedSampleTotal);
        });
    }

    taskGroup.wait();

    timingRec.finalizeRenderPasses(); // End record timing : compute average thread timing and other info
    driver->mProgressEstimation.updatePassInfo(timingRec); // update pass info

#   ifdef DEBUG
    // Verify that all queues have been completely drained by now.
    if (!canceled) {
        MNRY_ASSERT(verifyNoOutstandingWork(fs));
    }
#   endif // end DEBUG

    if (stopAtPassBoundary) return RenderPassesResult::STOP_AT_PASS_BOUNDARY;
    if (canceled) return RenderPassesResult::ERROR_OR_CANCEL;
    return RenderPassesResult::OK_AND_TRY_NEXT;
}

void
RenderDriver::runDisplayFiltersTile(RenderDriver *driver,
                                    size_t tileIdx,
                                    size_t threadId)
{
    if (driver->mDisplayFilterMutex.try_lock()) {
        // This thread will try to snapshot aovs.
        // The aov buffers are stored in mDisplayFilterDriver and all threads
        // read from those buffers to run the display filters.
        driver->getFrameState().mRenderContext->snapshotAovsForDisplayFilters(true, false);
        driver->mDisplayFilterMutex.unlock();
    }

    // Filter just this tile. It may require reading input buffer data from adjacent tiles.
    // It is ok if that data is out of date. This function is called only in progressive mode
    // and the display filter buffers get refined with each progessive pass.
    driver->mDisplayFilterDriver.runDisplayFilters(tileIdx, threadId);
}

//---------------------------------------------------------------------------------------------------------------

struct RenderSamplesParams
{
    RenderDriver *      mDriver;

    Film *              mFilm;

    unsigned            mTileIdx; // index of current processed tileId

    unsigned            mPx;
    unsigned            mPy;

    pbr::Sampler *      mSampler;

    unsigned            mConsistentSamplesPerPixel;
    unsigned            mTotalNumSamples;
    unsigned            mRealtimeSampleOfs;

    float               mShutterBias;
    unsigned            mRenderNodeTotal;
    unsigned            mRenderNodeSampleOfs;

    unsigned            mAovNumFloats;
    float *             mAovs;
    float *             mLocalAovs;
    float *             mDeepAovs;
    float *             mDeepVolumeAovs;

    mcrt_common::ScopedAccumulator * mNonRenderDriverAccumulator;
};

unsigned
RenderDriver::renderTiles(RenderDriver *driver,
                          mcrt_common::ThreadLocalState *tls,
                          const TileGroup &group) // return processed primary ray total
{
    pbr::TLState *pbrTls = tls->mPbrTls.get();
    const rndr::FrameState &fs = driver->getFrameState();
    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    // Used to track all the time not spent in the RenderDriver.
    ACCUMULATOR_GET_PAUSED(pbrTls, ACCUM_NON_RENDER_DRIVER, nonRenderDriverAccumulator);

    RenderSamplesParams params;
    { // setup params members
        // CPPCHECK -- Using memset() on struct which contains a floating point number.
        // cppcheck-suppress memsetClassFloat
        memset(&params, 0, sizeof(params));
        params.mDriver = driver;
        {
            float *aovs = nullptr;
            float *deepAovs = nullptr;
            float *deepVolumeAovs = nullptr;
            float *localAovs = nullptr;
            unsigned aovNumChannels = 0;
            if (!fs.mAovSchema->empty()) {
                aovNumChannels = fs.mAovSchema->numChannels();
                aovs = arena->allocArray<float>(aovNumChannels);
                deepAovs = arena->allocArray<float>(aovNumChannels);
                deepVolumeAovs = arena->allocArray<float>(aovNumChannels);
                localAovs = arena->allocArray<float>(aovNumChannels);
            }
            params.mAovNumFloats = aovNumChannels;
            params.mAovs = aovs;
            params.mDeepAovs = deepAovs;
            params.mDeepVolumeAovs = deepVolumeAovs;
            params.mLocalAovs = localAovs;
        }

        params.mNonRenderDriverAccumulator = &nonRenderDriverAccumulator;

        params.mConsistentSamplesPerPixel = fs.mRenderContext->getNumConsistentSamples();

        // shutter bias modifies the time of the primary ray sample
        params.mShutterBias = fs.mScene->getCamera()->getShutterBias();

        params.mRenderNodeTotal = fs.mNumRenderNodes;
        params.mRenderNodeSampleOfs = fs.mRenderNodeIdx;
    }

    unsigned processedSampleTotalFilm0 = 0;

    params.mFilm = driver->mFilm;

    pbr::DeepBuffer *deepBuffer = params.mFilm->getDeepBuffer();
    // This is not thread safe - it should really be done during frame setup!
    if (deepBuffer != nullptr) {
        unsigned totalNumSamples = fs.mMaxSamplesPerPixel;
        deepBuffer->setSamplesPerPixel(totalNumSamples);
        if (totalNumSamples >= 64) {
            deepBuffer->setSubpixelRes(8);
        } else if (totalNumSamples >= 16) {
            deepBuffer->setSubpixelRes(4);
        } else if (totalNumSamples >= 4) {
            deepBuffer->setSubpixelRes(2);
        } else {
            deepBuffer->setSubpixelRes(1);
        }
    }

    pbr::CryptomatteBuffer *cryptomatteBuffer = params.mFilm->getCryptomatteBuffer();

    // Loop over current batch of tiles, we execute tile batches in parallel.
    unsigned processedSampleTotal = 0;
    for (unsigned itile = group.mStartTileIdx; itile != group.mEndTileIdx; ++itile) {
        params.mTileIdx = itile;
        if (!renderTile(driver, tls, group, params, deepBuffer, cryptomatteBuffer, processedSampleTotal)) {
            return 0; // cancel return
        }
    }
    processedSampleTotalFilm0 = processedSampleTotal;

    return processedSampleTotalFilm0; // return processed sample total of film = 0
}

// static function
bool
RenderDriver::renderTile(RenderDriver *driver,
                         mcrt_common::ThreadLocalState *tls,
                         const TileGroup &group,
                         RenderSamplesParams &params,
                         pbr::DeepBuffer *deepBuffer,
                         pbr::CryptomatteBuffer *cryptomatteBuffer,
                         unsigned &processedSampleTotal)
//
// return cancel render condition : true=non-cancel false=canceled
//
{
    const rndr::FrameState &fs = driver->getFrameState();

    if (fs.mSamplingMode != SamplingMode::UNIFORM) {
        if (params.mFilm->getAdaptiveRenderTilesTable()->getTile(params.mTileIdx).isCompleted()) {
            return true;        // already completed tile.
        }
    }

    const Pass &pass = driver->mTileWorkQueue.getPass(group.mPassIdx);
    if (fs.mSamplingMode == SamplingMode::UNIFORM || pass.isCoarsePass()) {
        //
        // Uniform sampling tile mode
        //
        if (!renderTileUniformSamples<false>(driver, tls, group, params, deepBuffer, cryptomatteBuffer,
                                             pass.mStartSampleIdx, pass.mEndSampleIdx, processedSampleTotal)) {
            return false; // canceled render
        }
    } else {
        //
        // Adaptive sampling tile mode
        //
        const Film* const film = params.mFilm;
        if (!film->getAdaptiveDone()) {
            if (!renderTileAdaptiveStage(driver, tls, group, params, deepBuffer, cryptomatteBuffer,
                                         processedSampleTotal)) {
                return false;   // canceled render
            }
        }
    }
    driver->mDisplayFilterDriver.requestTileUpdate(params.mTileIdx);

    // Update display filters after each tile is rendered in progressive mode.
    // This is done for non coarse passes because each of the neighboring
    // tiles are guaranteed to have valid data, though it might be outdated data.
    if (driver->areCoarsePassesComplete() && fs.mRenderMode == RenderMode::PROGRESSIVE) {
        runDisplayFiltersTile(driver, params.mTileIdx, tls->mThreadIdx);
    }

    return true;
}

// static function
bool
RenderDriver::renderTileAdaptiveStage(RenderDriver* driver,
                                      mcrt_common::ThreadLocalState* tls,
                                      const TileGroup& group,
                                      RenderSamplesParams& params,
                                      pbr::DeepBuffer* deepBuffer,
                                      pbr::CryptomatteBuffer* cryptomatteBuffer,
                                      unsigned& processedSampleTotal)
{
    const Pass &pass = driver->mTileWorkQueue.getPass(group.mPassIdx);
#   ifdef PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE
    bool debug = params.mFilm->getAdaptiveRenderTilesTable()->isDebugTile(params.mTileIdx);
    if (debug) {
        std::cerr << ">> RenderFrame.cc renderTileAdaptiveStage() start ... group.mPassIdx:" << group.mPassIdx
                  << " pass(pix(" << pass.mStartPixelIdx << '~' << pass.mEndPixelIdx << ")"
                  << " smp(" << pass.mStartSampleIdx << '~' << pass.mEndSampleIdx << "))" << std::endl;
    }
#   endif // end PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE
    if (pass.mStartSampleIdx == pass.mEndSampleIdx) {
        return true;
    }

    Film* const film = params.mFilm;
    const scene_rdl2::fb_util::Tile &tile = (*driver->getTiles())[params.mTileIdx];

    ActivePixelMask adaptiveRegion;
    switch (updateTileCondition(driver, group, params, pass.mStartSampleIdx)) {
        case AdaptiveRenderTileInfo::Stage::COMPLETED:
            return true;
        case AdaptiveRenderTileInfo::Stage::UNIFORM_STAGE:
            adaptiveRegion = ActivePixelMask::all();
            break;
        case AdaptiveRenderTileInfo::Stage::ADAPTIVE_STAGE:
            adaptiveRegion = film->getAdaptiveSampleArea(tile, tls);
            break;

    }
    if (!renderTileUniformSamples<true>(driver,
                                        tls,
                                        group,
                                        params,
                                        deepBuffer,
                                        cryptomatteBuffer,
                                        pass.mStartSampleIdx,
                                        pass.mEndSampleIdx,
                                        processedSampleTotal,
                                        adaptiveRegion)) {
#       ifdef PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE
        if (debug) { std::cerr << ">> RenderFrame.cc renderTileAdaptiveStage() canceled" << std::endl; }
#       endif // end PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE
        return false; // canceled render
    }
    film->updateAdaptiveError(tile, film->getRenderBuffer(), *film->getRenderBufferOdd(), tls, pass.mEndSampleIdx);

#   ifdef PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE
    if (debug) { std::cerr << ">> RenderFrame.cc renderTileAdaptiveStage() done" << std::endl; }
#   endif // end PRINT_DEBUG_MESSAGE_ADAPTIVE_STAGE

    return true;
}

// static function
AdaptiveRenderTileInfo::Stage
RenderDriver::updateTileCondition(RenderDriver *driver,
                                  const TileGroup &group,
                                  RenderSamplesParams &params,
                                  const unsigned startSampleIdx)
{
    const rndr::FrameState &fs = driver->getFrameState();
    if (startSampleIdx < fs.mMinSamplesPerPixel) {
        // We have to do uniform sampling until minSamplesPerPixel
        return AdaptiveRenderTileInfo::Stage::UNIFORM_STAGE;
    }
    if (startSampleIdx >= fs.mMaxSamplesPerPixel) { // hit maximum samples per pixel -> complete this tile
        params.mFilm->getAdaptiveRenderTilesTable()->setTileCompleteAdaptiveStage(params.mTileIdx);
        return AdaptiveRenderTileInfo::Stage::COMPLETED;
    }

    // Still need to add sample uniformly to all pixels of this tile
    return AdaptiveRenderTileInfo::Stage::ADAPTIVE_STAGE;
}

// static function
// _adaptive_ does not strictly have to be a template parameter, but it's known at compile-time, so we might as well
// optimize it out.
template <bool adaptive>
bool
RenderDriver::renderTileUniformSamples(RenderDriver *driver,
                                       mcrt_common::ThreadLocalState *tls,
                                       const TileGroup &group,
                                       RenderSamplesParams &params,
                                       pbr::DeepBuffer *deepBuffer,
                                       pbr::CryptomatteBuffer *cryptomatteBuffer,
                                       const unsigned startSampleIdx,
                                       const unsigned endSampleIdx,
                                       unsigned &processedSampleTotal,
                                       const ActivePixelMask& inputRegion)
// return cancel render condition : true=non-cancel false=canceled
{
    if (adaptive && !inputRegion) {
        return true;
    }

    pbr::TLState *pbrTls = tls->mPbrTls.get();
    const rndr::FrameState &fs = *reinterpret_cast<const rndr::FrameState *>(pbrTls->mFs);
    const Pass &pass = driver->mTileWorkQueue.getPass(group.mPassIdx);
    Film &film = *params.mFilm;
    const scene_rdl2::fb_util::Tile &tile = (*driver->getTiles())[params.mTileIdx];

    for (unsigned ipix = pass.mStartPixelIdx; ipix != pass.mEndPixelIdx; ++ipix) {
        // Note we wrap around values over 64. This subtlety is actively used for realtime mode.
        unsigned pixelPerm = Film::getPixelFillOrder(params.mTileIdx, ipix);
        unsigned px = (tile.mMinX & ~0x07) + (pixelPerm & 7);
        unsigned py = (tile.mMinY & ~0x07) + (pixelPerm / 8);
        if (!fs.mViewport.contains(px, py)) continue; // Check this pixel is inside the viewport.

        if (adaptive && inputRegion && !inputRegion(px - tile.mMinX, py - tile.mMinY)) {
            continue;
        }

        scene_rdl2::alloc::Arena *pixelArena = pbrTls->mPixelArena;
        SCOPED_MEM(pixelArena);

        // update pixelInfo(=pixel center depth) if required
        computePixelInfo(driver, tls, film, px, py);

        unsigned currStartSampleIdx = startSampleIdx;
        unsigned currEndSampleIdx = endSampleIdx;
        if (film.getResumeStartSampleIdBuff().isValid()) {
            // resume rendering by adaptive sampled resume file
            unsigned initSampleIdx = film.getResumeStartSampleIdBuff().getSampleId(px, py);
            if (initSampleIdx > 0) {
                // This is a case of resume rendering.
                // First of all, we have to compare the sample range with the inital-start-sampleId which
                // came from  the resume file.
                if (currEndSampleIdx <= initSampleIdx) continue; // this pixel already done.
                if (currStartSampleIdx < initSampleIdx) {
                    currStartSampleIdx = initSampleIdx; // We don't want to run already finished sample range.
                }
            }
        }
        if (fs.mSamplingMode == SamplingMode::ADAPTIVE) {
            // In an adaptive sampling case, we have to schedule sample range to the MCRT threads as
            // continuous order. We have trouble for checkpoint data if some pixel does not sample some
            // middle sample range. This wrong pixel's last sampleId is not equal to weight value and
            // resume rendering will make a mistake start samplingId in this case.
            // The order of the scheduled sample span by each MCRT thread does not matter.
            // It is pretty important to get a continuous sampleId sequence without any missing span in
            // the middle. In order to keep all samplingId spans keeps continuous condition,
            // we use CAS access to the current pixel sampleId value. This makes it safe to get the next
            // sampleId range by multiple MCRT threads even 2 or more threads are working on the same tile
            // simultaneously.
            const unsigned nsamples = currEndSampleIdx - currStartSampleIdx;
            currStartSampleIdx = film.getCurrSampleIdBuff().atomicReserveSampleId(px, py, nsamples);
            currEndSampleIdx = currStartSampleIdx + nsamples;
        }

#       ifdef RUNTIME_VERIFY0
        PixSampleCountRuntimeVerify::get()->add(px, py, currEndSampleIdx - currStartSampleIdx);
#       endif // end RUNTIME_VERIFY0
#       ifdef RUNTIME_VERIFY1
        PixSampleSpanRuntimeVerify::get()->add(px, py, currStartSampleIdx, currEndSampleIdx);
#       endif // end RUNTIME_VERIFY1        

        // Get the number of samples for this pixel.
        unsigned totalNumSamples = computeTotalNumSamples(fs, 0, px, py);
        if (totalNumSamples < currStartSampleIdx) continue; // skip this pixel if it has enough samples

        // setup params
        params.mPx = px;
        params.mPy = py;
        params.mTotalNumSamples = totalNumSamples;

        // Each pixel gets its own sampler.
        pbr::Sampler sampler;
        const auto stereo = fs.mScene->getCamera()->getStereoView();
        pbr::DeepBuffer *deepBuffer = film.getDeepBuffer();
        bool use8x8Grid = deepBuffer ?
            (deepBuffer->getFormat() == pbr::DeepFormat::OpenDCX2_0) : false;
        sampler = pbr::Sampler(pbr::PixelScramble(px, py, fs.mFrameNumber, stereo),
                          fs.mPixelFilter, use8x8Grid, totalNumSamples);
        params.mSampler = &sampler;

        // Used for generating good sampling index values in the case of realtime.
        params.mRealtimeSampleOfs = (fs.mRenderMode == RenderMode::REALTIME) ? ipix : 0;
        bool renderPixCondition = false;
        if (fs.mRenderMode != RenderMode::PROGRESSIVE_FAST) {
            renderPixCondition = (fs.mExecutionMode == mcrt_common::ExecutionMode::SCALAR) ?
                                  renderPixelScalarSamples(pbrTls, currStartSampleIdx, currEndSampleIdx, &params) :
                                  renderPixelVectorSamples(pbrTls, currStartSampleIdx, currEndSampleIdx, &params,
                                                           group, deepBuffer, cryptomatteBuffer);
        } else {
            // renderPixelScalarSamplesFast is run for both SCALAR and VECTOR mode
            renderPixCondition = renderPixelScalarSamplesFast(pbrTls, currStartSampleIdx, currEndSampleIdx, &params);
        }
        if (!renderPixCondition) {
            return false; // canceled
        }

        if (deepBuffer) {
            deepBuffer->finishPixel(pbrTls->mThreadIdx);
        }

        // Update progress.
        pbrTls->mTilesRenderedTo.setBit(params.mTileIdx); // for moonray_gui debugging. for showTileProgress()
        // NonCheckpoint or nonAdaptiveCheckpoint case, we uses this code for progress update
        // Currently all adaptive sampling progress update information is provided by
        // film.getAdaptiveRenderTilesTable()
        unsigned samplesRendered = currEndSampleIdx - currStartSampleIdx;
        pbrTls->mPrimaryRaysSubmitted[group.mPassIdx] += samplesRendered; // used by non checkpoint case
        processedSampleTotal += samplesRendered;
        if (fs.mRenderMode == RenderMode::PROGRESS_CHECKPOINT) {
            driver->mProgressEstimation.atomicAddSamples(samplesRendered);
        }

        if (film.getAdaptiveRenderTilesTable()) { // adaptive sampling case
            unsigned addedSamples = currEndSampleIdx - currStartSampleIdx;
            film.getAdaptiveRenderTilesTable()->setTileUpdate(params.mTileIdx, addedSamples); // update
        }
    } // ipix

    return true;
}

// static function
bool
RenderDriver::renderPixelScalarSamples(pbr::TLState *pbrTls,
                                       const unsigned startSampleIdx,
                                       const unsigned endSampleIdx,
                                       RenderSamplesParams *params)
{
    //
    // Single ray execution start:
    //

    MNRY_ASSERT(startSampleIdx < endSampleIdx);

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    unsigned const px = params->mPx;
    unsigned const py = params->mPy;
    Film *film = params->mFilm;
    float *aovs = params->mAovs;

    float *localAovs = params->mLocalAovs;
    float *deepAovs = params->mDeepAovs;
    float localDepth = scene_rdl2::math::pos_inf;

    const rndr::FrameState &fs = *reinterpret_cast<const rndr::FrameState *>(pbrTls->mFs);
    const auto& schema = *fs.mAovSchema;

    // conditional heat map collection
    MNRY_ASSERT((fs.mRequiresHeatMap && film->getHeatMapBuffer()) ||
               !fs.mRequiresHeatMap);
    MCRT_COMMON_CLOCK_OPEN(fs.mRequiresHeatMap ? &(film->getHeatMap(px, py)) : nullptr);

    scene_rdl2::fb_util::RenderColor accRadiance = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::ZeroTy());
    scene_rdl2::fb_util::RenderColor accRadiance2 = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::ZeroTy());
    uint32_t numAccSamples = 0;
    if (params->mLocalAovs) fs.mAovSchema->initFloatArray(params->mLocalAovs);

#ifdef DEBUG_SAMPLE_REC_MODE
    DebugSamplesRecArray *debugSamplesRecArray = film->getDebugSamplesRecArray();
#endif // end DEBUG_SAMPLE_REC_MODE

    // Loop over samples in current pixel. It is important to note that all samples within
    // a single pass are guaranteed to be executed on the same thread. No single tile can
    // be executed on different threads at one point in time (the TileWorkQueue ensures
    // this), so we don't need to do any locking here.
    for (unsigned isamp = startSampleIdx; isamp != endSampleIdx; ++isamp) {

        // This line is for supporting the pixel sample map functionality.
        if (isamp >= params->mTotalNumSamples) break;

        CHECK_CANCELLATION(pbrTls, return false);

        // Push/pop memory arena for each sample.
        SCOPED_MEM(arena);

        if (aovs) {
            fs.mAovSchema->initFloatArray(aovs);
            fs.mAovSchema->initFloatArray(deepAovs);
        }

        float depth = scene_rdl2::math::pos_inf;
        float *depthPtr = nullptr;
        if (schema.hasClosestFilter()) {
            // we'll need a depth result
            depthPtr = &depth;
        }

        pbr::ComputeRadianceAovParams aovParams { /* alpha = */ 0.f,
            depthPtr, aovs, deepAovs, params->mDeepVolumeAovs };

        // We add offset based on the machineId under multi-machine context.
        // This is a core idea of how do we compute images by multi-machine.
        // (relataed United States Patent : 11,176,721 : Nov/16/2021)
        unsigned offset =
            (isamp + params->mRealtimeSampleOfs) * params->mRenderNodeTotal
            + params->mRenderNodeSampleOfs;

        const pbr::Sample sample =
            params->mSampler->getPrimarySample(px, py,
                                               fs.mFrameNumber,
                                               offset,
                                               fs.mDofEnabled,
                                               params->mShutterBias);

        ACCUMULATOR_UNPAUSE(*(params->mNonRenderDriverAccumulator));

        scene_rdl2::math::Color subSample =
            fs.mIntegrator->computeRadiance(pbrTls,
                                            px, py,
                                            int(offset),
                                            params->mTotalNumSamples,
                                            sample,
                                            aovParams,
                                            film->getDeepBuffer(),
                                            film->getCryptomatteBuffer());

        ACCUMULATOR_PAUSE(*(params->mNonRenderDriverAccumulator));

#ifdef DEBUG_SAMPLE_REC_MODE
        if (debugSamplesRecArray) {
            debugSamplesRecArray->pixSamples(px, py, isamp,
                                            subSample.r, subSample.g, subSample.b,
                                            aovParams.mAlpha); // for debug
        }
        if (film->getAdaptiveRenderTilesTable()->isDebugPixByPos(px, py)) {
            std::cerr << ">> RenderFrame.cc computeRadiance() result debug-pix(" << px << ',' << py << ')'
                    << " isamp:" << isamp
                    << " c(" << subSample.r << ',' << subSample.g << ',' << subSample.b << ')'
                    << " a:" << aovParams.mAlpha << std::endl;
        }
#endif // end DEBUG_SAMPLE_REC_MODE

        // If alpha is negative then we're dealing with an
        // invalid sample and we should skip adding it and
        // incrementing the sampleCount.
        const float alpha = aovParams.mAlpha;

        if (alpha < 0.f) continue;
        // copy color from unaligned subSample into
        // aligned sample
        scene_rdl2::fb_util::RenderColor sampleResult(subSample.r, subSample.g, subSample.b, alpha);

        accRadiance += sampleResult;

        // Record odd indexed samples in a secondary radiance buffer.
        if (isamp & 1) {
            accRadiance2 += sampleResult;
        }

        ++numAccSamples;

        if (params->mAovNumFloats) {
            unsigned aovFloatIndex = 0;

            // Fill in the AOVs

            for (std::size_t entryIdx = 0; entryIdx < schema.size(); ++entryIdx) {
                const pbr::AovSchema::Entry& entry = schema[entryIdx];
                aovFloatIndex += (entryIdx > 0) ? schema[entryIdx - 1].numChannels() : 0;


#ifdef DEBUG_SAMPLE_REC_MODE
                if (debugSamplesRecArray &&
                    debugSamplesRecArray->mode() == DebugSamplesRecArray::Mode::LOAD) {
                    if (entry.type() == pbr::AOV_TYPE_BEAUTY) {
                        std::memcpy((void *)(aovs + aovFloatIndex),
                                    (const void *)&sampleResult, sizeof(float) * 3);
                    }
                    if (entry.type() == pbr::AOV_TYPE_ALPHA) {
                        aovs[aovFloatIndex] = alpha;
                    }
                }
#endif // end DEBUG_SAMPLE_REC_MODE

                if (!schema.hasAovFilter()) {
                    // if there is no special aov filter, sum
                    // the values
                    for (unsigned j = 0; j < entry.numChannels(); ++j) {
                        const unsigned i = aovFloatIndex + j;
                        localAovs[i] += aovs[i];
                    }
                } else {
                    switch (entry.filter()) {
                        case pbr::AOV_FILTER_AVG:
                        case pbr::AOV_FILTER_SUM:
                            for (unsigned j = 0; j < entry.numChannels(); ++j) {
                                const unsigned i = aovFloatIndex + j;
                                localAovs[i] += aovs[i];
                            }
                            break;
                        case pbr::AOV_FILTER_MIN:
                            for (unsigned j = 0; j < entry.numChannels(); ++j) {
                                const unsigned i = aovFloatIndex + j;
                                localAovs[i] = std::min(localAovs[i], aovs[i]);
                            }
                            break;
                        case pbr::AOV_FILTER_MAX:
                            for (unsigned j = 0; j < entry.numChannels(); ++j) {
                                const unsigned i = aovFloatIndex + j;
                                localAovs[i] = std::max(localAovs[i], aovs[i]);
                            }
                            break;
                        case pbr::AOV_FILTER_FORCE_CONSISTENT_SAMPLING:
                            if (isamp < params->mConsistentSamplesPerPixel) {
                                for (unsigned j = 0; j < entry.numChannels(); ++j) {
                                    const unsigned i = aovFloatIndex + j;
                                    localAovs[i] += aovs[i];
                                }
                            }
                            break;
                        case pbr::AOV_FILTER_CLOSEST:
                            if (depth < localDepth) {
                                for (unsigned j = 0; j < entry.numChannels(); ++j) {
                                    const unsigned i = aovFloatIndex + j;
                                    localAovs[i] = aovs[i];
                                }
                            }
                            break;
                        default:
                            MNRY_ASSERT(0 && "unexpected filter type");
                            break;
                    }
                }
            }

            // update localDepth after all aovs are filled
            if (depth < localDepth) {
                localDepth = depth;
            }

        }  // if (params->mAovNumFloats) {
    }  // end sample loop

    if (numAccSamples) {
        // Update frame buffer. Scale the weights so that they are in
        // the same space as radiance.
        {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_ADD_SAMPLE_HANDLER);
            film->addSamplesToRenderBuffer(px, py, accRadiance,
                    numAccSamples, &accRadiance2);
        }
        // update aovs
        if (params->mAovNumFloats) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            film->addSamplesToAovBuffer(px, py, localDepth, localAovs);
        }
    }

    // conditional heat map collection
    MCRT_COMMON_CLOCK_CLOSE();

    return true;
}

// static function
bool
RenderDriver::renderPixelScalarSamplesFast(pbr::TLState *pbrTls,
                                           const unsigned startSampleIdx,
                                           const unsigned endSampleIdx,
                                           RenderSamplesParams *params)
{
    //
    // Single ray execution start:
    //

    MNRY_ASSERT(startSampleIdx < endSampleIdx);

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);

    unsigned const px = params->mPx;
    unsigned const py = params->mPy;
    Film *film = params->mFilm;
    float *aovs = params->mAovs;

    float *localAovs = params->mLocalAovs;
    float *deepAovs = params->mDeepAovs;
    float localDepth = scene_rdl2::math::pos_inf;

    const rndr::FrameState &fs = *reinterpret_cast<const rndr::FrameState *>(pbrTls->mFs);
    const auto& schema = *fs.mAovSchema;

    MCRT_COMMON_CLOCK_OPEN(fs.mRequiresHeatMap ? &(film->getHeatMap(px, py)) : nullptr);

    scene_rdl2::fb_util::RenderColor accRadiance = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::ZeroTy());
    unsigned numAccSamples = 0;
    if (params->mLocalAovs) fs.mAovSchema->initFloatArray(params->mLocalAovs);

#ifdef DEBUG_SAMPLE_REC_MODE
        DebugSamplesRecArray *debugSamplesRecArray = film->getDebugSamplesRecArray();
#endif // end DEBUG_SAMPLE_REC_MODE

    // Loop over samples in current pixel. It is important to note that all samples within
    // a single pass are guaranteed to be executed on the same thread. No single tile can
    // be executed on different threads at one point in time (the TileWorkQueue ensures
    // this), so we don't need to do any locking here.
    for (unsigned isamp = startSampleIdx; isamp != endSampleIdx; ++isamp) {

        // This line is for supporting the pixel sample map functionality.
        if (isamp >= params->mTotalNumSamples) break;

        CHECK_CANCELLATION(pbrTls, return false);

        // Push/pop memory arena for each sample.
        SCOPED_MEM(arena);

        if (aovs) {
            fs.mAovSchema->initFloatArray(aovs);
            fs.mAovSchema->initFloatArray(deepAovs);
        }

        float depth = scene_rdl2::math::pos_inf;
        float *depthPtr = nullptr;
        if (schema.hasClosestFilter()) {
            // we'll need a depth result
            depthPtr = &depth;
        }

        pbr::ComputeRadianceAovParams aovParams { /* alpha = */ 0.f,
            depthPtr, aovs, deepAovs, params->mDeepVolumeAovs };

        // We add offset based on the machineId under multi-machine context.
        // This is a core idea of how do we compute images by multi-machine.
        // (relataed United States Patent : 11,176,721 : Nov/16/2021)
        unsigned offset =
            (isamp + params->mRealtimeSampleOfs) * params->mRenderNodeTotal
            + params->mRenderNodeSampleOfs;

        const pbr::Sample sample =
            params->mSampler->getPrimarySample(px, py,
                                               fs.mFrameNumber,
                                               offset,
                                               fs.mDofEnabled,
                                               params->mShutterBias);

        ACCUMULATOR_UNPAUSE(*(params->mNonRenderDriverAccumulator));

        scene_rdl2::math::Color subSample =
            fs.mIntegrator->computeColorFromIntersection(pbrTls,
                                                         px, py,
                                                         int(offset),
                                                         params->mTotalNumSamples,
                                                         sample,
                                                         aovParams,
                                                         fs.mFastMode);

        ACCUMULATOR_PAUSE(*(params->mNonRenderDriverAccumulator));

        // If alpha is negative then we're dealing with an
        // invalid sample and we should skip adding it and
        // incrementing the sampleCount.
        const float alpha = aovParams.mAlpha;
        if (alpha >= 0.f) {

            // copy color from unaligned subSample into
            // aligned sample
            scene_rdl2::fb_util::RenderColor sampleResult(subSample.r, subSample.g, subSample.b, alpha);

            accRadiance += sampleResult;

            ++numAccSamples;
        }  // if (alpha >= 0.f)
    }  // end sample loop

    if (numAccSamples) {
        // Update frame buffer. Scale the weights so that they are in
        // the same space as radiance.
        {
            // Use constant black value since we don't care about adaptive samplinng in fast mode
            const scene_rdl2::fb_util::RenderColor blackConst = scene_rdl2::fb_util::RenderColor(scene_rdl2::math::ZeroTy());
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_ADD_SAMPLE_HANDLER);
            film->addSamplesToRenderBuffer(px, py, accRadiance,
                    numAccSamples, &blackConst);
        }
        // update aovs
        if (params->mAovNumFloats) {
            EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
            film->addSamplesToAovBuffer(px, py, localDepth, localAovs);
        }
    }

    // conditional heat map collection
    MCRT_COMMON_CLOCK_CLOSE();

    return true;
}

// static function
bool
RenderDriver::renderPixelVectorSamples(pbr::TLState *pbrTls,
                                       const unsigned startSampleIdx,
                                       const unsigned endSampleIdx,
                                       RenderSamplesParams *params,
                                       const TileGroup &group,
                                       const pbr::DeepBuffer *deepBuffer,
                                       pbr::CryptomatteBuffer *cryptomatteBuffer)
{
    unsigned px = params->mPx;
    unsigned py = params->mPy;
    const rndr::FrameState &fs = *reinterpret_cast<const rndr::FrameState *>(pbrTls->mFs);

    //
    // Bundled execution start:
    //

    uint32_t pixel = pbr::pixelLocationToUint32(px, py);

    // Loop over samples in current pixel.
    unsigned numSamples = endSampleIdx - startSampleIdx;
    if (params->mTotalNumSamples < endSampleIdx) {
        numSamples = params->mTotalNumSamples - startSampleIdx;
    }
    pbr::RayState **rayStates;
    if (numSamples > 0) {
        rayStates = pbrTls->allocRayStates(numSamples);
        for (unsigned i = 0; i < numSamples; i++) {
            pbr::RayState *rs = rayStates[i];
            rs->mDeepDataHandle = pbr::nullHandle;
            rs->mCryptomatteDataHandle = pbr::nullHandle;
        }
    }

    // invalid ray states will not be queued, we need to keep
    // a list of these and free them in bulk.
    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);
    pbr::RayState **rayStatesToFree = arena->allocArray<pbr::RayState*>(numSamples);
    unsigned numRayStatesToFree = 0;

    for (unsigned isub = startSampleIdx; isub != endSampleIdx; ++isub) {

    if (isub >= params->mTotalNumSamples) break;

        CHECK_CANCELLATION(pbrTls, return false);

        // We add offset based on the machineId under multi-machine context.
        // This is a core idea of how do we compute images by multi-machine.
        // (relataed United States Patent : 11,176,721 : Nov/16/2021)
        unsigned offset =
            (isub + params->mRealtimeSampleOfs) * params->mRenderNodeTotal
            + params->mRenderNodeSampleOfs;

        float shutterBias = fs.mScene->getCamera()->getShutterBias();
        const pbr::Sample sample =
            params->mSampler->getPrimarySample(px, py,
                                               fs.mFrameNumber,
                                               offset,
                                               fs.mDofEnabled,
                                               shutterBias);

        pbr::RayState *rs = rayStates[isub - startSampleIdx];

        // Partially fill in RayState data.
        rs->mSubpixel.mPixel = pixel;
        rs->mSubpixel.mSubpixelX = sample.pixelX;
        rs->mSubpixel.mSubpixelY = sample.pixelY;
        rs->mTilePass = pbr::makeTilePass(params->mTileIdx, group.mPassIdx);

        if (deepBuffer != nullptr) {
            rs->mDeepDataHandle = pbrTls->allocList(sizeof(pbr::DeepData), 1);
            pbr::DeepData *deepData = static_cast<pbr::DeepData*>(pbrTls->getListItem(rs->mDeepDataHandle, 0));
            deepData->mRefCount = 1;
            deepData->mHitDeep = 0;
            deepData->mLayer = 0;
        }

        if (cryptomatteBuffer != nullptr) {
            rs->mCryptomatteDataHandle = pbrTls->allocList(sizeof(pbr::CryptomatteData), 1);
            pbr::CryptomatteData *cryptomatteData =
                        static_cast<pbr::CryptomatteData*>(pbrTls->getListItem(rs->mCryptomatteDataHandle, 0));
            cryptomatteData->init(cryptomatteBuffer);

            rs->mCryptoRefP = scene_rdl2::math::Vec3f(0.f);
            rs->mCryptoRefN = scene_rdl2::math::Vec3f(0.f);
            rs->mCryptoUV = scene_rdl2::math::Vec2f(0.f);
        }

        // Queue up new primary ray.
        ACCUMULATOR_UNPAUSE(*(params->mNonRenderDriverAccumulator));
        bool queued =
            fs.mIntegrator->queuePrimaryRay(pbrTls,
                                            px, py,
                                            int(offset),
                                            params->mTotalNumSamples,
                                            sample,
                                            rs);
        if (!queued) {
            rayStatesToFree[numRayStatesToFree++] = rs;
        }
        ACCUMULATOR_PAUSE(*(params->mNonRenderDriverAccumulator));
    } // isub

    // Bulk free of raystates that were not queued
    pbrTls->freeRayStates(numRayStatesToFree, rayStatesToFree);

    //
    // Bundled execution end!
    //
    return true;
}

// static function
void
RenderDriver::computePixelInfo(RenderDriver *driver,
                               mcrt_common::ThreadLocalState *tls,
                               Film &film,
                               const unsigned px,
                               const unsigned py)
{
    const rndr::FrameState &fs = driver->getFrameState();

    // Update pixel info data (pixel center depth) buffer if required.
    if (film.hasPixelInfoBuffer() &&
        (!driver->mCoarsePassesComplete || fs.mRenderMode == RenderMode::BATCH)) {
            // Calculate the depth and add it to mPixelInfoBuffer
        float depth = moonray::pbr::computeOpenGLDepth(tls, fs.mScene, px, py);
        film.setPixelInfo(px, py, scene_rdl2::fb_util::PixelInfo(depth));
    }
}

// static function
unsigned
RenderDriver::computeTotalNumSamples(const rndr::FrameState &fs, const unsigned ifilm, unsigned px, unsigned py)
{
    unsigned totalNumSamples = fs.mMaxSamplesPerPixel;
    if (fs.mPixelSampleMap) {
        const scene_rdl2::fb_util::FloatBuffer *pixelSampleMap = fs.mPixelSampleMap;
        if (pixelSampleMap) {
            totalNumSamples = unsigned(scene_rdl2::math::floor(fs.mOriginalSamplesPerPixel *
                                                   scene_rdl2::math::max(0.0f, pixelSampleMap->getPixel(px,py))));
        }
    }
    return totalNumSamples;
}

} // namespace rndr
} // namespace moonray

