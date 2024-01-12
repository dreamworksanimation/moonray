// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// Progress reporting
// ------------------
// -   In non-bundled mode, we know how many primary samples we want to spawn
//     in total and we know how many have been spawned already. This simple
//     ratio is used to compute the overall progress.
// -   Bundled mode is similar to non-bundled except that total time is the sum
//     of the time taken to spawn all primary rays *plus* the time taken to
//     drain the queues when there are no more primary rays left to spawn.
//     The queue draining time is negligible currently (we default to relative
//     small queue sizes), so is ignored at the moment. Revisit as necessary.
// -   APIs exist to track individual per-pass progress also.
//
#include <scene_rdl2/render/util/AtomicFloat.h> // Needs to be included before any OpenImageIO file
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>

#include "CheckpointSigIntHandler.h"
#include "PixSampleRuntimeVerify.h"
#include "RenderContext.h"
#include "RenderDriver.h"
#include "ResumeHistoryMetaData.h"
#include "TileSampleSpecialEvent.h"

#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/pbr/core/DebugRay.h>

#include <tbb/task_arena.h>

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_SPAN // See RuntimeVerify.h
#define RUNTIME_VERIFY
#endif // end RUNTIME_VERIFY_PIX_SAMPLE_SPAN

namespace moonray {

using namespace mcrt_common;

namespace rndr {

namespace {

inline mcrt_common::ThreadLocalState *
getPrimaryTLS()
{
    // Treat the first TLS instance as our primary one for profiling and
    // cancellation purposes.
    return getTLSList();
}

#ifdef DEBUG
// Can only be called on master render thread.
bool
verifyNoBundledLeaks(const FrameState &fs)
{
    if (fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED) {
        pbr::forEachTLS([&](pbr::TLState *tls) {

            // If the frame finished naturally, we shouldn't have any outstanding allocations.
            MNRY_ASSERT(tls->verifyNoOutstandingAllocs());

            // ... and all TLS queues should be completely empty.
            MNRY_ASSERT(tls->areAllLocalQueuesEmpty());

            for (size_t i = 0; i < MAX_RENDER_PASSES; ++i) {
                MNRY_ASSERT(tls->mPrimaryRaysSubmitted[i] == 0);
            }
        });

        MNRY_ASSERT(shading::Material::areAllShadeQueuesEmpty());
    }

    return true;
}
#endif // end DEBUG

}   // End of anon namespace.

//-----------------------------------------------------------------------------

void
RenderDriver::renderFrame(RenderDriver *driver, const FrameState &fs)
{
#   ifdef VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC
    verifyPassesLogicForAdaptiveSampling();
#   endif // end VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC

    ACCUMULATOR_PROFILE(getPrimaryTLS(), ACCUM_ROOT);

    MNRY_ASSERT(fs.mIntegrator);

    // Start recording end to end frame time for realtime and progressive modes.
    driver->mMcrtStartTime = scene_rdl2::util::getSeconds();

    moonray::util::ProcessUtilization frameStartUtilization = moonray::util::ProcessStats().getProcessUtilization();

    Film *film = driver->mFilm;

    // Initialize TLS array with data for this frame.
    // RayStatePool should be already initialized by this point.
    pbr::forEachTLS([&](pbr::TLState *tls) {

        if (fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED ||
            fs.mExecutionMode == mcrt_common::ExecutionMode::XPU) {
 
            tls->setRadianceQueueHandler(Film::addSampleBundleHandler, film);
            if (fs.mAovSchema->hasAovFilter()) {
                tls->setAovQueueHandler(Film::addFilteredAovSampleBundleHandler, film);
            } else {
                tls->setAovQueueHandler(Film::addAovSampleBundleHandler, film);
            }
            tls->setHeatMapQueueHandler(Film::addHeatMapBundleHandler, film);
        }

        tls->mFs = &fs;

        // Initialize profile accumulators for each thread.
        tls->cacheThreadLocalAccumulators();

        mcrt_common::ExclusiveAccumulators *exclAcc = tls->getInternalExclusiveAccumulators();
        ThreadLocalAccumulator *ispcAcc = tls->mIspcAccumulator;

        auto geomTls = (mcrt_common::BaseTLState *)tls->mTopLevelTls->mGeomTls.get();
        if (geomTls) {
            geomTls->mExclusiveAccumulatorsPtr = exclAcc;
            geomTls->mIspcAccumulator = ispcAcc;
        }

        auto shadingTls = (mcrt_common::BaseTLState *)tls->mTopLevelTls->mShadingTls.get();
        if (shadingTls) {
            shadingTls->mExclusiveAccumulatorsPtr = exclAcc;
            shadingTls->mIspcAccumulator = ispcAcc;
        }

        // For progressive mode when running vectorized, reduce all queue sizes
        // temporarily so that samples are retired in a more predictable order.
        // See RenderDriver::renderPasses() regarding queue size back logic if we've finished coarse pass
        float queueInterp = ((fs.mRenderMode == RenderMode::PROGRESSIVE ||
                              fs.mRenderMode == RenderMode::PROGRESSIVE_FAST ||
                              fs.mRenderMode == RenderMode::PROGRESS_CHECKPOINT) &&
                             driver->getLastCoarsePassIdx() != MAX_RENDER_PASSES)
                             ? 0.f : 1.f;
        tls->setAllQueueSizes(queueInterp);
    });

    // Need to setup the accumulators for the GUI TLS, which is separate from the other TLSes
    pbr::TLState* guiTls = mcrt_common::getGuiTLS()->mPbrTls.get();
    if (guiTls) {
        guiTls->cacheThreadLocalAccumulators();
    }

    // Check to see if we should be recording the rays for this frame.
    if (driver->getDebugRayState() == REQUEST_RECORD) {
        pbr::forEachTLS([&](pbr::TLState *tls) {
            MNRY_ASSERT(tls->mRayVertexStack.empty());
            pbr::DebugRayRecorder *recorder = tls->mRayRecorder;
            recorder->record();
        });
        driver->switchDebugRayState(REQUEST_RECORD, RECORDING);
    }

    // Everything should have been cleaned up from the previous frame, verify that's still the case.
    MNRY_ASSERT(verifyNoBundledLeaks(fs));

    // We are guaranteed to have copied the previous frame buffer by this point if we needed it.
    film->clearAllBuffers();

    // Disable adjust adaptive tree update timing logic at this moment. we will enable later for checkpoint
    film->disableAdjustAdaptiveTreeUpdateTiming();

    /* useful debug code for adaptive render debug pixel (w/ non resume render)
    film->getAdaptiveRenderTilesTable()->setDebugPosition(*driver->getTiles()); // for debug
    */

    // This should be before workQueue reset because revert film object might change workQueue parameters.
    unsigned progressCheckpointStartTileSampleId = revertFilmObjectAndResetWorkQueue(driver, fs);

    if (fs.mSamplingMode == SamplingMode::ADAPTIVE) {
        // Initialize current sampleId buffer for adaptive sampling.
        film->getCurrSampleIdBuff().init(film->getWeightBuffer(), fs.mMaxSamplesPerPixel);
    }

    TileWorkQueue *workQueue = &driver->mTileWorkQueue;
    workQueue->reset();

    // Turn off extrapolation if there are no coarse passes.
    if (driver->mLastCoarsePassIdx == MAX_RENDER_PASSES) {
        MNRY_ASSERT(workQueue->getPass(0).isFinePass());
        driver->setCoarsePassesComplete();
    }

    //
    // Start rendering the frame.
    //

    bool canceled = false;

    switch(fs.mRenderMode) {
    case RenderMode::BATCH:
        canceled = !batchRenderFrame(driver, fs);
        break;

    case RenderMode::PROGRESSIVE:
        canceled = !progressiveRenderFrame(driver, fs);
        break;

    case RenderMode::PROGRESSIVE_FAST:
        canceled = !progressiveRenderFrame(driver, fs);
        break;

    case RenderMode::REALTIME:
        realtimeRenderFrame(driver, fs);
        break;

    case RenderMode::PROGRESS_CHECKPOINT: {
        const bool pgEnabled = fs.mIntegrator->getEnablePathGuide(); // is path guiding enabled?
        std::unique_ptr<TileSampleSpecialEvent> tileSampleSpecialEvent;
        if (pgEnabled) {
            //
            // When PathGuiding case, we set up TileSampleSpecialEvent information to the
            // checkpoint rendering main logic.
            //
            auto genSpecialEventTileSampleIdTable = [&](const unsigned maxPixSamples) -> UIntTable {
                // We would like to execute special event when we finished following pixel samples on each
                // pixels under PathGuiding context.
                // pixSample total = {8, 10, 14, 22, 38, 70, 134, 262, 518, 1030, 2054, 4102, ...}
                // (start from 8 as initial value, pixel sample delta step is start step=2 and x2 at every
                // iteration.)
                //
                // So these number should be converted to tile based sample Id (i.e. pixSample * 64 - 1)
                // tileSample id = {511, 639, 897, 1407, 2431, 4479, 8575, 16767, 33151, 65919, 131455,
                // 262527, ...}. Following logic constructs this tileSample id table
                //
                // On each tile rendering, after finish special tile sample id which provided by return of
                // this function, renderer calls call back function as special event (in this case, passReset())
                // regardless of original tile sampling schedule.
                // This means, passReset() is executed at just finished after every tile sample Id's you
                // provided by this function.
                //
                // If you need other interval control for call back. it's easy and you just change logic
                // how to generate table here.
                unsigned pixSampleId = 8;
                unsigned pixSampleSteps = 2;
                UIntTable table;
                table.push_back(pixSampleId * 64 - 1); // tile sample number (i.e. not pixel samples)
                while (1) {
                    pixSampleId += pixSampleSteps;
                    table.push_back(pixSampleId * 64 - 1); // convert to tile sample Id
                    if (pixSampleId >= maxPixSamples) break;
                    pixSampleSteps *= 2;
                }
                return table;
            };

            tileSampleSpecialEvent.reset
                (new TileSampleSpecialEvent
                 (genSpecialEventTileSampleIdTable(fs.mMaxSamplesPerPixel),
                  [&](const unsigned sampleId) -> bool {
                     // setup CallBack function which is executed at after each sampleId inside
                     // TileSampleIdTable

                     // we are responsible for thread-safety
                     const_cast<pbr::PathIntegrator *>(fs.mIntegrator)->passReset();

                     return true;
                 }));
        }

        if (driver->mCheckpointController.isMemorySnapshotActive()) {
            CheckpointSigIntHandler::enable();
            scene_rdl2::logging::Logger::info("enable signal-based checkpoint");
        } else {
            scene_rdl2::logging::Logger::info("disable signal-based checkpoint");
        }
        canceled = !progressCheckpointRenderFrame(driver, fs, progressCheckpointStartTileSampleId,
                                                  tileSampleSpecialEvent.get());
        if (driver->mCheckpointController.isMemorySnapshotActive()) {
            CheckpointSigIntHandler::disable();
        }
    } break;

    default:
        MNRY_ASSERT(0);
    }

    // This must always be updated before we leave this function.
    driver->setReadyForDisplay();

#   ifdef RUNTIME_VERIFY
    if (fs.mSamplingMode == SamplingMode::ADAPTIVE) {
        if (!film->getCurrSampleIdBuff().verify(film->getWeightBuffer())) {
            std::cerr << ">> RenderFrame.cc sampleId verify NG\n";
        } else {
            std::cerr << ">> RenderFrame.cc sampleId verify OK\n";
        }
    }
    {
        const SampleIdBuff *startSampleIdBuff = nullptr;
        unsigned startSampleId = 0;
        if (fs.mRenderContext->getSceneContext().getResumeRender()) {
            if (film->getResumeStartSampleIdBuff().isValid()) {
                startSampleIdBuff = &(film->getResumeStartSampleIdBuff());
            } else {
                startSampleId = progressCheckpointStartTileSampleId / 64; // 64 = tileWidht * tileHeight
            }
        }
        PixSampleSpanRuntimeVerify::get()->verify(startSampleIdBuff, startSampleId, film->getWeightBuffer());

        std::cerr << ">> RenderFrame.cc "
                  << PixSampleSpanRuntimeVerify::get()->show(436, 106,
                                                             startSampleIdBuff, startSampleId,
                                                             film->getWeightBuffer()) << '\n';
    }
#   endif // end RUNTIME_VERIFY

    //
    // Frame clean up.
    //

    if (fs.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED ||
        fs.mExecutionMode == mcrt_common::ExecutionMode::XPU) {
        if (canceled) {
            shading::forEachShadeQueue(nullptr, [](mcrt_common::ThreadLocalState *tls, shading::ShadeQueue *queue) {
                queue->reset();
            });
        } else {
            MNRY_ASSERT(verifyNoBundledLeaks(fs));
        }

        pbr::resetPools();
        shading::Material::printDeferredEntryWarnings();
        shading::Material::resetDeferredEntryState();
    }

    // If we were recording rays, we're done now
    if (driver->getDebugRayState() == RECORDING) {
        pbr::forEachTLS([&](pbr::TLState *tls) {
            pbr::DebugRayRecorder *recorder = tls->mRayRecorder;
            recorder->stopRecording();
            tls->mRayVertexStack.clear();
        });
        driver->switchDebugRayState(RECORDING, RECORDING_COMPLETE);
    }

    // Reset all TLS objects.
    forEachTLS([canceled](ThreadLocalState *tls) {
        if (!canceled) {
            MNRY_ASSERT_REQUIRE(tls->mHandlerStackDepth == 0);
        }
        tls->reset();
        MNRY_ASSERT(tls->mShadingTls->mOIIOThreadData == nullptr);
    });

    // Update frame duration stats.
    moonray::util::ProcessUtilization us = moonray::util::ProcessStats().getProcessUtilization();

    double endTime = scene_rdl2::util::getSeconds();
    double renderTime = endTime - driver->mMcrtStartTime;

    driver->mMcrtDuration = renderTime;
    driver->mMcrtUtilization =
        (us.getUserSeconds(frameStartUtilization) +
         us.getSystemSeconds(frameStartUtilization)) * 100 / renderTime;
}

// Returns true if we ran to completion or false if we were canceled.
bool
RenderDriver::batchRenderFrame(RenderDriver *driver, const FrameState &fs)
{
    RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();
    timingRec.reset(0);            // reset condition for new frame

    // store timing info for resume history
    fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintStartTime();

    // Submit all passes with cancellation.
    RenderPassesResult result = renderPasses(driver, fs, true);

    if (result != RenderPassesResult::ERROR_OR_CANCEL) {
        driver->setReadyForDisplay();
        driver->setFrameComplete();
    }

    // This is a debug purpose code and only support fileId = 0 so far.
    if (driver->mFilm->getDebugSamplesRecArray()) {
        std::cerr << ">> RenderFrame.cc batchRenderFrame completed and save DebugSamplesRecArray." << std::endl;
        std::cerr << driver->mFilm->getDebugSamplesRecArray()->show("") << std::endl;
        if (!driver->mFilm->getDebugSamplesRecArray()->save("./tmp.samples")) {
            std::cerr << ">> RenderFrame.cc DebugSamplesRecArray() failed." << std::endl;
        }
    }

    // store timing info for resume history : set dummy end tile sample id
    fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintEndTime(0, false);

    if (result == RenderPassesResult::ERROR_OR_CANCEL) return false;
    return true;
}

void
RenderDriver::runDisplayFiltersEndOfPass(RenderDriver *driver, const FrameState &fs)
{
    fs.mRenderContext->snapshotAovsForDisplayFilters(true, true);
    const DisplayFilterDriver& displayFilterDriver = driver->getDisplayFilterDriver();
    simpleLoop (true, 0u, (unsigned int)driver->getTiles()->size() - 1u, [&](unsigned tileIdx) {
        int threadId = tbb::task_arena::current_thread_index();
        displayFilterDriver.runDisplayFilters(tileIdx, threadId);
    });
}

// Returns true if we ran to completion or false if we were canceled.
bool
RenderDriver::progressiveRenderFrame(RenderDriver *driver, const FrameState &fs)
{
    RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();
    timingRec.reset(0);            // reset condition for new frame

    // store timing info for resume history
    fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintStartTime();

    TileWorkQueue *workQueue = &driver->mTileWorkQueue;

    // Only allow rendering of pass 0 initially. This will allow us to present a new frame
    // up on screen quickly. Pass 0 is dynamically configured with the same rules as for
    // realtime rendering.
    int clampPass = 0;
    workQueue->clampToPass(clampPass);

    // The passes array should be configured with a single pass containing just the
    // correct amount of samples we need to render to fit within our allocated frame time.
    renderPasses(driver, fs, false);

    // Check if we still need to run extrapolation before displaying.
    if (workQueue->getNumPasses() > 1) {
        const Pass &pass = workQueue->getPass(1);
        if (pass.isFinePass()) {
            driver->setCoarsePassesComplete();
        }
    }

    driver->setReadyForDisplay();
    if (driver->mStopAtFrameReadyForDisplay) {
        return false;
    }

    // Don't render any more if user has triggered a cancel.
    if (workQueue->getNumPasses() > 0) {

        if (!getPrimaryTLS()->mPbrTls->isCanceled()) {
            bool enablePathGuide = fs.mIntegrator->getEnablePathGuide();
            // Clamp coarse passes for display filters. Some pixels do not
            // yet have data during coarse passes so display filters
            // must be run at the end of the pass.
            bool hasDisplayFilters = driver->getDisplayFilterDriver().hasDisplayFilters()
                && !driver->areCoarsePassesComplete();
            bool clampPasses = enablePathGuide || hasDisplayFilters;
            if (clampPasses) {
                for (clampPass = 1; clampPass < workQueue->getNumPasses(); ++clampPass) {
                    // we need to reset the path guide after each pass
                    // we are responsible for ensuring thread-safety
                    if (enablePathGuide) {
                        const_cast<pbr::PathIntegrator *>(fs.mIntegrator)->passReset();
                    }
                    workQueue->clampToPass(clampPass);
                    RenderPassesResult result = renderPasses(driver, fs, true);
                    if (hasDisplayFilters) {
                        runDisplayFiltersEndOfPass(driver, fs);
                    }
                    if (result == RenderPassesResult::ERROR_OR_CANCEL) {
                        return false;
                    }
                    if (result == RenderPassesResult::STOP_AT_PASS_BOUNDARY) {
                        break; // render completed at pass boundary condition -> exit loop
                    }
                }
            } else {
                // Allow rendering of all remaining passes.
                workQueue->unclampPasses();

                // Render all remaining passes.
                RenderPassesResult result = renderPasses(driver, fs, true);
                if (result == RenderPassesResult::ERROR_OR_CANCEL) {
                    return false;
                }
            }
        }
    }

    // Run final DisplayFilter pass after the last render pass.
    // When the display filters are run during a pass they may
    // use outdated data from neighboring tiles.
    if (driver->getDisplayFilterDriver().hasDisplayFilters()) {
        // Request to update all tiles
        unsigned numTiles = driver->getFilm().getTiler().mNumTiles;
        for (unsigned tile = 0; tile < numTiles; ++tile) {
            driver->getDisplayFilterDriver().requestTileUpdate(tile);
        }

        runDisplayFiltersEndOfPass(driver, fs);
    }

    // Mark frame as fully complete.
    driver->setFrameComplete();

    // store timing info for resume history : set dummy end tile sample id
    fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintEndTime(0, false);
    return true;
}

// static function
// Returns true if we ran to completion or false if we were canceled.
void
RenderDriver::realtimeRenderFrame(RenderDriver *driver, const FrameState &fs)
{
    RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();
    timingRec.reset(0);   // reset condition for new frame : set renderFrameStartTime internally

    TileWorkQueue *workQueue = &driver->mTileWorkQueue;

    // There should only be a single coarse pass at this point. That pass is
    // assumed to only render a single sample per tile.
    MNRY_ASSERT(workQueue->getNumPasses() == 1);
    MNRY_ASSERT(workQueue->getPass(0).getNumSamplesPerTile() == 1);

    RealtimeFrameStats &rts = driver->getCurrentRealtimeFrameStats();
    rts.mRenderFrameStartTime = timingRec.getRenderFrameStartTime();

    // Compute end time for this frame
    {
        // Use information from the previous frame to derive how much time we've spent in the update portion of the
        // frame for this frame. Also accounts duration offset here. Duration offset sec is a extra time which need to
        // add to next frame time budget and defined by outside of renderDriver.
        // For more information about duration offset, please see GapInterval of McrtRtComputation
        // (moonray/dso/computation/mcrt_rt/McrtRtComputationRealtimeController.h)
        double lastFrameUpdateDuration = timingRec.getLastFrameUpdate() - driver->getLastFrameUpdateDurationOffset();
        double frameBudget = std::max(0.0, (1.0 / double(fs.mFps)) - std::max(lastFrameUpdateDuration, 0.0)); // sec
        double predictedEnd = timingRec.setRenderFrameTimeBudget(frameBudget);

        // update information for logging
        rts.mUpdateDuration = timingRec.getLastFrameUpdate();
        rts.mUpdateDurationOffset = driver->getLastFrameUpdateDurationOffset();
        rts.mRenderBudget = frameBudget;
        rts.mPredictedEndTime = predictedEnd;

        driver->mFrameEndTime = predictedEnd;
    }

    timingRec.setRenderPassesSamplesPerTile(1); // set sample total as 1 for estimation stage

    // The TileWorkQueue is already configured with a single pass which renders
    // a single sample per tile. We use this as our baseline sample cost estimate.
    renderPasses(driver, fs, false); // no cancelation requested

    rts.mPredictedSampleCost = timingRec.getEstimatedSampleCost();
    rts.mFirstPassStartTime = timingRec.getPasses(0).getStartTime();
    rts.mFirstPassEndTime = timingRec.getPasses(0).getEndTime();

    unsigned oldSamplesPerTileRendered = 0;
    const unsigned numTiles = unsigned(driver->getTiles()->size());

    while (1) {
        double now = scene_rdl2::util::getSeconds();

        // mFrameEndTime is completely dynamic, it is updated by calling
        // RenderDriver::requestStop or RenderDriver::stop.
        double remainingTime = driver->mFrameEndTime - now;

        // Do we have time to render more samples?
        if (timingRec.isComplete(remainingTime, now)) {
            break;
        }

        //
        // The paranoiaFactor variable is a factor for how conservative we
        // want to be with the remaining time available:
        //
        // - 0.0 is the least conservative/paranoid. Here we assume the current
        //       time estimate is accurate and try and render as many samples as
        //       possible without going over budget. This makes the best use
        //       of the remaining time but we risk exceeding our budget in
        //       some cases.
        //
        // - 1.0 is the most conservative/paranoid which will always render a
        //       single sample per tile regardless of the amount of time left.
        //       We lose coherence by doing this and spend more time flushing
        //       queues than is optimal, but we are also the least likely to
        //       exceed our time budget.
        //
        // TODO: We could refine the paranoia factor dynamically get larger
        //       as more samples are rendered. The more samples rendered,
        //       the higher confidence we can have in the cost estimate
        //       (since we're refining it continually for each new pass).
        //
        const double paranoiaFactor = 0.0;

        // Compute how many new samples we can safely render within the
        // remaining time interval.
        double estimatedSamplesPerTile = timingRec.estimateSamples(remainingTime);

        // How many *additional* samples per tiles should we attempt to render.
        unsigned numNewSamplesPerTile = scene_rdl2::math::max(unsigned(estimatedSamplesPerTile * (1.0 - paranoiaFactor)), 1u);
        timingRec.setRenderPassesSamplesPerTile(numNewSamplesPerTile);

        // Generate new pass (note: pixels wrap around after we exceed 64).
        Pass newPass;
        newPass.mStartPixelIdx = timingRec.getNumSamplesPerTile();
        newPass.mEndPixelIdx = timingRec.getNumSamplesPerTile() + numNewSamplesPerTile;
        newPass.mStartSampleIdx = 0;
        newPass.mEndSampleIdx = 1;

        // Add new samples to the tile work queue.
        workQueue->init(RenderMode::REALTIME, numTiles, 1, getNumTBBThreads(), &newPass);

        // Render new pass and drain queues.
        renderPasses(driver, fs, false); // no cancelation requested

        // Check if we still need to run extrapolation before displaying.
        if (oldSamplesPerTileRendered < 64 && timingRec.getNumSamplesPerTile() >= 64) {
            driver->setCoarsePassesComplete();
        }
        oldSamplesPerTileRendered = timingRec.getNumSamplesPerTile();
    }

    // update information for logging
    rts.mActualSampleCost = timingRec.actualSampleCost();
    rts.mNumRenderPasses = timingRec.getNumPassesRendered();
    rts.mSamplesPerTile = timingRec.getNumSamplesPerTile();
    rts.mSamplesAll = timingRec.getNumSamplesAll();
    rts.mActualEndTime = timingRec.getRenderFrameEndTime();
    rts.mOverheadDuration = timingRec.getTotalOverheadDuration();

    driver->setReadyForDisplay();

    // Mark frame as fully complete.
    driver->setFrameComplete();
}

} // namespace rndr
} // namespace moonray

