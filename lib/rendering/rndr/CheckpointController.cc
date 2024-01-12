// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "CheckpointController.h"
#include "ImageWriteDriver.h"
#include "RenderContext.h"
#include "RenderOutputDriver.h"

#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/render/util/TimeUtil.h>

#include <iomanip>
#include <sstream>

//#define DEBUG_MSG

#ifdef DEBUG_MSG
#include <iostream>
#endif // end DEBUG_MSG

namespace {

std::string    
secToMsStr(float sec)
{
    std::ostringstream ostr;
    ostr << std::setw(15) << std::fixed << std::setprecision(3) << (sec * 1000.0f) << " ms";
    return ostr.str();
}

} // namespace

namespace moonray {
namespace rndr {

std::string
CheckpointRayCostEvalEvent::show() const
{
    std::ostringstream ostr;
    ostr << "RayCostEvalEvent {"
         << "deltaTime:" << secToMsStr(mDeltaSec) << ' '
         << "SampleId (start:" << std::setw(7) << mDeltaSampleStartId << ' '
         << "end:" << std::setw(7) << mDeltaSampleEndId << ' '
         << "delta:" << std::setw(7) << getDeltaSamples() << ")}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------    

void
CheckpointRayCostEstimator::push(float deltaSec, // Time (sec) used for the sampling operation
                                 unsigned deltaSampleStartId, // tile sampleId
                                 unsigned deltaSampleEndId) // tile sampleId
{
    // Store sampling event info to the top of the list. The front item is new and the back item is old.
    mEventList.emplace_front
        (std::make_shared<CheckpointRayCostEvalEvent>(deltaSec, deltaSampleStartId, deltaSampleEndId));

    // We only keep up to some amount of history (i.e. we don't keep all of the histories).
    while (oldestEventRemoveTest()) {
        mEventList.pop_back();
    }

#   ifdef DEBUG_MSG
    std::cerr << ">> CheckpointController.cc CheckpointRayCostEstimator::push() " << show() << '\n';
#   endif // end DEBUG_MSG
}

float
CheckpointRayCostEstimator::estimateRayCost() const
//
// Simply calculate average ray cost based on the history information
//    
{
    float totalSec = 0.0f;
    unsigned totalSamples = 0;
    for (auto itr = mEventList.begin(); itr != mEventList.end(); ++itr) {
        totalSec += (*itr)->getDeltaSec();
        totalSamples += (*itr)->getDeltaSamples();
    }

    if (totalSamples == 0) {
        return 0.0f;            // special case
    }

    return totalSec / static_cast<float>(totalSamples);
}

std::string
CheckpointRayCostEstimator::show() const
{
    float totalDelta = 0.0f;
    unsigned totalSamples = 0;

    std::ostringstream ostr;
    ostr << "CheckpointRayCostEstimator {\n";
    for (auto itr = mEventList.begin(); itr != mEventList.end(); ++itr) {
        ostr << scene_rdl2::str_util::addIndent((*itr)->show()) << '\n';

        totalDelta += (*itr)->getDeltaSec();
        totalSamples += (*itr)->getDeltaSamples();
    }
    ostr << "}"
         << " totalDelta:" << totalDelta << " sec"
         << " totalSamples:" << totalSamples
         << " estimateRayCost:" << secToMsStr(estimateRayCost());
    return ostr.str();
}

bool
CheckpointRayCostEstimator::oldestEventRemoveTest()
{
    if (mEventList.size() <= 1) {
        // We should try to keep at least 1 event regardless of its sample condition.
        // If the event list size is 1 or less, we don't need to remove them.
        return false; // not remove
    }

    // This number was defined in a heuristic way based on several test scenes.
    // We would keep at least 4spp sampling results in the list in this case.
    constexpr unsigned thresholdSamples = 64 * 4; // 4 SPP

    unsigned totalSamples = calcDeltaSamplesTotal();
    if (totalSamples <= thresholdSamples) {
        return false; // keep all events if total samples are not enough
    }
    
    unsigned oldestSamples = mEventList.back()->getDeltaSamples();
    if ((totalSamples - oldestSamples) < thresholdSamples) {
        return false;
    }
    return true; // we should remove oldest event item
}

unsigned
CheckpointRayCostEstimator::calcDeltaSamplesTotal()
//
// Returns sampling total number based on the history info
//    
{
    unsigned total = 0;
    for (auto itr = mEventList.begin(); itr != mEventList.end(); ++itr) {
        total += (*itr)->getDeltaSamples();
    }
    return total;
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------    

void
CheckpointSnapshotEstimator::set(float snapshotIntervalMinute, float snapshotOverheadFraction)
{
    mSnapshotIntervalSec = std::max(snapshotIntervalMinute * 60.0f, 0.0f);
    mSnapshotOverheadThresholdFraction = std::max(snapshotOverheadFraction, 0.0f);

    if (isActive()) {
        // do extra snapshot for signal
        std::ostringstream ostr;
        ostr << "extra-snapshot parameters ";
        if (snapshotIntervalMinute > 0.0f) {
            ostr << "snapshotIntervalMinute:" << mSnapshotIntervalSec << " sec";
        } else {
            ostr << "snapshotOverheadFraction:" << mSnapshotOverheadThresholdFraction;
        }
        scene_rdl2::logging::Logger::info(ostr.str());
    } else {
        scene_rdl2::logging::Logger::info("no-extra-snapshot");
    }
}

bool
CheckpointSnapshotEstimator::isActive() const
//
// Returns status of we do snapshot action (true) or not (false)
//        
{
    return (mSnapshotIntervalSec > 0.0f || mSnapshotOverheadThresholdFraction > 0.0f);
}

void
CheckpointSnapshotEstimator::reset()
{
    mEstimateIntervalSec = 0.0f;
    mEventList.clear();
}

void
CheckpointSnapshotEstimator::pushSnapshotCost(float snapshotSec)
//
// Stores single snapshot cost event into the history list and maintains history list size as reasonable.
//    
{
    if (mSnapshotIntervalSec > 0.0f) {
        // snapshotInterval sec is defined and we don't need to tracking snapshot sec
        return;
    }

    mEventList.emplace_front(snapshotSec);

    constexpr size_t maxKeep = 10;
    while (mEventList.size() > maxKeep) {
        mEventList.pop_back();
    }

#   ifdef DEBUG_MSG    
    std::cerr << ">> CheckpointController.cc snapshotEstimator::pushSnapshotCost() " << show() << '\n';
#   endif // end DEBUG_MSG
}

float
CheckpointSnapshotEstimator::estimateSnapshotInterval()
//
// Calculates the best guess of snapshot interval based on the history information.
//    
{
    if (mSnapshotIntervalSec > 0.0f) {
        return mSnapshotIntervalSec; // snapshotInterval sec is defined. We don't need to estimate it.
    }

    float snapshotSec = estimateSnapshotSec();
    if (snapshotSec == 0.0f || mSnapshotOverheadThresholdFraction == 0.0f) {
        // We don't have any guess of the snapshot cost yet (This happens when history info is empty).
        // Or threshold fraction is not defined (Safety logic to avoid divide by ZERO).
        // This initial value was selected based on the 50AOVs with HD resolution scene.
        // I hope this value is not so bad.
        mEstimateIntervalSec = 30.0f;
    } else {
        // We have a cap to increase interval length from the previous value.
        // This scaleMax value was heuristically defined based on several tests.
        // We are concerned only if the interval is too long.
        // Actually, we don't care about too short intervals because too short interval does not mak
        // any serious issues to the micro checkpoint loop logic itself.
        // (RenderDriver::checkpointRenderMicroStintLoop())
        constexpr float scaleMax = 2.5f;

        float intervalSec = snapshotSec / mSnapshotOverheadThresholdFraction;
        mEstimateIntervalSec = std::min(mEstimateIntervalSec * scaleMax, intervalSec);
    }
    return mEstimateIntervalSec;
}

std::string    
CheckpointSnapshotEstimator::show() const
{
    float totalSec = 0.0f;
    float estimateSec = estimateSnapshotSec();

    std::ostringstream ostr;
    ostr << "CheckpointSnapshotEstimator {\n";
    for (auto itr = mEventList.begin(); itr != mEventList.end(); ++itr) {
        ostr << "  " << (*itr) << " sec\n";
        totalSec += (*itr);
    }
    ostr << "} totalSec:" << totalSec << " sec"
         << " estimateSnapshot:" << estimateSec << " sec (" << secToMsStr(estimateSec) << ")";
    return ostr.str();
}

float
CheckpointSnapshotEstimator::estimateSnapshotSec() const
//
// Return averaged snapshot time based on the history information
//    
{
    if (mEventList.size() == 0) {
        return 0.0f;            // special case
    }

    float totalSec = 0.0f;
    for (auto itr = mEventList.begin(); itr != mEventList.end(); ++itr) {
        totalSec += (*itr);
    }
    return totalSec / static_cast<float>(mEventList.size());
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

void
CheckpointController::set(float snapshotIntervalMinute,
                          float snapshotOverheadFraction)
{
    mSnapshotEstimator.set(snapshotIntervalMinute, snapshotOverheadFraction);
}

void
CheckpointController::reset()
{
    mRemainingIntervalSec = mSnapshotEstimator.estimateSnapshotInterval();

    mCurrDeltaSampleStartId = 0;
    mCurrDeltaSampleEndId = 0;
    mMaxDeltaSamples = 1;

    mRayCostEstimator.reset();

    mLastSnapshotIntervalSec = 0.0f;
    mSnapshotIntervalTime.start(); // for debug
}

bool
CheckpointController::isMemorySnapshotActive() const
{
    return mSnapshotEstimator.isActive();
}

unsigned
CheckpointController::estimateEndSampleId(const unsigned startSampleId, const unsigned endSampleId)
//
// Returns current best guess of next end tile sampleId for micro checkpoint loop based on the history.
//
{
    auto sampleClip = [&](unsigned samples) -> unsigned {
        //
        // We have a cap to increase sample delta.
        // The new sample total should be less than some scaled value of max sample deltas from the frame
        // start. This scale value (i.e. maxScale) was heuristically defined based on the several test
        // scenes.
        // This idea is safety logic to avoid executing too many samples in the early stage of rendering.
        // In most cases, the estimated sample delta is not accurate due to a lack of statistical information
        // at the early stage of rendering. However, the estimated result would be enhanced time goes by and
        // brush up to the pretty accurate value in the end.
        //        
        constexpr float maxScale = 2.5f;
        unsigned maxSamples =
        static_cast<unsigned>(static_cast<float>(std::max(mMaxDeltaSamples, 1U)) * maxScale);
        return scene_rdl2::math::clamp(samples, 1U, maxSamples);
    };

    float estimatedRayCost = mRayCostEstimator.estimateRayCost();
    unsigned deltaSamples = 1;
    if (estimatedRayCost > 0.0f) {
        deltaSamples = sampleClip(mRemainingIntervalSec / estimatedRayCost);

        // Tracking max delta samples value. This is important for sampleClip() operation
        if (deltaSamples > mMaxDeltaSamples) mMaxDeltaSamples = deltaSamples;
    }

    mCurrDeltaSampleStartId = startSampleId;
    mCurrDeltaSampleEndId = startSampleId + deltaSamples;
    if (endSampleId < mCurrDeltaSampleEndId) mCurrDeltaSampleEndId = endSampleId;

#   ifdef DEBUG_MSG
    std::cerr << ">> CheckpointController.cc estimateEndSampleId"
              << " rayCost:" << estimatedRayCost
              << " startId:" << mCurrDeltaSampleStartId
              << " endId:" << mCurrDeltaSampleEndId
              << " delta:" << (mCurrDeltaSampleEndId - mCurrDeltaSampleStartId)
              << " maxDelta:" << mMaxDeltaSamples
              << '\n';
#   endif // end DEBUG_MSG

    return mCurrDeltaSampleEndId;
}

void
CheckpointController::microStintStart()
{
    mRayCostEvalTime.start();
}

bool
CheckpointController::microStintEnd()
{
    float deltaSec = mRayCostEvalTime.end();
    mRayCostEstimator.push(deltaSec, mCurrDeltaSampleStartId, mCurrDeltaSampleEndId);

    mRemainingIntervalSec -= deltaSec;
    if (mRemainingIntervalSec <= 0.0f) {
        mLastSnapshotIntervalSec = mSnapshotIntervalTime.end(); // for debug
        return true;            // we need snapshot
    }
    return false;
}

void
CheckpointController::snapshotOnly(RenderContext *renderContext,
                                   const std::string &checkpointPostScript,
                                   const unsigned endSampleId)
//
// Creates new ImageWriteCache for snapshot action and stores it properly.
// No file output operation itself, just snapshot only.
//
{
    scene_rdl2::rec_time::RecTime time;
    time.start();
    fileOutputMain(true, // checkpointBgWrite
                   true, // twoStageOutput
                   true, // snapshotOnly
                   renderContext,
                   checkpointPostScript,
                   endSampleId);
    float snapshotActionSec = time.end();
    mSnapshotEstimator.pushSnapshotCost(snapshotActionSec);

    resetRemainingIntervalSec();

    //
    // output extra snapshot detailed info to the log
    //
    float overhead = snapshotActionSec / mLastSnapshotIntervalSec * 100.0f;
    std::ostringstream ostr;
    ostr << "executed extra snapshot "
         << scene_rdl2::time_util::timeStr(scene_rdl2::time_util::getCurrentTime())
         << " snapshot:" << snapshotActionSec << " sec"
         << " interval:" << mLastSnapshotIntervalSec << " sec"
         << " overhead:" << std::setw(6) << std::fixed << std::setprecision(3) << overhead << "%";
    // ostr << '\n' << ImageWriteDriver::get()->showMemUsage(); // for debug
    scene_rdl2::logging::Logger::info(ostr.str());
}

void
CheckpointController::output(bool checkpointBgWrite,
                             bool twoStageOutput,
                             RenderContext *renderContext,
                             const std::string &checkpointPostScript,
                             const unsigned endSampleId)
//
// This is a standard image output action that includes both checkpoint and non-checkpoint situations. 
//
{
    fileOutputMain(checkpointBgWrite,
                   twoStageOutput,
                   false, // snapshotOnly
                   renderContext,
                   checkpointPostScript,
                   endSampleId);

    resetRemainingIntervalSec();

#   ifdef DEBUG_MSG
    std::cerr << ">> CheckpointController.cc CheckpointController::output() "
              << scene_rdl2::time_util::timeStr(scene_rdl2::time_util::getCurrentTime()) << '\n';
#   endif // end DEBUG_MSG
}

//------------------------------------------------------------------------------------------

void
CheckpointController::resetRemainingIntervalSec()
{
    mRemainingIntervalSec = mSnapshotEstimator.estimateSnapshotInterval();
    mSnapshotIntervalTime.start(); // for debug
}

//
// CheckpointController file output main logic
//
// Implementation for writing logic is a bit complicated.
// We have to consider 3 flags (scene_variables) condition.
// They are checkpoint_bg_write, two_stage_output and checkpoint_overwrite scene_variable.
// So, we have to think about  2 * 2 * 2 = 8 cases total.
// Multi-version file specification does not have any impact on the output logic itself.
// a multi-version file only affects the name of the output checkpoint filename.
//
// Following list is detailed information about flag condition, output files and output detail
// procedure which includes write runMode (STD, ENQ, and DEQ) and finalize file (cp/rename).
// We have total 8 types (A ~ H).
//   bg = checkpoint_bg_write scene_variable condition
//   two = two_stage_output scene_variable condition
//   overwrite = checkpoint_overwrite scene_variable condition.
//
// A) bg=off, two=off, overwrite=off
//   out = checkpoint + multi-ver-checkpoint
//   procedure : We need 2 STD actions.
//     STD-0 : standard checkpoint file
//     STD-1 : multi-version checkpoint file
//
// B) bg=off, two=off, overwrite=on
//   out = checkpoint (no multi-version checkpoint file)
//   procedure : We only need 1 STD action.
//     STD : standard checkpoint file
//
// C) bg=off, two=on, overwrite=off
//   out = checkpoint + multi-ver-checkpoint
//   procedure : We need 1 STD and 2 cp/rename actions.
//     STD : tmpFile output due to two stage output on
//     cp/rename-0 : finalize for standard checkpoint file
//     cp/rename-1 : finalize for multi-version checkpoint file
//
// D) bg=off, two=on, overwrite=on
//   out = checkpoint (no multi-version checkpoint file)
//   procedure : We need 1 STD and 1 cp/rename actions.
//     STD : tmpFile output due to two stage output on
//     cp/rename : finalize for standard checkpoint file
//
// E) bg=on, two=off, overwrite=off
//   out = checkpoint + multi-ver-checkpoint
//   procedure : We need 1 ENQ and 2 DEQ actions.
//     ENQ : standard ENQ action
//     DEQ-0 : output standard checkpoint file
//     DEQ-1 : output multi-version checkpoint file
//
// F) bg=on, two=off, overwrite=on
//   output = checkpoint (no multi-version checkpoint file)
//   procedure : We need 1 ENQ and 1 DEQ actions.
//     ENQ : standard ENQ action
//     DEQ : output for standard checkpoint file (multi-ver=off)
//
// G) bg=on, two=on, overwrite=off
//   out = checkpoint + multi-ver-checkpoint
//   procedure :  We need 1 ENQ, 1 DEQ and 2 cp/rename actions.
//     ENQ : starndard ENQ action
//     DEQ : output to tmpFile due to two stage output
//     cp/rename-0 : finalize for standard checkpoint file
//     cp/rename-1 : finalize for multi-ver-checkpoint file
//
// H) bg=on, two=on, overwrite=on
//   out = checkpoint (no multi-version checkpoint file)
//   procedure : We need 1 ENQ, 1 DEQ and 1 cp/rename actions.
//     ENQ : standard ENQ action
//     DEQ : output to tmpFile (two stage output)
//     cp/rename : finalize for standard checkpoint file
//
void
CheckpointController::fileOutputMain(bool checkpointBgWrite,
                                     bool twoStageOutput,
                                     bool snapshotOnly,
                                     RenderContext *renderContext,
                                     const std::string &checkpointPostScript,
                                     const unsigned endSampleId)
{
    const RenderOutputDriver *renderOutputDriver = renderContext->getRenderOutputDriver();
    if (!renderOutputDriver) return; // skip output. just in case

    if (snapshotOnly) {
        // just in case. we use checkpointBgWrite=true, twoStageOutput=true when
        // snapshotOnly situation
        checkpointBgWrite = true;
        twoStageOutput = true;
    }

    // We can generate deep buffer checkpoint file data but not support deep buffer resume render yet.
    const pbr::DeepBuffer *deepBuffer = renderContext->getDeepBuffer();
    pbr::CryptomatteBuffer *cryptomatteBuffer = renderContext->getCryptomatteBuffer();
    scene_rdl2::fb_util::HeatMapBuffer heatMapBuffer;
    renderContext->snapshotHeatMapBuffer(&heatMapBuffer, true, true); // only do if it has data
    scene_rdl2::fb_util::FloatBuffer weightBuffer;
    renderContext->snapshotWeightBuffer(&weightBuffer, true, true); // only do if it has data
    scene_rdl2::fb_util::RenderBuffer renderBufferOdd;
    renderContext->snapshotRenderBufferOdd(&renderBufferOdd, true, true); // only do if it has data
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer> aovBuffers;
    renderContext->snapshotAovBuffers(aovBuffers, true, true);
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer> displayFilterBuffers;
    renderContext->snapshotDisplayFilterBuffers(displayFilterBuffers, true, true);

    if (!snapshotOnly && checkpointBgWrite) {
        // Non memorySnapshot only situation : conditional wait until we have enough bg cache memory capacity
        ImageWriteDriver::get()->waitUntilBgWriteReady();
    }

    ImageWriteDriver::ImageWriteCacheUqPtr cache;
    if (checkpointBgWrite ||
        twoStageOutput ||
        (!checkpointPostScript.empty())) {
        // checkpoint file is written by background thread and run parallel with MCRT threads
        // or post checkpoint script is set. Or two stage output mode is enable.
        // In this case, We need ImageWriteCache object.
        cache = ImageWriteDriver::get()->newImageWriteCache(renderOutputDriver, snapshotOnly);
        cache->setPostCheckpointScript(checkpointPostScript);
        cache->setTwoStageOutput(twoStageOutput);
        if (checkpointBgWrite) {
            cache->setupEnqMode(); // cache internal mode is ENQ
        } else {
            // cache internal mode is STD
        }

    } else {
        // Stop all MCRT threads and checkpoint file write is exclusively executed.
        // Also no post checkpoint script is specified. Also two stage output is disabled.
        // This case cache is nullptr
        MNRY_ASSERT(!cache);
    }

    renderOutputDriver->writeCheckpointEnq(false, // checkpointMultiVersion
                                           deepBuffer,
                                           cryptomatteBuffer,
                                           &heatMapBuffer,
                                           &weightBuffer,
                                           &renderBufferOdd,
                                           aovBuffers,
                                           displayFilterBuffers,
                                           endSampleId,
                                           cache.get());
    renderOutputDriver->loggingErrorAndInfo(cache.get());

    if (cache) {
        cache->calcFinalBlockInternalDataSize();
    }

    if (checkpointBgWrite) {
        if (snapshotOnly) {
            ImageWriteDriver::get()->updateSnapshotData(cache);
        } else {
            ImageWriteDriver::get()->resetSnapshotData(); // clear snapshot data
            ImageWriteDriver::get()->enqImageWriteCache(cache);
        }

    } else {
        if (!twoStageOutput) {
            // We need to output checkpointMultiVersion data here when non-bgOutput mode
            // with non-twoStage mode. If we are overwite=true case, following function
            // return quickly without doing anything.
            renderOutputDriver->writeCheckpointEnq(true, // checkpointMultiVersion
                                                   deepBuffer,
                                                   cryptomatteBuffer,
                                                   &heatMapBuffer,
                                                   &weightBuffer,
                                                   &renderBufferOdd,
                                                   aovBuffers,
                                                   displayFilterBuffers,
                                                   endSampleId,
                                                   nullptr);
            renderOutputDriver->loggingErrorAndInfo(nullptr);
        }

        if (cache) {
            // TwoStage=on
            if (cache->getTwoStageOutput()) {
                // copy and rename for checkpoint file
                cache->allFinalizeCheckpointFile();
            }
            if (!cache->getCheckpointOverwrite()) {
                // We need to output checkpoint multi-version files. Data already be written as tmpFile.
                // copy and rename for checkpoint multi version file
                cache->allFinalizeCheckpointMultiVersionFile();
            }
            if (cache->hasPostCheckpointScript()) cache->runPostCheckpointScript();

            ImageWriteDriver::get()->setLastImageWriteCache(cache);
        }
    }
}

} // namespace rndr
} // namespace moonray
