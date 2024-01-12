// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// This file includes logic about checkpoint/resume render.
// We need to think about TIME_BASED/QUALITY_BASED checkpoint mode and also
// UNIFORM/ADAPTIVE sampling mode. These mode required special care to construct multi-passes workQueue.
// After construct special multi-passes workQueue, finally calls RenderDriver::renderPasses().
//
#include "RenderDriver.h"

#include "AdaptiveRenderTilesTable.h"
#include "ImageWriteCache.h"
#include "ImageWriteDriver.h"
#include "PixSampleRuntimeVerify.h"
#include "RenderContext.h"
#include "RenderOutputDriver.h"
#include "ResumeHistoryMetaData.h"
#include "TileSampleSpecialEvent.h"
#include "TileWorkQueueRuntimeVerify.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/render/util/Random.h>

#ifdef RUNTIME_VERIFY_PIX_SAMPLE_COUNT // See RuntimeVerify.h
#define RUNTIME_VERIFY0
#endif // end RUNTIME_VERIFY_PIX_SAMPLE_COUNT

#ifdef RUNTIME_VERIFY_TILE_WORK_QUEUE // See RuntimeVerify.h
#define RUNTIME_VERIFY1
#endif // end RUNTIME_VERIFY_TILE_WORK_QUEUE

// Debug message display to trackdown progress checkpoint render passes and it's samples id
// This is only for debug purpose and should not turn on for release version.
//#define PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

// Debug message display to trackdown checkpoint micro stint loop logic
// This is only for debug purpose and should not turn on for release version.
//#define PRINT_DEBUG_MESSAGE_PROGRESS_MICROCHECKPOINTSTINTLOOP

namespace moonray {
namespace rndr {

namespace {

void
convertSampleIdRangeToPasses(const unsigned startSampleId, const unsigned endSampleId, std::vector<Pass> &passes)
{
    MNRY_ASSERT(3 <= MAX_RENDER_PASSES);

    unsigned sId = startSampleId;
    unsigned eId = endSampleId;
    if (sId == eId) return;   // early exit

    Pass pass;

    if ((sId & (unsigned)63) != 0) {
        unsigned baseId = sId & ~(unsigned)63;
        pass.mStartPixelIdx = (sId - baseId) & (unsigned)63;
        pass.mEndPixelIdx = std::min((int)(eId - baseId), 64);
        pass.mStartSampleIdx = sId / 64;
        pass.mEndSampleIdx = pass.mStartSampleIdx + 1;
        passes.push_back(pass);
        sId = baseId + 64;
        if (eId <= sId) return;
    }

    if (sId + 64 <= eId) {
        unsigned loop = (eId - sId) / 64;
        pass.mStartPixelIdx = 0;
        pass.mEndPixelIdx = 64;
        pass.mStartSampleIdx = sId / 64;
        pass.mEndSampleIdx = pass.mStartSampleIdx + loop;
        passes.push_back(pass);
        sId += loop * 64;
        if (eId <= sId) return;
    }

    pass.mStartPixelIdx = 0;
    pass.mEndPixelIdx = eId & (unsigned)63;
    pass.mStartSampleIdx = sId / 64;
    pass.mEndSampleIdx = pass.mStartSampleIdx + 1;
    passes.push_back(pass);
}

std::vector<unsigned>
createAdaptiveIterationPixSampleIdTable(const unsigned maxSampleId)
//
// Return pixel samples value table. Adaptive sampling logic would like to evaluate variance at
// this tables pixel sample count.
//
{
    std::vector<unsigned> adaptiveIterationPixSampleIdTable;

    unsigned currSampleId = 0;
    unsigned sampleStep = 4;
    for (unsigned i = 0; i < MAX_RENDER_PASSES; ++i) {
        if (maxSampleId <= currSampleId + 1) break;

        // we have to store endSampleId value here (i.e. sampleId + 1 value)
        adaptiveIterationPixSampleIdTable.push_back(currSampleId + 1);

        currSampleId += sampleStep;
        sampleStep += 2;
    }
    adaptiveIterationPixSampleIdTable.push_back(maxSampleId);

    /* useful debug dump
    auto dumpTbl = [&](const std::vector<unsigned> &tbl) -> std::string {
        int w0 = std::to_string(tbl.size()).size();
        int w1 = std::to_string(tbl.back()).size();
        int w2 = std::to_string(tbl.back() * 64).size();
        std::ostringstream ostr;
        ostr << "tbl (size:" << tbl.size() << ") {\n";
        for (size_t i = 0; i < tbl.size(); ++i) {
            ostr << "  i:" << std::setw(w0) << i
                 << " " << std::setw(w1) << tbl[i]
                 << " (tileSample:" << std::setw(w2) << tbl[i] * 64 << ")" << std::endl;
        }
        ostr << "}";
        return ostr.str();
    };
    std::cerr << dumpTbl(adaptiveIterationPixSampleIdTable) << std::endl;
    */
    /* useful debug dump 2
    auto dumpTblSimple = [&](const std::vector<unsigned> &tbl) -> std::string {
        std::ostringstream ostr;
        for (size_t i = 0; i < tbl.size(); ++i) {
            ostr << std::setw(5) << tbl[i];
            if (i != tbl.size() - 1) ostr << ',';
            if ((i + 1) % 10 == 0) ostr << '\n';
        }
        return ostr.str();
    };
    std::cerr << dumpTblSimple(adaptiveIterationPixSampleIdTable) << std::endl;
    */

    return adaptiveIterationPixSampleIdTable;
}

bool
isAdaptiveIterationBoundary(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl,
                            const unsigned pixSampleId)
//
// Return that the given pixSampleId is on the adaptive sample iteration boundary or not.
//
{
    return (pixSampleId == *std::lower_bound(adaptiveIterationPixSampleIdTbl.begin(),
                                             adaptiveIterationPixSampleIdTbl.end(),
                                             pixSampleId));
}

unsigned
findAdaptiveStartTableId(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl,
                         const unsigned startPixSampleId)
//
// Find table id of proper adaptive iteration sample boundary for start
//
{
    auto itr = std::lower_bound(adaptiveIterationPixSampleIdTbl.begin(), adaptiveIterationPixSampleIdTbl.end(),
                                startPixSampleId);
    if (startPixSampleId < *itr) --itr;
    return std::distance(adaptiveIterationPixSampleIdTbl.begin(), itr);
}

unsigned
findAdaptiveEndTableId(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl,
                       const unsigned endPixSampleId)
//
// Find table id of proper adaptive iteration sample boundary for end
//
{
    auto itr = std::lower_bound(adaptiveIterationPixSampleIdTbl.begin(), adaptiveIterationPixSampleIdTbl.end(),
                                endPixSampleId);
    return std::distance(adaptiveIterationPixSampleIdTbl.begin(), itr);
}

template <bool debugMsg>
void
convertSampleIdRangeToPassesAdaptive(const unsigned startSampleId, const unsigned endSampleId,
                                     const std::vector<unsigned> &adaptiveIterationPixSampleIdTable,
                                     std::vector<Pass> &passes,
                                     std::ostringstream *debugOstr)
{
    auto showPass = [](const Pass &pass) -> std::string {
        std::ostringstream ostr;
        ostr
        << "pix(" << pass.mStartPixelIdx << " ~ " << pass.mEndPixelIdx << ") "
        << "smp(" << pass.mStartSampleIdx << " ~ " << pass.mEndSampleIdx << ")";
        return ostr.str();
    };

    MNRY_ASSERT(3 <= MAX_RENDER_PASSES);

    unsigned sId = startSampleId; // tilebase sampleId (i.e. pixelbase sampleId * 64)
    unsigned eId = endSampleId;   // tilebase sampleId (i.e. pixelbase sampleId * 64)
    if (sId == eId) return; // early exit

    Pass pass;

    if ((sId & (unsigned)63) != 0) {
        //
        // pass-stage-A
        // Create first small pass to fill the gap to the appropriate pixel Id boundary of the tile.
        // (i.e. this pass should end by pixId = 63)
        //
        unsigned baseId = sId & ~(unsigned)63;
        pass.mStartPixelIdx = (sId - baseId) & (unsigned)63;
        pass.mEndPixelIdx = std::min((int)(eId - baseId), 64);
        pass.mStartSampleIdx = sId / 64;
        pass.mEndSampleIdx = pass.mStartSampleIdx + 1;
        if (debugMsg) *debugOstr << "  pass-stage-A:" << showPass(pass) << '\n';
        passes.push_back(pass);
        sId = baseId + 64;

        if (eId <= sId) {
            return; // Special case for single small pass
        }
    }

    // sId is tile pixel boundary position. (i.e. sId is pixId=0 position)

    if (sId + 64 <= eId) {
        //
        // We have samples request which cover at least one sample per pixel for all the pixels of the tile.
        //
        unsigned currPixSampleId = sId / 64;
        if (!isAdaptiveIterationBoundary(adaptiveIterationPixSampleIdTable, currPixSampleId)) {
            unsigned endTblId = findAdaptiveEndTableId(adaptiveIterationPixSampleIdTable, currPixSampleId);
            unsigned currEndPixSampleId = adaptiveIterationPixSampleIdTable[endTblId];
            if ((currEndPixSampleId << 6) <= eId) {
                //
                // pass-stage-B
                // We have to fill the gap to the appropriate adaptive iteration boundary
                //
                pass.mStartPixelIdx = 0;
                pass.mEndPixelIdx = 64;
                pass.mStartSampleIdx = currPixSampleId;
                pass.mEndSampleIdx = currEndPixSampleId;
                if (debugMsg) *debugOstr << "  pass-stage-B:" << showPass(pass) << '\n';
                passes.push_back(pass);
                sId = currEndPixSampleId * 64;
            } else {
                //
                // This is a case that we don't have enough sample request to reach next adaptive iteration
                // boundary. So just skip in this case.
                //
            }
        }

        currPixSampleId = sId / 64;
        unsigned endPixSampleId = eId / 64;
        unsigned startTblId = findAdaptiveStartTableId(adaptiveIterationPixSampleIdTable, currPixSampleId);
        // endTblId search is also using findAdaptiveStartTableId() function here. This is intended.
        // We need endTblId value by same logic of computing startTblId at this point.
        unsigned endTblId = findAdaptiveStartTableId(adaptiveIterationPixSampleIdTable, endPixSampleId);

        if (startTblId < endTblId) {
            //
            // pass-stage-C
            // Create main middle passes which contain main adaptive sampling passes.
            // All the middle passes have full tile pixels (pixId=0~63) with appropriate adaptive iteration
            // sampling range.
            //
            for (unsigned tblId = startTblId; tblId < endTblId; ++tblId) {
                unsigned currStartPixSampleId = adaptiveIterationPixSampleIdTable[tblId];
                unsigned currEndPixSampleId = adaptiveIterationPixSampleIdTable[tblId + 1];

                pass.mStartPixelIdx = 0;
                pass.mEndPixelIdx = 64;
                pass.mStartSampleIdx = currStartPixSampleId;
                pass.mEndSampleIdx = currEndPixSampleId;
                if (debugMsg) *debugOstr << "  pass-stage-C:" << showPass(pass) << '\n';
                passes.push_back(pass);

                sId = currEndPixSampleId << 6; // convert from pix sampleId to tile sampleId
            }
        }

        if (sId + 64 <= eId) {
            //
            // pass-stage-D
            // We still have some samples left and this samples covers entire pixels of tile.
            // (i.e. end sample is not match with adaptive iteration range boundary.
            //
            unsigned loop = (eId - sId) / 64;
            pass.mStartPixelIdx = 0;
            pass.mEndPixelIdx = 64;
            pass.mStartSampleIdx = sId / 64;
            pass.mEndSampleIdx = pass.mStartSampleIdx + loop;
            if (debugMsg) *debugOstr << "  pass-stage-D:" << showPass(pass) << '\n';
            passes.push_back(pass);
            sId += loop * 64;
        }

        if (eId <= sId) {
            return; // Special case. All the samples are converted into passes. we are done.
        }
    }

    //
    // pass-stage-E
    // Create last small pass to fill the remaining samples.
    //
    pass.mStartPixelIdx = 0;
    pass.mEndPixelIdx = eId & (unsigned)63;
    pass.mStartSampleIdx = sId / 64;
    pass.mEndSampleIdx = pass.mStartSampleIdx + 1;
    if (debugMsg) *debugOstr << "  pass-stage-E:" << showPass(pass) << '\n';
    passes.push_back(pass);
}

unsigned
findAdaptiveEndPixSampleId(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl,
                           const unsigned startPixSampleId,
                           const unsigned adaptiveIterationSteps)
//
// find adaptiveEndSampleId based on the adaptiveIterationSampleIdTbl data
//
{
    size_t tblId = findAdaptiveStartTableId(adaptiveIterationPixSampleIdTbl, startPixSampleId);
    if (!isAdaptiveIterationBoundary(adaptiveIterationPixSampleIdTbl, startPixSampleId)) {
        // startPixSampleId is not boundary of adaptiveIterationPixSampleId table.
        // In this case, we just return end pix sampleId as next boundary of
        // adaptiveIterationPixSampleId and just skip adaptiveIterationSteps.
        tblId += 1;
    } else {
        tblId += adaptiveIterationSteps;
    }
    return adaptiveIterationPixSampleIdTbl[std::min(tblId, (size_t)adaptiveIterationPixSampleIdTbl.size() - 1)];
}

unsigned
calcStartResumeUniformSampleTotal(const unsigned unAlignedW,
                                  const unsigned unAlignedH,
                                  const unsigned checkpointStartTileSampleId)
{
    //
    // We have to convert tileSampleId to pixel samples total of entire image.
    //
    // This function is only supported single machine environment so far.
    // Not support multi-machine execution yet.
    // Internally single machine version of Film::getPixelFillOrder() is used.
    // If we need to support multi-machine, we have to think bit more detail because each machine
    // has different pixel fill order (See Film.h)
    //
    if ((unAlignedW % 8) == 0 && (unAlignedH % 8) == 0) {
        // W, H are tile aligned.
        unsigned totalTiles = (unAlignedW / 8) * (unAlignedH / 8);
        return totalTiles * checkpointStartTileSampleId;
    }

    int totalTileX = (unAlignedW + 7) / 8;
    int totalTileY = (unAlignedH + 7) / 8;
    int totalFullTiles = (totalTileX - 1) * (totalTileY - 1);
    int totalXBoundTiles = totalTileY - 1;
    int totalYBoundTiles = totalTileX - 1;
    int totalXYBoundTile = 1;

    int xSize = unAlignedW % 8;
    int ySize = unAlignedH % 8;
    unsigned xBoundTileSamples = 0;
    unsigned yBoundTileSamples = 0;
    unsigned xyBoundTileSamples = 0;
    for (unsigned pixId = 0; pixId < checkpointStartTileSampleId; ++pixId) {
        unsigned pixelPerm = Film::getPixelFillOrder(pixId & 63);
        unsigned px = pixelPerm % 8;
        unsigned py = pixelPerm / 8;
        if (px < xSize) xBoundTileSamples++;
        if (py < ySize) yBoundTileSamples++;
        if (px < xSize && py < ySize) xyBoundTileSamples++;
    }

    int totalSamples =
        totalFullTiles   * checkpointStartTileSampleId +
        totalXBoundTiles * xBoundTileSamples +
        totalYBoundTiles * yBoundTileSamples +
        totalXYBoundTile * xyBoundTileSamples;
    return totalSamples;
}

#ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
// Useful debug function
std::string
showAllPasses(const std::string &hd, const std::vector<Pass> &passes)
{
    int w0 = std::to_string(passes.size()-1).size();
    int w1 = std::to_string(passes[0].mStartPixelIdx).size();
    int w2 = std::to_string(passes[0].mEndPixelIdx).size();
    int w3 = std::to_string(passes[passes.size()-1].mStartSampleIdx).size();
    int w4 = std::to_string(passes[passes.size()-1].mEndSampleIdx).size();
    std::ostringstream ostr;
    ostr << hd << "passes (total:" << passes.size() << ") {\n";
    for (size_t i = 0; i < passes.size(); ++i) {
        const Pass &currPass = passes[i];
        ostr << hd << "  i:" << std::setw(w0) << i;
        ostr << " pix("
             << std::setw(w1) << currPass.mStartPixelIdx << " ~ "
             << std::setw(w2) << currPass.mEndPixelIdx << ")"
             << " smp("
             << std::setw(w3) << currPass.mStartSampleIdx << " ~ "
             << std::setw(w4) << currPass.mEndSampleIdx << ")\n";
    }
    ostr << hd << "}";
    return ostr.str();
}

// Useful debug dump for debug
std::string
showAllPasses(const std::string &hd, const std::vector<Pass> &newPasses,
              const unsigned startSampleId, const unsigned endSampleId,
              const unsigned tileSamplesCap,
              const unsigned finalTileSamples)
{
    std::ostringstream ostr;
    ostr << showAllPasses(hd, newPasses)
         << " tileSampleId(start:" << startSampleId << " end:" << endSampleId
         << " delta:" << endSampleId - startSampleId
         << " tileSamplesCap:" << tileSamplesCap
         << " final:" << finalTileSamples << ')';
    return ostr.str();
}
#endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

} // namespace

//---------------------------------------------------------------------------------------------------------------

// static function
std::vector<unsigned>
RenderDriver::createKJSequenceTable(const unsigned maxSampleId)
{
    return createAdaptiveIterationPixSampleIdTable(maxSampleId);
}

// static function
unsigned
RenderDriver::revertFilmObjectAndResetWorkQueue(RenderDriver *driver, const FrameState &fs)
//
// Revert film object from file for resume render and
// reset workQueue for first renderPass based on reverted information when if checkpointMode is TIME_BASED
// return progressCheckpointStartTileSampleId number (this is a tile sample and not pixel sample).
//
{
    if (!fs.mRenderContext->getSceneContext().getResumeRender()) {
        return 0;               // this is not a resume render case.
    }

    //
    // Resume render mode
    //
    Film *film = driver->mFilm;
    unsigned progressCheckpointStartTileSampleId = 0;
    if (!film->getResumedFromFileCondition()) {
        driver->mProgressEstimation.revertFilm(true);

        // This is a very first time to render. So we try to revert film object from file.
        RenderOutputDriver *renderOutputDriver = fs.mRenderContext->getRenderOutputDriver();
        renderOutputDriver->resetErrors();  // clean up error buffer
        if (!driver->revertFilmData(renderOutputDriver, fs, progressCheckpointStartTileSampleId)) {
            // revert from file failed. -> fall back to standard operation.
            scene_rdl2::logging::Logger::error("Fall back on standard rendering");
            film->clearAllBuffers();
            driver->mProgressEstimation.revertFilm(false);
            return 0;           // fall back to normal render
        }

        // set tried condition and skip all additional revertFilmData() regardless of
        // the result of revertFilmData()
        film->setResumedFromFileCondition();

        if (fs.mSamplingMode == SamplingMode::ADAPTIVE) {
            unsigned pixelSampleId =
                film->getAdaptiveRenderTilesTable()->reset(*film,
                                                           *driver->getTiles(),
                                                           fs.mMinSamplesPerPixel,
                                                           fs.mMaxSamplesPerPixel,
                                                           fs.mTargetAdaptiveError,
                                                           fs.mViewport);

            // We need to enable adjust adaptiveTree update timing logic and execute adaptiveTreeUpdate operation by current film
            // condition before start rendering under resume adaptive sampling rendering mode.
            UIntTable adaptiveIterationPixSampleIdTable =
                createAdaptiveIterationPixSampleIdTable(fs.mMaxSamplesPerPixel);
            film->enableAdjustAdaptiveTreeUpdateTiming(adaptiveIterationPixSampleIdTable);
            film->updateAdaptiveErrorAll(film->getRenderBuffer(), *(film->getRenderBufferOdd()));

            // Adaptive sampling case, progressCheckpointStartTileSampleId is always boundary of
            // same pixel samples in the tile.
            progressCheckpointStartTileSampleId = pixelSampleId * 64;
            if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
                //
                // We only setup workQueue for TIME_BASED checkpoint mode.
                // If user select QUALITY_BASED checkpoint mode, all workQueue will be constructed
                // inside RenderDriver::progressCheckpointRenderFrame()
                //

                unsigned startSampleId = pixelSampleId * 64;
                // Added 64 samples to tile (i.e. = add 1 sample per pixel).
                // Why adding 64 samples to the tile?
                // This first render schedule is used by very first checkpoint render stint as estimation.
                // If we set 1 sample per tile like non resumable render estimation, this 1 sample per tile
                // is not enough in some condition and we can not get reasonable quality of estimation result.
                // If 1 sample per tile is skipped somehow depending on sampling parameter change and/or
                // other variance condition of the pixels, estimated averaged 1 sample cost is extremely
                // light (i.e. short time). As a result, estimated total sampling count becomes huge number.
                // This situation makes totally break of checkpoint-logic.
                // In order to reduce risk, Using 64 samples per tile for initial estimation stage.
                unsigned endSampleId = startSampleId + 64;
                std::vector<Pass> newPasses;
                convertSampleIdRangeToPassesAdaptive<false>(startSampleId, endSampleId,
                                                            adaptiveIterationPixSampleIdTable, newPasses,
                                                            nullptr);
#               ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
                std::cerr << ">> RenderFrameCheckpointResume.cc revertFilmObjectAndResetWorkQueue()"
                          << std::endl;
                std::cerr << showAllPasses("", newPasses, startSampleId, endSampleId,
                                           64, fs.mMaxSamplesPerPixel * 64)
                          << std::endl;                
#               endif                
                TileWorkQueue *workQueue = &driver->mTileWorkQueue;
                const unsigned numTiles = unsigned(driver->getTiles()->size());
                workQueue->init(RenderMode::PROGRESS_CHECKPOINT, numTiles,
                                newPasses.size(), mcrt_common::getNumTBBThreads(), &newPasses[0]);
            }

        } else {
            unsigned startSampleId = progressCheckpointStartTileSampleId;
            unsigned endSampleId = startSampleId + 1;

            // Setup start condition for uniform resume sampling. This is important for progress info
            unsigned startSamples =
                calcStartResumeUniformSampleTotal(driver->mUnalignedW,
                                                  driver->mUnalignedH,
                                                  progressCheckpointStartTileSampleId);
            driver->mProgressEstimation.setStartUniformSamples(startSamples);

            if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
                //
                // We only setup workQueue for TIME_BASED checkpoint mode. Otherwise workQueue will be
                // setup inside RenderDriver::progressCheckpointRenderFrame()
                //
                // re-create workQueue based on resume start sampleId for first renderPasses()
                std::vector<Pass> newPasses;
                convertSampleIdRangeToPasses(startSampleId, endSampleId, newPasses);
#               ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
                std::cerr << ">> RenderFrameCheckpointResume.cc 1st estimate renderPasses "
                          << showAllPasses("", newPasses) << std::endl;
#               endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
                TileWorkQueue *workQueue = &driver->mTileWorkQueue;
                const unsigned numTiles = unsigned(driver->getTiles()->size());
                workQueue->init(RenderMode::PROGRESS_CHECKPOINT, numTiles,
                                newPasses.size(), mcrt_common::getNumTBBThreads(), &newPasses[0]);
            }
        }

        driver->mProgressEstimation.revertFilm(false);
    }

    fs.mRenderContext->getResumeHistoryMetaData()->setStartTileSampleId(progressCheckpointStartTileSampleId);

    return progressCheckpointStartTileSampleId;
}

//---------------------------------------------------------------------------------------------------------------

// static function
bool
RenderDriver::progressCheckpointRenderFrame(RenderDriver *driver, const FrameState &fs,
                                            const unsigned progressCheckpointStartTileSampleId,
                                            const TileSampleSpecialEvent *tileSampleSpecialEvent)
//
// Rendering single frame by checkpoint mode. Create multiple checkpoint files inside this function.
// Support 2 different checkpoint mode so far. it's TIME_BASED and QUALITY_BASED checkpoint mode.
//
// Returns true if we ran to completion or false if we were canceled.
{
#ifdef RUNTIME_VERIFY0
    PixSampleCountRuntimeVerify::init(driver->getWidth(), driver->getHeight());
#endif // end RUNTIME_VERIFY0

    //------------------------------
    //
    // checkpoint related setup
    //
    RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();
    {
        scene_rdl2::fb_util::Tiler tiler(driver->getWidth(), driver->getHeight());
        unsigned resumeStartWholeSamples = progressCheckpointStartTileSampleId * tiler.mNumTiles;
        timingRec.reset(resumeStartWholeSamples); // reset condition for new frame. set FrameStartTime as well
    }

    // There should only be a single coarse pass at this point. That pass is
    // assumed to only render a single sample per tile.

    // Compute end time for estimation stage
    double frameBudget = 0.0;
    if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
        // We don't account renderPrep time for estimation. (This is a one of major difference between
        // REALTIME rendermode).
        if (fs.mCheckpointInterval * 60.0f > 5.0f) {
            // The 1st checkpoint interval is adjusted to 5 sec in order to create a pretty rough
            // immediate checkpoint data.
            frameBudget = 5.0; // sec : user defined but not include frame update duration for 1st stint.
        }
        double predictedEnd = timingRec.setRenderFrameTimeBudget(frameBudget);

        driver->mFrameEndTime = predictedEnd;
    } else {
        // QualityBased checkpoint render : basically we don't need to measure timing because we don't
        // do any cost estimation for quality based checkpoint render
        driver->mFrameEndTime = 0.0f;
    }
    unsigned firstTileSamples = driver->mTileWorkQueue.getTotalTileSamples();

    // The TileWorkQueue is already configured with a single pass which renders
    // a single sample per tile. We use this as our baseline sample cost estimate.
#   ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
    scene_rdl2::rec_time::RecTime recTime;
    recTime.start();
    if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
        std::cerr << ">> RenderFrameCheckpointResume.cc progressCheckpointRenderFrame() TIME_BASED"
                  << " 1st stint start (frameBudget:" << frameBudget << " sec) {" << std::endl;
    } else {
        std::cerr << ">> RenderFrameCheckpointResume.cc progressCheckpointRenderFrame() QUALITY_BASED"
                  << " start (qualitySteps:" << fs.mCheckpointQualitySteps << ")" << std::endl;
    }
#   endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

    // store timing info for resume history
    fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintStartTime();

    //------------------------------
    //
    // initial estimation renderPasses()
    //
    unsigned startSampleId = 0;
    unsigned endSampleId = 0;
    if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
        //
        // TIME_BASED checkpoint render
        // We do initial estimation phase before start main checkpoint stint loop 
        //
        startSampleId = progressCheckpointStartTileSampleId;
        endSampleId = startSampleId + firstTileSamples;

        driver->mParallelInitFrameUpdateTime.start();
        driver->mCheckpointEstimationStage = true;
        renderPasses(driver, fs, false); // no cancelation requested
        driver->mCheckpointEstimationStage = false;
        if (!driver->mParallelInitFrameUpdate) {
            float time = driver->mParallelInitFrameUpdateTime.end();
            driver->mCheckpointEstimationTime.set(time);
            /* useful debug message
            std::cerr << ">> RenderFrameCheckpointResume.cc READY-FOR-DISPLAY nonParallel"
                      << " time:" << time << " sec\n";
            */
        }
    } else {
        //
        // QUALITY_BASED checkpoint render
        // We skip all estimation conputation for quality based checkpoint render
        //
        startSampleId = progressCheckpointStartTileSampleId;
        endSampleId = startSampleId;
    }

#ifdef RUNTIME_VERIFY0
    std::cerr << ">> RenderFrameCheckpointResume.cc after initial estimation pass "
              << PixSampleCountRuntimeVerify::get()->show() << '\n';
#endif // end RUNTIME_VERIFY0

    driver->setReadyForDisplay();
    if (driver->mStopAtFrameReadyForDisplay) {
        return false;           // This is a equivalent with cancel condition
    }

    double renderStart = scene_rdl2::util::getSeconds(); // for time cap

    // initial adaptive tile sampleCap is same number of current sampling total
    if (fs.mSamplingMode != SamplingMode::UNIFORM) {
        driver->mAdaptiveTileSampleCap = firstTileSamples;
    }

    //------------------------------
    //
    // checkpoint stint loop
    //
    driver->mCheckpointController.reset();
    double frameBudgetBase = fs.mCheckpointInterval * 60.0f; // convert to second from minute
    int quickPhaseTotal = 0;
    if (driver->mMultiMachineCheckpointMainLoop) {
        frameBudgetBase = driver->mMultiMachineFrameBudgetSecShort;
        quickPhaseTotal = driver->mMultiMachineQuickPhaseLengthSec / frameBudgetBase;
        if (quickPhaseTotal < 1) quickPhaseTotal = 1;
    }
    for (size_t checkpointStintId = 0; ; ++checkpointStintId) {
        double durationSec = scene_rdl2::util::getSeconds() - renderStart;
        if (fs.mCheckpointTimeCap > 0.0f && durationSec / 60.0 > (double)fs.mCheckpointTimeCap) {
            std::ostringstream ostr;
            ostr << "===>>> Rendering time was exceeded to the time cap (" << fs.mCheckpointTimeCap
                 << " min) and finish render <<<===";
            scene_rdl2::logging::Logger::info(ostr.str());
#           ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
            std::cerr << ostr.str() << std::endl;
#           endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
            break;              // exceeded time cap -> end
        }
        if (fs.mCheckpointSampleCap > 0 && endSampleId == fs.mCheckpointSampleCap) {
            std::ostringstream ostr;
            ostr << "===>>> Rendering sample was exceeded to the sample cap (tileSample:"
                 << fs.mCheckpointSampleCap
                 << " pixSample:" << fs.mCheckpointSampleCap / 64
                 << ") and finish render <<<===";
            scene_rdl2::logging::Logger::info(ostr.str());
#           ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
            std::cerr << ostr.str() << std::endl;
#           endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
            break;              // exceeded sample cap -> end
        }

        //
        // One checkpoint render stint starting
        //
        RenderPassesResult stintResult =
            progressCheckpointRenderStint(driver, fs, progressCheckpointStartTileSampleId,
                                          tileSampleSpecialEvent, startSampleId, endSampleId);

#       ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
        std::cerr << ">> RenderFrameCheckpointResume.cc progressCheckpointRenderFrame() } checkpoint stint end."
                  << " interval:" << recTime.end() << " sec" << std::endl;
#       endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

        if (stintResult == RenderPassesResult::ERROR_OR_CANCEL) {
            return false;
        }
        if (stintResult == RenderPassesResult::STOP_AT_PASS_BOUNDARY) {
            driver->mFrameCompleteAtPassBoundary = true;
            break;              // We've completed
        }

#ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
        std::cerr << ">> RenderFrameCheckpointResume.cc checkpoint stint end. interval:" << recTime.end()
                  << " sec" << std::endl;
#endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

        //
        // output checkpoint file
        //
#ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
        std::cerr << ">> RenderFrameCheckpointResume.cc checkpoint file output."
                  << " endSampleId:" << endSampleId << '\n'
                  << "=========================================================" << std::endl;
#endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

        // record MCRT end timing for resumeHistory
        fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintEndTime(endSampleId - 1, false);

        checkpointFileOutput(driver, fs, endSampleId);

        if (stintResult == RenderPassesResult::COMPLETED_ALL_SAMPLES) {
            break; // We've completed whole samples and done.
        }

        // not completed. Let's record start time for next checkpoint stint
        fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintStartTime();

        frameBudget = frameBudgetBase;
        if (driver->mMultiMachineCheckpointMainLoop) {
            // Change frameBudget (time length) depending on the globalProgressFraction
            // under multi-machine configuration. This makes better CPU utilization and
            // also better MCRT stage stop response.
            if (driver->mMultiMachineGlobalProgressFraction < 0.5f) {
                if (checkpointStintId >= quickPhaseTotal - 1) {
                    frameBudgetBase = driver->mMultiMachineFrameBudgetSecLong;
                }
            } else if (driver->mMultiMachineGlobalProgressFraction > 0.9f) {
                frameBudgetBase = driver->mMultiMachineQuickPhaseLengthSec;
            } else {
                frameBudgetBase *= 0.5f;

                if (frameBudgetBase <= driver->mMultiMachineQuickPhaseLengthSec) {
                    frameBudgetBase = driver->mMultiMachineQuickPhaseLengthSec;
                }
            }
            std::cerr << ">> RenderFrameCheckpointResume.cc frameBudgetBase:" << frameBudgetBase << '\n';
        }

#       ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
        recTime.start();
        std::cerr << ">> RenderFrameCheckpointResume.cc progressCheckpointRenderFrame() checkpoint 2nd/later"
                  << " stint start frameBudget:" << frameBudget << " sec) {" << std::endl;
#       endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

        { // compute end time for next checkpoint render stint
            timingRec.newStint();

            double predictedEnd = timingRec.setRenderFrameTimeBudget(frameBudget);
            driver->mFrameEndTime = predictedEnd;
        }
    } // checkpoint stint loop
    fs.mRenderContext->getRenderOutputDriver()->setLastCheckpointRenderTileSamples(endSampleId);

    // finalize sync (wait all ImageWriteDriver task) here
    fs.mRenderContext->getResumeHistoryMetaData()->setFinalizeSyncStartTime();
    ImageWriteDriver::get()->conditionWaitUntilAllCompleted(); // condition wait
    fs.mRenderContext->getResumeHistoryMetaData()->setFinalizeSyncEndTime();

    // mark frame as fully complete.
    driver->setFrameComplete();
    driver->mProgressEstimation.setFrameComplete();

#ifdef RUNTIME_VERIFY0
    std::cerr << PixSampleCountRuntimeVerify::get()->show() << '\n';
#endif // end RUNTIME_VERIFY0

    return true;
}

// static function
RenderDriver::RenderPassesResult
RenderDriver::progressCheckpointRenderStint(RenderDriver *driver, const FrameState &fs,
                                            const unsigned processStartTileSampleId,
                                            const TileSampleSpecialEvent *tileSampleSpecialEvent,
                                            unsigned &startSampleId, unsigned &endSampleId)
//
// This function try to render some checkpoint stint computation. Not include checkpoint file output.
// Under TIME_BASED mode, try to occupied entire time budget based on the average cost of single
// sample. Under QUALITY_BASED mode, just used pre-computed quality tile sample id table then determines
// sample volumes.
//
// startSampleId, endSampleId
//   These 2 argument values are reference and updated inside this function.
//   These 2 values are not indicate start and end sampleId of this checkpoint render stint.
//   Start/end sampleId is computed based on estimation logic and not defined by argument.
//    
{
    RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();

    UIntTable adaptiveIterationPixSampleIdTable;
    const UIntTable *adaptiveIterationPixSampleIdTablePtr = nullptr;
    if (fs.mSamplingMode != SamplingMode::UNIFORM) {
        // Adaptive sampling case, we setup adaptiveIterationPixSampleIdTable
        adaptiveIterationPixSampleIdTable = createAdaptiveIterationPixSampleIdTable(fs.mMaxSamplesPerPixel);
        adaptiveIterationPixSampleIdTablePtr = &adaptiveIterationPixSampleIdTable;

        // We do enable adjust adaptiveTree update timing logic for checkpoint + adaptive sampling.
        driver->mFilm->enableAdjustAdaptiveTreeUpdateTiming(adaptiveIterationPixSampleIdTable);
    }

    if (startSampleId == processStartTileSampleId) {
        // This is a very first checkpoint stint just after estimation render has been completed.
        // We should initialize progress estimation logic here by some initial condition.
        unsigned finalTileSamples = fs.mMaxSamplesPerPixel * 64;
        driver->mProgressEstimation.startEstimation(timingRec, processStartTileSampleId, finalTileSamples);
    }

    if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
        //
        // TIME_BASED checkpoint rendering
        //
        while (1) { // Retry loop if still time-budget is not run out.
            double remainingTime = 0.0;

            if (isProgressCheckpointComplete(driver, fs)) {
                return RenderPassesResult::COMPLETED_ALL_SAMPLES; // We've completed -> done!
            }

            double now = scene_rdl2::util::getSeconds();
            remainingTime = driver->mFrameEndTime - now;
            if (timingRec.isComplete(remainingTime, now)) { // Do we have time to render more samples?
                break; // -> Time budget for this stint expired and end rendering loop
            }

            {
                unsigned numNewSamplesPerTile =
                    calcCheckpointStintStartEndSampleId(driver, fs,
                                                        remainingTime,
                                                        adaptiveIterationPixSampleIdTablePtr,
                                                        startSampleId, endSampleId);
                if (numNewSamplesPerTile == 0) {
                    return RenderPassesResult::COMPLETED_ALL_SAMPLES; // sample budget expired -> done!
                }
            }

            RenderPassesResult result =
                checkpointRenderMiniStintLoop(driver, fs,
                                              adaptiveIterationPixSampleIdTablePtr,
                                              tileSampleSpecialEvent,
                                              startSampleId, endSampleId);
            if (result != RenderPassesResult::OK_AND_TRY_NEXT) {
                return result;
            }

            if (endSampleId == 64) { // Check if we still need to run extrapolation before displaying.
                driver->setCoarsePassesComplete();
                break;              // endSampleId == 64 should be last pass
            }
            if (endSampleId == fs.mCheckpointSampleCap) { // checkpoint sample cap
                break;
            }
        } // while (1) : Retry loop if still time-budget is not run out

    } else if (fs.mCheckpointMode == CheckpointMode::QUALITY_BASED) {
        //
        // QUALITY_BASED checkpoint rendering : logic is bit simpler than time based mode
        //

        // Check finish condition first.
        if (isProgressCheckpointComplete(driver, fs)) {
            return RenderPassesResult::COMPLETED_ALL_SAMPLES; // We've completed -> done!
        }

        {
            unsigned numNewSamplesPerTile =
                calcCheckpointStintStartEndSampleId(driver, fs,
                                                    0.0, // This argument is not used by QUALITY_BASED
                                                    adaptiveIterationPixSampleIdTablePtr,
                                                    startSampleId, endSampleId);
            if (numNewSamplesPerTile == 0) {
                return RenderPassesResult::COMPLETED_ALL_SAMPLES; // no additional samples -> done!
            }
        }

        RenderPassesResult result =
            checkpointRenderMiniStintLoop(driver, fs,
                                          adaptiveIterationPixSampleIdTablePtr, tileSampleSpecialEvent,
                                          startSampleId, endSampleId);
        if (result != RenderPassesResult::OK_AND_TRY_NEXT) {
            return result;
        }

        // update pass info just in case even QUALITY_BASED checkpoint mode
        driver->mProgressEstimation.updatePassInfo(timingRec);

        if (endSampleId == 64) { // Check if we still need to run extrapolation before displaying.
            driver->setCoarsePassesComplete();
        }

        // We need to check all possible samples are completed here.
        // We have to return render finish condition from this function due to avoid
        // unnecessary empty checkpoint stint loop. If we do empty extra checkpoint stint loop,
        // exactly same checkpoint file is created twice and have to pay extra checkpoint write cost.
        // Even more, empty extra checkpoint stint loop executes exactly same post checkpoint script
        // twice. This is pretty bad idea and might creates critical situation.
        if (isProgressCheckpointComplete(driver, fs)) {
            return RenderPassesResult::COMPLETED_ALL_SAMPLES; // We've completed -> done!
        }
    }

    return RenderPassesResult::OK_AND_TRY_NEXT; // need to continue more checkpoint stint
}

// static function
bool
RenderDriver::isProgressCheckpointComplete(RenderDriver *driver, const FrameState &fs)
{
    if (fs.mSamplingMode == SamplingMode::UNIFORM) {
        //
        // UNIFORM sampling mode
        //
        const unsigned spp = fs.mMaxSamplesPerPixel;
        const unsigned long long finalWholeSamples =
            static_cast<unsigned long long>(driver->getWidth()) *
            static_cast<unsigned long long>(driver->getHeight()) *
            static_cast<unsigned long long>(spp);

        RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();

#       ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
        std::cerr << ">> RenderFrameCheckpointResume.cc isProgressCheckpointComplete() }  endTest: numSamples:"
                  << timingRec.getNumSamplesAll()
                  << " finalWholeSamples:" << finalWholeSamples << std::endl;
#       endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

        if (timingRec.getNumSamplesAll() >= finalWholeSamples) {
#           ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
            std::cerr << ">> RenderFrameCheckpointResume.cc isProgressCheckpointComplete() done" << std::endl;
#           endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
            return true; // We've completed whole samples -> end loop
        }

    } else {
        //
        // ADAPTIVE sampling mode
        //
        if (driver->mFilm->getAdaptiveDone()) {
            return true; // all tiles have been converged -> end loop
        }
    }

    return false;
}

// static function
unsigned
RenderDriver::calcCheckpointStintStartEndSampleId(RenderDriver *driver, const FrameState &fs,
                                                  const double remainingTime, // for time based mode
                                                  const UIntTable *adaptiveIterationPixSampleIdTable,
                                                  unsigned &startSampleId,
                                                  unsigned &endSampleId)
//
// Compute startSampleId and endSampleId for checkpoint stint based on the checkpoint mode (TIME_BASED /
// QUALITY_BASED) with related information.
//
// adaptiveIterationPixSampleIdTable is only set properly when adaptive sampling mode otherwise might be nullptr
//
{
    //
    // 1st phase : compute estimated samples per tile based on the checkpoint mode
    //
    double estimatedSamplesPerTile =
        estimateSamplesPerTile(driver, fs,
                               remainingTime, adaptiveIterationPixSampleIdTable, endSampleId);

    //
    // 2nd phase : apply some sample number special rules and check end condition.
    //
    unsigned numNewSamplesPerTile = scene_rdl2::math::max(unsigned(estimatedSamplesPerTile), 1u); // for next render

    // Compute startSampleId and endSampleId for this checkpoint stint.
    // (note: pixels wrap around after we exceed 64).
    const unsigned finalTileSamples = fs.mMaxSamplesPerPixel * 64;
    startSampleId = endSampleId;
    endSampleId = std::min(startSampleId + numNewSamplesPerTile, finalTileSamples);

    if (startSampleId == endSampleId) {
        return 0; // sample budget expired -> done!
    }

    if (startSampleId < 64 && 64 <= endSampleId) {
        endSampleId = 64;   // We do not render coarse pass and fine pass in same checkpoint stint
        numNewSamplesPerTile = endSampleId - startSampleId; // re-compute new samples per tile
    }
    if (startSampleId < fs.mCheckpointSampleCap && fs.mCheckpointSampleCap <= endSampleId) {
        // checkpoint sample cap operation.
        endSampleId = fs.mCheckpointSampleCap;
        numNewSamplesPerTile = endSampleId - startSampleId; // re-compute new samples per tile
    }

    return numNewSamplesPerTile;
}

// static function
double
RenderDriver::estimateSamplesPerTile(RenderDriver *driver, const FrameState &fs,
                                     const double remainingTime, // for time based mode
                                     const UIntTable *adaptiveIterationPixSampleIdTable,
                                     const unsigned startTileSampleId) // for quality based mode
//
// This function calculates samples per tile for this checkpoint stint based on the current checkpoint mode.
// Just return samplesPerTile number. Later stage convert this samplesPerTile number to the start/end sampleId.
// Return value uses double precision due to all timing related information has already double precision
// (and there is no special reason to cast float then reduce precision inside this function. Performance of
// this function is also not so important because this function is executed once or very few counts at every
// checkpoint stint).
// 
// adaptiveIterationPixSampleIdTable is only used when non SamplingMode::UNIFORM w/ QUALITY_BASED mode.
//
{
    double estimatedSamplesPerTile = 0.0;
    if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
        //
        // TIME_BASED checkpoint : estimation of samplePerTile based on timing history
        //
        RenderFrameTimingRecord &timingRec = driver->getRenderFrameTimingRecord();

        estimatedSamplesPerTile =
            (fs.mSamplingMode == SamplingMode::UNIFORM)?
            timingRec.estimateSamplesAllStint(remainingTime): // using whole result
            timingRec.estimateSamples(remainingTime); // only using previous phase

        // Sepcial adjustment for time based mode
        if (driver->mAdaptiveTileSampleCap > 0 &&
            estimatedSamplesPerTile > driver->mAdaptiveTileSampleCap) {
            // Only for adaptive sampling under TIME_BASED checkpoint render mode.
            // We can only increase 2x bigger CAP than current value at each checkpoint stint as max.
            // This is a safety logic to ignore wrong passes which includes massive tile samples
            // (sometimes massive passes are created if sample cost estimation is pretty pure case).
            driver->mAdaptiveTileSampleCap =
                std::min((unsigned)estimatedSamplesPerTile, driver->mAdaptiveTileSampleCap * 2);
            estimatedSamplesPerTile =
                std::min(estimatedSamplesPerTile, (double)driver->mAdaptiveTileSampleCap);
        }

    } else {
        //
        // QUALITY_BASED checkpoint : estimation of samplePerTile based on quality steps of iterations
        //
        unsigned overRunSamples = startTileSampleId % 64;
        unsigned fillUpSamples = 0;
        if (overRunSamples) {
            // startTileSampleId is not boundary of equal pixel samples inside one tile.
            // In this case, we should add special sampling in order to fill up
            // to next boundary of equal pixel samples inside one tile.
            fillUpSamples = 64 - overRunSamples;
        }

        switch (fs.mSamplingMode) {
        case SamplingMode::UNIFORM : // uniform sampling
            // Uniform sampling case, we use qualitySteps as increased pixel samples number.
            estimatedSamplesPerTile =
                fillUpSamples +
                fs.mCheckpointQualitySteps * 64.0; // converted tile samples
            break;

        default : { // adaptive sampling
            unsigned pixStartSampleId = (startTileSampleId + fillUpSamples) / 64;
            unsigned pixEndSampleId = findAdaptiveEndPixSampleId(*adaptiveIterationPixSampleIdTable,
                                                                 pixStartSampleId,
                                                                 fs.mCheckpointQualitySteps);
            estimatedSamplesPerTile =
                fillUpSamples +
                (pixEndSampleId - pixStartSampleId) * 64.0;
        } break;
        }
    }

#   ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
    if (fs.mCheckpointMode == CheckpointMode::TIME_BASED) {
        std::cerr << ">> RenderFrameCheckpointResume.cc TIME_BASED remainingTime:" << remainingTime << " sec"
                  << " estimatedSamplesPerTile:" << estimatedSamplesPerTile << std::endl;
    } else {
        std::cerr << ">> RenderFrameCheckpointResume.cc QUALITY_BASED"
                  << " qualitySteps:" << fs.mCheckpointQualitySteps
                  << " estimatedSamplesPerTile:" << estimatedSamplesPerTile << std::endl;
    }
#   endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

    return estimatedSamplesPerTile;
}

// static function
RenderDriver::RenderPassesResult
RenderDriver::checkpointRenderMiniStintLoop(RenderDriver *driver,
                                            const FrameState &fs,
                                            const UIntTable *adaptiveIterationPixSampleIdTable,
                                            const TileSampleSpecialEvent *tileSampleSpecialEvent,
                                            unsigned &startSampleId, unsigned &endSampleId) // tile sampleId
//
// In order to support tileSampleSpecialEvent, we have to break down start/end sampleId into mini-stint.
// This function creates mini-stint and render them sequentially. After finish every mini-stint,
// we do call specialEvent callBack function is needed.
// If no tileSampleSepcialEvent is specified, just process start/end sample id span as single stint
// without callBack.
//
// startSampleId, endSampleId
//   These 2 argument values are reference and updated inside this function.
//   These 2 values are not indicate start and end sampleId of this checkpoint render stint.
//   Start/end sampleId is computed based on estimation logic and not defined by argument.
//    
{
    // First of all, create miniStint sampleId table based on start/end sampleId with
    // tileSampleSpecialEvent information.
    UIntTable miniStintSampleIdTable =
        calcCheckpointMiniStintStartEndId(startSampleId, endSampleId, tileSampleSpecialEvent);

    // Execute checkpointRenderMiniStint() function based on the table which created above.
    for (size_t miniStintId = 0; miniStintId < miniStintSampleIdTable.size() - 1; ++miniStintId) {
        startSampleId = miniStintSampleIdTable[miniStintId];
        endSampleId = miniStintSampleIdTable[miniStintId + 1];
        RenderPassesResult result =
            checkpointRenderMicroStintLoop(driver, fs, adaptiveIterationPixSampleIdTable,
                                           startSampleId, endSampleId);
        if (result != RenderPassesResult::OK_AND_TRY_NEXT) {
            return result;
        }

        if (tileSampleSpecialEvent && tileSampleSpecialEvent->table().size()) {
            // Only test is specialEventTileSampleIdTable size is not 0 case.
            if (isRequiredSpecialEvent(tileSampleSpecialEvent->table(), endSampleId)) {
                // Execute specialEvent call back procedure. This call is done by single thread
                // and always run thread safe way.
                if (!tileSampleSpecialEvent->callBack()(endSampleId - 1)) {
                    return RenderPassesResult::ERROR_OR_CANCEL;
                }
            }
        }
    }

    return RenderPassesResult::OK_AND_TRY_NEXT;
}

// static function
RenderDriver::UIntTable
RenderDriver::calcCheckpointMiniStintStartEndId(const unsigned startSampleId,
                                                const unsigned endSampleId,
                                                const TileSampleSpecialEvent *tileSampleSpecialEvent)
//
// Create mini-stint tile sample id table based on tileSampleSpecialEvent information
//
{
    UIntTable miniStintSampleIdArray;
    if (!tileSampleSpecialEvent || !tileSampleSpecialEvent->table().size()) {
        //
        // no tileSampleSpecialEvent information or it's table is empty
        // -> no need to split start/end sampleId span.
        //
        miniStintSampleIdArray.push_back(startSampleId);
        miniStintSampleIdArray.push_back(endSampleId);
        return miniStintSampleIdArray;
    }

    // Find end sampleId based on current start/end sampleId span with specialEventTileSampleIdArray
    auto findEndId = [&](const UIntTable &tbl,
                         const unsigned startSampleId, const unsigned endSampleId) -> unsigned {
        auto itrStart = std::lower_bound(tbl.begin(), tbl.end(), startSampleId);
        auto itrEnd = std::lower_bound(itrStart, tbl.end(), endSampleId - 1);
        if (itrStart == itrEnd) {
            return endSampleId;
        }
        return (*itrStart) + 1;
    };

    miniStintSampleIdArray.push_back(startSampleId);
    unsigned currStartSampleId = startSampleId;
    while (1) {
        if (currStartSampleId == endSampleId) break;
        unsigned currEndSampleId = findEndId(tileSampleSpecialEvent->table(), currStartSampleId, endSampleId);
        miniStintSampleIdArray.push_back(currEndSampleId);
        currStartSampleId = currEndSampleId;
    }

    return miniStintSampleIdArray;
}

// static function
bool
RenderDriver::isRequiredSpecialEvent(const UIntTable &specialEventTileSampleIdArray,
                                     const unsigned endSampleId)
//
// Determine this endSampleId needs to do special event (=true) or not (=false).
//
{
    auto itrEnd =
        std::lower_bound(specialEventTileSampleIdArray.begin(), specialEventTileSampleIdArray.end(),
                         endSampleId - 1);
    if ((*itrEnd) + 1 == endSampleId) return true;
    return false;
}

RenderDriver::RenderPassesResult
RenderDriver::checkpointRenderMicroStintLoop(RenderDriver *driver,
                                             const FrameState &fs,
                                             const UIntTable *adaptiveIterationPixSampleIdTable,
                                             unsigned &startSampleId, unsigned &endSampleId)
{
    CheckpointController &checkpointController = driver->mCheckpointController;
    if (!checkpointController.isMemorySnapshotActive()) {
        // disable signal-based checkpoint functionality
        return checkpointRenderMicroStint(driver, fs, adaptiveIterationPixSampleIdTable,
                                          startSampleId, endSampleId);
    }

#   ifdef PRINT_DEBUG_MESSAGE_PROGRESS_MICROCHECKPOINTSTINTLOOP
    std::cerr << ">> RenderFrameCheckpointResume.cc microStintLoop start ... (start:" << startSampleId << " end:" << endSampleId << ")\n";
#   endif // end PRINT_DEBUG_MESSAGE_PROGRESS_MICROCHECKPOINTSTINTLOOP

    // signal-based checkpoint enable
    RenderPassesResult result;

    unsigned currStartSampleId = startSampleId;
    unsigned currEndSampleId = checkpointController.estimateEndSampleId(currStartSampleId, endSampleId);
    while (true) {
        checkpointController.microStintStart();
        result = checkpointRenderMicroStint(driver, fs, adaptiveIterationPixSampleIdTable,
                                            currStartSampleId, currEndSampleId);
        if (result != RenderPassesResult::OK_AND_TRY_NEXT) {
            // just in case, update startSampleId and endSampleId. They aren't important !OK_AND_TRYNEXT case
            startSampleId = currStartSampleId;
            endSampleId = currEndSampleId;
            return result;
        }
        if (checkpointController.microStintEnd()) {
            // Need to update MCRT end timing for resumeHistory here.
            fs.mRenderContext->getResumeHistoryMetaData()->setMCRTStintEndTime(currEndSampleId - 1, true);

            // do extra-snapshot for unexpected interruption by signal.
            checkpointController.snapshotOnly(fs.mRenderContext,
                                              driver->mCheckpointPostScript,
                                              currEndSampleId);
        }

        if (currEndSampleId == endSampleId) break; // completed all samples -> exit loop

        currStartSampleId = currEndSampleId;
        currEndSampleId = checkpointController.estimateEndSampleId(currStartSampleId, endSampleId);
    }

#   ifdef PRINT_DEBUG_MESSAGE_PROGRESS_MICROCHECKPOINTSTINTLOOP
    std::cerr << ">> RenderFrameCheckpointResume.cc microStintLoop ... end\n";
#   endif // end PRINT_DEBUG_MESSAGE_PROGRESS_MICROCHECKPOINTSTINTLOOP

    return result;
}

// static function
RenderDriver::RenderPassesResult
RenderDriver::checkpointRenderMicroStint(RenderDriver *driver,
                                        const FrameState &fs,
                                        const UIntTable *adaptiveIterationPixSampleIdTable,
                                        const unsigned &startSampleId, const unsigned &endSampleId)
//
// Render single checkpoint mini-stint
//
// adaptiveIterationPixSampleIdTable is only used by adaptive sampling mode
//
{
    //------------------------------
    //
    // Create rendering passes based on the start/end sampleId
    //
    std::vector<Pass> newPasses;
    if (fs.mSamplingMode == SamplingMode::UNIFORM) {
        convertSampleIdRangeToPasses(startSampleId, endSampleId, newPasses);
    } else {
        convertSampleIdRangeToPassesAdaptive<false>(startSampleId, endSampleId,
                                                    *adaptiveIterationPixSampleIdTable, newPasses,
                                                    nullptr);
    }
#   ifdef PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT
    {
        unsigned finalTileSamples = fs.mMaxSamplesPerPixel * 64;
        std::cerr << ">> RenderFrameCheckpointResume.cc before renderPasses()"
                  << " checkpointMode:" << checkpointModeStr(fs.mCheckpointMode)
                  << " startSampleId:" << startSampleId << " endSampleId:" << endSampleId
                  << " finalTileSamples:" << finalTileSamples << '\n'
                  << showAllPasses("", newPasses, startSampleId, endSampleId,
                                   driver->mAdaptiveTileSampleCap, finalTileSamples) << std::endl;
    }
#   endif // end PRINT_DEBUG_MESSAGE_PROGRESS_CHECKPOINT

    //------------------------------
    //
    // Setup new workQueue and render
    //        
    // Add new samples (which probably has multiple passes)
    TileWorkQueue *workQueue = &driver->mTileWorkQueue;
    workQueue->init(RenderMode::PROGRESS_CHECKPOINT,
                    unsigned(driver->getTiles()->size()), // numTiles
                    newPasses.size(), mcrt_common::getNumTBBThreads(), &newPasses[0]);

    RenderDriver::RenderPassesResult result = renderPasses(driver, fs, true); // rendering is done by renderPasses() by multi-threaded

#   ifdef RUNTIME_VERIFY1
    TileGroupRuntimeVerify::get()->verify();
#   endif // end RUNTIME_VERIFY1

#   ifdef RUNTIME_VERIFY0
    std::cerr << ">> RenderFrameCheckpointResume.cc " << PixSampleCountRuntimeVerify::get()->show() << '\n';
#   endif // end RUNTIME_VERIFY0

    return result;
}

// static function
void
RenderDriver::checkpointFileOutput(RenderDriver *driver, const FrameState &fs, const unsigned endSampleId)
//
// Write out checkpoint file
//
{
    // endSampleId is a tile based sampling number and tile size is 8x8.
    // So we need to multiply 64 here.
    if (endSampleId < fs.mCheckpointStartSPP * 64) {
        std::ostringstream ostr;
        ostr << "Skip checkpoint file output. tile-sampleTotal:" << endSampleId
             << " (pixel-sampleTotal:" << (float)endSampleId / 64.f << ")";
        scene_rdl2::logging::Logger::info(ostr.str());
        return; // sample total is not enough to dump out checkpoint file.
    }

    const RenderOutputDriver *renderOutputDriver = fs.mRenderContext->getRenderOutputDriver();
    if (!renderOutputDriver) return; // skip output. just in case

    if (static_cast<int>(endSampleId) == driver->mLastCheckpointFileEndSampleId) {
        // We skip duplicate output requests If this checkpoint file output is
        // exactly the same with previous checkpoint file out.
        return;
    }
    driver->mLastCheckpointFileEndSampleId = endSampleId; // update last checkpointFile endSampleId

    driver->mProgressEstimation.checkpointOutput(true, endSampleId);
    driver->mCheckpointController.output(fs.mCheckpointBgWrite,
                                         fs.mTwoStageOutput,
                                         fs.mRenderContext,
                                         driver->mCheckpointPostScript,
                                         endSampleId);
    driver->mProgressEstimation.checkpointOutput(false);
}

// static function
std::string
RenderDriver::checkpointModeStr(const CheckpointMode &mode)
{
    std::string str;
    switch (mode) {
    case CheckpointMode::TIME_BASED : str = "TIME_BASED"; break;
    case CheckpointMode::QUALITY_BASED : str = "QUALITY_BASED"; break;
    default : break;
    }
    return str;
}

#ifdef VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC
bool
RenderDriver::verifyPassesLogicForAdaptiveSampling()
//
// Verify adaptive sampling passes generation logic. This is test purpose function.
//    
{
    // Generate test sample range tables by random span. Using this table for verify main logic of
    // adaptive sampling passes construction.
    auto genTestTileSampleTbl = [](const unsigned tileSampleMin, const unsigned tileSampleMax,
                                   const unsigned stepMin, const unsigned stepMax) -> std::vector<unsigned> {
        std::random_device rd;
        scene_rdl2::util::Random rng(rd());
        std::uniform_int_distribution<unsigned> randDist(stepMin, stepMax);

        std::vector<unsigned> sampleTbl;
        unsigned currSampleId = tileSampleMin;
        while (1) {
            currSampleId = std::min(currSampleId, tileSampleMax);
            sampleTbl.push_back(currSampleId);
            if (currSampleId == tileSampleMax) break;
            currSampleId += randDist(rng);
        }
        return sampleTbl;
    };

    // create strings for dump passes information
    auto dumpPasses = [](const std::vector<Pass> &passes) -> std::string {
        auto getPassW = [](const std::vector<Pass> &passes, int offset) -> size_t {
            auto getPassMax = [](const std::vector<Pass> &passes, int offset) -> unsigned {
                auto getPassV = [](const Pass &p, int off) -> unsigned { return *((const unsigned *)&p + off); };
                return getPassV(*(std::max_element(passes.begin(), passes.end(),
                                                   [&](const Pass &a, const Pass &b) {
                                                       return getPassV(a, offset) < getPassV(b, offset);
                                                   })), offset);
            };
            return std::to_string(getPassMax(passes, offset)).size();
        };
        std::ostringstream ostr;
        ostr << "passes (total:" << passes.size() << ") {\n";
        for (size_t i = 0; i < passes.size(); ++i) {
            ostr << "  i:" << std::setw(std::to_string(passes.size()).size()) << i << " pix("
                 << std::setw(getPassW(passes, 0)) << passes[i].mStartPixelIdx << '-'
                 << std::setw(getPassW(passes, 1)) << passes[i].mEndPixelIdx << ") smp("
                 << std::setw(getPassW(passes, 2)) << passes[i].mStartSampleIdx << '-'
                 << std::setw(getPassW(passes, 3)) << passes[i].mEndSampleIdx << ")\n";
        }
        ostr << "}";
        return ostr.str();
    };

    auto countPassesTotalTileSamples = [](const std::vector<Pass> &passes) -> unsigned {
        unsigned total = 0;
        for (const auto &itr : passes) {
            total += (itr.mEndPixelIdx - itr.mStartPixelIdx) * (itr.mEndSampleIdx - itr.mStartSampleIdx);
        }
        return total;
    };

    constexpr unsigned finalPixSamples = 4096; // max pixel samples for this test.
    constexpr int testMax = 1024; // max verify count
    constexpr int testTileSampleMax = finalPixSamples * 64;
    constexpr int testStepMinTileSamples = 16;
    constexpr int testStepMaxTileSamples = 1024;

    std::vector<unsigned> adaptIterationPixSampleTbl = createAdaptiveIterationPixSampleIdTable(finalPixSamples);

    bool returnVerifyCondition = true;
    for (int testId = 0; testId < testMax; ++testId) {
        std::vector<unsigned> tileSampleTbl = genTestTileSampleTbl(0,
                                                                   testTileSampleMax,
                                                                   testStepMinTileSamples,
                                                                   testStepMaxTileSamples);
        std::ostringstream ostr;
        unsigned totalTileSamples = 0;
        for (size_t i = 0; i < tileSampleTbl.size() - 1; ++i) {
            unsigned currTileStartSampleId = tileSampleTbl[i];
            unsigned currTileEndSampleId = tileSampleTbl[i + 1];

            std::vector<Pass> passes;
            convertSampleIdRangeToPassesAdaptive<false>(currTileStartSampleId, currTileEndSampleId,
                                                        adaptIterationPixSampleTbl, passes,
                                                        nullptr);
            unsigned currTileSamples = countPassesTotalTileSamples(passes);
            totalTileSamples += currTileSamples;

            bool currStat = (currTileSamples == currTileEndSampleId - currTileStartSampleId);
            bool wholeStat = (totalTileSamples == tileSampleTbl[i + 1]);
            if (!currStat || !wholeStat) {
                ostr << "i:" << i
                     << " tileSample(" << currTileStartSampleId << '-' << currTileEndSampleId
                     << " currVerify:" << ((currStat)? "OK" : "NG")
                     << " wholeVerify:" << ((wholeStat)? "OK" : "NG") << ") "
                     << dumpPasses(passes) << '\n';
                std::vector<Pass> tmpPasses;
                convertSampleIdRangeToPassesAdaptive<true>(currTileStartSampleId, currTileEndSampleId,
                                                           adaptIterationPixSampleTbl, tmpPasses,
                                                           &ostr);
            }
        }
        bool finalStat = (totalTileSamples == tileSampleTbl[tileSampleTbl.size() - 1]);
        ostr << "testId:" << testId
             << " final verify tileSampleTbl.size():" << tileSampleTbl.size()
             << " result:" << ((finalStat)? "OK" : "NG");
        std::cerr << ostr.str() << std::endl; // We need to printout. This is verify function for debug
        if (!finalStat) returnVerifyCondition = false;
    }

    return returnVerifyCondition;
}
#endif // end VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC

} // namespace rndr
} // namespace moonray
