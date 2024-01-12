// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/render/util/AtomicFloat.h> // Needs to be included before any OpenImageIO file
#include "ExrUtils.h"
#include "ImageWriteCache.h"
#include "ImageWriteDriver.h"
#include "RenderContext.h"
#include "RenderDriver.h"
#include "RenderOutputDriverImpl.h"
#include "RenderOutputWriter.h"
#include "ResumeHistoryMetaData.h"

#include <moonray/rendering/pbr/core/DeepBuffer.h>
#include <scene_rdl2/common/grid_util/Sha1Util.h>
#include <scene_rdl2/render/util/Strings.h>

// This directive is used to verify logic correctness between ENQ and DEQ action by SHA1 hash value.
// We can control hash computation precision as well.
// See PRECISE_HASH_COMPARE directive (RenderOutputWriter.cc) for more detail.
// This directive should commented out for release version.
//#define RUNTIME_VERIFY_HASH

namespace moonray {
namespace rndr {

void
RenderOutputDriver::Impl::writeDeq(ImageWriteCache *cache,
                                   const bool checkpointOutputMultiVersion,
                                   scene_rdl2::grid_util::Sha1Gen::Hash *hashOut) const
{
    
    write(true, checkpointOutputMultiVersion,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
          cache, hashOut);
}

void
RenderOutputDriver::Impl::write(const bool checkpointOutput,
                                const bool checkpointOutputMultiVersion,
                                const pbr::DeepBuffer *deepBuffer,
                                pbr::CryptomatteBuffer *cryptomatteBuffer,
                                const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                                const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> *aovBuffers,
                                const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> *displayFilterBuffers,
                                const unsigned checkpointTileSampleTotals, // only used for STD and ENQ
                                ImageWriteCache *cache,
                                scene_rdl2::grid_util::Sha1Gen::Hash *hashOut) const
//
// weightBuffer : only has valid info when RenderOutputDriver has weightAOV.
// checkpointOutputMultiVersion : runtime multi-version checkpoint file output condition.
//                                This flag only used under checkpointOutput = on
//
// -- Write image by background thread --
// Now this write() function has 3 different run mode. STD, ENQ and DEQ.
// STD : standard output mode which is not using background write mode.
// ENQ : same logic of STD but not create file and just create ImageWriteCache data instead.
// DEQ : create file from ImageWriteCache.
//
// STD runMode is still used for final image output stage (i.e. not a checkpoint file out).
// And also used by checkpoint write with "checkpoint_bg_write" = false situation.
//
// Background checkpoint write is done by 2 passes. ENQ pass and DEQ pass.
// ENQ pass is done by renderDriver thread and create ImageWriteCache data. This is pretty quick.
// DEQ pass is done by ImageWriteDriver thread independent from renderDriver thread tree by parallel.
// Scene variable "checkpoint_bg_write" is used for disable/enable of background write.
// Default is "true".
//
// DeepFile output is only working under STD mode at this moment.
//
// -- Two stage output mode --
// In order to reduce the risk of kill/terminate signal during multiple image write situation,
// we have special logic calls "two stage output" mode. If multiple images are not update
// correctly due to terminate render process, all generated image files are not sync and
// we can not resume from this output images.
// This "two stage output" creates temporary file first and after that copy them to the final
// destination as temporary name. Then rename from destination temporary name to destination final
// name. If your renderOutput creates multiple files, all files are created as temporary files first
// then after finish all files are ready, copy to the final destination as temporary name files.
// After that, all destination temporary files are renamed to final name files.
// Typically final rename stage is much faster than directory create that files. So this solution can
// greatly reduce the risk of generate out of sync situation of the multiple image output.
//
// If "two stage output" mode is enabled, this write() function try to create file as tmp file.
// copy and rename stage is done by later. (See ImageWriteCache::allFinalize*File() API).
// moonray clean up the temporary files by itself but some case there is some possibility to leave
// the tmp file unfortunately.
// if you find the file like "*.part" with old timestamp, this is a garbage file and need
// to be removed. This file is destination temp file and usually renamed to final filename after
// copy is done.
//
// This "two stage output" is only support non DeepFile output so far.
//
{
    if (checkpointOutputMultiVersion && mOverwriteCheckpoint) {
        // checkpointOutputMultiVersion only available when checkpointOverwrite=false situation.
        return; // early exit. This behavior is requested. (See RenderDriver::checkpointFileOutput()).
    }

    std::vector<std::string> &errors = (cache)? cache->getErrors(): mErrors;
    std::vector<std::string> &infos = (cache)? cache->getInfos(): mInfos;

    scene_rdl2::grid_util::Sha1Gen sha1Gen;
    scene_rdl2::grid_util::Sha1Gen *sha1GenPtr = (hashOut) ? &sha1Gen : nullptr;

    if (cache) cache->timeStart();

    ImageWriteCache::Mode runMode = ImageWriteCache::runMode(cache);

    if (runMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        int predefinedWidth = 0;
        int predefinedHeight = 0;
        if (deepBuffer) {
            predefinedWidth = deepBuffer->getWidth();
            predefinedHeight = deepBuffer->getHeight();
        }

        //------------------------------
        // A 2D buffer needed by DeepBuffer and Cryptomatte that holds
        // the number of samples per pixel, which can vary when using adaptive sampling.
        scene_rdl2::fb_util::PixelBuffer<unsigned> samplesCount;
        if (deepBuffer || cryptomatteBuffer) {
            const Film& film = rndr::getRenderDriver()->getFilm();
            film.fillPixelSampleCountBuffer(samplesCount);
        }
        if (cache) cache->timeRec(0); // record timing into position id = 0

        //------------------------------
        if (cryptomatteBuffer) {
            cryptomatteBuffer->finalize(samplesCount);
        }
        if (cache) cache->timeRec(1); // record timing into position id = 1

        //------------------------------
        FileNameParam fileNameParam(checkpointOutput,
                                    checkpointOutputMultiVersion,
                                    mOverwriteCheckpoint,
                                    mFinalMaxSamplesPerPixel);

        RenderOutputWriter::CallBackCheckpointResumeMetadata callBackMetadata =
            [&](const unsigned currCheckpointTileSampleTotals) -> std::vector<std::string> {
            // callback for checkpointResumeMetaData
            std::vector<std::string> attrTbl;
            if (mCheckpointRenderActive) {
                attrTbl = writeCheckpointResumeMetadata(currCheckpointTileSampleTotals);

                // save timeSaveSecBySignalCheckpoint value into cache for statistical info dump
                float sec = mRenderContext->getResumeHistoryMetaData()->getTimeSaveSecBySignalCheckpoint();
                cache->setTimeSaveSecBySignalCheckpoint(sec);
            }
            return attrTbl;
        };

        RenderOutputWriter writer(runMode,
                                  cache,
                                  checkpointTileSampleTotals,
                                  predefinedWidth, predefinedHeight,
                                  &mFiles,
                                  &fileNameParam,
                                  callBackMetadata,
                                  cryptomatteBuffer, heatMap, weightBuffer, renderBufferOdd,
                                  aovBuffers, displayFilterBuffers,
                                  errors, infos,
                                  sha1GenPtr);
        if (cache) cache->timeRec(2); // record timing into position id = 2

        writer.main();
        if (cache) cache->timeRec(3); // record timing into position id = 3

        // Here we "unfinalize" the cryptomatte buffer, putting it into a state ready for further accumulation
        // of samples following a checkpoint (which would have incurred a finalize operation)
        if (cryptomatteBuffer) {
            cryptomatteBuffer->unfinalize(samplesCount);
        }

        if (deepBuffer && runMode == ImageWriteCache::Mode::STD) {
            deepWrite(checkpointOutput, checkpointOutputMultiVersion, checkpointTileSampleTotals,
                      samplesCount, deepBuffer);
        }

    } else { // DEQ
        if (cache) cache->timeRec(0); // record timing into position id = 0
        if (cache) cache->timeRec(1); // record timing into position id = 1

        FileNameParam fileNameParam(checkpointOutputMultiVersion);
        RenderOutputWriter writer(cache,
                                  &mFiles,
                                  &fileNameParam,
                                  errors, infos,
                                  sha1GenPtr);
        if (cache) cache->timeRec(2); // record timing into position id = 2

        writer.main();
        if (cache) cache->timeRec(3); // record timing into position id = 3
    }

    if (cache) {
        cache->timeRec(4); // record timing into position id = 4
        cache->timeEnd();
        // std::cerr << ">> RenderOutputDriver.cc " << cache->timeShow() << std::endl; // useful debug dump
    }

    if (hashOut) {
        *hashOut = sha1GenPtr->finalize();
    }
}

void
RenderOutputDriver::Impl::deepWrite(const bool checkpointOutput,
                                    const bool checkpointOutputMultiVersion,
                                    const unsigned checkpointTileSampleTotals,
                                    const scene_rdl2::fb_util::PixelBuffer<unsigned> &samplesCount,
                                    const pbr::DeepBuffer *deepBuffer) const
{
    std::vector<std::string> &errors = mErrors;
    std::vector<std::string> &infos = mInfos;

    int deepAOV = 0;
    for (const auto &f: mFiles) {

        // We are not supporting resume render of deep data but we support generating checkpoint deep file.
        std::string filename = ((checkpointOutput) ?
                                ((checkpointOutputMultiVersion) ?
                                 RenderOutputWriter::generateCheckpointMultiVersionFilename(f,
                                                                                            mOverwriteCheckpoint,
                                                                                            mFinalMaxSamplesPerPixel,
                                                                                            checkpointTileSampleTotals) :
                                 f.mCheckpointName) :
                                f.mName);

        // Make sure all of this file's outputs are "deep"
        // You cannot mix "deep" and non-deep outputs in the same file!  Deep
        // images are output using OpenDCX and non-deep images may be output with
        // a different library, e.g. OIIO.
        bool hasDeep = false;
        bool hasNonDeep = false;
        int numOutputs = 0;
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getOutputType() == std::string("deep")) {
                    hasDeep = true;
                } else {
                    hasNonDeep = true;
                }
                numOutputs++;
            }
        }
        if (!hasDeep && hasNonDeep) {
            // non-deep images are output elsewhere
            // need to skip over non-deep outputs but increment aov index
            for (const auto &i: f.mImages) {
                for (const auto &e: i.mEntries) {
                    if (e.mAovSchemaId != pbr::AOV_SCHEMA_ID_UNKNOWN) {
                        deepAOV++;
                    }
                }
            }
            continue;
        }
        if (hasDeep && hasNonDeep) {
            errors.push_back
                (scene_rdl2::util::buildString("ERROR: output file '", filename, "' has a mixture of deep and non-deep images."));
            continue; // next!
        }

        // Figure out what we're outputting in this deep file.
        //  It can be any combination of aovs, collect these up from the
        //  images + entries.
        std::vector<int> outputAOVs;
        std::vector<std::string> outputAOVChannelNames;
        const scene_rdl2::rdl2::Metadata *rdlMetadata = nullptr;

        const scene_rdl2::rdl2::SceneVariables& vars = f.mImages[0].mEntries[0].mRenderOutput->getSceneClass().getSceneContext()->getSceneVariables();
        const scene_rdl2::math::HalfOpenViewport aperture = vars.getRezedApertureWindow();
        const scene_rdl2::math::HalfOpenViewport region = vars.getRezedRegionWindow();

        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                const scene_rdl2::rdl2::SceneObject *metadata = ro->getExrHeaderAttributes();
                if (metadata) {
                    // use first valid metadata we find, same as flat case above
                    rdlMetadata = metadata->asA<scene_rdl2::rdl2::Metadata>();
                }

                outputAOVs.push_back(deepAOV);
                for (size_t cn = 0; cn < e.mChannelNames.size(); cn++) {
                    // need the channel names so they are named correctly in the deep file
                    outputAOVChannelNames.push_back(e.mChannelNames[cn]);
                }
                deepAOV++;
            }
        }

        if (!filename.empty()) {
            deepBuffer->write(filename, outputAOVs, outputAOVChannelNames, samplesCount, aperture, region, rdlMetadata);

            infos.push_back(scene_rdl2::util::buildString("Wrote deep: ", filename));
        }
    }
}

std::vector<std::string>
RenderOutputDriver::Impl::writeCheckpointResumeMetadata(const unsigned checkpointTileSampleTotals) const
{
    //
    // Dump checkpoint/resume related internal information into metadata.
    // This information is always write out when checkpoint mode is on
    // regardless of resumable_output condition.
    // If you need to resume from this output image, you need to generate this
    // output image with resumable_output = on.
    // Keep in mind these metadata information is not guarantee that this file
    // can be used for input of resume render.
    //

    std::vector<std::string> outTbl;

    unsigned tileSampleTotal = 0;
    {
        //
        // This is a information for uniform/adaptive sampling version of progress checkpoint rendering.
        // Keep to store how many samples are completed in each tile. This information is used by resumed
        // process and skip previously completed part of samplings and start resume render properly.
        //
        tileSampleTotal = checkpointTileSampleTotals;
        if (tileSampleTotal == 0) {
            if (mLastTileSamples > 0) {
                // Checkpoint render completed with normal/TimeCap situation, renderer
                // sets mLastTileSamples properly. Otherwise it's 0 (= non checkpoint render).
                tileSampleTotal = mLastTileSamples;
            } else {
                // This is a case of regular output file. We should re-compute tilebase sample total based on
                // finalMaxSamplesPerPixel information.
                tileSampleTotal = mFinalMaxSamplesPerPixel * 64; // tile = 8*8
            }
        }
        outTbl.emplace_back("progressCheckpointTileSamples");
        outTbl.emplace_back("int");
        outTbl.emplace_back(std::to_string(tileSampleTotal));
    }

    {
        //
        // This is a information of numConsistentSamples number for FORCE_CONSISTENT_SAMPLING Aov filter.
        // Regardress of this output uses this AOV filter or not, just in case we should output
        // this number into metadata.
        // If this AOV filter is used, at resume Film object stage, denormalize weight operation is used
        // this minAdaptiveSampes value insted of "on the fly" computation value at resumed process.
        // "On the fly" computation might have some risk to have different number due to change of sampling
        // parameters for re-run environment.
        //
        unsigned numConsistentSamples = mRenderContext->getNumConsistentSamples();
        outTbl.emplace_back("AovFilterNumConsistentSamples");
        outTbl.emplace_back("int");
        outTbl.emplace_back(std::to_string(numConsistentSamples));
    }

    {
        if (mRenderContext->getSamplingMode() != SamplingMode::UNIFORM) {
            //
            // This is a information of sampling type. We record sampling type and it's parameters
            // into metadata for resume render when non uniform sampling case.
            //
            if (mRenderContext->getSamplingMode() == SamplingMode::ADAPTIVE) {
                unsigned minSamples, maxSamples;
                float targetError;
                mRenderContext->getAdaptiveSamplingParam(minSamples, maxSamples, targetError);

                std::ostringstream ostr;
                ostr << minSamples << ' ' << maxSamples << ' ' << targetError;
                outTbl.emplace_back("adaptiveSamplingV1");
                outTbl.emplace_back("v3f");
                outTbl.emplace_back(ostr.str());
            }
        }
    }

    {
        //
        // This is a resume render history information. We should add current render information
        //
        ResumeHistoryMetaData *resumeHistoryMetaData =
            const_cast<ResumeHistoryMetaData *>(mRenderContext->getResumeHistoryMetaData());
        if (resumeHistoryMetaData) {
            outTbl.emplace_back("resumeHistory");
            outTbl.emplace_back("string");
            outTbl.emplace_back(resumeHistoryMetaData->
                                resumeHistory(mResumeHistory,
                                              mRenderContext->accumulatePbrStatistics()));
        }
    }

    return outTbl;
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

void
RenderOutputDriver::writeFinal(const pbr::DeepBuffer *deepBuffer,
                               pbr::CryptomatteBuffer *cryptomatteBuffer,
                               const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                               const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                               const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers,
                               ImageWriteCache *cache) const
//
// This function writes image as final output.
//
{
    mImpl->write(false, false,
                 deepBuffer, cryptomatteBuffer, heatMap, weightBuffer,
                 renderBufferOdd, &aovBuffers, &displayFilterBuffers, 0,
                 cache, nullptr);
}

void
RenderOutputDriver::writeCheckpointEnq(const bool checkpointMultiVersion,
                                       const pbr::DeepBuffer *deepBuffer,
                                       pbr::CryptomatteBuffer *cryptomatteBuffer,
                                       const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                                       const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                       const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                       const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                                       const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers,
                                       const unsigned tileSampleTotals,
                                       ImageWriteCache *outCache) const
{
    scene_rdl2::grid_util::Sha1Gen::Hash *hashPtr = nullptr;
#   ifdef RUNTIME_VERIFY_HASH
    if (outCache) {
        scene_rdl2::grid_util::Sha1Util::init(outCache->dataHash());
        hashPtr = &outCache->dataHash();
    }
#   endif // end RUNTIME_VERIFY_HASH

    if (hashPtr) {
        if (checkpointMultiVersion && mImpl->getOverwriteCheckpoint()) {
            // checkpointMultiVersion only available when checkpointOverwrite=false situation.
        } else {
            mImpl->write(false, false,
                         deepBuffer, cryptomatteBuffer, heatMap, weightBuffer,
                         renderBufferOdd, &aovBuffers, &displayFilterBuffers, tileSampleTotals,
                         nullptr, hashPtr);
        }
    }

    mImpl->write(true, checkpointMultiVersion,
                 deepBuffer, cryptomatteBuffer, heatMap, weightBuffer,
                 renderBufferOdd, &aovBuffers, &displayFilterBuffers, tileSampleTotals,
                 outCache, nullptr);
}

void
RenderOutputDriver::writeCheckpointDeq(ImageWriteCache *cache,
                                       const bool checkpointMultiVersion) const
{
    cache->setupDeqMode();

    scene_rdl2::grid_util::Sha1Gen::Hash hash;
    scene_rdl2::grid_util::Sha1Gen::Hash *hashPtr = nullptr;
    if (!scene_rdl2::grid_util::Sha1Util::isInit(cache->dataHash())) {
        // cache has Enq dataHash, so we will try to compute Deq data hash for verify
        hashPtr = &hash;
    }

    mImpl->writeDeq(cache, checkpointMultiVersion, hashPtr);

    if (hashPtr) {
        if (cache->dataHash() == *hashPtr) {
            std::cerr << "RenderOutputDriverImplWrite.cc : runtime Enq/Deq hash compare OK\n";
        } else {
            std::cerr << "RenderOutputDriverImplWrite.cc : runtime Enq/Deq hash compare failed\n";
        }
        std::cerr << "RenderOutputDriverImplWrite.cc Enq() " << scene_rdl2::grid_util::Sha1Util::show(cache->dataHash()) << '\n';
        std::cerr << "RenderOutputDriverImplWrite.cc Deq() " << scene_rdl2::grid_util::Sha1Util::show(*hashPtr) << '\n';
    }
}

} // namespace rndr
} // namespace moonray
