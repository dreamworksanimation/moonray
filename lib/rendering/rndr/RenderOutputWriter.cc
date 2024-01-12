// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ExrUtils.h"
#include "ImageWriteDriver.h"
#include "RenderOutputDriverImpl.h"
#include "RenderOutputWriter.h"

#include <moonray/rendering/mcrt_common/Clock.h>
#include <moonray/rendering/pbr/core/Cryptomatte.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/common/grid_util/Sha1Util.h>
#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/render/util/Strings.h>

// Useful runtime verify for single float to half float conversion for ImageWriteCache
//#define RUNTIME_VERIFY_FTOH

// Useful runtime verifies for imageSpecId calculation between STD/ENQ and DEQ.
// This is for debugging purposes.This should be commented out for the release version.
//#define RUNTIME_VERIFY_IMGSPECID

// We can compute write data hash and this hash is used for debugging purposes.
// (Hash computation should be off for the released version. We should set nullptr to RenderOutputWriter's
// sha1Gen argument).
// If this directive is commented out, write data hash computation is done as follows.
//   a) not include any timing / date related information into the hash target
//   b) pixel value precision is adjusted to a pretty low resolution.
// Typical use of commented out of this directive is that we want to compare result images correctness by
// hash value between different run of moonray. If we get the same hash result from 2 different versions of
// moonray, writer logic should be working precisely correctly.
// Yes, moonray has some non-determinism and different run might create different pixel value results
// run by run. This is the main reason why we need low-precision pixel data for SHA1 hash generation.
// Under the purpose of this hash, It is not so important about the pixel value itself is exactly the same
// or not. I only care about writing data logic is exactly the same between two different versions of moonray.
// If we can get the same SHA1 hash from 2 different versions of moonray, pixel value might be slightly
// different by non-determinism but we can guarantee the moonray did exactly the same data write out logic by
// the same order.
// Unfortunately, this low precision pixel value solution is not perfect and there is some possibility to
// create a different hash from 2 different run even no code changed due to non-determinism.
// But in my test, this low precision pixel solution is working well especially very 1st checkpoint data
// (like 1SPP image). This solution greatly reduced the testing cost during refactoring write logic.
//
// If this directive is not commented out, all the write-out data is targeted for hash computation.
// If you get the same hash value between ENQ and DEQ, ENQ and DEQ logic is working as we expected.
//#define PRECISE_HASH_COMPARE

namespace moonray {
namespace rndr {

std::string
FileNameParam::show() const
{
    auto boolStr = [](const bool b) -> std::string {
        return scene_rdl2::str_util::boolStr(b);
    };

    std::ostringstream ostr;
    ostr << "FileNameParam {\n"
         << "  mCheckpointOutput:" << boolStr(mCheckpointOutput) << '\n'
         << "  mCheckpointOutputMultiVersion:" << boolStr(mCheckpointOutputMultiVersion) << '\n'
         << "  mOverwriteCheckpoint:" << boolStr(mOverwriteCheckpoint) << '\n'
         << "  mFinalMaxSamplesPerPixel:" << mFinalMaxSamplesPerPixel << '\n'
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

bool
RenderOutputWriter::main() const
{
    bool result = true;
    for (size_t fileId = 0; fileId < mCurrBufferSpec->getFileTotal(); ++fileId) {
        if (!singleFileOutput(fileId)) {
            result = false;
            // we want to continue to write lest of the files even some file write is failed.
        }
    }

    return result;
}

// static
std::string
RenderOutputWriter::generateCheckpointMultiVersionFilename(const File& file,
                                                           const bool overwriteCheckpoint,
                                                           const unsigned finalMaxSamplesPerPixel,
                                                           const unsigned checkpointTileSampleTotals)
//
// This function generates a filename for multi-version output.
// Multi-version output is only available under checkpoint_overwrite = false.
//
{
    if (overwriteCheckpoint) {
        return ""; // empty if overwrite=true mode
    }

    std::string filename = file.mCheckpointName;
    if (!file.mCheckpointMultiVersionName.empty()) {
        // pick up checkpoint multi-version names when we have
        filename = file.mCheckpointMultiVersionName;
    }

    int w = std::to_string(finalMaxSamplesPerPixel * 64).size();
    std::ostringstream ostr;
    ostr << '_' << std::setw(w) << std::setfill('0') << checkpointTileSampleTotals;
    auto pos = filename.rfind(".exr");
    if (pos == std::string::npos) {
        filename.append(ostr.str());
    } else {
        filename.insert(pos, ostr.str());
    }
    return filename;
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

void
RenderOutputWriter::setupBuffOffsetTable()
{
    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        if (mRunMode == ImageWriteCache::Mode::STD) {
            mCurrBufferSpec = &mBufferSpec;
        } else { // ENQ
            mCurrBufferSpec = &mCache->bufferSpec();
        }

        // We have to track down the last bufferSpecSubImage by index. The pointer of the last
        // bufferSpecSubImage does not work in this case because it is a part of the vector array
        // and its address might be changed when the vector is growing.
        size_t lastBufferSpecSubImageFileId = ~0;
        size_t lastBufferSpecSubImageId = ~0;

        for (size_t fId = 0; fId < mFiles->size(); ++fId) {
            const File& file = (*mFiles)[fId];
            ImageWriteCacheBufferSpecFile* currBufferSpecFile = mCurrBufferSpec->newFile();

            for (size_t imgId = 0; imgId < file.mImages.size(); ++imgId) {
                const Image& img = file.mImages[imgId];
                
                ImageWriteCacheBufferSpecSubImage* currBufferSpecSubImage = currBufferSpecFile->newSubImage();

                size_t aovBuffSize = 0;
                size_t displayFilterBuffSize = 0;
                std::vector<ImageWriteCacheBufferSpecSubImage::ChanFormat> pixChanFormat;
                std::vector<int> pixNumChan;
                for (const auto &entry: img.mEntries) {
                    if (entry.mAovSchemaId != pbr::AOV_SCHEMA_ID_UNKNOWN) {
                        aovBuffSize++;
                    }
                    if (entry.mRenderOutput->getResult() ==
                        scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
                        displayFilterBuffSize++;
                    }
                    pixNumChan.push_back(entry.mChannelNames.size());
                    switch (entry.mRenderOutput->getChannelFormat()) {
                    case scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF :
                        pixChanFormat.push_back(ImageWriteCacheBufferSpecSubImage::ChanFormat::HALF);
                        break;
                    case scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_FLOAT :
                        pixChanFormat.push_back(ImageWriteCacheBufferSpecSubImage::ChanFormat::FULL);
                        break;
                    default :
                        pixChanFormat.push_back(ImageWriteCacheBufferSpecSubImage::ChanFormat::UNKNOWN);
                        break;
                    }
                } // loop entry

                size_t pixCacheSizeFull = 0;
                size_t pixCacheSizeHalf = 0;
                calcPixCacheSize(img.mEntries, pixCacheSizeFull, pixCacheSizeHalf);

                const ImageWriteCacheBufferSpecSubImage *lastBufferSpecSubImage = nullptr;
                if (lastBufferSpecSubImageFileId != ~0 && lastBufferSpecSubImageId != ~0) {
                    lastBufferSpecSubImage =
                        &mCurrBufferSpec->getBufferSpecFile(lastBufferSpecSubImageFileId).
                        getBufferSpecSubImage(lastBufferSpecSubImageId);
                }
                currBufferSpecSubImage->setup(lastBufferSpecSubImage,
                                              img.mName,
                                              aovBuffSize,
                                              displayFilterBuffSize,
                                              pixCacheSizeFull,
                                              pixCacheSizeHalf,
                                              pixChanFormat,
                                              pixNumChan);
                lastBufferSpecSubImage = currBufferSpecSubImage;

                lastBufferSpecSubImageFileId = fId;
                lastBufferSpecSubImageId = imgId;
            } // loop imgId
        } // loop fId
    } else { // DEQ
        mCurrBufferSpec = &mCache->bufferSpec();
    }

    if (mRunMode != ImageWriteCache::Mode::ENQ && mSha1Gen) { // STD/DEQ && mSha1Gen
        mCurrBufferSpec->updateHash(mSha1Gen);
    }

    // useful debug message
    // std::cerr << "RenderOutputWriter.cc setupBuffOffsetTable() " << mCurrBufferSpec->show() << '\n';
}

bool
RenderOutputWriter::dataValidityCheck(const int predefinedWidth, const int predefinedHeight) const
{
    auto isPredefinedDataActive = [&]() -> bool {
        return (predefinedWidth > 0 && predefinedHeight > 0);
    };

    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        return (isPredefinedDataActive() ||
                mCryptomatteBuffer ||
                mHeatMap ||
                mWeightBuffer ||
                mRenderBufferOdd ||
                !mAovBuffers->empty());
    } else { // DEQ
        return (!isPredefinedDataActive() &&
                !mCryptomatteBuffer &&
                !mHeatMap &&
                !mWeightBuffer &&
                !mRenderBufferOdd &&
                !mAovBuffers);
    }
}

void
RenderOutputWriter::setupCheckpointTileSampleTotals(const unsigned checkpointTileSampleTotals)
{
    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        mCheckpointTileSampleTotals = checkpointTileSampleTotals;
        if (mRunMode == ImageWriteCache::Mode::ENQ) {
            mCache->enq()->enqVLUInt(mCheckpointTileSampleTotals);
        }
        if (mCache) {
            // we store tileSample info to ImageWriteCache just in case
            mCache->setCheckpointTileSampleTotals(mCheckpointTileSampleTotals);
        }
    } else { // DEQ
        mCheckpointTileSampleTotals = mCache->deq()->deqVLUInt();
    }
}

void
RenderOutputWriter::setupWidthHeight(const int predefinedWidth,
                                     const int predefinedHeight)
//
// predefined{Width,Height} is usually defined by deepBuffer information.
// They are 0 when there are no predefined values (i.e. no deepBuffer).
//
{
    mWidth = 0;
    mHeight = 0;
    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        if (predefinedWidth > 0 && predefinedHeight > 0) {
            mWidth = predefinedWidth;
            mHeight = predefinedHeight;
        } else if (mCryptomatteBuffer) {
            mWidth = mCryptomatteBuffer->getWidth();
            mHeight = mCryptomatteBuffer->getHeight();
        } else if (mAovBuffers && !mAovBuffers->empty() &&
                   (*mAovBuffers)[0].getWidth() && (*mAovBuffers)[0].getHeight()) {
            mWidth = (*mAovBuffers)[0].getWidth();
            mHeight = (*mAovBuffers)[0].getHeight();
        } else if (mHeatMap && mHeatMap->getWidth() && mHeatMap->getHeight()) {
            mWidth = mHeatMap->getWidth();
            mHeight = mHeatMap->getHeight();
        } else if (mWeightBuffer && mWeightBuffer->getWidth() && mWeightBuffer->getHeight()) {
            mWidth = mWeightBuffer->getWidth();
            mHeight = mWeightBuffer->getHeight();
        } else if (mRenderBufferOdd && mRenderBufferOdd->getWidth() && mRenderBufferOdd->getHeight()) {
            mWidth = mRenderBufferOdd->getWidth();
            mHeight = mRenderBufferOdd->getHeight();
        }

        // In some cases, there is a situation of width = 0 and/or height = 0.
        // But it is not an error under some circumstances. We have to continue to execute code
        // even width = 0 and/or height = 0.

        if (mRunMode == ImageWriteCache::Mode::ENQ) {
            mCache->enq()->enqVLInt(mWidth);
            mCache->enq()->enqVLInt(mHeight);

            if (mWidth != 0 && mHeight != 0) {
                mCache->setupEnqCacheImageMemory(mWidth, mHeight);
            }
        }
    } else { // DEQ
        mWidth = mCache->deq()->deqVLInt();
        mHeight = mCache->deq()->deqVLInt();
    }
}

bool
RenderOutputWriter::singleFileOutput(const size_t fileId) const
//
// Single file data output action
//
{
    if (mCache) mCache->timeStartFile();

    //------------------------------
    // setup filename and verify
    ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem;
    std::string filename = calcFilename(fileId, tmpFileItem);
    bool verifyFilenameAndDataTypeReturnStatus;
    if (!verifyFilenameAndDataType(fileId, filename, tmpFileItem, verifyFilenameAndDataTypeReturnStatus)) {
        return verifyFilenameAndDataTypeReturnStatus;
    }
    if (mCache) mCache->timeRecFile(0); // record File timing into position id = 0

    //------------------------------
    bool oiioImgSetupResult;
    auto io = setupOiioImageSetup(filename, tmpFileItem, oiioImgSetupResult);
    if (!oiioImgSetupResult) {
        return false;
    }
    if (mCache) mCache->timeRecFile(1); // record File timing into position id = 1

    //------------------------------
    // create an OIIO::ImageSpec for each output image in the file
    std::vector<OIIO::ImageSpec> specs;
    setupOiioImageSpecTable(fileId, specs);
    if (mCache) mCache->timeRecFile(2); // record File timing into position id = 2

    //------------------------------
    // open the file
    if (!openFile(filename, tmpFileItem, io, specs)) {
        return false;
    }
    if (mCache) mCache->timeRecFile(3); // record File timing into position id = 3

    //------------------------------
    // create a buffer for each image, fill it, and write it.
    bool fillBufferAndWriteResult = fillBufferAndWrite(fileId, filename, tmpFileItem, io, specs);
    if (mCache) mCache->timeRecFile(4); // record File timing into position id = 4

    //------------------------------
    // close the file
    bool closeFileResult = closeFile(filename, tmpFileItem, io);
    if (mCache) mCache->timeRecFile(5); // record File timing into position id = 5

    return (fillBufferAndWriteResult && closeFileResult);
}

std::string
RenderOutputWriter::calcFilename(const size_t fileId,
                                 ImageWriteCache::ImageWriteCacheTmpFileItemShPtr& tmpFileItem) const
{
    std::string filename;
    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        const File& file = (*mFiles)[fileId];

        std::string finalFilename = file.mName;
        std::string checkpointFilename = file.mCheckpointName;
        std::string checkpointMultiVersionFilename =
            generateCheckpointMultiVersionFilename(file,
                                                   mFileNameParam->mOverwriteCheckpoint,
                                                   mFileNameParam->mFinalMaxSamplesPerPixel,
                                                   mCheckpointTileSampleTotals);

        filename = ((mFileNameParam->mCheckpointOutput) ?
                    ((mFileNameParam->mCheckpointOutputMultiVersion) ?
                     checkpointMultiVersionFilename :
                     checkpointFilename) :
                    finalFilename);
        if (mCache) {
            if (mCache->getTwoStageOutput()) {
                // update filename to tmpFilename
                filename = ImageWriteDriver::get()->genTmpFilename(fileId, filename);

                // Create tmpFileItem
                // tmpFileItem is only valid pointer when two stage output mode is enabled
                tmpFileItem = mCache->setTmpFileItem(fileId,
                                                     filename, // tmp filename
                                                     checkpointFilename,
                                                     checkpointMultiVersionFilename,
                                                     finalFilename);
            }

            if (mCache->hasPostCheckpointScript()) {
                // we store info to ImageWriteCache for postCheckpoint script execution
                if (!filename.empty()) {
                    mCache->setCheckpointFilename(checkpointFilename);
                }
            }

            mCache->setCheckpointOverwrite(mFileNameParam->mOverwriteCheckpoint);
        }

        if (mRunMode == ImageWriteCache::Mode::ENQ) {
            if (mCache->getTwoStageOutput()) {
                // We store filename (=tmpFilename) into cache when we are two stage output mode.
                mCache->enq()->enqString(filename); // enq output filename into cache.
            } else {
                // We store both of the filename into cache anyway and the final output filename will
                // be decided at runtime depending on che condition of checkpointOuptutMultiVersion flag.
                mCache->enq()->enqString(checkpointFilename);
                mCache->enq()->enqString(checkpointMultiVersionFilename);
            }
        }
    } else { // DEQ
        if (mCache->getTwoStageOutput()) {
            filename = mCache->deq()->deqString();
            // tmpFileItem is only valid pointer when two stage output mode is enabled
            tmpFileItem = mCache->getTmpFileItem(fileId);
        } else {
            std::string checkpointFilename = mCache->deq()->deqString();
            std::string checkpointMultiVersionFilename = mCache->deq()->deqString();
            filename = ((mFileNameParam->mCheckpointOutputMultiVersion) ?
                        checkpointMultiVersionFilename :
                        checkpointFilename);
        }
    }

    return filename;
}

bool    
RenderOutputWriter::verifyFilenameAndDataType(const size_t fileId,
                                              const std::string& filename,
                                              ImageWriteCache::ImageWriteCacheTmpFileItemShPtr& tmpFileItem,
                                              bool& returnStatus) const
{
    returnStatus = true;
    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        if (filename.empty()) {
            // don't write files that have empty filenames
            if (!mFileNameParam->mCheckpointOutput) {
                const File& file = (*mFiles)[fileId];
                for (const auto& img: file.mImages) {
                    for (const auto& entry: img.mEntries) {
                        // there is an error message for each render output
                        const std::string& name = entry.mRenderOutput->getName();
                        std::ostringstream ostr;
                        ostr << "File output disabled for RenderOutput(\'" << name << "\'), file name empty.";
                        mErrors.push_back(ostr.str());
                    }
                }
            }
            if (mRunMode == ImageWriteCache::Mode::ENQ) mCache->enq()->enqBool(false);
            returnStatus = false;
            return false;
        }
        if (mRunMode == ImageWriteCache::Mode::ENQ) mCache->enq()->enqBool(true);
    } else { // DEQ
        if (!mCache->deq()->deqBool()) {
            mErrors.push_back("File output disable : file name empty.");
            returnStatus = false;
            return false;
        }
    }

    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        // Make sure all of this file's outputs are "flat"
        // You cannot mix "flat" and non-flat outputs in the same file!  Flat
        // images are output using OIIO and non-flat (e.g. deep images) may be
        // output using a different library, e.g. OpenDCX.
        bool hasFlat = false;
        bool hasNonFlat = false;
        const File& file = (*mFiles)[fileId];
        for (const auto& img: file.mImages) {
            for (const auto& entry: img.mEntries) {
                const scene_rdl2::rdl2::RenderOutput* ro = entry.mRenderOutput;
                if (ro->getOutputType() == std::string("flat")) {
                    hasFlat = true;
                } else {
                    hasNonFlat = true;
                }
            }
        }
        if (!hasFlat && hasNonFlat) {
            // non-flat images are output elsewhere
            // need to skip over non-flat outputs
            if (mRunMode == ImageWriteCache::Mode::ENQ) mCache->enq()->enqBool(false);
            return false; // this is not a error
        }
        if (hasFlat && hasNonFlat) {
            mErrors.push_back(errMsg("Output file has a mixture of flat and non-float images", filename,
                                     tmpFileItem));
            if (mRunMode == ImageWriteCache::Mode::ENQ) mCache->enq()->enqBool(false);
            returnStatus = false;
            return false;
        }
        if (mRunMode == ImageWriteCache::Mode::ENQ) mCache->enq()->enqBool(true);
    } else { // DEQ
        if (!mCache->deq()->deqBool()) {
            mErrors.push_back("Output file has a mixture of flat and non-float images");
            returnStatus = false;
            return false;
        }
    }

    return true;
}

OIIO::ImageOutput::unique_ptr
RenderOutputWriter::setupOiioImageSetup(const std::string& filename,
                                        ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                                        bool& result) const
{
    OIIO::ImageOutput::unique_ptr imgOutput = nullptr;

    result = true;
    if (mRunMode != ImageWriteCache::Mode::ENQ) { // STD/DEQ
        imgOutput = OIIO::ImageOutput::create(filename.c_str());
        if (mSha1Gen) {
            mSha1Gen->updateStr("io_construct");
        }
        if (!imgOutput) {
            mErrors.push_back(errMsg("Failed to create OIIO::ImageOutput for ", filename, tmpFileItem));
            result = false;
        }
    } else { // ENQ
        // Skip OIIO::ImageOutput construction for ENQ runMode
    }

    return imgOutput;
}

void
RenderOutputWriter::setupOiioImageSpecTable(const size_t fileId,
                                            std::vector<OIIO::ImageSpec>& specs) const
{
    scene_rdl2::math::HalfOpenViewport aperture;
    scene_rdl2::math::HalfOpenViewport region;
    if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
        const auto& file = (*mFiles)[fileId];
        const scene_rdl2::rdl2::SceneVariables& vars =
            file.mImages[0].mEntries[0].mRenderOutput->getSceneClass().getSceneContext()->getSceneVariables();
        aperture = vars.getRezedApertureWindow();
        region = vars.getRezedRegionWindow();
    }

    const ImageWriteCacheBufferSpecFile& buffSpecFile = mCurrBufferSpec->getBufferSpecFile(fileId);
    for (size_t imgId = 0; imgId < buffSpecFile.getSubImgTotal(); ++imgId) {
        const ImageWriteCacheBufferSpecSubImage& buffSpecSubImg = buffSpecFile.getBufferSpecSubImage(imgId);

        ImageWriteCacheImageSpec localImgSpec;
        ImageWriteCacheImageSpec* imgSpec = nullptr;
        if (mRunMode == ImageWriteCache::Mode::ENQ) {
            imgSpec = mCache->newImgSpec(); // create new imgSpec

#           ifdef RUNTIME_VERIFY_IMGSPECID
            bool verifyResult = (mCache->getImgSpecTotal() - 1) == buffSpecSubImg.getImgSpecId();
            std::ostringstream ostr;
            ostr << "RUNTIME-VERIFY-"
                 << (verifyResult ? "OK" : "FAILED")
                 << " : RenderOutputWriter.cc setupOiioImageSpecTable() imgSpecId {\n"
                 << "  mCache->getImgSpecTotal()-1:" << (mCache->getImgSpecTotal() - 1)
                 << (verifyResult ? " ==" : " !=")
                 << " buffSpecSubImg.getImgSpecId():" << buffSpecSubImg.getImgSpecId() << '\n'
                 << "}";
            std::cerr << ostr.str() << '\n';
#           endif // end RUNTIME_VERIFY_IMGSPECID

        } else if (mRunMode == ImageWriteCache::Mode::DEQ) {
            imgSpec = mCache->getImgSpec(buffSpecSubImg.getImgSpecId()); // pick up saved imgSpec
        } else { // STD
            imgSpec = &localImgSpec; // pick local imgSpec
        }

        if (mRunMode != ImageWriteCache::Mode::DEQ) { // STD/ENQ
            setupImageSpecStdEnq(imgSpec, (*mFiles)[fileId].mImages[imgId], aperture, region);
        } // STD/ENQ

        if (mRunMode != ImageWriteCache::Mode::ENQ) { // STD/DEQ
            addOiioImageSpec(imgSpec, specs);
        }

#       ifdef PRECISE_HASH_COMPARE
        if (mRunMode == ImageWriteCache::Mode::DEQ && mSha1Gen) {
            imgSpec->updateHash(*mSha1Gen);
        }
#       endif // end PRECISE_HASH_COMPARE
    } // loop imgId

#   ifdef PRECISE_HASH_COMPARE
    if (mRunMode == ImageWriteCache::Mode::STD && mSha1Gen) {
        updateHashImageSpecOrg((*mFiles)[fileId], aperture, region);
    }
#   endif // end PRECISE_HASH_COMPARE
}

bool
RenderOutputWriter::openFile(const std::string& filename,
                             const ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                             OIIO::ImageOutput::unique_ptr& io,
                             const std::vector<OIIO::ImageSpec>& specs) const
{
    if (mRunMode == ImageWriteCache::Mode::ENQ) {
        return true;
    }

    //
    // STD/DEQ
    //
    if (mSha1Gen) {
        mSha1Gen->updateStr("io->open()");
    }
            
    if (!mSha1Gen || mRunMode == ImageWriteCache::Mode::DEQ) {
        if (!io->open(filename.c_str(), specs.size(), &specs[0])) {
            std::ostringstream ostr;
            ostr << "Failed to open '" << filename << "' for writing." << " oiioError:(" << io->geterror() << ")";
            mErrors.push_back(ostr.str());
            return false;
        }

        if (tmpFileItem) {
            // Two stage output mode. try to open 2ndary fd for final copy and unlink in order to
            // clean up the tmp file.
            if (!tmpFileItem->openTmpFileAndUnlink()) {
                std::ostringstream ostr;
                ostr << "Failed to open two stage output tmpFile '" << filename
                     << "' (finally copy to '" << tmpFileItem->getDestinationFilename() << "').";
                mErrors.push_back(ostr.str());
                io->close(); // close oiio file first.
                return false;
            }
        }
    }
    return true;
}

bool
RenderOutputWriter::fillBufferAndWrite(const size_t fileId,
                                       const std::string& filename,
                                       const ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                                       OIIO::ImageOutput::unique_ptr& io,
                                       const std::vector<OIIO::ImageSpec>& specs) const
{
    bool result = true;

    const ImageWriteCacheBufferSpecFile& bufferSpecFile = mCurrBufferSpec->getBufferSpecFile(fileId);
    for (size_t subImgId = 0; subImgId < bufferSpecFile.getSubImgTotal(); ++subImgId) {
        const ImageWriteCacheBufferSpecSubImage& bufferSpecSubImage =
            bufferSpecFile.getBufferSpecSubImage(subImgId);
        const OIIO::ImageSpec* spec = (mRunMode != ImageWriteCache::Mode::ENQ) ? &specs[subImgId] : nullptr;

        if (mCache) mCache->timeStartImage();

        // if not the first image, need to re-open to process the next sub-image
        if (subImgId > 0) {
            if (mRunMode != ImageWriteCache::Mode::ENQ) { // STD/DEQ
                if (mSha1Gen) {
                    mSha1Gen->updateStr("io->open()");
                }

                if (!mSha1Gen || mRunMode == ImageWriteCache::Mode::DEQ) {
                    if (!io->open(filename.c_str(), *spec, OIIO::ImageOutput::AppendSubimage)) {
                        std::ostringstream ostr;
                        ostr << "Failed to reopen file for writing part-name:"
                             << bufferSpecSubImage.getName();
                        mErrors.push_back(errMsg(ostr.str(), filename, tmpFileItem));
                        result = false;
                        if (mCache) {
                            mCache->timeRecImage(0); // record Image timing into position id = 0
                            mCache->timeRecImage(1); // record Image timing into position id = 1
                            mCache->timeRecImage(2); // record Image timing into position id = 2
                            mCache->timeRecImage(3); // record Image timing into position id = 3
                        }
                        continue; // next! but I don't have a good feeling about the next sub-images either.
                    }
                }
            }
        }
        if (mCache) mCache->timeRecImage(0); // record Image timing into position id = 0

        if (!subImageFillBufferAndWrite(fileId, subImgId, io, spec)) {
            if (tmpFileItem) { // two stage output mode
                tmpFileItem->closeTmpFile(); // clean up for two stage output
            }
            std::ostringstream ostr;
            ostr << "Failed to write buffer. part-name:"
                 << bufferSpecSubImage.getName();
            mErrors.push_back(errMsg(ostr.str(), filename, tmpFileItem));
            result = false;
        }
        if (mCache) mCache->timeRecImage(3); // record Image timing into position id = 3
    } // loop file.mImages

    return result;
}

bool
RenderOutputWriter::closeFile(const std::string& filename,
                              const ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                              OIIO::ImageOutput::unique_ptr& io) const
{
    bool result = true;

    if (mRunMode == ImageWriteCache::Mode::ENQ) {
        return result;
    }

    //
    // STD/DEQ
    //
    if (mSha1Gen) {
        mSha1Gen->updateStr("io->close()");
    }

    if (!mSha1Gen || mRunMode == ImageWriteCache::Mode::DEQ) {
        if (!io->close()) {
            if (tmpFileItem) { // two stage output mode
                tmpFileItem->closeTmpFile(); // clean up for two stage output
            }
            mErrors.push_back(errMsg("Failed to close file", filename, tmpFileItem));
            result = false;
        } else {
            if (tmpFileItem) {
#               ifdef IMAGE_WRITE_DETAIL_MESSAGE
                mInfos.push_back(scene_rdl2::util::buildString("Wrote: tmpFile for ",
                                                               tmpFileItem->getDestinationFilename()));
#               endif // end IMAGE_WRITE_DETAIL_MESSAGE
            } else {
                mInfos.push_back(scene_rdl2::util::buildString("Wrote: ", filename));
            }
        }
    }

    return result;
}

//------------------------------------------------------------------------------------------

void
RenderOutputWriter::setupImageSpecStdEnq(ImageWriteCacheImageSpec* imgSpec,
                                         const Image& img,
                                         const scene_rdl2::math::HalfOpenViewport& aperture,
                                         const scene_rdl2::math::HalfOpenViewport& region) const
{
    int numChans = 0;
    for (const auto &entry: img.mEntries) {
        const scene_rdl2::rdl2::RenderOutput *ro = entry.mRenderOutput;

        // number of channels
        int nc = entry.mChannelNames.size();

        // channel format
        imgSpec->pushBackChanFormatN(nc, ro->getChannelFormat());

        // channel names
        imgSpec->pushBackChanNames(entry.mChannelNames);

        numChans += nc;
    }
    imgSpec->setTotalNumChans(numChans);

    if (!img.mName.empty()) {
        imgSpec->setName(img.mName);
    }
                
    imgSpec->setSizeInfo(mWidth,
                         mHeight,
                         region.min().x,
                         aperture.max().y - region.max().y, // flip y relative to display window
                         aperture.min().x,
                         aperture.min().y,                                  
                         aperture.width(),
                         aperture.height());

    imgSpec->setCompression(getCompressionStr(img.mEntries[0].mRenderOutput));
    if (imgSpec->isCompressionDwa()) {
        imgSpec->setDwaLevel(std::max(0.0f, img.mEntries[0].mRenderOutput->getCompressionLevel()));
    }

    // add metadata to image spec
    // get first valid metadata we find.
    for (const auto& entry: img.mEntries) {
        const scene_rdl2::rdl2::SceneObject* metadata = entry.mRenderOutput->getExrHeaderAttributes();
        if (metadata) {
            const scene_rdl2::rdl2::Metadata* currMetadata = metadata->asA<scene_rdl2::rdl2::Metadata>();
            imgSpec->setExrHeaderInfo(currMetadata->getAttributeNames(),
                                      currMetadata->getAttributeTypes(),
                                      currMetadata->getAttributeValues(),
                                      currMetadata->getName());
            break;
        }
    }

    // add checkpoint resume metadata if call back returns data
    std::vector<std::string> metadata = (*mCallBackCheckpointResumeMetadata)(mCheckpointTileSampleTotals);
    if (!metadata.empty()) {
        imgSpec->resumeAttr() = metadata;
    }
}

// static function
void
RenderOutputWriter::addOiioImageSpec(const ImageWriteCacheImageSpec* imgSpec,
                                     std::vector<OIIO::ImageSpec>& specs)
{
    imgSpec->
        setupOIIOSpecs
        ([&](const std::string &name,
             int width,
             int height,
             int spec_x,
             int spec_y,
             int spec_full_x,
             int spec_full_y,
             int spec_full_width,
             int spec_full_height,
             int totalNumChans,
             const std::string &compression,
             float level,
             const std::vector<ImageWriteCacheImageSpec::ChannelFormat>& chanFormat,
             const std::vector<std::string>& chanNames,
             const std::vector<std::string>& attrNames,
             const std::vector<std::string>& attrTypes,
             const std::vector<std::string>& attrValues,
             const std::string& metaDataName,
             const std::vector<std::string>& resumeAttr)
         {
             auto calcOIIOChanFormat =
                 [](const ImageWriteCacheImageSpec::ChannelFormat& chanFormat) -> OIIO::TypeDesc {
                 if (chanFormat == scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF) {
                     return OIIO::TypeDesc::HALF;
                 } else {
                     return OIIO::TypeDesc::FLOAT;
                 }
             };

             // create the spec for this image
             specs.emplace_back(width, height, totalNumChans, OIIO::TypeDesc::FLOAT);

             specs.back().x = spec_x;
             specs.back().y = spec_y;
             specs.back().full_x = spec_full_x;
             specs.back().full_y = spec_full_y;
             specs.back().full_width = spec_full_width;
             specs.back().full_height = spec_full_height;

             std::vector<OIIO::TypeDesc> chanFormats;
             for (auto itr: chanFormat) {
                 chanFormats.push_back(calcOIIOChanFormat(itr));
             }
             specs.back().channelformats = chanFormats;
             specs.back().channelnames = chanNames;

             if (!name.empty()) {
                 specs.back().attribute("name", name);
             }

             specs.back().attribute("oiio::ColorSpace", "Linear");

             specs.back().attribute("compression", compression);
             // dwaa and dwab compression can also have compression levels
             if (imgSpec->isCompressionDwa()) {
                 // compression level must be >= 0.
                 specs.back().attribute("openexr:dwaCompressionLevel", level);
             }

             if (!metaDataName.empty()) {
                 // add metadata
                 writeExrHeader(specs.back(), attrNames, attrTypes, attrValues, metaDataName);
             }

             if (!resumeAttr.empty()) {
                 // add checkpoint resume metadata
                 for (size_t i = 0; i < resumeAttr.size(); i += 3) {
                     writeExrHeader(specs.back(), resumeAttr[i], resumeAttr[i+1], resumeAttr[i+2],
                                    "internal_resumable_output_file_write_logic"); // metadataName for error log
                 }
             }
         });
}

void
RenderOutputWriter::updateHashImageSpecOrg(const File& file,
                                           const scene_rdl2::math::HalfOpenViewport& aperture,
                                           const scene_rdl2::math::HalfOpenViewport& region) const
{
    int specs_x = region.min().x;
    int specs_y = aperture.max().y - region.max().y; // flip y coordinate relative to display window
    int specs_full_x = aperture.min().x;
    int specs_full_y = aperture.min().y;
    int specs_full_width = aperture.width();
    int specs_full_height = aperture.height();

    for (const auto &img: file.mImages) {
        int numChans = 0;
        std::vector<OIIO::TypeDesc> chanFormats;
        std::vector<std::string> chanNames;
        for (const auto &entry: img.mEntries) {
            const scene_rdl2::rdl2::RenderOutput* ro = entry.mRenderOutput;

            // number of channels and channel names
            int nc = entry.mChannelNames.size();
            chanNames.insert(chanNames.end(), entry.mChannelNames.begin(), entry.mChannelNames.end());

            // channel format
            OIIO::TypeDesc format = getChannelFormat(ro);
            chanFormats.insert(chanFormats.end(), nc, format);

            numChans += nc;
        }

        mSha1Gen->updateInt2(mWidth, mHeight);
        mSha1Gen->update<int>(numChans);
        mSha1Gen->updateInt2(specs_x, specs_y);
        mSha1Gen->updateInt2(specs_full_x, specs_full_y);
        mSha1Gen->updateInt2(specs_full_width, specs_full_height);

        for (size_t id = 0; id < chanFormats.size(); ++id) {
            using ChannelFormat = scene_rdl2::rdl2::RenderOutput::ChannelFormat;
            ChannelFormat chanFormat = ChannelFormat::CHANNEL_FORMAT_FLOAT;
            if (chanFormats[id] == OIIO::TypeDesc::HALF) {
                chanFormat = ChannelFormat::CHANNEL_FORMAT_HALF;
            }
            mSha1Gen->update<ChannelFormat>(chanFormat);
        }

        mSha1Gen->updateStrVec(chanNames);
        if (!img.mName.empty()) {
            mSha1Gen->updateStr(img.mName);
        }
        std::string compression = getCompressionStr(img.mEntries[0].mRenderOutput);
        mSha1Gen->updateStr(compression);
        if (compression == "dwaa" || compression == "dwab") {
            float level = std::max(0.0f, img.mEntries[0].mRenderOutput->getCompressionLevel());
            mSha1Gen->update<float>(level);
        }

        // add metadata to image spec
        // get first valid metadata we find.
        for (const auto &e: img.mEntries) {
            const scene_rdl2::rdl2::SceneObject *metadata = e.mRenderOutput->getExrHeaderAttributes();
            if (metadata) {
                const scene_rdl2::rdl2::Metadata *currMetaData = metadata->asA<scene_rdl2::rdl2::Metadata>();
                mSha1Gen->updateStrVec(currMetaData->getAttributeNames());
                mSha1Gen->updateStrVec(currMetaData->getAttributeTypes());
                mSha1Gen->updateStrVec(currMetaData->getAttributeValues());
                mSha1Gen->updateStr(currMetaData->getName());
                break;
            }
        }

        // add checkpoint resume metadata if call back returns data
        std::vector<std::string> metadata = (*mCallBackCheckpointResumeMetadata)(mCheckpointTileSampleTotals);
        if (!metadata.empty()) {
            for (size_t i = 0; i < metadata.size(); i += 3) {
                mSha1Gen->updateStr3(metadata[i], metadata[i+1], metadata[i+2]);
            }
        }
    } // loop file.mImages
}

// static function
OIIO::TypeDesc
RenderOutputWriter::getChannelFormat(const scene_rdl2::rdl2::RenderOutput* ro)
{
    if (ro->getChannelFormat() == scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF) {
        return OIIO::TypeDesc::HALF;
    }
    return OIIO::TypeDesc::FLOAT; 
}

// static function
std::string
RenderOutputWriter::getCompressionStr(const scene_rdl2::rdl2::RenderOutput* ro)
{
    switch (ro->getCompression()) {
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_NONE:  return "none";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_RLE:   return "rle";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_ZIPS:  return "zips";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_PIZ:   return "piz";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_PXR24: return "pxr24";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_B44:   return "b44";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_B44A:  return "b44a";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_DWAA:  return "dwaa";
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_DWAB:  return "dwab";
    default:
    case scene_rdl2::rdl2::RenderOutput::COMPRESSION_ZIP:   return "zip";
    }
}

//------------------------------------------------------------------------------------------

bool
progressCallBack(void* data, float fraction)
{
    if (data) {
        (static_cast<ImageWriteCache *>(data))->timeUpdateBuffWrite(fraction);
    }
    return false;
}

bool
RenderOutputWriter::subImageFillBufferAndWrite(const size_t fileId,
                                               const size_t subImgId,
                                               OIIO::ImageOutput::unique_ptr& io,
                                               const OIIO::ImageSpec* spec) const
{
    bool result = true;
    if (mRunMode != ImageWriteCache::Mode::ENQ) { // STD/DEQ
        OIIO::ImageBuf buffer(*spec);
        fillBuffer(fileId, subImgId, &buffer);
        if (mCache) mCache->timeRecImage(1); // record Image timing into position id = 1

        if (mSha1Gen) {
            mSha1Gen->updateStr("buffer.write()");
        }
        if (!mSha1Gen || mRunMode == ImageWriteCache::Mode::DEQ) {
            if (mCache) mCache->timeStartBuffWrite();
            result = buffer.write(io.get(), &progressCallBack, static_cast<void *>(mCache));
            if (mCache) mCache->timeEndBuffWrite();
        }
        if (mCache) mCache->timeRecImage(2); // record Image timing into position id = 2

    } else { // ENQ
        result = fillBuffer(fileId, subImgId, nullptr);
        if (mCache) {
            mCache->timeRecImage(1); // record Image timing into position id = 1
            mCache->timeRecImage(2); // record Image timing into position id = 2
        }
    }

    return result;
}

bool
RenderOutputWriter::fillBuffer(const size_t fileId,
                               const size_t subImgId,
                               OIIO::ImageBuf* buffer) const
{
    auto getBuffSpecSubImage = [&]() -> const ImageWriteCacheBufferSpecSubImage & {
        const ImageWriteCacheBufferSpecFile &buffSpecFile = mCurrBufferSpec->getBufferSpecFile(fileId);
        return buffSpecFile.getBufferSpecSubImage(subImgId);
    };
    auto getAovBuff = [&]() -> const VariablePixelBuffer * {
        return &(*mAovBuffers)[getBuffSpecSubImage().getAovBuffOffset()];
    };
    auto getDisplayFilterBuff = [&]() -> const VariablePixelBuffer * {
        return &(*mDisplayFilterBuffers)[getBuffSpecSubImage().getDisplayFilterBuffOffset()];
    };

    unsigned numchannels = getBuffSpecSubImage().getTotalNumChannels();
    size_t pixCacheSizeFull = getBuffSpecSubImage().getPixCacheFullSize();
    size_t pixCacheSizeHalf = getBuffSpecSubImage().getPixCacheHalfSize();
    if (mRunMode == ImageWriteCache::Mode::ENQ) {
        //
        // Convert imageWriteCache internal memory (dataFull/dataHalf) is done by multi threads under
        // ENQ mode.
        // Tested several taskSize but 1 was best. I could not find any reason to use 2 or more at
        // this moment.
        //
        constexpr size_t taskSize = 1;
        tbb::blocked_range<size_t> range(0, mCache->getYBlockTotal(), taskSize);
        bool fillPixBufferEnqErrorCondition = false;
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                PageAlignedBuff convertBuff;
                if (pixCacheSizeHalf) {
                    // construct internal buffer for half float conversion
                    size_t allocSize = pixCacheSizeHalf * 2;
                    allocSize = (allocSize + 3) & ~3; // aligned count 4 boundary.
                    convertBuff.resize(allocSize, 0x0);
                }
                size_t addByteFull = pixCacheSizeFull * mWidth * mCache->getYBlockSize();
                size_t addByteHalf = pixCacheSizeHalf * mWidth * mCache->getYBlockSize();

                for (int yBlockId = r.begin(); yBlockId < (int)r.end(); ++yBlockId) {
                    mCache->expandInternalDataIfNeeded(yBlockId, addByteFull, addByteHalf);

                    int yMin = yBlockId * mCache->getYBlockSize();
                    int yMax = yMin + mCache->getYBlockSize();
                    if (yMax > mHeight) yMax = mHeight;
                    for (int y = yMin; y < yMax; ++y) {
                        for (int x = 0; x < mWidth; ++x) {
                            void *dataFull =
                                (void *)mCache->enqFullBuff(yBlockId)->enqReserveMem(pixCacheSizeFull);
                            void *dataHalf =
                                (void *)mCache->enqHalfBuff(yBlockId)->enqReserveMem(pixCacheSizeHalf);
                            if (!fillPixBufferEnq(fileId, subImgId, x, y,
                                                  getAovBuff(), getDisplayFilterBuff(),
                                                  dataFull, pixCacheSizeFull,
                                                  dataHalf, pixCacheSizeHalf,
                                                  convertBuff)) {
                                fillPixBufferEnqErrorCondition = true;
                            }
                        }
                    }
                }
            });

        if (fillPixBufferEnqErrorCondition) {
            std::vector<std::string> &errors = mCache->getErrors();
            errors.push_back("fillPixBufferEnq detected problem of memory over-run. "
                             "Stopped data construction and skip some pixels to write.");
            return false;
        }

    } else { // STD/DEQ
        // dataStd is a float buffer which includes final pixel values to pass into the openimageio API.
        // STD mode simply fills dataStd from film and passes dataStd into openimageio API.
        // DEQ mode constructs dataStd from imageWriteCache's dataFull/dataHalf and passes dataStd into
        // openimageio API.
        std::vector<float> dataStd;
        dataStd.resize(numchannels);

        if (mRunMode == ImageWriteCache::Mode::DEQ) {
            // We are going to initialize the DEQ buffer setup based on each sub-Image independently.
            // This is important for error handling. Each sub-Image output does not have any dependency
            // on the previous sub-image output result. This behavior is pretty important If the previous
            // sub-Image (or file) failed to write.
            mCache->setupDeqBuff(fileId, subImgId);
        }

        //
        // buffer setpixel operation is done by single thread
        //
        for (int y = 0; y < mHeight; ++y) {
            for (int x = 0; x < mWidth; ++x) {
                if (mRunMode == ImageWriteCache::Mode::STD) { // STD
                    fillPixBufferStd(fileId, subImgId, x, y,
                                     getAovBuff(), getDisplayFilterBuff(), &dataStd[0]);
                } else { // DEQ
                    int yBlockId = mCache->calcYBlockId(y);
                    const void *dataFull =
                        (const void *)mCache->deqFullBuff(yBlockId)->skipByteData(pixCacheSizeFull);
                    const void *dataHalf =
                        (const void *)mCache->deqHalfBuff(yBlockId)->skipByteData(pixCacheSizeHalf);
                    fillPixBufferDeq(fileId, subImgId, dataFull, dataHalf, &dataStd[0]);
                }
                if (!mSha1Gen || mRunMode == ImageWriteCache::Mode::DEQ) {
                    buffer->setpixel(buffer->xbegin() + x, buffer->yend() - y - 1, &dataStd[0]);
                }
            }
        }
    }

    return true;
}

void    
RenderOutputWriter::fillPixBufferStd(const size_t fileId,
                                     const size_t subImgId,
                                     const int x,
                                     const int y,
                                     const VariablePixelBuffer* aovBuffers,
                                     const VariablePixelBuffer* displayFilterBuffers,
                                     void* outData) const
{
    const std::vector<Entry>& entries = (*mFiles)[fileId].mImages[subImgId].mEntries;

    size_t outDataOffset = 0;
    const scene_rdl2::fb_util::VariablePixelBuffer* aov = aovBuffers;
    const scene_rdl2::fb_util::VariablePixelBuffer* displayFilterBuffer = displayFilterBuffers;
    for (const auto& e: entries) {
        int numChan = e.mChannelNames.size();
        float* fPtr = (float *)((uintptr_t)outData + outDataOffset);
        outDataOffset += numChan * sizeof(float);

        fillPixBufferSingleEntry(e, x, y, aov, displayFilterBuffer, fPtr);

        if (mSha1Gen) {
            bool halfFloatMode =
                (e.mRenderOutput->getChannelFormat() == scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF);
            for (int i = 0; i < numChan; ++i) {
                if (halfFloatMode) {
                    mSha1Gen->update<unsigned short>(precisionAdjustH(ftoh(fPtr[i])));
                } else {
                    mSha1Gen->update<float>(precisionAdjustF(fPtr[i]));
                }
            }
        }
    } // for (const auto &e: entries)
}

bool
RenderOutputWriter::fillPixBufferEnq(const size_t fileId,
                                     const size_t subImgId,
                                     const int x,
                                     const int y,
                                     const VariablePixelBuffer* aovBuffers,
                                     const VariablePixelBuffer* displayFilterBuffers,
                                     void* outDataFull,
                                     size_t outDataFullSize, // pixCacheSizeByte for full float
                                     void* outDataHalf,
                                     size_t outDataHalfSize, // pixCacheSizeByte for half float
                                     PageAlignedBuff& convertBuff) const
//
// outDataFull and outDataHalf should have enough memory to store data.
// This function returns false immediately without completion of data setup when buffer size is not
// enough.
//
{
    MNRY_ASSERT(outDataHalfSize % sizeof(unsigned short) == 0);
    int halfFloatTotal = outDataHalfSize / sizeof(unsigned short);

    const std::vector<Entry>& entries = (*mFiles)[fileId].mImages[subImgId].mEntries;

    float* convertBuffPtr = nullptr;
    size_t convertBuffSize = 0; // byte
    if (halfFloatTotal) {
        convertBuffSize = halfFloatTotal * sizeof(float); // byte
        convertBuffPtr = (float *)&convertBuff[0]; // original float buffer for convert to half float
    }

    size_t outDataFullOffset = 0; // byte
    size_t convertBuffOffset = 0; // byte

    const scene_rdl2::fb_util::VariablePixelBuffer* aov = aovBuffers;
    const scene_rdl2::fb_util::VariablePixelBuffer* displayFilterBuffer = displayFilterBuffers;
    for (const auto &e: entries) {
        int numChan = e.mChannelNames.size();
        float* fPtr = nullptr;

        bool halfFloatMode =
            (e.mRenderOutput->getChannelFormat() == scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF);
        if (halfFloatMode) {
            fPtr = (float *)((uintptr_t)convertBuffPtr + convertBuffOffset);
            convertBuffOffset += numChan * sizeof(float);
            if (convertBuffOffset > convertBuffSize) {
                // original float buffer for half float conversion exceeded full.
                return false; // Return in order to avoid memory overrun.
            }
        } else {
            fPtr = (float *)((uintptr_t)outDataFull + outDataFullOffset);
            outDataFullOffset += numChan * sizeof(float);
            if (outDataFullOffset > outDataFullSize) {
                // output full data buffer exceeded full.
                return false; // Return in order to avoid overrun.
            }
        }

        fillPixBufferSingleEntry(e, x, y, aov, displayFilterBuffer, fPtr);
    } // for (const auto &e: entries)

    if (halfFloatTotal) {
        fVecToHVec(convertBuff, (unsigned short *)((uintptr_t)outDataHalf));
    }

    return true;
}

void
RenderOutputWriter::fillPixBufferSingleEntry(const Entry& e,
                                             const int x,
                                             const int y,
                                             const VariablePixelBuffer *&aov,
                                             const VariablePixelBuffer *&displayFilterBuffer,
                                             float* outPtr) const
//
// outPtr should have proper memory size already before call this function
//
{
    const scene_rdl2::rdl2::RenderOutput* ro = e.mRenderOutput;

    switch(ro->getResult()) {
    case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP:
        if (mHeatMap) {
            // time per pixel stat
            const int64_t* p = mHeatMap->getData();
            p = p + y * mWidth + x;
            outPtr[0] = mcrt_common::Clock::seconds(*p);
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT:
        if (mWeightBuffer) {
            const float* p = mWeightBuffer->getData();
            p = p + y * mWidth + x;
            outPtr[0] = *p;
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX:
        if (mRenderBufferOdd) {
            const scene_rdl2::fb_util::RenderColor* p = mRenderBufferOdd->getData();
            p = p + y * mWidth + x;
            outPtr[0] = p->x;
            outPtr[1] = p->y;
            outPtr[2] = p->z;
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX:
        if (mRenderBufferOdd) {
            const scene_rdl2::fb_util::RenderColor* p = mRenderBufferOdd->getData();
            p = p + y * mWidth + x;
            outPtr[0] = p->w;
        }
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY:
    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA:
    case scene_rdl2::rdl2::RenderOutput::RESULT_STATE_VARIABLE:
    case scene_rdl2::rdl2::RenderOutput::RESULT_DEPTH:
    case scene_rdl2::rdl2::RenderOutput::RESULT_PRIMITIVE_ATTRIBUTE:
    case scene_rdl2::rdl2::RenderOutput::RESULT_MATERIAL_AOV:
    case scene_rdl2::rdl2::RenderOutput::RESULT_WIREFRAME:
    case scene_rdl2::rdl2::RenderOutput::RESULT_LIGHT_AOV:
    case scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV:
        {
            MNRY_ASSERT(e.mAovSchemaId != pbr::AOV_SCHEMA_ID_UNKNOWN);
            switch (aov->getFormat()) {
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
                {
                    MNRY_ASSERT(e.mChannelNames.size() == 1);
                    const scene_rdl2::fb_util::FloatBuffer& fbuf = aov->getFloatBuffer();
                    const float* f = fbuf.getData();
                    f += y * mWidth + x;
                    outPtr[0] = *f;
                }
                break;
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
                {
                    MNRY_ASSERT(e.mChannelNames.size() == 2);
                    const scene_rdl2::fb_util::Float2Buffer& f2buf = aov->getFloat2Buffer();
                    const scene_rdl2::math::Vec2f* v2f = f2buf.getData();
                    v2f += y * mWidth + x;
                    outPtr[0] = v2f->x;
                    outPtr[1] = v2f->y;
                }
                break;
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
                {
                    MNRY_ASSERT(e.mChannelNames.size() == 3);
                    const scene_rdl2::fb_util::Float3Buffer& f3buf = aov->getFloat3Buffer();
                    const scene_rdl2::math::Vec3f* v3f = f3buf.getData();
                    v3f += y * mWidth + x;
                    outPtr[0] = v3f->x;
                    outPtr[1] = v3f->y;
                    outPtr[2] = v3f->z;
                }
                break;
            case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4:
                {
                    MNRY_ASSERT(e.mChannelNames.size() == 4);
                    const scene_rdl2::fb_util::Float4Buffer& f4buf = aov->getFloat4Buffer();
                    const scene_rdl2::math::Vec4f* v4f = f4buf.getData();
                    v4f += y * mWidth + x;
                    outPtr[0] = v4f->x;
                    outPtr[1] = v4f->y;
                    outPtr[2] = v4f->z;
                    outPtr[3] = v4f->w;
                }
                break;

            default:
                MNRY_ASSERT(0 && "unexpected variable pixel buffer format");
            } // switch (aov->getFormat())
            ++aov;                  // increment aov pointer
        }
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE:
        {
            if (mCryptomatteBuffer) {
                int numLayers = ro->getCryptomatteNumLayers();
                mCryptomatteBuffer->outputFragments(x, y, numLayers, outPtr, *ro);
            }
        }
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER:
        {
            // Read data from the DisplayFilter Buffer
            const scene_rdl2::fb_util::Float3Buffer& f3buf = displayFilterBuffer->getFloat3Buffer();
            const scene_rdl2::math::Vec3f& v3f = f3buf.getPixel(x, y);
            outPtr[0] = v3f.x;
            outPtr[1] = v3f.y;
            outPtr[2] = v3f.z;
            ++displayFilterBuffer; // increment displayFilterBuffer pointer
        }
        break;

    default:
        MNRY_ASSERT(0 && "unknown result type");
    } // switch(ro->getResult())
}

void
RenderOutputWriter::fillPixBufferDeq(const size_t fileId,
                                     const size_t subImgId,
                                     const void* cacheDataFull,
                                     const void* cacheDataHalf,
                                     float* dataOut) const
{
    auto getBuffSpecSubImage = [&]() -> const ImageWriteCacheBufferSpecSubImage & {
        const ImageWriteCacheBufferSpecFile &buffSpecFile = mCurrBufferSpec->getBufferSpecFile(fileId);
        return buffSpecFile.getBufferSpecSubImage(subImgId);
    };
    auto getNumEntries = [&]() -> size_t {
        return getBuffSpecSubImage().getPixNumChan().size();
    };
    auto getPixChanFormat = [&](const size_t entryId) -> ImageWriteCacheBufferSpecSubImage::ChanFormat {
        return getBuffSpecSubImage().getPixChanFormat()[entryId];
    };
    auto getPixNumChan = [&](const size_t entryId) -> int {
        return getBuffSpecSubImage().getPixNumChan()[entryId];
    };

    std::vector<float> fVec;

    size_t cacheDataFullOffset = 0;
    size_t cacheDataHalfOffset = 0;
    size_t dataOutOffset = 0;
    for (size_t entryId = 0; entryId < getNumEntries(); ++entryId) {
        int numChan = getPixNumChan(entryId);
        
        switch (getPixChanFormat(entryId)) {
        case ImageWriteCacheBufferSpecSubImage::ChanFormat::HALF :
            {
                const unsigned short* hPtr = (const unsigned short *)((uintptr_t)cacheDataHalf + cacheDataHalfOffset);
                hVecToFVec(hPtr, numChan, fVec);
                size_t dataSize = numChan * sizeof(float);
                std::memcpy((void *)((uintptr_t)dataOut + dataOutOffset), &fVec[0], dataSize);
                dataOutOffset += dataSize;
                cacheDataHalfOffset += numChan * sizeof(unsigned short);

                if (mSha1Gen) {
                    for (int i = 0; i < numChan; ++i) {
                        mSha1Gen->update<unsigned short>(precisionAdjustH(hPtr[i]));
                    }
                }
            }
            break;

        case ImageWriteCacheBufferSpecSubImage::ChanFormat::FULL :
            { 
                const float* fPtr = (const float *)((uintptr_t)cacheDataFull + cacheDataFullOffset);
                size_t dataSize = numChan * sizeof(float);
                std::memcpy((void *)((uintptr_t)dataOut + dataOutOffset), (const void *)(fPtr), dataSize);
                dataOutOffset += dataSize;
                cacheDataFullOffset += dataSize;

                if (mSha1Gen) {
                    for (int i = 0; i < numChan; ++i) {
                        mSha1Gen->update<float>(precisionAdjustF(fPtr[i]));
                    }
                }
            }
            break;

        default :
            break;
        }
    } // loop entries
}

// static function
void
RenderOutputWriter::calcPixCacheSize(const std::vector<Entry>& entries,
                                     size_t& fullBuffSize, size_t& halfBuffSize)
{
    fullBuffSize = 0;
    halfBuffSize = 0;
    for (const auto &e: entries) {
        size_t numChan = e.mChannelNames.size();        
        if (e.mRenderOutput->getChannelFormat() == scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF) {
            halfBuffSize += numChan * sizeof(unsigned short);
        } else {
            fullBuffSize += numChan * sizeof(float);
        }
    }
}

// static function
void
RenderOutputWriter::fVecToHVec(const PageAlignedBuff& fVec, unsigned short* hVec)
{
#if 1
    // fp16c instruction version :
    // You can check your cpu has fp16c instruction by ( lscpu | grep fp16c).
    auto f4toh4 = [](const __m128& in, unsigned short* h4, int count) {
        // This function basically converts 4 floats to 4 half floats however still need to maintain
        // less than 4 floats case. (i.e. it is the case that Input data has 4 float memory but only
        // set less than 4 float).
        __m128i out = _mm_cvtps_ph(in, 0); // An immediate value controlling rounding bits : 0=Nearest
        MNRY_ASSERT(count <= 4);
        std::memcpy(static_cast<void *>(h4),
                    static_cast<const void *>(&out), sizeof(unsigned short) * count);
    };
#else
    // non fp16c instruction version
    auto f4toh4 = [&](const __m128& in, unsigned short* h4, int count) {
        // This is a special version which does not use fp16c instruction (i.e. _mm_cvtps_ph).
        // We are not using a host that does not support fp16c instruction anymore.
        // However, just in case, I would like to keep this code in order to support non-fp16c hosts
        // in some cases.
        MNRY_ASSERT(count <= 4);
        const float* fptr = (const float*)&in;
        for (int i = 0; i < count; ++i) {
            h4[i] = ftoh(fptr[i]);
        }
    };
#endif

    const float* fPtr = (const float*)&fVec[0];

    size_t fCount = fVec.size() / sizeof(float); // total float count

#   ifdef RUNTIME_VERIFY_FTOH
    std::vector<unsigned short> hVerifyTarget;
    for (size_t i = 0; i < fCount; ++i) {
        hVerifyTarget.push_back(ftoh(fPtr[i]));
    }
#   endif // end RUNTIME_VERIFY_FTOH
    
    if (fCount == 1) {
        hVec[0] = ftoh(fPtr[0]);
    } else {
        const __m128* m128Ptr = (const __m128*)fPtr;

        size_t simd4LoopMax = fCount / 4;
        for (size_t i = 0; i < simd4LoopMax; ++i) {
            f4toh4(m128Ptr[i], &hVec[i * 4], 4);
        }

        if (fCount % 4) {
            f4toh4(m128Ptr[simd4LoopMax], &hVec[simd4LoopMax * 4], fCount % 4);
        }
    }

#   ifdef RUNTIME_VERIFY_FTOH
    for (size_t i = 0; i < fCount; ++i) {
        if (hVec[i] != hVerifyTarget[i]) {
            std::cerr << ">> RenderOutputDriverImplWrite.cc RUNTIME-VERIFY failed. fVecToHVec()\n";
            break;
        }
    }
#   endif // end RUNTIME_VERIFY_FTOH
}

// static function
float
RenderOutputWriter::htof(const unsigned short h)
{
    return _cvtsh_ss(h); // Convert half 16bit float to full 32bit float
}

// static function
unsigned short
RenderOutputWriter::ftoh(const float f)
{
    return _cvtss_sh(f, 0); // Convert full 32bit float to half 16bit float
                            // An immediate value controlling rounding using bits : 0=Nearest 
}

// static function
void
RenderOutputWriter::hVecToFVec(const unsigned short* hVec, const unsigned itemTotal, std::vector<float>& fVec)
{
    fVec.clear();
    fVec.resize(itemTotal);
    for (size_t i = 0; i < itemTotal; ++i) {
        fVec[i] = htof(hVec[i]);
    }
}

// static function
#ifdef PRECISE_HASH_COMPARE
float
RenderOutputWriter::precisionAdjustF(const float f, int /*reso*/)
{
    return f;
}
#else // else PRECISE_HASH_COMPARE
float
RenderOutputWriter::precisionAdjustF(const float f, int reso)
{
    return static_cast<float>(static_cast<long>(f * static_cast<float>(reso))) / static_cast<float>(reso);
}
#endif // end !PRECISE_HASH_COMPARE

// static function
#ifdef PRECISE_HASH_COMPARE
unsigned short
RenderOutputWriter::precisionAdjustH(const unsigned short h, int /*reso*/)
{
    return h;
}
#else // else PRECISE_HASH_COMPARE
unsigned short
RenderOutputWriter::precisionAdjustH(const unsigned short h, int reso)
{
    return ftoh(precisionAdjustF(htof(h), reso));
}
#endif // end !PRECISE_HASH_COMPARE

//------------------------------------------------------------------------------------------
    
std::string
RenderOutputWriter::errMsg(const std::string& msg,
                           const std::string& filename,
                           ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem) const
//
// error message utilities : add filename info to the end of the error message
//
{
    std::ostringstream ostr;
    ostr << msg << ' ';
    if (tmpFileItem) { // two stage output mode
        ostr << "filename='" << tmpFileItem->getDestinationFilename() << "'";
    } else {
        ostr << "filename='" << filename << "'";
    }
    return ostr.str();
}

} // namespace rndr
} // namespace moonray
