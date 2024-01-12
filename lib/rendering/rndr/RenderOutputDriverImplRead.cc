// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "Film.h"
#include "OiioReader.h"
#include "RenderContext.h"
#include "RenderOutputDriverImpl.h"
#include "RenderOutputHelper.h"

#include <moonray/rendering/mcrt_common/Clock.h>

namespace moonray {
namespace rndr {

namespace {

inline unsigned int f2ui(const float f)
// Simple cast from float value to unsigned int
{
    if (f < 0.0f) return 0; // just in case.
    return (unsigned int)f; // cast back to unsigned int
}

template <typename T, typename F>
void
setTiledBuf(OiioReader& reader, T* dstBuf, F pixCopyFunc)
// Copy all pixels by scanline segment order which is considered tiled memory layout
// and maximize memory access coherency for both of source and destination in parallel.
{
    scene_rdl2::fb_util::Tiler tiler(reader.getWidth(), reader.getHeight());
    reader.crawlAllTiledScanline([&](const int startX, const int endX, const int pixY, const int nChan, const float* pix) {
            T* dstPix = dstBuf + tiler.linearCoordsToTiledOffset(startX, pixY);
            for (int x = startX; x < endX; ++x) {
                pixCopyFunc(dstPix, pix);
                dstPix ++;
                pix += nChan;
            }
        });
}

} // namespace

bool
RenderOutputDriver::Impl::revertFilmData(Film& film,
                                         unsigned& resumeTileSamples, int& resumeNumConsistentSamples, bool& zeroWeightMask,
                                         bool& adaptiveSampling, float adaptiveSampleParam[3])
//
// Read all resume files which defined inside renderOutputDriver
// return 3 values by argument resumeTileSamples and resumeNumConsistentSamples come from file's metadata and
// zeroWeightMask condition is computed based on read weight AOV.
// Read data is stored into film and not do any denormalize/zeroWeight operation yet.
//
{
    // reset condition for read file
    bool errorCondition = false;
    mResumeTileSamples = -1;
    mResumeNumConsistentSamples = -1;
    mResumeAdaptiveSampling = -1;
    mRevertWeightAOV = false;
    mRevertBeautyAuxAOV = false;
    mZeroWeightMask = false;

    std::vector<std::string> resumeFiles;

    // Loop over all files and to read all resume file
    for (const auto& f: mFiles) {
        if (!read(f, film)) { // read one resume file
            std::ostringstream ostr;
            ostr << "Read file failed and could not revert film data from file. filename:" << f.mResumeName;
            mErrors.push_back(ostr.str());
            errorCondition = true;
        } else {
            resumeFiles.push_back(f.mResumeName);
        }
    }
    if (!errorCondition) {
        if (!resumeRenderReadyTest()) {
            errorCondition = true;
        }
    }

    if (!errorCondition) {
        resumeTileSamples = static_cast<unsigned>(mResumeTileSamples);
        resumeNumConsistentSamples = mResumeNumConsistentSamples; // if resume file does not include this info. this value is -1
        zeroWeightMask = mZeroWeightMask;
        adaptiveSampling = (mResumeAdaptiveSampling == 1)? true: false;
        adaptiveSampleParam[0] = mResumeAdaptiveParam[0];
        adaptiveSampleParam[1] = mResumeAdaptiveParam[1];
        adaptiveSampleParam[2] = mResumeAdaptiveParam[2]; // This is before multiply 10,000 value.

        // log output
        {
            std::ostringstream ostr;
            ostr << "resume render initial information {\n";
            if (resumeFiles.size() == 1) {
                ostr << "  read resume file : " << resumeFiles[0] << '\n';
            } else {
                ostr << "  read resume files (total:" << resumeFiles.size() << ") {\n";
                int iw = std::to_string(resumeFiles.size()).size();
                for (size_t i = 0; i < resumeFiles.size(); ++i) {
                    ostr << "    " << std::setw(iw) << i << ' ' << resumeFiles[i] << '\n';
                }
                ostr << "  }\n";
            }
            ostr << "  progressCheckpointTileSamples:" << resumeTileSamples << '\n'
                 << "  AovFilterNumConsistentSamples:" << resumeNumConsistentSamples << '\n';
            if (adaptiveSampling) {
                // display target adaptive error should be multiplied by 10,000.
                float targetAdaptiveError = adaptiveSampleParam[2] * 10000.0f;
                ostr << "  sampling:ADAPTIVE\n"
                     << "  adaptiveSampleParameters:"
                     << adaptiveSampleParam[0] << ' ' << adaptiveSampleParam[1] << ' ' << targetAdaptiveError << '\n';

            } else {
                ostr << "  sampling:UNIFORM\n";
            }
            ostr << "}";
            scene_rdl2::logging::Logger::info(ostr.str());
            // if (isatty(STDOUT_FILENO)) std::cout << ostr.str() << std::endl; // useful for debug
        }

        if (adaptiveSampling) {
            // We have to store resume render initial condition sampleId as a separate buffer.
            // 64 = tileWidth * tileHeight
            film.getResumeStartSampleIdBuff().init(film.getWeightBuffer(), mResumeTileSamples / 64);
        }
    }
    runOnResumeScript(!errorCondition, resumeFiles);

    return !errorCondition;
}

bool
RenderOutputDriver::Impl::resumeRenderReadyTest() const
//
// Can we continue to do resume render ?
//
{
    auto dumpResumeFileList = [](std::ostringstream &ostr, const std::vector<File> &files) {
        if (files.size() == 1) {
            ostr << "specified resume file:" << files[0].mResumeName;
        } else {
            ostr << "specified resume files { ";
            for (size_t i = 0; i < files.size(); ++i) {
                ostr << files[i].mResumeName;
                if (i != files.size() - 1) ostr << ", ";
            }
            ostr << " }";
        }
    };

    std::ostringstream ostr;
    bool errorCondition = false;

    if (!mRevertWeightAOV) {
        // We could not revert weight AOV from file. We can not resume render without weight.
        ostr << "Could not revert weight AOV data from resume file. We need weight AOV data for resume render. ";
        dumpResumeFileList(ostr, mFiles);
        mErrors.push_back(ostr.str());
        errorCondition = true;
    }

    if (mResumeTileSamples == -1) {
        ostr.str("");
        ostr << "Could not find \"progressCheckpointTileSamples\" metadata for resume. "
             << "Please generate resume file with proper options. ";
        dumpResumeFileList(ostr, mFiles);
        mErrors.push_back(ostr.str());
        errorCondition = true;
    }

    // mResumeNumConsistentSamples == -1 is not a critical error. We should continue to do resume render

    if (!mRevertBeautyAuxAOV) {
        ostr.str("");
        ostr << "Could not revert \"beauty aux\" AOV data from resume file."
             << " We need \"beauty aux\" AOV data for resume render. ";
        dumpResumeFileList(ostr, mFiles);
        mErrors.push_back(ostr.str());
        errorCondition = true;
    }

    return !errorCondition;
}

bool
RenderOutputDriver::Impl::read(const File& file, Film& film)
//
// Read one resume file and stored into film object as is.
// No de-normalization and other operations at this moment and just read from file.
//
{
    const std::string& filename = file.mResumeName;
    if (filename.empty()) return true; // early exit. skip this File data

    //
    // open file (not read image data yet)
    //
    OiioReader reader(filename);
    if (!reader) {
        mErrors.push_back("Can't open file");
        return false;
    }

    //
    // get parameters for resume render (read resume render related metadatas)
    //
    if (!readResumableParameters(reader)) {
        return false; // not enough information for resume render -> We can not resume render without them.
    }

    //
    // subImage loop
    //
    for (size_t imgId = 0; imgId < file.mImages.size(); ++imgId) {
        // read whole image data from file into memory inside reader here.
        if (!reader.readData(imgId)) {
            mErrors.push_back("read data failed.");
            return false;
        }
        // std::cerr << reader.showSpec("") << std::endl; // useful for debug

        // process reader internal memory data to proper way by multi-threaded operation
        if (!readSubImage(reader, file.mImages[imgId], film)) {
            return false;
        }
    }

    return true;
}

bool
RenderOutputDriver::Impl::readResumableParameters(OiioReader& reader)
{
    bool errorCondition = false;

    //
    // mResumeTileSamples
    //
    {
        //
        // Read tile sample information for resume rendering setup from resume file.
        // Checkpoint file already has metadata "progressCheckpointTileSamples" and this number indicates
        // how many tile samples are done. We need this number to restart redering.
        //
        int tileSamples;
        if (reader.getMetadata("progressCheckpointTileSamples", tileSamples)) {
            if (mResumeTileSamples == -1) {
                mResumeTileSamples = tileSamples;
            } else {
                if (mResumeTileSamples != tileSamples) {
                    std::ostringstream ostr;
                    ostr << "progressCheckpointTileSamples mismatch between multi resume input files. "
                         << "resumeTileSamples:" << mResumeTileSamples << " != " << tileSamples
                         << " of file:" << reader.filename();
                    mErrors.push_back(ostr.str());
                    errorCondition = true;
                }
            }
        }
    }

    //
    // mResumeNumConsistentSamples
    //
    {
        //
        // Read numConsistentSamples for resume rendering. This number is used for FORCE_CONSISTENT_SAMPLING
        // Aov filter to limit sampling count.
        // Regardress of this image's AOV uses this AOV filter or not, just in case this numConsistentSamples
        // value is saved as metadata. We should read here.
        int numConsistentSamples;
        if (reader.getMetadata("AovFilterNumConsistentSamples", numConsistentSamples)) {
            if (mResumeNumConsistentSamples == -1) {
                mResumeNumConsistentSamples = numConsistentSamples;
            } else {
                if (mResumeNumConsistentSamples != numConsistentSamples) {
                    std::ostringstream ostr;
                    ostr << "AovFilterNumConsistentSamples mismatch between multi resume input files. "
                         << "resumeNumConsistentSamples:" << mResumeNumConsistentSamples << " != " << numConsistentSamples
                         << " of file:" << reader.filename();
                    mErrors.push_back(ostr.str());
                    errorCondition = true;
                }
            }
        }
    }

    //
    // mResumeAdaptiveSampling, mResumeAdaptiveParam
    //
    {
        //
        // Read adaptiveSamplingV1 parameters. This info is used for control adaptive sampling resume rendering.
        //
        if (reader.getMetadata("adaptiveSamplingV1", mResumeAdaptiveParam)) {
            if (mResumeAdaptiveSampling == -1) {
                mResumeAdaptiveSampling = 1;
            } else {
                if (mResumeAdaptiveSampling != 1) {
                    std::ostringstream ostr;
                    ostr << "adaptiveSampleV1 mismatch between multi resume input files. "
                         << "resumeAdaptiveSampling:" << mResumeAdaptiveSampling << " != true"
                         << " of file:" << reader.filename();
                    mErrors.push_back(ostr.str());
                    errorCondition = true;
                }
            }
        }
    }

    //
    // resume render history
    //
    {
        std::string resumeHistory;
        if (reader.getMetadata("resumeHistory", resumeHistory)) {
            if (mResumeHistory.empty()) {
                mResumeHistory = std::move(resumeHistory);
            } else {
                if (mResumeHistory != resumeHistory) {
                    std::ostringstream ostr;
                    ostr << "resumeHistory information mismatch between multi resume input files."
                         << "resumeHistory:>" << mResumeHistory << "< != >" << resumeHistory
                         << " file:" << reader.filename();
                    mErrors.push_back(ostr.str());
                    errorCondition = true;
                }
            }
        }
    }

    /* useful debug dump
    std::cerr << ">> RenderOutputDriver.cc tileSamples:" << mResumeTileSamples << std::endl;
    std::cerr << ">> RenderOutputDriver.cc numConsistentSamples:" << mResumeNumConsistentSamples << std::endl;
    std::cerr << ">> RenderOutputDriver.cc"
              << " mResumeAdaptiveSampling:" << mResumeAdaptiveSampling
              << " mResumeAdaptiveParam:("
              << mResumeAdaptiveParam[0] << ' ' << mResumeAdaptiveParam[1] << ' ' << mResumeAdaptiveParam[2]
              << ")" << std::endl;
    std::cerr << ">> RenderOutputDriver.cc resumeHistory:>" << mResumeHistory << "<" << std::endl;
    */

    return !errorCondition;
}

bool
RenderOutputDriver::Impl::readSubImage(OiioReader& reader, const Image& currImage, Film& film)
{
    //
    // Test correctness of subImage by name attribute.
    //
    if (!readSubImageNameValidityTest(reader, currImage)) {
        return false;
    }

    //
    // Image's Entry loop : processing each entry in parallel
    //
    int roIdxStart = currImage.mStartRoIdx;

    bool errorCondition = false;

#ifdef SINGLE_THREAD_READ
    size_t taskSize = currImage.mEntries.size();
#else
    size_t taskSize = 1;
#endif
    tbb::blocked_range<size_t> range(0, currImage.mEntries.size(), taskSize);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t eId = r.begin(); eId < r.end(); ++eId) {
                if (!readSubImageOneEntry(reader, roIdxStart + eId, film)) {
                    errorCondition = true;
                }
            } // eId
        });

    return !errorCondition;
}

bool
RenderOutputDriver::Impl::readSubImageNameValidityTest(OiioReader& reader, const Image& currImage) const
{
    std::ostringstream ostr;

    std::string subImageName;
    if (!currImage.mName.empty()) {
        //
        // resume read information need to mach with currImage's subImage name.
        //
        if (!reader.getMetadata("name", subImageName)) {
            ostr << "Could not read subImge name attribute which should have (name:" << subImageName << "). "
                 << "This imply subImage order is wrong or improper resume file:" << reader.filename();
            mErrors.push_back(ostr.str());
            return false;
        }
        if (subImageName != currImage.mName) {
            ostr << "subImage name is wrong (!=" << subImageName << "). "
                 << "This imply subImage order is wrong or improper resume file:" << reader.filename();
            mErrors.push_back(ostr.str());
            return false;
        }
    } else {
        //
        // currImage does not have subImage name attribute. So resume read information also should not have name
        //
        if (reader.getMetadata("name", subImageName)) {
            ostr << "subImage should not have name attribute but it has (name:" << subImageName << "). "
                 << "This imply subImage order is wrong or improper resume file:" << reader.filename();
            mErrors.push_back(ostr.str());
            return false;
        }
    }
    return true;
}

bool
RenderOutputDriver::Impl::readSubImageOneEntry(OiioReader& reader,
                                               const int roIdx, // index of RenderOutputDriver::Impl::mEntries[] about currEntry
                                               Film& film)
{
    //
    // Properly setup one Entry data from already read resume data into proper destination
    // of memory inside Film object. Does not consider reverse normalization of weight value yet.
    // Result of destination is simply same as resume file data at this point.
    //

    const Entry& currEntry = *(mEntries[roIdx]);

    //
    // Try to find destination buffer address.
    //
    scene_rdl2::fb_util::HeatMapBuffer* heatMapBuffer = nullptr;
    scene_rdl2::fb_util::FloatBuffer* weightBuffer = nullptr;
    scene_rdl2::fb_util::VariablePixelBuffer* aovBuffer = nullptr;
    scene_rdl2::fb_util::RenderBuffer* renderBufferOdd = nullptr;
    if (!readSubImageSetDestinationBuffer(roIdx,
                                          film,
                                          &heatMapBuffer, &weightBuffer, &renderBufferOdd, &aovBuffer)) {
        return true; // we don't need to revert this entry from file (visibility_variance AOV). This is not a error
    }

    //
    // Validity check about buffer resolution
    //
    if (!readSubImageResoValidityTest(reader, currEntry,
                                      heatMapBuffer, weightBuffer, renderBufferOdd, aovBuffer,
                                      film.getCryptomatteBuffer())) {
        return false; // resolution check failed. This resume file is not sutable for resume render
    }

    //
    // Setup pixel channel offset table information
    //
    std::vector<int> chanOffset;
    if (!readSubImageSetPixChanOffset(reader, currEntry, chanOffset)) {
        return false; // Could not find channel inside resume file. So no way to revert data from resume file.
    }

    //
    // Loop over all pixels
    //
    // This logic is using same idea of RenderOutputDriver::Impl::fillBuffer() (but functionality opposite direction)
    // Actually destination buffer is tiled memory layout and input read resume data is untiled.
    // We have to think about memory layout conversion issue here. Main logic of tile <-> untile conversion is done
    // by inside of "setTiledBuf()" templates.
    //
    const scene_rdl2::rdl2::RenderOutput* ro = currEntry.mRenderOutput;
    switch (ro->getResult()) {
    case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP:
        {
            int64_t* dstBuf = heatMapBuffer->getData();
            setTiledBuf(reader, dstBuf, [&](int64_t* dstPix, const float* srcPix) {
                    // We stored seconds value into 32bit single float and already lost some precision.
                    // Simply convert seconds value to nanoseconds as int64_t here.
                    *dstPix = mcrt_common::Clock::nanoseconds((double)srcPix[chanOffset[0]]);
                });
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT:
        {
            float* dstBuf = weightBuffer->getData();
            bool zeroVal = false;
            setTiledBuf(reader, dstBuf, [&](float* dstPix, const float* srcPix) {
                    *dstPix = srcPix[chanOffset[0]];
                    if (*dstPix == 0.0f) zeroVal = true;
                });
            mRevertWeightAOV = true; // This flag indicates weight AOV is properly reverted.
            if (zeroVal) mZeroWeightMask = true;
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX:
        {
            if (renderBufferOdd) {  // only read if destination buffer is ready
                scene_rdl2::fb_util::RenderColor* dstBuf = renderBufferOdd->getData();
                setTiledBuf(reader, dstBuf, [&](scene_rdl2::fb_util::RenderColor* dstPix, const float* srcPix) {
                        (*dstPix)[0] = srcPix[chanOffset[0]];
                        (*dstPix)[1] = srcPix[chanOffset[1]];
                        (*dstPix)[2] = srcPix[chanOffset[2]];
                    });
                mRevertBeautyAuxAOV = true; // This flag indicates beauty aux AOV is properly reverted.
            }
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX:
        {
            if (renderBufferOdd) {
                scene_rdl2::fb_util::RenderColor* dstBuf = renderBufferOdd->getData();
                setTiledBuf(reader, dstBuf, [&](scene_rdl2::fb_util::RenderColor* dstPix, const float* srcPix) {
                        (*dstPix)[3] = srcPix[chanOffset[0]];
                    });
            }
        }
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE:
        {
            // Clear the cryptomatte buffer ready for loading resume data
            pbr::CryptomatteBuffer* cryptomatteBuf = film.getCryptomatteBuffer();
            cryptomatteBuf->clear();
            cryptomatteBuf->init(reader.getWidth(), reader.getHeight(), 1, cryptomatteBuf->getMultiPresenceOn());

            scene_rdl2::fb_util::Tiler tiler(reader.getWidth(), reader.getHeight());
            reader.crawlAllTiledScanline
                ([&](const int startX, const int endX, const int pixY, const int nChan, const float* pix) {
                    for (int x = startX; x < endX; ++x) {
                        // get the offsets to the cryptomatte data (the data isn't necessarily contiguous!)
                        const float* idAndCoverageData       = pix + reader.getPixChanOffset("Cryptomatte00.R");
                        const float* positionData            = pix + reader.getPixChanOffset("CryptoP00.R");
                        const float* normalData              = pix + reader.getPixChanOffset("CryptoN00.R");
                        const float* beautyData              = pix + reader.getPixChanOffset("CryptoB00.R");
                        const float* refPData                = pix + reader.getPixChanOffset("CryptoRefP00.R");
                        const float* refNData                = pix + reader.getPixChanOffset("CryptoRefN00.R");
                        const float* uvData                  = pix + reader.getPixChanOffset("CryptoUV00.R");
                        const float* resumeRenderSupportData = pix + reader.getPixChanOffset("CryptoS00.R");

                        cryptomatteBuf->addFragments(x, pixY, *ro, idAndCoverageData, positionData, normalData, 
                                                     beautyData, refPData, refNData, uvData, resumeRenderSupportData);
                        pix += nChan;
                    }
                });
            cryptomatteBuf->setFinalized(true);
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
        switch (aovBuffer->getFormat()) {
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT:
            {
                float* dstBuf = aovBuffer->getFloatBuffer().getData();
                setTiledBuf(reader, dstBuf, [&](float* dstPix, const float* srcPix) {
                        *dstPix = srcPix[chanOffset[0]];
                    });
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT2:
            {
                scene_rdl2::math::Vec2f* dstBuf = aovBuffer->getFloat2Buffer().getData();
                setTiledBuf(reader, dstBuf, [&](scene_rdl2::math::Vec2f* dstPix, const float* srcPix) {
                        dstPix->x = srcPix[chanOffset[0]];
                        dstPix->y = srcPix[chanOffset[1]];
                    });
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3:
            {
                scene_rdl2::math::Vec3f* dstBuf = aovBuffer->getFloat3Buffer().getData();
                setTiledBuf(reader, dstBuf, [&](scene_rdl2::math::Vec3f* dstPix, const float* srcPix) {
                        dstPix->x = srcPix[chanOffset[0]];
                        dstPix->y = srcPix[chanOffset[1]];
                        dstPix->z = srcPix[chanOffset[2]];
                    });
            }
            break;
        case scene_rdl2::fb_util::VariablePixelBuffer::FLOAT4:
            if (ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MathFilter::MATH_FILTER_CLOSEST) {
                // This is a special case and this AOV is using the closest math filter
                scene_rdl2::math::Vec4f* dstBuf = aovBuffer->getFloat4Buffer().getData();
                switch (film.getAovNumFloats(getAovBuffer(roIdx))) {
                case 1 :
                    setTiledBuf(reader, dstBuf, [&](scene_rdl2::math::Vec4f* dstPix, const float* srcPix) {
                            dstPix->x = srcPix[chanOffset[0]];
                            dstPix->y = 0.0f;
                            dstPix->z = 0.0f;
                            dstPix->w = srcPix[chanOffset[1]]; // depth
                        });
                    break;
                case 2 :
                    setTiledBuf(reader, dstBuf, [&](scene_rdl2::math::Vec4f* dstPix, const float* srcPix) {
                            dstPix->x = srcPix[chanOffset[0]];
                            dstPix->y = srcPix[chanOffset[1]];
                            dstPix->z = 0.0f;
                            dstPix->w = srcPix[chanOffset[2]]; // depth
                        });
                    break;
                case 3 :
                    setTiledBuf(reader, dstBuf, [&](scene_rdl2::math::Vec4f* dstPix, const float* srcPix) {
                            dstPix->x = srcPix[chanOffset[0]];
                            dstPix->y = srcPix[chanOffset[1]];
                            dstPix->z = srcPix[chanOffset[2]];
                            dstPix->w = srcPix[chanOffset[3]]; // depth
                        });
                    break;
                default : break; // unexpected aov size -> skip
                }
            }
            break;

        default:
            MNRY_ASSERT(0 && "unexpected variable pixel buffer format");
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER:
        // nothing needs to happen here because display filters are recomputed
        // from the aovs.
        break;
    default:
        MNRY_ASSERT(0 && "unknown result type");
    }

    return true;
}

bool
RenderOutputDriver::Impl::readSubImageSetDestinationBuffer(const int roIdx,
                                                           Film &film,
                                                           scene_rdl2::fb_util::HeatMapBuffer** heatMapBuffer,
                                                           scene_rdl2::fb_util::FloatBuffer** weightBuffer,
                                                           scene_rdl2::fb_util::RenderBuffer** renderBufferOdd,
                                                           scene_rdl2::fb_util::VariablePixelBuffer** aovBuffer) const
{
    bool returnVal = true; // continue to revert information from file

    switchAovType(*mRenderContext->getRenderOutputDriver(),
                  roIdx,
                  [&](const scene_rdl2::rdl2::RenderOutput *ro) { // non active AOV
                      switch (ro->getResult()) {
                      case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP:
                          *heatMapBuffer = film.getHeatMapBuffer();
                          break;
                      case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT:
                          *weightBuffer = &film.getWeightBuffer();
                          break;
                      case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX:
                          *renderBufferOdd = film.getRenderBufferOdd();
                          break;
                      case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX:
                          *renderBufferOdd = film.getRenderBufferOdd();
                          break;
                      }
                  },
                  [&](const int aovIdx) { // Visibility AOV
                      *aovBuffer = &film.getAovBuffer(aovIdx);
                  },
                  [&](const int aovIdx) { // regular AOV
                      *aovBuffer = &film.getAovBuffer(aovIdx);
                  });

    return returnVal;
}

bool
RenderOutputDriver::Impl::readSubImageResoValidityTest(OiioReader& reader,
                                                       const Entry& currEntry,
                                                       const scene_rdl2::fb_util::HeatMapBuffer* heatMapBuffer,
                                                       const scene_rdl2::fb_util::FloatBuffer* weightBuffer,
                                                       const scene_rdl2::fb_util::RenderBuffer* renderBufferOdd,
                                                       const scene_rdl2::fb_util::VariablePixelBuffer* aovBuffer,
                                                       const pbr::CryptomatteBuffer* cryptomatteBuffer) const
{
    std::ostringstream ostr;

    int w = reader.getTiledAlignedWidth();
    int h = reader.getTiledAlignedHeight();

    if (heatMapBuffer) {
        if (heatMapBuffer->getWidth() != w || heatMapBuffer->getHeight() != h) {
            ostr << "input heatMap buffer resolution is mismatch with internal buffer."
                 << " file:" << reader.filename()
                 << " internal(" << heatMapBuffer->getWidth() << "x" << heatMapBuffer->getHeight() << ") != "
                 << " file(tileAligned " << w << "x" << h << ")";
            mErrors.push_back(ostr.str());
            return false;
        }
    }
    if (weightBuffer) {
        if (weightBuffer->getWidth() != w || weightBuffer->getHeight() != h) {
            ostr << "input weight buffer resolution is mismatch with internal buffer."
                 << " file:" << reader.filename()
                 << " internal(" << weightBuffer->getWidth() << "x" << weightBuffer->getHeight() << ") != "
                 << " file(tileAligned " << w << "x" << h << ")";
            mErrors.push_back(ostr.str());
            return false;
        }
    }
    if (renderBufferOdd) {
        if (renderBufferOdd->getWidth() != w || renderBufferOdd->getHeight() != h) {
            ostr << "input renderBufferOdd buffer resolution is mismatch with internal buffer."
                 << " file:" << reader.filename()
                 << " internal(" << renderBufferOdd->getWidth() << "x" << renderBufferOdd->getHeight() << ") != "
                 << " file(tileAligned " << w << "x" << h << ")";
            mErrors.push_back(ostr.str());
            return false;
        }
    }
    if (aovBuffer) {
        if (aovBuffer->getWidth() != w || aovBuffer->getHeight() != h) {
            ostr << "input aovBuffer resolution (name:" << currEntry.mRenderOutput->getName() << ") is mismatch with "
                 << "internal buffer."
                 << " file:" << reader.filename()
                 << " internal(" << aovBuffer->getWidth() << "x" << aovBuffer->getHeight() << ") != "
                 << " file(tileAligned " << w << "x" << h << ")";
            mErrors.push_back(ostr.str());
            return false;
        }
    }
    if (cryptomatteBuffer) {
        int cryptoW = cryptomatteBuffer->getWidth();
        int cryptoH = cryptomatteBuffer->getHeight();
        int cryptoWTileAligned = (cryptoW + 7) & ~7;
        int cryptoHTileAligned = (cryptoH + 7) & ~7;
        if (cryptoWTileAligned != w || cryptoHTileAligned != h) {
            ostr << "input cryptomatte buffer resolution (name:"
                 << currEntry.mRenderOutput->getName() << ") is mismatch with "
                 << "internal buffer."
                 << " file:" << reader.filename()
                 << " internal(tileAligned " << cryptoWTileAligned << "x" << cryptoHTileAligned
                 << ") != "
                 << " file(tileAligned " << w << "x" << h << ")";
            mErrors.push_back(ostr.str());
            return false;
        }
    }

    return true;
}

bool
RenderOutputDriver::Impl::readSubImageSetPixChanOffset(OiioReader& reader,
                                                       const Entry& currEntry,
                                                       std::vector<int>& chanOffset) const
{
    for (size_t chanId = 0; chanId < currEntry.mChannelNames.size(); ++chanId) {
        const std::string& channelName = currEntry.mChannelNames[chanId];

        int pixChanOffset = reader.getPixChanOffset(channelName);
        if (pixChanOffset < 0) {
            std::ostringstream ostr;
            ostr << "Could not find pixel channel offset value for channelName:" << channelName << " "
                 << "Inproper resume file:" << reader.filename() << " ?";
            mErrors.push_back(ostr.str());
            return false;
        }
        chanOffset.push_back(pixChanOffset);
    } // chanId

    /* debug dump pixChanOffset table
    std::cerr << ">> RenderOutputDriver.cc chanOffset (total:" << chanOffset.size() << ") {";
    for (size_t i = 0; i < chanOffset.size(); ++i) {
        std::cerr << chanOffset[i];
        if (i != chanOffset.size() - 1) std::cerr << ",";
    }
    std::cerr << "}" << std::endl;
    */

    return true;
}

} // namespace rndr
} // namespace moonray

