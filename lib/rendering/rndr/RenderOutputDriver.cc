// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file RenderOutputDriver.cc

#include "Film.h"
#include "FrameState.h"
#include "ImageWriteDriver.h"
#include "RenderContext.h"
#include "RenderOutputDriverImpl.h"
#include "Util.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/mcrt_common/Clock.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Viewport.h>
#include <scene_rdl2/render/util/Files.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Strings.h>
#include <scene_rdl2/render/util/StrUtil.h>

#include <OpenImageIO/imageio.h>
#include <tbb/parallel_for.h>

#include <exception>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

#ifdef __INTEL_COMPILER
// We don't need any include for half float instructions
#else // else __INTEL_COMPILER
#include <x86intrin.h>          // _mm_cvtps_ph, _cvtph_ps : for GCC build
#endif // end else __INTEL_COMPILER

//#define SINGLE_THREAD_READ

using namespace moonray;

namespace {

// delete the return of realpath() with free()
auto freeChar = [](char *c) { free(c); };

// different filename attribute values can point to
// the same file. this attempts to catch this.
std::string
realfile(const std::string &filename)
{
    std::unique_ptr<char, decltype(freeChar)>
        rp(realpath(filename.c_str(), nullptr), freeChar);

    if (rp) {
        return std::string(rp.get());
    }

    return filename;
}

} // namespace

namespace moonray {
namespace rndr {

//----------------------------------------------------------------------------

namespace {
    template <typename... Additional>
    void skipRenderOutputMessage(const std::string& name, const std::string& message, Additional&&... additional)
    {
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] skipping active render output ", name, ": ", message, std::forward<Additional>(additional)...);
    }

    template <typename... Additional>
    void skipRenderOutputMessageImageConflict(const std::string& name0, const std::string& name1, const std::string& message, Additional&&... additional)
    {
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] skipping active render output ", name0, ". ",
                      name0, " and ", name1, " share the same output image but conflict: ",
                      message, std::forward<Additional>(additional)...);
    }
}

int
RenderOutputDriver::Impl::getAovSchemaID(const scene_rdl2::rdl2::RenderOutput* ro, int& lpePrefixFlags, pbr::AovSchemaId &stateAovId)
{
    // lpe prefix flags are really only used for light aovs:
    lpePrefixFlags = pbr::AovSchema::sLpePrefixNone;
    int roAovSchemaId = pbr::AOV_SCHEMA_ID_UNKNOWN;
    if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_STATE_VARIABLE ||
        ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DEPTH) {
        // state variables go in with the aovs
        // the depth result is just an alias for the state aov depth variable
        const scene_rdl2::rdl2::RenderOutput::StateVariable stateVariable =
            ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_STATE_VARIABLE ?
            ro->getStateVariable() : scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DEPTH;
        roAovSchemaId = stateVariableToSchema(stateVariable);
        if (roAovSchemaId == pbr::AOV_SCHEMA_ID_UNKNOWN) {
            skipRenderOutputMessage(ro->getName(), R"(unknown state variable enumeration ")",
                                                   static_cast<int>(ro->getStateVariable()), R"(".)");
            return pbr::AOV_SCHEMA_ID_UNKNOWN;
        }
        if (stateVariable == scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_MOTION) {
            mRequiresMotionVector = true;
        }
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_PRIMITIVE_ATTRIBUTE) {
        // primitive attribute go in with the aovs
        roAovSchemaId = primitiveAttributeToSchema(ro->getPrimitiveAttribute(),
                                                   ro->getPrimitiveAttributeType(),
                                                   mPrimAttrs);
        if (roAovSchemaId == pbr::AOV_SCHEMA_ID_UNKNOWN) {
            skipRenderOutputMessage(ro->getName(), R"(invalid primitive attribute expression ")",
                                   ro->getPrimitiveAttribute(), R"(" and type )",
                                   static_cast<int>(ro->getPrimitiveAttributeType()), ".");
            return pbr::AOV_SCHEMA_ID_UNKNOWN;
        }
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_MATERIAL_AOV) {
        // material aovs go in with the aovs
        roAovSchemaId = materialAovToSchema(ro, mPrimAttrs, lpePrefixFlags, stateAovId);
        if (roAovSchemaId == pbr::AOV_SCHEMA_ID_UNKNOWN) {
            skipRenderOutputMessage(ro->getName(), R"(invalid material aov expression ")",
                                   ro->getMaterialAov(), R"(")");
            return pbr::AOV_SCHEMA_ID_UNKNOWN;
        }
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_WIREFRAME) {
        // wireframe output is handled with state aovs
        roAovSchemaId = pbr::AOV_SCHEMA_ID_WIREFRAME;
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_LIGHT_AOV) {
        // light aovs go with the aovs
        roAovSchemaId = lightAovToSchema(ro, lpePrefixFlags);
        if (roAovSchemaId == pbr::AOV_SCHEMA_ID_UNKNOWN) {
            skipRenderOutputMessage(ro->getName(), R"(invalid light path expression ")",
                                   ro->getLpe(), R"(")");
            return pbr::AOV_SCHEMA_ID_UNKNOWN;
        }
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV) {
        // visibility aovs go with the aovs
        roAovSchemaId = visibilityAovToSchema(ro->getVisibilityAov());
        if (roAovSchemaId == pbr::AOV_SCHEMA_ID_UNKNOWN) {
            skipRenderOutputMessage(ro->getName(), R"(light path expression ")",
                                   ro->getVisibilityAov(), R"(" failed to be generated.)");
            return pbr::AOV_SCHEMA_ID_UNKNOWN;
        }
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY) {
        roAovSchemaId = pbr::AOV_SCHEMA_ID_BEAUTY;
    } else if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA) {
        roAovSchemaId = pbr::AOV_SCHEMA_ID_ALPHA;
    }
    return roAovSchemaId;
}

pbr::AovFilter
RenderOutputDriver::Impl::getAovFilter(const scene_rdl2::rdl2::RenderOutput *ro,
                                       int aovSchemaId,
                                       std::string *error) const
{
    // only state, primitive attribute, and material aovs support
    // special filters (non-avg, non-force consistent sampling)
    // extra_aovs are a special case of light aovs that can support avg, sum,
    // min and max.
    pbr::AovFilter filter = pbr::AOV_FILTER_AVG;
    if (aovSchemaId != pbr::AOV_SCHEMA_ID_UNKNOWN) {
        filter = pbr::AovFilter(static_cast<int>(ro->getMathFilter()));
        pbr::AovType type = pbr::aovType(aovSchemaId);
        // light aovs that have a 'U: label are extra aovs
        const bool isExtraAov = (type == pbr::AOV_TYPE_LIGHT_AOV &&
            ro->getLpe().find("'U:") != std::string::npos);
        const bool isMaterialLpeAov = (type == pbr::AOV_TYPE_LIGHT_AOV &&
            ro->getLpe().find("'M:") != std::string::npos);

        const bool extraAovUsingSupportedFilter =
            filter == pbr::AOV_FILTER_AVG || filter == pbr::AOV_FILTER_SUM ||
            filter == pbr::AOV_FILTER_MIN || filter == pbr::AOV_FILTER_MAX;

        if (type != pbr::AOV_TYPE_STATE_VAR &&
            type != pbr::AOV_TYPE_PRIM_ATTR &&
            type != pbr::AOV_TYPE_MATERIAL_AOV) {

            if (filter != pbr::AOV_FILTER_AVG &&
                filter != pbr::AOV_FILTER_FORCE_CONSISTENT_SAMPLING &&
                filter != pbr::AOV_FILTER_CLOSEST && 
                !isMaterialLpeAov &&
                !(isExtraAov && extraAovUsingSupportedFilter)) {
                if (error) {
                    *error = scene_rdl2::util::buildString("[MCRT-RENDER] active render output ",
                                               ro->getName(),
                                               ": unsupported math filter");
                }
                filter = pbr::AOV_FILTER_AVG;
            }
        }
    }
    return filter;
}

//---------------------------------------------------------------------------------------------------------------

RenderOutputDriver::Impl::Impl(const RenderContext *renderContext) :
    mLightAovs(renderContext->getActiveLayer()),
    mFinalMaxSamplesPerPixel(0),
    mResumeTileSamples(-1),
    mResumeNumConsistentSamples(-1),
    mResumeAdaptiveSampling(-1),
    mResumeAdaptiveParam{ 0.0f, 0.0f, 0.0f },
    mRevertWeightAOV(false),
    mZeroWeightMask(false),
    mRevertBeautyAuxAOV(false),
    mLastTileSamples(0),
    mRenderContext(renderContext)
{
    parserConfigure();

    const std::vector<const scene_rdl2::rdl2::RenderOutput *> &renderOutputs = renderContext->getSceneContext().getAllRenderOutputs();

    mCheckpointRenderActive = false; // Condition of this process is checkpoint render or not.
    bool resumeRenderActive = false;
    mResumableOutput = false;
    if (renderOutputs.size() >= 1 && renderOutputs[0]) {
        resumeRenderActive = renderOutputs[0]->getSceneClass().getSceneContext()->getResumeRender();
        if (resumeRenderActive) {
            mResumableOutput = true;
            mCheckpointRenderActive = true;
        } else {
            mResumableOutput = renderOutputs[0]->getSceneClass().getSceneContext()->getResumableOutput();
            mCheckpointRenderActive = renderOutputs[0]->getSceneClass().getSceneContext()->getCheckpointActive();
        }
    }
    ImageWriteDriver::get()->setResumableOutput(mResumableOutput); // set resumable condition

    const scene_rdl2::rdl2::SceneVariables &vars = renderContext->getSceneContext().getSceneVariables();
    mOverwriteCheckpoint = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointOverwrite);

    // clear our primitive attribute key set
    // this member is built up as we process the primitive attribute
    // render output objects
    mPrimAttrs.clear();
    mRequiresMotionVector = false;

    std::set<const scene_rdl2::rdl2::RenderOutput*> varianceAovReferences;

    bool defWeightAOV = false;
    bool defBeautyAuxAOV = false;
    bool defAlphaAuxAOV = false;

    for (auto ro: renderOutputs) {
        if (ro->getActive()) {
            // make the aov schema for those result types that end up
            // in the aov buffer.
            int lpePrefixFlags = pbr::AovSchema::sLpePrefixNone;
            pbr::AovSchemaId stateAovId = pbr::AovSchemaId::AOV_SCHEMA_ID_UNKNOWN;
            const int roAovSchemaId = getAovSchemaID(ro, lpePrefixFlags, stateAovId);
            if (roAovSchemaId == pbr::AOV_SCHEMA_ID_UNKNOWN) {
                if (ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP &&
                    ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT &&
                    ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX &&
                    ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX &&
                    ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE &&
                    ro->getResult() != scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
                    continue;
                }
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT) defWeightAOV = true;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX) defBeautyAuxAOV = true;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX) defAlphaAuxAOV = true;
            }

            if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER &&
                ro->getDisplayFilter() == nullptr) {
                // bad input. Cannot have display filter result without a display filter.
                continue;
            }

            // find or create the file
            const std::string filename = realfile(ro->getFileName());
            auto file = std::find_if(mFiles.begin(), mFiles.end(),
                                     [&filename](const File &out)
                                     {
                                         return out.mName == filename;
                                     });
            if (file == mFiles.end()) {
                mFiles.emplace_back(filename);
                file = std::prev(mFiles.end());
            }

            if (mCheckpointRenderActive) {
                // checkpoint output
                const std::string checkpointFilename = realfile(ro->getCheckpointFileName());
                if (file->mCheckpointName.empty()) {
                    file->mCheckpointName = checkpointFilename;
                } else if (file->mCheckpointName != checkpointFilename) {
                    std::ostringstream ostr;
                    ostr << "[MCRT-RENDER] Conflict checkpoint output filename. "
                         << "Defines \"" << file->mCheckpointName << "\" and \"" << checkpointFilename << "\""
                         << " for same checkpoint output at RenderOutput:" << ro->getName();
                    throw scene_rdl2::except::RuntimeError(ostr.str());
                }

                // checkpoint multi-version output
                const std::string checkpointMultiVersionFilename =
                    realfile(ro->getCheckpointMultiVersionFileName());
                if (file->mCheckpointMultiVersionName.empty()) {
                    file->mCheckpointMultiVersionName = checkpointMultiVersionFilename;
                } else if (file->mCheckpointMultiVersionName != checkpointMultiVersionFilename) {
                    std::ostringstream ostr;
                    ostr << "[MCRT-RENDER] Conflict checkpoint multi-version output filename. "
                         << "Defines \"" << file->mCheckpointMultiVersionName << "\" and \""
                         << checkpointMultiVersionFilename << "\""
                         << " for same checkpoint output at RenderOutput:" << ro->getName();
                    throw scene_rdl2::except::RuntimeError(ostr.str());
                }
                          
            } else {
                file->mCheckpointName.clear();
                file->mCheckpointMultiVersionName.clear();
            }

            if (resumeRenderActive) {
                // resume input
                if (ro->getResumeFileName() != "") {
                    const std::string resumeFilename = realfile(ro->getResumeFileName());
                    if (file->mResumeName.empty()) {
                        file->mResumeName = resumeFilename;
                    } else if (file->mResumeName != resumeFilename) {
                        std::ostringstream ostr;
                        ostr << "[MCRT-RENDER] Conflict resume input filename. "
                             << "Defines \"" << file->mResumeName << "\" and \"" << resumeFilename << "\""
                             << " for same resume render input at RenderOutput:" << ro->getName();
                        throw scene_rdl2::except::RuntimeError(ostr.str());
                    }
                }
            } else {
                file->mResumeName.clear();
            }

            // find or create the image
            auto image = std::find_if(file->mImages.begin(), file->mImages.end(),
                                      [ro](const Image &i)
                                      {
                                          return ro->getFilePart() == i.mName;
                                      });
            if (image == file->mImages.end()) {
                file->mImages.emplace_back(ro->getFilePart());
                image = std::prev(file->mImages.end());
            }
            // resolve channel names
            std::vector<std::string> roChannelNames;
            getChannelNames(ro, roAovSchemaId, roChannelNames);
            // check for conflicts
            auto conflict = std::find_if(image->mEntries.begin(), image->mEntries.end(),
                                         [ro, &roChannelNames](const Entry &entry)
                                         {
                                             // check for:
                                             //   1. channel name conflicts (if aov type is the same)
                                             //   2. different compressions
                                             //   3. different metadata for same file part
                                             //   4. file part specified in one
                                             //   output, but empty in another
                                             const scene_rdl2::rdl2::RenderOutput *ro2 = entry.mRenderOutput;
                                             bool sameChan = false;
                                             for (const auto &c0: roChannelNames) {
                                                 for (const auto &c1: entry.mChannelNames) {
                                                     if (c0 == c1 && ro->getResult() == ro2->getResult()) {
                                                         sameChan = true;
                                                         break;
                                                     }
                                                 }
                                                 if (sameChan) break;
                                             }
                                             const bool diffComp = ro->getCompression() != ro2->getCompression();
                                             const bool diffCompLevel = ro->getCompressionLevel() != ro2->getCompressionLevel();

                                             // check if metadata is different
                                             const bool diffMetadata = ro->getExrHeaderAttributes() !=
                                                     ro2->getExrHeaderAttributes()
                                                     // and neither metadata is null
                                                     && (ro->getExrHeaderAttributes() && ro2->getExrHeaderAttributes());

                                             bool partDiff = ro->getFilePart().empty() != ro2->getFilePart().empty();

                                             return sameChan || diffComp || diffCompLevel || diffMetadata || partDiff;
                                         });
            if (conflict != image->mEntries.end()) {
                const scene_rdl2::rdl2::RenderOutput *ro2 = conflict->mRenderOutput;
                // found a conflict
                if (ro->getCompression() != ro2->getCompression()) {
                    skipRenderOutputMessageImageConflict(ro->getName(), ro2->getName(), "specify different image compressions.");
                }
                if (ro->getCompressionLevel() != ro2->getCompressionLevel()) {
                    skipRenderOutputMessageImageConflict(ro->getName(), ro2->getName(), "specify different image compression levels.");
                }
                if (ro->getExrHeaderAttributes() != ro2->getExrHeaderAttributes() &&
                    ro->getFilePart() == ro2->getFilePart()) {
                    skipRenderOutputMessageImageConflict(ro->getName(), ro2->getName(), "specify different image metadata.");
                }
                if (!ro->getFilePart().empty() && ro2->getFilePart().empty()) {
                    skipRenderOutputMessageImageConflict(ro->getName(), ro2->getName(), ro->getName(), " does not specify a file part while ", ro2->getName(), " does.");
                }
                if (ro->getFilePart().empty() && !ro2->getFilePart().empty()) {
                    skipRenderOutputMessageImageConflict(ro->getName(), ro2->getName(), ro->getName(), " specifies a file part while ", ro2->getName(), " does not.");
                }
                for (const auto &roC : roChannelNames) {
                    for (const auto &conflictC: conflict->mChannelNames) {
                        if (roC == conflictC &&
                            ro->getResult() == ro2->getResult()) {
                                skipRenderOutputMessageImageConflict(ro->getName(), ro2->getName(), R"(both target channel ")", roC, R"(".)");
                        }
                    }
                }
            } else {
                // all good, add it
                image->mEntries.emplace_back(ro);
                image->mEntries.back().mChannelNames = roChannelNames;
                image->mEntries.back().mAovSchemaId = roAovSchemaId;
                image->mEntries.back().mLpePrefixFlags = lpePrefixFlags;
                image->mEntries.back().mStateAovId = stateAovId;
            }
        }
    }

    if (mCheckpointRenderActive) {
        //
        // Check duplicate checkpoint related output settings regarding the multiple file output entries.
        //
        for (size_t i = 0; i < mFiles.size() - 1; ++i) {
            for (size_t j = i + 1; j < mFiles.size(); ++j) {
                if (!mFiles[i].mCheckpointName.empty() && mFiles[i].mCheckpointName == mFiles[j].mCheckpointName) {
                    std::ostringstream ostr;
                    ostr << "[MCRT-RENDER] Conflict checkpoint output filename. "
                         << "Multiple checkpoint output files have same output name:" << mFiles[j].mCheckpointName;
                    throw scene_rdl2::except::RuntimeError(ostr.str());
                }

                if (!mFiles[i].mCheckpointMultiVersionName.empty() &&
                    mFiles[i].mCheckpointMultiVersionName == mFiles[j].mCheckpointMultiVersionName) {
                    std::ostringstream ostr;
                    ostr << "[MCRT-RENDER] Conflict checkpoint multi-version output filename. "
                         << "Multiple checkpoint multi-version output files have same output name:"
                         << mFiles[j].mCheckpointMultiVersionName;
                    throw scene_rdl2::except::RuntimeError(ostr.str());
                }
            }
        }
    }

    //
    // Create checkpoint related outfile path if it doesn't already exist
    //
    for (size_t i = 0; i < mFiles.size(); ++i){
        if (!scene_rdl2::util::writeTest(mFiles[i].mName, true)){
            throw std::runtime_error(std::string("Could not write to output file: ") + mFiles[i].mName);
        }
        if (mCheckpointRenderActive){
            if (!mFiles[i].mCheckpointName.empty()) {
                if (!scene_rdl2::util::writeTest(mFiles[i].mCheckpointName, true)){
                    throw std::runtime_error(std::string("Could not access checkpoint file: ") +
                                             mFiles[i].mCheckpointName);
                }
            }
            if (!mFiles[i].mCheckpointMultiVersionName.empty()) {
                if (!scene_rdl2::util::writeTest(mFiles[i].mCheckpointMultiVersionName, true)) {
                    throw std::runtime_error(std::string("Could not access checkpoint multi-version file:") +
                                             mFiles[i].mCheckpointMultiVersionName);
                }
            }
        }
    }

    //
    // check temporary directory and create it it doesn't exist
    //
    std::string testFile = vars.getTmpDir() + "/__TMPDIR_ACCESS_TEST__";
    if (!scene_rdl2::util::writeTest(testFile, true)) {
        throw std::runtime_error(std::string("Could not access temporary directory:") +
                                 vars.getTmpDir());
    }

    if (resumeRenderActive) {
        //
        // Check duplicate resume input filename setting regarding to the multiple file entries
        //
        for (size_t i = 0; i < mFiles.size() - 1; ++i) {
            for (size_t j = i + 1; j < mFiles.size(); ++j) {
                if (!mFiles[i].mResumeName.empty() && mFiles[i].mResumeName == mFiles[j].mResumeName) {
                    std::ostringstream ostr;
                    ostr << "[MCRT-RENDER] Conflict resume input filename. "
                         << "Multiple resume input files have same input name:" << mFiles[j].mResumeName;
                    throw scene_rdl2::except::RuntimeError(ostr.str());
                }
            }
        }
    }

    if (mResumableOutput) {
        if (!defWeightAOV || !defBeautyAuxAOV || !defAlphaAuxAOV) {
            std::vector<std::string> errorAOV;
            if (!defWeightAOV) errorAOV.push_back("weight");
            if (!defBeautyAuxAOV) errorAOV.push_back("beauty aux");
            if (!defAlphaAuxAOV) errorAOV.push_back("alpha aux");

            std::ostringstream ostr;
            for (size_t i = 0; i < errorAOV.size(); ++i) {
                ostr << "\"" << errorAOV[i] << "\"";
                if (i < errorAOV.size() - 1) {
                    ostr << ((i == errorAOV.size() - 2)? " and ": ", ");
                }
            }
            std::string errorAOVnames = ostr.str();

            ostr.str("");
            ostr << "[MCRT-RENDER] You should define " << errorAOVnames << " AOV for resumable_output mode. "
                 << "RenderOutput definition does not include " << errorAOVnames << " AOV."
                 << "You can not use this output file as resume file.";

            throw scene_rdl2::except::RuntimeError(ostr.str());
        }
    }

    // construct our entry order
    // set the aov schema
    // set our aov offsets
    // this assumes the iteration order in write()
    mEntries.clear();
    std::vector<pbr::AovSchema::EntryData> schemaData;
    mAovBuffers.clear();
    mDenoiserAlbedoInput = -1;
    mDenoiserNormalInput = -1;
    mDisplayFilterIndices.clear();

    unsigned int aovBuffer = 0;
    unsigned int dfIdx = 0;
    for (auto &file: mFiles) {
        for (auto &image: file.mImages) {
            image.mStartRoIdx = (int)mEntries.size(); // save roIdx of image.mEntries[0]
            for (const auto &entry: image.mEntries) {

                mEntries.push_back(&entry);

                // Non-aov entries do not contribute to the aov schema.
                // It is worth noting that the aov schema is not necessarily the
                // same size as the aovBuffers.  The aov schema contains just
                // the order of aovs in the aov buffer - the aov buffers array
                // contains an entry for every render output (using -1 to indicate
                // that the result is not in the aov buffer).
                if (entry.mAovSchemaId != pbr::AOV_SCHEMA_ID_UNKNOWN) {
                    pbr::AovSchema::EntryData data;
                    data.schemaID = entry.mAovSchemaId;
                    data.lpePrefixFlags = entry.mLpePrefixFlags;
                    data.stateAovId = entry.mStateAovId;

                    std::string filterWarning;
                    data.filter = getAovFilter(entry.mRenderOutput, entry.mAovSchemaId, &filterWarning);
                    if (!filterWarning.empty()) {
                        scene_rdl2::logging::Logger::warn(filterWarning);
                    }

                    data.storageType = getStorageType(entry.mRenderOutput, entry.mAovSchemaId, data.filter);

                    mAovBuffers.push_back(aovBuffer);

                    schemaData.push_back(data);
                    ++aovBuffer;

                    // denoiser input?
                    scene_rdl2::rdl2::RenderOutput::DenoiserInput di = entry.mRenderOutput->getDenoiserInput();
                    if (di == scene_rdl2::rdl2::RenderOutput::DENOISER_INPUT_ALBEDO) {
                        // do we already have a designated albedo input?
                        if (mDenoiserAlbedoInput != -1) {
                            scene_rdl2::logging::Logger::warn(
                                scene_rdl2::util::buildString(
                                    "[MCRT-RENDER] active render output ", entry.mRenderOutput->getName(),
                                    " specifies itself as the albedo denoiser input.  But so does ",
                                    mEntries[mDenoiserAlbedoInput]->mRenderOutput->getName(),
                                    ".  Using ", mEntries[mDenoiserAlbedoInput]->mRenderOutput->getName(),
                                    " as the albedo denoiser input."));
                        } else {
                            mDenoiserAlbedoInput = mEntries.size() - 1;
                        }
                    } else if (di == scene_rdl2::rdl2::RenderOutput::DENOISER_INPUT_NORMAL) {
                        // do we already have a designated normal input?
                        if (mDenoiserNormalInput != -1) {
                            scene_rdl2::logging::Logger::warn(
                                scene_rdl2::util::buildString(
                                    "[MCRT-RENDER] active render output ", entry.mRenderOutput->getName(),
                                    " specifies itself as the normal denoiser input.  But so does ",
                                    mEntries[mDenoiserNormalInput]->mRenderOutput->getName(),
                                    ".  Using ", mEntries[mDenoiserNormalInput]->mRenderOutput->getName(),
                                    " as the normal denoiser input."));
                        } else {
                            mDenoiserNormalInput = mEntries.size() - 1;
                        }
                    }

                    mDisplayFilterIndices.push_back(-1); // not a display filter

                } else {

                    // only aovs can be used as denoiser inputs
                    if (entry.mRenderOutput->getDenoiserInput() != scene_rdl2::rdl2::RenderOutput::DENOISER_INPUT_NONE) {
                        scene_rdl2::logging::Logger::warn(
                            scene_rdl2::util::buildString(
                                "[MCRT-RENDER] active render output ", entry.mRenderOutput->getName(),
                                " cannot be used as a denoiser input."));
                    }

                    mAovBuffers.push_back(-1); // not an aov

                    if (entry.mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
                        mDisplayFilterIndices.push_back(dfIdx);
                        ++dfIdx;
                    } else {
                        mDisplayFilterIndices.push_back(-1); // not a display filter
                    }
                }
            }
        }
    }

    // init the aov schema
    mAovSchema.init(schemaData);

    // finalize our light aov object
    mLightAovs.finalize();

#ifdef DEBUG_DUMP_ENTRIES_AND_FILES
    // Useful debug dump to trackdown all entry items and file info of renderOutputDriver
    {
        std::ostringstream ostr;
        ostr << "Entries (total:" << mEntries.size() << ") {\n";
        for (size_t i = 0; i < mEntries.size(); ++i) {
            ostr << "  i:" << i << '\n';
            ostr << mEntries[i]->showChannelNames("  ") << '\n';
        }
        ostr << "}\n";

        int totalFiles = 0;
        int totalParts = 0;
        int totalFloatChan = 0;
        int totalHalfChan = 0;

        ostr << "File (total:" << mFiles.size() << ") {\n";
        for (size_t i = 0; i < mFiles.size(); ++i) {
            totalFiles++;
            ostr << "  i:" << i
                 << " checkpoint:" << mFiles[i].mCheckpointName
                 << " checkpointMultiVer:" << mFiles[i].mCheckpointMultiVersionName
                 << " resume:" << mFiles[i].mResumeName << '\n';

            ostr << "  file {\n";
            const File &f = mFiles[i];
            for (const auto &i: f.mImages) {
                totalParts++;
                
                ostr << "    part {\n";
                for (const auto &e: i.mEntries) {
                    auto fmtStr = [](const rdl2::RenderOutput *ro) -> std::string {
                        if (ro->getChannelFormat() == rdl2::RenderOutput::CHANNEL_FORMAT_HALF) {
                            return "half";
                        }
                        return "float";
                    };
                    
                    const rdl2::RenderOutput *ro = e.mRenderOutput;
                    ostr << "      entry {";
                    for (size_t j = 0; j < e.mChannelNames.size(); ++j) {
                        if (j != 0) ostr << " ";
                        ostr << e.mChannelNames[j];
                        if (ro->getChannelFormat() == rdl2::RenderOutput::CHANNEL_FORMAT_HALF) {
                            totalHalfChan++;
                        } else {
                            totalFloatChan++;
                        }
                   }
                    ostr << "} fmt:" << fmtStr(ro) << "\n";
                }
                ostr << "    }\n";
            }
            ostr << "  }\n";
        }
        ostr << "}";
        ostr << " files:" << totalFiles << " parts:" << totalParts;
        ostr << " floatChan:" << totalFloatChan << " halfChan:" << totalHalfChan;

        size_t pixelSize = totalFloatChan * sizeof(float) + totalHalfChan * sizeof(unsigned short);
        ostr << " pix:" << pixelSize << " byte";

        std::cerr << ">> RenderOutputDriver.cc " << ostr.str() << std::endl;
    }
#endif // end DEBUG_DUMP_ENTRIES_AND_FILES
}

unsigned int
RenderOutputDriver::Impl::getNumberOfRenderOutputs() const
{
    return mEntries.size();
}

const scene_rdl2::rdl2::RenderOutput *
RenderOutputDriver::Impl::getRenderOutput(unsigned int indx) const
{
    return mEntries[indx]->mRenderOutput;
}

int
RenderOutputDriver::Impl::getRenderOutputIndx(const scene_rdl2::rdl2::RenderOutput *ro) const
{
    for (unsigned int i = 0; i < mEntries.size(); ++i) {
        if (mEntries[i]->mRenderOutput == ro) {
            return i;
        }
    }

    return -1; // not found
}

unsigned int
RenderOutputDriver::Impl::getNumberOfChannels(unsigned int indx) const
{
    return mEntries[indx]->mChannelNames.size();
}

int
RenderOutputDriver::Impl::getAovBuffer(unsigned int indx) const
{
    return mAovBuffers[indx];
}

int
RenderOutputDriver::Impl::getDisplayFilterIndex(unsigned int indx) const
{
    return mDisplayFilterIndices[indx];
}

float
RenderOutputDriver::Impl::getAovDefaultValue(unsigned int indx) const
{
    int aovIdx = getAovBuffer(indx); // aovIdx is aovSchema id
    if (aovIdx < 0) {
        return 0.0f;        // HEAT_MAP, WEIGHT, BEAUTY_AUX, ALPHA_AUX
    }

    const auto &entry = mAovSchema[aovIdx];
    return entry.defaultValue();
}

bool
RenderOutputDriver::Impl::requiresScaledByWeight(unsigned int indx) const
{
    int aovIdx = getAovBuffer(indx); // aovIdx is aovSchema id
    if (aovIdx < 0) {
        // HEAT_MAP, WEIGHT, BEAUTY_AUX, ALPHA_AUX
        // Note : We don't have enough information when RenderOutput is BEAUTY_AUX/ALPHA_AUX
        // which need to be scaled by weight.
        // So we need special care for BEAUTY_AUX/ALPHA_AUX case by caller function.
        return false;
    }
    return mAovSchema.requiresScaledByWeight(aovIdx);
}

bool
RenderOutputDriver::Impl::isVisibilityAov(unsigned int indx) const
{
    return mEntries[indx]->mAovSchemaId >= pbr::AOV_SCHEMA_ID_VISIBILITY_AOV &&
           mEntries[indx]->mAovSchemaId < pbr::AOV_SCHEMA_ID_LIGHT_AOV;
}

const shading::AttributeKeySet &
RenderOutputDriver::Impl::getPrimAttrs() const
{
    return mPrimAttrs;
}

unsigned int
RenderOutputDriver::Impl::getDisplayFilterCount() const
{
    return std::accumulate(mEntries.begin(), mEntries.end(), 0, [](int count, const Entry* entry) {
        return count + (entry->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER);
    });
}

bool
RenderOutputDriver::Impl::requiresDeepBuffer() const
{
    for (unsigned int indx = 0; indx < mEntries.size(); ++indx) {
        if (requiresDeepBuffer(indx)) {
            return true;
        }
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresDeepBuffer(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getOutputType() == std::string("deep")) {
        return true;
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresCryptomatteBuffer() const
{
    for (const auto &f: mFiles) {
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresCryptomatteBuffer(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE) {
        return true;
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresRenderBuffer() const
{
    for (unsigned int indx = 0; indx < mEntries.size(); ++indx) {
        if (requiresRenderBuffer(indx)) {
            return true;
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresRenderBuffer(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY ||
        mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA) {
        return true;
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresRenderBufferOdd() const
{
    for (unsigned int indx = 0; indx < mEntries.size(); ++indx) {
        if (requiresRenderBufferOdd(indx)) {
            return true;
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresRenderBufferOdd(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX ||
        mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX) {
        return true;
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresHeatMap() const
{
    for (const auto &f: mFiles) {
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresHeatMap(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP) {
        return true;
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresWeightBuffer() const
{
    for (const auto &f: mFiles) {
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresWeightBuffer(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT) {
        return true;
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresDisplayFilter(unsigned int indx) const
{
    MNRY_ASSERT(indx < mEntries.size());
    if (mEntries[indx]->mRenderOutput->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
        return true;
    }

    return false;
}

bool
RenderOutputDriver::Impl::requiresDisplayFilter() const
{
    for (const auto &f: mFiles) {
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresWireframe() const
{
    for (const auto &f: mFiles) {
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_WIREFRAME) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool
RenderOutputDriver::Impl::requiresMotionVector() const
{
    return mRequiresMotionVector;
}

std::string    
RenderOutputDriver::Impl::showWriteInfo(const int width, const int height) const
{
    int fileTotal = 0;
    int imageTotal = 0;
    int numChanHalf = 0;
    int numChanFull = 0;
    
    fileTotal = mFiles.size();
    for (const auto &f: mFiles) {

        imageTotal += f.mImages.size();
        for (auto i = f.mImages.begin(); i != f.mImages.end(); ++i) {

            for (const auto &e: i->mEntries) {
                if (e.mRenderOutput->getChannelFormat() == scene_rdl2::rdl2::RenderOutput::CHANNEL_FORMAT_HALF) {
                    numChanHalf += e.mChannelNames.size();
                } else {
                    numChanFull += e.mChannelNames.size();
                }
            } // i->mEntries
        } // f.mImages
    } // mFIles

    size_t size = width * height * (numChanFull * sizeof(float) + numChanHalf * sizeof(unsigned short));

    std::ostringstream ostr;
    ostr << "writeInfo "
         << " w:" << width
         << " h:" << height
         << " fileTotal:" << fileTotal
         << " imageTotal:" << imageTotal
         << " numChanHalf:" << numChanHalf
         << " numCHanFull:" << numChanFull
         << " size:" << size << " byte (" << scene_rdl2::str_util::byteStr(size) << ")";
    return ostr.str();
}

bool
RenderOutputDriver::Impl::loggingErrorAndInfo(ImageWriteCache *cache) const
{
    std::vector<std::string> &errors = (cache)? cache->getErrors(): mErrors;
    std::vector<std::string> &infos = (cache)? cache->getInfos(): mInfos;

    bool st = errors.empty();

    for (const auto &e: errors) {
        scene_rdl2::logging::Logger::error(e);
        // if (isatty(STDOUT_FILENO)) std::cout << e << std::endl; // useful for debug run from terminal
    }
    errors.clear();

    for (const auto &i: infos) {
        scene_rdl2::logging::Logger::info(i);
        // if (isatty(STDOUT_FILENO)) std::cout << i << std::endl; // useful for debug run from terminal
    }
    infos.clear();

    return st;
}

template <typename SRC_BUFFER_TYPE>
static void
initializeDestBuffer(scene_rdl2::fb_util::VariablePixelBuffer *destBuffer,
                     const SRC_BUFFER_TYPE *srcBuffer,
                     scene_rdl2::fb_util::VariablePixelBuffer::Format format)
{
    MNRY_ASSERT(srcBuffer);
    const unsigned int width  = srcBuffer->getWidth();
    const unsigned int height = srcBuffer->getHeight();

    destBuffer->init(format, width, height);
}

template<typename SRC_PIXEL_TYPE, typename DEST_PIXEL_TYPE, typename PIXEL_XFORM_FN>
static void
copyAndTransform(scene_rdl2::fb_util::PixelBuffer<DEST_PIXEL_TYPE> *destBuffer,
                 const scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> *srcBuffer,
                 PIXEL_XFORM_FN const &pixelXform,
                 bool parallel)
{
    const unsigned int h = srcBuffer->getHeight();
    const unsigned int w = srcBuffer->getWidth();

    simpleLoop(parallel, 0u, h, [&](unsigned int y) {
        DEST_PIXEL_TYPE      *dst = destBuffer->getRow(y);
        const SRC_PIXEL_TYPE *src = srcBuffer->getRow(y);

        for (unsigned int x = 0; x < w; ++x) {
            pixelXform(dst[x], src[x]);
        }
    });
}

void
RenderOutputDriver::Impl::finishSnapshot(scene_rdl2::fb_util::VariablePixelBuffer *destBuffer, unsigned int indx,
                                         const scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                                         const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                                         const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                         const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                         bool parallel) const
{
    MNRY_ASSERT(indx < mEntries.size());

    switch (mEntries[indx]->mRenderOutput->getResult()) {
    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY:
        // extract RGB from RenderBuffer
        initializeDestBuffer(destBuffer, renderBuffer, scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3);
        copyAndTransform(&destBuffer->getFloat3Buffer(), renderBuffer,
                         [](scene_rdl2::math::Vec3f &dst, const scene_rdl2::fb_util::RenderColor &src) {
                             dst.x = src.x;
                             dst.y = src.y;
                             dst.z = src.z;
                         }, parallel);
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX:
        // extract RGB from RenderBufferOdd
        initializeDestBuffer(destBuffer, renderBufferOdd, scene_rdl2::fb_util::VariablePixelBuffer::FLOAT3);
        copyAndTransform(&destBuffer->getFloat3Buffer(), renderBufferOdd,
                         [](scene_rdl2::math::Vec3f &dst, const scene_rdl2::fb_util::RenderColor &src) {
                             dst.x = src.x;
                             dst.y = src.y;
                             dst.z = src.z;
                         }, parallel);
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA:
        // extract alpha of RenderBuffer
        initializeDestBuffer(destBuffer, renderBuffer, scene_rdl2::fb_util::VariablePixelBuffer::FLOAT);
        copyAndTransform(&destBuffer->getFloatBuffer(), renderBuffer,
                         [](float &dst, const scene_rdl2::fb_util::RenderColor &src) {
                             dst = src.w;
                         }, parallel);
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX:
        // extract alpha of RenderBufferOdd
        initializeDestBuffer(destBuffer, renderBufferOdd, scene_rdl2::fb_util::VariablePixelBuffer::FLOAT);
        copyAndTransform(&destBuffer->getFloatBuffer(), renderBufferOdd,
                         [](float &dst, const scene_rdl2::fb_util::RenderColor &src) {
                             dst = src.w;
                         }, parallel);
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP:
        // heat map buffer values are in ticks, convert to seconds
        initializeDestBuffer(destBuffer, heatMapBuffer, scene_rdl2::fb_util::VariablePixelBuffer::FLOAT);
        copyAndTransform(&destBuffer->getFloatBuffer(), heatMapBuffer,
                         [](float &dst, int64_t src) {
                             dst = mcrt_common::Clock::seconds(src);
                         }, parallel);
        break;

    case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT:
        // extract weight
        initializeDestBuffer(destBuffer, weightBuffer, scene_rdl2::fb_util::VariablePixelBuffer::FLOAT);
        copyAndTransform(&destBuffer->getFloatBuffer(), weightBuffer,
                         [](float &dst, const float &src) {
                             dst = src;
                         }, parallel);
        break;

    default:
        MNRY_ASSERT(0 && "needless finishing of render output");
    }
}

std::string
RenderOutputDriver::Impl::defBaseChannelName(const scene_rdl2::rdl2::RenderOutput& ro)
{
    scene_rdl2::rdl2::RenderOutput::Result res = ro.getResult();
    switch (res) {
        case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY:
        case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX:
        case scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER:
            return "";
        case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA:
            return "A";
        case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX:
            return "Aaux";
        case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP:
            return "heat";
        case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT:
            return "weight";
        case scene_rdl2::rdl2::RenderOutput::RESULT_WIREFRAME:
            return "wireframe";
        case scene_rdl2::rdl2::RenderOutput::RESULT_STATE_VARIABLE:
        case scene_rdl2::rdl2::RenderOutput::RESULT_DEPTH:
            return defBaseChannelName(res == scene_rdl2::rdl2::RenderOutput::RESULT_DEPTH ?
                                      scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DEPTH :
                                      ro.getStateVariable());
        case scene_rdl2::rdl2::RenderOutput::RESULT_PRIMITIVE_ATTRIBUTE:
            return ro.getPrimitiveAttribute();
        case scene_rdl2::rdl2::RenderOutput::RESULT_MATERIAL_AOV:
            return ro.getMaterialAov();
        case scene_rdl2::rdl2::RenderOutput::RESULT_LIGHT_AOV:
            return ro.getLpe();
        case scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV:
            return "visiblility_" + ro.getVisibilityAov();
        case scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE:
            return "cryptomatte";
        default:
            MNRY_ASSERT(0 && "unknown result type");
            return "unknown";
    }
}

pbr::AovStorageType
RenderOutputDriver::Impl::getStorageType(const scene_rdl2::rdl2::RenderOutput *ro)
{
    // TODO: figure out if lpePrefixFlags should be returned by this function.
    int lpePrefixFlags = pbr::AovSchema::sLpePrefixNone;
    pbr::AovSchemaId stateAovId = pbr::AovSchemaId::AOV_SCHEMA_ID_UNKNOWN;
    const int aovSchemaId = getAovSchemaID(ro, lpePrefixFlags, stateAovId);
    pbr::AovFilter filter = getAovFilter(ro, aovSchemaId, nullptr);
    return getStorageType(ro, aovSchemaId, filter);
}

pbr::AovStorageType
RenderOutputDriver::Impl::getStorageType(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId, pbr::AovFilter filter)
{
    // When using a closest filter, we must store our results in aligned Float4 types.
    // This allows for lock-free frame buffer updating in bundled mode.

    switch (ro->getResult()) {
        case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY:
        case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX:
        case scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER:
            return pbr::AovStorageType::RGB;
        case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA:
        case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX:
        case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP:
        case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT:
        case scene_rdl2::rdl2::RenderOutput::RESULT_WIREFRAME:
            return pbr::AovStorageType::FLOAT;
        case scene_rdl2::rdl2::RenderOutput::RESULT_STATE_VARIABLE:
        case scene_rdl2::rdl2::RenderOutput::RESULT_DEPTH:
        {
            switch (pbr::aovNumChannels(aovSchemaId)) {
                case 1:
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::FLOAT;
                case 2:
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::VEC2;
                case 3:
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::VEC3;
                default:
                    MNRY_ASSERT(0 && "unexpected schema size");
                    break;
            }
        }
            break;
        case scene_rdl2::rdl2::RenderOutput::RESULT_PRIMITIVE_ATTRIBUTE:
        {
            switch (pbr::aovNumChannels(aovSchemaId)) {
                case 1:
                    // float
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::FLOAT;
                case 2:
                    // all are vec2f
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::VEC2;
                case 3:
                    // could be vec3f or rgb
                    switch (ro->getPrimitiveAttributeType()) {
                        case scene_rdl2::rdl2::RenderOutput::PRIMITIVE_ATTRIBUTE_TYPE_RGB:
                            return filter == pbr::AOV_FILTER_CLOSEST ?
                                pbr::AovStorageType::RGB4 :
                                pbr::AovStorageType::RGB;
                        case scene_rdl2::rdl2::RenderOutput::PRIMITIVE_ATTRIBUTE_TYPE_VEC3F:
                            return filter == pbr::AOV_FILTER_CLOSEST ?
                                pbr::AovStorageType::VEC4 :
                                pbr::AovStorageType::VEC3;
                        default:
                            MNRY_ASSERT(0 && "unsupported primitive attribute type");
                            break;
                    }
                default:
                    MNRY_ASSERT(0 && "unsupported primitive attribute type");
                    break;
            }
        }
            break;
        case scene_rdl2::rdl2::RenderOutput::RESULT_MATERIAL_AOV:
        {
            switch (pbr::aovNumChannels(aovSchemaId)) {
                case 1:
                    // float
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::FLOAT;
                case 2:
                    // all are vec2f
                    return filter == pbr::AOV_FILTER_CLOSEST ?
                        pbr::AovStorageType::VEC4 :
                        pbr::AovStorageType::VEC2;
                case 3:
                    // could be vec3f or rgb
                    switch (pbr::aovToRangeTypeSchemaId(aovSchemaId)) {
                        case pbr::AOV_SCHEMA_ID_MATERIAL_AOV_RGB:
                            return filter == pbr::AOV_FILTER_CLOSEST ?
                                pbr::AovStorageType::RGB4 :
                                pbr::AovStorageType::RGB;
                        case pbr::AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F:
                            return filter == pbr::AOV_FILTER_CLOSEST ?
                                pbr::AovStorageType::VEC4 :
                                pbr::AovStorageType::VEC3;
                        default:
                            MNRY_ASSERT(0 && "unsupported primitive attribute type");
                            break;
                    }
                default:
                    MNRY_ASSERT(0 && "unsupported primitive attribute type");
                    break;
            }
        }
            break;
        case scene_rdl2::rdl2::RenderOutput::RESULT_LIGHT_AOV:
            return filter == pbr::AOV_FILTER_CLOSEST ? 
                pbr::AovStorageType::RGB4 : 
                pbr::AovStorageType::RGB;
            break;
        case scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV:
            return pbr::AovStorageType::VISIBILITY;
            break;
        default:
            MNRY_ASSERT(0 && "unknown result type");
            break;
    }
    MNRY_ASSERT(0 && "unknown result type");
    return pbr::AovStorageType::FLOAT;
}

pbr::AovOutputType
RenderOutputDriver::Impl::getOutputType(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId)
{
    const pbr::AovFilter filter = getAovFilter(ro, aovSchemaId, nullptr);
    const pbr::AovStorageType storageType = getStorageType(ro, aovSchemaId, filter);
    MNRY_ASSERT(filter != pbr::AOV_FILTER_CLOSEST ||
               storageType == pbr::AovStorageType::VEC4 ||
               storageType == pbr::AovStorageType::RGB4);

    switch (storageType) {
    case pbr::AovStorageType::UNSPECIFIED:
        MNRY_ASSERT(0, "Unable to determine output type from unspecifed storage type.");
        return pbr::AovOutputType::FLOAT;
    case pbr::AovStorageType::VEC2:
        return pbr::AovOutputType::VEC2;
    case pbr::AovStorageType::VEC3:
        return pbr::AovOutputType::VEC3;
    case pbr::AovStorageType::VEC4:
        // Currently, the only VEC4 types are FLOAT, VEC2, or VEC3 types
        // that use closest filtering
        MNRY_ASSERT(filter == pbr::AOV_FILTER_CLOSEST);
        switch (pbr::aovNumChannels(aovSchemaId)) {
        case 1:
            return pbr::AovOutputType::FLOAT;
        case 2:
            return pbr::AovOutputType::VEC2;
        case 3:
            return pbr::AovOutputType::VEC3;
        default:
            MNRY_ASSERT(0 && "unexpected schema size");
            return pbr::AovOutputType::FLOAT;
        }
    case pbr::AovStorageType::RGB:
        return pbr::AovOutputType::RGB;
    case pbr::AovStorageType::RGB4:
        // Currently, the only RGB4 storage types are RGB output
        // types that use closest filtering.  We'll need to update
        // this assert if the situation changes.
        MNRY_ASSERT(filter == pbr::AOV_FILTER_CLOSEST);
        return pbr::AovOutputType::RGB;
    case pbr::AovStorageType::FLOAT:
    case pbr::AovStorageType::VISIBILITY:
        return pbr::AovOutputType::FLOAT;
    default:
        MNRY_ASSERT(0, "Unhandled storage type");
        return pbr::AovOutputType::FLOAT;
    }
}

void
RenderOutputDriver::Impl::getChannelNames(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId,
                                          std::vector<std::string> &chanNames)
{
    std::string baseName = ro->getChannelName();
    if (baseName.empty()) {
        baseName = defBaseChannelName(*ro);
    }

    if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY && baseName.empty()) {
        chanNames.push_back("R");
        chanNames.push_back("G");
        chanNames.push_back("B");
        return;
    }

    if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX && baseName.empty()) {
        chanNames.push_back("__beautyAux__.R");
        chanNames.push_back("__beautyAux__.G");
        chanNames.push_back("__beautyAux__.B");
        return;
    }

    if (mResumableOutput) {
        // Setup channel name for Visibility AOV fulldump mode
        if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV) {
            getChannelNamesVisibilityFulldump(baseName, chanNames);
            return;
        }
    }

    if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE) {
        char channelName[80]; // 80 chars more than enough to construct filenames below
        int numLayers = ro->getCryptomatteNumLayers();
        int numFragments = numLayers * 2;
        
        MNRY_ASSERT_REQUIRE(numLayers < 100);
        for (int layer = 0; layer < numLayers; layer++) {
            sprintf(channelName, "Cryptomatte%02d.R", layer);
            chanNames.push_back(channelName);
            sprintf(channelName, "Cryptomatte%02d.G", layer);
            chanNames.push_back(channelName);
            sprintf(channelName, "Cryptomatte%02d.B", layer);
            chanNames.push_back(channelName);
            sprintf(channelName, "Cryptomatte%02d.A", layer);
            chanNames.push_back(channelName);
        }
        // position/normal/beauty data is defined per fragment, not per layer
        for (int fragment = 0; fragment < numFragments; ++fragment) {
            if (ro->getCryptomatteOutputPositions()) {
                sprintf(channelName, "CryptoP%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoP%02d.G", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoP%02d.B", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoP%02d.A", fragment);
                chanNames.push_back(channelName);
            }
            if (ro->getCryptomatteOutputNormals()) {
                sprintf(channelName, "CryptoN%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoN%02d.G", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoN%02d.B", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoN%02d.A", fragment);
                chanNames.push_back(channelName);
            }
            if (ro->getCryptomatteOutputBeauty()) {
                sprintf(channelName, "CryptoB%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoB%02d.G", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoB%02d.B", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoB%02d.A", fragment);
                chanNames.push_back(channelName);
            }
            if (ro->getCryptomatteOutputRefP()) {
                sprintf(channelName, "CryptoRefP%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoRefP%02d.G", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoRefP%02d.B", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoRefP%02d.A", fragment);
                chanNames.push_back(channelName);
            }
            if (ro->getCryptomatteOutputRefN()) {
                sprintf(channelName, "CryptoRefN%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoRefN%02d.G", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoRefN%02d.B", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoRefN%02d.A", fragment);
                chanNames.push_back(channelName);
            }
            if (ro->getCryptomatteOutputUV()) {
                sprintf(channelName, "CryptoUV%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoUV%02d.G", fragment);
                chanNames.push_back(channelName);
            }
            if (ro->getCryptomatteSupportResumeRender()) {
                sprintf(channelName, "CryptoS%02d.R", fragment);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptoS%02d.G", fragment);
                chanNames.push_back(channelName);
            }
        }

        if (ro->getCryptomatteEnableRefract()) {
            for (int layer = 0; layer < numLayers; layer++) {
                sprintf(channelName, "CryptomatteRefract%02d.R", layer);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptomatteRefract%02d.G", layer);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptomatteRefract%02d.B", layer);
                chanNames.push_back(channelName);
                sprintf(channelName, "CryptomatteRefract%02d.A", layer);
                chanNames.push_back(channelName);
            }
            // position/normal/beauty data is defined per fragment, not per layer
            for (int fragment = 0; fragment < numFragments; ++fragment) {
                if (ro->getCryptomatteOutputPositions()) {
                    sprintf(channelName, "CryptoRefractP%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractP%02d.G", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractP%02d.B", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractP%02d.A", fragment);
                    chanNames.push_back(channelName);
                }
                if (ro->getCryptomatteOutputNormals()) {
                    sprintf(channelName, "CryptoRefractN%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractN%02d.G", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractN%02d.B", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractN%02d.A", fragment);
                    chanNames.push_back(channelName);
                }
                if (ro->getCryptomatteOutputBeauty()) {
                    sprintf(channelName, "CryptoRefractB%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractB%02d.G", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractB%02d.B", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractB%02d.A", fragment);
                    chanNames.push_back(channelName);
                }
                if (ro->getCryptomatteOutputRefP()) {
                    sprintf(channelName, "CryptoRefractRefP%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractRefP%02d.G", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractRefP%02d.B", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractRefP%02d.A", fragment);
                    chanNames.push_back(channelName);
                }
                if (ro->getCryptomatteOutputRefN()) {
                    sprintf(channelName, "CryptoRefractRefN%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractRefN%02d.G", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractRefN%02d.B", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractRefN%02d.A", fragment);
                    chanNames.push_back(channelName);
                }
                if (ro->getCryptomatteOutputUV()) {
                    sprintf(channelName, "CryptoRefractUV%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractUV%02d.G", fragment);
                    chanNames.push_back(channelName);
                }
                if (ro->getCryptomatteSupportResumeRender()) {
                    sprintf(channelName, "CryptoRefractS%02d.R", fragment);
                    chanNames.push_back(channelName);
                    sprintf(channelName, "CryptoRefractS%02d.G", fragment);
                    chanNames.push_back(channelName);
                }
            }
        }
        return;
    }

    scene_rdl2::rdl2::RenderOutput::SuffixMode m = ro->getChannelSuffixMode();
    const char *rgb[] = { ".R", ".G", ".B" };
    const char *xyz[] = { ".X", ".Y", ".Z" };
    const char *uvw[] = { ".U", ".V", ".W" };
    const char * const *suffix =
        (m == scene_rdl2::rdl2::RenderOutput::SUFFIX_MODE_RGB) ? rgb :
        (m == scene_rdl2::rdl2::RenderOutput::SUFFIX_MODE_XYZ) ? xyz :
        (m == scene_rdl2::rdl2::RenderOutput::SUFFIX_MODE_UVW) ? uvw :
        /* m == SUFFIX_MODE_AUTO */ nullptr;

    switch (getOutputType(ro, aovSchemaId)) {
    case pbr::AovOutputType::FLOAT:
        // MOONRAY-4077
        // add .vis for visibility aov to deal with how it shows up on nuke. 
        // if no extension is provided then nuke displays is as "<baseName>.unnamed"
        // rest of result types gets no extension
        chanNames.push_back(baseName + 
            (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV ? ".vis" : ""));
        if (mResumableOutput &&
            ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MathFilter::MATH_FILTER_CLOSEST) {
            chanNames.push_back(baseName + ".depth");
        }
        break;
    case pbr::AovOutputType::VEC2:
        chanNames.push_back(baseName + (suffix ? suffix[0] : ".X"));
        chanNames.push_back(baseName + (suffix ? suffix[1] : ".Y"));
        if (mResumableOutput &&
            ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MathFilter::MATH_FILTER_CLOSEST) {
            chanNames.push_back(baseName + ".depth");
        }
        break;
    case pbr::AovOutputType::VEC3:
        chanNames.push_back(baseName + (suffix ? suffix[0] : ".X"));
        chanNames.push_back(baseName + (suffix ? suffix[1] : ".Y"));
        chanNames.push_back(baseName + (suffix ? suffix[2] : ".Z"));
        if (mResumableOutput &&
            ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MathFilter::MATH_FILTER_CLOSEST) {
            chanNames.push_back(baseName + ".depth");
        }
        break;
    case pbr::AovOutputType::RGB:
        chanNames.push_back(baseName + (suffix ? suffix[0] : ".R"));
        chanNames.push_back(baseName + (suffix ? suffix[1] : ".G"));
        chanNames.push_back(baseName + (suffix ? suffix[2] : ".B"));
        if (mResumableOutput &&
            ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MathFilter::MATH_FILTER_CLOSEST) {
            chanNames.push_back(baseName + ".depth");
        }
        break;
    default :
        break;                  // never happen
    }
}

void
RenderOutputDriver::Impl::getChannelNamesVisibilityFulldump(const std::string &baseName,
                                                            std::vector<std::string> &chanNames)
{
    chanNames.push_back(baseName + ".hit"); // hits
    chanNames.push_back(baseName + ".att"); // attempts
    // MOONRAY-4077
    // add .vis extension to deal with how it shows up on nuke. 
    // if no extension is provided then nuke displays is as "<baseName>.unnamed"
    chanNames.push_back(baseName + ".vis"); // visibility
}

int
RenderOutputDriver::Impl::stateVariableToSchema(scene_rdl2::rdl2::RenderOutput::StateVariable sv)
{
    int result = pbr::AOV_SCHEMA_ID_UNKNOWN;

    switch (sv) {
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_P:
        result = pbr::AOV_SCHEMA_ID_STATE_P;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_NG:
        result = pbr::AOV_SCHEMA_ID_STATE_NG;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_N:
        result = pbr::AOV_SCHEMA_ID_STATE_N;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_ST:
        result = pbr::AOV_SCHEMA_ID_STATE_ST;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DPDS:
        result = pbr::AOV_SCHEMA_ID_STATE_DPDS;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DPDT:
        result = pbr::AOV_SCHEMA_ID_STATE_DPDT;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DSDX:
        result = pbr::AOV_SCHEMA_ID_STATE_DSDX;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DSDY:
        result = pbr::AOV_SCHEMA_ID_STATE_DSDY;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DTDX:
        result = pbr::AOV_SCHEMA_ID_STATE_DTDX;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DTDY:
        result = pbr::AOV_SCHEMA_ID_STATE_DTDY;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_WP:
        result = pbr::AOV_SCHEMA_ID_STATE_WP;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DEPTH:
        result = pbr::AOV_SCHEMA_ID_STATE_DEPTH;
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_MOTION:
        result = pbr::AOV_SCHEMA_ID_STATE_MOTION;
        break;
    default:
        MNRY_ASSERT(0 && "unknown state variable specified");
    }

    return result;
}

int
RenderOutputDriver::Impl::primitiveAttributeToSchema(const std::string &primAttr,
                                                     scene_rdl2::rdl2::RenderOutput::PrimitiveAttributeType type,
                                                     shading::AttributeKeySet &primAttrs)
{
    int result = pbr::AOV_SCHEMA_ID_UNKNOWN;

    // first, get an index for this particular attribute, the only error check
    // is for an empty string.  if the attribute key doesn't exist, we add it
    // to the database.
    if (primAttr.empty()) return result;

    switch (type) {
    case scene_rdl2::rdl2::RenderOutput::PRIMITIVE_ATTRIBUTE_TYPE_RGB:
        {
            shading::TypedAttributeKey<scene_rdl2::math::Color> key(primAttr);
            primAttrs.insert(key);
            int geomIndx = key.getIndex();
            result = pbr::aovFromGeomIndex(geomIndx);
            break;
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::PRIMITIVE_ATTRIBUTE_TYPE_VEC3F:
        {
            shading::TypedAttributeKey<scene_rdl2::math::Vec3f> key(primAttr);
            primAttrs.insert(key);
            int geomIndx = key.getIndex();
            result = pbr::aovFromGeomIndex(geomIndx);
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::PRIMITIVE_ATTRIBUTE_TYPE_VEC2F:
        {
            shading::TypedAttributeKey<scene_rdl2::math::Vec2f> key(primAttr);
            primAttrs.insert(key);
            int geomIndx = key.getIndex();
            result = pbr::aovFromGeomIndex(geomIndx);
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::PRIMITIVE_ATTRIBUTE_TYPE_FLOAT:
        {
            shading::TypedAttributeKey<float> key(primAttr);
            primAttrs.insert(key);
            int geomIndx = key.getIndex();
            result = pbr::aovFromGeomIndex(geomIndx);
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown prim attr type");
    }

    return result;
}

int
RenderOutputDriver::Impl::materialAovToSchema(const scene_rdl2::rdl2::RenderOutput *ro,
                                              shading::AttributeKeySet &primAttrs,
                                              int &lpePrefixFlags,
                                              pbr::AovSchemaId &stateAovId)
{
    MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::RenderOutput::MATH_FILTER_AVG) == pbr::AOV_FILTER_AVG);
    MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::RenderOutput::MATH_FILTER_SUM) == pbr::AOV_FILTER_SUM);
    MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::RenderOutput::MATH_FILTER_MIN) == pbr::AOV_FILTER_MIN);
    MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::RenderOutput::MATH_FILTER_MAX) == pbr::AOV_FILTER_MAX);
    MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::RenderOutput::MATH_FILTER_FORCE_CONSISTENT_SAMPLING) ==
                      pbr::AOV_FILTER_FORCE_CONSISTENT_SAMPLING);
    MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::RenderOutput::MATH_FILTER_CLOSEST) == pbr::AOV_FILTER_CLOSEST);

    const std::string &materialAov = ro->getMaterialAov();
    const pbr::AovFilter filter = pbr::AovFilter(int(ro->getMathFilter()));
    int result = pbr::AOV_SCHEMA_ID_UNKNOWN;
    if (materialAov.empty()) return result;

    pbr::AovSchemaId lpeSchemaId = pbr::AOV_SCHEMA_ID_UNKNOWN;
    int lpeLabelId = -1;
    std::string lpe = ro->getLpe();
    if(!lpe.empty()) {
        // An LPE has been specified.
        // materialAov is a material expression, e.g. "D.color"
        // We need to add a label to the LPE: 'M:materialAov'
        // e.g. lpe = "C<RD>", materialAov = "D.color" -> lpe = "C<RD>'M:D.color'"
        std::string lpeLabel = "M:" + materialAov;
        lpe += "'" + lpeLabel + "'";

        // The material AOV is added to the "light" AOVs so it can use the LPE
        // state machine.
        lpeSchemaId = (pbr::AovSchemaId)lightAovToSchema(lpe, lpePrefixFlags);

        lpeLabelId = mLightAovs.getLabelId(lpeLabel);
    }

    int primAttrKey = -1;
    result = mMaterialAovs.createEntry(materialAov, filter, lpeSchemaId, lpeLabelId, stateAovId, primAttrKey);
    if (result != pbr::AOV_SCHEMA_ID_UNKNOWN) {
        if (primAttrKey >= 0) {
            primAttrs.insert(primAttrKey);
        }
        if (stateAovId == pbr::AOV_SCHEMA_ID_STATE_MOTION) {
            mRequiresMotionVector = true;
        }
    }

    return result;
}

int
RenderOutputDriver::Impl::lightAovToSchema(const std::string &lpe, int &lpePrefixFlags)
{
    int result = pbr::AOV_SCHEMA_ID_UNKNOWN;

    if (lpe.empty()) return result;

    result = mLightAovs.createEntry(lpe, false /* not visibility*/, lpePrefixFlags);

    return result;
}

int
RenderOutputDriver::Impl::lightAovToSchema(const scene_rdl2::rdl2::RenderOutput *ro,
                                           int &lpePrefixFlags)
{
    return lightAovToSchema(ro->getLpe(), lpePrefixFlags);
}

int
RenderOutputDriver::Impl::visibilityAovToSchema(const std::string &visibilityAov)
{
    int result = pbr::AOV_SCHEMA_ID_UNKNOWN;

    if (visibilityAov.empty()) return result;

    int tempFlags;
    result = mLightAovs.createEntry(visibilityAov, true /* visibility */, tempFlags);

    return result;
}

std::string
RenderOutputDriver::Impl::defBaseChannelName(scene_rdl2::rdl2::RenderOutput::StateVariable sv)
{
    std::string result;
    switch (sv) {
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_P:
        result = "P";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_NG:
        result = "Ng";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_N:
        result = "N";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_ST:
        result = "ST";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DPDS:
        result = "dPds";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DPDT:
        result = "dPdt";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DSDX:
        result = "dSdx";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DSDY:
        result = "dSdy";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DTDX:
        result = "dTdx";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DTDY:
        result = "dTdy";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_WP:
        result = "Wp";
        break;
    case scene_rdl2::rdl2::RenderOutput::STATE_VARIABLE_DEPTH:
        result = "Z";
        break;
    default:
        MNRY_ASSERT(0 && "unknown state variable specified");
    }

    return result;
}

//------------------------------------------------------------------------------

int
RenderOutputDriver::Impl::getCryptomatteDepth() const
{
    for (const auto &f: mFiles) {
        for (const auto &i: f.mImages) {
            for (const auto &e: i.mEntries) {
                const scene_rdl2::rdl2::RenderOutput *ro = e.mRenderOutput;
                if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE) {
                    return ro->getCryptomatteDepth();
                }
            }
        }
    }
    return 0;
}

void
RenderOutputDriver::Impl::runOnResumeScript(bool resumeStartStatus,
                                            const std::vector<std::string> &resumeFiles) const
{
    const std::string &onResumeScript = mRenderContext->getOnResumeScript();

    if (onResumeScript.empty()) {
        return; // skip on-resume script execution
    }

    scene_rdl2::util::LuaScriptRunner luaRun;
    try {
        setupOnResumeLuaGlobalVariables(luaRun, resumeStartStatus, resumeFiles);
        luaRun.runFile(onResumeScript);
    }
    catch (std::runtime_error &e) {
        std::ostringstream ostr;
        ostr << "on-resume LuaScriptRunner runtime ERROR {\n"
             << scene_rdl2::str_util::addIndent(e.what()) << '\n'
             << "}";
        scene_rdl2::logging::Logger::error(ostr.str());
    }
}

void
RenderOutputDriver::Impl::setupOnResumeLuaGlobalVariables(scene_rdl2::util::LuaScriptRunner &luaRun,
                                                          bool resumeStartStatus,
                                                          const std::vector<std::string> &resumeFiles) const
{
    //
    // Setup some on-resume related parameters to the LUA global variables
    //
    // RenderOptions has option "-rdla_set" and can set Lua global variable before any RDLA run
    // however this -rdla_set does not affect the on-resume script execution.
    // -rdla_set and on-resume script are independent.
    //
    luaRun.beginDictionary("resume");
    {
        luaRun.setVarBool("resumeStartStatus", resumeStartStatus);

        if (resumeStartStatus) {
            luaRun.setArrayString("resumeFiles", resumeFiles);

            luaRun.beginDictionary("resumeParam"); { // define dictionary of "resumeParam"
                luaRun.setVarInt("progressCheckpointTileSamples", mResumeTileSamples);
                luaRun.setVarInt("AovFilterNumConsistentSamples", mResumeNumConsistentSamples);

                bool adaptiveSampling = (mResumeAdaptiveSampling == 1);
                if (adaptiveSampling) {                
                    luaRun.setVarString("sampling", "adaptive");
                    std::vector<float> adaptiveSampleParam = {
                        mResumeAdaptiveParam[0],
                        mResumeAdaptiveParam[1],
                        mResumeAdaptiveParam[2] * 10000.0f // target adaptive error should be *= 10,000
                    };
                    luaRun.setArrayFloat("adaptiveSampleParameters", adaptiveSampleParam);
                } else {
                    luaRun.setVarString("sampling", "uniform");
                }
            }
            luaRun.endDictionary(); // end of dictionary of "resumeParam"
            luaRun.setDictionaryByJson("resumeHistory", mResumeHistory);
        }
    }
    luaRun.endDictionary();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

RenderOutputDriver::RenderOutputDriver(const RenderContext *renderContext)
{
    mImpl.reset(new Impl(renderContext));
}

RenderOutputDriver::~RenderOutputDriver()
{
}

void
RenderOutputDriver::setFinalMaxSamplesPerPix(unsigned v)
{
    mImpl->setFinalMaxSamplesPerPix(v);
}

unsigned int
RenderOutputDriver::getNumberOfRenderOutputs() const
{
    return mImpl->getNumberOfRenderOutputs();
}

const scene_rdl2::rdl2::RenderOutput *
RenderOutputDriver::getRenderOutput(unsigned int indx) const
{
    return mImpl->getRenderOutput(indx);
}

int
RenderOutputDriver::getRenderOutputIndx(const scene_rdl2::rdl2::RenderOutput *ro) const
{
    return mImpl->getRenderOutputIndx(ro);
}

unsigned int
RenderOutputDriver::getNumberOfChannels(unsigned int indx) const
{
    return mImpl->getNumberOfChannels(indx);
}

int
RenderOutputDriver::getAovBuffer(unsigned int indx) const
{
    return mImpl->getAovBuffer(indx);
}

int
RenderOutputDriver::getDisplayFilterIndex(unsigned int indx) const
{
    return mImpl->getDisplayFilterIndex(indx);
}

bool
RenderOutputDriver::isVisibilityAov(unsigned int indx) const
{
    return mImpl->isVisibilityAov(indx);
}

const shading::AttributeKeySet &
RenderOutputDriver::getPrimAttrs() const
{
    return mImpl->getPrimAttrs();
}

const pbr::AovSchema &
RenderOutputDriver::getAovSchema() const
{
    return mImpl->mAovSchema;
}

float
RenderOutputDriver::getAovDefaultValue(const unsigned int indx) const
{
    return mImpl->getAovDefaultValue(indx);
}

bool
RenderOutputDriver::requiresScaledByWeight(const unsigned indx) const
{
    return mImpl->requiresScaledByWeight(indx);
}

bool
RenderOutputDriver::requiresDeepBuffer() const
{
    return mImpl->requiresDeepBuffer();
}

bool
RenderOutputDriver::requiresDeepBuffer(unsigned int indx) const
{
    return mImpl->requiresDeepBuffer(indx);
}

bool
RenderOutputDriver::requiresCryptomatteBuffer() const
{
    return mImpl->requiresCryptomatteBuffer();
}

bool
RenderOutputDriver::requiresCryptomatteBuffer(unsigned int indx) const
{
    return mImpl->requiresCryptomatteBuffer(indx);
}

bool
RenderOutputDriver::requiresRenderBuffer() const
{
    return mImpl->requiresRenderBuffer();
}

bool
RenderOutputDriver::requiresRenderBuffer(unsigned int indx) const
{
    return mImpl->requiresRenderBuffer(indx);
}

bool
RenderOutputDriver::requiresRenderBufferOdd() const
{
    return mImpl->requiresRenderBufferOdd();
}

bool
RenderOutputDriver::requiresRenderBufferOdd(unsigned int indx) const
{
    return mImpl->requiresRenderBufferOdd(indx);
}

const pbr::MaterialAovs &
RenderOutputDriver::getMaterialAovs() const
{
    return mImpl->mMaterialAovs;
}

const pbr::LightAovs &
RenderOutputDriver::getLightAovs() const
{
    return mImpl->mLightAovs;
}

bool
RenderOutputDriver::requiresHeatMap() const
{
    return mImpl->requiresHeatMap();
}

bool
RenderOutputDriver::requiresHeatMap(unsigned int indx) const
{
    return mImpl->requiresHeatMap(indx);
}

bool
RenderOutputDriver::requiresWeightBuffer() const
{
    return mImpl->requiresWeightBuffer();
}

bool
RenderOutputDriver::requiresWeightBuffer(unsigned int indx) const
{
    return mImpl->requiresWeightBuffer(indx);
}

bool
RenderOutputDriver::requiresDisplayFilter() const
{
    return mImpl->requiresDisplayFilter();
}

bool
RenderOutputDriver::requiresDisplayFilter(unsigned int indx) const
{
    return mImpl->requiresDisplayFilter(indx);
}

unsigned int
RenderOutputDriver::getDisplayFilterCount() const
{
    return mImpl->getDisplayFilterCount();
}

bool
RenderOutputDriver::requiresWireframe() const
{
    return mImpl->requiresWireframe();
}

bool
RenderOutputDriver::requiresMotionVector() const
{
    return mImpl->requiresMotionVector();
}

const std::vector<std::string> &
RenderOutputDriver::getErrors() const
{
    return mImpl->mErrors;
}

void
RenderOutputDriver::resetErrors() const
{
    mImpl->mErrors.clear();
}

const std::vector<std::string> &
RenderOutputDriver::getInfos() const
{
    return mImpl->mInfos;
}

void
RenderOutputDriver::resetInfos() const
{
    mImpl->mInfos.clear();
}

bool
RenderOutputDriver::loggingErrorAndInfo(ImageWriteCache *cache) const
{
    return mImpl->loggingErrorAndInfo(cache);
}

void
RenderOutputDriver::finishSnapshot(scene_rdl2::fb_util::VariablePixelBuffer *destBuffer, unsigned int indx,
                                   const scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                                   const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                                   const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                   const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                   bool parallel) const
{
    return mImpl->finishSnapshot(destBuffer, indx,
                                 renderBuffer, heatMapBuffer, weightBuffer, renderBufferOdd,
                                 parallel);
}

int
RenderOutputDriver::getDenoiserAlbedoInput() const
{
    return mImpl->getDenoiserAlbedoInput();
}

int
RenderOutputDriver::getDenoiserNormalInput() const
{
    return mImpl->getDenoiserNormalInput();
}

bool
RenderOutputDriver::revertFilmData(Film &film,
                                   unsigned &resumeTileSamples, int &resumeNumConsistentSamples, bool &zeroWeightMask,
                                   bool &adaptiveSampling, float adaptiveSampleParam[3])
{
    return mImpl->revertFilmData(film,
                                 resumeTileSamples, resumeNumConsistentSamples, zeroWeightMask,
                                 adaptiveSampling, adaptiveSampleParam);
}

void
RenderOutputDriver::setLastCheckpointRenderTileSamples(const unsigned samples)
{
    mImpl->setLastCheckpointRenderTileSamples(samples);
}

scene_rdl2::grid_util::Parser&
RenderOutputDriver::getParser()
{
    return mImpl->getParser();
}

} // namespace rndr
} // namespace moonray

