// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "OiioReader.h"
#include "RenderOutputDriver.h"
#include "RenderOutputWriter.h"

#include <moonray/rendering/pbr/core/Aov.h>
#include <scene_rdl2/common/grid_util/Sha1Util.h>
#include <scene_rdl2/render/util/LuaScriptRunner.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>

// Useful debug dump to trackdown all entry items and file info of renderOutputDriver
//#define DEBUG_DUMP_ENTRIES_AND_FILES

namespace moonray {
namespace rndr {

class ImageWriteCacheImageSpec;

// files contain 1 or more (sub-)images (parts).  the
// case of 1 un-named image per file is allowed, and is typically
// thought of as the "normal" case.  if more than one image
// is present in a file, all sub-images must be named.  a particular
// image may have only 1 compression scheme.  only a single render output
// object may target a particular file/image/channel tripple.

// an Entry is basically a render output that has been processed
// to include resolved channel names and aov schema
struct Entry
{
    explicit Entry(const scene_rdl2::rdl2::RenderOutput *renderOutput):
        mRenderOutput(renderOutput),
        mAovSchemaId(pbr::AOV_SCHEMA_ID_UNKNOWN),
        mLpePrefixFlags(pbr::AovSchema::sLpePrefixNone),
        mStateAovId(0)
    {
    }

#ifdef DEBUG_DUMP_ENTRIES_AND_FILES
    std::string showChannelNames(const std::string &hd) const { // for debug
        std::ostringstream ostr;
        ostr << hd << "Entry mChannelNames (total:" << mChannelNames.size() << ") {\n";
        for (size_t i = 0; i < mChannelNames.size(); ++i) {
            ostr << hd << "  i:" << i << " name:" << mChannelNames[i] << '\n';
        }
        ostr << hd << "}";
        return ostr.str();
    }
#endif // end DEBUG_DUMP_ENTRIES_AND_FILES

    const scene_rdl2::rdl2::RenderOutput *mRenderOutput;
    std::vector<std::string> mChannelNames; // resolved channel names
    int mAovSchemaId;                       // aov schema id, if any
    int mLpePrefixFlags;                    // lpe prefix flags, if any
    int mStateAovId;
};

struct Image
{
    explicit Image(const std::string &name): mName(name), mStartRoIdx(0) {}

    std::string mName; // may be empty if only 1 image per file

    // Entries in the Image
    std::vector<Entry> mEntries;

    int mStartRoIdx; // index of RenderOutputDriver::Impl::mEntries[] of this->mEntries[0]
};

struct File
{
    explicit File(const std::string &name) :
        mName(name),
        mCheckpointName(""),
        mCheckpointMultiVersionName(""),
        mResumeName("")
    {}

    // realpath() of output file
    std::string mName;

    // realpath() of checkpoint output file : If empty, should not output checkpoint file.
    std::string mCheckpointName;
    
    // realpath() of checkpoint multi-version output file. This is used for checkpoint file
    // overwrite = off condition. If this value is empty, all non-overwrite checkpoint files
    // go to same location of mCheckpointName
    std::string mCheckpointMultiVersionName;
    
    // realpath() of resume input file : If empty, should not input resume file.
    std::string mResumeName;

    std::vector<Image> mImages; // images in file
};

//------------------------------------------------------------------------------------------

class RenderOutputDriver::Impl
{
public:
    using Parser = scene_rdl2::grid_util::Parser;

    Impl(const RenderContext *renderContext);

    void setFinalMaxSamplesPerPix(unsigned v) { mFinalMaxSamplesPerPixel = v; }

    unsigned int getNumberOfRenderOutputs() const;
    const scene_rdl2::rdl2::RenderOutput *getRenderOutput(unsigned int indx) const;
    int getRenderOutputIndx(const scene_rdl2::rdl2::RenderOutput *ro) const;
    unsigned int getNumberOfChannels(unsigned int indx) const;
    int getAovBuffer(unsigned int indx) const;
    int getDisplayFilterIndex(unsigned int indx) const;
    float getAovDefaultValue(unsigned int indx) const;
    bool requiresScaledByWeight(unsigned int indx) const;
    bool isVisibilityAov(unsigned int indx) const;
    unsigned int getDisplayFilterCount() const;
    bool getOverwriteCheckpoint() const { return mOverwriteCheckpoint; }

    const shading::AttributeKeySet &getPrimAttrs() const;

    bool requiresDeepBuffer() const;
    bool requiresDeepBuffer(unsigned int indx) const;
    bool requiresCryptomatteBuffer() const;
    bool requiresCryptomatteBuffer(unsigned int indx) const;
    bool requiresRenderBuffer() const;
    bool requiresRenderBuffer(unsigned int indx) const;
    bool requiresRenderBufferOdd() const;
    bool requiresRenderBufferOdd(unsigned int indx) const;
    bool requiresHeatMap() const;
    bool requiresHeatMap(unsigned int indx) const;
    bool requiresWeightBuffer() const;
    bool requiresWeightBuffer(unsigned int indx) const;
    bool requiresDisplayFilter() const;
    bool requiresDisplayFilter(unsigned int indx) const;
    bool requiresWireframe() const;
    bool requiresMotionVector() const;

    void incrementAovs(const File& file,
                       const scene_rdl2::fb_util::VariablePixelBuffer *&aovs) const;
    void writeDeq(ImageWriteCache *cache,
                  const bool checkpointOutputMultiVersion,
                  scene_rdl2::grid_util::Sha1Gen::Hash *hashOut) const;
    void write(const bool checkpointOutput,
               const bool checkpointOutputMultiVersion,
               const pbr::DeepBuffer *deepBuffer,
               pbr::CryptomatteBuffer *cryptomatteBuffer,
               const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
               const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
               const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> *aovBuffers,
               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> *displayFilterBuffers,
               const unsigned checkpointTileSampleTotals,
               ImageWriteCache *cache,
               scene_rdl2::grid_util::Sha1Gen::Hash *hashOut) const;
    void deepWrite(const bool checkpointOutput,
                   const bool checkpointOutputMultiVersion,
                   const unsigned checkpointTileSampleTotals,
                   const scene_rdl2::fb_util::PixelBuffer<unsigned> &samplesCount,
                   const pbr::DeepBuffer *deepBuffer) const;

    bool loggingErrorAndInfo(ImageWriteCache *cache) const;

    void finishSnapshot(scene_rdl2::fb_util::VariablePixelBuffer *destBuffer, unsigned int indx,
                        const scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                        const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                        const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                        const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                        bool parallel) const;

    int getDenoiserAlbedoInput() const
    {
        return mDenoiserAlbedoInput;
    }

    int getDenoiserNormalInput() const
    {
        return mDenoiserNormalInput;
    }

    bool revertFilmData(Film &film,
                        unsigned &resumeTileSamples, int &resumeNumConsistentSamples, bool &zeroWeightMask,
                        bool &adaptiveSampling, float adaptiveSampleParam[3]);

    void setLastCheckpointRenderTileSamples(const unsigned samples) { mLastTileSamples = samples; }

    std::string showWriteInfo(const int width, const int height) const;

    Parser& getParser() { return mParser; }

private:
    using PageAlignedBuff = RenderOutputWriter::PageAlignedBuff;

    friend RenderOutputDriver;

    int getAovSchemaID(const scene_rdl2::rdl2::RenderOutput* ro, int& lpePrefixFlags, pbr::AovSchemaId &stateAovId);
    pbr::AovFilter getAovFilter(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId, std::string *error) const;
    pbr::AovOutputType getOutputType(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId);
    pbr::AovStorageType getStorageType(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId, pbr::AovFilter filter);
    pbr::AovStorageType getStorageType(const scene_rdl2::rdl2::RenderOutput *ro);
    void getChannelNames(const scene_rdl2::rdl2::RenderOutput *ro, int aovSchemaId,
                         std::vector<std::string> &chanNames);
    void getChannelNamesVisibilityFulldump(const std::string &baseName,
                                           std::vector<std::string> &chanNames);
    std::string generateCheckpointMultiVersionFilename(const File &f,
                                                       unsigned checkpointTileSampleTotals) const;
    std::vector<std::string> writeCheckpointResumeMetadata(const unsigned checkpointTileSampleTotals) const;
                              
    static int stateVariableToSchema(scene_rdl2::rdl2::RenderOutput::StateVariable sv);
    static int primitiveAttributeToSchema(const std::string &primAttr,
                                          scene_rdl2::rdl2::RenderOutput::PrimitiveAttributeType type,
                                          shading::AttributeKeySet &primAttrs);

    static std::string defBaseChannelName(scene_rdl2::rdl2::RenderOutput::StateVariable sv);
    static std::string defBaseChannelName(const scene_rdl2::rdl2::RenderOutput& ro);

    int materialAovToSchema(const scene_rdl2::rdl2::RenderOutput *ro,
                            shading::AttributeKeySet &primAttrs,
                            int &lpePrefixFlags,
                            pbr::AovSchemaId &stateAovId);
    int lightAovToSchema(const std::string &lpe, int &lpePrefixFlags);
    int lightAovToSchema(const scene_rdl2::rdl2::RenderOutput *ro, int &lpePrefixFlags);
    int visibilityAovToSchema(const std::string &visibilityAov);

    bool resumeRenderReadyTest() const;
    bool read(const File &file, Film &film);
    bool readResumableParameters(OiioReader &reader);
    bool readSubImage(OiioReader &reader, const Image &currImage, Film &film);
    bool readSubImageNameValidityTest(OiioReader &reader, const Image &currImage) const;
    bool readSubImageOneEntry(OiioReader &reader, const int roIdx, Film &film);
    bool readSubImageSetDestinationBuffer(const int roIdx,
                                          Film &film,
                                          scene_rdl2::fb_util::HeatMapBuffer **heatMapBuffer,
                                          scene_rdl2::fb_util::FloatBuffer **weightBuffer,
                                          scene_rdl2::fb_util::RenderBuffer **renderBufferOdd,
                                          scene_rdl2::fb_util::VariablePixelBuffer **aovBuffer) const;
    bool readSubImageResoValidityTest(OiioReader &reader,
                                      const Entry &currEntry,
                                      const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                                      const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                      const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                      const scene_rdl2::fb_util::VariablePixelBuffer *aovBuffer,
                                      const pbr::CryptomatteBuffer *cryptomatteBuffer) const;
    bool readSubImageSetPixChanOffset(OiioReader &reader, const Entry &currEntry, std::vector<int> &chanOffset) const;

    int getCryptomatteDepth() const;

    void runOnResumeScript(bool resumeStartStatus, const std::vector<std::string> &resumeFiles) const;
    void setupOnResumeLuaGlobalVariables(scene_rdl2::util::LuaScriptRunner &luaRun, bool resumeStartStatus,
                                         const std::vector<std::string> &resumeFiles) const;

    void parserConfigure();

    std::string showDenoiseInfo() const;

    //------------------------------

    std::vector<const Entry *>        mEntries;                 // our entry order
    std::vector<File>                 mFiles;                   // resolved output files
    mutable std::vector<std::string>  mErrors;                  // error messages
    mutable std::vector<std::string>  mInfos;                   // info messages
    pbr::AovSchema                    mAovSchema;               // aov schema codes
    std::vector<int>                  mAovBuffers;              // indx in aov buffer
    std::vector<int>                  mDisplayFilterIndices;    // indx in DisplayFilter buffer
    pbr::MaterialAovs                 mMaterialAovs;            // material aov management
    pbr::LightAovs                    mLightAovs;               // light aov management
    shading::AttributeKeySet          mPrimAttrs;               // needed primitive attributes
    int                               mDenoiserAlbedoInput;     // mEntries[] index of albedo input
    int                               mDenoiserNormalInput;     // mEntries[] index of normal input
    bool                              mRequiresMotionVector;

    bool mCheckpointRenderActive;   // flag of checkpoint rendering on or off
    bool mResumableOutput;          // flag to output resumable Visibility/ClosestFilter AOV buffer
                                    // This flag is not indicate that this process is a resume render.
    unsigned mFinalMaxSamplesPerPixel;
    int mResumeTileSamples;         // tile samples for resume render which is read from resume file
    int mResumeNumConsistentSamples; // numConsistentSamples for resume render which is read from resume file
    int mResumeAdaptiveSampling;    // use adaptive sampling (-1:init 0:false 1:true), read from resume file
    float mResumeAdaptiveParam[3];  // adaptive sampling parameters which are read from resume file
    std::string mResumeHistory;     // all resume render history information from resume file
    bool mRevertWeightAOV;          // condition of reverted weight AOV buffer for revertFilmData()
    bool mZeroWeightMask;           // condition of required zero weight mask operation for all AOV.
    bool mRevertBeautyAuxAOV;       // condition of reverted beauty aux AOV buffer for revertFilmData()

    bool mOverwriteCheckpoint;   // only used by checkpoint write.

    unsigned mLastTileSamples;  // last render tile samples at render finish under checkpoint render mode

    const RenderContext *mRenderContext;

    Parser mParser;
};

} // namespace rndr
} // namespace moonray

