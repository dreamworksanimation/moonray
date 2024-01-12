// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImageWriteCache.h"

#include <scene_rdl2/render/util/AlignedAllocator.h>

#include <OpenImageIO/imagebuf.h>

namespace scene_rdl2 {
namespace fb_util {
    class VariablePixelBuffer;
} // namespace fb_util
namespace grid_util {
    class Sha1Gen;
} // namespace grid_util
namespace rdl2 {
    class RenderOutput;
} // namespace rdl2
} // namespace scene_rdl2

namespace moonray {

namespace pbr {
    class CryptomatteBuffer;
} // namespace pbr

namespace rndr {

struct Entry;
struct File;
struct Image;

struct FileNameParam
//
// This class is used in order to specify output filename-related information to the RenderOutputWriter.
// There are 2 different types of constructors based on the runMode.
// You should select one of them depending on the runMode.
//
{
    // constructor for STD/ENQ operation
    FileNameParam(const bool checkpointOutput,
                  const bool checkpointOutputMultiVersion,
                  const bool overwriteCheckpoint,
                  const unsigned finalMaxSamplesPerPixel)
        : mCheckpointOutput(checkpointOutput)
        , mCheckpointOutputMultiVersion(checkpointOutputMultiVersion)
        , mOverwriteCheckpoint(overwriteCheckpoint)
        , mFinalMaxSamplesPerPixel(finalMaxSamplesPerPixel)
    {}

    // constructor for DEQ operation
    explicit FileNameParam(const bool checkpointOutputMultiVersion)
        : mCheckpointOutput(false) // not used
        , mCheckpointOutputMultiVersion(checkpointOutputMultiVersion)
        , mOverwriteCheckpoint(false) // not used
        , mFinalMaxSamplesPerPixel(0) // not used
    {}

    std::string show() const;

    const bool mCheckpointOutput;
    const bool mCheckpointOutputMultiVersion;
    const bool mOverwriteCheckpoint;
    const unsigned mFinalMaxSamplesPerPixel;
};

class RenderOutputWriter
//
// This class is used for writing RenderOutputData.
// This class executes 3 different runMode (STD, ENQ, and DEQ) internally and has 2 different types of
// constructors based on the runMode. You should select one of them depending on the runMode.
//
// runMode STD : standard run mode. output files are immediately written out inside main() call.
// runMode ENQ : enqueue data into the cache. ImageWriteCache is created instead of written out to the disk.
//               The actual file write action is postponed later and executed by the imageWriteDriver thread.
// runMode DEQ : Called from imageWriteDriver and execute cache data write action.
//
{
public:
    using PageAlignedAllocator = scene_rdl2::alloc::AlignedAllocator<char, 4096>; // pageSize = 4096
    using PageAlignedBuff = std::vector<char, PageAlignedAllocator>;
    using CallBackCheckpointResumeMetadata = std::function<std::vector<std::string>(const unsigned checkpointTileSampleTotals)>;
    using CallBackCheckpointMultiVersionFilename = std::function<std::string(const File& file)>;
    using VariablePixelBuffer = scene_rdl2::fb_util::VariablePixelBuffer;

    // constructor for STD/ENQ operation
    RenderOutputWriter(const ImageWriteCache::Mode runMode,
                       ImageWriteCache* cache,
                       const unsigned checkpointTileSampleTotals,
                       const int predefinedWidth,
                       const int predefinedHeight,
                       const std::vector<File>* files,
                       const FileNameParam* fileNameParam,
                       const CallBackCheckpointResumeMetadata& callBack,
                       const pbr::CryptomatteBuffer* cryptomatteBuffer,
                       const scene_rdl2::fb_util::HeatMapBuffer* heatMap,
                       const scene_rdl2::fb_util::FloatBuffer* weightBuffer,
                       const scene_rdl2::fb_util::RenderBuffer* renderBufferOdd,
                       const std::vector<scene_rdl2::fb_util::VariablePixelBuffer>* aovBuffers,
                       const std::vector<scene_rdl2::fb_util::VariablePixelBuffer>* displayFilterBuffers,
                       std::vector<std::string>& errors,
                       std::vector<std::string>& infos,
                       scene_rdl2::grid_util::Sha1Gen* sha1Gen)
        : mErrors(errors)
        , mInfos(infos)
        , mRunMode(runMode)
        , mCache(cache)
        , mFiles(files)
        , mFileNameParam(fileNameParam)
        , mCallBackCheckpointResumeMetadata(&callBack)
        , mCryptomatteBuffer(cryptomatteBuffer)
        , mHeatMap(heatMap)
        , mWeightBuffer(weightBuffer)
        , mRenderBufferOdd(renderBufferOdd)
        , mAovBuffers(aovBuffers)
        , mDisplayFilterBuffers(displayFilterBuffers)
        , mSha1Gen(sha1Gen)
    {
        MNRY_ASSERT(dataValidityCheck(predefinedWidth, predefinedHeight));

        setupBuffOffsetTable();
        setupCheckpointTileSampleTotals(checkpointTileSampleTotals);
        setupWidthHeight(predefinedWidth, predefinedHeight);
    }

    // constructor for DEQ operation
    RenderOutputWriter(ImageWriteCache* cache,
                       const std::vector<File>* files,
                       const FileNameParam* fileNameParam,
                       std::vector<std::string>& errors,
                       std::vector<std::string>& infos,
                       scene_rdl2::grid_util::Sha1Gen* sha1Gen)
        : mErrors(errors)
        , mInfos(infos)
        , mCache(cache)
        , mFiles(files)
        , mFileNameParam(fileNameParam)
        , mSha1Gen(sha1Gen)
    {
        MNRY_ASSERT(dataValidityCheck(0, 0));

        setupBuffOffsetTable();
        setupCheckpointTileSampleTotals(0);
        setupWidthHeight(0, 0);
    }

    //
    // Return result of write action. 
    // Usually, scene includes multiple file output. This API tries to do write-out actions for all files.
    // This API supports 3 modes (STD, ENQ, and DEQ).
    // STD and DEQ execute actual write-out action.
    // ENQ only creates cache data.
    //
    // Return true when all of the writing files action succeeded.
    // Return false if some of (or all of) the writing files action failed.
    // Partial failed condition case, some of the file writing action might be succeeded without error.
    // We don't stop writing action when we hit the first error, and we will try to write the data as much
    // as possible.
    //
    bool main() const;

    static std::string generateCheckpointMultiVersionFilename(const File& file,
                                                              const bool overwriteCheckpoint,
                                                              const unsigned finalMaxSamplesPerPixel,
                                                              const unsigned checkpointTileSampleTotals);

private:

    bool dataValidityCheck(const int predefinedWidth, const int predefinedHeight) const;
    void setupBuffOffsetTable();
    void setupCheckpointTileSampleTotals(const unsigned checkpointTileSampleTotals);
    void setupWidthHeight(const int deepBufferWidth, const int deepBufferHeight);

    //------------------------------

    bool singleFileOutput(const size_t fileId) const;

    std::string calcFilename(const size_t fileId,
                             ImageWriteCache::ImageWriteCacheTmpFileItemShPtr& tmpFileItem) const;

    bool verifyFilenameAndDataType(const size_t fileId,
                                   const std::string& filename,
                                   ImageWriteCache::ImageWriteCacheTmpFileItemShPtr& tmpFileItem,
                                   bool& returnStatus) const;

    OIIO::ImageOutput::unique_ptr
    setupOiioImageSetup(const std::string& filename,
                        ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                        bool& result) const;

    void setupOiioImageSpecTable(const size_t fileId,
                                 std::vector<OIIO::ImageSpec>& specs) const;

    bool openFile(const std::string& filename,
                  const ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                  OIIO::ImageOutput::unique_ptr& io,
                  const std::vector<OIIO::ImageSpec>& specs) const;

    bool fillBufferAndWrite(const size_t fileId,
                            const std::string& filename,
                            const ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                            OIIO::ImageOutput::unique_ptr& io,
                            const std::vector<OIIO::ImageSpec>& specs) const;

    bool closeFile(const std::string& filename,
                   const ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem,
                   OIIO::ImageOutput::unique_ptr& io) const;

    //------------------------------

    void setupImageSpecStdEnq(ImageWriteCacheImageSpec* imgSpec,
                              const Image& img,
                              const scene_rdl2::math::HalfOpenViewport& aperture,
                              const scene_rdl2::math::HalfOpenViewport& region) const;
    static void addOiioImageSpec(const ImageWriteCacheImageSpec* imgSpec,
                                 std::vector<OIIO::ImageSpec>& specs);
    void updateHashImageSpecOrg(const File& f,
                                const scene_rdl2::math::HalfOpenViewport& aperture,
                                const scene_rdl2::math::HalfOpenViewport& region) const;
    static OIIO::TypeDesc getChannelFormat(const scene_rdl2::rdl2::RenderOutput* ro);
    static std::string getCompressionStr(const scene_rdl2::rdl2::RenderOutput* ro);

    //------------------------------

    bool subImageFillBufferAndWrite(const size_t fileId,
                                    const size_t subImgId,
                                    OIIO::ImageOutput::unique_ptr& io,
                                    const OIIO::ImageSpec* spec) const;
    bool fillBuffer(const size_t fileId,
                    const size_t subImgId,
                    OIIO::ImageBuf* buffer) const;
    void fillPixBufferStd(const size_t fileId,
                          const size_t subImgId,
                          const int x,
                          const int y,
                          const VariablePixelBuffer* aovBuffers,
                          const VariablePixelBuffer* displayFilterBuffers,
                          void* outData) const;
    bool fillPixBufferEnq(const size_t fileId,
                          const size_t subImgId,
                          const int x,
                          const int y,
                          const VariablePixelBuffer* aovBuffers,
                          const VariablePixelBuffer* displayFilterBuffers,
                          void* outDataFull,
                          size_t outDataFullSize, // pixCacheSizeByte for full float
                          void* outDataHalf,
                          size_t outDataHalfSize, // pixCacheSizeByte for half float
                          PageAlignedBuff& convertBuff) const;
    void fillPixBufferSingleEntry(const Entry& e,
                                  const int x,
                                  const int y,
                                  const VariablePixelBuffer *&aov,
                                  const VariablePixelBuffer *&displayFilterBuffer,
                                  float* outPtr) const;

    void fillPixBufferDeq(const size_t fileId,
                          const size_t subImgId,
                          const void* cacheDataFull,
                          const void* cacheDataHalf,
                          float* dataOut) const;

    static void calcPixCacheSize(const std::vector<Entry>& entries, size_t& fullBuffSize, size_t& halfBuffSize);
    static float htof(const unsigned short h);
    static unsigned short ftoh(const float f);
    static void hVecToFVec(const unsigned short* hVec, const unsigned itemTotal, std::vector<float>& fVec);
    static void fVecToHVec(const PageAlignedBuff& fVec, unsigned short* hVec);

    static float precisionAdjustF(const float f, int reso=4096);
    static unsigned short precisionAdjustH(const unsigned short h, int reso=32);
    
    //------------------------------
    
    std::string errMsg(const std::string& msg,
                       const std::string& filename,
                       ImageWriteCache::ImageWriteCacheTmpFileItemShPtr tmpFileItem) const;

    //------------------------------

    std::vector<std::string>& mErrors;
    std::vector<std::string>& mInfos;

    const ImageWriteCache::Mode mRunMode {ImageWriteCache::Mode::DEQ};
    ImageWriteCache* mCache {nullptr};

    unsigned mCheckpointTileSampleTotals {0};
    int mWidth {0};
    int mHeight {0};

    ImageWriteCacheBufferSpec mBufferSpec;
    ImageWriteCacheBufferSpec* mCurrBufferSpec {nullptr};

    const std::vector<File>* mFiles {nullptr};

    const FileNameParam* mFileNameParam {nullptr};
    const CallBackCheckpointResumeMetadata* mCallBackCheckpointResumeMetadata {nullptr};

    const pbr::CryptomatteBuffer* mCryptomatteBuffer {nullptr};
    const scene_rdl2::fb_util::HeatMapBuffer* mHeatMap {nullptr};
    const scene_rdl2::fb_util::FloatBuffer* mWeightBuffer {nullptr};
    const scene_rdl2::fb_util::RenderBuffer* mRenderBufferOdd {nullptr};
    const std::vector<scene_rdl2::fb_util::VariablePixelBuffer>* mAovBuffers {nullptr};
    const std::vector<scene_rdl2::fb_util::VariablePixelBuffer>* mDisplayFilterBuffers {nullptr};
        
    scene_rdl2::grid_util::Sha1Gen* mSha1Gen {nullptr};
};

} // namespace rndr
} // namespace moonray
