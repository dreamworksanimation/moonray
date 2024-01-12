// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/grid_util/Sha1Util.h>
#include <scene_rdl2/common/rec_time/RecTime.h>
#include <scene_rdl2/render/cache/CacheDequeue.h>
#include <scene_rdl2/render/cache/CacheEnqueue.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>

#include <OpenImageIO/imageio.h>

#include <atomic>
#include <fstream>
#include <numeric> // std::accumulate()

// If this directive is enabled, showing detail message of file output sequence. 
// These messages are tmpfile write, tmpfile copy to destination and rename to the final filename.
// This is useful for debugging purpose.
// If this directive is disabled. only showing "Wrote" message when final rename is completed.
//#define IMAGE_WRITE_DETAIL_MESSAGE

namespace scene_rdl2 {

namespace util {
    class LuaScriptRunner;
}

namespace cache {
    class CacheDequeue;
    class CacheEnqueue;
}
}
    
namespace moonray {

namespace rndr {

class RenderOutputDriver;

//------------------------------------------------------------------------------------------

class ImageWriteCacheSpecs
{
public:
    ImageWriteCacheSpecs() {}

    std::vector<OIIO::ImageSpec> &specs() { return mSpecs; }
    
    size_t memSizeByte() const;

private:
    std::vector<OIIO::ImageSpec> mSpecs;
};

//------------------------------------------------------------------------------------------

class ImageWriteCacheImageSpec
{
public:
    using ChannelFormat = scene_rdl2::rdl2::RenderOutput::ChannelFormat;
    using CallBackSetup = std::function<void(const std::string &name,
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
                                             const std::vector<ChannelFormat> &chanFormat,
                                             const std::vector<std::string> &chanNames,
                                             const std::vector<std::string> &attrNames,
                                             const std::vector<std::string> &attrTypes,
                                             const std::vector<std::string> &attrValues,
                                             const std::string &metaDataName,
                                             const std::vector<std::string> &resumeAttr)>;

    ImageWriteCacheImageSpec() {}

    void setName(const std::string &name) { mName = name; }
    void setSizeInfo(int width, int height,
                     int x, int y, int fullX, int fullY, int fullWidth, int fullHeight);
    void setCompression(const std::string &compression) { mCompression = compression; }
    void setDwaLevel(float level) { mLevel = level; }

    bool isCompressionDwa() const { return mCompression == "dwaa" || mCompression == "dwab"; }

    void pushBackChanFormatN(int total, const ChannelFormat &chanFormat);
    void pushBackChanNames(const std::vector<std::string> &chanNames);

    void setTotalNumChans(int total) { mTotalNumChans = total; }

    void setExrHeaderInfo(const std::vector<std::string> &attrNames,
                          const std::vector<std::string> &attrTypes,
                          const std::vector<std::string> &attrValues,
                          const std::string &metaDataName);
    std::vector<std::string> &resumeAttr() { return mResumeAttr; }

    //------------------------------

    void setupOIIOSpecs(const CallBackSetup &callBack) const;

    void updateHash(scene_rdl2::grid_util::Sha1Gen &sha1Gen) const;

    size_t memSizeByte() const;

    std::string show(bool detail) const;

private:
    std::string mName;

    int mWidth {0};
    int mHeight {0};

    int mSpecX {0};
    int mSpecY {0};
    int mSpecFullX {0};
    int mSpecFullY {0};
    int mSpecFullWidth {0};
    int mSpecFullHeight {0};

    int mTotalNumChans {0};

    std::string mCompression;
    float mLevel {0.0f}; // for compression type 'dwaa' or 'dwab'

    std::vector<ChannelFormat> mChanFormat;
    std::vector<std::string> mChanNames;

    std::vector<std::string> mAttrNames;        // exr metadata
    std::vector<std::string> mAttrTypes;        // exr metadata
    std::vector<std::string> mAttrValues;       // exr metadata
    std::string mMetaDataName;

    std::vector<std::string> mResumeAttr; // exr metadata for resume
};

//------------------------------------------------------------------------------------------

class ImageWriteCacheBufferSpecSubImage
//
// Buffer information for a single sub-image inside file.
// The single file might have multiple sub-images.
//
{
public:
    enum class ChanFormat : int {
        UNKNOWN,
        HALF,
        FULL
    };

    ImageWriteCacheBufferSpecSubImage() {}

    void setup(const ImageWriteCacheBufferSpecSubImage *prevBuff,
               const std::string &name,
               const size_t aovBuffSize,
               const size_t displayFilterBuffSize,
               const size_t pixCacheFullSize,
               const size_t pixCacheHalfSize,
               const std::vector<ChanFormat> &pixChanFormat,
               const std::vector<int> &pixNumChan);

    const std::string &getName() const { return mName; }

    size_t getImgSpecId() const { return mImgSpecId; }
    size_t getAovBuffOffset() const { return mAovBuffOffset; }
    size_t getDisplayFilterBuffOffset() const { return mDisplayFilterBuffOffset; }
    size_t getPixCacheFullOffset() const { return mPixCacheFullOffset; }
    size_t getPixCacheHalfOffset() const { return mPixCacheHalfOffset; }
    size_t getPixCacheFullSize() const { return mPixCacheFullSize; }
    size_t getPixCacheHalfSize() const { return mPixCacheHalfSize; }
    size_t getTotalNumChannels() const { return std::accumulate(mPixNumChan.begin(), mPixNumChan.end(), 0); }
    const std::vector<ChanFormat> &getPixChanFormat() const { return mPixChanFormat; }
    const std::vector<int> &getPixNumChan() const { return mPixNumChan; }

    void updateHash(scene_rdl2::grid_util::Sha1Gen *sha1Gen) const;

    size_t memSizeByte() const;

    std::string show() const;
    static std::string showChanFormat(const ChanFormat &format);

private:

    std::string mName;

    size_t mImgSpecId {0};

    size_t mAovBuffOffset {0};           // offset from 1st image of 1st file
    size_t mDisplayFilterBuffOffset {0}; // offset from 1st image of 1st file
    size_t mAovBuffSize {0};             // size of this image
    size_t mDisplayFilterBuffSize {0};   // size of this image
    
    size_t mPixCacheFullOffset {0};      // offset from 1st image of 1st file
    size_t mPixCacheHalfOffset {0};      // offset from 1st image of 1st file
    size_t mPixCacheFullSize {0};        // size of this image
    size_t mPixCacheHalfSize {0};        // size of this image

    std::vector<ChanFormat> mPixChanFormat; // mPixChanFormat[entryId]
    std::vector<int> mPixNumChan;           // mPixNumChan[entryId]
};

class ImageWriteCacheBufferSpecFile
//
// Buffer information for a single file.
//
{
public:
    ImageWriteCacheBufferSpecSubImage *newSubImage()
    {
        mSubImgTbl.emplace_back();
        return &mSubImgTbl.back();
    }

    size_t getSubImgTotal() const { return mSubImgTbl.size(); }
    const ImageWriteCacheBufferSpecSubImage &getBufferSpecSubImage(const size_t subImgId) const
    {
        return mSubImgTbl[subImgId];
    }

    void updateHash(scene_rdl2::grid_util::Sha1Gen *sha1Gen) const;

    size_t memSizeByte() const;

    std::string show() const;

private:
    std::vector<ImageWriteCacheBufferSpecSubImage> mSubImgTbl;
};

class ImageWriteCacheBufferSpec
//
// Buffer information for all of the files
//
{
public:
    ImageWriteCacheBufferSpecFile *newFile() {
        mFileTbl.emplace_back();
        return &mFileTbl.back();
    }

    size_t getFileTotal() const { return mFileTbl.size(); }
    const ImageWriteCacheBufferSpecFile &getBufferSpecFile(const size_t fileId) const
    {
        return mFileTbl[fileId];
    }

    void updateHash(scene_rdl2::grid_util::Sha1Gen *sha1Gen) const;

    size_t memSizeByte() const;

    std::string show() const;

private:
    std::vector<ImageWriteCacheBufferSpecFile> mFileTbl;
};

//------------------------------------------------------------------------------------------

class ImageWriteCacheTimingLog
{
public:
    inline void init(const bool deq);
    inline void rec(scene_rdl2::rec_time::RecTime &recTime, const bool deq, int id, bool add);

    size_t size(bool deq) const { return (deq)? mDeqTimes.size(): mEnqTimes.size(); }
    float get(bool deq, size_t id) const { return (deq)? mDeqTimes[id]: mEnqTimes[id]; }

    size_t memSizeByte() const;

private:
    std::vector<float> mEnqTimes;
    std::vector<float> mDeqTimes;
};

inline void
ImageWriteCacheTimingLog::init(const bool deq)
{
    if (deq) mDeqTimes.clear();
    else     mEnqTimes.clear();
}

inline void
ImageWriteCacheTimingLog::rec(scene_rdl2::rec_time::RecTime &recTime,
                              const bool deq,
                              int id,
                              bool add)
{
    auto recMain = [](scene_rdl2::rec_time::RecTime &recTime, std::vector<float> &array, int id, bool add) {
        if ((int)array.size() <= id) array.resize(id + 1, 0.0f);
        if (add) array[id] += recTime.end();
        else     array[id] =  recTime.end();
        recTime.start();
    };
    recMain(recTime, (deq)? mDeqTimes: mEnqTimes, id, add);
}

//------------------------------------------------------------------------------------------

class ImageWriteCacheTmpFileItem
{
public:
    ImageWriteCacheTmpFileItem(const std::string &tmpFilename,
                               const std::string &checkpointFilename,
                               const std::string &checkpointMultiVersionFilename,
                               const std::string &finalFilename)
        : mTmpFilename(tmpFilename)
        , mCheckpointFilename(checkpointFilename)
        , mCheckpointMultiVersionFilename(checkpointMultiVersionFilename)
        , mFinalFilename(finalFilename)
    {}
    ~ImageWriteCacheTmpFileItem() { if (mTmpFileFd != -1) closeTmpFile(); }

    const std::string getCheckpointFilename() const { return mCheckpointFilename; }
    const std::string getFinalFilename() const { return mFinalFilename; }
    inline const std::string getDestinationFilename() const;

    bool openTmpFileAndUnlink();
    
    inline bool copyCheckpointFile(std::string &errMsg) const;
    inline bool copyCheckpointMultiVersionFile(std::string &errMsg) const;
    inline bool copyFinalFile(std::string &errmsg) const;
    inline bool renameCheckpointFile(std::string &errMsg) const;
    inline bool renameCheckpointMultiVersionFile(std::string &errMsg) const;
    inline bool renameFinalFile(std::string &errMsg) const;

    bool closeTmpFile();

    size_t memSizeByte() const;

private:
    int mTmpFileFd {-1};
    std::string mTmpFilename;        // tmp file name
    std::string mCheckpointFilename; // checkpoint file name
    std::string mCheckpointMultiVersionFilename; // checkpoint multi version file name
    std::string mFinalFilename;      // regular final image output name

    bool copyFile(const std::string &dstName, std::string &errMsg) const;
    bool renameFile(const std::string &dstName, std::string &errMsg) const;

    static std::string genCopyDestName(const std::string &dstName) { return dstName + ".part"; }
};

inline const std::string
ImageWriteCacheTmpFileItem::getDestinationFilename() const
{
    return (!mCheckpointFilename.empty())? mCheckpointFilename: mFinalFilename;
}

inline bool
ImageWriteCacheTmpFileItem::copyCheckpointFile(std::string &errMsg) const
{
    return copyFile(mCheckpointFilename, errMsg);
}

inline bool
ImageWriteCacheTmpFileItem::copyCheckpointMultiVersionFile(std::string &errMsg) const
{
    return copyFile(mCheckpointMultiVersionFilename, errMsg);
}

inline bool
ImageWriteCacheTmpFileItem::copyFinalFile(std::string &errMsg) const
{
    return copyFile(mFinalFilename, errMsg);
}

inline bool
ImageWriteCacheTmpFileItem::renameCheckpointFile(std::string &errMsg) const
{
    return renameFile(mCheckpointFilename, errMsg);
}

inline bool
ImageWriteCacheTmpFileItem::renameCheckpointMultiVersionFile(std::string &errMsg) const
{
    return renameFile(mCheckpointMultiVersionFilename, errMsg);
}

inline bool
ImageWriteCacheTmpFileItem::renameFinalFile(std::string &errMsg) const
{
    return renameFile(mFinalFilename, errMsg);
}

//------------------------------------------------------------------------------------------

class ImageWriteCache
{
public:
    using ImageWriteCacheTmpFileItemShPtr = std::shared_ptr<ImageWriteCacheTmpFileItem>;

    enum class Type : int { OUTPUT, SNAPSHOT_ONLY };
    enum class Mode : int { STD, ENQ, DEQ };
    enum class ProgressStage : int {
        INIT,
        WRITE, // RenderOutputDriver::Impl::write()
        FILE,  // RenderOutputWriter::singleFileOutput()
        IMAGE, // RenderOutputWriter::fillBufferAndWrite()
        BUFF0, // OIIO::ImageBuf::write() 1st progress = 0
        BUFF1, // OIIO::ImageBuf::write() 2nd progress = 0, oiio returns 2 different progress=0
        BUFF2  // OIIO::ImageBuf::write() progress > 0
    };

    ImageWriteCache(Type type,
                    const RenderOutputDriver *renderOutputDriver, bool resumableOutput,
                    size_t initialBlockDataSizeFull, size_t initialBlockDataSizeHalf)
        : mType(type)
        , mRenderOutputDriver(renderOutputDriver)
        , mBlockDataSizeFull(initialBlockDataSizeFull) // if set 0, memory is allocated on the fly
        , mBlockDataSizeHalf(initialBlockDataSizeHalf) // if set 0, memory is allocated on the fly
        , mResumableOutput(resumableOutput)
    {}
    ~ImageWriteCache() {}

    std::vector<std::string> &getErrors() { return mErrors; }
    std::vector<std::string> &getInfos() { return mInfos; }

    inline void setupEnqMode();
    inline void setupDeqMode();

    ImageWriteCacheBufferSpec &bufferSpec() { return mBufferSpec; }

    void setupEnqCacheImageMemory(const int width, const int height);
    void expandInternalDataIfNeeded(int yBlockId, size_t addByteFull, size_t addByteHalf); // byte
    void getCapacityInternalData(int yBlockId, size_t &sizeFull, size_t &sizeHalf); // byte
    void freeInternalData();

    void calcFinalBlockInternalDataSize();
    size_t getBlockInternalDataSizeFull() const { return mBlockDataSizeFull; }
    size_t getBlockInternalDataSizeHalf() const { return mBlockDataSizeHalf; }

    static Mode runMode(const ImageWriteCache *src) { return (src)? src->mMode: Mode::STD; }

    scene_rdl2::cache::CacheEnqueue *enq() { return mCacheQueue.mCEnq.get(); }
    scene_rdl2::cache::CacheDequeue *deq() { return mCacheQueue.mCDeq.get(); }

    ImageWriteCacheImageSpec *newImgSpec(); // return newly allocated data
    size_t getImgSpecTotal() const { return mImgSpecTbl.size(); }
    ImageWriteCacheImageSpec *getImgSpec(int id);

    int getYBlockSize() const { return mYBlockSize; }
    int getYBlockTotal() const { return mYBlockTotal; }
    int calcYBlockId(const int y) const { return y / mYBlockSize; }
    void setupDeqBuff(const size_t fileId, const size_t subImgId);
    scene_rdl2::cache::CacheEnqueue *enqFullBuff(int yBlockId) {
        return mCacheQueueFullBuffArray[yBlockId].mCEnq.get();
    }
    scene_rdl2::cache::CacheDequeue *deqFullBuff(int yBlockId) {
        return mCacheQueueFullBuffArray[yBlockId].mCDeq.get();
    }
    scene_rdl2::cache::CacheEnqueue *enqHalfBuff(int yBlockId) {
        return mCacheQueueHalfBuffArray[yBlockId].mCEnq.get();
    }
    scene_rdl2::cache::CacheDequeue *deqHalfBuff(int yBlockId) {
        return mCacheQueueHalfBuffArray[yBlockId].mCDeq.get();
    }

    scene_rdl2::grid_util::Sha1Gen::Hash &dataHash() { return mDataHash; }

    //------------------------------

    void setPostCheckpointScript(const std::string &filename) { mPostCheckpointScript = filename; }
    bool hasPostCheckpointScript() const { return !mPostCheckpointScript.empty(); }
    void setCheckpointTileSampleTotals(unsigned n) { mCheckpointTileSampleTotals = n; }
    inline void setCheckpointFilename(const std::string &filename);

    //------------------------------

    void createLastImageWriteCacheSpecs() { mImageWriteCacheSpecs.emplace_back(); }
    void incrementLastImageWriteCacheSpecs() { mLastImageWriteCacheSpecsId++; }
    inline ImageWriteCacheSpecs &getLastImageWriteCacheSpecs();

    //------------------------------

    void setTwoStageOutput(bool flag) { mTwoStageOutput = flag; }
    bool getTwoStageOutput() const { return mTwoStageOutput; }

    void setCheckpointOverwrite(bool flag) { mCheckpointOverwrite = flag; }
    bool getCheckpointOverwrite() const { return mCheckpointOverwrite; }

    ImageWriteCacheTmpFileItemShPtr setTmpFileItem(const int fileSequenceId,
                                                   const std::string &tmpFilename,
                                                   const std::string &checkpointFilename,
                                                   const std::string &checkpointMultiVersionFilename,
                                                   const std::string &finalFilename);
    ImageWriteCacheTmpFileItemShPtr getTmpFileItem(const int fileSequenceId);

    bool allFinalizeCheckpointFile() const;
    bool allFinalizeCheckpointMultiVersionFile() const;
    bool allFinalizeFinalFile() const;

    //------------------------------

    void outputDataFinalize(); // for ImageWriteDriver
    void runPostCheckpointScript();

    //------------------------------

    // timing record and write action progress update APIs
    void timeStart();
    void timeRec(int id);
    void timeEnd();
    void timeStartFile();
    void timeRecFile(int id);
    void timeStartImage();
    void timeRecImage(int id);
    void timeStartBuffWrite();
    void timeUpdateBuffWrite(float fraction);
    void timeEndBuffWrite();
    std::string timeShow() const;

    void setTimeSaveSecBySignalCheckpoint(const float sec) { mTimeSaveSecBySignalCheckpoint = sec; }
    float getTimeSaveSecBySignalCheckpoint() const { return mTimeSaveSecBySignalCheckpoint; }

    unsigned getProgressCounter() const { return mProgressCounter.load(std::memory_order_relaxed); }
    ProgressStage getProgressStage() const { return mCurrProgressStage.load(std::memory_order_relaxed); }
    int getProgressBuffFraction() const
    {
        return mCurrBuffWriteProgressFraction.load(std::memory_order_relaxed); // return 0 ~ 10
    }

    //------------------------------

    size_t memSizeByte() const; // return entire size of ImageWriteCache
    size_t memSizeDataByte() const; // return image data cache size only

    std::string showMemUsage() const;

    static std::string modeStr(const Mode &mode);
    static std::string typeStr(const Type &type);

private:

    Type mType {Type::OUTPUT};
    Mode mMode {Mode::STD};

    const RenderOutputDriver *mRenderOutputDriver {nullptr};

    std::vector<std::string> mErrors;
    std::vector<std::string> mInfos;

    //------------------------------

    std::string mPostCheckpointScript; // post checkpoint script name
    unsigned mCheckpointTileSampleTotals {0};
    std::vector<std::string> mCheckpointFilename;

    //------------------------------

    ImageWriteCacheBufferSpec mBufferSpec;

    int mWidth {0};             // output image width
    int mHeight {0};            // output image height
    int mYBlockSize {0};        // single yblock height
    int mYBlockTotal {0};       // total number of blocks in height

    // Following std::string and vector of std::string are cache data itself and used by cacheQueue.
    // std::string is used as byte data storage (and not ASCII string data storage).
    std::string mData;
    std::vector<std::string> mDataFullArray; // mDataFullArray[yBlockId]
    std::vector<std::string> mDataHalfArray; // mDataHalfArray[yBlockId]

    size_t mBlockDataSizeFull {0}; // byte : final data size of mDataFullArray's one item
    size_t mBlockDataSizeHalf {0}; // byte : final data size of mDataHalfArray's one item

    std::vector<ImageWriteCacheImageSpec> mImgSpecTbl;

    union CacheQueue {
        CacheQueue() {}
        ~CacheQueue() {}
        CacheQueue(const CacheQueue &) {} // copy constructor does not work at this moment
        std::unique_ptr<scene_rdl2::cache::CacheEnqueue> mCEnq; // We only need one of the other
        std::unique_ptr<scene_rdl2::cache::CacheDequeue> mCDeq;
    };

    CacheQueue mCacheQueue;
    std::vector<CacheQueue> mCacheQueueFullBuffArray; // mCacheQueueFullBuffArray[yBlockId]
    std::vector<CacheQueue> mCacheQueueHalfBuffArray; // mCacheQueueHalfBuffArray[yBlockId]

    size_t mLastImageWriteCacheSpecsId {0};
    std::vector<ImageWriteCacheSpecs> mImageWriteCacheSpecs;

    // output data hash for runtime verify
    scene_rdl2::grid_util::Sha1Gen::Hash mDataHash {scene_rdl2::grid_util::Sha1Util::init()};

    //------------------------------

    const bool mResumableOutput;
    bool mTwoStageOutput {true};
    bool mCheckpointOverwrite {true};
    std::vector<ImageWriteCacheTmpFileItemShPtr> mTmpFileItemArray;
    
    //------------------------------

    scene_rdl2::rec_time::RecTime mRecTimeAll, mRecTime, mRecTimeFile, mRecTimeImage, mRecTimeBuffWrite;
    float mTimeAll[2]; // 0:STD/ENQ 1:DEQ
    ImageWriteCacheTimingLog mTime, mTimeFile, mTimeImage;

    float mTimeSaveSecBySignalCheckpoint {0.0f};

    // write action progress update related parameters
    std::atomic<unsigned> mProgressCounter {0};
    std::atomic<ProgressStage> mCurrProgressStage {ProgressStage::INIT};
    std::atomic<int> mCurrBuffWriteProgressCounter {0};
    std::atomic<int> mCurrBuffWriteProgressFraction {0}; // 0 ~ 10

    //------------------------------

    inline void initAllTimeArray();
    void setupLuaGlobalVariables(scene_rdl2::util::LuaScriptRunner &luaRun);

    //------------------------------

    using TmpFileActionCallBack = 
        std::function<bool (const ImageWriteCacheTmpFileItemShPtr, std::string &)>;

    bool allFinalizeFile(bool resumable,
                         const TmpFileActionCallBack &copyCallBack,
                         const TmpFileActionCallBack &renameCallBack) const;
    bool crawlAllTmpFileItems(const TmpFileActionCallBack &callBack) const;

    void updateProgress(const ProgressStage &stage); // w/ memory barrier
};

inline void
ImageWriteCache::setupEnqMode()
{
    //
    // We need to construct mCacheQueue here however we construct mCacheQueueFullBuff and
    // mCacheQueueHalfBuff later due to the need for width/height information for construction.
    // They are executed by setupEnqCacheImageMemory()
    //
    new(&mCacheQueue.mCEnq) std::unique_ptr<scene_rdl2::cache::CacheEnqueue>;
    mCacheQueue.mCEnq.reset(new scene_rdl2::cache::CacheEnqueue(&mData));

    mMode = Mode::ENQ;
}

inline void
ImageWriteCache::setupDeqMode()
{
    if (mMode == Mode::ENQ) {
        // We need to finalize CacheEnqueue when previous condition is ENQ
        enq()->finalize();
        for (int yBlockId = 0; yBlockId < mYBlockTotal; ++yBlockId) {
            enqFullBuff(yBlockId)->finalize();
            enqHalfBuff(yBlockId)->finalize();
        }
    }

    // Re-construct CacheDequeue
    new(&mCacheQueue.mCDeq) std::unique_ptr<scene_rdl2::cache::CacheDequeue>;
    mCacheQueue.mCDeq.reset(new scene_rdl2::cache::CacheDequeue(mData.data(), mData.size()));

    for (int yBlockId = 0; yBlockId < mYBlockTotal; ++yBlockId) {
        new(&mCacheQueueFullBuffArray[yBlockId].mCDeq) std::unique_ptr<scene_rdl2::cache::CacheDequeue>;
        mCacheQueueFullBuffArray[yBlockId].mCDeq.reset
            (new scene_rdl2::cache::CacheDequeue(mDataFullArray[yBlockId].data(), mDataFullArray[yBlockId].size()));

        new(&mCacheQueueHalfBuffArray[yBlockId].mCDeq) std::unique_ptr<scene_rdl2::cache::CacheDequeue>;
        mCacheQueueHalfBuffArray[yBlockId].mCDeq.reset
            (new scene_rdl2::cache::CacheDequeue(mDataHalfArray[yBlockId].data(), mDataHalfArray[yBlockId].size()));
    }

    mMode = Mode::DEQ;
    mLastImageWriteCacheSpecsId = 0;
}

void
ImageWriteCache::setCheckpointFilename(const std::string &filename)
{
    mCheckpointFilename.push_back(filename);
}

inline ImageWriteCacheSpecs &
ImageWriteCache::getLastImageWriteCacheSpecs()
{
    return mImageWriteCacheSpecs[mLastImageWriteCacheSpecsId];
}

inline void
ImageWriteCache::initAllTimeArray()
{
    bool deq = (mMode == Mode::DEQ);
    mTimeAll[(deq)? 1: 0] = 0.0f;
    mTime.init(deq);
    mTimeFile.init(deq);
    mTimeImage.init(deq);
}
    
} // namespace rndr
} // namespace moonray
