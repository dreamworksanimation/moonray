// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImageWriteCache.h"

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>
#include <time.h>

namespace moonray {
namespace rndr {

class RenderContext;
class RenderOutputDriver;

class ImageWriteDriver
{
public:
    using ImageWriteDriverShPtr = std::shared_ptr<ImageWriteDriver>;
    using ImageWriteCacheUqPtr = std::unique_ptr<ImageWriteCache>;
    using ImageWriteCacheList = std::list<ImageWriteCacheUqPtr>;

    enum class ThreadState : int { INIT, IDLE, BUSY };

    static void init();
    static ImageWriteDriverShPtr get();

    // Non-copyable
    ImageWriteDriver& operator=(const ImageWriteDriver&) = delete;
    ImageWriteDriver(const ImageWriteDriver&) = delete;

    ImageWriteDriver();
    ~ImageWriteDriver();
    
    void setResumableOutput(bool flag) { mResumableOutput = flag; }

    void setTwoStageOutput(bool flag) { mTwoStageOutput = flag; }
    bool getTwoStageOutput() const { return mTwoStageOutput; }

    void setTmpDirectory(const std::string& directoryName) { mTmpDirectory = directoryName; }

    void setMaxBgCache(size_t max) { mMaxBgCache = max; }

    void setRenderContext(RenderContext* renderContext) { mRenderContext = renderContext; }

    // Called from signal handler. This function should consists of async-signal-safe operations.
    void interruptBySignal(const siginfo_t& sigInfo);

    //------------------------------

    ImageWriteCacheUqPtr newImageWriteCache(const RenderOutputDriver* renderOutputDriver,
                                            bool snapshotOnly = false); // MTsafe

    void updateSnapshotData(ImageWriteCacheUqPtr& imageWriteCachePtr); // MTsafe
    void resetSnapshotData(); // MTsafe

    void enqImageWriteCache(ImageWriteCacheUqPtr& imageWriteCachePtr); // MTsafe
    void waitUntilBgWriteReady(); // MTsafe

    size_t getBlockDataSizeFull() const { return mBlockDataSizeFull; }
    size_t getBlockDataSizeHalf() const { return mBlockDataSizeHalf; }

    void setLastImageWriteCache(ImageWriteCacheUqPtr& imageWriteCachePtr); // MTsafe
    const ImageWriteCache* getLastImageWriteCache() const { return mLastImageWriteCache.get(); }

    void conditionWaitUntilAllCompleted(); // called by RenderDriver::progressCheckpointRenderFrame()

    //------------------------------

    std::string genTmpFilename(const int fileSequenceId, const std::string& finalFilename);

    static size_t getProcMemUsage(); // return rss size by byte for debug

    std::string showMemUsage() const;

 private:
    std::thread mThread;
    std::atomic<ThreadState> mThreadState {ThreadState::INIT};
    bool mThreadShutdown {false};

    bool mResumableOutput {false};
    bool mTwoStageOutput {false};

    mutable std::mutex mMutex;
    std::condition_variable mCvBoot; // using at boot threadMain sequence
    size_t mMaxBgCache {2};   // Default is 2 ImageWriteCache capacity
    std::condition_variable mCvList; // for max ImageWriteCache control
    ImageWriteCacheList mImageWriteCacheList; // ImageWriteCache list
    ImageWriteCacheUqPtr mLastImageWriteCache;

    // memory size of curently processed imageWriteCache by ImageWriteDriver thread
    size_t mCurrImageWriteCacheMemSize {0};

    // BlockData size for ImageWriteCache cache data memory pre-allocation.
    size_t mBlockDataSizeFull {0};  // byte : ready after 1st call of enqImageWriteCache()
    size_t mBlockDataSizeHalf {0};  // byte : ready after 1st call of enqImageWriteCache()

    std::string mTmpDirectory;  // temporary file directory
    std::string mTmpFilePrefix; // temporary filename prefix

    std::mutex mSnapshotDataMutex;
    scene_rdl2::rec_time::RecTime mSnapshotDataFreshness;
    ImageWriteCacheUqPtr mSnapshotData; // snapshot for signal interruption

    //------------------------------
    //
    // signal interruption related parameters
    //
    std::atomic<bool> mSigReceived {false};
    siginfo_t mSigInfo; // received signal information
    time_t mSigTime {0}; // signal received time

    RenderContext* mRenderContext {nullptr}; // signalAction() uses this pointer

    //------------------------------

    static void threadMain(ImageWriteDriver* driver);
    ImageWriteCacheUqPtr deqImageWriteCache(); // MTsafe
    bool isImageWriteCacheEmpty() const; // MTsafe

    std::string getTmpFilePrefix();
    std::string getTmpFileExtension(const std::string& finalFilename);

    void signalAction();

    static std::string threadStateStr(const ThreadState& v);
};

} // namespace rndr
} // namespace moonray
