// Copyright 2023-2025 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ImageWriteCache.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
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
    //
    // tmp directory management related
    //
    bool openTmpDirControlFile(const float maxExpirationDeltaTimeMinutes = 15.0f,
                               const float minExpirationDeltaTimeMinutes = 5.0f);

    bool getProtectTmpDir() const { return mProtectTmpDir; };
    void cleanUpTmpDirControlFile(std::string& msg);

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
    //
    // tmp directory management for signal-checkpointing
    //
    // In order to reduce the risk of generating collapse image output, we are using a 2-stage output solution
    // as a default for all the image output from MoonRay. Under a 2-stage output solution, MoonRay creates
    // temporary files into the tmp directory first, and copy/rename to the final destination next. This means
    // MoonRay needs a tmp directory to output the checkpoint files.
    // Under the signal-checkpointing, The queue system sends SIGINT to MoonRay in some situations and MoonRay
    // starts the dumping checkpoint files as soon as receiving SIGINT. However, the default of the
    // DreamWorksAnimation queue system behavior deletes the tmp directory when sending SIGINT to the MoonRay.
    // (Under the queue system controlled farm job, the tmp directory is not a regular directory and is managed
    // by the queue system. This is a DreamWorksAnimation specific environment)
    // This is no good because MoonRay needs the tmp directory to dump the checkpoint file. To properly output
    // the checkpoint file under signal-based checkpointing, MoonRay creates DO_NOT_TMP_CLEAN file inside the
    // tmp directory at the beginning of the MCRT stage with the expiration timestamp on the 1st line of this
    // file. If this file exists in the tmp directory, the queue system does not clean up the tmp directory and
    // preserve the tmp directory files when sending the SIGINT.
    // If MoonRay crashes during file output for an unknown reason and temporary files remain in the tmp
    // directory, these files are eventually cleaned up by another independent cron process later based on the
    // expiration date/time info of DO_NOT_TMP_CLEAN file.
    // MoonRay removes this DO_NOT_TMP_CLEAN file if the process is finished the normal way or the signal
    // handler finishes properly under signal-based checkpointing.
    //
    // MoonRay tries to keep the expiration timestamp of DO_NOT_TMP_CLEAN a reasonable length (like 15 minutes)
    // but we don't know the exact rendering time before finishing rendering. So, MoonRay tries to update the
    // expiration timestamp during the MCRT phase frequently as needed. This update operation is done by atomic
    // file operation. As a result, DO_NOT_TMP_CLEAN has always kept a reasonable expiration timestamp.
    //
    bool mProtectTmpDir {false}; // Status of tmp directory protection against clean up by queue system
    std::string mTmpDirControlFileName;
    std::string mTmpDirControlFileNameTmp; // Initial filename before rename
    std::chrono::seconds mTmpDirMaxExpirationDelta {0};
    std::chrono::seconds mTmpDirMinExpirationDelta {0};
    std::chrono::time_point<std::chrono::system_clock> mTmpDirExpiration;

    //------------------------------

    static void threadMain(ImageWriteDriver* driver);
    ImageWriteCacheUqPtr deqImageWriteCache(); // MTsafe
    bool isImageWriteCacheEmpty() const; // MTsafe

    std::string getTmpFilePrefix();
    std::string getTmpFileExtension(const std::string& finalFilename);

    bool updateTmpDirControlFile();
    bool manageTmpDirControlFileTimeStamp();

    void signalAction();

    static std::string threadStateStr(const ThreadState& v);
};

} // namespace rndr
} // namespace moonray
