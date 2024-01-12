// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ImageWriteDriver.h"

#include "CheckpointSigIntHandler.h"
#include "ImageWriteCache.h"
#include "ProcKeeper.h"
#include "RenderContext.h"
#include "RenderProgressEstimation.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/mcrt_common/Util.h>

#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/render/util/TimeUtil.h>

#include <cstdlib>              // getenv, EXIT_SUCCESS
#include <malloc.h>             // malloc_trim
#include <sys/types.h>          // getpid
#include <sys/sysinfo.h>        // sysinfo
#include <unistd.h>             // getpid, gethostname

// This directive is used to dump all ImageWriteDriver related memory tracking information
//#define DEBUG_MSG_MEMUSAGE

// This directive is used to dump max bgCache control related condition
//#define DEBUG_MSG_MAXBGCACHE

namespace moonray {
namespace rndr {

ImageWriteDriver::ImageWriteDriver()
{
    memset(&mSigInfo, 0x0, sizeof(siginfo_t));

    // We have to build thread after finish mMutex and mCvBoot initialization completed
    mThread = std::move(std::thread(threadMain, this));

    // Wait until thread is booted
    std::unique_lock<std::mutex> uqLock(mMutex);
    mCvBoot.wait(uqLock, [&]{
            return (mThreadState != ThreadState::INIT); // Not wait if already non INIT condition
        });
}

ImageWriteDriver::~ImageWriteDriver()
{
    mThreadShutdown = true; // This is a only place set true to mThreadShutdown
    if (mThread.joinable()) {
        mThread.join();
    }
}

void
ImageWriteDriver::interruptBySignal(const siginfo_t& sigInfo)
//
// This function is called from a signal handler
// So we should only use the async-signal-safe operation inside this function
//
{
    if (mSigReceived) {
        return; // We already received a signal and avoid duplicate execution.
    }

    CheckpointSigIntHandler::handlerActionStarted(); // We record that handler action has already started.

    if (mRenderContext) {
        // We want to stop all MCRT threads however we don't want to block execution here.
        // So we should use requestStop instead of the stopFrame()
        mRenderContext->requestStopAsyncSignalSafe(); // non blocking and async-signal-safe function

        mRenderContext->getFrameProgressEstimation()->signalInterruption(); // for banner update
    }

    mSigTime = time(NULL); // save signal received time : this is an async-signal-safe function
    mSigInfo = sigInfo;

    // Value update of sigReceived should be last of this function because this atomic parameter
    // is beeing watched from ImageWriteDriver main thread.
    mSigReceived = true;
}

ImageWriteDriver::ImageWriteCacheUqPtr
ImageWriteDriver::newImageWriteCache(const RenderOutputDriver* renderOutputDriver, bool snapshotOnly)
{
    std::lock_guard<std::mutex> lock(mMutex);

#   ifdef  DEBUG_MSG_MEMUSAGE
    std::cerr << ">> ImageWriteDriver.cc newImageWriteCache() " << showMemUsage() << std::endl;
#   endif // end  DEBUG_MSG_MEMUSAGE

    using Type = ImageWriteCache::Type;
    auto calcType = [&](bool snapshotOnly) -> Type {
        return (snapshotOnly) ? Type::SNAPSHOT_ONLY : Type::OUTPUT;
    };

    return ImageWriteCacheUqPtr(new ImageWriteCache(calcType(snapshotOnly),
                                                    renderOutputDriver,
                                                    mResumableOutput,
                                                    mBlockDataSizeFull,
                                                    mBlockDataSizeHalf));
}

void
ImageWriteDriver::updateSnapshotData(ImageWriteCacheUqPtr& imageWriteCachePtr)
{
    {
        std::lock_guard<std::mutex> lock(mSnapshotDataMutex);
        mSnapshotData = std::move(imageWriteCachePtr);
        mSnapshotDataFreshness.start();
    }
    malloc_trim(0); // Return unused memory from malloc() arena to OS 
}

void
ImageWriteDriver::resetSnapshotData()
{
    {
        std::lock_guard<std::mutex> lock(mSnapshotDataMutex);
        mSnapshotData.reset();
    }
    malloc_trim(0); // Return unused memory from malloc() arena to OS 
}

void
ImageWriteDriver::enqImageWriteCache(ImageWriteCacheUqPtr& imageWriteCachePtr) // MTsafe
{
    std::lock_guard<std::mutex> lock(mMutex);

    // update blockDataSizeFull/Half by 1st call of this function
    // We use this info for preallocation of 2nd or later ImageWriteCache.
    // mBlockDatasizefull/mBlockDataSizeHalf size is constant during entire render process.
    if (mBlockDataSizeFull == 0) {
        mBlockDataSizeFull = imageWriteCachePtr->getBlockInternalDataSizeFull();
    }
    if (mBlockDataSizeHalf == 0) {
        mBlockDataSizeHalf = imageWriteCachePtr->getBlockInternalDataSizeHalf();
    }

    mImageWriteCacheList.push_back(std::move(imageWriteCachePtr));

    malloc_trim(0); // Return unused memory from malloc() arena to OS 

#   ifdef DEBUG_MSG_MEMUSAGE
    std::cerr << ">> ImageWriteDriver.cc enqImageWriteCache() " << showMemUsage() << std::endl;    
#   endif // end DEBUG_MSG_MEMUSAGE
}

void
ImageWriteDriver::waitUntilBgWriteReady() // MTsafe
//
// Wait until we have enough bg cache capacity by conditional wait
// This function is used by RenderDriver::checkpointFileOutput()
//
{
    std::unique_lock<std::mutex> uqLock(mMutex);

    auto totalImageWriteCache = [&]() -> size_t {
        size_t total = mImageWriteCacheList.size();
        if (mThreadState == ThreadState::BUSY) {
            // We have to account for the current working ImageWriteCache in this case.
            total++;
        }
        return total;
    };

#   ifdef DEBUG_MSG_MAXBGCACHE
    auto debugMsg = [&]() -> std::string {
        std::ostringstream ostr;
        ostr
        << " backLog:" << mImageWriteCacheList.size()
        << " maxBgCache:" << mMaxBgCache
        << " engine:" << threadStateStr(mThreadState)
        << " total:" << totalImageWriteCache();
        return ostr.str();
    };

    {
        std::ostringstream ostr;
        ostr << ((totalImageWriteCache() >= mMaxBgCache) ? ">Suspend<" : "---Run---") << debugMsg();
        std::cerr << ostr.str() << '\n';
    }
#   endif // end DEBUG_MSG_MAXBGCACHE

    // conditional wait until ready to save more ImageWriteCache
    mCvList.wait(uqLock, [&]{
            return (totalImageWriteCache() < mMaxBgCache); // No wait if we have room.
        });

#   ifdef DEBUG_MSG_MAXBGCACHE
    {
        std::ostringstream ostr;
        ostr << "-Finish--" << debugMsg();
        std::cerr << ostr.str() << '\n';
    }
#   endif // end DEBUG_MSG_MAXBGCACHE
}

void
ImageWriteDriver::setLastImageWriteCache(ImageWriteCacheUqPtr& imageWriteCachePtr) // MTsafe
{
    std::lock_guard<std::mutex> lock(mMutex);

    mCurrImageWriteCacheMemSize = 0; // reset mem size for debug

    imageWriteCachePtr->freeInternalData();
    mLastImageWriteCache = std::move(imageWriteCachePtr);

#   ifdef  DEBUG_MSG_MEMUSAGE
    std::cerr << ">> ImageWriteDriver.cc setLastImageWriteCache() " << showMemUsage() << std::endl;    
#   endif // end  DEBUG_MSG_MEMUSAGE
}

void
ImageWriteDriver::conditionWaitUntilAllCompleted()
// This function is used by RenderDriver::progressCheckpointRenderFrame()
{
    std::unique_lock<std::mutex> uqLock(mMutex);
    mCvBoot.wait(uqLock, [&]{
            // Not wait if cacheList is empty and also threadState == IDLE
            return (mImageWriteCacheList.empty() && mThreadState == ThreadState::IDLE);
        });
}

std::string
ImageWriteDriver::genTmpFilename(const int fileSequenceId,
                                 const std::string& finalFilename)
{
    return (mTmpDirectory + "/" + getTmpFilePrefix() +
            std::to_string(fileSequenceId) + getTmpFileExtension(finalFilename));
}

// static function
size_t
ImageWriteDriver::getProcMemUsage()
// Return current process memory size (rss size and not include vsize) for debug
{
    std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);

    std::string pid, comm, state, ppid, pgrp, session, tty_nr;
    std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    std::string utime, stime, cutime, cstime, priority, niceval;
    std::string O, itrealvalue, starttime;

    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> niceval
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
    stat_stream.close();

    size_t rssByteSize = rss * sysconf(_SC_PAGE_SIZE); // byte

    // return vsize + rssByteSize;
    return rssByteSize;         // Only return rss size
}

std::string
ImageWriteDriver::showMemUsage() const
{
    size_t imageWriteCacheMemTotalSize = 0;
    for (auto itr = mImageWriteCacheList.begin(); itr != mImageWriteCacheList.end(); ++itr) {
        imageWriteCacheMemTotalSize += (*itr)->memSizeByte();
    }

    size_t lastImageWriteCacheMemSize = 0;
    if (mLastImageWriteCache) {
        lastImageWriteCacheMemSize = mLastImageWriteCache->memSizeByte();
    }

    size_t currImageWriteCacheMemSize = mCurrImageWriteCacheMemSize;

    size_t totalCacheMemSize =
        imageWriteCacheMemTotalSize + lastImageWriteCacheMemSize + currImageWriteCacheMemSize;

    std::ostringstream ostr;
    ostr << "ImageWriteDriver memUsage info {\n";
    if (mLastImageWriteCache) {
        ostr << "  mLastImageWriteCache:ON"
             << " memSize:" << scene_rdl2::str_util::byteStr(lastImageWriteCacheMemSize) << "\n";
    } else {
        ostr << "  mLastImageWriteCache:OFF\n";
    }
    if (currImageWriteCacheMemSize) {
        ostr << "  mCurrImageWriteCache:ON"
             << " memSize:" << scene_rdl2::str_util::byteStr(currImageWriteCacheMemSize) << "\n";
    } else {
        ostr << "  mCurrImageWriteCache:OFF\n";
    }
    ostr << "  mImageWriteCacheList:"
         << " total:" << mImageWriteCacheList.size()
         << " memSize:" << scene_rdl2::str_util::byteStr(imageWriteCacheMemTotalSize) << "\n";
    ostr << "  totalImageCacheMemSize:" << scene_rdl2::str_util::byteStr(totalCacheMemSize) << "\n"
         << "  procMem:" << scene_rdl2::str_util::byteStr(getProcMemUsage()) << "\n";
    if (mSnapshotData) {
        ostr << "  mSnapshotData:ON"
             << " memSize:" << scene_rdl2::str_util::byteStr(mSnapshotData->memSizeByte()) << '\n';
    } else {
        ostr << "  mSnapshotData:OFF\n";
    }
    ostr << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

// static
void
ImageWriteDriver::threadMain(ImageWriteDriver* driver)
{
    // First of all change driver's threadState condition and do notify_one to caller.
    driver->mThreadState = ThreadState::IDLE;
    driver->mCvBoot.notify_one(); // notify to ImageWriteDriver's constructor

    //
    // This imageWriteDriver thread main function does not have interactive boot/shutdown
    // functionality yet. It is difficult to call this API from RenderContext::startFrame()
    // at this moment.
    // So we booted ImageWriteDriver thread by rndr::initGlobalDriver() regardless of
    // checkpoint_bg_write scene variable is true or false.
    //
    while (1) {
        if (driver->mThreadShutdown) {
            // We are in the shutdown sequence of ImageWriteDriver thread.
            if (driver->isImageWriteCacheEmpty()) { // MTsafe
                break; // We can safely complete this thread
            }
            // There is some ImageWriteCache which need to process
        }

        driver->mThreadState = ThreadState::BUSY;
        ImageWriteCacheUqPtr cache = std::move(driver->deqImageWriteCache());
        if (cache) {
            // We have regular checkpoint data
            cache->outputDataFinalize();

            driver->setLastImageWriteCache(cache); // free internal cache memory as well
            driver->mThreadState = ThreadState::IDLE;

            if (driver->mImageWriteCacheList.size() < driver->mMaxBgCache) {
                driver->mCvList.notify_one();   // send notify if total ImageWriteCache is less than maxBgCache        
            }

        } else if (driver->mSigReceived) {
            // We received signal!
            // All regular checkpoint files were finished to write out.
            // We are ready to dump snapshot data.
            driver->signalAction(); // never return this function

        } else {
            // We should not use thread conditional wait in order to yield CPU resources.
            // If we implement CPU yield logic by thread condition variable, we have to call
            // std::condition_variable::notify_one() from a signal handler.
            // I don't have confidence that this call is async-signal-safe at this moment.
            // It would be better not to use the thread condition variable here.

            // first of all change condition then yield CPU
            driver->mThreadState = ThreadState::IDLE;
            if (!driver->mThreadShutdown) {
                driver->mCvBoot.notify_one(); // nofity to RenderDriver::progressCheckpointRenderFrame()

                usleep(1000); // 1000us = 1ms : wake up every 1ms and check ImageWriteCache
            } else {
                // We are in the shutdown sequence and simply skip all CPU yield logic
            }
        }
    }
}

ImageWriteDriver::ImageWriteCacheUqPtr
ImageWriteDriver::deqImageWriteCache() // MTsafe
{ 
    std::lock_guard<std::mutex> lock(mMutex);

    if (mImageWriteCacheList.empty()) return ImageWriteCacheUqPtr(nullptr);
    ImageWriteCacheUqPtr ptr = std::move(mImageWriteCacheList.front());
    mImageWriteCacheList.pop_front();

    mCurrImageWriteCacheMemSize = ptr->memSizeByte(); // for debug

#   ifdef  DEBUG_MSG_MEMUSAGE
    std::cerr << ">> ImageWriteDriver.cc deqImageWriteCache() " << showMemUsage() << std::endl;    
#   endif // end  DEBUG_MSG_MEMUSAGE
    return ptr;
}

bool
ImageWriteDriver::isImageWriteCacheEmpty() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mImageWriteCacheList.empty();
}

std::string
ImageWriteDriver::getTmpFilePrefix()
{
    if (mTmpFilePrefix.empty()) {
        char buff[HOST_NAME_MAX];
        if (gethostname(buff, HOST_NAME_MAX) == -1) std::strcpy(buff, "unknown");
        std::string hostname(buff);
        size_t pos = hostname.find(".");
        if (pos != std::string::npos) hostname.erase(pos);
        pid_t myPid = getpid();
        mTmpFilePrefix = "moonray_" + hostname + "_" + std::to_string((size_t)myPid) + "_";
    }
    return mTmpFilePrefix;
}

std::string
ImageWriteDriver::getTmpFileExtension(const std::string& finalFilename)
{
    size_t pos = finalFilename.rfind(".");
    if (pos == std::string::npos) return ".exr";
    return finalFilename.substr(pos);
}

void
ImageWriteDriver::signalAction()
//
// Dump snapshot data to the disk if we have it.
// This function never return.
//
{
    auto showSigInfo = [](const siginfo_t& info) -> std::string {
        std::ostringstream ostr;
        ostr
        << "siginfo {\n"
        << "  si_signo:" << info.si_signo << " // received signal number\n"
        << "  si_code:" << info.si_code << " // received signal code\n"
        << "  si_pid:" << static_cast<int>(info.si_pid) << " // signal sending process ID\n"
        << "  si_uid:" << static_cast<int>(info.si_uid) << " // Real user ID of sending signal process\n"
        << "}";
        return ostr.str();
    };
    auto msgOut = [](const std::string& str) {
        scene_rdl2::logging::Logger::info(str);
    };

    timeval signalActionStartTime = scene_rdl2::time_util::getCurrentTime();

    std::ostringstream ostr;
    ostr << "=====>>>>> signalAction start <<<<<=====\n"
         << "SigRecv:" << scene_rdl2::time_util::timeStr(mSigTime) << '\n'
         << "current:" << scene_rdl2::time_util::timeStr(signalActionStartTime) << '\n'
         << showSigInfo(mSigInfo) << '\n';

    ImageWriteCacheUqPtr currSnapshotData;
    float freshnessSec = 0.0f;
    {
        std::lock_guard<std::mutex> lock(mSnapshotDataMutex);
        currSnapshotData = std::move(mSnapshotData);
        mSnapshotData.reset();
        if (currSnapshotData) {
            freshnessSec = mSnapshotDataFreshness.end();
        }
    }

    if (currSnapshotData) {
        ostr << "dump snapshot data sequence start ...";
        msgOut(ostr.str());
        ostr.str("");

        scene_rdl2::rec_time::RecTime time;
        time.start();
        currSnapshotData->outputDataFinalize();
        float fileOutputSec = time.end();
        ostr << "snapshot data freshness:" << freshnessSec << " sec old\n";

        float timeSaveSecBySignalCheckpoint = currSnapshotData->getTimeSaveSecBySignalCheckpoint();
        if (timeSaveSecBySignalCheckpoint > 0.0f) {
            ostr << "signal-based checkpoint saved "
                 << scene_rdl2::str_util::secStr(timeSaveSecBySignalCheckpoint)
                 << " time compared to the ordinal checkpoint\n";
        }
        ostr << "checkpoint file output action:" << fileOutputSec << " sec\n";

        // Useful debug message for timing
        // ostr << ">> ImageWriteDriver.cc signalAction() " << currSnapshotData->timeShow() << '\n';
    } else {
        ostr << "snapshotData is empty\n";
    }

    ProcKeeper::get()->closeWriteProgressFile(); // Cleanup operation regarding write action progress file

    ostr << "=====>>>>> signalAction done <<<<<=====";
    msgOut(ostr.str());

    _exit(EXIT_SUCCESS);
}

// static function
std::string
ImageWriteDriver::threadStateStr(const ThreadState& v)
{
    switch (v) {
    case ThreadState::INIT: return "INIT";
    case ThreadState::IDLE: return "IDLE";
    case ThreadState::BUSY: return "BUSY";
    default : break;
    }
    return "?";
}

//==========================================================================================

ImageWriteDriver::ImageWriteDriverShPtr gImageWriteDriver;

// static function
void
ImageWriteDriver::init()
{
    MOONRAY_THREADSAFE_STATIC_WRITE(gImageWriteDriver.reset(new ImageWriteDriver));
}

// static
ImageWriteDriver::ImageWriteDriverShPtr
ImageWriteDriver::get()
{
    return gImageWriteDriver;
}

} // namespace rndr
} // namespace moonray
