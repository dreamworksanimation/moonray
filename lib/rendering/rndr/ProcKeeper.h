// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace moonray {
namespace rndr {

class ImageWriteCache;

class ProcKeeper
//
// This class boots a thread and this thread is in charge of write-progress-update and signal-based
// SceneContext dump.
//
// Basically, inside a signal handler, we can only execute async-signal-safe functions and it is not
// enough for executing the desired actions at signal received time.
// Instead, we set up an internal flag of this class from signal handler. And the thread which is
// booted by this class's constructor executes the main actions that we need.
// Under this solution, we basically can run any code without getting any limitations of a signal
// handler when we received the signal.
// Also, we can pretty easy to add other signal actions to this class if we need in the future.
//
{
public:
    using ProcKeeperShPtr = std::shared_ptr<ProcKeeper>;

    enum class ThreadState : int { INIT, IDLE, BUSY, DONE };
    enum class RunState : int { WAIT, START };

    static void init();
    static ProcKeeperShPtr get();

    // Non-copyable
    ProcKeeper &operator =(const ProcKeeper &) = delete;
    ProcKeeper(const ProcKeeper &) = delete;

    ProcKeeper();
    ~ProcKeeper();

    //------------------------------
    //
    // write-action progress report related
    //
    bool openWriteProgressFile();
    bool closeWriteProgressFile();

    void startImageWrite(const ImageWriteCache *imageWriteCache);
    void finishImageWrite(); // busy wait blocking function

    //------------------------------
    //
    // sceneContext dump related
    //
    void signalActionSceneContextDump(); // Request to execute sceneContext dump

protected:
    static void threadMain(ProcKeeper *keeper);

    void main();

    void setupImageWriteProgressFilename();
    bool openWriteProgressFileMain();
    bool updateWriteProgressFile(const std::string &str) const;

    static std::string runStateStr(const RunState &state);
    static std::string progressSymbol(const ImageWriteCache *cache);

    void signalActionSceneContextDumpMain();

    //------------------------------

    std::thread mThread;
    std::atomic<ThreadState> mThreadState;
    std::atomic<RunState> mRunState;
    std::atomic<bool> mThreadShutdown;

    // This code will be refactored when working regarding MOONRAY-4417.
    // See comment of ProcKeeper constructor comments.
    /*
    mutable std::mutex mMutexBoot;
    std::condition_variable mCvBoot; // using at boot threadMain sequence
    */

    mutable std::mutex mMutexRun;
    std::condition_variable mCvRun; // using at run threadMain 

    //------------------------------

    bool mImageWriteProgress;

    std::atomic<bool> mMainTaskCancel;
    const ImageWriteCache *mImageWriteCache;

    std::string mImageWriteProgressFilename;
    int mImageWriteProgressFd;

    //------------------------------

    bool mSceneContextDump;
};

} // namespace rndr
} // namespace moonray

