// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "CheckpointSigIntHandler.h"
#include "ImageWriteDriver.h"
#include "ProcKeeper.h"

#include <scene_rdl2/render/logging/logging.h>

#include <atomic>
#include <cstdlib> // EXIT_FAILURE
#include <cstring> // memset
#include <signal.h>

namespace moonray {
namespace rndr {

static
void
checkpointSigActionFunction(int /*sig*/, siginfo_t *info, void */*ctx*/)
{
    ImageWriteDriver::get()->interruptBySignal(*info);
}

#ifdef TEST_SCENE_CONTEXT_DUMP
static
void
sceneContextDumpSigActionFunction(int sig, siginfo_t */*info*/, void */*ctx*/)
{
    auto sigStr = [](int sig) -> std::string {
        switch (sig) {
        case SIGABRT : return "SIGABRT";
        case SIGSEGV : return "SIGSEGV";
        case SIGILL : return "SIGILL";
        case SIGFPE : return "SIGFPE";
        default : return "?";
        }
    };

    std::cerr << ">> CheckpointSigIntHandler.cc sceneContextDumpSigActionFunction() sig:" << sigStr(sig) << '\n';

    ProcKeeper::get()->signalActionSceneContextDump();

    std::cerr << ">> CheckpointSigIntHandler.cc startSceneContextDump() end\n";
}
#endif // end TEST_SCENE_CONTEXT_DUMP

//-----------------------------------------------------------------------------------------

static std::atomic<bool> gCheckpointSigIntHandlerActionStarted(false);

// static function
void
CheckpointSigIntHandler::enable()
//
// setup handler for SIGINT
//
{
    struct sigaction newSigIntAction;
    std::memset(&newSigIntAction, 0, sizeof(newSigIntAction));

    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGINT);

    newSigIntAction.sa_mask = block;
    newSigIntAction.sa_sigaction = checkpointSigActionFunction;
    newSigIntAction.sa_flags = SA_SIGINFO | SA_RESTART;

    if (sigaction(SIGINT, &newSigIntAction, NULL) < 0) {
        scene_rdl2::logging::Logger::fatal("setup new SIGINT handler for checkpoint-dump failed.");
        exit(EXIT_FAILURE);
    }

    if (!ProcKeeper::get()->openWriteProgressFile()) {
        scene_rdl2::logging::Logger::fatal("writeProgressFile open failed.");
    }
}

// static function
void
CheckpointSigIntHandler::disable()
//
// fallback to default signal handler for SIGINT
//
{
    if (gCheckpointSigIntHandlerActionStarted.load(std::memory_order_relaxed)) {
        // checkpointSigIntHandler action has been started already.
        // We skip disabling.
        return;
    }

    if (!ProcKeeper::get()->closeWriteProgressFile()) {
        scene_rdl2::logging::Logger::fatal("writeProgressFile close failed.");
    }

    struct sigaction newSigIntAction;
    std::memset(&newSigIntAction, 0, sizeof(newSigIntAction));

    newSigIntAction.sa_sigaction = reinterpret_cast<void (*)(int, siginfo_t *, void *)>(SIG_DFL);
    newSigIntAction.sa_flags = SA_SIGINFO;
    
    if (sigaction(SIGINT, &newSigIntAction, NULL) < 0) {
        scene_rdl2::logging::Logger::fatal("fall back to default SIGINT handler failed.");
        exit(EXIT_FAILURE);
    }
}

// static function
void
CheckpointSigIntHandler::handlerActionStarted()
{
    gCheckpointSigIntHandlerActionStarted.store(true, std::memory_order_relaxed);
}

//------------------------------------------------------------------------------------------

#ifdef TEST_SCENE_CONTEXT_DUMP

static std::atomic<bool> gSceneContextDumpSigHandlerActionStarted = false;

// static function
void
SceneContextDumpSigHandler::enable()
//
// setup handler for SIGABRT, SIGSEGV, SIGILL, and SIGFPE
//
{
    std::cerr << ">> CheckpointSigIntHandler.cc SceneContextDumpSigHandler::enable() start\n";

    struct sigaction newSigIntAction;
    std::memset(&newSigIntAction, 0, sizeof(newSigIntAction));

    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGABRT);
    sigaddset(&block, SIGSEGV);
    sigaddset(&block, SIGILL);
    sigaddset(&block, SIGFPE);

    newSigIntAction.sa_mask = block;
    newSigIntAction.sa_sigaction = sceneContextDumpSigActionFunction;
    newSigIntAction.sa_flags = SA_SIGINFO | SA_RESTART;

    if (sigaction(SIGABRT, &newSigIntAction, NULL) < 0 ||
        sigaction(SIGSEGV, &newSigIntAction, NULL) < 0 ||
        sigaction(SIGILL, &newSigIntAction, NULL) < 0 ||
        sigaction(SIGFPE, &newSigIntAction, NULL) < 0) {
        scene_rdl2::logging::Logger::fatal("setup new SIGABRT,SIGSEGV,SIGILL, and SIGFPE handler "
                                        "for sceneContext-dump failed.");
        exit(EXIT_FAILURE);
    }
    std::cerr << ">> CheckpointSigIntHandler.cc SceneContextDumpSigHandler::enable() done\n";
}

// static function
void
SceneContextDumpSigHandler::disable()
//
// fallback to default signal handler for SIGABRT, SIGSEGV, SIGILL, and SIGFPE
//
{
    std::cerr << ">> CheckpointSigIntHandler.cc SceneContextDumpSigHandler::disable() start\n";

    if (gSceneContextDumpSigHandlerActionStarted.load(std::memory_order_relaxed)) {
        // sceneContextDumpSigHandler action has been started already.
        // We skip disabling.
        return;
    }

    struct sigaction newSigIntAction;
    std::memset(&newSigIntAction, 0, sizeof(newSigIntAction));

    newSigIntAction.sa_sigaction = reinterpret_cast<void (*)(int, siginfo_t *, void *)>(SIG_DFL);
    newSigIntAction.sa_flags = SA_SIGINFO;
    
    if (sigaction(SIGABRT, &newSigIntAction, NULL) < 0 ||
        sigaction(SIGSEGV, &newSigIntAction, NULL) < 0 ||
        sigaction(SIGILL, &newSigIntAction, NULL) < 0 ||
        sigaction(SIGFPE, &newSigIntAction, NULL) < 0) {
        scene_rdl2::logging::Logger::fatal("fall back to default handler failed for "
                                        "SIGABRT, SIGSEGV, SIGILL, and SIGFPE.");
        exit(EXIT_FAILURE);
    }

    std::cerr << ">> CheckpointSigIntHandler.cc SceneContextDumpSigHandler::disable() done\n";
}

// static function
void
SceneContextDumpSigHandler::handlerActionStarted()
{
    gSceneContextDumpSigHandlerActionStarted.store(true, std::memory_order_relaxed);
}

#endif // end TEST_SCENE_CONTEXT_DUMP

} // namespace rndr
} // namespace moonray

