// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

// Enable a test code for SceneContext dump action by the signal for MOONRAY-4417.
// Keep this code until we mainly start working for MOONRAY-4417.
// There is similar definition in side ProcKeeper.cc. Please check them.
//#define TEST_SCENE_CONTEXT_DUMP

namespace moonray {
namespace rndr {

class CheckpointSigIntHandler
{
public:
    static void enable();
    static void disable();
    static void handlerActionStarted();
};

#ifdef TEST_SCENE_CONTEXT_DUMP
class SceneContextDumpSigHandler
{
public:
    static void enable();
    static void disable();
    static void handlerActionStarted();
};
#endif // end TEST_SCENE_CONTEXT_DUMP

} // namespace rndr
} // namespace moonray

