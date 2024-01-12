// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/grid_util/DebugConsoleDriver.h>

namespace moonray {
namespace rndr {

class RenderContext;

class RenderContextConsoleDriver : public scene_rdl2::grid_util::DebugConsoleDriver
//
// This class provides server side functionality of interactive moonray_gui control feature by
// telnet connection. This functionality was designed to simulate arras environment action
// (i.e. progmcrt computation) without arras as a simple procedure. This is a very powerful
// solution in order to isolate issues that are related to arras or purely moonray trouble.
// We can send commands to moonray_gui via telnet connection while moonray_gui is actively rendering.
// It is pretty easy to add a new command to this class if you need to.
//    
// This object boots thread inside the DebugConsoleDriver::initialize() and this is a main thread
// to handle telnet connection. This thread is shut down automatically when this object is destructed.
// Only support single incoming telnet connection at this moment.
//
{
public:
    using RenderContextConsoleDriverShPtr = std::shared_ptr<RenderContextConsoleDriver>;

    // Negative port number disables debug console features and never constructs
    // RenderContextConsoleDriver internally (In this case, get() return nullptr).
    // Port = 0 is special, in this case, the kernel finds the available port for you.
    // Usually you specify the port number which you want to use.
    static void init(int port);
    static RenderContextConsoleDriverShPtr get();
    
    RenderContextConsoleDriver() :
        DebugConsoleDriver(),
        mRenderContext(nullptr)
    {}
    ~RenderContextConsoleDriver() {}

    void setRenderContext(RenderContext *renderContext); // MTsafe

private:
    using Parser = scene_rdl2::grid_util::Parser;
    using Arg = scene_rdl2::grid_util::Arg;

    void parserConfigure(Parser &parser) override;

    mutable std::mutex mMutexRenderContext;
    RenderContext *mRenderContext;
};

} // namespace rndr
} // namespace moonray

