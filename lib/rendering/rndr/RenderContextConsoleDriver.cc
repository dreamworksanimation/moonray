// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RenderContextConsoleDriver.h"
#include "RenderContext.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/texturing/sampler/TextureTLState.h>

namespace moonray {
namespace rndr {

void
RenderContextConsoleDriver::setRenderContext(RenderContext *renderContext)
{
    std::lock_guard<std::mutex> lock(mMutexRenderContext);
    mRenderContext = renderContext;
}

//-----------------------------------------------------------------------------------------

void    
RenderContextConsoleDriver::parserConfigure(Parser &parser)
{
    parser.description("renderContext console command");

    parser.opt("renderContext", "...command...", "renderContext command",
               [&](Arg &arg) -> bool {
                   std::lock_guard<std::mutex> lock(mMutexRenderContext);
                   if (!mRenderContext) return arg.msg("renderContext is nullptr\n");
                   return mRenderContext->getParser().main(arg.childArg());
               });
    parser.opt("invalidateAllTexture", "", "invalidate all texture resources",
               [&](Arg &arg) -> bool {
                   std::lock_guard<std::mutex> lock(mMutexRenderContext);
                   if (!mRenderContext) return arg.msg("renderContext is nullptr\n");

                   bool restart = false;
                   if (mRenderContext->isFrameRendering()) {
                       mRenderContext->stopFrame(); // blocking
                       restart = true;
                   }
                   arg.msg("invalidate start ...");
                   mRenderContext->invalidateAllTextureResources();
                   arg.msg(" done\n");
                   if (restart) {
                       mRenderContext->setForceCallStartFrame();
                   }
                   return true;
               });
    parser.opt("shmFbOutput", "...command...", "shmFbOutput command (only works moonray_gui)",
               [&](Arg& arg) {
                   if (getShmFbOutput()) {
                       return getShmFbOutput()->getParser().main(arg.childArg());
                   } else {
                       return arg.msg("ShmFbOutput is empty\n");
                   }
               });
}

//==========================================================================================

RenderContextConsoleDriver::RenderContextConsoleDriverShPtr gRenderContextConsoleDriver;
std::shared_ptr<scene_rdl2::grid_util::ShmFbOutput> gShmFbOutput;

// static function    
void
RenderContextConsoleDriver::init(int port)
{
    // We only activate the debug console when we specify proper port number (0 or positive).
    if (port < 0) return;

    MOONRAY_THREADSAFE_STATIC_WRITE(gRenderContextConsoleDriver.reset(new RenderContextConsoleDriver()));
    gRenderContextConsoleDriver->initialize(port); // boot driver thread and setup port
}

// static function
RenderContextConsoleDriver::RenderContextConsoleDriverShPtr
RenderContextConsoleDriver::get()
{
    return gRenderContextConsoleDriver;
}

// static function
void
RenderContextConsoleDriver::setShmFbOutput(std::shared_ptr<scene_rdl2::grid_util::ShmFbOutput> shmFbOutput)
{
    gShmFbOutput = shmFbOutput;
}

// static function
RenderContextConsoleDriver::ShmFbOutputShPtr
RenderContextConsoleDriver::getShmFbOutput()
{
    return gShmFbOutput;
}

} // namespace rndr
} // namespace moonray

