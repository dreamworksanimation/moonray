// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/rndr/RenderOptions.h>
#include <moonray/rendering/rndr/Types.h>

#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>

#include <sstream>
#include <vector>


namespace scene_rdl2 {
namespace rdl2 {
class RenderOutput;
class SceneObject;
}
}

namespace moonray {
class ChangeWatcher;

namespace rndr {
class RenderContext;
class RenderOutputDriver;
}

namespace pbr {
class CryptomatteBuffer;
class DeepBuffer;
}

class RaasApplication
{
public:
    RaasApplication();
    virtual ~RaasApplication() {}

    int main(int argc, char** argv);

protected:
    virtual void parseOptions() = 0;
    virtual void run() = 0;

    void printStatusLine(rndr::RenderContext& renderContext, double startTime, bool done);

    void parseOptions(bool guiMode);

    void logInitMessages();
    std::string secStr(double etaSec) const;

    int mArgc;
    char** mArgv;
    std::stringstream mInitMessages;
    rndr::RenderOptions mOptions;

    double mNextLogProgressTime;
    double mNextLogProgressPercentage;
};


int // return 0 on success, 1 on failure
writeImageWithMessage(const scene_rdl2::fb_util::RenderBuffer* frame,
                      const std::string& filename,
                      const scene_rdl2::rdl2::SceneObject *metadata,
                      const scene_rdl2::math::HalfOpenViewport& aperture,
                      const scene_rdl2::math::HalfOpenViewport& region);

int // return 0 on success, 1 on failure
writeRenderOutputsWithMessages(const rndr::RenderOutputDriver *renderOutputs,
                               const pbr::DeepBuffer *deepBuffer,
                               pbr::CryptomatteBuffer *cryptomatteBuffer,
                               const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                               const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                               const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers);

void
watchShaderDsos(ChangeWatcher& watcher,
                const rndr::RenderContext& renderContext);

} // namespace moonray

