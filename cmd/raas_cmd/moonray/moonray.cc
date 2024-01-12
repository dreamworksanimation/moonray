// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/application/RaasApplication.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/rndr/PixelBufferUtils.h>
#include <moonray/rendering/rndr/RenderContext.h>
#include <moonray/rendering/rndr/RenderDriver.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Files.h>

#include <cstdlib>
#include <iostream>

using scene_rdl2::logging::Logger;

namespace moonray {

class RaasCommandLineApplication : public RaasApplication
{
public:
    RaasCommandLineApplication() : RaasApplication() {}
    ~RaasCommandLineApplication() {}
protected:
    void parseOptions();
    void render(rndr::RenderContext & renderContext);
    void renderOutput(rndr::RenderContext &renderContext);
    void run();
};


void
RaasCommandLineApplication::parseOptions()
{
    RaasApplication::parseOptions(false);
}

void
RaasCommandLineApplication::render(rndr::RenderContext & renderContext)
{
    // Don't even bother if we can't write the output file in non interactive mode

    scene_rdl2::rdl2::SceneVariables& sceneVars = renderContext.getSceneContext().getSceneVariables();

    if (!scene_rdl2::util::writeTest(sceneVars.get(scene_rdl2::rdl2::SceneVariables::sOutputFile), true)) {
        std::ostringstream oss;
        oss << "Output file '" <<
            sceneVars.get(scene_rdl2::rdl2::SceneVariables::sOutputFile) <<
            "' is not writable!";
        scene_rdl2::Logger::error(oss.str());
        throw scene_rdl2::except::IoError(oss.str());
    }

    std::shared_ptr<rndr::RenderDriver> driver = rndr::getRenderDriver();

    renderContext.startFrame();

    while (!renderContext.isFrameComplete()) {
        // Poll for completion every 100 ms.
        usleep(100000);
        // TODO: Do we want a 'verbosity' setting.  Maybe we don't want to print status always
        printStatusLine(renderContext, driver->getLastFrameMcrtStartTime(), false);
    }
    // Inform the status line that we're done rendering.
    printStatusLine(renderContext, driver->getLastFrameMcrtStartTime(), true);

    renderContext.stopFrame();
    renderOutput(renderContext);
}

void
RaasCommandLineApplication::renderOutput(rndr::RenderContext &renderContext)
{
    scene_rdl2::fb_util::RenderBuffer outputBuffer;
    renderContext.snapshotRenderBuffer(&outputBuffer, true, true);

    // write the main output file
    scene_rdl2::rdl2::SceneContext const &sceneContext = renderContext.getSceneContext();
    std::string outputFile =
        sceneContext.getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sOutputFile);
    const scene_rdl2::rdl2::SceneObject *metadata = sceneContext.getSceneVariables().getExrHeaderAttributes();
    const scene_rdl2::math::HalfOpenViewport aperture = renderContext.getRezedApertureWindow();
    const scene_rdl2::math::HalfOpenViewport region = renderContext.getRezedRegionWindow();

    int error = writeImageWithMessage(&outputBuffer, outputFile, metadata, aperture, region);

    // write any arbitrary RenderOutput objects
    const pbr::DeepBuffer *deepBuffer = renderContext.getDeepBuffer();
    pbr::CryptomatteBuffer *cryptomatteBuffer = renderContext.getCryptomatteBuffer();
    scene_rdl2::fb_util::HeatMapBuffer heatMapBuffer;
    renderContext.snapshotHeatMapBuffer(&heatMapBuffer, /*untile*/ true, /*parallel*/ true); // internally only do snapshot when it has heatmapAOV
    scene_rdl2::fb_util::FloatBuffer weightBuffer;
    renderContext.snapshotWeightBuffer(&weightBuffer, /*untile*/ true, /*parallel*/ true); // internally only do snapshot when it has weightAOV
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer> aovBuffers;
    renderContext.snapshotAovBuffers(aovBuffers, /*untile*/ true, /*parallel*/ true);
    scene_rdl2::fb_util::RenderBuffer renderBufferOdd;
    renderContext.snapshotRenderBufferOdd(&renderBufferOdd, /*untile*/ true, /*parallel*/ true);
    std::vector<scene_rdl2::fb_util::VariablePixelBuffer> displayFilterBuffers;
    renderContext.snapshotDisplayFilterBuffers(displayFilterBuffers, /*untile*/ true, /*parallel*/ true);
    error += writeRenderOutputsWithMessages(renderContext.getRenderOutputDriver(),
                                            deepBuffer, cryptomatteBuffer, &heatMapBuffer,
                                            &weightBuffer, &renderBufferOdd, aovBuffers,
                                            displayFilterBuffers);

    // throw a file io error if anything failed to write, this will cause main to
    // exit with a non-zero error code.
    if (error) {
        throw scene_rdl2::except::IoError("Failed to write output images");
    }
}

void
RaasCommandLineApplication::run()
{
    // Run global init (creates a RenderDriver) This *must* be called on the same thread we
    // intend to call RenderContext::startFrame from.
    rndr::initGlobalDriver(mOptions);

    logInitMessages();

    // Create a RenderContext. Since the RenderContext internally holds a
    // ref on the RenderDriver, scope it to this block so it gets destroyed
    // before we call rndr::cleanUpRenderDriver.
    {
        rndr::RenderContext renderContext(mOptions, &mInitMessages);
#ifdef ENABLE_ATHENA_LOGGING // option defined by build system
        constexpr auto loggingConfig = rndr::RenderContext::LoggingConfiguration::ATHENA_ENABLED;
#else
        constexpr auto loggingConfig = rndr::RenderContext::LoggingConfiguration::ATHENA_DISABLED;
#endif
        renderContext.initialize(mInitMessages, loggingConfig);

        // TODO: allow progressive mode rendering in moonray also.
        renderContext.setRenderMode(rndr::RenderMode::BATCH);

        render(renderContext);

        for (const std::string & deltasFile : mOptions.getDeltasFiles()) {
            Logger::info("Applying deltas from '" + deltasFile + "'.");
            renderContext.updateScene(deltasFile);
            render(renderContext);
        }
    }

    rndr::cleanUpGlobalDriver();
}

} // namespace

int main(int argc, char* argv[])
{
    moonray::RaasCommandLineApplication app;
    try {
        return app.main(argc, argv);
    } catch (const std::exception& e) {
        Logger::error(e.what());
        std::exit(EXIT_FAILURE);
    }
}

