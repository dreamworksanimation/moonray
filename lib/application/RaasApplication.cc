// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "RaasApplication.h"
#include "ChangeWatcher.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/common/time/Timer.h>

#include <moonray/rendering/mcrt_common/Util.h>
#include <moonray/rendering/rndr/ImageWriteCache.h>
#include <moonray/rendering/rndr/ImageWriteDriver.h>
#include <moonray/rendering/rndr/rndr.h>
#include <moonray/rendering/rndr/RenderOutputDriver.h>
#include <moonray/rendering/rndr/RenderProgressEstimation.h>
#include <moonray/rendering/rndr/RenderStatistics.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <string>
#include <sys/ioctl.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

// In seconds.
#define LOG_PROGRESS_TIME_INTERVAL  60.0

namespace moonray {


using namespace scene_rdl2::math;
using namespace scene_rdl2;


RaasApplication::RaasApplication()
    : mArgc(0)
    , mArgv(nullptr)
    , mNextLogProgressTime(0.0)
    , mNextLogProgressPercentage(0.0)
{
}

void
RaasApplication::parseOptions(bool guiMode)
{
    // Check for no flags or help flag.
    if (mArgc == 1 || std::string(mArgv[1]) == "-h" || std::string(mArgv[1]) == "-help" || std::string(mArgv[1]) == "--help") {
        std::cerr << rndr::RenderOptions::getUsageMessage(mArgv[0], guiMode) << std::endl;
        const int status = (mArgc == 1 ? EXIT_FAILURE : EXIT_SUCCESS);
        exit(status);
    }

    // Initialize our logging
    scene_rdl2::logging::Logger::init();

    // Parse the command line.
    mOptions.parseFromCommandLine(mArgc, mArgv);

    // Check for scene files and "-in" flag because
    // calls to main()::run() without any scene files will cause 
    // Moonray to exit with 'Error: scene context contains no cameras' 
    if (mOptions.getSceneFiles().empty()) {
        std::cerr << "No scene file specified. Did you forget the `-in` argument?. " << std::endl;
        exit(EXIT_FAILURE);
    } 
}

void
RaasApplication::logInitMessages()
{
    mInitMessages << "Using OpenImageIO Texture System\n";
}

void 
stackTraceHandler(int sig) 
{
    std::string crashReason;
    switch (sig) {
        break;
    case SIGSEGV:
        mcrt_common::debugPrintCallstack("\nSIGSEGV(segfault) callstack");
        crashReason = "SIGSEGV (Segmentation fault)";
        break;
    case SIGILL:
        mcrt_common::debugPrintCallstack("\nSIGILL(illegal instruction) callstack");
        crashReason = "SIGILL (Illegal instruction)";
        break;
    case SIGFPE:
        mcrt_common::debugPrintCallstack("\nSIGFPE(float point exception) callstack");
        crashReason = "SIGFPE (Floating point exception)";
        break;
    default:
        mcrt_common::debugPrintCallstack(nullptr);
        break;
    }

    signal(SIGABRT, SIG_DFL);
    abort();
}

int
RaasApplication::main(int argc, char** argv)
{
    // register stack trace dumping
    signal(SIGABRT, stackTraceHandler); // abort
    signal(SIGSEGV, stackTraceHandler); // segfault
    signal(SIGILL, stackTraceHandler);  // illegal instruction
    signal(SIGFPE, stackTraceHandler);  // floating point exception 

    mArgc = argc;
    mArgv = argv;
    parseOptions();

    // Virtual
    run();

    return EXIT_SUCCESS;
}

void
RaasApplication::printStatusLine(rndr::RenderContext& renderContext, double startTime, bool done)
{
    bool estimationActive =
        renderContext.getSceneContext().getCheckpointActive() || renderContext.getSceneContext().getResumeRender();

    rndr::RenderProgressEstimation *progEst = renderContext.getFrameProgressEstimation();

    float progressFraction = 0.0f;
    {
        std::size_t total = 0;
        progressFraction = renderContext.getFrameProgressFraction(nullptr, &total);
        if (total == 0) {
            return;
        }
        progressFraction = scene_rdl2::math::clamp(progressFraction, 0.0f, 1.0f);
    }

    //------------------------------

    const float pctComplete = progressFraction * 100.f;

    const double currentTime = scene_rdl2::util::getSeconds();
    const double elapsedMcrtTime = currentTime - startTime;

    // Should we log progress?
    bool logProgress = false;
    if (elapsedMcrtTime >= mNextLogProgressTime || pctComplete >= mNextLogProgressPercentage) {
        mNextLogProgressTime = elapsedMcrtTime + LOG_PROGRESS_TIME_INTERVAL;
        mNextLogProgressPercentage = pctComplete + 1.0;
        logProgress = true;
    }

    // In a live terminal, print the progress bar
    if (isatty(STDOUT_FILENO)) {

        // Get the terminal width.
        winsize win;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &win);

        if (win.ws_col == 0) {
            win.ws_col = 80; // situation like emacs shell
        }

        // Compute progress bar layout.
        int barWidth = win.ws_col
                       - 6  // "  [/] " (spinner)
                       - 9  // "Rendering"
                       - 4  // " [" + "] " (bar borders and spacing)
                       - 7  // "100.0% "
                       - 14 // "xx:xx:xx ETA  " Estimated Time Of Accomplishment
                       ;

        // Have we underflowed?
        if (barWidth < 0) {
            barWidth = 0;
        }

        int fullBars = progressFraction * barWidth + 0.5f;
        static int spinnerIdx = 0;
        static char spinnerAnim[] = "-\\|/";

        char spinner = (done) ? '+' : spinnerAnim[spinnerIdx];

        // Print the status line.
        if (estimationActive) {
            std::printf("  [%c] %s [", spinner, progEst->getModeBanner().c_str());
        } else {
            std::printf("  [%c] Rendering [", spinner);
        }
        for (int i = 0; i < fullBars; ++i) {
            std::printf("=");
        }
        for (int i = 0; i < barWidth - fullBars; ++i) {
            std::printf(" ");
        }
        if (done) {
            std::printf("] 100.0%% \n");
        } else {
            if (estimationActive) {
                if (progEst->isAdaptiveSampling()) {
                    std::printf("] %5.1f%%\r", pctComplete);
                } else {
                    std::printf("] %5.1f%% %s ETA\r", pctComplete,
                                secStr(progEst->getEstimatedSecOfCompletionFromNow()).c_str());
                }
            } else {
                std::printf("] %5.1f%% \r", pctComplete);
            }
            std::fflush(stdout);
        }

        MOONRAY_THREADSAFE_STATIC_WRITE(spinnerIdx = ++spinnerIdx % std::strlen(spinnerAnim));

    } else if (logProgress) {
        std::stringstream ss;
        ss << "Rendering [" << std::fixed << std::setw(3) << std::setprecision(0) << pctComplete << "%]";            
        if (estimationActive) {
            ss << " " << secStr(progEst->getEstimatedSecOfCompletionFromNow()) << " ETA";
        }
        renderContext.getSceneRenderStats().logString(ss.str());
    }

    // Update Athena.
    if (logProgress) {
        renderContext.getSceneRenderStats().logMcrtProgress(float(elapsedMcrtTime), pctComplete);
    }
}

std::string
RaasApplication::secStr(double sec) const
{
    std::ostringstream ostr;

    if (sec <= 0.0) {
        ostr << "- : -";
    } else {
        unsigned secInt = static_cast<unsigned>(sec);

        if (secInt >= 3600) { // hour
            ostr << std::setw(2) << std::setfill('0') << (secInt / 3600) << ':';
            secInt %= 3600;
        }

        if (secInt >= 60) { // minute
            ostr << std::setw(2) << std::setfill('0') << (secInt / 60) << ':';
            secInt %= 60;
        } else {
            ostr << "00:";
        }

        ostr << std::setw(2) << std::setfill('0') << secInt; // secound
    }
    
    return ostr.str();
}

int
writeImageWithMessage(const scene_rdl2::fb_util::RenderBuffer* frame,
                      const std::string& filename,
                      const scene_rdl2::rdl2::SceneObject *metadata,
                      const scene_rdl2::math::HalfOpenViewport& aperture,
                      const scene_rdl2::math::HalfOpenViewport& region)
{
    int error = 0;

    // disable write if filename is empty
    if (!filename.empty()) {
        // Write the final frame out.
        try {
            rndr::writePixelBuffer(*frame, filename, metadata, aperture, region);
            std::cout << "Wrote " << filename << std::endl;
        } catch (...) {
            scene_rdl2::Logger::error("Failed to write out ", filename);
            error = 1;
        }
    }

    return error;
}

int
writeRenderOutputsWithMessages(const rndr::RenderOutputDriver *rod,
                               const pbr::DeepBuffer *deepBuffer,
                               pbr::CryptomatteBuffer *cryptomatteBuffer,
                               const scene_rdl2::fb_util::HeatMapBuffer *heatMap,
                               const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                               const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                               const std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers)
{
    int err = 0;

    const rndr::ImageWriteCache *lastImageWriteCache =
        rndr::ImageWriteDriver::get()->getLastImageWriteCache();
    if (lastImageWriteCache && lastImageWriteCache->getTwoStageOutput()) {
        //
        // Checkpoint render with two stage output mode
        //
        // We already have lastImageWriteCache which was created by checkpoint output with two stage
        // output mode (i.e. checkpoint was created as tmpfile first then copied to final location).
        // This last checkpoint output is exactly same as final output.
        // We only need copy instead of generate file here.
        if (!lastImageWriteCache->allFinalizeFinalFile()) err = 1;

    } else {
        //
        // Non checkpoint render or checkpoint render w/ two stage output off
        // This case, we don't have any previously written temporary file data and we
        // need to output final data by standard way.
        //
        rndr::ImageWriteDriver::ImageWriteCacheUqPtr cache;
        if (!rod->requiresDeepBuffer() && rndr::ImageWriteDriver::get()->getTwoStageOutput()) {
            // Two stage output mode for final output w/ STD mode
            cache = rndr::ImageWriteDriver::get()->newImageWriteCache(rod);
            cache->setTwoStageOutput(true);
        } else {
            // standard final output (STD mode) with out two stage output
            MNRY_ASSERT(!cache);
        }
            
        rod->writeFinal(deepBuffer,
                        cryptomatteBuffer,
                        heatMap,
                        weightBuffer,
                        renderBufferOdd,
                        aovBuffers,
                        displayFilterBuffers,
                        cache.get());
        err = rod->loggingErrorAndInfo(cache.get())? 0: 1;

        if (cache && cache->getTwoStageOutput()) {
            if (!cache->allFinalizeFinalFile()) err = 1;
        }
    }

    return err;
}


void
watchShaderDsos(ChangeWatcher& watcher, const rndr::RenderContext& renderContext)
{
    const scene_rdl2::rdl2::SceneContext& sceneContext = renderContext.getSceneContext();

    // For each active SceneClass, install a file watcher for its DSO.
    for (auto iter = sceneContext.beginSceneClass(); iter != sceneContext.endSceneClass(); ++iter) {
        const scene_rdl2::rdl2::SceneClass* sceneClass = iter->second;
        std::string sourcePath(sceneClass->getSourcePath());
        if (!sourcePath.empty()) {
            watcher.watchFile(sourcePath);
        }
    }
}

} // namespace moonray

