// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "McrtRtComputation.h"
#include "McrtRtComputationStatistics.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/pbr/core/Statistics.h>
#include <moonray/rendering/rndr/RenderDriver.h>
#include <moonray/rendering/rndr/rndr.h>
#include <moonray/rendering/rndr/TileScheduler.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>

#include <moonray/client/protocol/viewport_message/ViewportMessage.h>
#include <moonray/common/log/logging.h>
#include <moonray/common/object/Object.h>
#include <moonray/common/web/client/HttpRequest.h>
#include <moonray/engine/messages/base_frame/BaseFrame.h>
#include <moonray/engine/messages/json_message/JSONUtils.h>
#include <moonray/engine/messages/partial_frame/PartialFrame.h>
#include <moonray/engine/messages/rdl_message_lefteye/RDLMessage_LeftEye.h>
#include <moonray/engine/messages/rdl_message_righteye/RDLMessage_RightEye.h>
#include <moonray/engine/messages/render_messages/RenderMessages.h>
#include <engine/computation/Computation.h>
#include <scene_rdl2/common/fb_util/PixelBufferUtilsGamma8bit.h>
#include <scene_rdl2/common/fb_util/SparseTiledPixelBuffer.h>
#include <scene_rdl2/common/rec_time/RecTime.h>
#include <scene_rdl2/render/util/GetEnv.h>
#include <scene_rdl2/render/util/Strings.h>
#include <scene_rdl2/scene/rdl2/BinaryWriter.h>
#include <scene_rdl2/scene/rdl2/Camera.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/Light.h>
#include <scene_rdl2/scene/rdl2/Material.h>

#include <json/reader.h>
#include <json/value.h>
#include <logging_base/macros.h>

#include <cstdlib>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#define USE_RAAS_DEBUG_FILENAME

using moonray::engine::Computation;
using moonray::network::BaseFrame;
using moonray::network::GenericMessage;
using moonray::network::GeometryData;
using moonray::network::JSONMessage;
using moonray::network::JSONUtils::ValueVector;
using moonray::network::Message;
using moonray::network::RDLMessage;
using moonray::network::RDLMessage_LeftEye;
using moonray::network::RDLMessage_RightEye;
using moonray::network::RenderedFrame;
using moonray::network::RenderMessages;
using moonray::network::PartialFrame;
using moonray::network::ViewportMessage;


CREATOR_FUNC(moonray::mcrt_rt_computation::McrtRtComputation);

//------------------------------------------------------------------------------

#define SHOW_TIMING_INFO        // show interval timing info for onIdle() and onMessage()
//#define SHOW_PRIMARY_RAY_TOTAL  // show primary ray total count info for frame 0 ~ 99
#define SHOW_BUSYMESSAGE_INFO   // show multiple onMessage execution status

#ifdef SHOW_TIMING_INFO
namespace moonray {
namespace mcrt_rt_computation {
    static McrtRtComputationStatistics ms;
} // namespace mcrt_rt_computation
} // namespace moonray
#define TEST_SHOW_TIMING_INFO( f ) f
#else // else SHOW_TIMING_INFO
#define TEST_SHOW_TIMING_INFO( f )
#endif // end !SHOW_TIMING_INFO

#ifdef SHOW_PRIMARY_RAY_TOTAL
namespace moonray {
namespace mcrt_rt_computation {
    static StatisticsPrimaryRayTotal sr;
} // namespace mcrt_rt_computation
} // namespace moonray
#define TEST_SHOW_PRIMARY_RAY_TOTAL( f ) f
#else  // else SHOW_PRIMARY_RAY_TOTAL
#define TEST_SHOW_PRIMARY_RAY_TOTAL( f )
#endif // end !SHOW_PRIMARY_RAY_TOTAL

#ifdef SHOW_BUSYMESSAGE_INFO
namespace moonray {
namespace mcrt_rt_computation {
    static StatisticsBusyMessage sb;
} // namespace mcrt_rt_computation
} // namespace moonray
#define TEST_SHOW_BUSYMESSAGE_INFO( f ) f
#else // else SHOW_BUSYMESSAGE_INFO
#define TEST_SHOW_BUSYMESSAGE_INFO( f )
#endif // end !SHOW_BUSYMESSAGE_INFO

//------------------------------------------------------------------------------

namespace moonray {

namespace {

    // Configuration constants
    const std::string sConfigScene = "scene";
    const std::string sConfigDsopath = "dsopath";
    const std::string sConfigEnableDepthBuffer = "enableDepthBuffer";
    const std::string sConfigCamera = "camera";
    const std::string sConfigLayer = "layer";
    const std::string sConfigFps = "fps";
    const std::string sConfigGamma = "applyGamma";
    const std::string sConfigNumMachines = "numMachines";
    const std::string sConfigMachineId = "machineId";
    const std::string sConfigFrameGating = "frameGating";
    const std::string sConfigSendLogMessages = "sendLogMessages";
    const std::string sImageEncoding = "imageEncoding";
    const std::string sFastGeometryUpdate = "fastGeometryUpdate";
    const std::string sRenderedEye = "renderedEye";
    const std::string sRenderMode = "renderMode";
    const std::string sApplicationMode = "applicationMode";
    const std::string sExecMode = "exec_mode";
    const std::string sRayStreaming = "ray_streaming";
    const std::string sTextureCacheSize = "textureCacheSize";
    const std::string sTextureSystem = "textureSystem"; 

    const std::string AOV_BEAUTY = "beauty";
    const std::string AOV_DEPTH = "depth";

    // TODO: This function duplicated in McrtMergeComputation.cc, where is a
    //       good place to shared it?
    inline fb_util::VariablePixelBuffer::Format
    convertImageEncoding(BaseFrame::ImageEncoding encoding)
    {
        switch (encoding)
        {
        case BaseFrame::ENCODING_RGBA8:     return fb_util::VariablePixelBuffer::RGBA8888;
        case BaseFrame::ENCODING_RGB888:    return fb_util::VariablePixelBuffer::RGB888;
        default:                            MNRY_ASSERT(0);
        }
        return fb_util::VariablePixelBuffer::UNINITIALIZED;
    }


    // KEY used to indicate that a RenderSetupMessage originated from an upstream computation, and not a client
    // Used to get around the lack of message intents: http://jira.anim.dreamworks.com/browse/NOVADEV-985
    const std::string RENDER_SETUP_KEY = "776CD313-6D4B-40A4-82D2-C61F2FD055A9";
}


namespace mcrt_rt_computation {

McrtRtComputation::McrtRtComputation() :
    mOptions(),
    mImageEncoding(RenderedFrame::ImageEncoding::ENCODING_RGB888),
    mRenderContext(nullptr),
    mGeometryUpdate(nullptr),
    mFrameCount(0),
    mFpsSet(false),
    mFps(5.0f),
    mLastFps(-1.0),             // initial special value
    mDispatchGatesFrame(false),
    mLastTime(0.0),
    mReceivedSnapshotRequest(false),
    mReceivedFinalPixelInfoBuffer(false),
    mSentFinalPixelInfoBuffer(false),
    mFirstFrame(true),
    mCompleteRendering(false),
    mApplyGamma(true),
    mFrameStarted(false),
    mNumMachinesOverride(-1),
    mMachineIdOverride(-1),
    mRenderTimestamp(0),
    mLastSnapshotTimestamp(0),
    mLastFilmActivity(0),
    mEye(moonray::network::RenderedFrame::RenderedEye::RIGHT_EYE),
    mTotalReceivedGeometryData(0),
    mTotalSkipGeometryData(0),
    mGeoFrameId(0),
    mRenderFrameId(0),
    mGeoUpdateMode(true),
    mResetCounter(true),
    mRTController(24.0)
{
    std::cout << ">>> McrtRtComputation.cc constructor() ..." << std::endl;

#ifdef USE_RAAS_DEBUG_FILENAME
    if (const char* const delayFilename = scene_rdl2::util::getenv<const char*>("RAAS_DEBUG_FILENAME")) {
        while (access(delayFilename, R_OK)) {
            unsigned int const DELTA_TIME = 1;
            sleep(DELTA_TIME);
        }
    }
#endif

    // Preallocate some space for several RDL updates.
    mUpdates.reserve(10);
}

void
McrtRtComputation::setFps(const float fps)
{
    mFps = fps;
    mFpsSet = true;
    mOptions.setFps(mFps);
}

void
McrtRtComputation::configure(const object::Object& aConfig)
{   
    MOONRAY_LOG_INFO("McrtRt computation ...");

    // Turn off the use of the depth buffer by default.
    mOptions.setGeneratePixelInfo(false);

    // Optionally, enable the depth buffer
    if (!aConfig[sConfigEnableDepthBuffer].isNull()) {
        mOptions.setGeneratePixelInfo(aConfig[sConfigEnableDepthBuffer].value().asInt());
    }

    // Override defaults with settings from the config.
    if (!aConfig[sConfigScene].isNull()) {
        mOptions.setSceneFiles({(const char*)aConfig[sConfigScene]});
    }
    if (!aConfig[sConfigDsopath].isNull()) {
        mOptions.setDsoPath((const char*)aConfig[sConfigDsopath]);
    }
    if (!aConfig[sConfigFps].isNull()) {
        float currFps = (float)aConfig[sConfigFps];
        setFps(currFps);
    }
    if (!aConfig[sConfigNumMachines].isNull()) {
        mNumMachinesOverride = aConfig[sConfigNumMachines].value().asInt();
    }
    if (!aConfig[sConfigMachineId].isNull()) {
        mMachineIdOverride = aConfig[sConfigMachineId].value().asInt();
    }

    // mcrt wants to run with hyperthreading. Enable hyperthreading if it is
    // available and increase the number of threads.
    int threads = maxCores();
    int perCore = threadsPerCore();
    if (perCore > 1) {
        setHyperthreading(true);
        threads *= perCore;
    }
    mOptions.setThreads(threads);
    if (!aConfig[sImageEncoding].isNull()) {
        mImageEncoding = static_cast<BaseFrame::ImageEncoding>(static_cast<int>((aConfig[sImageEncoding])));
    }
    if (!aConfig[sConfigGamma].isNull()) {
        mApplyGamma = aConfig[sConfigGamma];
    }
    if (!aConfig[sConfigSendLogMessages].isNull()) {
        mSendLogMessages = aConfig[sConfigSendLogMessages];
    }
    if (!aConfig[sConfigFrameGating].isNull()) {
        mDispatchGatesFrame = aConfig[sConfigFrameGating].value().asInt();
    }

    mOptions.setFastGeometry();

    if (!aConfig[sRenderedEye].isNull()) {
        //  by default eye is set to right so only change it if we have a left
        //  eye setting.
        if (!strcmp("left", (const char*) aConfig[sRenderedEye])) {
            mEye = moonray::network::RenderedFrame::RenderedEye::LEFT_EYE;
        } else if (!strcmp("right", (const char*) aConfig[sRenderedEye])) {
            mEye = moonray::network::RenderedFrame::RenderedEye::RIGHT_EYE;
        } else if (!strcmp("center", (const char*) aConfig[sRenderedEye])) {
            mEye = moonray::network::RenderedFrame::RenderedEye::CENTER_EYE;
        }
    }
    mOptions.setRenderMode(rndr::RenderMode::PROGRESSIVE);
    if (!aConfig[sRenderMode].isNull()) {
        // we only set the mode to realtime if we find the config option
        // this ensures the previous default behavior of PROGRESSIVE rendering
        if (!strcmp("realtime", (const char*) aConfig[sRenderMode])) {
            mOptions.setRenderMode(rndr::RenderMode::REALTIME);
        } else {
            MOONRAY_LOG_INFO("Unrecognized render mode, setting to default Progressive Mode");
        }
    }

    // Undefined mode is backward compatible with previous behavior and has been added as such.
    mOptions.setApplicationMode(rndr::ApplicationMode::UNDEFINED);
    if (!aConfig[sApplicationMode].isNull()) {
        if (aConfig[sApplicationMode].value().asInt() == 1) {
            mOptions.setApplicationMode(rndr::ApplicationMode::MOTIONCAPTURE);
        } else {
            MOONRAY_LOG_ERROR("APPLICATION MODE SET TO UNDEFIND");
        }
    }

    // "bundled" is deprecated, it's now just an alias for vectorized.
    std::string execMode = "scalar";
    bool vectorized = false;
    if (!aConfig[sExecMode].isNull()) {
        execMode = aConfig[sExecMode].value().asString();
    }
    mOptions.setDesiredExecutionMode(execMode);

    {
        if (!aConfig[sRayStreaming].isNull()) {
            const bool rayStreaming = aConfig[sRayStreaming].value().asBool();
            bool setFlag = rayStreaming && vectorized;
            mOptions.setRayStreaming(setFlag);
        }
    }

    if (!aConfig[sTextureCacheSize].isNull()) {
        mOptions.setTextureCacheSizeMb(aConfig[sTextureCacheSize].value().asInt());
    }

#ifdef DEBUG_CONSOLE_MODE
    mDebugConsole.open(20000, this);
#endif  // end DEBUG_CONSOLE_MODE

    SHOW_TIMING_INFO(ms.mLap.setFileDumpId(mMachineIdOverride));
}

void
McrtRtComputation::append(log::Logger::Level level, const std::string& message)
{
    if (mSendLogMessages) {
        JSONMessage::Ptr logMsg = RenderMessages::createLogMessage(level, message);
        send(logMsg);
    }
}

McrtRtComputation::~McrtRtComputation()
{
}

void
McrtRtComputation::onStart()
{
    // Setup the logger only after the client has connected
    using namespace std::placeholders;
    log::Logger::instance().logEvent += std::bind(&McrtRtComputation::append, this, _1, _2);

    // Run global init (creates a RenderDriver) This *must* be called on the same thread we
    // intend to call RenderContext::startFrame from.
    rndr::initGlobalDriver(mOptions);

    // TODO: Move this to onRenderSetupMessage and make all the clients send the message
    mRenderContext.reset(new rndr::RenderContext(mOptions));

    if (!mDispatchGatesFrame) {
        mRTController.setConstantFps(mFps); // non frameGating mode need to set constant fps to RTController
    }
}

void
McrtRtComputation::onStop()
{
    // Shutdown the renderer
    mRenderContext = nullptr;
    rndr::cleanUpGlobalDriver();

    // Remove the log appender
    using namespace std::placeholders;
    log::Logger::instance().logEvent -= std::bind(&McrtRtComputation::append, this, _1, _2);
}

void
McrtRtComputation::applyConfigOverrides()
{
    // we now honor the setting from the config file and can now support
    // a render mode besides progressive
    mRenderContext->setRenderMode(mOptions.getRenderMode());

    rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
    
    rdl2::SceneObject::UpdateGuard guard(&sceneVars);
    
    if (mNumMachinesOverride >= 0) {
        sceneVars.set(rdl2::SceneVariables::sNumMachines, mNumMachinesOverride);
    }
    if (mMachineIdOverride >= 0) {
        sceneVars.set(rdl2::SceneVariables::sMachineId, mMachineIdOverride);
    }

    // TODO: This tile scheduler type is hardcoded here since the merge computation
    // assumes it. We should really be sending over the scheduling pattern
    // the partial frames were generated with in a message to avoid having
    // to hardcode anything.
    int tileSchedulerType = rndr::TileScheduler::SPIRAL_SQUARE;
    sceneVars.set(rdl2::SceneVariables::sBatchTileOrder, tileSchedulerType);
    sceneVars.set(rdl2::SceneVariables::sProgressiveTileOrder, tileSchedulerType);
}

bool 
McrtRtComputation::fpsIntervalPassed()
{
    if (isMultiMachine() && mDispatchGatesFrame) {
        // In a multimachine setup, frame gating is handled upstream and
        // receiving an update indicates it's time to render the next frame;
        // Receiving a snapshot request indicates it's time to make another 
        // snapshot

        // Before finish fist frame, we have to always return w/ true condition. This required mocap
        // realtime rendering. But not tested under TORCH environment yet. Toshi (Mar/31/16)
        if (mFirstFrame) {
            return true;
        }

        if (!mGeometryUpdate && mUpdates.empty() && !mReceivedSnapshotRequest) {
            return false;
        }
    } else {
        // In a single machine setup, frame gating is handled here.
        double now = util::getSeconds();
        if (now - mLastTime < (1.0f / mFps)) {
            return false;
        }
        mLastTime = now;
    }

    return true;
}

void
McrtRtComputation::setFrameStateVariables()
{
    mReceivedFinalPixelInfoBuffer = false;
    mSentFinalPixelInfoBuffer = false;
    // Update the render timestamp
    mRenderTimestamp++;
    mLastFilmActivity = 0;
    // Mark frame as started so we send the appropriate status message
    mFrameStarted = true;
}

void
McrtRtComputation::processControlMessages()
{
    MNRY_ASSERT(mRenderContext && "Cannot control the render without a render context");
    for (auto iter = mRenderControlMessages.begin(); iter != mRenderControlMessages.end(); ++iter) {
        if (*iter == STOP && mRenderContext->isFrameRendering()) {
            MOONRAY_LOG_DEBUG("onIdle(); Stopping frame...");
            mRenderContext->stopFrame();
        } else  if (*iter == START && !mRenderContext->isFrameRendering()) {
            MOONRAY_LOG_DEBUG("onIdle(); Starting frame...");
            mRenderContext->startFrame();
            TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startEnd));
            setFrameStateVariables();
        }
    }
    mRenderControlMessages.clear();
}

void
McrtRtComputation::onIdle()
{
#   ifdef DEBUG_CONSOLE_MODE
    mDebugConsole.eval();
    if (mResetCounter) {
        TEST_SHOW_PRIMARY_RAY_TOTAL(sr.reset());
        TEST_SHOW_TIMING_INFO(ms.mLap.reset());
        if (mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME) {
            auto driver = rndr::getRenderDriver();
            driver->saveRealtimeStats();
            driver->resetRealtimeStats();
        }
        mResetCounter = false;
    }
#   endif // end DEBUG_CONSOLE_MODE    

    TEST_SHOW_TIMING_INFO(ms.mLap.passStartingLine());
    TEST_SHOW_BUSYMESSAGE_INFO(sb.onIdleUpdate());

    BaseFrame::Status status = BaseFrame::ERROR;

#ifdef RTT_TEST_MODE
    if (mGeometryUpdate) {
        processGeoUpdateAck();
    }
#endif // end RTT_TEST_MODE    

    // Process queues first

    // Do we have pending updates?
    bool haveUpdates = mGeometryUpdate || !mUpdates.empty();

    // If so, we don't want to process any render control messages because they supersede render control
    if (mRenderContext && !haveUpdates) {
        processControlMessages();
    }

    // Make sure enough time has passed since the last frame
    // additionally only use the fps gate if it was explicity set
    // otherwise we are counting on updates messages to trigger
    // a render frame restart.
    if (mRenderContext->getRenderMode() != rndr::RenderMode::REALTIME) {
        if (mFpsSet && !fpsIntervalPassed()) {
            return;
        }
    }

    // Application specific behavior for motion capture, we do not want to reset render unless updates exist.
    //     the implication here is that rendering fps can only go as fast as animation fps -- b.stastny
    if ((mOptions.getApplicationMode() == rndr::ApplicationMode::MOTIONCAPTURE) && (!mFirstFrame && !haveUpdates)) {
        if (mRenderContext->getRenderMode() != rndr::RenderMode::REALTIME) {
            return;
        }
    }

    if (mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME) {
        if (!mRenderContext->isFrameComplete()) {
            return;
        }
        // rendering is completed.
        
        if (!mRTController.isFrameComplete()) { // Main function to adjust timing.
            return;             // still inside gap interval. need to wait bit more
        }

        // Estimated result of gapIntervap value is crucial to change time budget for next frame.
        // So we passed value into renderDriver here for next frame.
        rndr::getRenderDriver()->setLastFrameUpdateDurationOffset(mRTController.getGapInterval());

        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startEnd));

        TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_frameInterval, mRTController.getFrameInterval()));
        TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_pureGap, mRTController.getPureGapInterval()));
        TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_overrun, mRTController.getOverrun()));
        TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_adjust, mRTController.getAdjustDuration()));
        {
            TEST_SHOW_PRIMARY_RAY_TOTAL(sr.update(rnder::getRenderDriver()->mTimeRec.getNumSamplesAll(), 5, 100));

            TEST_SHOW_TIMING_INFO(auto &timeRec = rndr::getRenderDriver()->getRenderFrameTimingRecord());
            TEST_SHOW_TIMING_INFO(ms.mLap.auxUInt64SectionAdd(ms.mIdAuxL_primRayTotal, timeRec.getNumSamplesAll()));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_overhead, timeRec.getTotalOverheadDuration()));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_active, timeRec.getTotalActiveDuration()));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxUInt64SectionAdd(ms.mIdAuxL_passesTotal,timeRec.getNumPassesRendered()));
        }
    }

    TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_whole));

    bool isRenderingOrFinished = mRenderContext && 
        (mRenderContext->isFrameRendering() || mRenderContext->isFrameComplete());
    // Make sure we are at least rendering or done rendering.

    if (isRenderingOrFinished) {

        // Use RenderContext to access low level Film in the RenderDriver
        // TODO: Send messages per film @see http://jira.anim.dreamworks.com/browse/MOONRAY-1542
        const unsigned filmActivity = mRenderContext->getFilmActivity();
        bool renderSamplesPending = (filmActivity != mLastFilmActivity);
        if (mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME) {
            renderSamplesPending = true;
        }

        // Make sure pixel information actually changes and we are ready to display
        bool isFrameReadyForDisplay = mRenderContext->isFrameReadyForDisplay();
        if (mOptions.getApplicationMode() == rndr::ApplicationMode::MOTIONCAPTURE) {
            if (mRenderContext->getRenderMode() != rndr::RenderMode::REALTIME) {
                if (isMultiMachine()) {
                    isFrameReadyForDisplay = true;    // always true even still coarse pass is not finished yet.
                }
            }
        }

        if (renderSamplesPending && isFrameReadyForDisplay && !mCompleteRendering) {
            TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_endStart));

            if (mOptions.getApplicationMode() == rndr::ApplicationMode::MOTIONCAPTURE) {
                if (mRenderContext->getRenderMode() != rndr::RenderMode::REALTIME) {
                    TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startEnd));
                    TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_stop));
                    mRenderContext->stopFrame();
                    TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_stop));
                }
            }

            // Update the snapshot timestamp
            mLastSnapshotTimestamp = mRenderTimestamp;
            mLastFilmActivity = filmActivity;

            // The first frame that we are going to send should have the STARTED status
            status = mFrameStarted ? BaseFrame::STARTED : BaseFrame::RENDERING;
            mFrameStarted = false;
            // We are ready to take some snapshots
            TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_snapshot));
            snapshotBuffers();
            TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_snapshot));

            // Decide on which machine type to process
            ++mFrameCount;
            if (isMultiMachine()) {
                processMultimachine(status);
            } else {
                processSingleMachine(status);
            }

            if (mRenderContext->getRenderMode() != rndr::RenderMode::REALTIME) {
                // We have to update primary ray total here when non realtime renderMode
                size_t total = mRenderContext->getPbrStatistics().getCounter(pbr::STATS_PIXEL_SAMPLES);
                TEST_SHOW_PRIMARY_RAY_TOTAL(sr.update(total, 5, 100));
                TEST_SHOW_TIMING_INFO(ms.mLap.auxUInt64SectionAdd(ms.mIdAuxL_primRayTotal, total));

                TEST_SHOW_TIMING_INFO(auto &timeRec = rndr::getRenderDriver()->getRenderFrameTimingRecord());
                TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_overhead, timeRec.getTotalOverheadDuration()));
                TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_active, timeRec.getTotalActiveDuration()));
                TEST_SHOW_TIMING_INFO(ms.mLap.auxUInt64SectionAdd(ms.mIdAuxL_passesTotal,timeRec.getNumPassesRendered()));

#ifdef SHOW_TIMING_INFO
                static double time = -1.0;
                double oldTime = time;
                MOONRAY_THREADSAFE_STATIC_WRITE(time = util::getSeconds(););
                double duration = (oldTime < 0.0)? 0.0: time - oldTime;
#endif
                TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_frameInterval, duration));
            }

            if (mRenderContext->isFrameComplete()) {
                mCompleteRendering = true;
            }
        }
    }

    // Apply updates if needed
    if (haveUpdates && mLastSnapshotTimestamp >= mRenderTimestamp ||
        mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME) {
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_start));
        applyUpdatesAndRestartRender();
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_start));
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_endStart));
    }

    TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_whole));
    TEST_SHOW_TIMING_INFO(if(ms.mLap.showLapInfo(mFps, [](const std::string &msg) { std::cerr << msg << std::endl; })) {
            float pct = 0.0;
            if (mTotalReceivedGeometryData > 100) {
                pct = (float)mTotalSkipGeometryData / (float)(mTotalReceivedGeometryData - 100) * 100.0f;
            }
            std::cout << "skip frame info {\n"
                      << "  total received geo:" << mTotalReceivedGeometryData - 100 << '\n'
                      << "      total skip geo:" << mTotalSkipGeometryData << " (" << pct << "%)\n"
                      << "}" << std::endl;
        });
}

#ifdef RTT_TEST_MODE
void
McrtRtComputation::processGeoUpdateAck()
{
    int geoFrameId = mGeoFrameId;

    std::ostringstream os;
    os << "geoMsgAck " << mMachineIdOverride << " " << geoFrameId;

    GenericMessage::Ptr geoUpdateAckMsg(new GenericMessage);
    geoUpdateAckMsg->mValue = os.str();
    send(geoUpdateAckMsg);

    //    std::cout << "McrtRtComputation.cc ack:>" << geoUpdateAckMsg->mValue << "<" << std::endl;
}
#endif // end RTT_TEST_MODE

void
McrtRtComputation::snapshotBuffers()
{
    // Untiling is done on the merge node for the multi machine case but
    // on this node if rendering is done on a single machine.
    bool untileDuringSnapshot = !isMultiMachine();

    // If we're in applicationMode == MOTIONCAPTURE or renderMode == realtime then all
    // rendering should have stopped by this point, so use all threads for the snapshot.
    bool parallel = mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME;
    if (mOptions.getApplicationMode() == rndr::ApplicationMode::MOTIONCAPTURE) {
        parallel = true;
    }

    {
        bool extrapolationStat = !mRenderContext->areCoarsePassesComplete();
        if (extrapolationStat) {
            TEST_SHOW_TIMING_INFO(ms.mLap.auxUInt64SectionAdd(ms.mIdAuxL_extrapTotal,1));
        } else {
            TEST_SHOW_TIMING_INFO(ms.mLap.auxUInt64SectionAdd(ms.mIdAuxL_extrapTotal,0));
        }
    }

    mRenderContext->snapshotRenderBuffer(&mRenderBuffer, untileDuringSnapshot, parallel);

    // we only want to perform snapshots once when requested; always setting this to false
    // doesn't cost anything significant and avoids unnecessary additional logic
    mReceivedSnapshotRequest = false;

    if (mOptions.getApplicationMode() != rndr::ApplicationMode::MOTIONCAPTURE) {
        // Make sure that the frame is ready for display before we start sending the depth buffer
        if (!mReceivedFinalPixelInfoBuffer && mRenderContext->hasPixelInfoBuffer()) {
            // If we finished the coarse passes then we have a sample for each pixel and
            // we can send one more depth buffer and then stop sending anymore.
            if (mRenderContext->areCoarsePassesComplete()) {
                mReceivedFinalPixelInfoBuffer = true;
            }
            mRenderContext->snapshotPixelInfoBuffer(&mPixelInfoBuffer,
                                                    untileDuringSnapshot,
                                                    parallel);
        }
    }

    // Gamma correct and quantize to 8-bit before sending downstream.
    // Note that we are doing this here for the multi-machine case also so
    // the merge node shouldn't do it again.
    // TODO: Does it make more sense to have the merge node do this for the
    //       multi-machine path?
    if (mImageEncoding == BaseFrame::ENCODING_RGBA8 ||
        mImageEncoding == BaseFrame::ENCODING_RGB888) {
        fb_util::PixelBufferUtilOptions options = fb_util::PIXEL_BUFFER_UTIL_OPTIONS_NONE;
        if (mApplyGamma) options |= fb_util::PIXEL_BUFFER_UTIL_OPTIONS_APPLY_GAMMA;
        if (parallel)    options |= fb_util::PIXEL_BUFFER_UTIL_OPTIONS_PARALLEL;
        mPixelBuffer.gammaAndQuantizeTo8bit(mRenderBuffer, options, 0.f, 1.f);
    }
}

void
McrtRtComputation::applyUpdatesAndRestartRender()
{
    if (!mRenderContext) {
        return;
    }

    TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startA));

    if (mOptions.getApplicationMode() != rndr::ApplicationMode::MOTIONCAPTURE) {
        // Stop rendering.  The frame may not have started yet, so only stop it if it has
        if (mRenderContext->isFrameRendering()) {
            mRenderContext->stopFrame();
        }
    }

    // APPLY ALL THE UPDATES HERE
    for (auto it = mUpdates.begin(); it != mUpdates.end(); ++it) {
        it->second(it->first);
    }

    // Make sure our updates didn't trample our overrides
    if (!mUpdates.empty()) {
        applyConfigOverrides();
        mUpdates.clear();
    }

    TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startA));
    TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startB[0]));

    // We did at least receive one rdl message to begin rendering
    if (mRenderContext->isInitialized()) {
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startB[1]));

        // Start rendering.
        MOONRAY_LOG_INFO("Starting Rendering");
        {
            if (mRenderContext->isFrameRendering()) {
                mRenderContext->stopFrame();
            }
        }

        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startB[1])); // save 1
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startB[2]));

        // Apply geometry updates.
        if (mGeometryUpdate) {
            mRenderContext->updateGeometry(mGeometryUpdate->mObjectData);
            mGeometryUpdate = nullptr;
        }

        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startB[2])); // save 2
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startB[3]));

        if (mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME) {
            // update display time info to realtimeFrameStats
            auto driver = rndr::getRenderDriver();
            driver->commitCurrentRealtimeStats();
        }

        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startB[3])); // save 3
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startB[4]));
        mRenderContext->startFrame();
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startB[4])); // save 4
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startB[5]));
        {
#ifdef SHOW_TIMING_INFO
            for (int i = 0; i < mRenderContext->getRenderPrepTimingStatus().getStopFrameTotal(); ++i) {
                ms.mLap.auxSectionAdd(ms.mIdAux_stopFrame[i], mRenderContext->getRenderPrepTimingStatus().getStopFrameVal(i));
            }

            ms.mLap.auxSectionAdd(ms.mIdAux_sfWhole, mRenderContext->getRenderPrepTimingStatus().getWholeStartFrame());
            for (int i = 0; i < mRenderContext->getRenderPrepTimingStatus().getRenderPrepTotal(); ++i) {
                ms.mLap.auxSectionAdd(ms.mIdAux_renderPrep[i], mRenderContext->getRenderPrepTimingStatus().getRenderPrepVal(i));
            }

            // float loadScnAve = mRenderContext->getSceneRenderStats().mLoadSceneTime.getAverage();
            // unsigned long loadScnTotal = mRenderContext->getSceneRenderStats().mLoadSceneTime.getCount();
            float buildPrimAttrTblAve = mRenderContext->getSceneRenderStats().mBuildPrimAttrTableTime.getAverage();
            // unsigned long buildPrimAttrTblTotal = mRenderContext->getSceneRenderStats().mBuildPrimAttrTableTime.getCount();
            float loadProcAve = mRenderContext->getSceneRenderStats().mLoadProceduralsTime.getAverage();
            // unsigned long loadProcTotal = mRenderContext->getSceneRenderStats().mLoadProceduralsTime.getCount();

            float tessAve = mRenderContext->getSceneRenderStats().mTessellationTime.getAverage();
            // unsigned long tessTotal = mRenderContext->getSceneRenderStats().mTessellationTime.getCount();
            float buildAve = mRenderContext->getSceneRenderStats().mBuildAcceleratorTime.getAverage();
            // unsigned long buildTotal = mRenderContext->getSceneRenderStats().mBuildAcceleratorTime.getCount();

            float buildProc = mRenderContext->getSceneRenderStats().mBuildProceduralTime;
            float rtcCommit = mRenderContext->getSceneRenderStats().mRtcCommitTime;

            float rebuildGeoAve = mRenderContext->getSceneRenderStats().mRebuildGeometryTime.getAverage();
            // unsigned long rebuildGeoTotal = mRenderContext->getSceneRenderStats().mRebuildGeometryTime.getCount();
            // float updateScnAve = mRenderContext->getSceneRenderStats().mUpdateSceneTime.getAverage();
            // unsigned long updateScnTotal = mRenderContext->getSceneRenderStats().mUpdateSceneTime.getCount();
#endif // end SHOW_TIMING_INFO

            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_primAttrTbl, buildPrimAttrTblAve));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_loadProc, loadProcAve));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_tessellation, tessAve));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_buildBVH, buildAve));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_buildProc, buildProc));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_rtcCommit, rtcCommit));
            TEST_SHOW_TIMING_INFO(ms.mLap.auxSectionAdd(ms.mIdAux_rebuildGeo, rebuildGeoAve));
        }
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startEnd));
        setFrameStateVariables();

        mRenderFrameId = mGeoFrameId; // update fameId for PartialFrame message
        mCompleteRendering = false;
        TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startB[5])); // save 5
    }

    TEST_SHOW_TIMING_INFO(ms.mLap.sectionEnd(ms.mId_startB[0])); // whole
}

void
McrtRtComputation::processMultimachine(BaseFrame::Status& status)
{
    // Don't bother sending message if this machine doesn't have any tiles to
    // work on.
    if (mRenderContext->getTiles()->empty()) {
        return;
    }
    assert(mImageEncoding != BaseFrame::ENCODING_LINEAR_FLOAT);

    // In a multimachine setup, we output a partial frame for reassembly
    // downstream by the merge computation.
    // TODO: Handle sending a partial linear float frame in a multi-machine setup
    unsigned numTiles = (unsigned)(mRenderContext->getTiles()->size());
    if (!numTiles) {
        return;
    }

    PartialFrame::Ptr frameMsg(new PartialFrame);

    rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
    frameMsg->mMachineId = sceneVars.getMachineId();

    MOONRAY_LOG_DEBUG("Setting machine_id: %d", frameMsg->mMachineId);

    if (mRenderContext->isFrameComplete() && mRenderContext->isFrameRendering()) {
        status = BaseFrame::FINISHED;
    }

    {
        math::HalfOpenViewport halfOpenViewport = mRenderContext->getRezedRegionWindow();
        math::Viewport rezedViewport = math::convertToClosedViewport(halfOpenViewport);
        frameMsg->mHeader.setRezedViewport(rezedViewport.mMinX, rezedViewport.mMinY, rezedViewport.mMaxX, rezedViewport.mMaxY);
    }
    frameMsg->mHeader.mStatus = status;
    frameMsg->mHeader.mProgress = getRenderProgress();

    // Get the ROI
    math::HalfOpenViewport roiHalfOpenViewport;
    const bool usingRoi = sceneVars.getSubViewport(roiHalfOpenViewport);
    if (usingRoi) {
        math::Viewport roiViewport = math::convertToClosedViewport(roiHalfOpenViewport);
        frameMsg->mHeader.setViewport(roiViewport.mMinX, roiViewport.mMinY, roiViewport.mMaxX, roiViewport.mMaxY);
    } else {
        frameMsg->mHeader.mViewport.reset();
    }

    unsigned dataLength = numTiles * (COARSE_TILE_SIZE * COARSE_TILE_SIZE * mPixelBuffer.getSizeOfPixel());
    uint8_t* data = new uint8_t[dataLength];
    mPixelBuffer.packSparseTiles(data, *mRenderContext->getTiles());

    frameMsg->addBuffer(network::makeValPtr(data), dataLength, AOV_BEAUTY.c_str(), mImageEncoding);
    frameMsg->mHeader.mFrameId = mRenderFrameId;

    if (mOptions.getApplicationMode() != rndr::ApplicationMode::MOTIONCAPTURE) {
        // Add the depth buffer if it's ready
        if (mRenderContext->hasPixelInfoBuffer() && !mSentFinalPixelInfoBuffer) {
            unsigned dataLength = numTiles * (COARSE_TILE_SIZE * COARSE_TILE_SIZE * sizeof(fb_util::PixelInfoBuffer::PixelType));
            data = new uint8_t[dataLength];
            fb_util::packSparseTiles((fb_util::PixelInfoBuffer::PixelType *)data, mPixelInfoBuffer, *mRenderContext->getTiles());

            if (mReceivedFinalPixelInfoBuffer) {
                mSentFinalPixelInfoBuffer = true;
            }

            frameMsg->addBuffer(network::makeValPtr(data), dataLength, AOV_DEPTH.c_str(), BaseFrame::ENCODING_FLOAT);
        }
    }

    send(frameMsg);

    if (mFirstFrame) {
        mFirstFrame = false;
    }
}

void
McrtRtComputation::processSingleMachine(BaseFrame::Status& status)
{
    MNRY_ASSERT_REQUIRE(mRenderBuffer.getData());

    // In a single machine setup, we output a complete frame.
    RenderedFrame::Ptr frameMsg(new RenderedFrame);
    {
        math::HalfOpenViewport halfOpenViewport = mRenderContext->getRezedRegionWindow();
        math::Viewport rezedViewport = math::convertToClosedViewport(halfOpenViewport);
        frameMsg->mHeader.setRezedViewport(rezedViewport.mMinX, rezedViewport.mMinY, rezedViewport.mMaxX, rezedViewport.mMaxY);
    }

    // Image buffer as unsigned char
    moonray::network::DataPtr buffer;

    // Size of image buffer in bytes
    size_t bufferLength;

    // Get the ROI
    rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();

    math::HalfOpenViewport roiHalfOpenViewport;
    const bool usingRoi = sceneVars.getSubViewport(roiHalfOpenViewport);
    math::Viewport roiViewport = math::convertToClosedViewport(roiHalfOpenViewport);

    if (usingRoi) {
        // Set the ROI viewport
        MOONRAY_LOG_DEBUG("Using ROI Viewport: (%d, %d, %d, %d) (%d x %d)", roiViewport.mMinX, roiViewport.mMinY, roiViewport.mMaxX, roiViewport.mMaxY, roiViewport.width(), roiViewport.height());
        frameMsg->mHeader.setViewport(roiViewport.mMinX, roiViewport.mMinY, roiViewport.mMaxX, roiViewport.mMaxY);
        switch (mImageEncoding) {
        case BaseFrame::ENCODING_RGBA8:
        case BaseFrame::ENCODING_RGB888:
            fb_util::copyRoiBuffer<uint8_t>
                       (roiViewport, mViewport, mPixelBuffer.getSizeOfPixel(),
                        mRoiPixelBuffer.getData(),
                        mPixelBuffer.getData(), bufferLength);
            buffer = mRoiPixelBuffer.getDataShared();
            break;
        case BaseFrame::ENCODING_LINEAR_FLOAT:
            fb_util::copyRoiBuffer<float>
                        (roiViewport, mViewport, sizeof(float),
                         reinterpret_cast<float *>(mRoiRenderBuffer.getData()),
                         reinterpret_cast<float *>(mRenderBuffer.getData()), bufferLength);
            buffer = mRoiRenderBuffer.getDataSharedAs<uint8_t>();
            break;
        default:
            assert(0 && "Invalid Image Encoding");
        }
    } else {
        // Otherwise, make sure we're not using a region of interest
        frameMsg->mHeader.mViewport.reset();
        switch (mImageEncoding) {
        case BaseFrame::ENCODING_RGBA8:
        case BaseFrame::ENCODING_RGB888:
            buffer = mPixelBuffer.getDataShared();
            bufferLength = mPixelBuffer.getArea() * mPixelBuffer.getSizeOfPixel();
            break;
        case BaseFrame::ENCODING_LINEAR_FLOAT:
            buffer = mRenderBuffer.getDataSharedAs<uint8_t>();
            bufferLength = mRenderBuffer.getArea() * sizeof(decltype(mRenderBuffer)::PixelType); // 4 floats per pixel (4 bytes per float)
            break;
        default:
            assert(0 && "Invalid Image Encoding");
        }
    }

    // Finally add the buffer to the message
    frameMsg->addBuffer(buffer, bufferLength, AOV_BEAUTY.c_str(), mImageEncoding);

    if (mRenderContext->isFrameComplete()) {
        status = BaseFrame::FINISHED;
    }

    frameMsg->mHeader.mStatus = status;
    frameMsg->mHeader.mProgress = getRenderProgress();

    if (mOptions.getApplicationMode() != rndr::ApplicationMode::MOTIONCAPTURE) {
        // Only do this part if we want to send a depth buffer, which
        // we do only if not in batch mode.
        if (mRenderContext->hasPixelInfoBuffer() && !mSentFinalPixelInfoBuffer) {
            if (usingRoi) {
                fb_util::copyRoiBuffer<float>
                    (roiViewport, mViewport, 1,
                     reinterpret_cast<float *>(mRoiPixelInfoBuffer.getData()),
                     reinterpret_cast<float *>(mPixelInfoBuffer.getData()), bufferLength);
                buffer = mRoiPixelInfoBuffer.getDataSharedAs<uint8_t>();
            } else {
                buffer = mPixelInfoBuffer.getDataSharedAs<uint8_t>();
                bufferLength = mPixelInfoBuffer.getArea() * sizeof(decltype(mPixelInfoBuffer)::PixelType); // 1 float per pixel
            }

            MOONRAY_LOG_DEBUG("Adding depth buffer");
            frameMsg->addBuffer(buffer, bufferLength, AOV_DEPTH.c_str(), BaseFrame::ENCODING_FLOAT);

            if (mReceivedFinalPixelInfoBuffer) {
                mSentFinalPixelInfoBuffer = true;
            }
        }
    }

    MOONRAY_LOG_DEBUG("Sending frame #%d", ++mFrameCount);

    frameMsg->mEye = mEye;
    send(frameMsg);

    if (mFirstFrame == true) {
        GenericMessage::Ptr firstFrameMsg(new GenericMessage);
        firstFrameMsg->mValue = "MCRT Rendered First Frame";
        send(firstFrameMsg);
        mFirstFrame = false;
        MOONRAY_LOG_INFO("Mcrt Sent first frame message");
    }

    if (mRenderContext->isFrameComplete() && mRenderContext->isFrameRendering()) {
        mRenderContext->stopFrame();
    }
}

float
McrtRtComputation::getRenderProgress()
{

    if (!mRenderContext) {
        return 0.0f; // Nothing to do
    }

    // Defaults to finished
    float progress = 1.0f;

    // If we're rendering, then we can fetch the current frame progress
    if (mRenderContext->isFrameRendering()) {
        progress = mRenderContext->getFrameProgressFraction(nullptr, nullptr);
    }

    return progress;
}

void
McrtRtComputation::onMessage(const Message::Ptr aMsg)
{
    TEST_SHOW_BUSYMESSAGE_INFO(sb.onMessageUpdate());
    
    if (aMsg->id() == GeometryData::ID) {
        TEST_SHOW_TIMING_INFO(ms.geoRecvInterval.showInterval("Mcrt : geo msg recv", 10.0f, [](const std::string &msg) {
                    std::cerr << msg << std::endl;
                    // MOONRAY_LOG_INFO(msg);
                }));

        if (mRenderContext->getRenderMode() == rndr::RenderMode::REALTIME) {
            mRTController.markReceivedGeometryData();
            if (mDispatchGatesFrame) {
                setFps(mRTController.getFps());
            }
        }

        if (!mGeoUpdateMode) return;
        
        if (mGeometryUpdate) {
            std::cout << ">>> McrtRtComputation.cc skip GeoFrame:" << mGeometryUpdate->mFrame << std::endl;
            mTotalSkipGeometryData++;
        }

        mGeometryUpdate = std::static_pointer_cast<GeometryData>(aMsg);
        mGeoFrameId = mGeometryUpdate->mFrame;
        mTotalReceivedGeometryData++;

        if (mTotalReceivedGeometryData == 100) {
            std::cerr << "\n\n>>> McrtRtComputation.cc warm-up run completed : reset()" << std::endl;
            TEST_SHOW_TIMING_INFO(ms.mLap.reset());
            mTotalSkipGeometryData = 0;
        }

    } else if (aMsg->id() == RDLMessage::ID) {
        RDLMessage::Ptr rdlMsg = std::static_pointer_cast<RDLMessage>(aMsg);
        onRDLMessage(rdlMsg);
    } else if (aMsg->id() == RDLMessage_LeftEye::ID) {
        RDLMessage::Ptr rdlMsg = std::static_pointer_cast<RDLMessage>(aMsg);
        onRDLMessage(rdlMsg);
    } else if (aMsg->id() == RDLMessage_RightEye::ID) {
        RDLMessage::Ptr rdlMsg = std::static_pointer_cast<RDLMessage>(aMsg);
        onRDLMessage(rdlMsg);
    } else if (aMsg->id() == GenericMessage::ID) {
        GenericMessage::Ptr gm = std::static_pointer_cast<GenericMessage>(aMsg);
        onGenericMessage(gm);
    } else if (aMsg->id() == JSONMessage::ID) {
        JSONMessage::Ptr jm = std::static_pointer_cast<JSONMessage>(aMsg);
        onJSONMessage(jm);
    } else if (aMsg->id() == ViewportMessage::ID) {
       ViewportMessage& vpMsg = static_cast<ViewportMessage&>(*aMsg);
       onViewportChanged(vpMsg.width(), vpMsg.height());
    }
}

void
McrtRtComputation::processRenderSetupMessage(const moonray::network::Message::Ptr& msg)
{
    moonray::network::JSONMessage::Ptr jm = std::static_pointer_cast<moonray::network::JSONMessage>(msg);
    const std::string sandbox = jm->messagePayload()[RenderMessages::RENDER_SETUP_PAYLOAD_SANDBOX].asString();

    // Initialize the render context
    mRenderContext.reset(new rndr::RenderContext(mOptions));
}

void
McrtRtComputation::onRenderSetupMessage(const moonray::network::Message::Ptr& msg)
{
    // HACK! Verify that this message came with the right KEY.  This ensures that messages from the Client
    // aren't delivered before messages from upstream computations.  This is a hack until we have proper
    // intent-message routing rules:  @see http://jira.anim.dreamworks.com/browse/NOVADEV-985

    moonray::network::JSONMessage::Ptr jm = std::static_pointer_cast<moonray::network::JSONMessage>(msg);
    if (jm->mClientData != RENDER_SETUP_KEY) {
        return; // ignore message as it originated from the client
    }

    // Clean up the queue since we do not want to process further updates
    mUpdates.clear();

    // Push it to the queue
    mUpdates.push_back(std::make_pair(msg, std::bind(&McrtRtComputation::processRenderSetupMessage, this, std::placeholders::_1)));
}

void
McrtRtComputation::processRdlMessage(const moonray::network::Message::Ptr& msg)
{
    moonray::network::RDLMessage::Ptr rdlMsg = std::static_pointer_cast<moonray::network::RDLMessage>(msg);
    // If we haven't begun rendering yet (mRenderContext is NULL),
    // or we need to restart rendering from scratch, do so here
    if (rdlMsg->mForceReload || !mRenderContext) {
        mRenderContext.reset(new rndr::RenderContext(mOptions));
    }

    MNRY_ASSERT(mRenderContext && "Cannot apply a scene update without a RenderContext!");
    mRenderContext->updateScene(rdlMsg->mManifest, rdlMsg->mPayload);

    if (!mRenderContext->isInitialized()) {
        std::stringstream initMessages;
        mRenderContext->initialize(initMessages);
        applyConfigOverrides();
        initializeBuffers();
    }
}

void
McrtRtComputation::onRDLMessage(const RDLMessage::Ptr msg)
{
    // If we're forcing a reload, clear the message queue and only use this message
    if (msg->mForceReload) {
        mUpdates.clear();
    }
    // Finally, add the message.  If it's a normal delta update message, queue it normally
    mUpdates.push_back(std::make_pair(msg, std::bind(&McrtRtComputation::processRdlMessage, this, std::placeholders::_1)));
}

void
McrtRtComputation::onGenericMessage(const GenericMessage::Ptr msg)
{
    // Parse all 'non-json' messages first.  TODO, make everything a json payload
    if (msg->mValue == "snapshot") {
        mReceivedSnapshotRequest = true;
        return;
    }
    return;
}

void
McrtRtComputation::processROIMessage(const moonray::network::Message::Ptr& msg)
{
    moonray::network::JSONMessage::Ptr jm = std::static_pointer_cast<moonray::network::JSONMessage>(msg);
    const std::string messageID = jm->messageId();

    rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();

    // Set a new region of interest
    if (messageID == RenderMessages::SET_ROI_OPERATION_ID) {
        auto& payload = jm->messagePayload();
        math::Viewport viewport = math::Viewport(payload[Json::Value::ArrayIndex(0)].asInt(), payload[Json::Value::ArrayIndex(1)].asInt(),
                                                         payload[Json::Value::ArrayIndex(2)].asInt(), payload[Json::Value::ArrayIndex(3)].asInt());
        MOONRAY_LOG_INFO("Setting ROI viewport to (%d, %d, %d, %d) (%d x %d)", viewport.mMinX, viewport.mMinY, viewport.mMaxX, viewport.mMaxY, viewport.width(), viewport.height());
        rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
        rdl2::SceneVariables::UpdateGuard guard(&sceneVars);
        std::vector<int> viewportVector = {viewport.mMinX, viewport.mMinY, viewport.mMaxX, viewport.mMaxY};
        sceneVars.set(rdl2::SceneVariables::sSubViewport, viewportVector);
        mLastRoiViewport = viewport;

        // Only reinitialize these buffers if we're in single-machine mode.  ROI buffers
        // are handled downstream in the Merge computation for Multimachine configuratinos
        if (!isMultiMachine()) {
            unsigned w = (unsigned)viewport.width();
            unsigned h = (unsigned)viewport.height();

            mRoiRenderBuffer.init(w, h);
            mRoiPixelBuffer.init(convertImageEncoding(mImageEncoding), w, h);

            if (mRenderContext->hasPixelInfoBuffer()) {
                mRoiPixelInfoBuffer.init(w, h);
            }
        }

    // Turn off the region of interest
    } else if (messageID == RenderMessages::SET_ROI_STATUS_OPERATION_ID) {
        sceneVars.disableSubViewport();
    }
}

void
McrtRtComputation::handlePick(const RenderMessages::PickMode mode,
                              const int x, const int y,
                              JSONMessage::Ptr& result)
{
    switch (mode) {
        case RenderMessages::PickMode::QUERY_LIGHT_CONTRIBUTIONS:
        {
            moonray::shading::LightContribArray rdlLights;
            mRenderContext->handlePickLightContributions(x, y, rdlLights);

            Json::Value lights;
            Json::Value contributions;
            // Loop through the lights and populate the json
            // values
            for (uint i = 0; i < rdlLights.size(); ++i) {
                lights.append(rdlLights[i].first->getName());
                contributions.append(rdlLights[i].second);
            }

            result->messagePayload()[RenderMessages::PICK_DATA_MESSAGE_PAYLOAD_LIGHTS] = lights;
            result->messagePayload()[RenderMessages::PICK_DATA_MESSAGE_PAYLOAD_LIGHT_CONTRIBUTIONS] = contributions;
        }
        break;
        case RenderMessages::PickMode::QUERY_GEOMETRY:
        {
            const rdl2::Geometry* geometry = mRenderContext->handlePickGeometry(x, y);

            Json::Value geom = geometry ? geometry->getName() : "";
            result->messagePayload()[RenderMessages::PICK_DATA_MESSAGE_PAYLOAD_GEOMETRY] = geom;
        }
        break;
        case RenderMessages::PickMode::QUERY_GEOMETRY_PART:
        {
            std::string parts;
            const rdl2::Geometry* geometry = mRenderContext->handlePickGeometryPart(x, y, parts);

            Json::Value part = parts;
            Json::Value geom = geometry ? geometry->getName() : "";
            result->messagePayload()[RenderMessages::PICK_DATA_MESSAGE_PAYLOAD_GEOMETRY_PARTS] = part;
            result->messagePayload()[RenderMessages::PICK_DATA_MESSAGE_PAYLOAD_GEOMETRY] = geom;
        }
        break;
        case RenderMessages::PickMode::QUERY_POSITION_AND_NORMAL:
        break;
        case RenderMessages::PickMode::QUERY_CELL_INSPECTOR:
        break;
        case RenderMessages::PickMode::QUERY_MATERIAL:
        {
            const rdl2::Material* materials = mRenderContext->handlePickMaterial(x, y);

            Json::Value material = materials ? materials->getName() : "";
            result->messagePayload()[RenderMessages::PICK_DATA_MESSAGE_PAYLOAD_MATERIALS] = material;
        }
        break;
        default:
        break;
    };
}

void
McrtRtComputation::onJSONMessage(const JSONMessage::Ptr jm)
{
    const std::string messageID = jm->messageId();
    if (messageID == RenderMessages::RENDER_CONTROL_ID) {
        const std::string operation = jm->messagePayload()[RenderMessages::getRenderControlPayloadOperation()].asString();
        if (operation == RenderMessages::getRenderControlPayloadOperationStart()) {
            MOONRAY_LOG_INFO("Msg-> Start Rendering!");
            mRenderControlMessages.push_back(START);
        } else if (operation == RenderMessages::getRenderControlPayloadOperationStop()) {
            MOONRAY_LOG_INFO("Msg-> Stop Rendering")
            mRenderControlMessages.push_back(STOP);
        }

#if 0
        // Handle render controls
    } else if (messageID == SELECT_AOV_ID) {
        // Handle AOV selection
#endif

    } else if (messageID == RenderMessages::PICK_MESSAGE_ID) {
        // Handle Picking
        auto& payload = jm->messagePayload();
        int x = payload[RenderMessages::PICK_MESSAGE_PAYLOAD_PIXEL][Json::Value::ArrayIndex(0)].asInt();
        int y = payload[RenderMessages::PICK_MESSAGE_PAYLOAD_PIXEL][Json::Value::ArrayIndex(1)].asInt();
        RenderMessages::PickMode mode = static_cast<RenderMessages::PickMode>(payload[RenderMessages::PICK_MESSAGE_PAYLOAD_MODE].asInt());
        MOONRAY_LOG_INFO("ClientID: %s  (x: %d, y: %d) mode: %d", jm->mClientData.c_str(), x, y, static_cast<int>(mode));

        auto result = RenderMessages::createPickDataMessage(x, y, jm->mClientData);
        handlePick(mode, x, y, result);
        send(result);

    } else if (messageID == RenderMessages::LOGGING_CONFIGURATION_MESSAGE_ID) {
        auto& payload = jm->messagePayload();
        log::Logger::Level level = static_cast<log::Logger::Level>(payload[Json::Value::ArrayIndex(0)].asInt());
        moonray::log::Logger::instance().setThreshold(level);

    } else if (messageID == RenderMessages::SET_ROI_OPERATION_ID) {
        MOONRAY_LOG_INFO("Msg-> Roi changed");
        auto& payload = jm->messagePayload();

        // Check if we need to apply an update
        rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
        math::HalfOpenViewport curViewport;
        math::Viewport viewport = math::Viewport(payload[Json::Value::ArrayIndex(0)].asInt(), payload[Json::Value::ArrayIndex(1)].asInt(),
                                                 payload[Json::Value::ArrayIndex(2)].asInt(), payload[Json::Value::ArrayIndex(3)].asInt());
        MOONRAY_LOG_INFO("sceneVars.getSubViewport(curViewport): %d", sceneVars.getSubViewport(curViewport));
        MOONRAY_LOG_INFO("mLastRoiViewport: (%d, %d, %d, %d) (%d x %d)", mLastRoiViewport.mMinX, mLastRoiViewport.mMinY, mLastRoiViewport.mMaxX, mLastRoiViewport.mMaxY, mLastRoiViewport.width(), mLastRoiViewport.height());
        MOONRAY_LOG_INFO("viewport        : (%d, %d, %d, %d) (%d x %d)", viewport.mMinX, viewport.mMinY, viewport.mMaxX, viewport.mMaxY, viewport.width(), viewport.height());

        // Either we haven't set an ROI, or our ROI has changed
        if (!sceneVars.getSubViewport(curViewport) || mLastRoiViewport != viewport) {
            // Push it to the queue
            mUpdates.push_back(std::make_pair(jm, std::bind(&McrtRtComputation::processROIMessage, this, std::placeholders::_1)));
        }

    } else if (messageID == RenderMessages::SET_ROI_STATUS_OPERATION_ID) {
        MOONRAY_LOG_INFO("Msg-> Roi status changed");
        auto& payload = jm->messagePayload();

        // Check if we're disabling the ROI
        if (!payload[Json::Value::ArrayIndex(0)].asBool()) {
            // Push it to the queue
            mUpdates.push_back(std::make_pair(jm, std::bind(&McrtRtComputation::processROIMessage, this, std::placeholders::_1)));
        }
    } else if (messageID == RenderMessages::getInvalidateResourcesId()) {
        MOONRAY_LOG_INFO("Invalidating resources");
        std::vector<std::string> resourcesToInvalidate =
            ValueVector(jm->messagePayload()[RenderMessages::INVALIDATE_RESOURCES_PAYLOAD_LIST]);
        mRenderContext->invalidateTextureResources(resourcesToInvalidate);
    } else if (messageID == RenderMessages::RENDER_CAMERA_TRANSFORM_ID) {
        MOONRAY_LOG_INFO("Msg-> Transform camera");

        // Must name camera "/rdla/camera" in rdla until this is added to a config.
        moonray::rdl2::Camera* camera = 
            mRenderContext->getSceneContext().getSceneObject("/rdla/camera")->asA<moonray::rdl2::Camera>();

        const auto& value = jm->messagePayload()[RenderMessages::getRenderCameraTransformPayload()];

        // Convert the json to a matrix
        rdl2::Mat4f matrix;
        size_t i = 0;

        // cppcheck-suppress StlMissingComparison
        for (auto it = value.begin(); it != value.end(); ++it) {
            for (size_t j = 0; j < 4; ++j) {
                matrix[i][j] = (*it).asDouble();
                if (j != 3) ++it;
            }
            ++i;
        }
        camera->beginUpdate();
        camera->set("node xform", matrix);
        camera->endUpdate();

        //We want to trigger an update when we get a camera update
        RDLMessage* rdlMsg = new RDLMessage;
        moonray::rdl2::BinaryWriter w((mRenderContext->getSceneContext()));
        w.setDeltaEncoding(true);
        w.toBytes(rdlMsg->mManifest, rdlMsg->mPayload);
        mRenderContext->getSceneContext().commitAllChanges();
        mUpdates.push_back(std::make_pair(Message::Ptr(rdlMsg), std::bind(&McrtRtComputation::processRdlMessage, this, std::placeholders::_1)));

    } else if (messageID == RenderMessages::RENDER_SETUP_ID) {
        /* test
        MOONRAY_LOG_INFO("Render Setup");
        onRenderSetupMessage(jm);
        */
    }
    
}

void
McrtRtComputation::initializeBuffers()
{
    rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
    mViewport = math::convertToClosedViewport(sceneVars.getRezedRegionWindow());

    unsigned w = mViewport.width();
    unsigned h = mViewport.height();

    // Since we do the frame reconstruction and untiling on the merge node
    // in the distributed rendering case, keep these buffers tile aligned.
    if (isMultiMachine()) {
        w = util::alignUp(w, COARSE_TILE_SIZE);
        h = util::alignUp(h, COARSE_TILE_SIZE);
    }

    mRenderBuffer.init(w, h);
    mPixelBuffer.init(convertImageEncoding(mImageEncoding), w, h);

    if (mRenderContext->hasPixelInfoBuffer()) {
        mPixelInfoBuffer.init(w, h);
    }
}

void
McrtRtComputation::onViewportChanged(int width, int height)
{
    if (mRenderContext && mRenderContext->isFrameRendering()) {
        mRenderContext->stopFrame();
    }

    // Slight-hack to deal with viewport message creating the RenderContext and forcing a reload
    if (!mOptions.getSceneFiles().empty()) {
        mRenderContext.reset(new rndr::RenderContext(mOptions));
        std::stringstream initMessages;
        mRenderContext->initialize(initMessages);
    }
 
    // Apply new width and height to the potentially new RenderContext
    {
        rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
        rdl2::SceneVariables::UpdateGuard guard(&sceneVars);
        
        sceneVars.set(rdl2::SceneVariables::sImageWidth, width);
        sceneVars.set(rdl2::SceneVariables::sImageHeight, height);
    }
    
    applyConfigOverrides();
    initializeBuffers();
    
    mLastTime = util::getSeconds();
    mRenderContext->startFrame();
    TEST_SHOW_TIMING_INFO(ms.mLap.sectionStart(ms.mId_startEnd));
    setFrameStateVariables();
}


bool
McrtRtComputation::isMultiMachine()
{
    if (mRenderContext) {
        rdl2::SceneVariables& sceneVars = mRenderContext->getSceneContext().getSceneVariables();
        int numMachines = sceneVars.get(rdl2::SceneVariables::sNumMachines);
        if (numMachines > -1) { // "num machines" is set
            return numMachines > 1;
        }
    }

    //fall back to override
    return mNumMachinesOverride > 1;
}

} // namespace mcrt_rt_computation
} // namespace moonray

