// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "McrtRtComputationRealtimeController.h"

//#define DEBUG_CONSOLE_MODE
#ifdef DEBUG_CONSOLE_MODE
#include "McrtRtComputationDebugConsole.h"
#endif  // end end DEBUG_CONSOLE_MODE

#include <moonray/rendering/rndr/rndr.h>

#include <moonray/common/log/logging.h>
#include <moonray/engine/messages/generic_message/GenericMessage.h>
#include <moonray/engine/messages/geometry_data/GeometryData.h>
#include <moonray/engine/messages/json_message/JSONMessage.h>
#include <moonray/engine/messages/rdl_message/RDLMessage.h>
#include <moonray/engine/messages/render_messages/RenderMessages.h>
#include <moonray/engine/messages/rendered_frame/RenderedFrame.h>
#include <engine/computation/Computation.h>
#include <scene_rdl2/common/math/Viewport.h>

//#define RTT_TEST_MODE

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moonray {
namespace mcrt_rt_computation {

class McrtRtComputation : public moonray::engine::Computation
{
public:
    McrtRtComputation();
    virtual ~McrtRtComputation();

    void setFps(const float fps);
    float getFps() const { return mFps; }
    void setGeoUpdateMode(const bool mode) { mGeoUpdateMode = mode; }
    bool getGeoUpdateMode() const { return mGeoUpdateMode; }
    void setResetCounter() { mResetCounter = true; }

protected:
    virtual void configure(const moonray::object::Object& aConfiguration);
    virtual void onIdle();
    virtual void onMessage(const moonray::network::Message::Ptr msg);
    virtual void onStart();
    virtual void onStop();

private:
    enum RenderControl
    {
        START = 0,
        STOP,
        PAUSE
    };

    void handlePick(const moonray::network::RenderMessages::PickMode mode,
                    const int x, const int y, moonray::network::JSONMessage::Ptr& result);
    void onViewportChanged(int width, int height);
    void onRDLMessage(const moonray::network::RDLMessage::Ptr msg);
    void onGenericMessage(const moonray::network::GenericMessage::Ptr msg);
    void onJSONMessage(const moonray::network::JSONMessage::Ptr msg);
    bool isMultiMachine();
    void initializeBuffers();
    // Apply scene updates
    void applyUpdatesAndRestartRender();
    // Return whether or not we been idle long enough
    bool fpsIntervalPassed();
    // Snapshot the render and pixel info buffers
    void snapshotBuffers();
    // Set the state variables at the beginning of a render
    void setFrameStateVariables();
    void processControlMessages();
    void processMultimachine(network::BaseFrame::Status& status);
    void processSingleMachine(network::BaseFrame::Status& status);
    void processROIMessage(const moonray::network::Message::Ptr& msg);
    float getRenderProgress();
    void applyConfigOverrides();
    virtual void append(log::Logger::Level level, const std::string& message);

    void processRdlMessage(const moonray::network::Message::Ptr&);
    void onRenderSetupMessage(const moonray::network::Message::Ptr& msg);
    void processRenderSetupMessage(const moonray::network::Message::Ptr&);

#ifdef DEBUG_CONSOLE_MODE
    McrtRtComputationDebugConsole mDebugConsole;
#endif // end DEBUG_CONSOLE_MODE    

#ifdef RTT_TEST_MODE
    void processGeoUpdateAck();
#endif // end RTT_TEST_MODE

    rndr::RenderOptions mOptions;
    moonray::network::RenderedFrame::ImageEncoding mImageEncoding;
    std::unique_ptr<rndr::RenderContext> mRenderContext;

    fb_util::RenderBuffer mRenderBuffer;
    fb_util::RenderBuffer mRoiRenderBuffer;

    fb_util::VariablePixelBuffer mPixelBuffer;
    fb_util::VariablePixelBuffer mRoiPixelBuffer;

    fb_util::PixelInfoBuffer mPixelInfoBuffer;
    fb_util::PixelInfoBuffer mRoiPixelInfoBuffer;

    moonray::math::Viewport mViewport;
    moonray::math::Viewport mLastRoiViewport;

    moonray::network::GeometryData::Ptr mGeometryUpdate;
    typedef std::pair<moonray::network::Message::Ptr, std::function<void(const moonray::network::Message::Ptr&)> > Update;
    std::vector<Update> mUpdates;
    std::vector<RenderControl> mRenderControlMessages;

    uint mFrameCount;
    bool mFpsSet;
    float mFps;
    float mLastFps;   // for realtime renderMode
    // Flag to indicate if images are gated upstream
    bool mDispatchGatesFrame;
    double mLastTime;
    bool mReceivedSnapshotRequest;

    // flag used to determine if we have got the final depth
    // buffer after all coarse passes are finished.
    bool mReceivedFinalPixelInfoBuffer;

    // Set to true when we've sent a final pixel info buffer for this frame.
    // At that point we no longer need to snapshot the pixel info buffer any 
    // further.
    bool mSentFinalPixelInfoBuffer;

    // flag use to determine if the first rendered frame has
    //  left the renderer to other downstream computations.
    //  this is used to signal that the time heavy on demand
    //  loading of assets has finished and the render has
    //  actually begun rendering
    bool mFirstFrame;
    bool mCompleteRendering;
    bool mApplyGamma;
    
    // Flag for sending the proper status with the first sent frame
    bool mFrameStarted;

    // Stores "num machines" and "machine id" SceneVariable overrides
    // to be applied once the RenderContext is set up.
    int mNumMachinesOverride;
    int mMachineIdOverride;

    // Timestamps for when we start a render and when we send a snapshot
    int mRenderTimestamp;
    int mLastSnapshotTimestamp;
    int mLastFilmActivity;

    bool mSendLogMessages = true;

    moonray::network::RenderedFrame::RenderedEye  mEye;

    int mTotalReceivedGeometryData;
    int mTotalSkipGeometryData;
    int mGeoFrameId;            // received geometry message's frameId
    int mRenderFrameId;         // rendering data's frameId

    bool mGeoUpdateMode;
    bool mResetCounter;

    McrtRtComputationRealtimeController mRTController;
};


} // namespace mcrt_rt_computation
} // namespace moonray

