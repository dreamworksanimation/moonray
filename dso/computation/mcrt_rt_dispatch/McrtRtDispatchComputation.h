// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define DEBUG_CONSOLE_MODE
#ifdef DEBUG_CONSOLE_MODE
#include "McrtRtDispatchComputationDebugConsole.h"
#endif  // end DEBUG_CONSOLE_MODE

#include <engine/computation/Computation.h>
#include <engine/messages/geometry_data/GeometryData.h>
#include <engine/messages/json_message/JSONMessage.h>
#include <engine/messages/rdl_message/RDLMessage.h>

//#define RTT_TEST_MODE // This is a Round Trip Time test mode for message communication

#ifdef RTT_TEST_MODE
#include <engine/messages/generic_message/GenericMessage.h>
#endif // end RTT_TEST_MODE

#include <string>

namespace moonray {
namespace mcrt_rt_dispatch_computation {

class McrtRtDispatchComputation : public moonray::engine::Computation
{
public:
    McrtRtDispatchComputation();
    virtual ~McrtRtDispatchComputation();

    void setFps(const float fps) { mFps = fps; }
    float getFps() const { return mFps; }
    void setMotionCaptureMode(const bool mode) { mMotionCaptureMode = mode; }
    bool getMotionCaptureMode() const { return mMotionCaptureMode; }
    void setGeoUpdateMode(const bool mode) { mGeoUpdateMode = mode; }
    bool getGeoUpdateMode() const { return mGeoUpdateMode; }

protected:
    virtual void configure(const moonray::object::Object& aConfiguration);
    virtual void onIdle();
    virtual void onMessage(const moonray::network::Message::Ptr msg);

#ifdef RTT_TEST_MODE
    void processGenericMessage();
#endif // end RTT_TEST_MODE

private:
    moonray::network::GeometryData::Ptr mGeometryUpdate;
    std::vector<moonray::network::RDLMessage::Ptr> mRdlUpdates;
    moonray::network::JSONMessage::Ptr mCameraUpdate;
    bool mReceivedGeometryUpdate;
    bool mReceivedRdlUpdate;
    bool mReceivedCameraUpdate;
    float mFps;
    double mLastTime;
    bool mContinuous;
    bool mMotionCaptureMode;
    bool mGeoUpdateMode;
    int mFrameId;

#ifdef DEBUG_CONSOLE_MODE
    McrtRtDispatchComputationDebugConsole mDebugConsole;
#endif  // end DEBUG_CONSOLE_MODE

#ifdef RTT_TEST_MODE
    moonray::network::GenericMessage::Ptr mGenericMessage;
    bool mReceivedGenericMessage;
#endif // end RTT_TEST_MODE
};

} // namespace mcrt_rt_dispatch_computation
} // namespace moonray

