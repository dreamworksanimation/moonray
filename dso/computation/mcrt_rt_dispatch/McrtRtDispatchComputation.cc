// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "McrtRtDispatchComputation.h"
#include "McrtRtDispatchComputationStatistics.h"

#include <scene_rdl2/common/log/logging.h>
#include <scene_rdl2/common/object/Object.h>
#include <engine/computation/Computation.h>
#include <engine/messages/generic_message/GenericMessage.h>
#include <engine/messages/geometry_data/GeometryData.h>
#include <engine/messages/rdl_message/RDLMessage.h>
#include <engine/messages/render_messages/RenderMessages.h>
#include <scene_rdl2/common/platform/Platform.h>

#include <logging_base/macros.h>

#include <cstdlib>
#include <iostream>             // test
#include <string>

using moonray::engine::Computation;
using moonray::network::GenericMessage;
using moonray::network::GeometryData;
using moonray::network::JSONMessage;
using moonray::network::Message;
using moonray::network::RDLMessage;
using moonray::network::RenderMessages;

CREATOR_FUNC(moonray::mcrt_rt_dispatch_computation::McrtRtDispatchComputation);

//------------------------------------------------------------------------------
//
// Some useful debugging and statistical analyzing directives
//

#define SHOW_DOWNSTREAM_MESSAGE_INTERVAL // show interval info about geometry update message send to downstream
#define SHOW_UPSTREAM_MESSAGE_INTERVAL   // show interval info about geometry update message recv from upstream
#define SHOW_ONIDLE_STAT                 // show detail interval status for onIdle()

#ifdef SHOW_DOWNSTREAM_MESSAGE_INTERVAL
namespace scene_rdl2 {
    static rec_time::RecTimeAutoInterval geoSendInterval; // outgoing geometry message
} // namespace scene_rdl2

#define TEST_SHOW_DOWNSTREAM_MESSAGE_INTERVAL(f) f
#else // else SHOW_DOWNSTREAM_MESSAGE_INTERVAL
#define TEST_SHOW_DOWNSTREAM_MESSAGE_INTERVAL(f)
#endif // end !SHOW_DOWNSTREAM_MESSAGE_INTERVAL

#ifdef SHOW_UPSTREAM_MESSAGE_INTERVAL
namespace scene_rdl2 {
    static rec_time::RecTimeAutoInterval geoRecvInterval; // incoming geometry message
} // namespace scene_rdl2
#define TEST_SHOW_UPSTREAM_MESSAGE_INTERVAL(f) f
#else  // else SHOW_UPSTREAM_MESSAGE_INTERVAL
#define TEST_SHOW_UPSTREAM_MESSAGE_INTERVAL(f)
#endif // end !SHOW_UPSTREAM_MESSAGE_INTERVAL

#ifdef SHOW_ONIDLE_STAT
namespace moonray {
namespace mcrt_rt_dispatch_computation {
    static McrtRtDispatchComputationStatistics ms;
} // namespace mcrt_rt_dispatch_computation
} // namespace moonray
#define TEST_SHOW_ONIDLE(f) f
#else  // else SHOW_ONIDLE_STAT 
#define TEST_SHOW_ONIDLE(f)
#endif // end !SHOW_ONIDLE_STAT

//------------------------------------------------------------------------------

namespace moonray {
namespace mcrt_rt_dispatch_computation {

McrtRtDispatchComputation::McrtRtDispatchComputation() :
    mReceivedGeometryUpdate(false),
    mReceivedRdlUpdate(false),
    mReceivedCameraUpdate(false),
    mFps(5.0f),
    mLastTime(0.0),
    mContinuous(false),
    mMotionCaptureMode(false),
    mGeoUpdateMode(true),
    mFrameId(0)
#ifdef RTT_TEST_MODE
    , mReceivedGenericMessage(false)
#endif // end RTT_TEST_MODE    
{
    mRdlUpdates.reserve(10);
}

McrtRtDispatchComputation::~McrtRtDispatchComputation()
{
}

void
McrtRtDispatchComputation::configure(const object::Object& aConfig)
{
    std::cout << ">>> McrtRtDispatchComputation ..." << std::endl;
    MOONRAY_LOG_INFO("McrtRtDispatchComputation ...\n");

    if (!aConfig["fps"].isNull()) {
        mFps = aConfig["fps"];
    }

    mContinuous = aConfig["continuous"].value().asBool();

    mMotionCaptureMode = false;
    const std::string sApplicationMode = "applicationMode";
    if (!aConfig[sApplicationMode].isNull()) {
        if (aConfig[sApplicationMode].value().asInt() == 1) {
            mMotionCaptureMode = true;
        } else {
            MOONRAY_LOG_ERROR("APPLICATION MODE SET TO UNDEFIND");
        }
    }

#ifdef DEBUG_CONSOLE_MODE
    mDebugConsole.open(20000, this);
#endif // end DEBUG_CONSOLE_MODE
}

void
McrtRtDispatchComputation::onIdle()
{
#ifdef DEBUG_CONSOLE_MODE
    mDebugConsole.eval();
#endif // end DEBUG_CONSOLE_MODE

    //------------------------------

    TEST_SHOW_ONIDLE(ms.mLap.passStartingLine());

#ifdef RTT_TEST_MODE
    if (mReceivedGenericMessage) {
        processGenericMessage();
        mReceivedGenericMessage = false;
    }
#endif // end RTT_TEST_MODE    

    // Is it time to kick out a frame yet?
    double now = util::getSeconds();
    if (mMotionCaptureMode && mFps == 0.0f) {
        if (!mReceivedGeometryUpdate && !mReceivedCameraUpdate) {
            return;
        }
    } else {
        if (now - mLastTime < (1.0f / mFps)) {
            return;
        }
    }

    TEST_SHOW_ONIDLE(ms.mLap.sectionStart(ms.mId_whole));

    if (mContinuous) {
        // otherwise, request a snapshot (if we haven't received any other updates)
        if (!(mReceivedGeometryUpdate || mReceivedRdlUpdate || mReceivedCameraUpdate)) {
            GenericMessage* msg = new GenericMessage;
            msg->mValue = "snapshot";
            send(Message::Ptr(msg));
        }
    }

    // We have to process camera update before send geometry update
    if (mReceivedCameraUpdate) {
        send(mCameraUpdate);
        mReceivedCameraUpdate = false;
        std::cout << ">>> mcrtRtDispatchComputation.cc send camera update" << std::endl;
    }

    // We have to process rdlUpdate before send geometry update
    if (mReceivedRdlUpdate) {
        TEST_SHOW_ONIDLE(ms.mLap.sectionStart(ms.mId_rdl));
        for (auto iter = mRdlUpdates.begin(); iter != mRdlUpdates.end(); ++iter) {
            send(*iter);
        }
        TEST_SHOW_ONIDLE(ms.mLap.sectionEnd(ms.mId_rdl));
        
        // MOONRAY_LOG_INFO("[MCRT-DISPATCH] RDL Update Dispatch sent");
        mRdlUpdates.clear();
        mReceivedRdlUpdate = false;
    }

    // Forward the most geometry update, if received.
    bool sendGeomST = false;
    if (mMotionCaptureMode) {
        // mocap mode, always send current geomtry 
        sendGeomST = (mGeometryUpdate != 0x0)? true: false;
    } else {
        sendGeomST = mReceivedGeometryUpdate;
    }
    if (sendGeomST) {
        mFrameId++;
        mGeometryUpdate->mFrame = mFrameId;

        TEST_SHOW_ONIDLE(ms.mLap.sectionStart(ms.mId_geo));
        send(mGeometryUpdate);
        TEST_SHOW_ONIDLE(ms.mLap.sectionEnd(ms.mId_geo));
        TEST_SHOW_DOWNSTREAM_MESSAGE_INTERVAL(geoSendInterval.showInterval("Dispatcher : geo msg send", ms.MSG_INTERVAL_SEC,
                                                                           [](const std::string &msg){
                                                                               MOONRAY_LOG_INFO(msg);
                                                                           }));

        // MOONRAY_LOG_INFO("[MCRT-DISPATCH] Geometry Update Dispatch sent");
        mReceivedGeometryUpdate = false;
    }

    mLastTime = now;

    TEST_SHOW_ONIDLE(ms.mLap.sectionEnd(ms.mId_whole));
    TEST_SHOW_ONIDLE(ms.mLap.showLapInfo(mFps, [](const std::string &msg) { std::cerr << msg << std::endl; }));
}

void
McrtRtDispatchComputation::onMessage(const Message::Ptr msg)
{
    if (msg->id() == GeometryData::ID) {
        TEST_SHOW_UPSTREAM_MESSAGE_INTERVAL(geoRecvInterval.showInterval("Dispatcher : geo msg recv", ms.MSG_INTERVAL_SEC,
                                                                         [](const std::string &msg){
                                                                             MOONRAY_LOG_INFO(msg);
                                                                         }));

        if (mGeoUpdateMode) {
            mGeometryUpdate = std::static_pointer_cast<GeometryData>(msg);
            mReceivedGeometryUpdate = true;
        }
    }
    else if (msg->id() == RDLMessage::ID) {
        mRdlUpdates.push_back(std::static_pointer_cast<RDLMessage>(msg));
        mReceivedRdlUpdate = true;
    }
    else if (msg->id() == JSONMessage::ID) {
        JSONMessage::Ptr jm = std::static_pointer_cast<JSONMessage>(msg);
        const std::string messageID = jm->messageId();        
        if (messageID == RenderMessages::getRenderCameraTransformId()) {
            mCameraUpdate = jm;
            mReceivedCameraUpdate = true;
        } // end renderCameraTransformId
    } // end JSONMessage
#ifdef RTT_TEST_MODE
    else if (msg->id() == GenericMessage::ID) {
        mGenericMessage = std::static_pointer_cast<GenericMessage>(msg);
        mReceivedGenericMessage = true;
    }
#endif // end RTT_TEST_MODE
}

#ifdef RTT_TEST_MODE
void
McrtRtDispatchComputation::processGenericMessage()
{
    const GenericMessage::Ptr gm = mGenericMessage;
    if (gm->mValue.find("geoMsgAck") != std::string::npos) {
        std::ostringstream ostr;
        ostr << "McrtRtDispatcher GenericMessage:>" << gm->mValue << "<" << std::endl;
        MOONRAY_LOG_INFO(ostr.str().c_str());
    }
}
#endif // end RTT_TEST_MODE

} // namespace mcrt_rt_dispatch_computation
} // namespace moonray

