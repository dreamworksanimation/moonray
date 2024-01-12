// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#ifndef RENDERPREPEXECTRACKER_H
#define RENDERPREPEXECTRACKER_H

#include <memory>               // unique_ptr
#include <functional>           // function

namespace scene_rdl2 {

namespace grid_util {
    class Parser;
    class RenderPrepStats;
}
}

namespace moonray {
namespace rndr {

class RenderPrepExecTracker
//
// This class is used to track down renderPrep execution status. This class is using impl stype
// implementation because this class is used from RenderContext.h and RenderContext.h is a public header.
// RenderPrepStatsCallBack is a call back function to send stats to the downstream computation.
// RenderPrepStatsCallBack is properly called from each start/end function.
// We can set nullptr to RenderPrepStatsCallBack and disable report stats logic when we are not under
// the arras context. 
//
{
public:
    using RenderPrepStatsCallBack = std::function<void(const scene_rdl2::grid_util::RenderPrepStats &rPrepStats)>;
    using RenderPrepCancelCallBack = std::function<bool()>;
    using MsgHandlerCallBack = std::function<void(const std::string &msg)>;

    enum class RESULT // function execution result
    {
        CANCELED, // canceled middle of the function
        FINISHED  // function has been completed
    };

    RenderPrepExecTracker();
    ~RenderPrepExecTracker();

    void setRenderPrepStatsCallBack(const RenderPrepStatsCallBack &callBack);
    RenderPrepStatsCallBack getRenderPrepStatsCallBack() const;

    void setRenderPrepCancelCallBack(const RenderPrepCancelCallBack &callBack);
    RenderPrepCancelCallBack getRenderPrepCancelCallBack() const;

    void setMsgHandlerCallBack(const MsgHandlerCallBack &callBack);
    MsgHandlerCallBack getMsgHandlerCallBack() const;

    void init();

    RESULT startRenderPrep();

    RESULT startApplyUpdate();
    RESULT endApplyUpdate();

    // We have 2 loadGeom stages. loadGeom0 for regular layer and loadGeom1 for meshLight layer.
    // Probably some detailed progress update API will be added between startLoadGeom and endLoadGeom
    // in the future in order to capture more detailed progress updates.
    RESULT startLoadGeom0();
    RESULT endLoadGeom0();
    RESULT startLoadGeom1();
    RESULT endLoadGeom1();

    // We have 2 finalizeChange stages. finalizeChange0 for regular layer and finalizeChange1
    // for meshLight layer.
    RESULT startFinalizeChange0();
    RESULT endFinalizeChange0();
    RESULT startFinalizeChange1();
    RESULT endFinalizeChange1();

    RESULT endRenderPrep();

    //------------------------------

    scene_rdl2::grid_util::Parser& getParser();

    std::string cancelInfoEncode() const; // for debug console
    void cancelInfoDecode(const std::string &data); // for debug console

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace rndr
} // namespace moonray

#endif // RENDERPREPEXECTRACKER_H
