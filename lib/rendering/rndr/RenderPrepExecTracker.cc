// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RenderPrepExecTracker.h"

#include <scene_rdl2/common/grid_util/Arg.h>
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/common/grid_util/RenderPrepStats.h>
#include <scene_rdl2/scene/rdl2/ValueContainerDeq.h>
#include <scene_rdl2/scene/rdl2/ValueContainerEnq.h>

//#define DEBUG_MSG

#ifdef DEBUG_MSG
#include <iostream>
#endif // end DEBUG_MSG

namespace moonray {
namespace rndr {

class RenderPrepExecTracker::Impl
{
public:
    // This is for a call back function in order to report renderPrep progress information
    // to the downstream computation. This functionality is only used under arras context.
    using RenderPrepStatsCallBack = std::function<void(const scene_rdl2::grid_util::RenderPrepStats &rPrepStats)>;
    using RenderPrepCancelCallBack = std::function<bool()>;
    using MsgHandlerCallBack = std::function<void(const std::string &msg)>;
    using Parser = scene_rdl2::grid_util::Parser;
    using Arg = scene_rdl2::grid_util::Arg;
    using RESULT = RenderPrepExecTracker::RESULT;

    Impl() :
        mRenderPrepStatsCallBack(nullptr),
        mRenderPrepCancelCallBack(nullptr),
        mMsgHandlerCallBack(nullptr),
        mCancelCodePos(CancelCodePos::EMPTY)
    {
        parserConfigure();
        init();
    }

    void setRenderPrepStatsCallBack(const RenderPrepStatsCallBack &callBack);
    RenderPrepStatsCallBack getRenderPrepStatsCallBack() const { return mRenderPrepStatsCallBack; }

    void setRenderPrepCancelCallBack(const RenderPrepCancelCallBack &callBack);
    RenderPrepCancelCallBack getRenderPrepCancelCallBack() const { return mRenderPrepCancelCallBack; }

    void setMsgHandlerCallBack(const MsgHandlerCallBack &callBack);
    MsgHandlerCallBack getMsgHandlerCallBack() const { return mMsgHandlerCallBack; }

    void init();

    RESULT startRenderPrep();

    RESULT startApplyUpdate();
    RESULT endApplyUpdate();

    RESULT startLoadGeom0();
    RESULT endLoadGeom0();
    RESULT startLoadGeom1();
    RESULT endLoadGeom1();

    RESULT startFinalizeChange0();
    RESULT endFinalizeChange0();
    RESULT startFinalizeChange1();
    RESULT endFinalizeChange1();

    RESULT endRenderPrep();

    Parser& getParser() { return mParser; }

    std::string cancelInfoEncode() const; // for debug console
    void cancelInfoDecode(const std::string &data); // for debug console

    std::string show() const;

private:
    enum class Condition : int { INIT, START, START_CANCELED, END, END_CANCELED };
    enum class CancelCodePos : int {
        EMPTY = 0,
        RENDER_PREP_START,
        APPLY_UPDATE_START,
        APPLY_UPDATE_END,
        LOAD_GEOM0_START,
        LOAD_GEOM0_END,
        LOAD_GEOM1_START,
        LOAD_GEOM1_END,
        FINALIZE_CHANGE0_START,
        FINALIZE_CHANGE0_END,
        FINALIZE_CHANGE1_START,
        FINALIZE_CHANGE1_END,
        RENDER_PREP_END,
        MAX
    };

    RESULT updateRunStatus(const CancelCodePos &callerCodePos,
                           Condition &updateTarget,
                           const Condition &finishCondition,
                           const Condition &cancelCondition) const;
    RESULT checkRunStatus(const CancelCodePos &callerCodePos) const;
    void renderPrepStatsUpdate() const;
    scene_rdl2::grid_util::RenderPrepStats calcRenderPrepStats() const;

    void parserConfigure();
    std::string showCancelCodePosIdList() const;

    static std::string showCondition(const Condition &condition);
    static std::string showCancelCodePos(const CancelCodePos &cancelCodePos);
    static std::string showResult(const RESULT &result);

    RenderPrepStatsCallBack mRenderPrepStatsCallBack;
    RenderPrepCancelCallBack mRenderPrepCancelCallBack;
    MsgHandlerCallBack mMsgHandlerCallBack;

    //------------------------------

    // renderPrep top level execution condition.
    Condition mRunRenderPrep;

    // internal of renderPrep stage condition
    Condition mRunApplyUpdate;
    Condition mRunLoadGeom0;
    Condition mRunLoadGeom1;
    Condition mRunFinalizeChange0;
    Condition mRunFinalizeChange1;

    //------------------------------

    Parser mParser;
    CancelCodePos mCancelCodePos;
};

void
RenderPrepExecTracker::Impl::setRenderPrepStatsCallBack(const RenderPrepStatsCallBack &callBack)
{
    mRenderPrepStatsCallBack = callBack;
}

void
RenderPrepExecTracker::Impl::setRenderPrepCancelCallBack(const RenderPrepCancelCallBack &callBack)
{
    mRenderPrepCancelCallBack = callBack;
}

void
RenderPrepExecTracker::Impl::setMsgHandlerCallBack(const MsgHandlerCallBack &callBack)
{
    mMsgHandlerCallBack = callBack;
}

void
RenderPrepExecTracker::Impl::init()
{
    mRunRenderPrep = Condition::INIT;
    mRunApplyUpdate = Condition::INIT;
    mRunLoadGeom0 = Condition::INIT;
    mRunLoadGeom1 = Condition::INIT;
    mRunFinalizeChange0 = Condition::INIT;
    mRunFinalizeChange1 = Condition::INIT;
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::startRenderPrep()
{
    return updateRunStatus(CancelCodePos::RENDER_PREP_START,
                           mRunRenderPrep,
                           Condition::START,
                           Condition::START_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::startApplyUpdate()
{
    return updateRunStatus(CancelCodePos::APPLY_UPDATE_START,
                           mRunApplyUpdate,
                           Condition::START,
                           Condition::START_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::endApplyUpdate()
{
    return updateRunStatus(CancelCodePos::APPLY_UPDATE_END,
                           mRunApplyUpdate,
                           Condition::END,
                           Condition::END_CANCELED);
}    

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::startLoadGeom0()
{
    return updateRunStatus(CancelCodePos::LOAD_GEOM0_START,
                           mRunLoadGeom0,
                           Condition::START,
                           Condition::START_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::endLoadGeom0()
{
    return updateRunStatus(CancelCodePos::LOAD_GEOM0_END,
                           mRunLoadGeom0,
                           Condition::END,
                           Condition::END_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::startLoadGeom1()
{
    return updateRunStatus(CancelCodePos::LOAD_GEOM1_START,
                           mRunLoadGeom1,
                           Condition::START,
                           Condition::START_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::endLoadGeom1()
{
    return updateRunStatus(CancelCodePos::LOAD_GEOM1_END,
                           mRunLoadGeom1,
                           Condition::END,
                           Condition::END_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::startFinalizeChange0()
{
    return updateRunStatus(CancelCodePos::FINALIZE_CHANGE0_START,
                           mRunFinalizeChange0,
                           Condition::START,
                           Condition::START_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::endFinalizeChange0()
{
    return updateRunStatus(CancelCodePos::FINALIZE_CHANGE0_END,
                           mRunFinalizeChange0,
                           Condition::END,
                           Condition::END_CANCELED);
}    

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::startFinalizeChange1()
{
    return updateRunStatus(CancelCodePos::FINALIZE_CHANGE1_START,
                           mRunFinalizeChange1,
                           Condition::START,
                           Condition::START_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::endFinalizeChange1()
{
    return updateRunStatus(CancelCodePos::FINALIZE_CHANGE1_END,
                           mRunFinalizeChange1,
                           Condition::END,
                           Condition::END_CANCELED);
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::endRenderPrep()
{
    return updateRunStatus(CancelCodePos::RENDER_PREP_END,
                           mRunRenderPrep,
                           Condition::END,
                           Condition::END_CANCELED);
}

std::string
RenderPrepExecTracker::Impl::cancelInfoEncode() const
{
    std::string data;
    scene_rdl2::rdl2::ValueContainerEnq vcEnq(&data);
    vcEnq.enqInt(static_cast<int>(mCancelCodePos));
    vcEnq.finalize();
    return data;
}

void
RenderPrepExecTracker::Impl::cancelInfoDecode(const std::string &data)
{
    scene_rdl2::rdl2::ValueContainerDeq vcDeq(data.data(), data.size());
    mCancelCodePos = static_cast<CancelCodePos>(vcDeq.deqInt());
}

std::string
RenderPrepExecTracker::Impl::show() const
{
    auto showCancelCodePosWithId = [&]() -> std::string {
        std::ostringstream ostr;
        ostr << showCancelCodePos(mCancelCodePos)
             << " (cancelCodePos:" << static_cast<int>(mCancelCodePos) << ")";
        return ostr.str();
    };
    auto showCurrentCancelCondition = [&]() -> std::string {
        if (!mRenderPrepCancelCallBack) return "?";
        if (mRenderPrepCancelCallBack()) return "cancel";
        return "run";
    };

    std::ostringstream ostr;
    ostr << "RenderPrepExecTracker {\n"
         << "  mRenderPrepStatsCallBack:" << (mRenderPrepStatsCallBack ? "set" : "empty") << '\n'
         << "  mRenderPrepCancelCallBack:" << (mRenderPrepCancelCallBack ? "set" : "empty")
         << " status:" << showCurrentCancelCondition() << '\n'
         << "  mMsgHandlerCallBack:" << (mMsgHandlerCallBack ? "set" : "empty") << '\n'
         << "  mRunRenderPrep:" << showCondition(mRunRenderPrep) << '\n'
         << "  mRunApplyUpdate:" << showCondition(mRunApplyUpdate) << '\n'
         << "  mRunLoadGeom0:" << showCondition(mRunLoadGeom0) << '\n'
         << "  mRunLoadGeom1:" << showCondition(mRunLoadGeom1) << '\n'
         << "  mRunFinalizeChange0:" << showCondition(mRunFinalizeChange0) << '\n'
         << "  mRunFinalizeChange1:" << showCondition(mRunFinalizeChange1) << '\n'
         << "  mCancelCodePos:" << showCancelCodePosWithId() << '\n'
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::updateRunStatus(const CancelCodePos &callerCodePos,
                                             Condition &updateTarget,
                                             const Condition &finishCondition,
                                             const Condition &cancelCondition) const
{
    RESULT result = checkRunStatus(callerCodePos);
    if (result == RESULT::FINISHED) {
        updateTarget = finishCondition;
    } else {
        updateTarget = cancelCondition;
    }

    renderPrepStatsUpdate();

    if (mMsgHandlerCallBack && result == RESULT::CANCELED) {
        std::ostringstream ostr;
        ostr << ">> RenderPrepExecTracker.cc RenderPrepExecTracker::Impl::updateRunStatus()"
             << " callerCodePos:" << showCancelCodePos(callerCodePos)
             << " result:" << showResult(result);
        (mMsgHandlerCallBack)(ostr.str());
#       ifdef DEBUG_MSG
        std::cerr << ostr.str() << '\n';
#       endif // end DEBUG_MSG    
    }

    return result;
}

RenderPrepExecTracker::Impl::RESULT
RenderPrepExecTracker::Impl::checkRunStatus(const CancelCodePos &callerCodePos) const
{
    if (callerCodePos == mCancelCodePos) {
        // This is a case of cancelTest enable for debugging purpose.
        std::cerr << ">> RenderPrepExecTracker.cc checkRunStatus debug CANCELED\n";
        return RESULT::CANCELED;
    }

    if (!mRenderPrepCancelCallBack) {
        // no cancel condition check -> always run
        return RESULT::FINISHED;
    }

    if (mRenderPrepCancelCallBack()) {
        return RESULT::CANCELED;
    }

    return RESULT::FINISHED;
}

void
RenderPrepExecTracker::Impl::renderPrepStatsUpdate() const
{
    if (!mRenderPrepStatsCallBack) return;

    scene_rdl2::grid_util::RenderPrepStats rPrepStats = calcRenderPrepStats();
    (mRenderPrepStatsCallBack)(rPrepStats);

#   ifdef DEBUG_MSG
    std::cerr << "RenderPrepExecTracker::Impl::renderPrepStatsUpdate() stage:" << rPrepStats.show() << '\n';
#   endif // end DEBUG_MSG    
}

scene_rdl2::grid_util::RenderPrepStats
RenderPrepExecTracker::Impl::calcRenderPrepStats() const
//
// convert this class's internal status to the RenderPrepStats which we need to set as
// the argument of the callback function.
//    
{
    using RenderPrepStats = scene_rdl2::grid_util::RenderPrepStats;
    using Stage = RenderPrepStats::Stage;

    switch (mRunRenderPrep) {
    case Condition::INIT :           return RenderPrepStats(Stage::NOT_ACTIVE);
    case Condition::START_CANCELED : return RenderPrepStats(Stage::RENDER_PREP_START_CANCELED);
    case Condition::END :            return RenderPrepStats(Stage::RENDER_PREP_DONE);          // early exit
    case Condition::END_CANCELED :   return RenderPrepStats(Stage::RENDER_PREP_DONE_CANCELED); // early exit
    default : break;
    }

    // renderPrep started
    switch (mRunApplyUpdate) {
    case Condition::INIT :           return RenderPrepStats(Stage::RENDER_PREP_START);
    case Condition::START :          return RenderPrepStats(Stage::RENDER_PREP_APPLYUPDATE);
    case Condition::START_CANCELED : return RenderPrepStats(Stage::RENDER_PREP_APPLYUPDATE_CANCELED);
    case Condition::END_CANCELED :   return RenderPrepStats(Stage::RENDER_PREP_APPLYUPDATE_DONE_CANCELED);
    default : break;
    }

    // applyUpdate endded
    switch (mRunLoadGeom0) {
    case Condition::INIT :           return RenderPrepStats(Stage::RENDER_PREP_APPLYUPDATE_DONE);
    case Condition::START :          return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM0);
    case Condition::START_CANCELED : return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM0_CANCELED);
    case Condition::END_CANCELED:    return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM0_DONE_CANCELED);
    default : break;
    }

    // loadGeom0 ended
    switch (mRunLoadGeom1) {
    case Condition::INIT :           return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM0_DONE);
    case Condition::START :          return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM1);
    case Condition::START_CANCELED : return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM1_CANCELED);
    case Condition::END_CANCELED :   return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM1_DONE_CANCELED);
    default : break;
    }
        
    // loadGeom1 ended
    switch (mRunFinalizeChange0) {
    case Condition::INIT :           return RenderPrepStats(Stage::RENDER_PREP_LOAD_GEOM1_DONE);
    case Condition::START :          return RenderPrepStats(Stage::GM_FINALIZE0_START);
    case Condition::START_CANCELED : return RenderPrepStats(Stage::GM_FINALIZE0_START_CANCELED);
    case Condition::END_CANCELED :   return RenderPrepStats(Stage::GM_FINALIZE0_DONE_CANCELED);
    default : break;
    }

    // finalizeChange0 ended
    switch (mRunFinalizeChange1) {
    case Condition::INIT :           return RenderPrepStats(Stage::GM_FINALIZE0_DONE);
    case Condition::START :          return RenderPrepStats(Stage::GM_FINALIZE1_START);
    case Condition::START_CANCELED : return RenderPrepStats(Stage::GM_FINALIZE1_START_CANCELED);
    case Condition::END_CANCELED :   return RenderPrepStats(Stage::GM_FINALIZE1_DONE_CANCELED);
    default : break;
    }

    // finalizeChange1 ended
    return RenderPrepStats(Stage::GM_FINALIZE1_DONE);
}

void
RenderPrepExecTracker::Impl::parserConfigure()
{
    mParser.description("renderPrepExecTracker command");

    mParser.opt("cancelTest", "<posId>", "set test cancel action, posId = 0 disables cancelTest",
                [&](Arg &arg) -> bool {
                    int posId = (arg++).as<int>(0);
                    int max = static_cast<int>(CancelCodePos::MAX);
                    if (posId < 0 || max <= posId) {
                        return arg.msg(std::string("WARNING : posId out of range:") + std::to_string(posId) +
                                       " max:" + std::to_string(max - 1) + '\n');
                    }
                    mCancelCodePos = static_cast<CancelCodePos>(posId);
                    return true;
                });
    mParser.opt("cancelCodePosIdList", "", "show cancel code position id list",
                [&](Arg &arg) -> bool { return arg.msg(showCancelCodePosIdList() + '\n'); });
    mParser.opt("show", "", "show internal parameters",
                [&](Arg &arg) -> bool { return arg.msg(show() + '\n'); });
}

std::string
RenderPrepExecTracker::Impl::showCancelCodePosIdList() const
{
    std::ostringstream ostr;
    int size = static_cast<int>(CancelCodePos::MAX);
    int w = std::to_string(size).size();
    ostr << "cancelCodePos id table (size:" << size << ") {\n";
    for (int i = 0; i < size; ++i) {
        ostr << "  " << std::setw(w) << i << " : " << showCancelCodePos(static_cast<CancelCodePos>(i)) << '\n';
    }
    ostr << "}";
    return ostr.str();
}

// static function
std::string
RenderPrepExecTracker::Impl::showCondition(const Condition &condition)
{
    switch (condition) {
    case Condition::INIT : return "INIT";
    case Condition::START : return "START";
    case Condition::START_CANCELED : return "START_CANCELED";
    case Condition::END : return "END";
    case Condition::END_CANCELED : return "END_CANCELED";
    default : return "?";
    }
}

// static function
std::string
RenderPrepExecTracker::Impl::showCancelCodePos(const CancelCodePos &cancelCodePos)
{
    switch (cancelCodePos) {
    case CancelCodePos::EMPTY : return "EMPTY";
    case CancelCodePos::RENDER_PREP_START : return "RENDER_PREP_START";
    case CancelCodePos::APPLY_UPDATE_START : return "APPLY_UPDATE_START";
    case CancelCodePos::APPLY_UPDATE_END : return "APPLY_UPDATE_END";
    case CancelCodePos::LOAD_GEOM0_START : return "LOAD_GEOM0_START";
    case CancelCodePos::LOAD_GEOM0_END : return "LOAD_GEOM0_END";
    case CancelCodePos::LOAD_GEOM1_START : return "LOAD_GEOM1_START";
    case CancelCodePos::LOAD_GEOM1_END : return "LOAD_GEOM1_END";
    case CancelCodePos::FINALIZE_CHANGE0_START : return "FINALIZE_CHANGE0_START";
    case CancelCodePos::FINALIZE_CHANGE0_END : return "FINALIZE_CHANGE0_END";
    case CancelCodePos::FINALIZE_CHANGE1_START : return "FINALIZE_CHANGE1_START";
    case CancelCodePos::FINALIZE_CHANGE1_END : return "FINALIZE_CHANGE1_END";
    case CancelCodePos::RENDER_PREP_END : return "RENDER_PREP_END";
    case CancelCodePos::MAX : return "MAX";
    default : return "?";
    }
}

// static function
std::string
RenderPrepExecTracker::Impl::showResult(const RESULT &result)
{
    switch (result) {
    case RESULT::CANCELED : return "CANCELED";
    case RESULT::FINISHED : return "FINISHED";
    default : return "?";
    }
}

//------------------------------------------------------------------------------------------

RenderPrepExecTracker::RenderPrepExecTracker()
{
    mImpl.reset(new Impl);
}

RenderPrepExecTracker::~RenderPrepExecTracker()    
{
}

void
RenderPrepExecTracker::setRenderPrepStatsCallBack(const RenderPrepStatsCallBack &callBack)
{
    mImpl->setRenderPrepStatsCallBack(callBack);
}

RenderPrepExecTracker::RenderPrepStatsCallBack
RenderPrepExecTracker::getRenderPrepStatsCallBack() const
{
    return mImpl->getRenderPrepStatsCallBack();
}

void
RenderPrepExecTracker::setRenderPrepCancelCallBack(const RenderPrepCancelCallBack &callBack)
{
    mImpl->setRenderPrepCancelCallBack(callBack);
}

RenderPrepExecTracker::RenderPrepCancelCallBack
RenderPrepExecTracker::getRenderPrepCancelCallBack() const
{
    return mImpl->getRenderPrepCancelCallBack();
}

void
RenderPrepExecTracker::setMsgHandlerCallBack(const MsgHandlerCallBack &callBack)
{
    mImpl->setMsgHandlerCallBack(callBack);
}

RenderPrepExecTracker::MsgHandlerCallBack    
RenderPrepExecTracker::getMsgHandlerCallBack() const
{
    return mImpl->getMsgHandlerCallBack();
}

void
RenderPrepExecTracker::init()
{
    mImpl->init();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::startRenderPrep()
{
    return mImpl->startRenderPrep();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::startApplyUpdate()
{
    return mImpl->startApplyUpdate();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::endApplyUpdate()
{
    return mImpl->endApplyUpdate();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::startLoadGeom0()
{
    return mImpl->startLoadGeom0();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::endLoadGeom0()
{
    return mImpl->endLoadGeom0();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::startLoadGeom1()
{
    return mImpl->startLoadGeom1();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::endLoadGeom1()
{
    return mImpl->endLoadGeom1();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::startFinalizeChange0()
{
    return mImpl->startFinalizeChange0();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::endFinalizeChange0()
{
    return mImpl->endFinalizeChange0();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::startFinalizeChange1()
{
    return mImpl->startFinalizeChange1();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::endFinalizeChange1()
{
    return mImpl->endFinalizeChange1();
}

RenderPrepExecTracker::RESULT
RenderPrepExecTracker::endRenderPrep()
{
    return mImpl->endRenderPrep();
}

scene_rdl2::grid_util::Parser &
RenderPrepExecTracker::getParser()
{
    return mImpl->getParser();
}

std::string
RenderPrepExecTracker::cancelInfoEncode() const
{
    return mImpl->cancelInfoEncode();
}

void    
RenderPrepExecTracker::cancelInfoDecode(const std::string &data)
{
    mImpl->cancelInfoDecode(data);
}

} // namespace rndr
} // namespace moonray

