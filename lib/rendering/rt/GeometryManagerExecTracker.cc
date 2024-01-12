// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "GeometryManagerExecTracker.h"

#include <scene_rdl2/common/grid_util/RenderPrepStats.h>
#include <scene_rdl2/scene/rdl2/ValueContainerDeq.h>
#include <scene_rdl2/scene/rdl2/ValueContainerEnq.h>

//#Define DEBUG_MSG

#ifdef DEBUG_MSG
#include <iostream>
#endif // end DEBUG_MSG

namespace moonray {
namespace rt {

void
GeometryManagerExecTracker::initLoadGeometries(int stageId)
{
    mStageId = stageId;

    for (int i = stageId; i < mStageMax; ++i) {
        mRunLoadGeometries[i] = Condition::INIT;
        mRunLoadGeometriesTotal[i] = 0;
        mRunLoadGeometriesItem[i] = Condition::INIT;
        mRunLoadGeometriesProcessed[i] = 0;
    }

    mRenderPrepStatsCallBack = nullptr;
    mRenderPrepCancelCallBack = nullptr;
}

void
GeometryManagerExecTracker::initFinalizeChange(int stageId)
{
    mStageId = stageId;

    for (int i = stageId; i < mStageMax; ++i) {
        mRunTessellation[i] = Condition::INIT;
        mRunBVHConstruction[i] = Condition::INIT;
    }

    mRenderPrepStatsCallBack = nullptr;
    mRenderPrepCancelCallBack = nullptr;
}

void
GeometryManagerExecTracker::setRenderPrepStatsCallBack(const RenderPrepStatsCallBack &callBack)
{
    mRenderPrepStatsCallBack = callBack;
}

void
GeometryManagerExecTracker::setRenderPrepCancelCallBack(const RenderPrepCancelCallBack &callBack)
{
    mRenderPrepCancelCallBack = callBack;
}

void
GeometryManagerExecTracker::setMsgHandlerCallBack(const MsgHandlerCallBack &callBack)
{
    mMsgHandlerCallBack = callBack;
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::startLoadGeometries(int totalGeometries)
{
    mRunLoadGeometriesTotal[mStageId] = totalGeometries;
    mRunLoadGeometriesProcessed[mStageId] = 0; // just in case

    return updateRunStatus(CancelCodePos::LOADGEOMETRIES_0_START,
                           CancelCodePos::LOADGEOMETRIES_1_START,
                           mRunLoadGeometries[mStageId],
                           Condition::START,
                           Condition::START_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::startLoadGeometriesItem()
{
    // We update mRunLoadGeometriesItem condition by special value (Condition::ETC)
    return updateRunStatus(CancelCodePos::LOADGEOMETRIES_ITEM_0_START,
                           CancelCodePos::LOADGEOMETRIES_ITEM_1_START,
                           mRunLoadGeometriesItem[mStageId],
                           Condition::ETC,
                           Condition::ETC_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::endLoadGeometriesItem()
{
    mRunLoadGeometriesProcessed[mStageId]++; // atomic operation

    return updateRunStatus(CancelCodePos::LOADGEOMETRIES_ITEM_0_END,
                           CancelCodePos::LOADGEOMETRIES_ITEM_1_END,
                           mRunLoadGeometriesItem[mStageId],
                           Condition::ETC,
                           Condition::ETC_CANCELED);
}

void
GeometryManagerExecTracker::finalizeLoadGeometriesItem(bool canceled)
{
    Condition condition = (canceled) ? Condition::END_CANCELED : Condition::END;
    mRunLoadGeometriesItem[mStageId] = condition;
    mRunLoadGeometries[mStageId] = condition;
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::endLoadGeometries()
{
    return updateRunStatus(CancelCodePos::LOADGEOMETRIES_0_END,
                           CancelCodePos::LOADGEOMETRIES_1_END,
                           mRunLoadGeometries[mStageId],
                           Condition::END,
                           Condition::END_CANCELED);
}

//------------------------------

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::startFinalizeChange()
{
    return updateRunStatus(CancelCodePos::FINALIZE_CHANGE_0_START,
                           CancelCodePos::FINALIZE_CHANGE_1_START,
                           mRunFinalizeChange[mStageId],
                           Condition::START,
                           Condition::START_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::startTessellation(int totalTessellation)
{
    mRunTessellationTotal[mStageId] = totalTessellation;
    mRunTessellationProcessed[mStageId] = 0; // just in case

    return updateRunStatus(CancelCodePos::TESSELLATION_0_START,
                           CancelCodePos::TESSELLATION_1_START,
                           mRunTessellation[mStageId],
                           Condition::START,
                           Condition::START_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::startTessellationItem()
{
    // We update mRunTessellationItem condition by special value (Condition::ETC)
    return updateRunStatus(CancelCodePos::TESSELLATION_ITEM_0_START,
                           CancelCodePos::TESSELLATION_ITEM_1_START,
                           mRunTessellationItem[mStageId],
                           Condition::ETC,
                           Condition::ETC_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::endTessellationItem()
{
    mRunTessellationProcessed[mStageId]++; // atomic operation

    return updateRunStatus(CancelCodePos::TESSELLATION_ITEM_0_END,
                           CancelCodePos::TESSELLATION_ITEM_1_END,
                           mRunTessellationItem[mStageId],
                           Condition::ETC,
                           Condition::ETC_CANCELED);
}

void
GeometryManagerExecTracker::finalizeTessellationItem(bool canceled)
{
    Condition condition = (canceled) ? Condition::END_CANCELED : Condition::END;
    mRunTessellationItem[mStageId] = condition;
    mRunTessellation[mStageId] = condition;
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::endTessellation()
{
    return updateRunStatus(CancelCodePos::TESSELLATION_0_END,
                           CancelCodePos::TESSELLATION_1_END,
                           mRunTessellation[mStageId],
                           Condition::END,
                           Condition::END_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::startBVHConstruction()
{
    return updateRunStatus(CancelCodePos::BVH_CONSTRUCTION_0_START,
                           CancelCodePos::BVH_CONSTRUCTION_1_START,
                           mRunBVHConstruction[mStageId],
                           Condition::START,
                           Condition::START_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::endBVHConstruction()
{
    return updateRunStatus(CancelCodePos::BVH_CONSTRUCTION_0_END,
                           CancelCodePos::BVH_CONSTRUCTION_1_END,
                           mRunBVHConstruction[mStageId],
                           Condition::END,
                           Condition::END_CANCELED);
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::endFinalizeChange()
{
    return updateRunStatus(CancelCodePos::FINALIZE_CHANGE_0_END,
                           CancelCodePos::FINALIZE_CHANGE_1_END,
                           mRunFinalizeChange[mStageId],
                           Condition::END,
                           Condition::END_CANCELED);
}

std::string
GeometryManagerExecTracker::cancelInfoEncode() const
{
    std::string data;
    scene_rdl2::rdl2::ValueContainerEnq vcEnq(&data);
    vcEnq.enqInt(static_cast<int>(mCancelCodePos));
    vcEnq.enqInt(mCancelCodePosLoadGeomCounter);
    vcEnq.enqInt(mCancelCodePosTessellationCounter);
    vcEnq.finalize();
    return data;
}

void
GeometryManagerExecTracker::cancelInfoDecode(const std::string &data)
{
    scene_rdl2::rdl2::ValueContainerDeq vcDeq(data.data(), data.size());
    mCancelCodePos = static_cast<CancelCodePos>(vcDeq.deqInt());
    mCancelCodePosLoadGeomCounter = vcDeq.deqInt();
    mCancelCodePosTessellationCounter = vcDeq.deqInt();
}

std::string
GeometryManagerExecTracker::show() const
{
    auto showConditionVec = [](const Condition vec[mStageMax]) -> std::string {
        std::ostringstream ostr;
        for (int i = 0; i < mStageMax; ++i) {
            if (i != 0) ostr << " ";
            ostr << showCondition(vec[i]);
        }
        return ostr.str();
    };
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
    ostr << "GeometryManagerExecTracker (0x:"
         << std::hex << reinterpret_cast<uintptr_t>(this) << std::dec << ") {\n"
         << "  mRenderPrepStatsCallBack:" << (mRenderPrepStatsCallBack ? "set" : "empty") << '\n'
         << "  mRenderPrepCancelCallBack:" << (mRenderPrepCancelCallBack ? "set" : "empty")
         << " status:" << showCurrentCancelCondition() << '\n'
         << "  mMsgHandlerCallBack:" << (mMsgHandlerCallBack ? "set" : "empty") << '\n'
         << "  mStageId:" << mStageId << '\n'
         << "  stage_0 loadGeometries {\n"
         << "    mRunLoadGeometries:" << showCondition(mRunLoadGeometries[0]) << '\n'
         << "    mRunLoadGeometriesTotal:" << mRunLoadGeometriesTotal[0] << '\n'
         << "    mRunLoadGeometriesItem:" << showCondition(mRunLoadGeometriesItem[0]) << '\n'
         << "    mRunLoadGeometriesProcessed:" << mRunLoadGeometriesProcessed[0] << '\n'
         << "  }\n"
         << "  stage_1 loadGeometries {\n"
         << "    mRunLoadGeometries:" << showCondition(mRunLoadGeometries[1]) << '\n'
         << "    mRunLoadGeometriesTotal:" << mRunLoadGeometriesTotal[1] << '\n'
         << "    mRunLoadGeometriesItem:" << showCondition(mRunLoadGeometriesItem[1]) << '\n'
         << "    mRunLoadGeometriesProcessed:" << mRunLoadGeometriesProcessed[1] << '\n'
         << "  }\n"
         << "  stage_0 finalizeChange {\n"
         << "    mRunFinalizeChange:" << showCondition(mRunFinalizeChange[0]) << '\n' 
         << "    mRunTessellation:" << showCondition(mRunTessellation[0]) << '\n'
         << "    mRunTessellationTotal:" << mRunTessellationTotal[0] << '\n'
         << "    mRunTessellationItem:" << showCondition(mRunTessellationItem[0]) << '\n'
         << "    mRunTessellationProcessed:" << mRunTessellationProcessed[0] << '\n'
         << "    mRunBVHConstruction:" << showCondition(mRunBVHConstruction[0]) << '\n'
         << "  }\n"
         << "  stage_1 finalizeChange {\n"
         << "    mRunFinalizeChange:" << showCondition(mRunFinalizeChange[1]) << '\n' 
         << "    mRunTessellation:" << showCondition(mRunTessellation[1]) << '\n'
         << "    mRunTessellationTotal:" << mRunTessellationTotal[1] << '\n'
         << "    mRunTessellationItem:" << showCondition(mRunTessellationItem[1]) << '\n'
         << "    mRunTessellationProcessed:" << mRunTessellationProcessed[1] << '\n'
         << "    mRunBVHConstruction:" << showCondition(mRunBVHConstruction[1]) << '\n'
         << "  }\n"
         << "  mCancelCodePos:" << showCancelCodePosWithId() << '\n'
         << "  mCancelCodePosLoadGeomCounter:" << mCancelCodePosLoadGeomCounter << '\n'
         << "  mCancelCodePosTessellationCounter:" << mCancelCodePosTessellationCounter << '\n'
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

void
GeometryManagerExecTracker::parserConfigure()
{
    mParser.description("finalizeChangeExecTracker command");

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
    mParser.opt("cancelLoadGeomCounter", "<id>", "set loadGeomCounter-id for loadGeometry cancelTest",
                [&](Arg &arg) -> bool {
                    mCancelCodePosLoadGeomCounter = (arg++).as<int>(0);
                    return true;
                });
    mParser.opt("cancelTessellationCounter", "<id>", "set tessellationCounter-id for tessellation cacelTest",
                [&](Arg &arg) -> bool {
                    mCancelCodePosTessellationCounter = (arg++).as<int>(0);
                    return true;
                });
    mParser.opt("cancelCodePosIdList", "", "show cancel code position id list",
                [&](Arg &arg) -> bool { return arg.msg(showCancelCodePosIdList() + '\n'); });
    mParser.opt("show", "", "show internal parameters",
                [&](Arg &arg) -> bool { return arg.msg(show() + '\n'); });
}

void
GeometryManagerExecTracker::updateRunStatusCancelMessage(const CancelCodePos &callerCodePos,
                                                         RESULT result) const
{
    std::ostringstream ostr;
    ostr << ">> GeometryManagerExecTracer.cc GeometryManagerExecTracker::updateRunStatus()"
         << " mStageId:" << mStageId
         << " callerCodePos:" << showCancelCodePos(callerCodePos)
         << " result:" << showResult(result);
    (mMsgHandlerCallBack)(ostr.str());

#   ifdef DEBUG_MSG
    std::cerr << ostr.str() << '\n';
#   endif // end DEBUG_MSG    
}

GeometryManagerExecTracker::RESULT
GeometryManagerExecTracker::checkRunStatus(const CancelCodePos &callerCodePos) const
{
    if (callerCodePos == mCancelCodePos) {
        // This is a case of cancelTest enable for debugging purpose.
        switch (callerCodePos) {
        case CancelCodePos::LOADGEOMETRIES_ITEM_0_START :
        case CancelCodePos::LOADGEOMETRIES_ITEM_0_END :
        case CancelCodePos::LOADGEOMETRIES_ITEM_1_START :
        case CancelCodePos::LOADGEOMETRIES_ITEM_1_END :
            // We have to check cancelCodePosLoadGeomCounter
            if (mCancelCodePosLoadGeomCounter <= mRunLoadGeometriesProcessed[mStageId]) {
                return RESULT::CANCELED;
            }
            break;
        case CancelCodePos::TESSELLATION_ITEM_0_START :
        case CancelCodePos::TESSELLATION_ITEM_0_END :
        case CancelCodePos::TESSELLATION_ITEM_1_START :
        case CancelCodePos::TESSELLATION_ITEM_1_END :
            // We have to check cancelCodePosTessellateCounter
            if (mCancelCodePosTessellationCounter <= mRunTessellationProcessed[mStageId]) {
                return RESULT::CANCELED;
            }
            break;
        default :
            return RESULT::CANCELED;
        }
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
GeometryManagerExecTracker::renderPrepStatsUpdate() const
{
    if (!mRenderPrepStatsCallBack) return;

    scene_rdl2::grid_util::RenderPrepStats rPrepStats = calcRenderPrepStats();
    (mRenderPrepStatsCallBack)(rPrepStats);

#   ifdef DEBUG_MSG
    std::cerr << "GeometryManagerExecTracker::update() stage:" << rPrepStats.show() << '\n';
#   endif // end DEBUG_MSG    
}

scene_rdl2::grid_util::RenderPrepStats
GeometryManagerExecTracker::calcRenderPrepStats() const
//
// convert this class's internal status to the RenderPrepStats which we need to set as
// the argument of the callback function.
//    
{
    using Stage = scene_rdl2::grid_util::RenderPrepStats::Stage;

    auto isLoadGeometriesStage = [&]() -> bool {
        return (mRunLoadGeometries[0] != Condition::END || mRunLoadGeometries[1] != Condition::END);
    };

    auto loadGeometriesRenderPrepStats = [&](const int stageId,
                                             scene_rdl2::grid_util::RenderPrepStats &rPrepStats) -> bool {
        bool flag = true;

        switch (mRunLoadGeometries[stageId]) {
        case Condition::INIT :
            // not start loadGeometries yet
            flag = false;
            break;

        //------------------------------
        // loadGeometries started
        case Condition::START_CANCELED :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_LOADGEO0_START_CANCELED;
            else              rPrepStats.stage() = Stage::GM_LOADGEO1_START_CANCELED;
            rPrepStats.loadGeometriesTotal(stageId) = mRunLoadGeometriesTotal[stageId];
            break;
        case Condition::END :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_LOADGEO0_DONE;
            else              rPrepStats.stage() = Stage::GM_LOADGEO1_DONE;
            rPrepStats.loadGeometriesProcessed(stageId) = mRunLoadGeometriesProcessed[stageId];
            break;
        case Condition::END_CANCELED :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_LOADGEO0_DONE_CANCELED;
            else              rPrepStats.stage() = Stage::GM_LOADGEO1_DONE_CANCELED;
            rPrepStats.loadGeometriesProcessed(stageId) = mRunLoadGeometriesProcessed[stageId];
            break;

        //------------------------------
        // loadGeometri started but not finished yet
        default :
            if (mRunLoadGeometriesItem[stageId] == Condition::INIT) {
                if (stageId == 0) rPrepStats.stage() = Stage::GM_LOADGEO0_START;
                else              rPrepStats.stage() = Stage::GM_LOADGEO1_START;
                rPrepStats.loadGeometriesTotal(stageId) = mRunLoadGeometriesTotal[stageId];
            } else {
                if (stageId == 0) rPrepStats.stage() = Stage::GM_LOADGEO0_PROCESS;
                else              rPrepStats.stage() = Stage::GM_LOADGEO1_PROCESS;
                rPrepStats.loadGeometriesProcessed(stageId) = mRunLoadGeometriesProcessed[stageId];
            }
            break;
        }

        return flag;
    };

    auto finalizeChangeRenderPrepStats = [&](const int stageId,
                                             scene_rdl2::grid_util::RenderPrepStats &rPrepStats) -> bool {
        switch (mRunFinalizeChange[stageId]) {
        case Condition::INIT :
            return false; // not start finalizeChange yet

        //------------------------------
        // finalizeChange started
        case Condition::START_CANCELED :
            // early exit. finalizeChange stage has been canceled already at start timing
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_START_CANCELED;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_START_CANCELED;
            return true;
        case Condition::END :
            // early exit. finalizeChange stage has been completed already
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_DONE;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_DONE;
            return true;
        case Condition::END_CANCELED :
            // early exit. finalizeChange stage has been canceled already at end timing
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_DONE_CANCELED;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_DONE_CANCELED;
            return true;

        default : break;
        }

        //------------------------------
        // tessellation started
        switch (mRunTessellation[stageId]) {
        case Condition::INIT :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_START;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_START;
            return true;
        case Condition::START_CANCELED :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_TESSELLATION_CANCELED;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_TESSELLATION_CANCELED;
            rPrepStats.tessellationTotal(stageId) = mRunTessellationTotal[stageId];
            return true;
        case Condition::END_CANCELED :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_TESSELLATION_DONE_CANCELED;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_TESSELLATION_DONE_CANCELED;
            rPrepStats.tessellationProcessed(stageId) = mRunTessellationProcessed[stageId];
            return true;

        default : break;
        }
        if (mRunTessellation[stageId] != Condition::END) {
            // Timing is between starting tessellation and completing all tessellation
            if (mRunTessellationItem[stageId] == Condition::INIT) {
                // tessellate item computation is not started yet
                if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_TESSELLATION;
                else              rPrepStats.stage() = Stage::GM_FINALIZE1_TESSELLATION;
                rPrepStats.tessellationTotal(stageId) = mRunTessellationTotal[stageId];
            } else {
                // tessellation is on going
                if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_TESSELLATION_PROCESS;
                else              rPrepStats.stage() = Stage::GM_FINALIZE1_TESSELLATION_PROCESS;
                rPrepStats.tessellationProcessed(stageId) = mRunTessellationProcessed[stageId];
            }
            return true;
        }
        rPrepStats.tessellationProcessed(stageId) = mRunTessellationProcessed[stageId];

        //------------------------------
        // BVH construction started
        switch (mRunBVHConstruction[stageId]) {
        case Condition::INIT :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_TESSELLATION_DONE;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_TESSELLATION_DONE;
            break;
        case Condition::START :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_BVH;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_BVH;
            break;
        case Condition::START_CANCELED :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_BVH_CANCELED;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_BVH_CANCELED;
            break;
        case Condition::END_CANCELED :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_BVH_DONE_CANCELED;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_BVH_DONE_CANCELED;
            break;
        default :
            if (stageId == 0) rPrepStats.stage() = Stage::GM_FINALIZE0_BVH_DONE;
            else              rPrepStats.stage() = Stage::GM_FINALIZE1_BVH_DONE;
            break;
        }
        return true;
    };

    //
    // Special logic to setup renderPrepStats.
    // loadGeometries stage is executed first and after that finalizeChange stage is executed.
    // And inside each stage, stageId=0 is executed first and stageId=1 is next.
    // We evaluate loadGeometries stage if execution is before finalizeChange.
    // We setup renderPrepStats by stageId = 1 first,
    // We don't need to setup by stageId = 0 if stageId = 1 updates status.
    // If not, we should setup by stageId = 0 next.
    //
    scene_rdl2::grid_util::RenderPrepStats rPrepStats;
    if (isLoadGeometriesStage()) {
        if (!loadGeometriesRenderPrepStats(1, rPrepStats)) {
            loadGeometriesRenderPrepStats(0, rPrepStats);
        }
    } else {
        if (!finalizeChangeRenderPrepStats(1, rPrepStats)) {
            finalizeChangeRenderPrepStats(0, rPrepStats);
        }
    }

    return rPrepStats;
}

std::string
GeometryManagerExecTracker::showCancelCodePosIdList() const
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
GeometryManagerExecTracker::showCondition(const Condition &condition)
{
    switch (condition) {
    case Condition::INIT : return "INIT";
    case Condition::START : return "START";
    case Condition::START_CANCELED : return "START_CANCELED";
    case Condition::END : return "END";
    case Condition::END_CANCELED : return "END_CANCELED";
    case Condition::ETC : return "ETC";
    case Condition::ETC_CANCELED : return "ETC_CANCELED";
    default : return "?";
    }
}

// static function
std::string
GeometryManagerExecTracker::showCancelCodePos(const CancelCodePos &cancelCodePos)
{
    switch (cancelCodePos) {
    case CancelCodePos::EMPTY : return "EMPTY";

    case CancelCodePos::LOADGEOMETRIES_0_START : return "LOADGEOMETRIES_0_START";
    case CancelCodePos::LOADGEOMETRIES_ITEM_0_START : return "LOADGEOMETRIES_ITEM_0_START";
    case CancelCodePos::LOADGEOMETRIES_ITEM_0_END : return "LOADGEOMETRIES_ITEM_0_END";
    case CancelCodePos::LOADGEOMETRIES_0_END : return "LOADGEOMETRIES_0_END";
    case CancelCodePos::LOADGEOMETRIES_1_START : return "LOADGEOMETRIES_1_START";
    case CancelCodePos::LOADGEOMETRIES_ITEM_1_START : return "LOADGEOMETRIES_ITEM_1_START";
    case CancelCodePos::LOADGEOMETRIES_ITEM_1_END : return "LOADGEOMETRIES_ITEM_1_END";
    case CancelCodePos::LOADGEOMETRIES_1_END : return "LOADGEOMETRIES_1_END";

    case CancelCodePos::FINALIZE_CHANGE_0_START : return "FINALIZE_CHANGE_0_START";
    case CancelCodePos::TESSELLATION_0_START : return "TESSELLATION_0_START";
    case CancelCodePos::TESSELLATION_ITEM_0_START : return "TESSELLATION_ITEM_0_START";
    case CancelCodePos::TESSELLATION_ITEM_0_END : return "TESSELLATION_ITEM_0_END";
    case CancelCodePos::TESSELLATION_0_END : return "TESSELLATION_0_END";
    case CancelCodePos::BVH_CONSTRUCTION_0_START : return "BVH_CONSTRUCTION_0_START";
    case CancelCodePos::BVH_CONSTRUCTION_0_END : return "BVH_CONSTRUCTION_0_END";
    case CancelCodePos::FINALIZE_CHANGE_0_END : return "FINALIZE_CHANGE_0_END";
    case CancelCodePos::FINALIZE_CHANGE_1_START : return "FINALIZE_CHANGE_1_START";
    case CancelCodePos::TESSELLATION_1_START : return "TESSELLATION_1_START";
    case CancelCodePos::TESSELLATION_ITEM_1_START : return "TESSELLATION_ITEM_1_START";
    case CancelCodePos::TESSELLATION_ITEM_1_END : return "TESSELLATION_ITEM_1_END";
    case CancelCodePos::TESSELLATION_1_END : return "TESSELLATION_1_END";
    case CancelCodePos::BVH_CONSTRUCTION_1_START : return "BVH_CONSTRUCTION_1_START";
    case CancelCodePos::BVH_CONSTRUCTION_1_END : return "BVH_CONSTRUCTION_1_END";
    case CancelCodePos::FINALIZE_CHANGE_1_END : return "FINALIZE_CHANGE_1_END";

    case CancelCodePos::MAX : return "MAX";
    default : return "?";
    }
}

// static function
std::string
GeometryManagerExecTracker::showResult(const RESULT &result)
{
    switch (result) {
    case RESULT::CANCELED : return "CANCELED";
    case RESULT::FINISHED : return "FINISHED";
    default : return "?";
    }
}

} // namespace rt
} // namespace moonray

