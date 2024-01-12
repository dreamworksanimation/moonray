// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/grid_util/Parser.h>

#include <atomic>
#include <functional>
#include <limits>

namespace scene_rdl2 {

namespace grid_util {
    class Arg;
    class RenderPrepStats;
}
}

namespace moonray {
namespace rt {

class GeometryManagerExecTracker
//
// This class is used to track down GeometryManager execution status.
// RenderprepStatsCallBack is a call back function to send stats to the downstream computation.
// RenderprepStatsCallBack is properly called from each {start,end}* function.
// We can set nullptr to RenderPrepStatsCallBack and disable report stats logic when we are not under
// the arras context.
//
{
public:
    using RenderPrepStatsCallBack = std::function<void(const scene_rdl2::grid_util::RenderPrepStats &rPrepStats)>;
    using RenderPrepCancelCallBack = std::function<bool()>;
    using MsgHandlerCallBack = std::function<void(const std::string &msg)>;
    using Parser = scene_rdl2::grid_util::Parser;
    using Arg = scene_rdl2::grid_util::Arg;

    enum class RESULT // function execution result
    {
        CANCELED, // canceled middle of the function
        FINISHED  // function has been completed
    };

    GeometryManagerExecTracker() :
        mRenderPrepStatsCallBack(nullptr),
        mRenderPrepCancelCallBack(nullptr),
        mMsgHandlerCallBack(nullptr),
        mStageId(0),
        mRunLoadGeometries{Condition::INIT, Condition::INIT},
        mRunLoadGeometriesTotal{0, 0},
        mRunFinalizeChange{Condition::INIT, Condition::INIT},
        mRunTessellation{Condition::INIT, Condition::INIT},
        mRunTessellationTotal{0, 0},
        mRunBVHConstruction{Condition::INIT, Condition::INIT},
        mCancelCodePos(CancelCodePos::EMPTY),
        mCancelCodePosLoadGeomCounter(std::numeric_limits<int>::max()),
        mCancelCodePosTessellationCounter(std::numeric_limits<int>::max())
    {
        // The following parameters need initialize here in order to avoid GCC compile error.
        for (int i = 0; i < mStageMax; ++i) {
            mRunLoadGeometriesItem[i] = Condition::INIT;
            mRunLoadGeometriesProcessed[i] = 0;
            mRunTessellationItem[i] = Condition::INIT;
            mRunTessellationProcessed[i] = 0;
        }
    }

    GeometryManagerExecTracker(const GeometryManagerExecTracker &src) : // copy constructor
        mRenderPrepStatsCallBack(src.mRenderPrepStatsCallBack),
        mRenderPrepCancelCallBack(src.mRenderPrepCancelCallBack),
        mMsgHandlerCallBack(src.mMsgHandlerCallBack),
        mStageId(src.mStageId),
        mRunLoadGeometries{src.mRunLoadGeometries[0], src.mRunLoadGeometries[0]},
        mRunLoadGeometriesTotal{src.mRunLoadGeometriesTotal[0], src.mRunLoadGeometriesTotal[1]},
        mRunFinalizeChange{src.mRunFinalizeChange[0], src.mRunFinalizeChange[1]},
        mRunTessellation{src.mRunTessellation[0], src.mRunTessellation[1]},
        mRunTessellationTotal{src.mRunTessellationTotal[0], src.mRunTessellationTotal[1]},
        mRunBVHConstruction{src.mRunBVHConstruction[0], src.mRunBVHConstruction[1]},
        mCancelCodePos(src.mCancelCodePos),
        mCancelCodePosLoadGeomCounter(src.mCancelCodePosLoadGeomCounter),
        mCancelCodePosTessellationCounter(src.mCancelCodePosTessellationCounter)
    {
        // The following parameters need initialize here in order to avoid GCC compile error.
        for (int i = 0; i < mStageMax; ++i) {
            Condition condition;
            int processed;
            condition = src.mRunLoadGeometriesItem[i]; mRunLoadGeometriesItem[i] = condition;
            processed = src.mRunLoadGeometriesProcessed[i]; mRunLoadGeometriesProcessed[i] = processed;
            condition = src.mRunTessellationItem[i]; mRunTessellationItem[i] = condition;
            processed = src.mRunTessellationProcessed[i]; mRunTessellationProcessed[i] = processed;
        }

        // parserConfigure() is only executed inside copy constructor so far.
        // This is very dependent on the logic of how GeometryManager is initialized.
        // We always use copy constructed version of GeometryManagerExecTracker instead of the
        // very 1st initialized one.
        parserConfigure();
    }

    void initLoadGeometries(int stageId);
    void initFinalizeChange(int stageId);

    void setRenderPrepStatsCallBack(const RenderPrepStatsCallBack &callBack);
    void setRenderPrepCancelCallBack(const RenderPrepCancelCallBack &callBack);
    void setMsgHandlerCallBack(const MsgHandlerCallBack &callBack);
    MsgHandlerCallBack getMsgHandlerCallBack() const { return mMsgHandlerCallBack; }

    RESULT startLoadGeometries(int totalGeometries);
    RESULT startLoadGeometriesItem(); // called from multi-threaded function
    RESULT endLoadGeometriesItem(); // called from multi-threaded function
    void   finalizeLoadGeometriesItem(bool canceled);
    RESULT endLoadGeometries();

    RESULT startFinalizeChange();

    // Probably some detailed progress update API will be added between startBVHConstruction
    // and endBVHConstruction in the future in order to capture more detailed progress updates.
    RESULT startTessellation(int totalTessellation);
    RESULT startTessellationItem(); // called from multi-threaded function
    RESULT endTessellationItem(); // called from multi-threaded function
    void   finalizeTessellationItem(bool canceled);
    RESULT endTessellation();
    RESULT startBVHConstruction();
    RESULT endBVHConstruction();

    RESULT endFinalizeChange();

    scene_rdl2::grid_util::Parser& getParser() { return mParser; }

    std::string cancelInfoEncode() const;
    void cancelInfoDecode(const std::string &data);

    std::string show() const;

private:
    enum class Condition : int { INIT, START, START_CANCELED, END, END_CANCELED, ETC, ETC_CANCELED };
    enum class CancelCodePos : int {
        EMPTY = 0,

        // load geometries stage 
        LOADGEOMETRIES_0_START,
        LOADGEOMETRIES_ITEM_0_START,
        LOADGEOMETRIES_ITEM_0_END,
        LOADGEOMETRIES_0_END,
        LOADGEOMETRIES_1_START,
        LOADGEOMETRIES_ITEM_1_START,
        LOADGEOMETRIES_ITEM_1_END,
        LOADGEOMETRIES_1_END,

        // finalizeChange stage
        FINALIZE_CHANGE_0_START,
        TESSELLATION_0_START,
        TESSELLATION_ITEM_0_START,
        TESSELLATION_ITEM_0_END,
        TESSELLATION_0_END,
        BVH_CONSTRUCTION_0_START,
        BVH_CONSTRUCTION_0_END,
        FINALIZE_CHANGE_0_END,
        FINALIZE_CHANGE_1_START,
        TESSELLATION_1_START,
        TESSELLATION_ITEM_1_START,
        TESSELLATION_ITEM_1_END,
        TESSELLATION_1_END,
        BVH_CONSTRUCTION_1_START,
        BVH_CONSTRUCTION_1_END,
        FINALIZE_CHANGE_1_END,

        MAX
    };

    void parserConfigure();

    template <typename T0, typename T1>
    RESULT updateRunStatus(const CancelCodePos &callerCodePos0,
                           const CancelCodePos &callerCodePos1,
                           T0 &updateTarget,
                           const T1 &finishCondition,
                           const T1 &cancelCondition) const {
        const CancelCodePos &callerCodePos =
            (mStageId == 0) ? callerCodePos0 : callerCodePos1;

        RESULT result = checkRunStatus(callerCodePos);
        if (result == RESULT::FINISHED) {
            updateTarget = finishCondition;
        } else {
            updateTarget = cancelCondition;
        }

        renderPrepStatsUpdate();

        if (mMsgHandlerCallBack && result == RESULT::CANCELED) {
            updateRunStatusCancelMessage(callerCodePos, result);
        }

        return result;
    }
    void updateRunStatusCancelMessage(const CancelCodePos &callerCodePos, RESULT result) const;

    RESULT checkRunStatus(const CancelCodePos &callerCodePos) const;
    void renderPrepStatsUpdate() const;
    scene_rdl2::grid_util::RenderPrepStats calcRenderPrepStats() const;

    std::string showCancelCodePosIdList() const;

    static std::string showCondition(const Condition &condition);
    static std::string showCancelCodePos(const CancelCodePos &cancelCodePos);
    static std::string showResult(const RESULT &result);

    //------------------------------

    RenderPrepStatsCallBack mRenderPrepStatsCallBack;
    RenderPrepCancelCallBack mRenderPrepCancelCallBack;
    MsgHandlerCallBack mMsgHandlerCallBack;

    int mStageId;
    static constexpr int mStageMax = 2;

    // loadGeometries top level execution condition
    Condition mRunLoadGeometries[mStageMax];
    int mRunLoadGeometriesTotal[mStageMax];

    // internal of loadGeometries stage condition
    std::atomic<Condition> mRunLoadGeometriesItem[mStageMax];
    std::atomic<int> mRunLoadGeometriesProcessed[mStageMax];

    // finalizeChange top level execution condition
    Condition mRunFinalizeChange[mStageMax];

    // internal of finalizeChange stage condition for tessellation
    Condition mRunTessellation[mStageMax];
    int mRunTessellationTotal[mStageMax];
    std::atomic<Condition> mRunTessellationItem[mStageMax];
    std::atomic<int> mRunTessellationProcessed[mStageMax];

    // internal of finalizeChange stage condition for BVH construction
    Condition mRunBVHConstruction[mStageMax];

    //------------------------------

    Parser mParser;
    CancelCodePos mCancelCodePos;
    int mCancelCodePosLoadGeomCounter;
    int mCancelCodePosTessellationCounter;
};

} // namespace rt
} // namespace moonray

