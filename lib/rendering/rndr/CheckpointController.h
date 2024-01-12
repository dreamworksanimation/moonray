// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/rec_time/RecTime.h>

#include <list>
#include <memory> // shared_ptr

namespace moonray {
namespace rndr {

class RenderContext;

class CheckpointRayCostEvalEvent
//
// This is a single sampling event information that is stored inside CheckpointRayCostEstimator.
//
{
public:
    CheckpointRayCostEvalEvent(float deltaSec, // time used for the sampling operation
                               unsigned deltaSampleStartId, // tile sampleId
                               unsigned deltaSampleEndId) : // tile sampleId
        mDeltaSec(deltaSec),
        mDeltaSampleStartId(deltaSampleStartId),
        mDeltaSampleEndId(deltaSampleEndId)
    {}

    float getDeltaSec() const { return mDeltaSec; }
    unsigned getDeltaSamples() const { return mDeltaSampleEndId - mDeltaSampleStartId; }
    
    std::string show() const;

private:
    float mDeltaSec; // time used for the sampling operation
    unsigned mDeltaSampleStartId; // tile sampleId
    unsigned mDeltaSampleEndId; // tile sampleId
};

class CheckpointRayCostEstimator
//
// This class calculates single averaged ray sample cost based on the multiple
// sampling events results in history.
// The sampling event is stored in the list. It is maintained as reasonable size automatically
// in order to avoid using a huge memory.
//
{
public:

    void reset() { mEventList.clear(); }

    void push(float deltaSec, // time used for the sampling operation
              unsigned deltaSampleStart, // tile sampleId
              unsigned deltaSampleEnd); // tile sampleId

    float estimateRayCost() const;

    std::string show() const;

private:
    using CheckpointRayCostEvalEventShPtr = std::shared_ptr<CheckpointRayCostEvalEvent>;

    bool oldestEventRemoveTest();
    unsigned calcDeltaSamplesTotal();

    std::list<CheckpointRayCostEvalEventShPtr> mEventList;
};

//------------------------------------------------------------------------------------------

class CheckpointSnapshotEstimator
//
// This class computes the best snapshot intervals based on the user-defined information
// and history of already executed snapshot costs.
// Snapshot costs are stored in the list. It is maintained as a reasonable size automatically
// in order to avoid using a huge memory.
//
{
public:
    CheckpointSnapshotEstimator() :
        mSnapshotIntervalSec(0.0f),
        mSnapshotOverheadThresholdFraction(0.0f),
        mEstimateIntervalSec(0.0f)
    {}

    void set(float snapshotIntervalMinute, float snapshotOverheadFraction);

    bool isActive() const;

    void reset();

    void pushSnapshotCost(float snapshotSec);

    float estimateSnapshotInterval(); // return sec

    std::string show() const;

private:
    float estimateSnapshotSec() const;

    //------------------------------
    //
    // User-defined parameters in order to control snapshot intervals.
    // We use mSnapshotOverheadThresholdFraction if mSnapshotIntervalSec is ZERO or negative.
    // This is a fraction. For example, If you set 0.01, this means snapshot interval is automatically
    // adjusted and snapshot cost overhead would be less than 1% of entire MCRT calculation.
    // We use mSnapshotIntervalSec if mSnapshotIntervalSec value is more than ZERO and ignore
    // mSnapshotOverheadThresholdFraction. This is basically used for debugging purposes.
    // We disable snapshot operation when both of the parameters are ZERO or negative.
    //
    float mSnapshotIntervalSec;
    float mSnapshotOverheadThresholdFraction;

    //------------------------------

    float mEstimateIntervalSec; // last estimated snapshot interval (sec)

    std::list<float> mEventList; // snapshot interval history : unit is sec
};

//------------------------------------------------------------------------------------------

class CheckpointController
//
// This class provides APIs for executes snapshots by some intervals automatically and creates snapshot data.
// Also provides regular file output that includes both checkpoint and non-checkpoint.
//
{
public:
    CheckpointController() :
        mRemainingIntervalSec(0.0f),
        mCurrDeltaSampleStartId(0),
        mCurrDeltaSampleEndId(0),
        mMaxDeltaSamples(0),
        mLastSnapshotIntervalSec(0.0f)
    {}        

    // See CheckpointSnapshotEstimator comments for more detail.
    void set(float snapshotIntervalMinute, float snapshotOverheadFraction);

    void reset();

    //------------------------------

    bool isMemorySnapshotActive() const; // We execute snapshot or not.

    // Returns current best guess of next end tile sampleId for micro checkpoint loop based on the history.
    unsigned estimateEndSampleId(const unsigned startSampleId, const unsigned endSampleId);

    void microStintStart();
    bool microStintEnd(); // returns true if we need memory snapshot, otherwise returns false

    float getLastSnapshotIntervalSec() const { return mLastSnapshotIntervalSec; } // for debug

    //------------------------------

    // Creates new ImageWriteCache for snapshot action and stores it properly.
    // No file output operation itself, just snapshot only.
    void snapshotOnly(RenderContext *renderContext,
                      const std::string &checkpointPostScript,
                      const unsigned endSampleId);

    // This is a standard image output action that includes both checkpoint and non-checkpoint situations. 
    void output(bool checkpointBgWrite,
                bool twoStageOutput,
                RenderContext *renderContext,
                const std::string &checkpointPostScript,
                const unsigned endSampleId);

private:
    void resetRemainingIntervalSec();

    void fileOutputMain(bool checkpointBgWrite,
                        bool twoStageOutput,
                        bool memorySnapshot,
                        RenderContext *renderContext,
                        const std::string &checkpointPostScript,
                        const unsigned endSampleId);

    //------------------------------

    float mRemainingIntervalSec;
    unsigned mCurrDeltaSampleStartId;
    unsigned mCurrDeltaSampleEndId;
    unsigned mMaxDeltaSamples; // maximum delta samples of micro checkpoint loop from the frame start.

    CheckpointRayCostEstimator mRayCostEstimator;
    CheckpointSnapshotEstimator mSnapshotEstimator;

    scene_rdl2::rec_time::RecTime mRayCostEvalTime;

    float mLastSnapshotIntervalSec; // for debug
    scene_rdl2::rec_time::RecTime mSnapshotIntervalTime; // for debug
};

} // namespace rndr
} // namespace moonray

