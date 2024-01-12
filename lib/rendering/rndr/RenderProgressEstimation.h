// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <atomic>
#include <string>
#include <cstdint>

namespace moonray {
namespace rndr {

class RenderFrameTimingRecord;

class RenderProgressEstimation
//
// End time estimation logic
//
{
public:
    RenderProgressEstimation() :
        mFrameComplete(false),
        mAdaptiveSampling(false),
        mStartTileSamples(0),
        mFinalTileSamples(0),
        mRenderStartTime(0.0),
        mLastPassCompleteTime(0.0),
        mCompletedSamplesPerTile(0),
        mAveragedSampleCost(0.0),
        mEstimatedTimeOfCompletion(0.0),
        mStartUniformSamples(0),
        mSamplesTotal(0),
        mMode(ModeType::RENDERING),
        mLastMode(ModeType::RENDERING),
        mCallCounter(0),
        mCheckpointTileSamples(0)
    {}

    void setAdaptiveSampling(const bool adaptive) { mAdaptiveSampling = adaptive; }
    bool isAdaptiveSampling() const { return mAdaptiveSampling; }

    //------------------------------

    // We should call after very first estimation renderPasses() call
    void startEstimation(const RenderFrameTimingRecord &timingRec,
                         const unsigned startTileSamples,
                         const unsigned finalTileSamples);

    // We should call just after every pass is done in order to update pass info
    void updatePassInfo(const RenderFrameTimingRecord &timingRec);

    void setFrameComplete();

    //------------------------------

    void atomicAddSamples(const unsigned samples) { mSamplesTotal += samples; } // MTsafe
    unsigned getSamplesTotal() const { return mSamplesTotal; }

    //------------------------------

    void revertFilm(const bool sw) { mMode = (sw)? ModeType::REVERT_FILM: ModeType::RENDERING; }
    void checkpointOutput(const bool sw, unsigned tileSamples = 0) {
        if (sw) {
            mMode = ModeType::CHECKPOINT_OUTPUT;
            mCheckpointTileSamples = tileSamples;
        } else {
            mMode = ModeType::RENDERING;
        }
    }
    void signalInterruption() { mMode = ModeType::SIGNAL_INTERRUPTION; } // async-signal-safe function

    // only set start samples when resume uniform sample case.
    void setStartUniformSamples(const unsigned samples) { mStartUniformSamples = samples; }
    unsigned getStartUniformSamples() const { return mStartUniformSamples; }

    //------------------------------

    // Estimation APIs for end time.
    double getEstimatedSecOfCompletion() const { return mEstimatedTimeOfCompletion; } // from time of Epoch by sec
    double getEstimatedSecOfCompletionFromNow() const; // sec

    // Estimation API for percentage of progress based on processing time (not sampled total).
    double getEstimatedCompletedFractionOfNow() const; // return 0.0 ~ 1.0

    // Useful time related APIs
    double getElapsedRenderTime() const;
    double getWholeRenderTime() const { return mEstimatedTimeOfCompletion - mRenderStartTime; }

    std::string getModeBanner(); // called from single thread

    //------------------------------

    std::string show(const std::string &hd) const; // mainly used for debug

protected:
    enum class ModeType {
        RENDERING,              // rendering
        REVERT_FILM,            // reverting film object from file
        CHECKPOINT_OUTPUT,      // writing out checkpoint file
        SIGNAL_INTERRUPTION     // interrupt by signal
    };

    bool mFrameComplete;
    bool mAdaptiveSampling;

    unsigned mStartTileSamples;
    unsigned mFinalTileSamples;
    double mRenderStartTime;    // sec from EPIC

    double mLastPassCompleteTime;
    unsigned mCompletedSamplesPerTile;
    double mAveragedSampleCost;

    double mEstimatedTimeOfCompletion; // current best guess of end time (sec from EPIC)

    //------------------------------

    unsigned mStartUniformSamples; // total samples which already done at start render

    std::atomic<unsigned> mSamplesTotal; // current samples total

    //------------------------------

    ModeType mMode;

    ModeType mLastMode;         // last mode of getModeBunner() request
    size_t mCallCounter;        // used by getModeBunner() request
    unsigned mCheckpointTileSamples;
}; // class RenderProgressEstimation

} // namespace rndr
} // namespace moonray

