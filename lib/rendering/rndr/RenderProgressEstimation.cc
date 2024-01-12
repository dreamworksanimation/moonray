// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RenderProgressEstimation.h"
#include "RenderTimingRecord.h"

#include <scene_rdl2/common/platform/Platform.h> // scene_rdl2::util::getSeconds()

namespace moonray {
namespace rndr {

void    
RenderProgressEstimation::startEstimation(const RenderFrameTimingRecord &timingRec,
                                          const unsigned startTileSamples,
                                          const unsigned finalTileSamples)
{
    mFrameComplete = false;

    mStartTileSamples = startTileSamples;
    mFinalTileSamples = finalTileSamples;
    mRenderStartTime = timingRec.getRenderFrameStartTime();
}

void    
RenderProgressEstimation::updatePassInfo(const RenderFrameTimingRecord &timingRec)
{
    mLastPassCompleteTime = scene_rdl2::util::getSeconds();
    mCompletedSamplesPerTile = timingRec.getNumSamplesPerTile();
    mAveragedSampleCost = timingRec.getAveragedSampleCost();

    unsigned remainedSamplesPerTile = 0;
    if (mFinalTileSamples > mCompletedSamplesPerTile) {
        remainedSamplesPerTile = mFinalTileSamples - mCompletedSamplesPerTile - mStartTileSamples;
    }

    mEstimatedTimeOfCompletion = remainedSamplesPerTile * mAveragedSampleCost + mLastPassCompleteTime; // current best guess
}

void
RenderProgressEstimation::setFrameComplete()
{
    mFrameComplete = true;
}

double
RenderProgressEstimation::getEstimatedSecOfCompletionFromNow() const
{
    if (mFrameComplete) return 0.0;

    return std::max(mEstimatedTimeOfCompletion - scene_rdl2::util::getSeconds(), 0.0);
}

double
RenderProgressEstimation::getEstimatedCompletedFractionOfNow() const // return 0.0 ~ 1.0
{
    if (mFrameComplete) return 1.0;
    if (getWholeRenderTime() <= 0.0) return 0.0; // not finished estimation stage yet. (We don't have estimation info yet).
    return getElapsedRenderTime() / getWholeRenderTime();   // compute fraction based on completion render time estimation
}

double
RenderProgressEstimation::getElapsedRenderTime() const
{
    if (mRenderStartTime <= 0.0) return 0.0; // not start rendering yet.
    return scene_rdl2::util::getSeconds() - mRenderStartTime;
}

std::string
RenderProgressEstimation::getModeBanner()
{
    static constexpr char msgRen[] = "Rendering";
    static constexpr int  msgLen = sizeof(msgRen) - 1;

    std::string banner;
    if (mMode == ModeType::RENDERING) {
        banner = msgRen;
    } else {
        mCallCounter = (mLastMode != mMode)? 0: mCallCounter + 1;
        std::string work = std::string(msgLen, ' ');
        switch (mMode) {
        case ModeType::REVERT_FILM:
            work += "Reading resume file ...";
            break;
        case ModeType::CHECKPOINT_OUTPUT:
            work += "Checkpoint out " + std::to_string(mCheckpointTileSamples) + " tileSamples ...";
            break;
        case ModeType::SIGNAL_INTERRUPTION:
            work += "Interrupt by SIGNAL ...";
            break;
        default : break; // never happened
        }
        work += std::string(msgLen, ' ');
        banner = work.substr(mCallCounter % (work.size() - msgLen), msgLen);
    }
    mLastMode = mMode;
    
    return banner;
}

std::string
RenderProgressEstimation::show(const std::string &hd) const
{
    double currTime = scene_rdl2::util::getSeconds();

    std::ostringstream ostr;
    ostr << hd << "RenderProgressEstimation {\n"; {
        ostr << hd << "  mFrameComplete:" << mFrameComplete << '\n';
        ostr << hd << "  mFinalTileSamples:" << mFinalTileSamples << '\n';
        ostr << hd << "  mRenderStartTime:" << mRenderStartTime << " sec from EPIC "
             << mRenderStartTime - currTime << " sec\n";
        ostr << hd << "  mLastPassCompleteTime:" << mLastPassCompleteTime << " sec "
             << mLastPassCompleteTime - currTime << " sec\n";
        ostr << hd << "  mCompletedSamplesPerTile:" << mCompletedSamplesPerTile << '\n';
        ostr << hd << "  mAveragedSampleCost:" << mAveragedSampleCost << '\n';
        ostr << hd << "  mEstimatedTimeOfCompletion:" << mEstimatedTimeOfCompletion << " sec "
             << mEstimatedTimeOfCompletion - currTime << " sec\n";

        ostr << hd << "  getEstimatedSecOfCompletionFromNow():" << getEstimatedSecOfCompletionFromNow() << " sec\n";
        ostr << hd << "  getEstimatedCompletedFractionOfNow():" << getEstimatedCompletedFractionOfNow() << '\n';
    }
    ostr << hd << "}";
    return ostr.str();
}

} // namespace rndr
} // namespace moonray

