// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Types.h"

#include <moonray/rendering/pbr/Types.h>
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/math/Viewport.h>

#include <vector>

namespace moonray {
namespace rndr {

class RenderContext;
class RenderOutputDriver;

// "Conditioned" scene variables and other state which is constant over the
// render frame.

// for warning 444: destructor for base class "moonray::pbr::FrameState"
// (declared at line 104 of "moonray/lib/rendering/pbr/Types.h") is not virtual
#pragma warning(push)
#pragma warning(disable:444)

struct FrameState : public pbr::FrameState
{
    unsigned                mNumRenderThreads;
    unsigned                mNumRenderNodes;
    unsigned                mRenderNodeIdx;
    unsigned                mTileSchedulerType; // TileScheduler::Type
    unsigned                mTaskDistributionType; // Film::TaskDistribType

    RenderMode              mRenderMode;
    FastRenderMode          mFastMode;
    bool                    mRequiresDeepBuffer;
    bool                    mRequiresCryptomatteBuffer;
    bool                    mGeneratePixelInfo;
    unsigned                mWidth;
    unsigned                mHeight;
    scene_rdl2::math::Viewport mViewport;

    SamplingMode            mSamplingMode;
    unsigned                mMinSamplesPerPixel;
    unsigned                mMaxSamplesPerPixel;
    float                   mTargetAdaptiveError;

    // This only exists for backward compatibility in the cases where a pixel
    // sample map contains values above 1. It would be nice to disallow that
    // functionality and remove this member.
    unsigned                mOriginalSamplesPerPixel;

    bool                    mDofEnabled;
    const pbr::PixelFilter *mPixelFilter;

    // This functionality is mutually exclusive with adaptive sampling.
    scene_rdl2::fb_util::PixelBuffer<float> *mPixelSampleMap;

    int                     mDeepFormat;
    float                   mDeepCurvatureTolerance;
    float                   mDeepZTolerance;
    uint                    mDeepVolCompressionRes;
    std::vector<std::string> *mDeepIDChannelNames;
    int                     mDeepMaxLayers;
    float                   mDeepLayerBias;

    float mFps; // The desired frames per second for RENDER_MODE_PROGRESSIVE and RENDER_MODE_REALTIME modes.

    double                  mFrameStartTime;

    RenderContext          *mRenderContext;

    float mCheckpointInterval;  // minute
    // pixel sample delta count for UNIFORM sampling.
    // pixel sample delta steps on adaptive pixel sampling ID table for ADAPTIVE sampling.
    unsigned mCheckpointQualitySteps;
    float mCheckpointTimeCap;   // minute
    unsigned mCheckpointSampleCap; // sample count
    CheckpointMode mCheckpointMode;
    unsigned mCheckpointStartSPP; // start pixel sample count for checkpoint dump
    bool mCheckpointBgWrite;

    bool mTwoStageOutput;

    unsigned mDisplayFilterCount;
};

#pragma warning(pop)

}
}

