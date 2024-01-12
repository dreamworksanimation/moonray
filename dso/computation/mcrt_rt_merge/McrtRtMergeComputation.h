// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "McrtRtMergeComputationFb.h"

#include <moonray/rendering/rndr/rndr.h>
#include <moonray/rendering/rndr/TileScheduler.h>

#include <moonray/engine/messages/base_frame/BaseFrame.h>
#include <engine/computation/Computation.h>
#include <scene_rdl2/common/math/Viewport.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moonray {

namespace network {
    class PartialFrame;
}

namespace rndr { class TileScheduler; }

namespace mcrt_rt_merge_computation {

class McrtRtMergeComputation : public moonray::engine::Computation
{
public:
    McrtRtMergeComputation();

protected:
    virtual void configure(const moonray::object::Object& aConfiguration);
    virtual void onIdle();
    virtual void onMessage(const moonray::network::Message::Ptr msg);

private:
    void onViewportChanged(const network::PartialFrame& msg);
    bool fpsIntervalPassed();
    void onIdle_mocap();        // onIdle function for mocap mode
    void onMessage_mocap(const moonray::network::Message::Ptr msg); // onMessage function for mocap mode

    util::Ref<alloc::ArenaBlockPool> mArenaBlockPool;
    alloc::Arena mArena;

    math::Viewport mViewport;

    std::unique_ptr<rndr::TileScheduler>  mTileScheduler;

    std::unique_ptr<RenderFbArray> mFbArray; // for mocap mode

    // Holds all the partial frame buffers we've received from the upstream
    // MCRT computations.
    std::vector<std::vector<uint8_t>> mUpstreamBuffers;
    std::vector<std::vector<fb_util::PixelInfoBuffer::PixelType>> mUpstreamPixelInfoBuffers;
    std::vector<std::vector<fb_util::Tile>> mTiles;

    // Type of frame buffers to encode.
    network::BaseFrame::ImageEncoding mImageEncoding;

    // Holds the final reassembled frame.
    fb_util::VariablePixelBuffer mTiledFrame;
    fb_util::VariablePixelBuffer mFinalFrame;

    fb_util::PixelInfoBuffer mPixelInfoTiledFrame;
    fb_util::PixelInfoBuffer mFinalPixelInfoFrame;

    /// Holds buffers for ROI
    fb_util::VariablePixelBuffer mRoiPixelBuffer;
    fb_util::PixelInfoBuffer mRoiPixelInfoBuffer;

    // The number of upstream MCRT computations.
    unsigned int mNumMachines;

    std::unique_ptr<network::BaseFrame::Status[]> mStatus;
    std::unique_ptr<float[]> mProgress;
    // Flag if we got a pixel info buffer
    bool mHasPixelInfo;
    bool mHasPartialFrame;

    network::BaseFrame::Viewport mRezedViewport;
    unsigned int mFrameWidth; // Full frame width
    unsigned int mFrameHeight; // Full frame height

    // flag use to determine if the first rendered frame has
    //  left the renderer to other downstream computations.
    //  this is used to signal that the time heavy on demand
    //  loading of assets has finished and the render has
    //  acutally begun rendering
    bool mFirstFrame;
    bool mNeedsStartedSent;
    double mLastTime;
    float mFps;

    /// flag to indicate that we have a region of interest viewport set
    bool mUsingROI;

    bool mMotionCaptureMode;
};

} // namespace mcrt_rt_merge_computation
} // namespace moonray

CREATOR_FUNC(moonray::mcrt_rt_merge_computation::McrtRtMergeComputation);

