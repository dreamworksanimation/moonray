// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <moonray/rendering/rndr/rndr.h>

#include <moonray/engine/messages/partial_frame/PartialFrame.h>

#include <vector>

using moonray::network::Message;
using moonray::network::PartialFrame;

namespace moonray {
namespace mcrt_rt_merge_computation {

//
// Frame buffer which containes entire image. This is one frame of rendered image.
// Under distributed MCRT situation, entire image is split into multiple PartialFrames.
// This RenderFb is used for re-comstruct perticular frame from multiple PartialFrame messages which
// are sent from upstream MCRT computations.
// Internally keeps track of PartialFrame receive condition from different upstream MCRT computations.
//
class RenderFb
{
public:
 RenderFb() : mNumMachines(0), mReceived(), mProgress(), mCompleteTotal(0), mProgressTotal(0.0f) {}
    ~RenderFb() {}

    void setup(int numMachines);

    void reset();

    bool storePartialFrame(const PartialFrame &partialFrame); // return complete condition
    bool isActive() const { return (mCompleteTotal > 0)? true: false; }
    bool isComplete() const { return (mCompleteTotal == mNumMachines)? true: false; }
    
    // return complete percentage (0.0~1.0)
    float unpackSparseTiles(const std::vector<std::vector<fb_util::Tile>> &tiles, fb_util::VariablePixelBuffer &tiledFrame);

    void show() const;

protected:
    int mNumMachines;           // number of upstream MCRT computation

    std::vector<std::vector<uint8_t>> mUpstreamBuffers;
    std::unique_ptr<bool []> mReceived;
    std::unique_ptr<float []> mProgress;

    int mCompleteTotal;
    float mProgressTotal;
};

//
// Multiple frames of RenderFb.
//
// We need some number (like ~ 10frames) of multiple frames RenderFb array in order to manage
// re-construct render frame more robustly.
// In general, under high number of MCRT computation case, It's pretty difficult to sync all
// MCRT computations are working on same frame. Sometimes Merger might have possibility to received
// different frame from different upstream MCRT computations. This means Merger need to be smart enough
// to understand frameId difference to re-construct frame. This is the main purpose of RenderFbArray.
// 
// RenderFbArray keeps multiple frames which defined by startFrameId and endFrameId.
// Capacity of array is totalCacheFrame = endFrameId - startFrameId + 1.
// startFrameId/endFrameId information is updated by shiftFbTbl() API. one shiftFbTbl() call increments
// both of mStartFrameId and mEndFrameId. In this case oldest frameFb data is destructed and recycled
// by new frame.
// 
class RenderFbArray
{
public:
    RenderFbArray(int totalCacheFrame, int numMachines);

    // re-construct image from PartialFrame Message.
    // return true when all frames are received and frame is completed.
    // return false when still some partialFrame is not ready yet.
    bool storePartialFrame(const PartialFrame &partial, const math::Viewport &vp); // return complete frame condition

    // Shift internal frame buffer table and dispose oldest fb data and create empty new fb
    // Both mStartFrameId and mEndFrameId are incremented.
    void shiftFbTbl();

    // Test fbArray is already accessed or not.
    bool isActive() { return (mStartFrameId >= 0)? true: false; }

    // Try to find complete frame and set first complete frame as startFrame and return true
    // If can not find complete frame, return false
    bool completeTestAndShift();
    bool isCompleteLocal(int localFrameId) { return mFbTbl[localFrameId]->isComplete(); }
    bool isComplete(int globalFrameId) { return getRenderFb(globalFrameId)->isComplete(); }

    RenderFb *getLocal(int localFrameId) { return mFbTbl[localFrameId]; }

    int getStartFrameId() const { return mStartFrameId; }

    void show() const;

protected:
    int getLocalFrameId(int globalFrameId) { return globalFrameId - mStartFrameId; }
    RenderFb *getRenderFb(int globalFrameId) { return mFbTbl[getLocalFrameId(globalFrameId)]; }

    //------------------------------

    int mTotalCacheFrame;
    int mNumMachines;

    std::unique_ptr<RenderFb []> mFbArray;

    int mStartFrameId;          // global frameId
    int mEndFrameId;            // global frameId
    std::vector<RenderFb *> mFbTbl;
};

} // namespace mcrt_rt_merge_computation
} // namespace moonray


