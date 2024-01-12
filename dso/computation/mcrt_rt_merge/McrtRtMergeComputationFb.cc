// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "McrtRtMergeComputationFb.h"

#include <scene_rdl2/common/log/logging.h>

namespace moonray {
namespace mcrt_rt_merge_computation {

void    
RenderFb::setup(int numMachines)
{
    mNumMachines = numMachines;
    mUpstreamBuffers.resize(mNumMachines);
    mReceived.reset(new bool[mNumMachines]);
    mProgress.reset(new float[mNumMachines]);
    reset();
}

void    
RenderFb::reset()
{
    for (int i = 0; i < mNumMachines; ++i) {
        mReceived[i] = false;
        mProgress[i] = 0.0f;
    }
    mCompleteTotal = 0;
    mProgressTotal = 0.0f;
}

bool
RenderFb::storePartialFrame(const PartialFrame &partial)
{
    int machineId = partial.mMachineId;
    if (machineId < 0 || machineId >= mNumMachines) {
        return false;
    }

    for (const auto &buffer: partial.mBuffers) {
        // extract buffers by type
        if (!std::strcmp(buffer.mName, "beauty")) {
            auto &buf = mUpstreamBuffers[machineId];
            buf.resize(buffer.mDataLength);
            memcpy(&buf[0], buffer.mData.get(), buffer.mDataLength);
            // we always send status as STARTED.

        } else {
            // skip so far
        }
    }
    mReceived[machineId] = true;
    mProgress[machineId] = partial.getProgress();
    
    mCompleteTotal = 0;
    mProgressTotal = 0.0f;
    for (int i = 0; i < mNumMachines; ++i) {
        if (mReceived[i]) {
            mCompleteTotal++;
            mProgressTotal += mProgress[i];
        }
    }

    if (mCompleteTotal == mNumMachines) {
        return true;            // complete all data
    }
    return false;               // not yet or error
}    

float
RenderFb::unpackSparseTiles(const std::vector<std::vector<fb_util::Tile>> &tiles,
                            fb_util::VariablePixelBuffer &tiledFrame)
{
    // Copy each upstream buffer into the final frame buffer.
    for (size_t i = 0; i < mUpstreamBuffers.size(); ++i) {
        if (!tiles[i].empty() && mUpstreamBuffers[i].size() != 0) {
            tiledFrame.unpackSparseTiles(&mUpstreamBuffers[i][0], tiles[i]);
        }
    }

    return mProgressTotal;
}

void    
RenderFb::show() const
{
    std::ostringstream ostr;
    ostr << "  RenderFb {\n";
    {
        ostr << "      mNumMachines:" << mNumMachines << '\n';
        ostr << "         mReceived:";
        for (int i = 0; i < mNumMachines; ++i) {
            if (mReceived[i]) {
                ostr << "X";
            } else {
                ostr << ".";
            }
        }
        if (isComplete()) {
            ostr << " complete !\n";
        } else {
            ostr << "\n";
        }
        ostr << "    mCompleteTotal:" << mCompleteTotal << " progressTotal:" << mProgressTotal << '\n';
    }
    ostr << "  }" << std::endl;

    MOONRAY_LOG_INFO(ostr.str().c_str());
}

//------------------------------------------------------------------------------

RenderFbArray::RenderFbArray(int totalCacheFrame, int numMachines) :
    mTotalCacheFrame(totalCacheFrame),
    mNumMachines(numMachines),
    mFbArray(),
    mStartFrameId(-1),
    mEndFrameId(-1)
{
    mFbArray.reset(new RenderFb[totalCacheFrame]);
    mFbTbl.resize(totalCacheFrame);
}

bool
RenderFbArray::storePartialFrame(const PartialFrame &partial,
                                 const math::Viewport &vp)
{
    math::Viewport partialVp(partial.mHeader.mViewport.mMinX,
                             partial.mHeader.mViewport.mMinY,
                             partial.mHeader.mViewport.mMaxX,
                             partial.mHeader.mViewport.mMaxY);
    if (vp != partialVp) {
        /* Useful debug code
        std::cout << ">>>=== McrtRtMergeComputationFb.cc vp != partialVp" << std::endl;
        std::cout << ">>> partial"
                  << " mMinX:" << partial.mHeader.mViewport.mMinX
                  << " mMinY:" << partial.mHeader.mViewport.mMinY
                  << " mMaxX:" << partial.mHeader.mViewport.mMaxX
                  << " mMaxY:" << partial.mHeader.mViewport.mMaxY
                  << " vp"
                  << " minX:" << vp.mMinX
                  << " minY:" << vp.mMinY
                  << " maxX:" << vp.mMaxX
                  << " maxY:" << vp.mMaxY
                  << std::endl;
        */
        // Somehow partialFrame's viewport information is empty.
        // We skip this test for temporal quick fix reason. Toshi (22/Sep/16)
        // return false;           // invalid viewport data
    }

    int globalFrameId = partial.mHeader.mFrameId;

    {
        int machineId = partial.mMachineId;
        MOONRAY_LOG_INFO("Merge partialFrame machineId:%d globalFrameId:%d", machineId, globalFrameId );
    }

    if (mStartFrameId <= -1) {   // initialize table
        MOONRAY_LOG_INFO("Merge initialize multi-fb table");
        mStartFrameId = globalFrameId;
        mEndFrameId = mStartFrameId + mTotalCacheFrame - 1;
        for (int i = 0; i < mTotalCacheFrame; ++i) {
            mFbTbl[i] = &mFbArray[i];
            mFbTbl[i]->setup(mNumMachines);
        }
    }
    if (globalFrameId < mStartFrameId) {
        return false;           // too old data
    }

    int shiftCount = 0;                // for debug
    if (globalFrameId > mEndFrameId) { // we have to update table
        int shiftOffset = globalFrameId - mEndFrameId;
        for (int i = 0; i < shiftOffset; ++i) {
            shiftFbTbl();
            shiftCount++;       // for debug
        }
    }

    int localFrameId = getLocalFrameId(globalFrameId);
    RenderFb *cFb = mFbTbl[localFrameId];

    //------------------------------

    if (cFb->storePartialFrame(partial)) {
        return true;
    }

    return false;
}

void
RenderFbArray::shiftFbTbl()
{
    RenderFb *ptr = mFbTbl[0];

    {
        int cFrameId = mStartFrameId;
        if (!ptr->isComplete()) {
            MOONRAY_LOG_ERROR("drop uncomplete frame:%d", cFrameId);
        }
    }

    int lastId = mFbTbl.size() - 1;
    for (int i = 0; i < lastId; ++i) {
        mFbTbl[i] = mFbTbl[i+1];
    }
    ptr->reset();               // do reset for new recycled entry
    mFbTbl[lastId] = ptr;

    mStartFrameId++;
    mEndFrameId++;
}

bool
RenderFbArray::completeTestAndShift()
{
    int total = mFbTbl.size();
    int shiftTotal = -1;
    for (int i = 0; i < total; ++i) {
        RenderFb * cFb = mFbTbl[i];
        if (cFb->isComplete()) {
            shiftTotal = i;
            break;
        }
    }

    if (shiftTotal == -1) {
        // could not find completed renderFb
        return false;
    }

    for (int i = 0; i < shiftTotal; ++i) {
        shiftFbTbl();
    }

    return true;                // found completed renderFb and already shifted
}

void
RenderFbArray::show() const
{
    std::ostringstream ostr;
    ostr << "RenderFbArray {\n";
    {
        ostr << "  mTotalCacheFrame:" << mTotalCacheFrame << '\n';
        ostr << "      mNumMachines:" << mNumMachines << '\n';
        ostr << "     mStartFrameId:" << mStartFrameId << '\n';
        ostr << "       mEndFrameId:" << mEndFrameId << '\n';

        int fId = mStartFrameId;
        for (int i = 0; i < mTotalCacheFrame; ++i) {
            if (mFbTbl[i]->isActive()) {
                ostr << "  fId:" << fId << " ";
                mFbTbl[i]->show();
            }
            fId++;
        }
    }
    ostr << "}" << std::endl;

    MOONRAY_LOG_INFO(ostr.str().c_str());
}

    
} // namespace mcrt_rt_merge_computation
} // namespace moonray

