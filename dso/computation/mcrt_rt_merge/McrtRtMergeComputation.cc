// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "McrtRtMergeComputation.h"

#include <moonray/rendering/rndr/rndr.h>

#include <moonray/client/protocol/viewport_message/ViewportMessage.h>
#include <moonray/common/log/logging.h>
#include <moonray/common/object/Object.h>
#include <moonray/engine/messages/base_frame/BaseFrame.h>
#include <moonray/engine/messages/generic_message/GenericMessage.h>
#include <moonray/engine/messages/partial_frame/PartialFrame.h>
#include <moonray/engine/messages/rendered_frame/RenderedFrame.h>
#include <engine/computation/Computation.h>
#include <scene_rdl2/common/fb_util/PixelBufferUtilsGamma8bit.h>
#include <scene_rdl2/common/fb_util/SparseTiledPixelBuffer.h>
#include <scene_rdl2/common/fb_util/Tiler.h>

#include <dwa/Assert.h>
#include <logging_base/macros.h>

#include <cstdlib>
#include <stdint.h>
#include <string>

using moonray::engine::Computation;
using moonray::network::BaseFrame;
using moonray::network::Message;
using moonray::network::PartialFrame;
using moonray::network::RenderedFrame;
using moonray::network::ViewportMessage;

//#define ALPHA_VIEW_TEST

namespace moonray {

namespace {

// TODO: This function is duplicated in McrtComputation.cc, where is a good
//       place to shared it?
inline fb_util::VariablePixelBuffer::Format
convertImageEncoding(BaseFrame::ImageEncoding encoding)
{
    switch (encoding)
    {
    case BaseFrame::ENCODING_RGBA8:     return fb_util::VariablePixelBuffer::RGBA8888;
    case BaseFrame::ENCODING_RGB888:    return fb_util::VariablePixelBuffer::RGB888;
    default:                                MNRY_ASSERT(0);
    }
    return fb_util::VariablePixelBuffer::UNINITIALIZED;
}

#ifdef ALPHA_VIEW_TEST
void tmpHack(fb_util::VariablePixelBuffer *frame)
{
    unsigned area = frame->getArea();
    uint8_t *src = frame->getData();
    uint8_t *dst = src;

    for (unsigned i = 0; i < area; ++i, dst += 4, src += 4) {
        dst[0] = dst[1] = dst[2] = src[3];
    }
}
#endif // end ALPHA_VIEW_TEST

const char* boolToString(bool v) {
    static const char* YES = "yes";
    static const char* NO = "no";
    return (v ? YES : NO);
}

}

namespace mcrt_rt_merge_computation {


McrtRtMergeComputation::McrtRtMergeComputation() :
    mArenaBlockPool(util::alignedMallocCtorArgs<alloc::ArenaBlockPool>(CACHE_LINE_SIZE)),
    mFbArray(),
    mUpstreamBuffers(),
    mUpstreamPixelInfoBuffers(),
    mImageEncoding(BaseFrame::ImageEncoding::ENCODING_RGB888),
    mNumMachines(0),
    mStatus(),
    mProgress(),
    mHasPixelInfo(false),
    mHasPartialFrame(false),
    mFrameWidth(0),
    mFrameHeight(0),
    mFirstFrame(true),
    mNeedsStartedSent(false),
    mLastTime(0.0),
    mFps(5.0f),
    mUsingROI(false),
    mMotionCaptureMode(false)
{
    mArena.init(mArenaBlockPool.get());
}

void
McrtRtMergeComputation::configure(const object::Object& aConfig)
{
    MOONRAY_LOG_INFO(">>> McrtRtMergeComputation <<< ...");

    mMotionCaptureMode = false;
    const std::string sApplicationMode = "applicationMode";
    if (!aConfig[sApplicationMode].isNull()) {
        if (aConfig[sApplicationMode].value().asInt() == 1) {
            mMotionCaptureMode = true;
        } else {
        	MOONRAY_LOG_ERROR("APPLICATION MODE SET TO UNDEFIND");
        }
    }

    if (!aConfig["numMachines"].isNull()) {
        mNumMachines = aConfig["numMachines"];
        MNRY_ASSERT_REQUIRE(mNumMachines > 0);
        mUpstreamBuffers.resize(mNumMachines);
        mUpstreamPixelInfoBuffers.resize(mNumMachines);
        mTiles.resize(mNumMachines);
        mStatus.reset(new BaseFrame::Status[mNumMachines]);
        mProgress.reset(new float[mNumMachines]);
        for (unsigned int i = 0; i < mNumMachines; ++i) {
            mStatus[i] = BaseFrame::ERROR;
            mProgress[i] = 0.0;
        }
        int totalCacheFrame = 6;
        mFbArray.reset(new RenderFbArray(totalCacheFrame, mNumMachines));
        
    } else {
        MNRY_ASSERT_REQUIRE(false,
            "numMachines is a required config setting for the mcrt_merge computation");
    }

    if (!aConfig["imageEncoding"].isNull()) {
        mImageEncoding = static_cast<BaseFrame::ImageEncoding>(static_cast<int>((aConfig["imageEncoding"])));
    }

    if (!aConfig["fps"].isNull()) {
        mFps = aConfig["fps"];
    }
}

bool
McrtRtMergeComputation::fpsIntervalPassed()
{
    // In a single machine setup, frame gating is handled here.
    double now = util::getSeconds();
    if (now - mLastTime < (1.0f / mFps)) {
        return false;
    }
    mLastTime = now;
    return true;
}

void
McrtRtMergeComputation::onIdle()
{
    // If we don't have any upstream buffers, don't bother.
    if (mMotionCaptureMode) {
        onIdle_mocap();
        return;
    }

    if (mNumMachines == 0 || mUpstreamBuffers.empty() || mFinalFrame.getArea() == 0 || !fpsIntervalPassed()) {
        //MOONRAY_LOG_DEBUG("Returning early: mNumMachines: %d | mUpstreamBuffers.empty(): %s | mFinalFrame.getArea(): %d | fpsIntervalPassed: %s", mNumMachines, boolToString(mUpstreamBuffers.empty()), mFinalFrame.getArea(), boolToString(fpsIntervalPassed()));
        //MOONRAY_LOG_DEBUG_STR("Returning early: mNumMachines:" << mNumMachines << " | mUpstreamBuffers.empty(): " << boolToString(mUpstreamBuffers.empty()) << " | mFinalFrame.getArea(): " << mFinalFrame.getArea() << " | fpsIntervalPassed: " << boolToString(fpsIntervalPassed()));
        return;
    }

    float progress = 0.f;
    bool hasUpstreamBuffer = false;
    for (unsigned int i = 0; i < mNumMachines; ++i) {
        // This makes sure we at least have some
        // part of the frame to show.
        if (mUpstreamBuffers[i].size() != 0) {
            hasUpstreamBuffer = true;
        }
        progress += mProgress[i];
    }
    
    if (!hasUpstreamBuffer) {
        MOONRAY_LOG_DEBUG("Returning early: hasUpstreamBuffer %s", boolToString(hasUpstreamBuffer));
        return;
    }
    progress = std::min(progress, 1.0f);

    BaseFrame::Status status = BaseFrame::FINISHED;
    for (unsigned int i = 0; i < mNumMachines; ++i) {
        // It seems like it is possible that by the time
        // the first frame is ready to be sent after applying
        // an update all the nodes could be on 
        // BaseFrame::RENDERING but it is
        // Important that at least one STARTED notice is sent
        // at the begining
        if (mNeedsStartedSent) {
            status = BaseFrame::STARTED;
            mNeedsStartedSent = false;
            break;
        }
        // If at least one machine is not finished make sure we don't report finished.
        if (mStatus[i] != BaseFrame::FINISHED) {
            status = mStatus[i];
        }
    }

     // If we haven't gotten any new frames just return
    if (!mHasPartialFrame) {
        return;
    }

    mHasPartialFrame = false;

    // Send it downstream.
    RenderedFrame::Ptr frameMsg(new RenderedFrame);

    frameMsg->mHeader.mRezedViewport = mRezedViewport;
    frameMsg->mHeader.mStatus = status;
    frameMsg->mHeader.mProgress = progress;

    // Copy each upstream buffer into the final frame buffer.
    for (size_t i = 0; i < mUpstreamBuffers.size(); ++i) {
        if (!mTiles[i].empty() && mUpstreamBuffers[i].size() != 0) {
            mTiledFrame.unpackSparseTiles(&mUpstreamBuffers[i][0], mTiles[i]);
        }
    }

    // Untile buffer.
    fb_util::Tiler tiler(mFinalFrame.getWidth(), mFinalFrame.getHeight());
    mFinalFrame.untile(mTiledFrame, tiler, true);

#   ifdef ALPHA_VIEW_TEST
    tmpHack(&mFinalFrame);      // test code to view alpha value
#   endif // end ALPHA_VIEW_TEST


    // Image buffer as unsigned char
    moonray::network::DataPtr buffer;

    // Size of image buffer in bytes
    size_t bufferLength;

    math::Viewport fullViewport(0, 0, mFrameWidth - 1, mFrameHeight - 1);

    // If we're using an ROI viewport, then copy the contents of the final frame into our smaller buffer
    if (mUsingROI) {
        frameMsg->mHeader.setViewport(mViewport.mMinX, mViewport.mMinY, mViewport.mMaxX, mViewport.mMaxY);
        MOONRAY_LOG_DEBUG("Using ROI Viewport: (%d, %d, %d, %d) (%d x %d)", mViewport.mMinX, mViewport.mMinY, mViewport.mMaxX, mViewport.mMaxY, mViewport.width(), mViewport.height());
            fb_util::copyRoiBuffer<uint8_t>
                       (mViewport, fullViewport, mFinalFrame.getSizeOfPixel(),
                        mRoiPixelBuffer.getData(),
                        mFinalFrame.getData(), bufferLength);
            buffer = mRoiPixelBuffer.getDataShared();
    } else { // Otherwise, use the full buffer
        frameMsg->mHeader.mViewport.reset();
        buffer = mFinalFrame.getDataShared();
        bufferLength = mFinalFrame.getArea() * mFinalFrame.getSizeOfPixel();
    }

    frameMsg->addBuffer(buffer, bufferLength, "beauty", mImageEncoding);

    if (mHasPixelInfo) {
        // Copy each upstream pixel buffer into the final pixel info buffer.
        for (size_t i = 0; i < mUpstreamPixelInfoBuffers.size(); ++i) {
            if (!mTiles[i].empty() && mUpstreamPixelInfoBuffers[i].size() != 0) {
                fb_util::unpackSparseTiles(&mPixelInfoTiledFrame, &mUpstreamPixelInfoBuffers[i][0], mTiles[i]);
            }
        }

        // Untile buffer.
        fb_util::Tiler tiler(mFinalPixelInfoFrame.getWidth(), mFinalPixelInfoFrame.getHeight());
        fb_util::untile(&mFinalPixelInfoFrame, mPixelInfoTiledFrame, tiler, true,
                        [](const fb_util::PixelInfoBuffer::PixelType &pixel, unsigned)
                        -> const fb_util::PixelInfoBuffer::PixelType & {
                            return pixel;
                        });


        // Handle ROI for the pixel buffer
        if (mUsingROI) {
            fb_util::copyRoiBuffer<float>
                        (mViewport, fullViewport, 1,
                         reinterpret_cast<float *>(mRoiPixelInfoBuffer.getData()),
                         reinterpret_cast<float *>(mFinalPixelInfoFrame.getData()), bufferLength);
            buffer = mRoiPixelInfoBuffer.getDataSharedAs<uint8_t>();
        } else {
            buffer = mFinalPixelInfoFrame.getDataSharedAs<uint8_t>();
            bufferLength = mFinalPixelInfoFrame.getArea() * sizeof(decltype(mFinalPixelInfoFrame)::PixelType); // 1 float per pixel
        }

        MOONRAY_LOG_DEBUG("Adding pixel info buffer");
        frameMsg->addBuffer(buffer, bufferLength, "depth", BaseFrame::ENCODING_FLOAT);
        mHasPixelInfo = false;
    }

    // TODO... At this point we have a full sized linear buffer containing the final frame.
    // If there is an active ROI, we'll need to extract the relevant pixels into the
    // smaller ROI buffer right here.

    MOONRAY_LOG_DEBUG("Sending frame!");
    send(frameMsg);

    if (mFirstFrame == true) {
        moonray::network::GenericMessage::Ptr firstFrameMsg(new moonray::network::GenericMessage);
        firstFrameMsg->mValue = "MCRT Rendered First Frame";
        send (firstFrameMsg);
        mFirstFrame = false;
        MOONRAY_LOG_INFO("McrtMerge Sent first frame message");
    }
}

void
McrtRtMergeComputation::onIdle_mocap()
{
    if (mNumMachines == 0 || !mFbArray || mFinalFrame.getArea() == 0) {
        return;
    }

    if (!mFbArray->isActive()) {
        return;
    }

    double now = util::getSeconds();
    if (now - mLastTime < (1.0f / 60.0f)) {
        return;
    }

    if (!mFbArray->completeTestAndShift()) {
        return;                 // could not find completed renderFb yet.
    }

    mLastTime = now;

    RenderFb * cRenderFb = mFbArray->getLocal(0); // get first frame data in the fbArray
    float progress = cRenderFb->unpackSparseTiles(mTiles, mTiledFrame);
    mFbArray->shiftFbTbl();     // shift one frame

    // basically, mocap case, we always report as FINISHED.
    /*
    BaseFrame::Status status = BaseFrame::FINISHED;
    if (mNeedsStartedSent) {
        // It is important that at least one STARTED notice is sent at the begining
        status = BaseFrame::STARTED;
        mNeedsStartedSent = false;
    }
    */
    BaseFrame::Status status = BaseFrame::STARTED;

    // Untile buffer.
    fb_util::Tiler tiler(mFinalFrame.getWidth(), mFinalFrame.getHeight());
    mFinalFrame.untile(mTiledFrame, tiler, true);

    // Send it downstream.
    RenderedFrame::Ptr frameMsg(new RenderedFrame);

    frameMsg->mHeader.mRezedViewport = mRezedViewport;
    frameMsg->mHeader.mStatus = status;
    frameMsg->mHeader.mProgress = progress;

    // Image buffer as unsigned char
    moonray::network::DataPtr buffer;

    // Size of image buffer in bytes
    size_t bufferLength;

    math::Viewport fullViewport(0, 0, mFrameWidth - 1, mFrameHeight - 1);

    // If we're using an ROI viewport, then copy the contents of the final frame into our smaller buffer
    if (mUsingROI) {
        frameMsg->mHeader.setViewport(mViewport.mMinX, mViewport.mMinY, mViewport.mMaxX, mViewport.mMaxY);
        MOONRAY_LOG_DEBUG("Using ROI Viewport: (%d, %d, %d, %d) (%d x %d)",
                        mViewport.mMinX, mViewport.mMinY, mViewport.mMaxX, mViewport.mMaxY,
                        mViewport.width(), mViewport.height());
        fb_util::copyRoiBuffer<uint8_t>(mViewport, fullViewport, mFinalFrame.getSizeOfPixel(),
                                     mRoiPixelBuffer.getData(),
                                     mFinalFrame.getData(), bufferLength);
        buffer = mRoiPixelBuffer.getDataShared();
    } else { // Otherwise, use the full buffer
        frameMsg->mHeader.mViewport.reset();
        buffer = mFinalFrame.getDataShared();
        bufferLength = mFinalFrame.getArea() * mFinalFrame.getSizeOfPixel();
    }

    frameMsg->addBuffer(buffer, bufferLength, "beauty", mImageEncoding);

    // Not support pixelInfo mode for mocap

    MOONRAY_LOG_DEBUG("Sending frame!");
    send(frameMsg);

    if(mFirstFrame == true ) {
        moonray::network::GenericMessage::Ptr firstFrameMsg(new moonray::network::GenericMessage);
        firstFrameMsg->mValue = "MCRT Rendered First Frame";
        send(firstFrameMsg);
        mFirstFrame = false;
        MOONRAY_LOG_INFO("McrtMerge Sent first frame message");
    }
}

void
McrtRtMergeComputation::onMessage(const Message::Ptr aMsg)
{
    if (mMotionCaptureMode) {
        onMessage_mocap(aMsg);
        return;
    }

    if (aMsg->id() == PartialFrame::ID) {
        PartialFrame& partial = static_cast<PartialFrame&>(*aMsg);
        onViewportChanged(partial);

        int machineId = partial.mMachineId;
        //MOONRAY_LOG_DEBUG("machineId: %s vs mNumMachines; %d", machineId, mNumMachines);
        MNRY_ASSERT_REQUIRE((unsigned)machineId < mNumMachines);

        for (const auto& buffer: partial.mBuffers) {
            // TODO: Extract buffers by type
            if (!std::strcmp(buffer.mName, "beauty")) {
                auto &buf = mUpstreamBuffers[machineId];
                MNRY_ASSERT(buffer.mDataLength);
                buf.resize(buffer.mDataLength);
                memcpy(&buf[0], buffer.mData.get(), buffer.mDataLength);
                if (partial.getStatus() == BaseFrame::STARTED) {
                    mNeedsStartedSent = true;
                }
                mHasPartialFrame = true;
                mStatus[machineId] = partial.getStatus();
                mProgress[machineId] = partial.getProgress();
            } else if (!std::strcmp(buffer.mName, "depth")) {
                auto &buf = mUpstreamPixelInfoBuffers[machineId];
                MNRY_ASSERT(buffer.mDataLength &&
                           (buffer.mDataLength % sizeof(fb_util::PixelInfoBuffer::PixelType) == 0));
                buf.resize(buffer.mDataLength);
                memcpy(&buf[0], buffer.mData.get(), buffer.mDataLength);
                mHasPixelInfo = true;
            }
        }
    }
}

void
McrtRtMergeComputation::onMessage_mocap(const Message::Ptr aMsg)
{
    MOONRAY_LOG_INFO("Merge received message: %s", aMsg->name());

    if (aMsg->id() == PartialFrame::ID) {
        PartialFrame& partial = static_cast<PartialFrame&>(*aMsg);
        onViewportChanged(partial);
        mFbArray->storePartialFrame(partial, mViewport);
        // mFbArray->show();
    }
}

void
McrtRtMergeComputation::onViewportChanged(const PartialFrame& msg)
{

    bool shouldReinit = false;
    const BaseFrame::Viewport& viewport(msg.getViewport());

    // Check if we're using an ROI, and/or if it's different than our current viewport
    math::Viewport roiViewport;
    if (viewport.hasViewport()) {
        roiViewport = math::Viewport(viewport.minX(), viewport.minY(), viewport.maxX(), viewport.maxY());
        if (roiViewport != mViewport) {
            shouldReinit = true;
        }
    // If we were using ROI, and we no longer have an input viewport, then we're cancelling ROI
    } else if (mUsingROI) {
        shouldReinit = true;
    }

    if (msg.getRezedViewport().height() != mFrameHeight || msg.getRezedViewport().width() != mFrameWidth) {
        shouldReinit = true;
    }

    // If we haven't changed our frame width or height, or we're not using a different ROI, early exit
    if (!shouldReinit) {
        return;
    }

    /* Useful debug code
    std::cout << ">>> McrtRtMergeComputation.cc onViewportChanged() viewport changed!" << std::endl;
    std::cout << ">>> McrtRtMergeComputation.cc"
              << " mFrameWidth:" << mFrameWidth << " -> " << msg.getRezedViewport().width()
              << " mFrameHeight:" << mFrameHeight << " -> " << msg.getRezedViewport().height()
              << " minX:" << viewport.minX()
              << " minY:" << viewport.minY()
              << " maxX:" << viewport.maxX()
              << " maxY:" << viewport.maxY()
              << std::endl;
    */

    mRezedViewport = msg.getRezedViewport();
    mFrameWidth = mRezedViewport.width();
    mFrameHeight = mRezedViewport.height();

    if (viewport.hasViewport()) {
        mViewport = roiViewport;
        mUsingROI = true;
    } else {
        mViewport = math::Viewport(0, 0, mFrameWidth - 1, mFrameHeight - 1);
        mUsingROI = false;
    }

    MOONRAY_LOG_DEBUG_STR("onViewportChanged (" << mFrameWidth << " x " <<  mFrameHeight << ")");

    MNRY_ASSERT(mUpstreamBuffers.size() == mNumMachines);

    // Reset upstream buffers.
    for (unsigned int i = 0; i < mNumMachines; ++i) {
        mUpstreamBuffers[i].clear();
        mUpstreamPixelInfoBuffers[i].clear();
    }

    // Create tile configurations for each upstream buffer.
    // 
    // TODO: The tiling mode here needs to be kept in sync with the tiling
    // mode which the partial frames were generated with. Ideally this
    // information would be transmitted in a message but for now it's
    // just hardcoded.
    auto tileSchedulerType = rndr::TileScheduler::SPIRAL_SQUARE;
    mTileScheduler = rndr::TileScheduler::create(tileSchedulerType);

    for (unsigned int i = 0; i < mNumMachines; ++i) {
        SCOPED_MEM(&mArena);
        mTiles[i].clear();
        if (mTileScheduler->generateTiles(&mArena, mFrameWidth, mFrameHeight, mViewport, i, mNumMachines)) {
            mTiles[i] = mTileScheduler->getTiles();
        }
    }

    // Init buffers.
    unsigned tiledWidth  = util::alignUp(mFrameWidth, COARSE_TILE_SIZE);
    unsigned tiledHeight = util::alignUp(mFrameHeight, COARSE_TILE_SIZE);

    auto format = convertImageEncoding(mImageEncoding);
    mTiledFrame.init(format, tiledWidth, tiledHeight);
    mFinalFrame.init(format, mFrameWidth, mFrameHeight);

    mPixelInfoTiledFrame.init(tiledWidth, tiledHeight);
    mFinalPixelInfoFrame.init(mFrameWidth, mFrameHeight);

    // Reinitialize our ROI buffers on any change
    if (mUsingROI) {
        mRoiPixelBuffer.init(convertImageEncoding(mImageEncoding), (unsigned)mViewport.width(), (unsigned)mViewport.height());
        mRoiPixelInfoBuffer.init((unsigned)mViewport.width(), (unsigned)mViewport.height());
    } else {
        // We're not using ROI at the moment, so free the buffers that may have been allocated
        mRoiPixelBuffer.cleanUp();
        mRoiPixelInfoBuffer.cleanUp();
    }

    mLastTime = util::getSeconds();
}

} // namespace mcrt_rt_merge_computation
} // namespace moonray

