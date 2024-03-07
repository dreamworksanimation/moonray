// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once
#include <moonray/rendering/mcrt_common/ThreadLocalState.hh>

#define ACTIVE_ACC_STACK_SIZE   64

//
//  mTopLevelTls                        Backpointer to top level TLS.
//  mRayStatePool                       Pooled memory allocator.
//  mCL1Pool                            Single cache line allocator.
//  mRayQueue                           All local rays are sent through this queue.
//  mOcclusionQueue                     All local occlusion queries are sent through this queue.
//  mRadianceQueue                      All local radiance samples are sent through these queues.
//  mAovQueue
//  mHeatMapQueue
//  mXPUOcclusionRayQueue               Pointer to XPU occlusion ray queue (owned by the RenderDriver)
//  mXPURayQueue                        Pointer to XPU ray queue (owned by the RenderDriver)
//  mPrimaryRaysSubmitted               Primary rays submitted.
//  mFs                                 Constant for entire frame.
//  mTilesRenderedTo                    Tiles which have had any samples rendered to them for gui diagnostics purposes.
//  mCancellationState
//  mCurrentPassIdx                     The current pass we are rendering.
//  mStatistics
//  mRayRecorder                        Ray recording functionality.
//  mRayVertexStack
//  mRayEntries
//  mOcclusionEntries
//  mPresenceShadowsEntries
//  mRadianceEntries
//  mAovEntries
//  mHeatMapEntries
//
#define PBR_TL_STATE_MEMBERS                                                        \
    HUD_PTR(ThreadLocalState *, mTopLevelTls);                                      \
    HUD_CPP_PTR(ExclusiveAccumulators *, mExclusiveAccumulators);                   \
    HUD_CPP_MEMBER(RayStatePool, mRayStatePool, 96);                                \
    HUD_CPP_MEMBER(CL1Pool, mCL1Pool, 96);                                          \
    HUD_CPP_MEMBER(RayQueue, mRayQueue, 40);                                        \
    HUD_PRIVATE()                                                                   \
    HUD_CPP_MEMBER(OcclusionQueue, mOcclusionQueue, 40);                            \
    HUD_CPP_MEMBER(PresenceShadowsQueue, mPresenceShadowsQueue, 40);                \
    HUD_CPP_PTR(RadianceQueue *, mRadianceQueue);                                   \
    HUD_CPP_PTR(AovQueue *, mAovQueue);                                             \
    HUD_CPP_PTR(HeatMapQueue *, mHeatMapQueue);                                     \
    HUD_CPP_PTR(XPUOcclusionRayQueue *, mXPUOcclusionRayQueue);                     \
    HUD_CPP_PTR(XPURayQueue *, mXPURayQueue);                                       \
    HUD_PUBLIC()                                                                    \
    HUD_CPP_ARRAY(size_t, mPrimaryRaysSubmitted, MAX_RENDER_PASSES, 8000);          \
    HUD_PTR(const FrameState *, mFs);                                               \
    HUD_CPP_MEMBER(scene_rdl2::util::BitArray, mTilesRenderedTo, 16);               \
    HUD_MEMBER(CancellationState, mCancellationState);                              \
    HUD_MEMBER(uint32_t, mCurrentPassIdx);                                          \
    HUD_MEMBER(PbrStatistics, mStatistics);                                         \
    HUD_CPP_PTR(DebugRayRecorder *, mRayRecorder);                                  \
    HUD_CPP_MEMBER(std::vector<DebugRayVertex *>, mRayVertexStack, 24);             \
    HUD_PRIVATE()                                                                   \
    HUD_CPP_PTR(RayQueue::EntryType *, mRayEntries);                                \
    HUD_CPP_PTR(OcclusionQueue::EntryType *, mOcclusionEntries);                    \
    HUD_CPP_PTR(PresenceShadowsQueue::EntryType *, mPresenceShadowsEntries);        \
    HUD_CPP_PTR(RadianceQueue::EntryType *, mRadianceEntries);                      \
    HUD_CPP_PTR(AovQueue::EntryType *, mAovEntries);                                \
    HUD_CPP_PTR(HeatMapQueue::EntryType *, mHeatMapEntries);                        \
    HUD_ISPC_PAD(mPad, 40)


#define PBR_TL_STATE_VALIDATION                                 \
    HUD_BEGIN_VALIDATION(PbrTLState);                           \
    HUD_VALIDATE(PbrTLState, mTopLevelTls);                     \
    HUD_VALIDATE(PbrTLState, mExclusiveAccumulators);           \
    HUD_VALIDATE(PbrTLState, mRayStatePool);                    \
    HUD_VALIDATE(PbrTLState, mCL1Pool);                         \
    HUD_VALIDATE(PbrTLState, mRayQueue);                        \
    HUD_VALIDATE(PbrTLState, mOcclusionQueue);                  \
    HUD_VALIDATE(PbrTLState, mPresenceShadowsQueue);            \
    HUD_VALIDATE(PbrTLState, mRadianceQueue);                   \
    HUD_VALIDATE(PbrTLState, mAovQueue);                        \
    HUD_VALIDATE(PbrTLState, mHeatMapQueue);                    \
    HUD_VALIDATE(PbrTLState, mXPUOcclusionRayQueue);            \
    HUD_VALIDATE(PbrTLState, mXPURayQueue);                     \
    HUD_VALIDATE(PbrTLState, mPrimaryRaysSubmitted);            \
    HUD_VALIDATE(PbrTLState, mFs);                              \
    HUD_VALIDATE(PbrTLState, mTilesRenderedTo);                 \
    HUD_VALIDATE(PbrTLState, mCancellationState);               \
    HUD_VALIDATE(PbrTLState, mCurrentPassIdx);                  \
    HUD_VALIDATE(PbrTLState, mStatistics);                      \
    HUD_VALIDATE(PbrTLState, mRayRecorder);                     \
    HUD_VALIDATE(PbrTLState, mRayVertexStack);                  \
    HUD_VALIDATE(PbrTLState, mRayEntries);                      \
    HUD_VALIDATE(PbrTLState, mOcclusionEntries);                \
    HUD_VALIDATE(PbrTLState, mPresenceShadowsEntries);          \
    HUD_VALIDATE(PbrTLState, mRadianceEntries);                 \
    HUD_VALIDATE(PbrTLState, mAovEntries);                      \
    HUD_VALIDATE(PbrTLState, mHeatMapEntries);                  \
    HUD_END_VALIDATION

#define PBR_TL_STATE_NULL_HANDLE 0xffffffff

