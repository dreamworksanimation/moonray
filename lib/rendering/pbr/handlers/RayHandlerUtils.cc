// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "RayHandlerUtils.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>


namespace moonray {
namespace pbr {

void
accumLightAovs(pbr::TLState* pbrTls, const BundledOcclRay& occlRay, const FrameState& fs,
               int numItems, const scene_rdl2::math::Color& matchMultiplier,
               const scene_rdl2::math::Color* nonMatchMultiplier,
               int flags)
{
    if (!fs.mLightAovs->hasEntries()) {
        return;
    }

    for (int i = 0; i < numItems; ++i) {
        const BundledOcclRayData *b =
            static_cast<BundledOcclRayData *>(pbrTls->getListItem(occlRay.mDataPtrHandle, i));
        if (b->mFlags & BundledOcclRayDataFlags::LPE) {
            if (!(b->mLpeStateId)) break; // no more valid items in list

            MNRY_ASSERT(!fs.mAovSchema->empty());
            int lpeStateId = b->mLpeStateId;
            MNRY_ASSERT(lpeStateId > 0);
            if (lpeStateId > 0) {

                scene_rdl2::math::Color nonMatchValue;
                if (nonMatchMultiplier) {
                    nonMatchValue = b->mLpeRadiance * (*nonMatchMultiplier);
                }
                const scene_rdl2::math::Color *pNonMatchValue = nonMatchMultiplier ? &nonMatchValue : nullptr;

                aovAccumLightAovsBundled(pbrTls, *fs.mAovSchema, b->mCameraId,
                                         *fs.mLightAovs, b->mLpeRadiance * matchMultiplier,
                                         pNonMatchValue, flags, lpeStateId, occlRay.mPixel, occlRay.mDeepDataHandle,
                                         pbr::getFilm(occlRay.mTilePassAndFilm));
            }
        }
    }
}

void
accumVisibilityAovs(pbr::TLState* pbrTls, const BundledOcclRay& occlRay,
                    const FrameState& fs, const int numItems, float value)
{
    if (!fs.mLightAovs->hasVisibilityEntries()) {
        return;
    }

    for (int i = 0; i < numItems; ++i) {
        const BundledOcclRayData *b =
            static_cast<BundledOcclRayData *>(pbrTls->getListItem(occlRay.mDataPtrHandle, i));
        if ((b->mFlags & BundledOcclRayDataFlags::LPE) &&
            (b->mFlags & BundledOcclRayDataFlags::LIGHT_SAMPLE)) {
            if (!(b->mLpeStateId)) break; // no more valid items in list

            MNRY_ASSERT(!fs.mAovSchema->empty());
            const int lpeStateId = b->mLpeStateId;
            MNRY_ASSERT(lpeStateId > 0);
            if (lpeStateId > 0) {
                if (aovAccumVisibilityAovsBundled(pbrTls, *fs.mAovSchema, b->mCameraId, *fs.mLightAovs,
                    scene_rdl2::math::Vec2f(value, 1.0f), lpeStateId, occlRay.mPixel, occlRay.mDeepDataHandle,
                    pbr::getFilm(occlRay.mTilePassAndFilm))) {

                    // we only need to add to the visibility buffer once per shadow ray
                    break;
                }
            }
        }
    }
}

void
accumVisibilityAovsHit(pbr::TLState* pbrTls, const BundledOcclRay& occlRay,
                       const FrameState& fs, const int numItems)
{
    accumVisibilityAovs(pbrTls, occlRay, fs, numItems, 1.0f /* hit light */);
}

void
accumVisibilityAovsOccluded(pbr::TLState* pbrTls, const BundledOcclRay& occlRay,
                            const FrameState& fs, const int numItems)
{
    accumVisibilityAovs(pbrTls, occlRay, fs, numItems, 0.0f /*did not hit light */);
}

void
fillBundledRadiance(pbr::TLState* pbrTls, BundledRadiance* dst, const BundledOcclRay& occlRay)
{
    const BundledOcclRayData *b = static_cast<BundledOcclRayData *>(
                pbrTls->getListItem(occlRay.mDataPtrHandle, 0));

    scene_rdl2::math::Color radiance = occlRay.mRadiance;


    dst->mRadiance = RenderColor(radiance.r,
                                 radiance.g,
                                 radiance.b,
                                 0.f);

    dst->mPathPixelWeight = 0.f;
    dst->mPixel = occlRay.mPixel;
    dst->mSubPixelIndex = occlRay.mSubpixelIndex;
    dst->mDeepDataHandle = pbrTls->acquireDeepData(occlRay.mDeepDataHandle);
    // Cryptomatte only records pixel coverage information as seen from the camera; i.e. only primary rays add to
    // the cryptomatte buffer, so occlusion rays don't need to acquire a handle to the cryptomatte data. (This
    // differs from deep data, which records radiance information, and hence must acquire the deep data handle even
    // for secondary rays.)
    dst->mCryptomatteDataHandle = nullHandle;
    dst->mTilePassAndFilm = occlRay.mTilePassAndFilm;
    dst->mCameraId = b->mCameraId;
}

scene_rdl2::math::Color
getTransmittance(pbr::TLState* pbrTls, const BundledOcclRay& occlRay)
{
    // is light uniform?
    const BundledOcclRayData *b = static_cast<BundledOcclRayData *>(
                pbrTls->getListItem(occlRay.mDataPtrHandle, 0));
    mcrt_common::Ray trRay(occlRay.mOrigin, occlRay.mDir, occlRay.mMinT, occlRay.mMaxT, occlRay.mTime,
        occlRay.mDepth + 1);

    return pbrTls->mFs->mIntegrator->transmittance(pbrTls, trRay, occlRay.mPixel,
        occlRay.mSubpixelIndex, occlRay.mSequenceID, b->mLight);
}

} // namespace pbr
} // namespace moonray

