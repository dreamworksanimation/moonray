// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "Types.isph"

export uniform uint32_t
BundledOcclRay_hvdValidation(uniform bool verbose)
{
    BUNDLED_OCCL_RAY_VALIDATION(VLEN);
}

export uniform uint32_t
BundledRadiance_hvdValidation(uniform bool verbose)
{
    BUNDLED_RADIANCE_VALIDATION(VLEN);
}

export uniform uint32_t
FrameState_hudValidation(uniform bool verbose)
{
    FRAME_STATE_VALIDATION;
}

// ----------------------------------------------------------------------------
void
BundledOcclRayData_init(uniform BundledOcclRayData * varying this,
                        uint32_t flags,
                        const uniform Light * varying light)
{
    this->mFlags = flags;
    this->mLpeRadiance = sBlack;
    this->mLpeStateId = 0;
    this->mRayEpsilon = 0.f;
    this->mLight = light;
}

void
BundledOcclRay_init(varying BundledOcclRay * uniform this,
                    const varying Vec3f &origin,
                    const varying Vec3f &dir,
                    varying float minT,
                    varying float maxT,
                    varying float time,
                    varying int depth,
                    const varying Color &radiance,
                    varying uint32_t pixel,
                    varying int subpixelIndex,
                    varying uint32_t sequenceID,
                    varying uint32_t tilePass,
                    varying uint32_t dataPtrHandle,
                    varying uint32_t deepDataHandle,
                    varying uint32_t cryptomatteDataHandle,
                    const varying Vec3f &cryptoRefP,
                    const varying Vec3f &cryptoRefN,
                    const varying Vec2f &cryptoUV,
                    varying uint32_t occlTestType,
                    const varying int32_t shadowReceiverId)
{
    this->mOrigin = origin;
    this->mDir = dir;
    this->mMinT = minT;
    this->mMaxT = maxT;
    this->mTime = time;
    this->mDepth = depth;
    this->mRadiance = radiance;
    this->mPixel = pixel;
    this->mSubpixelIndex = subpixelIndex;
    this->mSequenceID = sequenceID;
    this->mTilePass = tilePass;
    this->mDataPtrHandle = dataPtrHandle;
    this->mDeepDataHandle = deepDataHandle;
    this->mCryptomatteDataHandle = cryptomatteDataHandle;
    this->mCryptoRefP = cryptoRefP;
    this->mCryptoRefN = cryptoRefN;
    this->mCryptoUV = cryptoUV;
    this->mOcclTestType = occlTestType;
    this->mShadowReceiverId = shadowReceiverId;
}


