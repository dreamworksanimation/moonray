// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#include "Types.h"
#include <scene_rdl2/common/platform/HybridUniformData.h>
#include <scene_rdl2/common/platform/HybridVaryingData.h>

namespace ispc {
extern "C" uint32_t BundledOcclRay_hvdValidation(bool);
extern "C" uint32_t BundledRadiance_hvdValidation(bool);
extern "C" uint32_t FrameState_hudValidation(bool);
}

using namespace scene_rdl2;
using scene_rdl2::math::isfinite;

namespace moonray {
namespace pbr {

bool
BundledOcclRay::isValid() const
{
    MNRY_ASSERT(isFinite(mOrigin));
    MNRY_ASSERT(isNormalized(mDir));
    MNRY_ASSERT(math::isfinite(mMinT));
    MNRY_ASSERT(math::isfinite(mMaxT) && mMinT >= 0.f);
    MNRY_ASSERT(mMinT <= mMaxT);
    MNRY_ASSERT(math::isfinite(mTime));
    MNRY_ASSERT(isFinite(mRadiance));
    return true;
}

HVD_VALIDATOR(BundledOcclRay);
HVD_VALIDATOR(BundledRadiance);
HUD_VALIDATOR(FrameState);

} // namespace pbr
} // namespace moonray


