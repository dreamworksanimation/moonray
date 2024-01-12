// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Bsdfv.h

// Convenience methods for operating on Bsdfv, BsdfLobev, and BsdfSlicev objects from c++ code
// Mostly this is to allow scalar implementations to make use of vectorized shading
// results without needing an error prone conversion to Bsdf and BsdfLobe objects

#pragma once

#include "Bsdf.h"
#include "BsdfSlice.h"

#include <scene_rdl2/common/math/ispc/Typesv.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Math.h>

namespace moonray {
namespace shading {


// Bsdfv Methods
void Bsdfv_init(Bsdfv *bsdv);
inline void addLobe(Bsdfv &bsdfv, BsdfLobev *lobev) { bsdfv.mLobes[bsdfv.mNumLobes++] = lobev; }
inline int getLobeCount(const Bsdfv &bsdfv) { return bsdfv.mNumLobes; }
inline BsdfLobev *getLobe(Bsdfv &bsdfv, int lobeIndex) { return bsdfv.mLobes[lobeIndex]; }
inline bool getIsSpherical(const Bsdfv &bsdfv, int lane) { return bsdfv.mIsSpherical[lane]; }
inline int getLpeMaterialLabelId(const Bsdfv &bsdfv) { return bsdfv.mLpeMaterialLabelId; }
void Bsdfv_setPostScatterExtraAovs(Bsdfv *bsdfv, int numExtraAovs, int *labelIds, scene_rdl2::math::Colorv *colors);

// BsdfLobev Methods
inline BsdfLobe::Type getType(const BsdfLobev &lobev) { return static_cast<BsdfLobe::Type>(lobev.mType); }
inline bool
matchesFlags(const BsdfLobev &lobev, int flags)
{
    return ((lobev.mType & BsdfLobe::ALL_SURFACE_SIDES & flags)  &&
        (lobev.mType & BsdfLobe::ALL_LOBES & flags));
}
inline bool matchesFlags(const BsdfLobev &lobev, BsdfLobe::Type flags) { return matchesFlags(lobev, static_cast<int>(flags)); }
inline int getLabel(const BsdfLobev &lobev) { return lobev.mLabel; }
inline bool isActive(const BsdfLobev &lobev, int lane) { return (1 << lane) & lobev.mMask; }
scene_rdl2::math::Color albedo(const BsdfLobev &lobev, const BsdfSlicev &slicev, int lane);
scene_rdl2::math::Color eval(const BsdfLobev &lobev, const BsdfSlicev &slicev, const scene_rdl2::math::Vec3f &wi, int lane, float *pdf);
scene_rdl2::math::Color sample(
    const BsdfLobev &lobev, const BsdfSlicev &slicev, float r1, float r2, int lane,
    scene_rdl2::math::Vec3f &wi, float &pdf);

// BsdfSlicev Methods
inline bool getIncludeCosineTerm(const BsdfSlicev &slicev, int lane) { return slicev.mIncludeCosineTerm[lane]; }
inline BsdfLobe::Type getFlags(const BsdfSlicev &slicev, int lane) { return BsdfLobe::Type(slicev.mFlags[lane]); }

inline BsdfLobe::Type
getSurfaceFlags(const BsdfSlicev &slicev, const Bsdfv &bsdfv, const scene_rdl2::math::Vec3f &wi, int lane)
{
    if (getIsSpherical(bsdfv, lane)) {
        return static_cast<BsdfLobe::Type>(slicev.mFlags[lane]);
    }

    const scene_rdl2::math::Vec3f ng(slicev.mNg.x[lane], slicev.mNg.y[lane], slicev.mNg.z[lane]);
    const scene_rdl2::math::Vec3f wo(slicev.mWo.x[lane], slicev.mWo.y[lane], slicev.mWo.z[lane]);
    return BsdfLobe::Type(slicev.mFlags[lane] &
                          ((scene_rdl2::math::dot(ng, wo) * scene_rdl2::math::dot(ng, wi) > 0.0f) ?
                           ~BsdfLobe::TRANSMISSION : ~BsdfLobe::REFLECTION));

}



} // namespace shading
} // namespace moonray

