// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Bsfdv.cc

#include "Bsdfv.h"

#include <moonray/rendering/shading/ispc/bsdf/Bsdfv_ispc_stubs.h>

namespace moonray {
namespace shading {

//==---------------------------------------------------------------------------
// Bsdfv Methods

void
Bsdfv_init(Bsdfv *bsdfv)
{
    ispc::Bsdfv_init(bsdfv);
}

void
Bsdfv_setPostScatterExtraAovs(Bsdfv *bsdfv, int numExtraAovs, int *labelIds, scene_rdl2::math::Colorv *colors)
{
    ispc::Bsdfv_setPostScatterExtraAovs(bsdfv, numExtraAovs, labelIds, colors);
}

//==---------------------------------------------------------------------------
// BsdfLobev Methods

scene_rdl2::math::Color
albedo(const BsdfLobev &lobev, const BsdfSlicev &slicev, int lane)
{
    scene_rdl2::math::Color result;
    ispc::BsdfLobev_albedo(&lobev, &slicev, lane, &result[0]);

    return result;
}

scene_rdl2::math::Color
eval(const BsdfLobev &lobev, const BsdfSlicev &slicev, const scene_rdl2::math::Vec3f &wi, int lane, float *pdf)
{
    scene_rdl2::math::Color result;
    ispc::BsdfLobev_eval(&lobev, &slicev, &wi[0], lane, &result[0], pdf);

    return result;
}

scene_rdl2::math::Color
sample(const BsdfLobev &lobev, const BsdfSlicev &slicev, float r1, float r2, int lane, scene_rdl2::math::Vec3f &wi, float &pdf)
{
    scene_rdl2::math::Color result;
    ispc::BsdfLobev_sample(&lobev, &slicev, r1, r2, lane, &wi[0], &result[0], &pdf);

    return result;
}

} // namespace shading
} // namespace moonray

