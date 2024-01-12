// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Bsdfv.ispc

// Exported functions that enable the use of Bsdfv, BsdfLobev, and BsdfSlicev
// objects in C++.  The purpose is to allow scalar C++ code to operate
// on vector obbject without the need for an expensive and error prone
// conversion process.

#include "Bsdf.isph"
#include "BsdfSlice.isph"

#include <scene_rdl2/common/math/ispc/Color.isph>
#include <scene_rdl2/common/math/ispc/Vec3.isph>

//==---------------------------------------------------------------------------
// Bsdfv Methods

export void
#pragma ignore warning(all)
Bsdfv_init(varying Bsdf * uniform bsdf)
{
    Bsdf_init(bsdf);
}


export void
#pragma ignore warning(all)
Bsdfv_setPostScatterExtraAovs(varying Bsdf * uniform bsdf,
                              uniform int numExtraAovs,
                              const uniform int * uniform labelIds,
#pragma ignore warning(all)
                              const varying Color * uniform colors)
{
    Bsdf_setPostScatterExtraAovs(bsdf, numExtraAovs, labelIds, colors);
}

//==---------------------------------------------------------------------------
// BsdfLobev Methods

export void
#pragma ignore warning(all)
BsdfLobev_albedo(const varying BsdfLobe * uniform lobe, const varying BsdfSlice * uniform slice,
    uniform int lane, uniform float * uniform result)
{
    varying Color color;
    if (lane == programIndex) {
        color = BsdfLobe_albedo(lobe, *slice);
    }
    result[0] = extract(color.r, lane);
    result[1] = extract(color.g, lane);
    result[2] = extract(color.b, lane);
}

export void
#pragma ignore warning(all)
BsdfLobev_eval(const varying BsdfLobe * uniform lobe, const varying BsdfSlice * uniform slice,
    const uniform float * uniform wi, uniform int lane, uniform float * uniform result,
    uniform float * uniform pdf)
{
    varying Vec3f vWi = Vec3f_ctor(wi[0], wi[1], wi[2]);
    varying Color color;
    varying float vPdf;
    if (lane == programIndex) {
        color = BsdfLobe_eval(lobe, *slice, vWi, &vPdf);
    }
    result[0] = extract(color.r, lane);
    result[1] = extract(color.g, lane);
    result[2] = extract(color.b, lane);
    *pdf = extract(vPdf, lane);
}

export void
#pragma ignore warning(all)
BsdfLobev_sample(const varying BsdfLobe * uniform lobe, const varying BsdfSlice * uniform slice,
    uniform float r1, uniform float r2, uniform int lane, uniform float * uniform wi,
    uniform float * uniform result, uniform float * uniform pdf)
{
    varying float vR1 = r1;
    varying float vR2 = r2;
    varying Vec3f vWi;
    varying Color color;
    varying float vPdf;

    if (lane == programIndex) {
        color = BsdfLobe_sample(lobe, *slice, vR1, vR2, vWi, vPdf);
    }

    wi[0] = extract(vWi.x, lane);
    wi[1] = extract(vWi.y, lane);
    wi[2] = extract(vWi.z, lane);
    result[0] = extract(color.r, lane);
    result[1] = extract(color.g, lane);
    result[2] = extract(color.b, lane);
    *pdf = extract(vPdf, lane);
}

