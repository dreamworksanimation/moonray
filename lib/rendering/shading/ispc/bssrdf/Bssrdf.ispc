// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Bssrdf.ispc

#include "Bssrdf.isph"
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

varying float
Bssrdf_sampleGlobal(const varying Bssrdf * uniform bssrdf,
                    varying float r1, varying float r2, varying float cosThetaMax,
                    varying Vec3f &ws)
{
    // Sample direction within a spherical cap below the surface
    ws = sampleLocalSphericalCapUniform(r1, r2, cosThetaMax);

    float pdf = rcp(sTwoPi * (1.0f - cosThetaMax));

    return pdf;
}

uniform bool
Bssrdf_getProperty(const varying Bssrdf * uniform bssrdf,
                   uniform SubsurfaceProperty property,
                   varying float * uniform dest)
{
    uniform bool handled = true;

    BssrdfGetPropertyFn getPropertyFn = (BssrdfGetPropertyFn) bssrdf->mGetPropertyFn;
    if (!getPropertyFn(bssrdf, property, dest)) {
        // no properties handled by the base class
        MNRY_ASSERT(0);
        handled = false;
    }

    return handled;
}

//----------------------------------------------------------------------------
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(Bssrdf);

