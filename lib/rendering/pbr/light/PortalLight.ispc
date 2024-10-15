// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include "RectLight.isph"
#include <moonray/rendering/pbr/core/RayState.isph>
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(PortalLight);

export uniform uint32_t
PortalLight_hudValidation(uniform bool verbose)
{
    PORTAL_LIGHT_VALIDATION;
}

varying Color
PortalLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls, const varying Vec3f &wi,
                 const varying Vec3f &p, const varying LightFilterRandomValues& filterR, varying float time,
                 const varying LightIntersection &isect, uniform bool fromCamera,
                 const uniform LightFilterList * uniform lightFilterList, const varying PathVertex * uniform pv,
                 varying float rayDirFootprint, varying float * uniform pdf)
{
    const uniform PortalLight * uniform light = (const uniform PortalLight * uniform)li;
    const uniform Light * uniform refLight = (const uniform Light * uniform)light->mRefLight;

    MNRY_ASSERT(refLight && li->mOn);

    Color radiance = sBlack;

    // ------- Eval reference light -----------
    Vec3f n;
    LightIntersection refIsect;
    // Find the intersection of wi (sampled from the portal) with the associated light
    Light_intersect(refLight, p, n, wi, time, sEnvLightDistance, refIsect);
    // Evaluate the light at the found intersection
    radiance = Light_eval(refLight, tls, wi, p, filterR, time, refIsect, fromCamera,
                          lightFilterList, pv, rayDirFootprint, nullptr);

    // -------- Eval portal -- any radiance will be a multiplier on reference light's radiance -----------
    radiance = radiance * RectLight_eval(li, tls, wi, p, filterR, time, isect, fromCamera,
                                         lightFilterList, pv, rayDirFootprint, pdf);

    return radiance;
}

//----------------------------------------------------------------------------

// The PortalLight mostly behaves like a RectLight, except for its eval function, which
// finds radiance from its reference light, mRefLight.
export void
PortalLight_init(uniform PortalLight * uniform light)
{
    Light_init((uniform Light * uniform)light, RectLight_canIlluminate,
                RectLight_intersect, RectLight_sample, PortalLight_eval,
                RectLight_getThetaO, RectLight_getThetaE);
}

//----------------------------------------------------------------------------

