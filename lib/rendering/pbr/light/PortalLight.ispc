// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include "RectLight.isph"
#include <moonray/rendering/pbr/core/RayState.isph>
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(PortalLight);

export uniform uint32_t
PortalLight_hudValidation(uniform bool verbose)
{
    PORTAL_LIGHT_VALIDATION;
}


static void
PortalLight_computeCorners(const uniform PortalLight * uniform light,
                           varying Vec3f * uniform corners, varying float time)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) light;

    corners[0] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor( light->mHalfWidth,  light->mHalfHeight, 0.f), time);
    corners[1] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor( light->mHalfWidth, -light->mHalfHeight, 0.f), time);
    corners[2] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(-light->mHalfWidth,  light->mHalfHeight, 0.f), time);
    corners[3] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(-light->mHalfWidth, -light->mHalfHeight, 0.f), time);
}


varying bool
PortalLight_canIlluminate(const uniform Light * uniform li,
                          const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
                          varying float radius, const uniform LightFilterList * uniform lightFilterList,
                          const varying PathVertex * uniform pv)
{
    const uniform PortalLight * uniform light = (const uniform PortalLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;
    const uniform Light * uniform refLight = (const uniform Light * uniform)light->mRefLight;

    MNRY_ASSERT(li->mOn && refLight && (refLight->mType == LIGHT_TYPE_ENV || refLight->mType == LIGHT_TYPE_DISTANT));

    Vec3f localP = LocalParamLight_xformVectorRender2Local(lpl, p - Light_getPosition(li, time), time);
    if (localP.z <= 0.0f) {
        // No lighting beyond the portal.
        return false;
    }

    // call shared RectLight code
    return RectLight_canIlluminateHelper(li, p, cullingNormal, time, radius, lightFilterList, pv);
}


varying bool
PortalLight_intersect(const uniform Light * uniform li,
                      const varying Vec3f &p, const varying Vec3f& cullingNormal,
                      const varying Vec3f &wi, varying float time,
                      varying float maxDistance, varying LightIntersection &isect)
{
    const uniform PortalLight * uniform light = (const uniform PortalLight * uniform)li;
    const uniform Light * uniform refLight = (const uniform Light * uniform)light->mRefLight;

    MNRY_ASSERT(li->mOn && refLight && (refLight->mType == LIGHT_TYPE_ENV || refLight->mType == LIGHT_TYPE_DISTANT));

    // Intersect against both the rectangle and the ref light, but set the isect by the ref light.
    LightIntersection isectRect;
    return RectLight_intersect(  li,     p, cullingNormal, wi, time, maxDistance, isectRect)
           &&  Light_intersect(refLight, p, cullingNormal, wi, time, maxDistance, isect);
}


varying bool
PortalLight_sample(const uniform Light * uniform li,
                   const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
                   const varying Vec3f &r, varying Vec3f &wi, varying LightIntersection &isect,
                   varying float rayDirFootprint)
{
    const uniform PortalLight * uniform light = (const uniform PortalLight * uniform)li;
    const uniform Light * uniform refLight = (const uniform Light * uniform)light->mRefLight;

    MNRY_ASSERT(li->mOn && refLight && (refLight->mType == LIGHT_TYPE_ENV || refLight->mType == LIGHT_TYPE_DISTANT));

    if (refLight->mType == LIGHT_TYPE_ENV) {
        // Env light currently samples the portal's rectangle, but we follow up with a call to intersect()
        // to generate the isect values on the env light.
        LightIntersection isectRect;
        return RectLight_sample(li, p, cullingNormal, time, r, wi, isectRect, rayDirFootprint)
            && Light_intersect(refLight, p, cullingNormal, wi, time, sMaxValue, isect);
    }

    // Distant light is just sampled in the usual way.
    return Light_sample(refLight, p, cullingNormal, time, r, wi, isect, rayDirFootprint);
}



varying Color
PortalLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls, const varying Vec3f &wi,
                 const varying Vec3f &p, const varying LightFilterRandomValues& filterR, varying float time,
                 const varying LightIntersection &isect, uniform bool fromCamera,
                 const uniform LightFilterList * uniform lightFilterList, const varying PathVertex * uniform pv,
                 varying float rayDirFootprint, varying float * uniform visibility, varying float * uniform pdf)
{
    const uniform PortalLight * uniform light = (const uniform PortalLight * uniform)li;
    const uniform Light * uniform refLight = (const uniform Light * uniform)light->mRefLight;

    MNRY_ASSERT(li->mOn && refLight && (refLight->mType == LIGHT_TYPE_ENV || refLight->mType == LIGHT_TYPE_DISTANT));

    // Get rect light intersection.
    // Also test the result and reject accordingly, since distant light may have sampled outside the portal region.
    LightIntersection isectRect;
    Vec3f dummyNormal;
    setInvalidCullingNormal(&dummyNormal);
    if (!RectLight_intersect(li, p, dummyNormal, wi, time, sMaxValue, isectRect)) return sBlack;
   
    // If the ref light is a distant light, we'll compute the pdf here in the ref light eval call
    // otherwise we set it to null to skip the redundant calculation
    varying float * uniform pdfRef = (refLight->mType == LIGHT_TYPE_DISTANT) ? pdf : nullptr;

    // Evaluate reference light
    // TODO: Support the ref light's lightfilterlist?
    Color radiance = Light_eval(refLight, tls, wi, p, filterR, time, isect, fromCamera,
                                nullptr, pv, rayDirFootprint, visibility, pdfRef);

    // If the ref light is an env light, we'll compute the pdf here in the rect light eval call
    // because the env light was sampled using the rectangle
    varying float * uniform pdfRect = (refLight->mType == LIGHT_TYPE_ENV) ? pdf : nullptr;

    // Evaluate portal rect light
    radiance = radiance * RectLight_eval(li, tls, wi, p, filterR, time, isectRect, fromCamera,
                                         lightFilterList, pv, rayDirFootprint, visibility, pdfRect);
    return radiance;
}

//----------------------------------------------------------------------------

// The PortalLight mostly behaves like a RectLight, except for its eval function, which
// finds radiance from its reference light, mRefLight.
export void
PortalLight_init(uniform PortalLight * uniform light)
{
    Light_init((uniform Light * uniform)light,
               LIGHT_TYPE_PORTAL,
               PortalLight_canIlluminate,
               PortalLight_intersect,
               PortalLight_sample,
               PortalLight_eval,
               RectLight_getThetaO,
               RectLight_getThetaE);
}

//----------------------------------------------------------------------------

