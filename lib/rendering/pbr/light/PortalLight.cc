// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "PortalLight.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/light/PortalLight_ispc_stubs.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;
using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

namespace moonray {
namespace pbr {

bool                                                            PortalLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*>  PortalLight::sRefLight;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>          PortalLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>          PortalLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         PortalLight::sWidthKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         PortalLight::sHeightKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         PortalLight::sSpreadKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         PortalLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         PortalLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>           PortalLight::sClearRadiusInterpolationKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>           PortalLight::sSidednessKey;

//----------------------------------------------------------------------------

HUD_VALIDATOR(PortalLight);

PortalLight::PortalLight(const scene_rdl2::rdl2::Light* rdlLight)
    : mRefLight(nullptr), mRefRdlLight(nullptr), RectLight(rdlLight)
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::PortalLight_init(this->asIspc());
}

PortalLight::~PortalLight() {}

bool PortalLight::update(const Mat4d& world2render)
{
    if (!RectLight::update(world2render)) return false;

    scene_rdl2::rdl2::SceneObject* refLightSO = mRdlLight->get<scene_rdl2::rdl2::SceneObject*>(sRefLight);

    if (!refLightSO) return false;

    auto name = refLightSO->getSceneClass().getName();
    if (name != "EnvLight" && name != "DistantLight") {
        scene_rdl2::logging::Logger::warn("PortalLight can currently reference only EnvLight or DistantLight");
        return false;
    }

    mRefRdlLight = refLightSO->asA<scene_rdl2::rdl2::Light>();

    return isOn();
}


bool
PortalLight::canIlluminate(const Vec3f p, const Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList, const PathVertex* pv) const
{
    MNRY_ASSERT(mOn && mRefLight && (mRefLight->isEnv() || mRefLight->isDistant()));

    Vec3f localP = xformVectorRender2Local(p - getPosition(time), time);
    if (localP.z <= 0.0f) {
        // No lighting beyond the portal.
        return false;
    }

    // call shared RectLight code
    return RectLight::canIlluminateHelper(p, n, time, radius, lightFilterList, pv);
}


bool
PortalLight::intersect(const Vec3f &p, const Vec3f *n,  const Vec3f &wi, float time,
        float maxDistance, LightIntersection &isect) const
{
    MNRY_ASSERT(mOn && mRefLight && (mRefLight->isEnv() || mRefLight->isDistant()));

    // Intersect against both the rectangle and the ref light, but set the isect by the ref light.
    LightIntersection isectRect;
    return RectLight::intersect(p, n, wi, time, maxDistance, isectRect)
        && mRefLight->intersect(p, n, wi, time, maxDistance, isect);
}


bool PortalLight::sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time,
                         const scene_rdl2::math::Vec3f& r, scene_rdl2::math::Vec3f &wi,
                         LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn && mRefLight && (mRefLight->isEnv() || mRefLight->isDistant()));

    if (mRefLight->isEnv()) {
        // Env light currently samples the portal's rectangle, but we follow up with a call to intersect()
        // to generate the isect values on the env light.
        LightIntersection isectRect;
        return RectLight::sample(p, n, time, r, wi, isectRect, rayDirFootprint)
            && mRefLight->intersect(p, n, wi, time, sMaxValue, isect);
    }

    // Distant light is just sampled in the usual way.
    return mRefLight->sample(p, n, time, r, wi, isect, rayDirFootprint);
}


Color PortalLight::eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi,
            const scene_rdl2::math::Vec3f &p, const LightFilterRandomValues& filterR, float time,
            const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList,
            const PathVertex *pv, float rayDirFootprint, float *visibility, float *pdf) const
{
    MNRY_ASSERT(mOn && mRefLight && (mRefLight->isEnv() || mRefLight->isDistant()));

    // Get rect light intersection.
    // Also test the result and reject accordingly, since distant light may have sampled outside the portal region.
    LightIntersection isectRect;
    if (!RectLight::intersect(p, nullptr, wi, time, sMaxValue, isectRect)) return sBlack;

    // If the ref light is a distant light, we'll compute the pdf here in the ref light eval call
    // otherwise we set it to null to skip the redundant calculation
    float *pdfRef = mRefLight->isDistant() ? pdf : nullptr;
   
    // Evaluate reference light
    // TODO: Support the ref light's lightfilterlist?
    Color radiance = mRefLight->eval(tls, wi, p, filterR, time, isect, fromCamera,
                                     nullptr, pv, rayDirFootprint, visibility, pdfRef);

    // If the ref light is an env light, we'll compute the pdf here in the rect light eval call
    // because the env light was sampled using the rectangle
    float *pdfRect = mRefLight->isEnv() ? pdf : nullptr;

    // Evaluate portal rect light
    radiance *= RectLight::eval(tls, wi, p, filterR, time, isectRect, fromCamera,
                                    lightFilterList, pv, rayDirFootprint, visibility, pdfRect);
    return radiance;
}


void
PortalLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sRefLight           = sc.getAttributeKey<scene_rdl2::rdl2::SceneObject*> ("light");
    sNormalizedKey      = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("normalized");
    sApplySceneScaleKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool> ("apply_scene_scale");
    sWidthKey           = sc.getAttributeKey<scene_rdl2::rdl2::Float>("width");
    sHeightKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("height");
    sSpreadKey          = sc.getAttributeKey<scene_rdl2::rdl2::Float>("spread");
    sSidednessKey       = sc.getAttributeKey<scene_rdl2::rdl2::Int>  ("sidedness");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

} // namespace pbr
} // namespace moonray

