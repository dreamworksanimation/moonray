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
    mRefRdlLight = refLightSO->asA<scene_rdl2::rdl2::Light>();

    return isOn();
}

Color PortalLight::eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi, 
            const scene_rdl2::math::Vec3f &p, const LightFilterRandomValues& filterR, float time, 
            const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, 
            float rayDirFootprint, float *pdf) const
{
    MNRY_ASSERT(mRefLight && mOn);
    Color radiance;
   
    // ------- Eval reference light -----------
    LightIntersection refIsect;
    // Find the intersection of wi (sampled from the portal) with the associated light
    mRefLight->intersect(p, nullptr, wi, time, sEnvLightDistance, refIsect);
    // Evaluate the light at the found intersection
    radiance = mRefLight->eval(tls, wi, p, filterR, time, refIsect, fromCamera, 
                               lightFilterList, rayDirFootprint, nullptr);

    // -------- Eval portal -- any radiance will be a multiplier on env light radiance -----------
    radiance *= RectLight::eval(tls, wi, p, filterR, time, isect, fromCamera, 
                                lightFilterList, rayDirFootprint, pdf);

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

