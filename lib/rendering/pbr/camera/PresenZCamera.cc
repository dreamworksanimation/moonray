// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// @file PresenZCamera.cc

#include "PresenZCamera.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/mcrt_common/PresenZSettings.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include "API/PzCameraApi.h"

namespace moonray {
namespace pbr {
using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool                               PresenZCamera::sAttributeKeyInitialized = false;

rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sPresenZEnabledKey;
rdl2::AttributeKey<rdl2::Int>      PresenZCamera::sPhaseKey;
rdl2::AttributeKey<rdl2::String>   PresenZCamera::sDetectFileKey;
rdl2::AttributeKey<rdl2::String>   PresenZCamera::sRenderFileKey;
rdl2::AttributeKey<rdl2::Float>    PresenZCamera::sRenderScaleKey;
rdl2::AttributeKey<rdl2::Vec3f>    PresenZCamera::sZOVScaleKey;
rdl2::AttributeKey<rdl2::Float>    PresenZCamera::sDistanceToGroundKey;

rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sDraftRenderingKey;
rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sFroxtrumRenderingKey;
rdl2::AttributeKey<rdl2::Int>      PresenZCamera::sFroxtrumDepthKey;
rdl2::AttributeKey<rdl2::Int>      PresenZCamera::sFroxtrumResolutionKey;
rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sRenderInsideZOVKey;
rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sEnableDeepReflectionsKey;
rdl2::AttributeKey<rdl2::Float>    PresenZCamera::sInterPupillaryDistanceKey;

rdl2::AttributeKey<rdl2::Int>      PresenZCamera::sZOVOffsetXKey;
rdl2::AttributeKey<rdl2::Int>      PresenZCamera::sZOVOffsetYKey;
rdl2::AttributeKey<rdl2::Int>      PresenZCamera::sZOVOffsetZKey;
rdl2::AttributeKey<rdl2::Vec3f>    PresenZCamera::sSpecularPointOffsetKey;

rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sEnableClippingSphereKey;
rdl2::AttributeKey<rdl2::Float>    PresenZCamera::sClippingSphereRadiusKey;
rdl2::AttributeKey<rdl2::Vec3f>    PresenZCamera::sClippingSphereCenterKey;
rdl2::AttributeKey<rdl2::Bool>     PresenZCamera::sClippingSphereRenderInsideKey;

PresenZCamera::PresenZCamera(const rdl2::Camera* rdlCamera) :
    Camera(rdlCamera),
    mPresenZSettings(fauxstd::make_unique<mcrt_common::PresenZSettings>())
{
    initAttributeKeys(rdlCamera->getSceneClass());
}

void PresenZCamera::initAttributeKeys(const rdl2::SceneClass& sceneClass)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sPresenZEnabledKey = sceneClass.getAttributeKey<rdl2::Bool>("presenz_enabled");
    sPhaseKey = sceneClass.getAttributeKey<rdl2::Int>("phase");
    sDetectFileKey = sceneClass.getAttributeKey<rdl2::String>("detect_file");
    sRenderFileKey = sceneClass.getAttributeKey<rdl2::String>("render_file");
    sRenderScaleKey = sceneClass.getAttributeKey<rdl2::Float>("render_scale");
    sZOVScaleKey = sceneClass.getAttributeKey<rdl2::Vec3f>("zov_scale");
    sDistanceToGroundKey = sceneClass.getAttributeKey<rdl2::Float>("distance_to_ground");

    sDraftRenderingKey = sceneClass.getAttributeKey<rdl2::Bool>("draft_rendering");
    sFroxtrumRenderingKey = sceneClass.getAttributeKey<rdl2::Bool>("froxtrum_rendering");
    sFroxtrumDepthKey = sceneClass.getAttributeKey<rdl2::Int>("froxtrum_depth");
    sFroxtrumResolutionKey = sceneClass.getAttributeKey<rdl2::Int>("froxtrum_resolution");
    sRenderInsideZOVKey = sceneClass.getAttributeKey<rdl2::Bool>("render_inside_zov");
    sEnableDeepReflectionsKey = sceneClass.getAttributeKey<rdl2::Bool>("enable_deep_reflections");
    sInterPupillaryDistanceKey = sceneClass.getAttributeKey<rdl2::Float>("inter_pupillary_distance");

    sZOVOffsetXKey = sceneClass.getAttributeKey<rdl2::Int>("zov_offset_x");
    sZOVOffsetYKey = sceneClass.getAttributeKey<rdl2::Int>("zov_offset_y");
    sZOVOffsetZKey = sceneClass.getAttributeKey<rdl2::Int>("zov_offset_z");
    sSpecularPointOffsetKey = sceneClass.getAttributeKey<rdl2::Vec3f>("specular_point_offset");

    sEnableClippingSphereKey = sceneClass.getAttributeKey<rdl2::Bool>("enable_clipping_sphere");
    sClippingSphereRadiusKey = sceneClass.getAttributeKey<rdl2::Float>("clipping_sphere_radius");
    sClippingSphereCenterKey = sceneClass.getAttributeKey<rdl2::Vec3f>("clipping_sphere_center");
    sClippingSphereRenderInsideKey = sceneClass.getAttributeKey<rdl2::Bool>("clipping_sphere_render_inside");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

bool
PresenZCamera::getIsDofEnabledImpl() const
{
    return false;
}

void
PresenZCamera::updateImpl(const Mat4d& world2render)
{
    mPresenZSettings->setEnabled(getRdlCamera()->get(sPresenZEnabledKey));
    mPresenZSettings->setPhase(getRdlCamera()->get(sPhaseKey));
    mPresenZSettings->setDetectFile(getRdlCamera()->get(sDetectFileKey));
    mPresenZSettings->setRenderFile(getRdlCamera()->get(sRenderFileKey));
    mPresenZSettings->setCamToWorld(getCamera2World());
    mPresenZSettings->setRenderScale(getRdlCamera()->get(sRenderScaleKey));
    mPresenZSettings->setZOVScale(getRdlCamera()->get(sZOVScaleKey));
    mPresenZSettings->setDistanceToGround(getRdlCamera()->get(sDistanceToGroundKey));

    mPresenZSettings->setDraftRendering(getRdlCamera()->get(sDraftRenderingKey));
    mPresenZSettings->setFroxtrumRendering(getRdlCamera()->get(sFroxtrumRenderingKey));
    mPresenZSettings->setFroxtrumDepth(getRdlCamera()->get(sFroxtrumDepthKey));
    mPresenZSettings->setFroxtrumResolution(getRdlCamera()->get(sFroxtrumResolutionKey));
    mPresenZSettings->setRenderInsideZOV(getRdlCamera()->get(sRenderInsideZOVKey));
    mPresenZSettings->setEnableDeepReflections(getRdlCamera()->get(sEnableDeepReflectionsKey));
    mPresenZSettings->setInterpupillaryDistance(getRdlCamera()->get(sInterPupillaryDistanceKey));

    mPresenZSettings->setZOVOffset(
        getRdlCamera()->get(sZOVOffsetXKey),
        getRdlCamera()->get(sZOVOffsetYKey),
        getRdlCamera()->get(sZOVOffsetZKey));
    mPresenZSettings->setSpecularPointOffset(getRdlCamera()->get(sSpecularPointOffsetKey));

    mPresenZSettings->setEnableClippingSphere(getRdlCamera()->get(sEnableClippingSphereKey));
    mPresenZSettings->setClippingSphereRadius(getRdlCamera()->get(sClippingSphereRadiusKey));
    mPresenZSettings->setClippingSphereCenter(getRdlCamera()->get(sClippingSphereCenterKey));
    mPresenZSettings->setClippingSphereRenderInside(getRdlCamera()->get(sClippingSphereRenderInsideKey));
}

void
PresenZCamera::createRayImpl(mcrt_common::RayDifferential* dstRay,
                             float x,
                             float y,
                             float time,
                             float /*lensU*/,
                             float /*lensV*/) const
{
    // Set an invalid ray as default
    Vec3f rayOrigin = Vec3f(0.0f, 0.0f, 0.0f);
    Vec3f rayDirection = Vec3f(0.0f, 0.0f, 1.0f);
    *dstRay = mcrt_common::RayDifferential(
        rayOrigin, rayDirection,
        rayOrigin, rayDirection,
        rayOrigin, rayDirection,
        math::sMaxValue, math::sMaxValue, time, 0);

    if (mPresenZSettings == nullptr || !mPresenZSettings->getEnabled()) {
        return;
    }

    const PresenZ::Camera::PzCameraRay pzRay = PresenZ::Camera::PzGetCameraRay(x, y);
    if (!pzRay.isValid()) {
        return;
    }

    const Vec2f res = mPresenZSettings->getResolution();
    rayOrigin = Vec3f(pzRay.origin.x, pzRay.origin.y, pzRay.origin.z);
    rayDirection = math::normalize(Vec3f(pzRay.dir.x, pzRay.dir.y, pzRay.dir.z));
    const Vec3f rayDirectionX = math::normalize(Vec3f(pzRay.dDdx.x, pzRay.dDdx.y, pzRay.dDdx.z) / res.x);
    const Vec3f rayDirectionY = math::normalize(Vec3f(pzRay.dDdy.x, pzRay.dDdy.y, pzRay.dDdy.z) / res.y);

    float minZ, maxZ;
    bool isVoxtrum = false;
    PresenZ::Camera::PzGetRayMinMaxZ(x, y, isVoxtrum, minZ, maxZ);
    float nearClip = getNear();
    if (isVoxtrum) {
        nearClip = 0.0001f;
    }

    if (!pzRay.isValid()) {
        return;
    }


    *dstRay = mcrt_common::RayDifferential(
        rayOrigin, rayDirection,
        rayOrigin, rayDirectionX,
        rayOrigin, rayDirectionY,
        nearClip + pzRay.minZ,
        getFar(),
        time, 0);

}

} // namespace pbr
} // namespace moonray
