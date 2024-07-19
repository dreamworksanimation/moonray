// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "SphericalCamera.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

bool SphericalCamera::sAttributeKeyInitialized = false;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sMinLatitudeKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sMaxLatitudeKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sLatitudeZoomOffsetKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sMinLongitudeKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sMaxLongitudeKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sLongitudeZoomOffsetKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sFocalKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  SphericalCamera::sInsideOutKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> SphericalCamera::sOffsetRadiusKey;


SphericalCamera::SphericalCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
    Camera(rdlCamera),
    mInsideOut(false),
    mOffsetRadius(0.0f)
{
    initAttributeKeys(rdlCamera->getSceneClass());
}

void SphericalCamera::initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE
    sAttributeKeyInitialized = true;
    sMinLatitudeKey         = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("min_latitude");
    sMaxLatitudeKey         = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("max_latitude");
    sLatitudeZoomOffsetKey  = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("latitude_zoom_offset");
    sMinLongitudeKey        = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("min_longitude");
    sMaxLongitudeKey        = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("max_longitude");
    sLongitudeZoomOffsetKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("longitude_zoom_offset");
    sFocalKey               = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("focal");
    sInsideOutKey           = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool >("inside_out");
    sOffsetRadiusKey        = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("offset_radius");
    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

bool SphericalCamera::getIsDofEnabledImpl() const
{
    return false;
}

void SphericalCamera::updateImpl(const Mat4d& world2render)
{
    float focal_length = getRdlCamera()->get(sFocalKey);
    float zoom = 30.0f / focal_length;   // ratio vs the default value

    float thetaMin = (sPi / 180.0f) * getRdlCamera()->get(sMinLatitudeKey);
    float thetaMax = (sPi / 180.0f) * getRdlCamera()->get(sMaxLatitudeKey);
    float thetaOfs = (sPi / 180.0f) * getRdlCamera()->get(sLatitudeZoomOffsetKey);
    float thetaMid = 0.5f * (thetaMin + thetaMax) + thetaOfs;

    mThetaScale  = zoom * (thetaMax - thetaMin) / getApertureWindowHeight();
    mThetaOffset = lerp(thetaMid, thetaMin, zoom);

    float phiMin = (sPi / 180.0f) * getRdlCamera()->get(sMinLongitudeKey);
    float phiMax = (sPi / 180.0f) * getRdlCamera()->get(sMaxLongitudeKey);

    // Maintain backwards compatibility:
    // Lecacy SphericalCamera points down the negative x-axis (in the sense that the negative x-direction appears in
    // the middle of the rendered map); current gen SphericalCamera points down the negative z-axis to match the other
    // camera types. If a legacy sphere is indicated (by the default latitude & longitude ranges), we assume we're
    // using the old convention and add 90 degrees to the min & max longitudes to compensate
    if ((getRdlCamera()->get(sMinLatitudeKey)  ==  -90.0f) &&
        (getRdlCamera()->get(sMaxLatitudeKey)  ==   90.0f) &&
        (getRdlCamera()->get(sMinLongitudeKey) == -180.0f) &&
        (getRdlCamera()->get(sMaxLongitudeKey) ==  180.0f)) {

        phiMin = -3.0f * sHalfPi;
        phiMax = sHalfPi;
    }

    float phiOfs = (sPi / 180.0f) * getRdlCamera()->get(sLongitudeZoomOffsetKey);
    float phiMid = 0.5f * (phiMin + phiMax) + phiOfs;

    mPhiScale  = zoom * (phiMax - phiMin) / getApertureWindowWidth();
    mPhiOffset = lerp(phiMid, phiMin, zoom);

    mInsideOut    = getRdlCamera()->get(sInsideOutKey);
    mOffsetRadius = getRdlCamera()->get(sOffsetRadiusKey);
}

void SphericalCamera::createRayImpl(mcrt_common::RayDifferential* dstRay,
                                    float x,
                                    float y,
                                    float time,
                                    float /*lensU*/,
                                    float /*lensV*/) const
{
    // Compute transforms
    Mat4f ct2render;    // "camera space at ray time" --> render space
    if (getMotionBlur()) {
        ct2render = computeCamera2Render(time);
    } else {
        time = 0.0f;
        ct2render = getCamera2Render();
    }

    const Vec3f org    = transformPoint(ct2render, Vec3f(0, 0, 0));
    const Vec3f dir    = transformVector(ct2render, createDirection(x, y));
    const Vec3f dir_dx = transformVector(ct2render, createDirection(x+1.0f, y));
    const Vec3f dir_dy = transformVector(ct2render, createDirection(x, y+1.0f));

    if (mInsideOut) {
        const Vec3f offset_org = org + mOffsetRadius * dir;
        *dstRay = mcrt_common::RayDifferential(
            offset_org, -dir,
            offset_org, -dir_dx,
            offset_org, -dir_dy,
            getNear(), getFar(), time, 0);
    } else {
        *dstRay = mcrt_common::RayDifferential(
            org, dir,
            org, dir_dx,
            org, dir_dy,
            getNear(), getFar(), time, 0);
    }
}

Vec3f SphericalCamera::createDirection(float x, float y) const
{
    // Compute spherical polar coords corresponding to pixel coords (x,y)
    // (theta is latitude, phi is longitude)
    const float theta = mThetaScale * y + mThetaOffset;
    const float phi   = mPhiScale   * x + mPhiOffset;

    float sinTheta, cosTheta;
    float sinPhi, cosPhi;

    sincos(theta, &sinTheta, &cosTheta);
    sincos(phi, &sinPhi, &cosPhi);

    // Generate unit vector from polar coords.
    // Note that when theta = phi = 0, the vector points down the negative z-axis
    return Vec3f(cosTheta * sinPhi, sinTheta, cosTheta * -cosPhi);
}

} // namespace pbr
} // namespace moonray

