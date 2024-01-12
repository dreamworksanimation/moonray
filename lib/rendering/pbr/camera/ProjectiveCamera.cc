// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"
#include "ProjectiveCamera.h"
#include <moonray/rendering/pbr/core/Distribution.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

bool                            ProjectiveCamera::sAttributeKeyInitialized = false;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  ProjectiveCamera::sDofKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> ProjectiveCamera::sDofApertureKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> ProjectiveCamera::sDofFocusDistance;

scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> ProjectiveCamera::sHorizontalFilmOffset;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> ProjectiveCamera::sVerticalFilmOffset;

ProjectiveCamera::ProjectiveCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
    Camera(rdlCamera),
    mWindow{0.0f, 0.0f, 0.0f, 0.0f},
    mRenderRegion{0.0f, 0.0f, 0.0f, 0.0f},
    mDof(false),
    mDofLensRadius(0.0f),
    mDofFocusDistance(0.0f),
    mLensDistribution(getRdlCamera())
{
    initAttributeKeys(rdlCamera->getSceneClass());
}

void ProjectiveCamera::initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sHorizontalFilmOffset = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("horizontal_film_offset");
    sVerticalFilmOffset = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("vertical_film_offset");

    sDofKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool>("dof");
    sDofApertureKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("dof_aperture");
    sDofFocusDistance = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("dof_focus_distance");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

void
ProjectiveCamera::updateImpl(const Mat4d &world2render)
{
    // aperture window: derive from aperture viewport
    float invAspectRatio = getApertureWindowHeight() / getApertureWindowWidth();
    mWindow[0] = -1.0f;
    mWindow[1] = -invAspectRatio;
    mWindow[2] = 1.0f;
    mWindow[3] = invAspectRatio;

    // region window: derived from region viewport
    float offsetX = 2 * getRegionToApertureOffsetX() / getApertureWindowWidth();
    float offsetY = 2 * getRegionToApertureOffsetY() / getApertureWindowHeight();
    float widthRatio = 2 * getRegionWindowWidth() / getApertureWindowWidth();
    float heightRatio = 2 * getRegionWindowHeight() / getApertureWindowHeight();
    mRenderRegion[0] = -1.0f + offsetX;
    mRenderRegion[1] = invAspectRatio * (-1.0f + offsetY);
    mRenderRegion[2] = -1.0f + offsetX + widthRatio;
    mRenderRegion[3] = invAspectRatio * (-1.0f + offsetY + heightRatio);

    // Compute screen space to raster space (s2r)
    float zoom[2];
    zoom[0] = getApertureWindowWidth() * 0.5f;
    zoom[1] = getApertureWindowHeight() * 0.5f;
    float offset[2];
    offset[0] = zoom[0];
    offset[1] = zoom[1];
    Mat4f s2r = Mat4f(one);
    s2r[0][0] = zoom[0];
    s2r[1][1] = zoom[1];
    s2r[3][0] = offset[0];
    s2r[3][1] = offset[1];

    // Compute mC0toS
    mC0toS = computeC2S(0.0f);

    // Compute mC1toS
    mC1toS = computeC2S(1.0f);

    // Compute mRtoC0
    Mat4f c0toR = mC0toS * s2r;
    mRtoC0 = c0toR.inverse();

    // Compute mRtoC1
    Mat4f c1toR = mC1toS * s2r;
    mRtoC1 = c1toR.inverse();

    // Depth-of-field settings
    // TODO: check focus node attribute
    const scene_rdl2::rdl2::SceneVariables& vars =
        getRdlCamera()->getSceneClass().getSceneContext()->getSceneVariables();

    mDof = vars.get(scene_rdl2::rdl2::SceneVariables::sEnableDof) && getRdlCamera()->get(sDofKey);

    if (mDof) {
        const float focalLength = getFocalDistance();
        const float fstop       = getRdlCamera()->get(sDofApertureKey);

        // One unit in world space == sceneScale meters
        // Formally:
        //     world space unit * sceneScale = meter
        //     world space unit = meter / sceneScale
        //     world space unit = millimeter / (1000.0f * sceneScale)
        const float worldToMm   = vars.get(scene_rdl2::rdl2::SceneVariables::sSceneScaleKey) * 1000.0f;

        // Convert fstop ratio to entrance pupil diameter in mm (fstop equation):
        const float entrancePupilDiameter = (focalLength / fstop);

        // Transform entrance pupil diameter to world-space radius:
        mDofLensRadius = (entrancePupilDiameter / 2.0f) / worldToMm;
        mDofFocusDistance = getRdlCamera()->get(sDofFocusDistance);

        mLensDistribution.update();
    } else {
        mDofLensRadius = 0.0f;
        mDofFocusDistance = 0.0f;
    }

}

Mat4f ProjectiveCamera::computeRaster2Camera(float time) const
{
    if (getMotionBlur()) {
        return lerp(mRtoC0, mRtoC1, time);
    } else {
        return mRtoC0;
    }
}

Mat4f ProjectiveCamera::computeRaster2Render(float time,
                                             const Mat4f& ct2render) const
{
    // "raster space at ray time" --> render space
    if (getMotionBlur()) {
        // Lerp camera projection.
        Mat4f rt2ct = lerp(mRtoC0, mRtoC1, time);
        // Then apply camera to render transform
        return rt2ct * ct2render;
    } else {
        return mRtoC0 * getCamera2Render();
    }
}

Mat4f
ProjectiveCamera::computeCamera2Screen(float time) const
{
    if (getMotionBlur()) {
        // Lerp camera projection
        return lerp(mC0toS, mC1toS, time);
    } else {
        return mC0toS;
    }
}

void
ProjectiveCamera::createRayImpl(mcrt_common::RayDifferential *dstRay, float x, float y,
                                float time, float lensU, float lensV) const
{ 
    Vec3f Pr(x, y, -1.0f);

    const bool doDof = getIsDofEnabled();

    if (doDof) {
        mLensDistribution.sampleLens(lensU, lensV);
    }

    const float lensX = lensU * mDofLensRadius;
    const float lensY = lensV * mDofLensRadius;

    if (doDof) {
        createDOFRay(dstRay, Pr, lensX, lensY, time);
    } else {
        createSimpleRay(dstRay, Pr, time);
    }

}

float
ProjectiveCamera::computeZDistanceImpl(const Vec3f &p, const Vec3f &o, float time) const
{
    // This result is typically thought of as the "depth" value used in
    // depth maps.  The ray origin is not needed as the camera is
    // the origin of all camera spaces.  Transform p from render space
    // to the camera space at time t, and return the negative z component
    // of that result.
    const Mat4f render2camera = computeRender2Camera(time);
    const float cpz = dot(Vec4f(p.x, p.y, p.z, 1.f), render2camera.col2());
    return -cpz;
}

} // namespace pbr
} // namespace moonray


