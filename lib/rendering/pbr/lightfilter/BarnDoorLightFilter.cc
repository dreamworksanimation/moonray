// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "BarnDoorLightFilter.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/lightfilter/BarnDoorLightFilter_ispc_stubs.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool BarnDoorLightFilter::sAttributeKeyInitialized;
rdl2::AttributeKey<rdl2::Mat4d> BarnDoorLightFilter::sProjectorXformKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sProjectorWidthKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sProjectorHeightKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sSizeTopKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sSizeBottomKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sSizeLeftKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sSizeRightKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sProjectorFocalDistKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sRotationKey;
rdl2::AttributeKey<rdl2::Int> BarnDoorLightFilter::sProjectorTypeKey;
rdl2::AttributeKey<rdl2::Int> BarnDoorLightFilter::sModeKey;
rdl2::AttributeKey<rdl2::Bool> BarnDoorLightFilter::sUseLightXformKey;
rdl2::AttributeKey<rdl2::Int> BarnDoorLightFilter::sPreBarnModeKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sPreBarnDistKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sRadiusKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sEdgeKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sEdgeScaleTopKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sEdgeScaleBottomKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sEdgeScaleLeftKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sEdgeScaleRightKey;
rdl2::AttributeKey<rdl2::Float> BarnDoorLightFilter::sDensityKey;
rdl2::AttributeKey<rdl2::Bool> BarnDoorLightFilter::sInvertKey;
rdl2::AttributeKey<rdl2::Rgb> BarnDoorLightFilter::sColorKey;


HUD_VALIDATOR(BarnDoorLightFilter);

BarnDoorLightFilter::BarnDoorLightFilter(const rdl2::LightFilter* rdlLightFilter) :
    LightFilter(rdlLightFilter)
{
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::BarnDoorLightFilter_init((ispc::BarnDoorLightFilter *)this->asIspc());
}

void
BarnDoorLightFilter::initAttributeKeys(const rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sProjectorXformKey = sc.getAttributeKey<rdl2::Mat4d>("node_xform");
    sProjectorWidthKey = sc.getAttributeKey<rdl2::Float>("projector_width");
    sProjectorHeightKey = sc.getAttributeKey<rdl2::Float>("projector_height");
    sSizeTopKey = sc.getAttributeKey<rdl2::Float>("size_top");
    sSizeBottomKey = sc.getAttributeKey<rdl2::Float>("size_bottom");
    sSizeLeftKey = sc.getAttributeKey<rdl2::Float>("size_left");
    sSizeRightKey = sc.getAttributeKey<rdl2::Float>("size_right");
    sProjectorFocalDistKey = sc.getAttributeKey<rdl2::Float>("projector_focal_distance");
    sRotationKey = sc.getAttributeKey<rdl2::Float>("rotation");
    sProjectorTypeKey = sc.getAttributeKey<rdl2::Int>("projector_type");
    sModeKey = sc.getAttributeKey<rdl2::Int>("mode");
    sUseLightXformKey = sc.getAttributeKey<rdl2::Bool>("use_light_xform");
    sPreBarnModeKey = sc.getAttributeKey<rdl2::Int>("pre_barn_mode");
    sPreBarnDistKey = sc.getAttributeKey<rdl2::Float>("pre_barn_distance");
    sRadiusKey = sc.getAttributeKey<rdl2::Float>("radius");
    sEdgeKey = sc.getAttributeKey<rdl2::Float>("edge");
    sEdgeScaleTopKey = sc.getAttributeKey<rdl2::Float>("edge_scale_top");
    sEdgeScaleBottomKey = sc.getAttributeKey<rdl2::Float>("edge_scale_bottom");
    sEdgeScaleLeftKey = sc.getAttributeKey<rdl2::Float>("edge_scale_left");
    sEdgeScaleRightKey = sc.getAttributeKey<rdl2::Float>("edge_scale_right");
    sDensityKey = sc.getAttributeKey<rdl2::Float>("density");
    sInvertKey = sc.getAttributeKey<rdl2::Bool>("invert");
    sColorKey = sc.getAttributeKey<rdl2::Rgb>("color");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

Xform3f
BarnDoorLightFilter::getSlerpXformRender2Local(float time) const
{
    // The result of applying the inverse of Local2Render to a point p is
    // (p + trans) * rot * scale, where trans = -position,
    // rot = render2localRot.l, and scale = render2localScale.
    const Vec3f trans = (mMb & LIGHTFILTER_MB_TRANSLATION) ?
        -lerp(mPosition[0], mPosition[1], time) :
        -mPosition[0];
    const Mat3f rot = (mMb & LIGHTFILTER_MB_ROTATION) ?
        // transpose is inverse
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() :
        mRender2LocalRot0.l;
    const Mat3f scale = (mMb & LIGHTFILTER_MB_SCALE) ?
        // The inverse() call below, while unfortunate, is necessary.
        // Reversing the order of operations (and lerping precomputed inverses)
        // does not look correct.
        lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time).inverse() :
        mRender2LocalScale0;

    return Xform3f(one, trans) * Xform3f(rot * scale, zero);
}

bool
BarnDoorLightFilter::updateTransforms(const Mat4f &local2Render, int ti)
{
    // update transforms at time index (ti)
    // ti == 0 is at normalized rayTime = 0.f
    // ti == 1 is at normalized rayTime = 1.f;
    MNRY_ASSERT(mRdlLightFilter);

    Quaternion3f rot;
    Mat3f scale;
    Xform3f local2RenderXform = xform<Xform3f>(local2Render);
    decompose(local2RenderXform.l, scale, rot);
    
    mLocal2RenderScale[ti] = scale;

    // Single shutter-time items (for when there is no MB)
    if (ti == 0) {
        mRender2Local0 = local2RenderXform.inverse();
        mRender2LocalScale0 = scale.inverse();
        mRender2LocalRot0 = Mat3f(rot).transposed();
    }

    mPosition[ti] = asVec3(local2Render.row3());

    Xform3f local2RenderRot = Xform3f(Mat3f(rot), zero);
    mOrientation[ti] = Quaternion3f(local2RenderRot.row0(),
                                    local2RenderRot.row1(),
                                    local2RenderRot.row2());

    // This code relies on update() being called twice: first with ti == 0,
    // and then with ti == 1.
    //
    // After setting the second quaternion (ti==1), check if the angle from the
    // first quaternion is negative and flip accordingly.
    if (ti==1 && dot(mOrientation[0], mOrientation[1]) < 0) {
        mOrientation[1] *= -1.f;
    }

    return true;
}

bool
BarnDoorLightFilter::updateParamAndTransforms(const Mat4f &local2Render0,
                                              const Mat4f &local2Render1)
{
    if (!updateTransforms(local2Render0, /* ti = */ 0)) return false;

    // setup mMb
    mMb = LIGHTFILTER_MB_NONE;
    const scene_rdl2::rdl2::SceneVariables &vars =
        mRdlLightFilter->getSceneClass().getSceneContext()->getSceneVariables();
    const bool mb = vars.get(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur) &&
        !isEqual(local2Render0, local2Render1);

    if (mb) {
        if (!updateTransforms(local2Render1, /* ti = */ 1)) return false;

        // set specific mb bits
        if (!isEqual(mPosition[0], mPosition[1]))
            mMb |= LIGHTFILTER_MB_TRANSLATION;
        if (!isEqual(mOrientation[0], mOrientation[1]))
            mMb |= LIGHTFILTER_MB_ROTATION;
        if (!isEqual(mLocal2RenderScale[0], mLocal2RenderScale[1]))
            mMb |= LIGHTFILTER_MB_SCALE;
    }

    return true;
}

// Similar to CookieLightFilter::computePerspectiveProjectionMatrix()
math::Mat4f
BarnDoorLightFilter::getPerspectiveProjectionMatrix()
{
    // See Appendix C.5 of the SGI Graphics Library Programming Guide
    // for format of projection matrix.
    // http://irix7.com/techpubs/007-1702-020.pdf  (pg. 238)
    // where:
    //   left = bottom = -1
    //   right = top = near = 1
    //   far = inf
    // and Y and Z columns are negated for barnRotMat.
    // The Z column (3rd) doesn't matter since the Barn Door only concerns itself
    // with X and Y values after applying this matrix (see use of mProjector2Screen).
    return Mat4f(1.f,  0.f, 0.f, 0.f,
                 0.f, -1.f, 0.f, 0.f,
                 0.f,  0.f, 1.f, 1.f,
                 0.f,  0.f, 2.f, 0.f);
}

// Similar to CookieLightFilter::computeOrthoProjectionMatrix()
math::Mat4f
BarnDoorLightFilter::getOrthoProjectionMatrix()
{
    // See Appendix C.5 of the SGI Graphics Library Programming Guide
    // for format of projection matrix.
    // http://irix7.com/techpubs/007-1702-020.pdf  (pg. 238)
    // where:
    //   left = bottom = -1
    //   right = top = near = 1
    //   far = inf
    // and Y and Z columns are negated for barnRotMat.
    // The Z column (3rd) doesn't matter since the Barn Door only concerns itself
    // with X and Y values after applying this matrix (see use of mProjector2Screen).
    return Mat4f(1.f,  0.f, 0.f, 0.f,
                 0.f, -1.f, 0.f, 0.f,
                 0.f,  0.f, 0.f, 0.f,
                 0.f,  0.f, 1.f, 1.f);
}

void
BarnDoorLightFilter::update(const LightFilterMap& /*lightFilters*/,
                            const Mat4d& world2Render)
{
    if (!mRdlLightFilter) {
        return;
    }

    // Get flap-opening width (mm)
    const float inputFilmWidth = max(mRdlLightFilter->get(sProjectorWidthKey), 0.00001f);

    // Get flap-opening height (mm)
    const float inputFilmHeight = max(mRdlLightFilter->get(sProjectorHeightKey), 0.00001f);

    // Get per-edge size adjustments
    float sizes[4];
    sizes[BARNDOOR_EDGE_LEFT]   = mRdlLightFilter->get<rdl2::Float>(sSizeLeftKey);
    sizes[BARNDOOR_EDGE_BOTTOM] = mRdlLightFilter->get<rdl2::Float>(sSizeBottomKey);
    sizes[BARNDOOR_EDGE_RIGHT]  = mRdlLightFilter->get<rdl2::Float>(sSizeRightKey);
    sizes[BARNDOOR_EDGE_TOP]    = mRdlLightFilter->get<rdl2::Float>(sSizeTopKey);

    // Determine sizes
    float effectiveWidth  = inputFilmWidth  + sizes[BARNDOOR_EDGE_LEFT]   + sizes[BARNDOOR_EDGE_RIGHT];
    float effectiveHeight = inputFilmHeight + sizes[BARNDOOR_EDGE_BOTTOM] + sizes[BARNDOOR_EDGE_TOP];
    float smallestEdge = min(effectiveWidth, effectiveHeight);

    // Convert radius from relative ([0,1]) to absolute size
    float relativeRadius = mRdlLightFilter->get(sRadiusKey, 0.f);
    mRadius = relativeRadius * (smallestEdge * 0.5f);

    // Convert edge from relative ([0,1]) to absolute size
    float relativeEdge = mRdlLightFilter->get(sEdgeKey, 0.f);
    mEdge = relativeEdge * smallestEdge;

    // Compute inverse edge scales
    float edgeScaleLeft   = mRdlLightFilter->get<rdl2::Float>(sEdgeScaleLeftKey);
    float edgeScaleBottom = mRdlLightFilter->get<rdl2::Float>(sEdgeScaleBottomKey);
    float edgeScaleRight  = mRdlLightFilter->get<rdl2::Float>(sEdgeScaleRightKey);
    float edgeScaleTop    = mRdlLightFilter->get<rdl2::Float>(sEdgeScaleTopKey);

    mReciprocalEdgeScales[BARNDOOR_EDGE_LEFT]   = rcp(max(1e-9f, mRadius + mEdge * edgeScaleLeft));
    mReciprocalEdgeScales[BARNDOOR_EDGE_BOTTOM] = rcp(max(1e-9f, mRadius + mEdge * edgeScaleBottom));
    mReciprocalEdgeScales[BARNDOOR_EDGE_RIGHT]  = rcp(max(1e-9f, mRadius + mEdge * edgeScaleRight));
    mReciprocalEdgeScales[BARNDOOR_EDGE_TOP]    = rcp(max(1e-9f, mRadius + mEdge * edgeScaleTop));

    // Get focal length (mm). Will use as distance to projector plane
    mFocalDist = mRdlLightFilter->get(sProjectorFocalDistKey, 0.f);
    mFocalDist = max(mFocalDist, 0.1f);

    // Setup mProjector2Screen
    mProjector2Screen = Mat4f(math::one);
    switch (mRdlLightFilter->get(sProjectorTypeKey)) {
    case PERSPECTIVE:
        mProjector2Screen = getPerspectiveProjectionMatrix();
        break;
    case ORTHOGRAPHIC:
        mProjector2Screen = getOrthoProjectionMatrix();
        break;
    default:
        MNRY_ASSERT(false);
    }

    // Make radius inscribe box: reduce width and height by 2X radius
    Vec2f filmSize(effectiveWidth  - 2.f * mRadius,
                   effectiveHeight - 2.f * mRadius);

    // Calculate center
    Vec2f center = Vec2f(sizes[BARNDOOR_EDGE_RIGHT] - sizes[BARNDOOR_EDGE_LEFT  ],
                         sizes[BARNDOOR_EDGE_TOP  ] - sizes[BARNDOOR_EDGE_BOTTOM]) * 0.5f;

    // Precompute screen-space aperture corners (at virtual focal length of 1)
    mMinCorner = center - filmSize * 0.5f;
    mMaxCorner = center + filmSize * 0.5f;

    // A note about rotations
    //
    // Most lights (Spot, Disk, Rect, Sphere, Distant) have a 180 degree
    // rotation about X in their transform, called sRotateX180 which makes the
    // light spaces' +z direction point -z in render space. That rotation is
    // baked into the light's xform used by the barn door when
    // mUseLightXform==true.
    //
    // When the barn door is not bound to a light, we apply an equal rotation,
    // called barnRotMat below, instead. That way the barn door will point in
    // the same direction regardless of mUseLightXform if the light's and barn
    // door's node_xforms match.
    //
    // Unfortunately that means:
    //
    //    1) barnRotMat is applied to barn doors on lights that don't use a
    //       baked rotation, namely Cyl, Mesh, and Env, and the filter will not
    //       point in the same direction as those lights' mDirection. To re-aim
    //       the barn door in those cases, either rotate the light or unbind the
    //       filter and set its node_xform. I matched the bound/unbound barn
    //       door light filter rotations in the RaTS test by rotating the
    //       lights:
    //
    //           Cyl:                                           rotate(180,1,0,0)
    //           Mesh:                                          rotate(180,1,0,0)
    //           Env:  EnvLight::sLocalOrientation.inverse()  * rotate(180,1,0,0)
    //                 = rotate(-90,0,1,0) * rotate(90,1,0,0) * rotate(180,1,0,0)
    //                 = rotate(-90,0,1,0) * rotate(-90,1,0,0)
    //
    //           The Env rotation was found by solving this eqn:
    //
    //                 sLocalOrientation * EnvLight.xform = barnRotMat * BarnDoor.xform
    //
    //           for EnvLight.xform:
    //
    //                 EnvLight.xform = sLocalOrientation.inverse() * barnRotMat * BarnDoor.xform
    //
    //    2) The projection matrices had to be inverted in Y and Z directions
    //       (as noted in their comments) in order for the filter not to appear
    //       backwards or upside down.
    //

    // Apply twist rotation about the filter's direction
    float rotAngle = math::degreesToRadians(mRdlLightFilter->get<rdl2::Float>(sRotationKey));
    Mat4f twistRotMat = Mat4f::rotate(Vec3f(0, 0, 1), rotAngle);
    mProjector2Screen = twistRotMat * mProjector2Screen;

    // This should match Light::sRotateX180
    Mat4d barnRotMat = Mat4d::rotate(Vec3d(1, 0, 0), math::sPi);

    // local space is projector space
    Mat4d local2World0 = barnRotMat * mRdlLightFilter->get(sProjectorXformKey, 0.f);
    Mat4d local2World1 = barnRotMat * mRdlLightFilter->get(sProjectorXformKey, 1.f);

    Mat4f local2Render0 = toFloat(local2World0 * world2Render);
    Mat4f local2Render1 = toFloat(local2World1 * world2Render);

    if (!updateParamAndTransforms(local2Render0, local2Render1)) {
        return;
    }

    mMode = mRdlLightFilter->get<rdl2::Int>(sModeKey);
    mUseLightXform = mRdlLightFilter->get<rdl2::Bool>(sUseLightXformKey);
    mPreBarnMode = mRdlLightFilter->get<rdl2::Int>(sPreBarnModeKey);
    mPreBarnDist = mRdlLightFilter->get<rdl2::Float>(sPreBarnDistKey);
    mDensity = clamp(mRdlLightFilter->get<rdl2::Float>(sDensityKey), 0.f, 1.f);
    mInvert = mRdlLightFilter->get<rdl2::Bool>(sInvertKey);
    mColor =  mRdlLightFilter->get<rdl2::Rgb>(sColorKey);
}

bool
BarnDoorLightFilter::canIlluminate(const CanIlluminateData& data) const
{

    // compute barn intersection in projector space
    Xform3f r2l = mUseLightXform ? 
        data.lightRender2LocalXform : getXformRender2Local(data.time);

    // z distance from projector
    float shadingDist = transformPoint(r2l, data.shadingPointPosition).z;

    // No illumination behind barn
    if (shadingDist < 0.f) {
        return false;
    }

    if (shadingDist < mPreBarnDist) {
        // The shading point is behind the projector.
        switch (mPreBarnMode) {
        case BLACK:
            return false;
        case WHITE:
            return true;
        case DEFAULT:
        default:
            break;     // Do nothing
        }
    }

    return true;
}

namespace {   // functions for local use only

    // Compute a (Gaussian) blur fall off value
    //
    // Input: Distance t in the range [0,1]
    // Return: Fall off value, decreasing from 1 to 0 as t goes from 0 to 1.
    //
    // To calculate the blur value we convolve a step function f, where:
    //     f(x) = 1 for x <= 0 (the interior of the BarnDoor portal) and
    //     f(x) = 0 for x >  0 (the exterior of the BarnDoor portal)
    // with g, a blur kernel chosen to be the Gaussian normal distribution PDF.
    //
    // To evaluate the blur value we center the blur kernel
    // at x=t and integrate f(x)*g(x-t) where they overlap. Because f=1 where they
    // overlap, this amounts to just integrating g, which is given by its CDF.

    // The CDF of a normal function can be found here:
    // https://en.wikipedia.org/wiki/Normal_distribution
    float gaussianCDF(float x) {

        // We choose the variance = sigma^2 = 0.1 to make the PDF approach zero
        // for |x|>1 so that the blur radius is 1.
        const float variance = 0.1f;

        return 0.5f * (1.f + erf( x / ( sqrtf(variance) * sqrtf(2.f) ) ) );
    }

    // Unfortunately, the normal distribution has infinite extent which is
    // inconvenient because we want the blur to have finite radius.
    //
    // Therefore we clip the domain from [-inf,inf] to [-1,1]. That region
    // is finite and contains 99.8% of the area under the PDF, but not 100%.
    // As a result the CDF range is about [0.0008, 1-0.0008] instead of [0,1].
    //
    // Without compensation this will produce noticeable clipping at the beginning
    // and end of the blur region. To compensate we expand the CDF slightly.
    float gaussianBlur(float t) {

        // Remap input from [0, 1] to [1, -1] for input to the CDF
        // We turn the CDF around so it decreases (instead of increases)
        // and scale it horizontally to fit our desired domain.
        t = 1.f - 2.f * t;

        // Evaluate the CDF
        float val = gaussianCDF(t);

        // Scale and offset the CDF so values at output to expand its range
        // from ~[0.0008, 0.9992] to [0,1].
        // (gcc and icc compile minVal to a constant with -O2 or -O3)
        float minVal = gaussianCDF(-1.f);         // 0.000786
        return val * (1.f + 2.f * minVal) - minVal;
    }
}

Color
BarnDoorLightFilter::eval(const EvalData& data) const
{
    // get render to projector space xform
    Xform3f r2l = mUseLightXform ? 
        data.lightRender2LocalXform : getXformRender2Local(data.time);

    // shading point in projector space
    Vec3f pt = transformPoint(r2l, data.shadingPointPosition);

    // z distance from projector
    float shadingDist = pt.z;

    if (shadingDist < mPreBarnDist) {
        // The shading point is behind the projector, we don't need to evaluate the
        // barn door.
        switch (mPreBarnMode) {
        case BLACK:
            return sBlack;
        case WHITE:
            return sWhite;
        case DEFAULT:
        default:
            break;     // Do nothing
        }
    }

    // In analytic mode we use the projection of the shading point directly into screen space.

    // For physical mode we intersect the light ray with the focal plane,
    // then project that point into screen space.
    if (mMode == PHYSICAL) {

        // Find intersection of the light ray with the focal plane
        //
        // pt = data.shadingPointPosition
        // wi = direction of light sample from pt
        // po = projectorPos = (0,0,0)
        // pd = projectorDir = (0,0,1)
        // sv = shadingVec = pt - po
        // f = distance to from projector center to focal plane
        // X = intersection of wi with focal plane
        // t = distance along wi from pt to X
        //
        // Points along ray from pt in direction wi:
        // X = pt + t * wi
        //
        // Points in the projector focal plane:
        // dot(X - po, pd) = f
        //
        // Substitute ray equation into plane equation and solve for t:
        //          dot(pt + t * wi - po, pd) = f
        //          dot(pt - po + t * wi, pd) = f
        //               dot(sv + t * wi, pd) = f
        //  dot(sv, pd) + dot(t * wi, pd) - f = 0
        //                    dot(sv, pd) - f = -dot(t * wi, pd)
        //                    dot(sv, pd) - f = t * -dot(wi, pd)
        //   (dot(sv, pd) - f) / -dot(wi, pd) = t
        //   (shadingDist - f) / -wi.z = t
        //   (f - shadingDist) / wi.z = t
        //
        // Then plug t into ray equation to find intersection point X
        Vec3f wi = transformVector(r2l, data.wi);                   // light direction
        float t = (mFocalDist - shadingDist) / wi.z;
        pt = pt + t * wi;
        MNRY_ASSERT(isEqual(pt.z, mFocalDist));           // true if things went as planned
    }

    // We need the screen space position of the render space point.
    Vec3f screenP = transformH(mProjector2Screen, pt);   // projection using a homogenous divide

    // Convert to 2D
    Vec2f shadingPoint(screenP.x, screenP.y);

    // Find point of inner rectangle closest to shading point
    Vec2f clampedShadingPoint = clamp(shadingPoint, mMinCorner, mMaxCorner);

    // Take the difference
    Vec2f diffVec = shadingPoint - clampedShadingPoint;

    // Choose the scales for the relevant corner
    Vec2f reciprocalScales(
        mReciprocalEdgeScales[diffVec.x < 0.f ? BARNDOOR_EDGE_LEFT   : BARNDOOR_EDGE_RIGHT],
        mReciprocalEdgeScales[diffVec.y < 0.f ? BARNDOOR_EDGE_BOTTOM : BARNDOOR_EDGE_TOP  ]);

    // Inversely scale according to ellipse dimensions
    Vec2f scaledVec = diffVec * reciprocalScales;

    // Compute the 3 distances we need:

    // Distance from the inner box to the first (inner)
    // rounded-rectangle-with-circular-corners (the edge start).
    float r0 = mRadius;

    // Distance from the inner box to shading point.
    float r  = length(diffVec);

    // Distance from the inner box to the second (outer)
    // rounded-rectangle-with-ellipse-corners (the edge end).
    // The distance to the ellipse is related to r by a ratio. That
    // ratio can be computed in the frame of a unit circle, since we know the
    // distance to the circle in that frame. Also the ratio is preserved during
    // transformation since it's affine. Then we just apply the ratio to r.
    float r1 = r / length(scaledVec);

    // Compute scale value
    float scale = 0.f;            // Exterior, r >= r1. 0 means no light allowed.
    if (r <= r0) {                // Interior
        scale = 1.f;
    } else if (r < r1) {          // Edge transition
        float lerpParam = (r - r0) / (r1 - r0);
        scale = gaussianBlur(lerpParam);
    }

    // invert
    if (mInvert) {
        scale = 1.f - scale;
    }

    // apply color
    Color result = mColor * scale;

    // apply density (lerp)
    result = math::lerp(sWhite, result, mDensity);

    return result;
}

} //namespace pbr
} //namespace moonray

