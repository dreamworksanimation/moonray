// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightFilter.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/shading/ispc/Shading.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(BarnDoorLightFilter);

export uniform uint32_t
BarnDoorLightFilter_hudValidation(uniform bool verbose)
{
    BARN_DOOR_LIGHT_FILTER_VALIDATION;
}

varying Xform3f
BarnDoorLightFilter_getSlerpXformRender2Local(const uniform BarnDoorLightFilter * uniform bdlif,
                                              varying float time)
{
    // The result of applying the inverse of Local2Render to a point p is
    // (p + trans) * rot * scale, where trans = -position,
    // rot = render2localRot.l, and scale = render2localScale.
    const Vec3f trans = (bdlif->mMb & LIGHTFILTER_MB_TRANSLATION) ?
        neg(lerp(bdlif->mPosition[0], bdlif->mPosition[1], time)) :
        neg(bdlif->mPosition[0]);
    const Mat3f rot = (bdlif->mMb & LIGHTFILTER_MB_ROTATION) ?
        transpose(Mat3f_ctor(slerp(bdlif->mOrientation[0], bdlif->mOrientation[1], time))) : // inverse is transpose
        bdlif->mRender2LocalRot0.l;
    const Mat3f scale = (bdlif->mMb & LIGHTFILTER_MB_SCALE) ?
        inverse(lerp(bdlif->mLocal2RenderScale[0], bdlif->mLocal2RenderScale[1], time)) :
        bdlif->mRender2LocalScale0;

    return Xform3f_ctor(Mat3f_identity(), trans) * Xform3f_ctor(rot * scale, Vec3f_ctor(0.f));
}

inline varying Xform3f
BarnDoorLightFilter_getXformRender2Local(const uniform BarnDoorLightFilter * uniform bdlif,
                                         varying float time)
{
    return (bdlif->mMb != 0) ? 
        BarnDoorLightFilter_getSlerpXformRender2Local(bdlif, time) : bdlif->mRender2Local0;
}

//----------------------------------------------------------------------------

varying bool
BarnDoorLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
                                  const varying CanIlluminateData &data)
{
    const uniform BarnDoorLightFilter * uniform lf = (const uniform BarnDoorLightFilter * uniform)lif;

    //  get render to projector space xform
    Xform3f r2l = lf->mUseLightXform ?
        data.lightRender2LocalXform : BarnDoorLightFilter_getXformRender2Local(lf, data.time);

    // shading point in projector space
    Vec3f pt = transformPoint(r2l, data.shadingPointPosition);

    // z distance from projector
    float shadingDist = pt.z;

    // No illumination behind barn
    if (shadingDist < 0.f) {
        return false;
    }

    if (shadingDist < lf->mPreBarnDist) {
        // The shading point is behind the projector, we don't need to evaluate the
        // map shader.
        switch (lf->mPreBarnMode) {
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
float BarnDoorLightFilter_gaussianCDF(varying float x) {

    // We choose the variance = sigma^2 = 0.1 to make the PDF approach zero
    // for |x|>1 so that the blur radius is 1.
    const float variance = 0.1f;
    return 0.5f * (1.f + erf( x / ( sqrt(variance) * sqrt(2.f) ) ) );
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
float BarnDoorLightFilter_gaussianBlur(varying float t) {

    // Remap input from [0, 1] to [1, -1] for input to the CDF
    // We turn the CDF around so it decreases (instead of increases)
    // and scale it horizontally to fit our desired domain.
    t = 1.f - 2.f * t;

    // Evaluate the CDF
    float val = BarnDoorLightFilter_gaussianCDF(t);

    // Scale and offset the CDF so values at output to expand its range
    // from ~[0.0008, 0.9992] to [0,1]
    // (I don't know if ISPC will compile minVal to a constant, 
    //  so I've made it a literal constant until we can be sure.)
    float minVal = 0.0007862747f; //BarnDoorLightFilter_gaussianCDF(-1.f);
    return val * (1.f + 2.f * minVal) - minVal;
}

void
BarnDoorLightFilter_eval(const uniform LightFilter * uniform lif,
                         const varying EvalData& data,
                         varying Color * uniform filterValue)
{
    const uniform BarnDoorLightFilter * uniform lf = (const uniform BarnDoorLightFilter * uniform)lif;

    // get render to projector space xform
    Xform3f r2l = lf->mUseLightXform ?
        data.lightRender2LocalXform : BarnDoorLightFilter_getXformRender2Local(lf, data.time);

    // shading point in projector space
    Vec3f pt = transformPoint(r2l, data.shadingPointPosition);

    // z distance from projector
    float shadingDist = pt.z;

    if (shadingDist < lf->mPreBarnDist) {
        // The shading point is behind the projector, we don't need to evaluate the
        // map shader.
        switch (lf->mPreBarnMode) {
        case BLACK:
            *filterValue = sBlack;
            return;
        case WHITE:
            *filterValue = sWhite;
            return;
        case DEFAULT:
        default:
            break;     // Do nothing
        }
    }

    if (lf->mMode == PHYSICAL) {

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
        varying float t = (lf->mFocalDist - shadingDist) / wi.z;
        pt = pt + t * wi;
    }

    // We need the screen space position of the render space point.
    Vec4f sppl = { pt.x, pt.y, pt.z, 1.f };
    Vec3f screenP = transformH(lf->mProjector2Screen, sppl); // projection using a homogenous divide

    // Convert to 2D
    Vec2f shadingPoint = {screenP.x, screenP.y};

    // Find point of inner rectangle closest to shading point
    Vec2f clampedShadingPoint = clamp(shadingPoint, lf->mMinCorner, lf->mMaxCorner);

    // Take the difference
    Vec2f diffVec = shadingPoint - clampedShadingPoint;

    // Choose the scales for the relevant corner
    Vec2f reciprocalScales =
        {lf->mReciprocalEdgeScales[diffVec.x < 0.f ? BARNDOOR_EDGE_LEFT   : BARNDOOR_EDGE_RIGHT],
         lf->mReciprocalEdgeScales[diffVec.y < 0.f ? BARNDOOR_EDGE_BOTTOM : BARNDOOR_EDGE_TOP]  };

    // Inversely scale it according to ellipse dimensions
    Vec2f scaledVec = diffVec * reciprocalScales;  // shadingPoint in frame of a unit circle

    // Compute the 3 distances we need:

    // Distance from the inner box to the first (inner)
    // rounded-rectangle-with-circular-corners (the edge start).
    float r0 = lf->mRadius;

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
    float scale = 0.f;       // Exterior, r >= r1. 0 means no light allowed.
    if (r <= r0) {           // Interior
        scale = 1.f;
    } else if (r < r1) {     // Edge transition
        float lerpParam = (r - r0) / (r1 - r0);
        scale = BarnDoorLightFilter_gaussianBlur(lerpParam);
    }

    // invert
    if (lf->mInvert) {
        scale = 1.f - scale;
    }

    // apply color
    *filterValue = lf->mColor * scale;

    // apply density (lerp)
    *filterValue = lerp(sWhite, *filterValue, lf->mDensity);
}

//----------------------------------------------------------------------------

export void
BarnDoorLightFilter_init(uniform BarnDoorLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter,
            BarnDoorLightFilter_canIlluminate, BarnDoorLightFilter_eval);
}

//----------------------------------------------------------------------------

