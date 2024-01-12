// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"

#include <scene_rdl2/common/math/ispc/Quaternion.isph>
#include <scene_rdl2/common/math/ispc/Vec3.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>



//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(Light);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(LocalParamLight);


export uniform uint32_t
Light_hudValidation(uniform bool verbose)
{
    LIGHT_VALIDATION;
}

export uniform uint32_t
LocalParamLight_hudValidation(uniform bool verbose)
{
    LOCAL_PARAM_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

varying Vec3f
Light_lerpPosition(const uniform Light * uniform light, varying float time)
{
    if (!(light->mMb & LIGHT_MB_TRANSLATION)) return light->mPosition[0];

    return lerp(light->mPosition[0], light->mPosition[1], time);
}

varying Vec3f
Light_slerpDirection(const uniform Light * uniform light, varying float time)
{
    if (!(light->mMb & LIGHT_MB_TRANSLATION)) return light->mPosition[0];

    Quaternion3f q = slerp(light->mOrientation[0], light->mOrientation[1], time);
    // row 2 of Mat3(q) - see Mat3.h
    return Vec3f_ctor(2.0f * (q.i * q.k + q.r * q.j),
                      2.0f * (q.j * q.k - q.r * q.i),
                      1.0 - 2.0f * (q.i * q.i + q.j * q.j));
}

// input parameters contain uniform p
varying Vec3f
LocalParamLight_slerpPointLocal2Render(const uniform LocalParamLight * uniform light,
                                       const uniform Vec3f &p, varying float time)
{
    // transformPoint(Xform3f x, Vec3f p) is  p * x.l + x.t
    const float s = (light->mMb & LIGHT_MB_SCALE) ?
        lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time) :
        light->mLocal2RenderScale[0];
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time)) :
        light->mLocal2RenderRot[0].l;
    const Vec3f t = (light->mMb & LIGHT_MB_TRANSLATION) ?
        lerp(light->mPosition[0], light->mPosition[1], time) :
        light->mPosition[0];
    
    return (p * s) * r + t;
}

// input parameters contain varying p
varying Vec3f
LocalParamLight_slerpPointLocal2Render(const uniform LocalParamLight * uniform light,
                                       const varying Vec3f &p, varying float time)
{
    // transformPoint(Xform3f x, Vec3f p) is  p * x.l + x.t
    const float s = (light->mMb & LIGHT_MB_SCALE) ?
        lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time) :
        light->mLocal2RenderScale[0];
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time)) :
        light->mLocal2RenderRot[0].l;
    const Vec3f t = (light->mMb & LIGHT_MB_TRANSLATION) ?
        lerp(light->mPosition[0], light->mPosition[1], time) :
        light->mPosition[0];
    
    return (p * s) * r + t;
}

varying Vec3f
LocalParamLight_slerpPointRender2Local(const uniform LocalParamLight * uniform light,
                                       const varying Vec3f &p, varying float time)
{
    // inverse of Local2Render is (p + t) * r * s
    // where t = -position, r = render2localRot.l, s = render2localScale
    const Vec3f t = (light->mMb & LIGHT_MB_TRANSLATION) ?
        -1.f * lerp(light->mPosition[0], light->mPosition[1], time) :
        -1.f * light->mPosition[0];
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION)?
        transpose(Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time))) : // inverse is transpose
        light->mRender2LocalRot[0].l;
    const float s = (light->mMb & LIGHT_MB_SCALE) ?
        rcp(lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time)) :
        light->mRender2LocalScale[0];

    return ((p + t) * r) * s;
}

varying Vec3f
LocalParamLight_slerpVectorLocal2Render(const uniform LocalParamLight * uniform light,
                                        const varying Vec3f &v, varying float time)
{
    // inverse of Local2Render is v * s * r
    // where r = render2localRot.l, s = render2localScale
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time)) :
        light->mLocal2RenderRot[0].l;
    const float s = (light->mMb & LIGHT_MB_SCALE) ?
        lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time) :
        light->mRender2LocalScale[0];

    return (v * s) * r;
}

varying Vec3f
LocalParamLight_slerpVectorRender2Local(const uniform LocalParamLight * uniform light,
                                        const varying Vec3f &v, varying float time)
{
    // inverse of Local2Render is v * r * s
    // where r = render2localRot.l, s = render2localScale
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        transpose(Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time))) : // inverse is transpose
        light->mRender2LocalRot[0].l;
    const float s = (light->mMb & LIGHT_MB_SCALE) ?
        rcp(lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time)) :
        light->mRender2LocalScale[0];

    return (v * r) * s;
}

varying Vec3f
LocalParamLight_slerpVectorLocal2RenderRot(const uniform LocalParamLight * uniform light,
                                           const varying Vec3f &v, varying float time)
{
    // transformVector(Xform3f x, Vec3f v) is v * x.l
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time)) :
        light->mLocal2RenderRot[0].l;

    return v * r;
}

varying Vec3f
LocalParamLight_slerpVectorRender2LocalRot(const uniform LocalParamLight * uniform light,
                                           const varying Vec3f &v, varying float time)
{
    // inverse of Local2RenderRot is v * r
    // where r = render2localRot
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        transpose(Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time))) : // inverse is transpose
        light->mRender2LocalRot[0].l;

    return v * r;
}

varying Vec3f
LocalParamLight_slerpNormalLocal2Render(const uniform LocalParamLight * uniform light,
                                        const varying Vec3f &n, varying float time)
{
    // note: transformNormal in the math library simply does M * n,
    // assuming it is passed the inverse of the desired transform.
    // xformNormalLocal2Render means to transform a normal from local
    // to render space.

    // transformNormal is r * s * n
    // where r is render2localRot and s is the render2localScale
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION) ?
        transpose(Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time))) : // inverse is transpose
        light->mRender2LocalRot[0].l;
    const float s = (light->mMb & LIGHT_MB_SCALE) ?
        rcp(lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time)) :
        light->mRender2LocalScale[0];

    return r * (s * n);
}

varying Vec3f
LocalParamLight_slerpNormalLocal2RenderRot(const uniform LocalParamLight * uniform light,
                                           const varying Vec3f &n, varying float time)
{
    // note: transformNormal in the math library simply does M * n,
    // assuming it is passed the inverse of the desired transform.
    // xformNormalLocal2RenderRot means to transform a normal from local
    // to render space orientation.

    // transformNormal is r * n
    // where r is the render2localRot
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION)?
        transpose(Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time))) : // inverse is transpose
        light->mRender2LocalRot[0].l;

    return r * n;
}

varying Vec3f
LocalParamLight_slerpNormalRender2LocalRot(const uniform LocalParamLight * uniform light,
                                           const varying Vec3f &n, varying float time)
{
    // note: transformNormal in the math library simply does M * n,
    // assuming it is passed the inverse of the desired transform.
    // xformNormalRender2LocalRot means to transform a normal from render
    // to local space orientation.

    // transformNormal is r * n
    // where r is the local2renderRot
    const Mat3f r = (light->mMb & LIGHT_MB_ROTATION)?
        Mat3f_ctor(slerp(light->mOrientation[0], light->mOrientation[1], time)) :
        light->mLocal2RenderRot[0].l;

    return r * n;
}

varying float
LocalParamLight_lerpLocal2RenderScale(const uniform LocalParamLight * uniform light,
                                      varying float s, varying float time)
{
    if (!(light->mMb & LIGHT_MB_SCALE)) return light->mLocal2RenderScale[0] * s;

    return lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time) * s;
}

varying float
LocalParamLight_lerpRender2LocalScale(const uniform LocalParamLight * uniform light,
                                      varying float s, varying float time)
{
    if (!(light->mMb & LIGHT_MB_SCALE)) return light->mRender2LocalScale[0] * s;

    return rcp(lerp(light->mLocal2RenderScale[0], light->mLocal2RenderScale[1], time)) * s;
}

//----------------------------------------------------------------------------


