// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// @file TestLightUtil.ispc
//

#include <moonray/rendering/pbr/light/LightUtil.isph>

//----------------------------------------------------------------------------

inline uniform bool
assertIsEqual(float a, float b)
{
    bool equal = isEqual(a, b, 1.0e-5f);
    if (all(equal)) {
        print("% ==\n%\n", a, b);
        return true;
    } else {
        print("Error:\n% !=\n%\n", a, b);
        return false;
    }
}


//----------------------------------------------------------------------------

static uniform bool
testUniformPlaneDistance(const uniform Vec3f &o, const uniform Vec3f &n, float l)
{
    uniform Plane plane;
    Plane_init(&plane, o, n);

    Vec3f p = o + n * l;
    float d = Plane_getDistance(&plane, p);
    return assertIsEqual(d, l);
}

static uniform bool
testVaryingPlaneDistance(const Vec3f &n, const uniform Vec3f &p, float l)
{
    Vec3f o = p - n * l;
    Plane plane;
    Plane_init(&plane, o, n);

    float d = Plane_getDistance(&plane, p);
    return assertIsEqual(d, l);
}

static uniform bool
testUniformPlaneDistance(const uniform Vec3f &o, const uniform Vec3f &n,
        float t1, float t2, float l)
{
    uniform Plane plane;
    Plane_init(&plane, o, n);
    uniform Vec3f T1 = cross(n, Vec3f_ctor(1.0f, 0.0f, 0.0f));
    uniform Vec3f T2 = cross(n, Vec3f_ctor(0.0f, 0.0f, 1.0f));

    Vec3f p = o + T1 * t1 + T2 * t2 + n * l;
    float d = Plane_getDistance(&plane, p);
    return assertIsEqual(d, l);
}

static uniform bool
testVaryingPlaneDistance(const Vec3f &n, const uniform Vec3f &p,
        float t1, float t2, float l)
{
    Vec3f T1 = cross(n, Vec3f_ctor(1.0f, 0.0f, 0.0f));
    Vec3f T2 = cross(n, Vec3f_ctor(0.0f, 0.0f, 1.0f));
    Vec3f o = p - T1 * t1 - T2 * t2 - n * l;
    Plane plane;
    Plane_init(&plane, o, n);

    float d = Plane_getDistance(&plane, p);
    return assertIsEqual(d, l);
}


export uniform bool
testPlane()
{
    uniform bool result = true;

    // range = [-1.0, ..., 1.0]
    float range = (float)programIndex / (programCount - 1) * 2.0f - 1.0f;

    // Uniform plane, varying points
    uniform Vec3f ou = Vec3f_ctor(1.0f, 1.0f, 1.0f);
    uniform Vec3f nu = normalize(Vec3f_ctor(1.0f, 1.0f, 1.0f));
    float l = range * 1.234f;
    result &= testUniformPlaneDistance(ou, nu, l);
    l = range * 3.456789f;
    result &= testUniformPlaneDistance(ou, nu, range * 1.629, range * -3.263, l);
    ou = Vec3f_ctor(3.2f, -61.5f, 18.0f);
    nu = normalize(Vec3f_ctor(3.0f, 2.0f, 1.0f));
    l = range * -4.321f;
    result &= testUniformPlaneDistance(ou, nu, l);
    l = range * 6.543f;
    result &= testUniformPlaneDistance(ou, nu, l);

    // Varying plane, uniform point
    uniform Vec3f pu = Vec3f_ctor(12.0f, 6.0f, 3.0f);
    Vec3f n = normalize(range * Vec3f_ctor(1.0f, 2.0f, 3.0f));
    l = range * 1.234f;
    result &= testVaryingPlaneDistance(n, pu, l);
    l = range * 3.456789f;
    result &= testVaryingPlaneDistance(n, pu, range * 1.629, range * -3.263, l);
    pu = Vec3f_ctor(3.2f, -61.5f, 18.0f);
    n = normalize(range * Vec3f_ctor(3.0f, 2.0f, 1.0f));
    l = range * -4.321f;
    result &= testVaryingPlaneDistance(n, pu, l);
    l = range * 6.543f;
    result &= testVaryingPlaneDistance(n, pu, l);

    return result;
}


//----------------------------------------------------------------------------

// Samples the falloff curve, and checks that the results are equal to the
// given samples. Returns true if all samples are the same, and false otherwise.
// In any case sets the samples array with the values computed herein.
export uniform bool
testFalloffCurve(const uniform FalloffCurve * uniform curve,
        uniform float * uniform samples, uniform int sampleCount)
{
    bool equal = true;

    foreach (i = 0 ... sampleCount) {
        float x = (float)i / (sampleCount - 1);
        float y = FalloffCurve_eval(curve, x);

        equal &= isEqual(samples[i], y);
        samples[i] = y;
    }

    return all(equal);
}

//----------------------------------------------------------------------------

