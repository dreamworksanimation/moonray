// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// @file TestLightUtil.cc
//

#include "TestLightUtil.h"
#include "TestUtil.h"
#include "TestLightUtil_ispc_stubs.h"

#include <moonray/rendering/pbr/light/LightUtil.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Vec3.h>


namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;


static const int sSampleCount = 8;


//----------------------------------------------------------------------------

static void
testPlaneDistance(const Vec3f &o, const Vec3f &n, float l)
{
    Plane plane(o, n);

    Vec3f p = o + n * l;
    float d = plane.getDistance(p);
    testAssert(isEqual(d, l), "%f != %f", d, l);
}


static void
testPlaneDistance(const Vec3f &o, const Vec3f &n, float t1, float t2, float l, float eps=epsilon)
{
    Plane plane(o, n);
    Vec3f T1 = cross(n, Vec3f(1.0f, 0.0f, 0.0f));
    Vec3f T2 = cross(n, Vec3f(0.0f, 0.0f, 1.0f));

    Vec3f p = o + T1 * t1 + T2 * t2 + n * l;
    float d = plane.getDistance(p);
    testAssert(isEqual(d, l, eps), "%f != %f", d, l);
}


void
TestLightUtil::testPlane()
{
    fprintf(stderr, "#### testPlane ####\n");

    Vec3f o = Vec3f(1.0f, 1.0f, 1.0f);
    Vec3f n = normalize(Vec3f(1.0f, 1.0f, 1.0f));
    float l = -1.234f;
    testPlaneDistance(o, n, l);
    testPlaneDistance(o, n, 1.629, -3.263, l);
    l = 3.456789f;
    testPlaneDistance(o, n, l);
    testPlaneDistance(o, n, -123.6279, 321.2264, l, 3e-6f);

    o = Vec3f(3.2f, -61.5f, 18.0f);
    n = normalize(Vec3f(3.0f, 2.0f, 1.0f));
    l = -4.321f;
    testPlaneDistance(o, n, l);
    l = 6.543f;
    testPlaneDistance(o, n, l);

    CPPUNIT_ASSERT(asCppBool(ispc::testPlane()));
}


//----------------------------------------------------------------------------

static void
printFalloffCurveSamples(FalloffCurveType type, float samples[sSampleCount])
{
    fprintf(stderr, "FalloffCurve type %d = \n    [", type);
    for (int i=0; i < sSampleCount; i++) {
        fprintf(stderr, "%f", samples[i]);
        if (i < sSampleCount - 1) {
            fprintf(stderr, ", ");
        } else {
            fprintf(stderr, "]\n");
        }
    }
}


static void
computeFalloffCurveSamples(const FalloffCurve &fc, float samples[sSampleCount])
{
    for (int i=0; i < sSampleCount; i++) {
        float x = i / float(sSampleCount - 1);
        samples[i] = fc.eval(x);
    }
}


static void
testFalloffCurveType(FalloffCurveType type, ispc::FalloffCurveType ispcType)
{
    CACHE_ALIGN float samples[sSampleCount];
    bool equal;

    FalloffCurve fc(type);
    computeFalloffCurveSamples(fc, samples);
    printFalloffCurveSamples(type, samples);
    equal = asCppBool(ispc::testFalloffCurve(fc.asIspc(), samples, sSampleCount));
    if (!equal) {
        fprintf(stderr, "Error: ISPC samples are different:\n");
        printFalloffCurveSamples(type, samples);
    }
    CPPUNIT_ASSERT(equal);
}


void
TestLightUtil::testFalloffCurve()
{
    testFalloffCurveType(FALLOFF_CURVE_TYPE_NONE,
                   ispc::FALLOFF_CURVE_TYPE_NONE);
    testFalloffCurveType(FALLOFF_CURVE_TYPE_LINEAR,
                   ispc::FALLOFF_CURVE_TYPE_LINEAR);
    testFalloffCurveType(FALLOFF_CURVE_TYPE_EASEIN,
                   ispc::FALLOFF_CURVE_TYPE_EASEIN);
    testFalloffCurveType(FALLOFF_CURVE_TYPE_EASEOUT,
                   ispc::FALLOFF_CURVE_TYPE_EASEOUT);
    testFalloffCurveType(FALLOFF_CURVE_TYPE_EASEINOUT,
                   ispc::FALLOFF_CURVE_TYPE_EASEINOUT);
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestLightUtil);

