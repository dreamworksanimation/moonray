// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TestLightTree.h"

#include <moonray/rendering/pbr/light/LightTreeUtil.h>
#include <moonray/rendering/pbr/light/LightTreeUtil.cc>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

static bool equal(const float& a, const float& b)
{
    return scene_rdl2::math::abs(a - b) < 0.000001f;
}

static bool conesAreEqual(const LightTreeCone& c1, const LightTreeCone& c2)
{
    return isEqual(c1.mAxis, c2.mAxis) && equal(c1.mCosThetaE, c2.mCosThetaE) &&
           equal(c1.mCosThetaO, c2.mCosThetaO) && equal(c1.mTwoSided, c2.mTwoSided);
}

static void printCone(const char* nameOfCone, const LightTreeCone& cone)
{
    std::cerr << "LightTreeCone " << nameOfCone << " = ( mAxis: (" << cone.mAxis.x << ", " << cone.mAxis.y << ", " 
              << cone.mAxis.z << "), mCosThetaO: " << cone.mCosThetaO << ", mCosThetaE: " << cone.mCosThetaE 
              << ", mTwoSided: " << cone.mTwoSided << ")\n";
}

void TestLightTree::testCone()
{
    fprintf(stderr, "=========================== Testing LightTree Cone ==============================\n");
    // If I combine identical cones, I should get back the same cone
    {
        std::cerr << "---- Combine identical cones to get the same cone ----\n";
        const LightTreeCone a{ Vec3f(0.f, 0.f, 1.f), /*cos(0rad)*/ 1.f, /*cos(pi/2rad)*/ 0.f, false };
        const LightTreeCone b{ a };
        const LightTreeCone result = combineCones(a, b);
        printCone("a", a);
        printCone("b", b);
        CPPUNIT_ASSERT(conesAreEqual(a, b));
        std::cerr << "LightTreeCone a and LightTreeCone b are equal\n";
        printCone("b", b);
        printCone("result", result);
        CPPUNIT_ASSERT(conesAreEqual(b, result));
        std::cerr << "LightTreeCone b and the resulting LightTreeCone are equal\n";
    }
    // If I combine an empty cone and a non-empty cone, I should get back the non-empty cone
    {
        std::cerr << "---- Combine an empty cone and non-empty cone, and get back the non-empty cone ----\n";
        const LightTreeCone emptyCone{};
        const LightTreeCone a{ Vec3f(0.f, 0.f, 1.f), /*cos(0rad)*/ 1.f, /*cos(pi/2rad)*/ 0.f, false };
        const LightTreeCone result = combineCones(a, emptyCone);
        printCone("empty", emptyCone);
        printCone("a", a);
        printCone("result", result);
        CPPUNIT_ASSERT(conesAreEqual(a, result));
    }
    // If one cone entirely contains the other cone, we should get back the larger cone, plus any extra emission
    {
        std::cerr << "---- Larger cone should subsume smaller cone ----\n";
        const LightTreeCone largerCone{ Vec3f(0.f, 0.f, 1.f), /*cos(pi/2rad)*/ 0.f, /*cos(0rad)*/ 1.f, false };
        const LightTreeCone smallerCone{ Vec3f(0.f, 0.f, 1.f), /*cos(0rad)*/ 1.f, /*cos(pi/2rad)*/ 0.f, false };
        const LightTreeCone result = combineCones(largerCone, smallerCone);
        printCone("larger", largerCone);
        printCone("smaller", smallerCone);
        printCone("result", result);
        const LightTreeCone expectedCone{ Vec3f(0.f, 0.f, 1.f), /*cos(pi/2rad)*/ 0.f, /*cos(pi/2rad)*/ 0.f, false };
        CPPUNIT_ASSERT(conesAreEqual(result, expectedCone));
    }
    // If one cone does not completely subsume the other, the resulting axis should be in between
    // the two axes, and theta_o and theta_e should encompass both cones's scopes
    {
        std::cerr << "---- Two cones, whose axes are orthogonal, both with thetaO 20deg ----\n";
        // LightTreeCone pointing in +x direction, thetaO is 20deg, thetaE is 70deg
        const LightTreeCone a{ Vec3f(1.f, 0.f, 0.f), 0.9396926f, 0.34202014f, false };
        // LightTreeCone pointing in +y direction, thetaO is 20deg, thetaE is 90deg
        const LightTreeCone b{ Vec3f(0.f, 1.f, 0.f), 0.9396926f, 0.f, false };
        // Result: LightTreeCone with axis (sqrt2/2, sqrt2/2, 0), thetaO is 65deg (45deg + 20deg), thetaE is 90deg
        const LightTreeCone result = combineCones(a, b);
        const LightTreeCone expected{ Vec3f(0.7071068f, 0.7071068f, 0.f), 0.42261826f, 0.f, false };
        printCone("a", a);
        printCone("b", b);
        printCone("result", result);
        CPPUNIT_ASSERT(conesAreEqual(result, expected));
    }
    // If one cone does not completely subsume the other, the resulting axis should be in between
    // the two axes, and theta_o and theta_e should encompass both cones's scopes
    {
        std::cerr << "---- Two cones, whose axes are orthogonal, one with thetaO 20deg, one with thetaO 30deg ----\n";
        // LightTreeCone pointing in +x direction, thetaO is 20deg, thetaE is 70deg
        const LightTreeCone a{ Vec3f(1.f, 0.f, 0.f), 0.9396926f, 0.34202014f, false };
        // LightTreeCone pointing in +y direction, thetaO is 30deg, thetaE is 90deg
        const LightTreeCone b{ Vec3f(0.f, 1.f, 0.f), 0.866025403f, 0.f, false };
        // Result: LightTreeCone whose axis is at 50deg on the unit circle, thetaO is 70deg ((90deg + 30deg + 20deg) / 2),
        // thetaE is 90deg
        // to get 50deg for the axis: thetaO is 70deg -- since cone a starts at -20deg on the unit circle, the axis
        // ends up being at positive 50deg on the unit circle
        const LightTreeCone result = combineCones(a, b);
        const LightTreeCone expected{ Vec3f(0.642787f, 0.766044f, 0.f), 0.342020f, 0.f, false };
        printCone("a", a);
        printCone("b", b);
        printCone("result", result);
        CPPUNIT_ASSERT(conesAreEqual(result, expected));
    }
}

}
}
CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestLightTree);
