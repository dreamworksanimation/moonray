// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// @file TestLightSetSampler.h
//

#pragma once

#include <moonray/rendering/pbr/integrator/LightSetSampler.h>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>


namespace moonray {
namespace pbr {


class Light;


//----------------------------------------------------------------------------

class TestLightSetSampler : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestLightSetSampler);

#if 1
    CPPUNIT_TEST(testRadiance);
#endif

    CPPUNIT_TEST_SUITE_END();

public:
    TestLightSetSampler();
    ~TestLightSetSampler();

    void setUp();
    void tearDown();

    void testRadiance();

private:

    typedef std::vector<std::shared_ptr<Light> > SharedLightVector;
    typedef std::vector<const Light *> LightVector;

    void testLightSetRadiance(const LightSet &lightSet,
                              const scene_rdl2::math::Vec3f &p,
                              const scene_rdl2::math::Vec3f &n);

    scene_rdl2::rdl2::SceneContext mContext;
    SharedLightVector mLights;
    LightVector mLightSetA;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

