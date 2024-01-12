// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestHair.h
/// $Id$
///

#pragma once

#include <scene_rdl2/common/math/Vec3.h>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

#include <memory>

namespace moonray {

namespace shading {
// forward declarations
class BsdfSlice;
class Fresnel;
class HairBsdfLobe;
class HairState;
}

namespace shading {

//----------------------------------------------------------------------------

///
/// @class TestHair TestHair.h <shading/TestHair.h>
/// @brief This class tests that the vectorized and scalar functions
/// have the same outputs given the same inputs
///
class TestHair : public CppUnit::TestFixture
{
public:
    TestHair();
    ~TestHair();

    CPPUNIT_TEST_SUITE(TestHair);
    CPPUNIT_TEST(testHairUtil);
    CPPUNIT_TEST(testRLobe);
    CPPUNIT_TEST(testTRTLobe);
    CPPUNIT_TEST(testTTLobe);
    CPPUNIT_TEST(testTRRTLobe);
    CPPUNIT_TEST(testHairState);
    CPPUNIT_TEST_SUITE_END();

    void testHairUtil();
    void testRLobe();
    void testTRTLobe();
    void testTTLobe();
    void testTRRTLobe();
    void testHairState();

private:
    // helper for wrapping create/destroy-style resources into
    // RAII unique_ptr with custom deleter
    template<typename T>
    using deleted_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

    void testLobe(shading::HairBsdfLobe * lobeCpp, void * lobeIspc);

    // cpp objects
    std::unique_ptr<shading::BsdfSlice> mBsdfSliceCpp;
    std::unique_ptr<shading::HairState> mHairStateCpp;
    std::unique_ptr<shading::Fresnel> mFresnelCpp;

    // ispc objects
    deleted_unique_ptr<void> mBsdfSliceIspc;
    deleted_unique_ptr<void> mHairStateIspc;
    deleted_unique_ptr<void> mFresnelIspc;

    scene_rdl2::math::Vec3f mNg;
    scene_rdl2::math::Vec3f mWi;
    scene_rdl2::math::Vec3f mWo;
};


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray



