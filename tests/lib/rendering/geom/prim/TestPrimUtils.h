// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <scene_rdl2/render/util/Random.h>

#include <random>

namespace moonray {
namespace geom {
namespace unittest {

class TestAttributes
{
public:
    /**
     * Initializes the test attributes. This function should be
     * called soon after program start. The test attributes
     * are not valid until init() is called.
     */
    static finline void init();
    // attributes for unit test
    static shading::TypedAttributeKey<bool> sTestBool0;
    static shading::TypedAttributeKey<int> sTestInt0;
    static shading::TypedAttributeKey<long> sTestLong0;
    static shading::TypedAttributeKey<float> sTestFloat0;
    static shading::TypedAttributeKey<float> sTestFloat1;
    static shading::TypedAttributeKey<std::string> sTestString0;
    static shading::TypedAttributeKey<scene_rdl2::math::Color> sTestColor0;
    static shading::TypedAttributeKey<scene_rdl2::math::Color> sTestColor1;
    static shading::TypedAttributeKey<scene_rdl2::math::Color4> sTestRGBA0;
    static shading::TypedAttributeKey<scene_rdl2::math::Vec2f> sTestVec2f0;
    static shading::TypedAttributeKey<scene_rdl2::math::Vec3f> sTestVec3f0;
    static shading::TypedAttributeKey<scene_rdl2::math::Mat4f > sTestMat4f0;

};

void
TestAttributes::init()
{
    // Disabling this warning:
    // warning #1711: assignment to statically allocated variable
#pragma warning push
#pragma warning disable 1711
    sTestBool0      = shading::TypedAttributeKey<bool>("test_bool_0");
    sTestInt0       = shading::TypedAttributeKey<int>("test_int_0");
    sTestLong0      = shading::TypedAttributeKey<long>("test_long_0");
    sTestFloat0     = shading::TypedAttributeKey<float>("test_float_0");
    sTestFloat1     = shading::TypedAttributeKey<float>("test_float_1");
    sTestString0    = shading::TypedAttributeKey<std::string>("test_string_0");
    sTestColor0     = shading::TypedAttributeKey<scene_rdl2::math::Color>("test_color_0");
    sTestColor1     = shading::TypedAttributeKey<scene_rdl2::math::Color>("test_color_1");
    sTestRGBA0      = shading::TypedAttributeKey<scene_rdl2::math::Color4>("test_rgba_0");
    sTestVec2f0     = shading::TypedAttributeKey<scene_rdl2::math::Vec2f>("test_vec2f_0");
    sTestVec3f0     = shading::TypedAttributeKey<scene_rdl2::math::Vec3f>("test_vec3f_0");
    sTestMat4f0     = shading::TypedAttributeKey<scene_rdl2::math::Mat4f>("test_mat4f_0");
#pragma warning pop

}

class RNG {
public:
    bool randomBool()
    {
        return mBoolDist(mGenerator);
    }

    int randomInt()
    { 
        return mIntDist(mGenerator);
    }

    int randomLong()
    { 
        return mLongDist(mGenerator);
    }

    float randomFloat()
    {
        return mGenerator.getNextFloat();
    }

    scene_rdl2::math::Color randomColor()
    {
        return scene_rdl2::math::Color(randomFloat(), randomFloat(), randomFloat());
    }

    scene_rdl2::math::Color4 randomColor4()
    {
        return scene_rdl2::math::Color4(randomFloat(), randomFloat(), randomFloat(), randomFloat());
    }

    scene_rdl2::math::Vec2f randomVec2f()
    {
        return scene_rdl2::math::Vec2f(randomFloat(), randomFloat()); 
    }


    scene_rdl2::math::Vec3f randomVec3f()
    {
        return scene_rdl2::math::Vec3f(randomFloat(), randomFloat(), randomFloat()); 
    }


    scene_rdl2::math::Mat4f randomMat4f()
    {
        return scene_rdl2::math::Mat4f(
            randomFloat(), randomFloat(), randomFloat(), randomFloat(),
            randomFloat(), randomFloat(), randomFloat(), randomFloat(),
            randomFloat(), randomFloat(), randomFloat(), randomFloat(),
            randomFloat(), randomFloat(), randomFloat(), randomFloat());
    }
private:
    scene_rdl2::util::Random mGenerator;
    std::bernoulli_distribution mBoolDist;
    std::uniform_int_distribution<int> mIntDist;
    std::uniform_int_distribution<long> mLongDist;
};


__forceinline bool isEqual(const scene_rdl2::math::Color4& lhs, const scene_rdl2::math::Color4& rhs) {
    return scene_rdl2::math::isEqual(lhs.r, rhs.r) && scene_rdl2::math::isEqual(lhs.g, rhs.g) &&
       scene_rdl2::math::isEqual(lhs.b, rhs.b) && scene_rdl2::math::isEqual(lhs.a, rhs.a);
}


// Force this function not to inline, otherwise small rounding diffs may break one of the unit tests.
// (This directive works under icc, gcc and clang without generating any warnings.)
[[gnu::noinline]]
scene_rdl2::math::Vec3f transformPointUtil(const scene_rdl2::math::Xform3f xform, const scene_rdl2::math::Vec3f point);


} // namespace unittest 
} // namespace geom
} // namespace moonray

