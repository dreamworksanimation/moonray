// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TestPrimUtils.h"

namespace moonray {
namespace geom {
namespace unittest {

using shading::TypedAttributeKey;

TypedAttributeKey<bool> TestAttributes::sTestBool0;
TypedAttributeKey<int> TestAttributes::sTestInt0;
TypedAttributeKey<long> TestAttributes::sTestLong0;
TypedAttributeKey<float> TestAttributes::sTestFloat0;
TypedAttributeKey<float> TestAttributes::sTestFloat1;
TypedAttributeKey<std::string> TestAttributes::sTestString0;
TypedAttributeKey<scene_rdl2::math::Color> TestAttributes::sTestColor0;
TypedAttributeKey<scene_rdl2::math::Color> TestAttributes::sTestColor1;
TypedAttributeKey<scene_rdl2::math::Color4> TestAttributes::sTestRGBA0;
TypedAttributeKey<scene_rdl2::math::Vec2f> TestAttributes::sTestVec2f0;
TypedAttributeKey<scene_rdl2::math::Vec3f> TestAttributes::sTestVec3f0;
TypedAttributeKey<scene_rdl2::math::Mat4f> TestAttributes::sTestMat4f0;


scene_rdl2::math::Vec3f transformPointUtil(const scene_rdl2::math::Xform3f xform, const scene_rdl2::math::Vec3f point)
{
    return transformPoint(xform, point);
}


} // namespace unittest 
} // namespace geom
} // namespace moonray

