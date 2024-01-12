// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "InstanceAttributes.h"

#include <moonray/rendering/mcrt_common/Types.h>

#include <numeric>

using namespace scene_rdl2;
using scene_rdl2::math::Vec2f;
using scene_rdl2::math::Vec3f;


namespace moonray {
namespace shading {

static void
transformVec3(Vec3f& vec3,
              Vec3Type type,
              const mcrt_common::Mat43& xform,
              const mcrt_common::Mat43& invXform)
{
    if (type == Vec3Type::POINT) {
        vec3 = transformPoint(xform, vec3);
    } else if (type == Vec3Type::VECTOR) {
        vec3 = transformVector(xform, vec3);
    } else if (type == Vec3Type::NORMAL) {
        vec3 = transformNormal(invXform, vec3);
    } else {
        MNRY_ASSERT(false, "unknown Vec3Type");
    }
}

void
InstanceAttributes::transformAttributes(const XformSamples& xforms,
                                        float shutterOpenDelta,
                                        float shutterCloseDelta,
                                        const std::vector<Vec3KeyType>& keyTypePairs)
{
    shading::XformSamples invXforms;
    invXforms.push_back(xforms[0].inverse());
    for (auto keyTypePair : keyTypePairs) {
        TypedAttributeKey<Vec3f> key = keyTypePair.first;
        if (!isSupported(key)) {
            continue;
        }
        Vec3f& vec3 = getAttribute(key);
        Vec3Type type = keyTypePair.second;
        transformVec3(vec3, type, xforms[0], invXforms[0]);
    }
}

} // namespace shading
} // namespace rendering


