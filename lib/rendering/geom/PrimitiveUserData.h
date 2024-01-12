// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveUserData.h

#pragma once
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <scene_rdl2/scene/rdl2/SceneObject.h>

namespace moonray {
namespace geom {

using moonray::shading::AttributeRate;

/// List of sizes for each rate (or 0 for unsupported rates). It is assumed
/// constant=1 (or put 1 in partCount if parts are otherwise not supported).
struct RateCounts {
    size_t partCount;
    size_t uniformCount; // faceCount
    size_t varyingCount;
    size_t vertexCount;
    size_t faceVaryingCount;
};

/// Choose a rate based on the size of an array. If it does not match any size choose
/// the largest one that works and produce a warning message.  Zero size always
/// returns RATE_UNKNOWN and prints an error.  The object and keyname are only used
/// for the warning messages.
AttributeRate pickRate(const scene_rdl2::rdl2::SceneObject* object,
                       const std::string& keyName,
                       size_t size,
                       const geom::RateCounts& rates);

/// Same but also compare the two sizes, complaining if they are unequal and using
/// the smaller one to choose the rate. This is for time sampled primvars. If one of
/// them is zero it is ignored.
AttributeRate pickRate(const scene_rdl2::rdl2::SceneObject* object,
                       const std::string& keyName,
                       size_t size0,
                       size_t size1,
                       const geom::RateCounts& rates);

/// This is used for motion blur data. Return true if the sizes are equal. If different
/// and size is non-zero, print an error message and return false.
bool sizeCheck(const scene_rdl2::rdl2::SceneObject* object,
               const std::string& keyName,
               size_t size,
               size_t correctSize);

/// Convenience function to turn AttributeKey<T> into it's name
template<typename T>
const std::string& getName(const scene_rdl2::rdl2::SceneObject* object, const scene_rdl2::rdl2::AttributeKey<T> attributeKey)
{
    return object->getSceneClass().getAttribute(attributeKey)->getName();
}

// Add all the attributes from a list of UserData objects to the table
void processArbitraryData(const scene_rdl2::rdl2::SceneObject* geometry,
                          const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObjectVector> attributeKey,
                          shading::PrimitiveAttributeTable& primitiveAttributeTable,
                          const geom::RateCounts& rates,
                          bool useFirstFrame,
                          bool useSecondFrame);

} // namespace geom
} // namespace moonray

