// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BakedAttribute.h
/// $Id$
///
#pragma once

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/VertexBuffer.h>

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>
#include <moonray/rendering/shading/Shading.h>

namespace moonray {
namespace geom {


// BakedAttribute is used for geometry baking, i.e.
// RenderContext->bakeGeometry().

class BakedAttribute
{
public:
    BakedAttribute() : mData(nullptr) {}

    ~BakedAttribute()
    {
        switch (mType) {
        case AttributeType::TYPE_BOOL:
            delete[] static_cast<bool *>(mData);
            break;
        case AttributeType::TYPE_INT:
            delete[] static_cast<int *>(mData);
            break;
        case AttributeType::TYPE_LONG:
            delete[] static_cast<long *>(mData);
            break;
        case AttributeType::TYPE_FLOAT:
            delete[] static_cast<float *>(mData);
            break;
        case AttributeType::TYPE_DOUBLE:
            delete[] static_cast<double *>(mData);
            break;
        case AttributeType::TYPE_STRING:
            delete[] static_cast<std::string *>(mData);
            break;
        case AttributeType::TYPE_RGB:
            delete[] static_cast<scene_rdl2::math::Color *>(mData);
            break;
        case AttributeType::TYPE_RGBA:
            delete[] static_cast<scene_rdl2::math::Color4 *>(mData);
            break;
        case AttributeType::TYPE_VEC2F:
            delete[] static_cast<Vec2f *>(mData);
            break;
        case AttributeType::TYPE_VEC3F:
            delete[] static_cast<Vec3f *>(mData);
            break;
        case AttributeType::TYPE_VEC4F:
            delete[] static_cast<Vec4f *>(mData);
            break;
        case AttributeType::TYPE_MAT4F:
            delete[] static_cast<scene_rdl2::math::Mat4f *>(mData);
            break;
        }
    }

    // The layout of the attribute buffers depends on the attribute rate and
    // the number of motion samples.  The examples below assume two motion
    // samples t0 and t1.

    // For RATE_CONSTANT: The number of elements is the number of motion samples.
    // Format is c_t0, c_t1.

    // For RATE_UNIFORM: # elements = numFaces * mMotionSampleCount.
    // Format is f0_t0, f0_t1, f1_t0, f1_t1, f2_t0, f2_t1...

    // For RATE_VERTEX: #elements = mVertexCount * mMotionSampleCount
    // Format is v0_t0, v0_t1, v1_t0, v1_t1, v2_t0, v2_t1...

    // For RATE_VARYING: same as RATE_VERTEX after baking

    // For RATE_FACE_VARYING: #elements = numFaces * mVertsPerFace * mMotionSampleCount
    // E.g. for mVertsPerFace=3 and mMotionSampleCount=2:
    // Format is f0_v0_t0, f0_v0_t1, f0_v1_t0, f0_v1_t1, f0_v2_t0, f0_v2_t1,
    //           f1_v0_t0, f1_v0_t1, f1_v1_t0, f1_v1_t1, f1_v2_t0, f1_v2_t1...

    // For RATE_PART: #elements = numParts * mMotionSampleCount
    // Format is: p0_t0, p0_t1, p1_t0, p1_t1...

    std::string mName;
    size_t mTimeSampleCount;
    AttributeType mType;
    shading::AttributeRate mRate;
    size_t mNumElements;
    void *mData;   // You must static_cast<> this to the appropriate type
};

} // namespace geom
} // namespace moonray


