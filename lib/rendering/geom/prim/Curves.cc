// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Curves.cc
/// $Id$
///

#include "Curves.h"

#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/MeshTessellationUtil.h>
#include <moonray/rendering/geom/BakedAttribute.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace scene_rdl2::math;
using namespace mcrt_common;
using namespace shading;

//------------------------------------------------------------------------------

// Utility interpolation function for attribute baking
template <typename T>
T
bilinearInterpolate(const Vec2f& uv,
                    const T& a0, const T& a1, const T& a2, const T& a3)
{
    return (1.0f - uv[0]) * (1.0f - uv[1]) * a0 +
           (       uv[0]) * (1.0f - uv[1]) * a1 +
           (       uv[0]) * (       uv[1]) * a2 +
           (1.0f - uv[0]) * (       uv[1]) * a3;
}

template <typename T>
void*
Curves::getBakedAttributeData(const TypedAttributeKey<T>& key,
                                size_t& numElements, AttributeRate &newRate) const
{
    size_t curvesCount = mCurvesVertexCount.size(); 
    size_t vertexCount = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        vertexCount += mCurvesVertexCount[i];
    }

    Attributes *attributes = getAttributes();
    size_t timeSamples = attributes->getTimeSampleCount(key);

    void* data;
    newRate = attributes->getRate(key);

    switch (newRate) {
    case RATE_CONSTANT:
    {
        numElements = timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t t = 0; t < timeSamples; t++) {
            tdata[t] = attributes->getConstant(key, t);
        }
        break;
    }
    case RATE_UNIFORM:
    {
        numElements = curvesCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t c = 0, dstIdx = 0; c < curvesCount; c++) {
            for (size_t t = 0; t < timeSamples; t++) {
                tdata[dstIdx++] = attributes->getUniform(key, c, t);
            }
        }
        break;
    }
    case RATE_VARYING:
    case RATE_VERTEX:
    {
        if (key == StandardAttributes::sNormal) {
            // reformat to FACE_VARYING if the normals are vertex rate
            numElements = vertexCount * timeSamples;
            T* tdata = new T[numElements];
            data = tdata;
            for (size_t v = 0, dstIdx = 0; v < vertexCount; v++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    tdata[dstIdx++] = attributes->getVertex(key, v, t);
                }
            }
        } else {
            numElements = getVertexCount() * timeSamples;
            T* tdata = new T[numElements];
            data = tdata;
            for (size_t v = 0, dstIdx = 0; v < vertexCount; v++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    tdata[dstIdx++] = attributes->getVertex(key, v, t);
                }
            }
        }
        break;
    }
    default:
        MNRY_ASSERT(false, "unknown attribute rate");
        break;
    }

    return data;
}

// Template specialization because we can't interpolate strings...
template <>
void*
Curves::getBakedAttributeData(const TypedAttributeKey<std::string>& key,
                              size_t& numElements, AttributeRate &newRate) const
{
    size_t curvesCount = mCurvesVertexCount.size(); 
    size_t vertexCount = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        vertexCount += mCurvesVertexCount[i];
    }

    Attributes *attributes = getAttributes();
    size_t timeSamples = attributes->getTimeSampleCount(key);

    void* data;
    newRate = attributes->getRate(key);

    switch (attributes->getRate(key)) {
    case RATE_CONSTANT:
    {
        numElements = timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t t = 0; t < timeSamples; t++) {
            tdata[t] = attributes->getConstant(key, t);
        }
        break;
    }
    case RATE_UNIFORM:
    {
        numElements = curvesCount * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t c = 0, dstIdx = 0; c < curvesCount; c++) {
            for (size_t t = 0; t < timeSamples; t++) {
                tdata[dstIdx++] = attributes->getUniform(key, c, t);
            }
        }
        break;
    }
    case RATE_VARYING:
    case RATE_VERTEX:
    {
        numElements = vertexCount * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t v = 0, dstIdx = 0; v < vertexCount; v++) {
            for (size_t t = 0; t < timeSamples; t++) {
                tdata[dstIdx++] = attributes->getVertex(key, v, t);
            }
        }
        break;
    }
    default:
        MNRY_ASSERT(false, "unknown attribute rate");
        break;
    }

    return data;
}

std::unique_ptr<BakedAttribute>
Curves::getBakedAttribute(const AttributeKey& key) const
{
    Attributes *attributes = getAttributes();
    size_t timeSamples = attributes->getTimeSampleCount(key);

    std::unique_ptr<BakedAttribute> battr = fauxstd::make_unique<BakedAttribute>();

    battr->mName = key.getName();
    battr->mTimeSampleCount = timeSamples;
    battr->mType = key.getType();
    battr->mData = nullptr;

    switch (battr->mType) {
    case AttributeType::TYPE_BOOL:
        battr->mData = getBakedAttributeData(TypedAttributeKey<bool>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_INT:
        battr->mData = getBakedAttributeData(TypedAttributeKey<int>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_LONG:
        battr->mData = getBakedAttributeData(TypedAttributeKey<long>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_FLOAT:
        battr->mData = getBakedAttributeData(TypedAttributeKey<float>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_DOUBLE:
        battr->mData = getBakedAttributeData(TypedAttributeKey<double>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_STRING:
        battr->mData = getBakedAttributeData(TypedAttributeKey<std::string>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_RGB:
        battr->mData = getBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Color>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_RGBA:
        battr->mData = getBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Color4>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_VEC2F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<Vec2f>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_VEC3F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<Vec3f>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_VEC4F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<Vec4f>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    case AttributeType::TYPE_MAT4F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                                             battr->mNumElements, battr->mRate);
        break;
    default:
        MNRY_ASSERT(false, (std::string("unsupported attribute type ") +
            std::string(attributeTypeName(key.getType())) +
            std::string(" for attribute ") + std::string(key.getName())).c_str());
        break;
    }

    return battr;
}

void
Curves::getBakedCurves(BakedCurves& bakedCurves) const
{
    bakedCurves.mName = mName;
    bakedCurves.mType = static_cast<BakedCurves::Type>(mType);
    bakedCurves.mMotionSampleCount = getMotionSamplesCount();
    bakedCurves.mCurvesVertexCount.resize(getCurvesVertexCount().size());
    bakedCurves.mVertexCount = 0;
    for (size_t i = 0; i < getCurvesVertexCount().size(); ++i) {
        bakedCurves.mCurvesVertexCount[i] = getCurvesVertexCount()[i];
        bakedCurves.mVertexCount += bakedCurves.mCurvesVertexCount[i];
    }
    bakedCurves.mVertexBuffer.resize(bakedCurves.mVertexCount * bakedCurves.mMotionSampleCount);
    bakedCurves.mRadii.resize(bakedCurves.mVertexCount * bakedCurves.mMotionSampleCount);
    for (size_t v = 0; v < bakedCurves.mVertexCount; v++) {
        for (size_t t = 0; t < bakedCurves.mMotionSampleCount; t++) {
            const scene_rdl2::math::Vec3fa& vtx = mVertexBuffer(v, t);
            scene_rdl2::math::Vec3f p(vtx.x, vtx.y, vtx.z);
            bakedCurves.mVertexBuffer[v * bakedCurves.mMotionSampleCount + t] = p;
            bakedCurves.mRadii[v * bakedCurves.mMotionSampleCount + t] = vtx.w;
        }
    }

    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        bakedCurves.mLayerAssignmentIds = std::vector<int>(bakedCurves.mCurvesVertexCount.size(),
                                                           mLayerAssignmentId.getConstId());
    } else {
        bakedCurves.mLayerAssignmentIds = mLayerAssignmentId.getVaryingId();
    }

    Attributes *attributes = getAttributes();
    const PrimitiveAttributeTable* patable = getPrimitiveAttributeTable();
    for (const auto& entry : *patable) {
        const AttributeKey& key = entry.first;
        if (attributes->hasAttribute(key)) {
            bakedCurves.mAttrs.push_back(getBakedAttribute(key));
        }
    }
}

//------------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray


