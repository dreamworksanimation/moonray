// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Curves.h
///

#pragma once

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/prim/BufferDesc.h>
#include <moonray/rendering/geom/prim/NamedPrimitive.h>

namespace moonray {
namespace geom {
namespace internal {
///
/// @class Curves Curves.h <geom/Curves.h>
/// @brief The Curves class is the highest level differentiation of primitive
///         for a group of curves.  It represents a "handoff-able" element from
///         the procedural system to the rendering system.
///

class Curves : public NamedPrimitive
{
public:
    enum class Type
    {
        LINEAR,
        BEZIER,
        BSPLINE,
        UNKNOWN
    };

    enum class SubType {
        RAY_FACING,
        ROUND,
        NORMAL_ORIENTED
    };

    explicit Curves(Curves::Type type,
                    SubType subtype,
                    geom::Curves::CurvesVertexCount&& curvesVertexCount,
                    geom::Curves::VertexBuffer&& vertices,
                    LayerAssignmentId&& layerAssignmentId,
                    shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        NamedPrimitive(std::move(layerAssignmentId)),
        mType(type),
        mSubType(subtype),
        mCurvesVertexCount(std::move(curvesVertexCount)),
        mVertexBuffer(std::move(vertices)),
        mIndexBuffer({}),
        mSpanCount(0),
        mPrimitiveAttributeTable(std::move(primitiveAttributeTable)),
        mCurvedMotionBlurSampleCount(0)
    {
        if (mSubType == SubType::NORMAL_ORIENTED) {
            const shading::PrimitiveAttribute<scene_rdl2::math::Vec3f>& normalAttr =
                mPrimitiveAttributeTable.getAttribute<scene_rdl2::math::Vec3f>(shading::StandardAttributes::sNormal);

            switch(normalAttr.getRate()) {
            case shading::AttributeRate::RATE_CONSTANT:
                for (size_t i = 0; i < mVertexBuffer.size(); ++i) {
                    mNormalBuffer.push_back(normalAttr[0]);
                }
                break;
            case shading::AttributeRate::RATE_UNIFORM:
                {
                    size_t curveIndex = 0;
                    size_t vCount = mCurvesVertexCount[curveIndex];
                    for (size_t i = 0; i < mVertexBuffer.size(); ++i) {
                        if (i == vCount) {
                            curveIndex++;
                            vCount += mCurvesVertexCount[curveIndex];
                        }
                        mNormalBuffer.push_back(normalAttr[curveIndex]);
                    }
                }
                break;
            case shading::AttributeRate::RATE_VERTEX:
            case shading::AttributeRate::RATE_VARYING:
                for (size_t i = 0; i < mVertexBuffer.size(); ++i) {
                    mNormalBuffer.push_back(normalAttr[i]);
                }
                break;
            default:
                // If the rate is not one of the above it should
                // have been checked for already
                MNRY_ASSERT(false);
                break;
            }
        }
    }

    /// The "Spans" struct and functions provide a service analogous
    /// to the "TriangleMesh" facilities -- they allow primitives to convey a
    /// response to the "tessellate()" request (as an alternative to callbacks)
    class Spans
    {
    public:
        std::vector<BufferDesc> mVertexBufferDesc;
        BufferDesc mIndexBufferDesc;
        BufferDesc mNormalBufferDesc;
        size_t mVertexCount;
        size_t mSpanCount;
    };

    virtual PrimitiveType getType() const override
    {
        return CURVES;
    }

    SubType getSubType() const
    {
        return mSubType;
    }

    void getTessellatedSpans(Spans& spans) const
    {
        spans.mSpanCount = getSpanCount();

        spans.mIndexBufferDesc.mData = static_cast<const void*>(mIndexBuffer.data());
        spans.mIndexBufferDesc.mOffset = 0;
        spans.mIndexBufferDesc.mStride = sizeof(IndexData);

        if (mSubType == SubType::NORMAL_ORIENTED) {
            spans.mNormalBufferDesc.mData = static_cast<const void*>(mNormalBuffer.data());;
            spans.mNormalBufferDesc.mOffset = 0;
            spans.mNormalBufferDesc.mStride = sizeof(scene_rdl2::math::Vec3f);
        }

        spans.mVertexCount = getVertexCount();
        size_t motionSampleCount = getMotionSamplesCount();
        size_t vertexSize = sizeof(geom::Curves::VertexBuffer::value_type);
        size_t vertexStride = motionSampleCount * vertexSize;
        const void* data = mVertexBuffer.data();
        for (size_t t = 0; t < motionSampleCount; ++t) {
            size_t offset = t * vertexSize;
            spans.mVertexBufferDesc.emplace_back(data, offset, vertexStride);
        }
    }

    virtual void getBakedCurves(BakedCurves &bakedCurves) const;

    geom::Primitive::size_type getCurvesCount() const
    {
        return mCurvesVertexCount.size();
    }

    const geom::Curves::CurvesVertexCount& getCurvesVertexCount() const
    {
        return mCurvesVertexCount;
    }

    geom::Curves::VertexBuffer& getVertexBuffer()
    {
        return mVertexBuffer;
    }

    const geom::Curves::VertexBuffer& getVertexBuffer() const
    {
        return mVertexBuffer;
    }

    uint32_t getSpanCount() const
    {
        return mSpanCount;
    }

    uint32_t getVertexCount() const
    {
        return mVertexBuffer.size();
    }

    const shading::PrimitiveAttributeTable* getPrimitiveAttributeTable() const
    {
        return &mPrimitiveAttributeTable;
    }

    void setCurvedMotionBlurSampleCount(int count)
    {
        mCurvedMotionBlurSampleCount = count;
    }

    int getCurvedMotionBlurSampleCount()
    {
        return mCurvedMotionBlurSampleCount;
    }

    scene_rdl2::rdl2::MotionBlurType getMotionBlurType()
    {
        return mMotionBlurType;
    }

    virtual int
    getIntersectionAssignmentId(int primID) const override
    {
        int assignmentId =
            mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
            mLayerAssignmentId.getConstId() :
            mLayerAssignmentId.getVaryingId()[mIndexBuffer[primID].mChain];
        return assignmentId;
    }

    virtual const scene_rdl2::rdl2::Material *
    getIntersectionMaterial(const scene_rdl2::rdl2::Layer *pRdlLayer,
            const mcrt_common::Ray &ray) const override
    {
        int assignmentId =
            mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
            mLayerAssignmentId.getConstId() :
            mLayerAssignmentId.getVaryingId()[mIndexBuffer[ray.primID].mChain];

        MNRY_ASSERT(assignmentId > -1);
        const scene_rdl2::rdl2::Material *pMaterial = MNRY_VERIFY(pRdlLayer->lookupMaterial(assignmentId));
        return pMaterial;
    }

    virtual size_t getMemory() const override
    {
        size_t mem = sizeof(Curves) - sizeof(NamedPrimitive) + NamedPrimitive::getMemory();
        // get memory for vertex buffer
        mem += mVertexBuffer.get_memory_usage();
        // get memory for mIndexBuffer;
        mem += scene_rdl2::util::getVectorElementsMemory(mIndexBuffer);
        mem += scene_rdl2::util::getVectorElementsMemory(mCurvesVertexCount);
        return mem;
    }

    virtual size_t getMotionSamplesCount() const override
    {
        return mVertexBuffer.get_time_steps();
    }

    virtual BBox3f computeAABB() const override
    {
        if (mVertexBuffer.empty()) {
            return BBox3f(scene_rdl2::math::zero);
        }
        float maxRadius = 0.0f;
        size_t motionSampleCount = getMotionSamplesCount();
        BBox3f result(mVertexBuffer(0));
        for (size_t v = 0; v < mVertexBuffer.size(); ++v) {
            for (size_t t = 0; t < motionSampleCount; ++t) {
                result.extend(mVertexBuffer(v, t));
                maxRadius = std::max(maxRadius, mVertexBuffer(v, t).w);
            }
        }
        // padding the bounding box with curves radius
        result.lower -= Vec3f(maxRadius);
        result.upper += Vec3f(maxRadius);
        return result;
    }

    virtual BBox3f computeAABBAtTimeStep(int timeStep) const override
    {
        if (mVertexBuffer.empty()) {
            return BBox3f(scene_rdl2::math::zero);
        }
        float maxRadius = 0.0f;
        MNRY_ASSERT(timeStep >= 0 && timeStep < static_cast<int>(getMotionSamplesCount()), "timeStep out of range");
        BBox3f result(scene_rdl2::util::empty);
        for (size_t v = 0; v < mVertexBuffer.size(); ++v) {
            result.extend(mVertexBuffer(v, timeStep));
            maxRadius = std::max(maxRadius, mVertexBuffer(v, timeStep).w);
        }
        // padding the bounding box with curves radius
        result.lower -= Vec3f(maxRadius);
        result.upper += Vec3f(maxRadius);
        return result;
    }

protected:

    struct IndexData {
        IndexData(uint32_t vertex, uint32_t chain, uint32_t span):
            mVertex(vertex), mChain(chain), mSpan(span)
        {}
        uint32_t mVertex;
        uint32_t mChain;
        uint32_t mSpan;
    };

    std::unique_ptr<BakedAttribute> getBakedAttribute(const shading::AttributeKey& key) const;

    template <typename T>
    void* getBakedAttributeData(const shading::TypedAttributeKey<T>& key,
                                size_t& numElements, shading::AttributeRate &newRate) const;

    Curves::Type mType;
    SubType mSubType;
    geom::Curves::CurvesVertexCount mCurvesVertexCount;
    geom::Curves::VertexBuffer mVertexBuffer;
    std::vector<IndexData> mIndexBuffer;
    std::vector<scene_rdl2::math::Vec3f> mNormalBuffer;
    uint32_t mSpanCount;
    shading::PrimitiveAttributeTable mPrimitiveAttributeTable;
    uint32_t mCurvedMotionBlurSampleCount;
    scene_rdl2::rdl2::MotionBlurType mMotionBlurType;
};

} // namespace internal
} // namespace geom
} // namespace moonray

