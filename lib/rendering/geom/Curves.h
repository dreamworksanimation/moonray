// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Curves.h
/// $Id$
///
#pragma once
#ifndef GEOM_CURVES_H
#define GEOM_CURVES_H

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/VertexBuffer.h>

#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>

namespace moonray {
namespace geom {

// forward declaration
class BakedAttribute;

// BakedCurves is used for geometry baking, i.e.
// RenderContext->bakeGeometry().

struct BakedCurves
{
    const scene_rdl2::rdl2::Geometry* mRdlGeometry;
    const scene_rdl2::rdl2::Layer* mLayer;

    // Curves name
    std::string mName;

    /// enum for specifying the curve type of a Curves primitive
    enum class Type
    {
        LINEAR,
        BEZIER,
        BSPLINE,
        UNKNOWN
    };
    BakedCurves::Type mType;

    size_t mVertexCount;

    std::vector<size_t> mCurvesVertexCount;

    // 0 if no motion blur
    size_t mMotionSampleCount;

    // The vertex buffer always has the format v0_t0, v0_t1, v1_t0, v1_t1...
    // (if there are two motion samples t0 and t1.)
    std::vector<Vec3f> mVertexBuffer;

    std::vector<float> mRadii;

    std::vector<float> mMotionFrames;

    // Assigment Ids
    std::vector<int> mLayerAssignmentIds;

    // Curves attributes.
    std::vector<std::unique_ptr<BakedAttribute>> mAttrs;
};

/// @class Curves
/// @brief Primitive that contains multiple curves
///
/// Each curve in Curves primitive is specified by a set of control vertices
class Curves : public moonray::geom::Primitive
{
public:
    typedef typename moonray::geom::VertexBuffer<Vec3fa, InterleavedTraits> VertexBuffer;
    typedef std::vector<size_type> CurvesVertexCount;

    /// enum for specifying the curve type of a Curves primitive
    enum class Type
    {
        LINEAR,
        BEZIER,
        BSPLINE,
        UNKNOWN
    };

    enum class SubType
    {
        RAY_FACING,
        ROUND,
        NORMAL_ORIENTED,
        UNKNOWN
    };

    Curves(Type type,
           SubType subtype,
           int tessellationRate,
           CurvesVertexCount&& curvesVertexCount,
           VertexBuffer&& vertices,
           LayerAssignmentId&& layerAssignmentId,
           shading::PrimitiveAttributeTable&& = shading::PrimitiveAttributeTable());

    ~Curves();

    virtual void accept(PrimitiveVisitor& v) override;

    /// return the number of curves in this Curves primitive
    size_type getCurvesCount() const;

    /// query the CurvesVertexCount that record vertex count in each curve
    /// @return CurvesVertexCount const reference that record the vertex count
    ///     for each curve in this Curves primitive
    const CurvesVertexCount& getCurvesVertexCount() const;

    /// return the curves type for this Curves primitive
    Type getCurvesType() const;

    /// return the curves subtype for this Curves primitive
    SubType getCurvesSubType() const;

    /// return the tessellation rate
    int getTessellationRate() const;

    /// return the memory usage of this primitive in bytes
    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    /// retrieve the reference of VertexBuffer
    /// @return VertexBuffer reference that contains the control vertices
    ///     for this Curves primitive
    VertexBuffer& getVertexBuffer();

    /// retrieve the const reference of VertexBuffer
    /// @return VertexBuffer const reference that contains the control vertices
    ///     for this Curves primitive
    const VertexBuffer& getVertexBuffer() const;

    /// set the primitive name
    /// @param name the name of this primitive
    void setName(const std::string& name);

    /// get the primitive name
    /// @return name the name of this primitive
    const std::string& getName() const;

    void setCurvedMotionBlurSampleCount(int count);

    /// @brief an util function to verify whether the input curves data meet
    ///     the criteria of the specified Curves type
    /// @param type the type of Curves to create
    /// @param subType the subtype of Curves to create
    /// @param tessellationRate the tessellation rate
    /// @param curvesVertexCount vector to specify the curve vertex count.
    ///     The size of the vector is the number of curves in created primitive.
    ///     Each entry of curveVertexCount is the vertex count of that curve.
    /// @param vertices VertexBuffer that contains the curve control points
    /// @param attributeTable a lookup table that can be used to attach
    ///     arbitrary primitive attribute (per face id, per vertex uv...etc)
    /// @param message optional diagnose message when encountering invalid data
    /// @return Primitive::DataValidness enum
    ///     Curves type
    static DataValidness checkPrimitiveData(Type type,
        SubType subtype,
        int tessellationRate,
        const CurvesVertexCount& curvesVertexCount,
        const VertexBuffer& vertices,
        const shading::PrimitiveAttributeTable& attributeTable = shading::PrimitiveAttributeTable(),
        std::string* message = nullptr);

private:
    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    virtual internal::Primitive* getPrimitiveImpl() override;

    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    virtual void transformPrimitive(
            const MotionBlurParams& motionBlurParams,
            const shading::XformSamples& prim2render) override;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};


} // namespace geom
} // namespace moonray

#endif // GEOM_CURVES_H
