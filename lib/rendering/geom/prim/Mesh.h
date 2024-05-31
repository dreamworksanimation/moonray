// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Mesh.h
/// $Id$
///

#pragma once

#ifndef GEOM_MESH_HAS_BEEN_INCLUDED
#define GEOM_MESH_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/internal/InterleavedTraits.h> // must include first
#include <moonray/rendering/geom/prim/BufferDesc.h>
#include <moonray/rendering/geom/prim/NamedPrimitive.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/VertexBuffer.h>
#include <scene_rdl2/common/math/Math.h>

namespace moonray {
namespace geom {
namespace internal {

static constexpr size_t sQuadVertexCount = 4;
static constexpr size_t sTriangleVertexCount = 3;
//----------------------------------------------------------------------------
enum class MeshIndexType
{
    UNDEFINED,
    TRIANGLE,
    QUAD,
};
///
/// @class Mesh Mesh.h <geom/Mesh.h>
/// @brief Brief description of the class Mesh.
///
class Mesh : public NamedPrimitive
{
public:
    /// Constructor / Destructor
    // Short-term convenience
    explicit Mesh(LayerAssignmentId&& layerAssignmentId):
        NamedPrimitive(std::move(layerAssignmentId)), mIsSingleSided(true),
        mIsNormalReversed(false), mIsOrientationReversed(false), mIsMeshFinalized(false),
        mAdaptiveError(0.0f), mMeshResolution(1), mPrimToRender(scene_rdl2::math::one)
    {}

    Mesh(const Mesh& other) = delete;

    const Mesh& operator=(const Mesh& other) = delete;

    virtual ~Mesh() = default;

    // get the memory in byte
    virtual size_t getMemory() const override;

    virtual bool canIntersect() const override
    {
        return false;
    }

    virtual void updateVertexData(const std::vector<float>& pVertexData,
            const shading::XformSamples& prim2render) = 0;

    /// TessellatedMesh object allows primitives to convey a response to the
    /// "tessellate()" request (as an alternative to callbacks)
    /// it contains the vertex/index buffer info rt library relies on
    /// to build the intersection accelerator.
    /// It can be either all quads or all triangles
    struct TessellatedMesh
    {
        std::vector<BufferDesc> mVertexBufferDesc;
        BufferDesc mIndexBufferDesc;
        /// Mesh indexing flag to distinguish between triangle and quad
        MeshIndexType mIndexBufferType;
        size_t mVertexCount;
        size_t mFaceCount;
        virtual ~TessellatedMesh() = default;
    };

    virtual const scene_rdl2::rdl2::Material* getIntersectionMaterial(
            const scene_rdl2::rdl2::Layer* pRdlLayer,
            const mcrt_common::Ray &ray) const override;

    virtual PrimitiveType getType() const override
    {
        return POLYMESH;
    }

    const scene_rdl2::math::Mat4f& getTransform() const
    {
        return mPrimToRender;
    }

    virtual void getTessellatedMesh(TessellatedMesh &tessMesh) const = 0;

    virtual void getBakedMesh(BakedMesh &bakedMesh) const = 0;

    // Bake positions and optionally normals in uv space.  Width and height
    // are the size of the image you want to bake.  Udim is the uv udim tile.
    // StKey is the uv attribute to use as the parameterization.
    // PosResult must be non-null and be of size width * height.
    // For any given pixel, a non-zero alpha in posResult indicates that
    // there is geometry at that particular uv coordinate.
    // NrmResult is optional. If non-null it should be of size width * height.
    // For any pixel that has geometry (posResult[p].w != 0.f) nrmResult[p]
    // will contain the normal.
    virtual bool bakePosMap(int width, int height, int udim,
            shading::TypedAttributeKey<Vec2f> stKey,
            scene_rdl2::math::Vec3fa *posResult, scene_rdl2::math::Vec3f *nrmResult) const
    {
        return false;
    }

    virtual size_t getTessellatedMeshFaceCount() const = 0;

    void setIsSingleSided(bool isSingleSided)
    {
        mIsSingleSided = isSingleSided;
    }

    bool getIsSingleSided() const
    {
        return mIsSingleSided;
    }

    void setIsNormalReversed(bool isNormalReversed)
    {
        if (mIsMeshFinalized) {
            scene_rdl2::logging::Logger::warn("Mesh \'" + getName() + "\' cannot reverse "
                         "normals because it is already finalized.");
            return;
        }
        mIsNormalReversed = isNormalReversed;
    }

    bool getIsNormalReversed() const
    {
        return mIsNormalReversed;
    }

    void setIsOrientationReversed(bool reverseOrientation)
    {
        if (mIsMeshFinalized) {
            scene_rdl2::logging::Logger::warn("Mesh \'" + getName() + "\' cannot reverse "
                         "orientation because it is already finalized.");
            return;
        }
        mIsOrientationReversed = reverseOrientation;
    }

    bool getIsOrientationReversed() const
    {
        return mIsOrientationReversed;
    }

    virtual void setMeshResolution(int meshResolution)
    {
        mMeshResolution = scene_rdl2::math::max(1, meshResolution);
    }

    void setAdaptiveError(float adaptiveError) {
        mAdaptiveError = adaptiveError;
    }

    // If this mesh is bound to a volume shader with a map binding, and the
    // mesh exhibits motion blur, we must bake the velocity grid so that the
    // volume shader can be blurred with the mesh.
    // We do this by using openvdb's mesh to volume conversion tool. The mesh
    // is a shell and openvdb fills in the voxel grid bound by that shell.
    // The grid can expand the shell by an exterior bandwith (the bandwith is
    // number of voxel thick) and expand into the interior of the grid by an
    // interior bandwidth. A bandwidth of 1 means only the voxels on the surface
    // of the mesh are active. We want an interior bandwidth such that the entire
    // interior voxel grid is active.
    // @param interiorBandwidth : width of the grid in voxels / 2.
    virtual void createVelocityGrid(const float interiorBandwidth,
                                    const geom::MotionBlurParams& motionBlurParams,
                                    const std::vector<int>& volumeIds) override;

    virtual void getST(int tessFaceId, float u, float v, Vec2f& st) const = 0;
    virtual scene_rdl2::math::Vec3f getVelocity(int tessFaceId, int vIndex, float time = 0.0f) const = 0;
    virtual int getFaceAssignmentId(int faceId) const = 0;
    virtual void setRequiredAttributes(int primId, float time, float u, float v,
        float w, bool isFirst, shading::Intersection& intersection) const = 0;

protected:
    // reverse orientation by reversing the order of vertex data on each
    // face
    static void reverseOrientation(const size_t faceVertexCount,
                                   std::vector<uint32_t>& indices,
                                   std::unique_ptr<shading::Attributes>& attributes);

    static void reverseOrientation(const std::vector<uint32_t>& faceVertexCount,
                                   std::vector<uint32_t>& indices,
                                   std::unique_ptr<shading::Attributes>& attributes);

    // compute a motion vector
    // id1, id2, and id3 are indices into vertBuf defining a triangle
    // vertexW1, W2, and W3 are the weighting factors of the vertices
    // ray contains additional info about the hit
    scene_rdl2::math::Vec3f computeMotion(const VertexBuffer<scene_rdl2::math::Vec3fa, InterleavedTraits> &vertBuf,
            uint32_t id1, uint32_t id2, uint32_t id3,
            float vertexW1, float vertexW2, float vertexW3,
            const mcrt_common::Ray &ray) const;

    bool faceHasAssignment(uint faceId);

    template<typename T> void
    computeVaryingAttributeDerivatives(shading::TypedAttributeKey<T> key,
            uint32_t vid1, uint32_t vid2, uint32_t vid3, float time,
            const std::array<float, 4>& invA,
            shading::Intersection& intersection) const
    {
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f1 = mAttributes->getMotionBlurVarying(key, vid1, time);
            T f2 = mAttributes->getMotionBlurVarying(key, vid2, time);
            T f3 = mAttributes->getMotionBlurVarying(key, vid3, time);
            computeDerivatives(key, f1, f2, f3, invA, intersection);
        } else {
            const T& f1 = mAttributes->getVarying(key, vid1);
            const T& f2 = mAttributes->getVarying(key, vid2);
            const T& f3 = mAttributes->getVarying(key, vid3);
            computeDerivatives(key, f1, f2, f3, invA, intersection);
        }
    }

    template<typename T> void
    computeFaceVaryingAttributeDerivatives(shading::TypedAttributeKey<T> key,
            uint32_t fid, uint32_t fvid1, uint32_t fvid2, uint32_t fvid3,
            float time, const std::array<float, 4>& invA,
            shading::Intersection& intersection) const
    {
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f1 = mAttributes->getMotionBlurFaceVarying(key, fid, fvid1, time);
            T f2 = mAttributes->getMotionBlurFaceVarying(key, fid, fvid2, time);
            T f3 = mAttributes->getMotionBlurFaceVarying(key, fid, fvid3, time);
            computeDerivatives(key, f1, f2, f3, invA, intersection);
        } else {
            const T& f1 = mAttributes->getFaceVarying(key, fid, fvid1);
            const T& f2 = mAttributes->getFaceVarying(key, fid, fvid2);
            const T& f3 = mAttributes->getFaceVarying(key, fid, fvid3);
            computeDerivatives(key, f1, f2, f3, invA, intersection);
        }
    }

    template<typename T> void
    computeVertexAttributeDerivatives(shading::TypedAttributeKey<T> key,
            uint32_t vid1, uint32_t vid2, uint32_t vid3, float time,
            const std::array<float, 4>& invA,
            shading::Intersection& intersection) const
    {
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f1 = mAttributes->getMotionBlurVertex(key, vid1, time);
            T f2 = mAttributes->getMotionBlurVertex(key, vid2, time);
            T f3 = mAttributes->getMotionBlurVertex(key, vid3, time);
            computeDerivatives(key, f1, f2, f3, invA, intersection);
        } else {
            const T& f1 = mAttributes->getVertex(key, vid1);
            const T& f2 = mAttributes->getVertex(key, vid2);
            const T& f3 = mAttributes->getVertex(key, vid3);
            computeDerivatives(key, f1, f2, f3, invA, intersection);
        }
    }

    template<typename T> void
    computeDerivatives(shading::TypedAttributeKey<T> key,
            const T& f1, const T& f2, const T& f3,
            const std::array<float, 4>& invA,
            shading::Intersection& intersection) const
    {
        T df0 = f2 - f1;
        T df1 = f3 - f1;
        T dfds = invA[0] * df0 + invA[1] * df1;
        T dfdt = invA[2] * df0 + invA[3] * df1;
        intersection.setdAttributeds(key, dfds);
        intersection.setdAttributedt(key, dfdt);
    }

    template<typename T> T
    getConstantAttribute(shading::TypedAttributeKey<T> key,
            float time = 0.0f) const
    {
        const shading::Attributes* attributes = getAttributes();
        MNRY_ASSERT(attributes->isSupported(key) &&
            attributes->getRate(key) == shading::RATE_CONSTANT);
        if (attributes->getTimeSampleCount(key) > 1 && !scene_rdl2::math::isZero(time)) {
            return attributes->getMotionBlurConstant(key, time);
        } else {
            return attributes->getConstant(key);
        }
    }

    template <typename T> T
    getUniformAttribute(shading::TypedAttributeKey<T> key, int fid,
            float time = 0.0f) const
    {
        const shading::Attributes* attributes = getAttributes();
        MNRY_ASSERT(attributes->isSupported(key) &&
            attributes->getRate(key) == shading::RATE_UNIFORM);
        if (attributes->getTimeSampleCount(key) > 1 && !scene_rdl2::math::isZero(time)) {
            return attributes->getMotionBlurUniform(key, fid, time);
        } else {
            return attributes->getUniform(key, fid);
        }
    }

    template <typename T> T
    getVaryingAttribute(shading::TypedAttributeKey<T> key, int vid,
            float time = 0.0f) const
    {
        const shading::Attributes* attributes = getAttributes();
        MNRY_ASSERT(attributes->isSupported(key) &&
            attributes->getRate(key) == shading::RATE_VARYING);
        if (attributes->getTimeSampleCount(key) > 1 && !scene_rdl2::math::isZero(time)) {
            return attributes->getMotionBlurVarying(key, vid, time);
        } else {
            return attributes->getVarying(key, vid);
        }
    }

    template <typename T> T
    getVertexAttribute(shading::TypedAttributeKey<T> key, int vid,
            float time = 0.0f) const
    {
        const shading::Attributes* attributes = getAttributes();
        MNRY_ASSERT(attributes->isSupported(key) &&
            attributes->getRate(key) == shading::RATE_VERTEX);
        if (attributes->getTimeSampleCount(key) > 1 && !scene_rdl2::math::isZero(time)) {
            return attributes->getMotionBlurVertex(key, vid, time);
        } else {
            return attributes->getVertex(key, vid);
        }
    }

    // fill in primitive attributes displacement shader graph requests
    virtual void fillDisplacementAttributes(int tessFaceId, int vIndex,
            shading::Intersection& intersection) const {}

    // helper function for baking uv position maps on meshes
    // normal output is optional.
    static void rasterizeTrianglePos(const scene_rdl2::math::BBox2f &roiST,
            int width, int height,
            const Vec2f &a, const Vec2f &b, const Vec2f &c,
            const scene_rdl2::math::Vec3f &posA, const scene_rdl2::math::Vec3f &posB, const scene_rdl2::math::Vec3f &posC,
            const scene_rdl2::math::Vec3f *nrmA, const scene_rdl2::math::Vec3f *nrmB, const scene_rdl2::math::Vec3f *nrmC,
            scene_rdl2::math::Vec3fa *posResult, scene_rdl2::math::Vec3f *nrmResult);

    // helper function for converting st coordinates based on udim into
    // a [0,0]x[1,1] range
    // returns false if input st coordinates are outside the udim tile
    static bool udimxform(int udim, Vec2f &st);

    bool mIsSingleSided;
    bool mIsNormalReversed;
    bool mIsOrientationReversed;
    bool mIsMeshFinalized;
    // criteria for stopping further tessellation (adaptive tessellation case)
    float mAdaptiveError;
    // tessellation resolution
    int mMeshResolution;
    // This is the transform used for the baked volume shader grid.  If
    // the mesh is not a shared primitive (i.e not an instancing ref) this
    // is the first motion sample of the primitive's local to render transform.
    // For shared references, since the prim2world xform is the identity, primToRender
    // is equivalent to world2Render.
    scene_rdl2::math::Mat4f mPrimToRender;

private:

    openvdb::VectorGrid::Ptr createTriMeshVelocityGrid(float interiorBandwidth, const TessellatedMesh& mesh,
        size_t motionSample, float invFps);
    openvdb::VectorGrid::Ptr createQuadMeshVelocityGrid(float interiorBandwidth, const TessellatedMesh& mesh,
        size_t motionSample, float invFps);
};


// solve the 2x2 system Ax = b where A is [st2 - st1, st3 - st1],
// x is the differential we are looking for and b is [f2 - f1, f3 - f1]
// we can reuse the A^-1 for arbitrary attriutes derivatives calculation
finline bool
computeStInverse(const Vec2f& st1, const Vec2f& st2, const Vec2f& st3,
        std::array<float, 4>& invA)
{
    std::array<float , 4> A = {
        st2[0] - st1[0], st2[1] - st1[1],
        st3[0] - st1[0], st3[1] - st1[1]};
    float detA = A[0] * A[3] - A[1] * A[2];
    if (!std::isnormal(detA)) {
        return false;
    }
    float invDet = 1.0f / detA;
    invA = {invDet *  A[3], invDet * -A[1],
            invDet * -A[2], invDet *  A[0]};
    return true;
}

class MeshInterpolator : public shading::Interpolator
{
public:
    // For quad face interpolation in subdivision mesh and polygonal mesh
    // quad --(tessellated quads)--> tessellated triangles
    // subdivision is tessellated as quad now,
    // this constructor is currently not in use
    MeshInterpolator(const shading::Attributes *attr, float time, int part,
            int coarseFace, int varying0, int varying1, int varying2, int varying3,
            float varyingW0, float varyingW1, float varyingW2, float varyingW3,
            int tessellatedFace, int vertex0, int vertex1, int vertex2,
            float vertexW0, float vertexW1, float vertexW2):
            shading::Interpolator(attr, time, part, coarseFace, 4,
            mVaryingIndex, mVaryingWeights,
            4, mFaceVaryingIndex, mVaryingWeights,
            tessellatedFace, 3, mVertexIndex, mVertexWeights)
    {
        mVaryingIndex[0] = varying0;
        mVaryingIndex[1] = varying1;
        mVaryingIndex[2] = varying2;
        mVaryingIndex[3] = varying3;
        mFaceVaryingIndex[0] = 0;
        mFaceVaryingIndex[1] = 1;
        mFaceVaryingIndex[2] = 2;
        mFaceVaryingIndex[3] = 3;
        mVaryingWeights[0] = varyingW0;
        mVaryingWeights[1] = varyingW1;
        mVaryingWeights[2] = varyingW2;
        mVaryingWeights[3] = varyingW3;

        mVertexIndex[0] = vertex0;
        mVertexIndex[1] = vertex1;
        mVertexIndex[2] = vertex2;
        mVertexWeights[0] = vertexW0;
        mVertexWeights[1] = vertexW1;
        mVertexWeights[2] = vertexW2;
    }

    // For quad face interpolation in subdivision mesh and polygonal mesh
    // quad -> tessellated quad mesh,
    // and original quad
    MeshInterpolator(const shading::Attributes *attr, float time, int part,
            int coarseFace, int varying0, int varying1, int varying2, int varying3,
            float varyingW0, float varyingW1, float varyingW2, float varyingW3,
            int quadFace, int vertex0, int vertex1, int vertex2, int vertex3,
            float vertexW0, float vertexW1, float vertexW2, float vertexW3):
            shading::Interpolator(attr, time, part, coarseFace, 4,
            mVaryingIndex, mVaryingWeights,
            4, mFaceVaryingIndex, mVaryingWeights,
            quadFace, 4, mVertexIndex, mVertexWeights)
    {
        mVaryingIndex[0] = varying0;
        mVaryingIndex[1] = varying1;
        mVaryingIndex[2] = varying2;
        mVaryingIndex[3] = varying3;
        mFaceVaryingIndex[0] = 0;
        mFaceVaryingIndex[1] = 1;
        mFaceVaryingIndex[2] = 2;
        mFaceVaryingIndex[3] = 3;
        mVaryingWeights[0] = varyingW0;
        mVaryingWeights[1] = varyingW1;
        mVaryingWeights[2] = varyingW2;
        mVaryingWeights[3] = varyingW3;

        mVertexIndex[0] = vertex0;
        mVertexIndex[1] = vertex1;
        mVertexIndex[2] = vertex2;
        mVertexIndex[3] = vertex3;
        mVertexWeights[0] = vertexW0;
        mVertexWeights[1] = vertexW1;
        mVertexWeights[2] = vertexW2;
        mVertexWeights[3] = vertexW3;
    }

    // For triangle face interpolation in subdivision mesh and polygonal mesh
    // triangle -> tessellated triangle mesh,
    // or original triangle.
    // used only for polygonal mesh,
    // subdivision mesh is no longer tessellated as triangles
    MeshInterpolator(const shading::Attributes *attr, float time, int part,
            int coarseFace, int varying0, int varying1, int varying2,
            float varyingW0, float varyingW1, float varyingW2,
            int tessellatedFace, int vertex0, int vertex1, int vertex2,
            float vertexW0, float vertexW1, float vertexW2):
            shading::Interpolator(attr, time, part, coarseFace, 3,
            mVaryingIndex, mVaryingWeights,
            3, mFaceVaryingIndex, mVaryingWeights,
            tessellatedFace, 3, mVertexIndex, mVertexWeights)
    {
        mVaryingIndex[0] = varying0;
        mVaryingIndex[1] = varying1;
        mVaryingIndex[2] = varying2;
        mFaceVaryingIndex[0] = 0;
        mFaceVaryingIndex[1] = 1;
        mFaceVaryingIndex[2] = 2;
        mVaryingWeights[0] = varyingW0;
        mVaryingWeights[1] = varyingW1;
        mVaryingWeights[2] = varyingW2;

        mVertexIndex[0] = vertex0;
        mVertexIndex[1] = vertex1;
        mVertexIndex[2] = vertex2;
        mVertexWeights[0] = vertexW0;
        mVertexWeights[1] = vertexW1;
        mVertexWeights[2] = vertexW2;
    }

    // For triangle face interpolation in subdivision mesh
    // triangle -> tessellated quad mesh
    // the coarse face is triangle, but the tessellated faces are quads
    MeshInterpolator(const shading::Attributes *attr, float time, int part,
            int coarseFace, int varying0, int varying1, int varying2,
            float varyingW0, float varyingW1, float varyingW2,
            int quadFace, int vertex0, int vertex1, int vertex2, int vertex3,
            float vertexW0, float vertexW1, float vertexW2, float vertexW3):
            shading::Interpolator(attr, time, part, coarseFace, 3,
            mVaryingIndex, mVaryingWeights,
            3, mFaceVaryingIndex, mVaryingWeights,
            quadFace, 4, mVertexIndex, mVertexWeights)
    {
        mVaryingIndex[0] = varying0;
        mVaryingIndex[1] = varying1;
        mVaryingIndex[2] = varying2;
        mFaceVaryingIndex[0] = 0;
        mFaceVaryingIndex[1] = 1;
        mFaceVaryingIndex[2] = 2;
        mVaryingWeights[0] = varyingW0;
        mVaryingWeights[1] = varyingW1;
        mVaryingWeights[2] = varyingW2;

        mVertexIndex[0] = vertex0;
        mVertexIndex[1] = vertex1;
        mVertexIndex[2] = vertex2;
        mVertexIndex[3] = vertex3;
        mVertexWeights[0] = vertexW0;
        mVertexWeights[1] = vertexW1;
        mVertexWeights[2] = vertexW2;
        mVertexWeights[3] = vertexW3;
    }

private:
    int mVaryingIndex[4];
    int mFaceVaryingIndex[4];
    float mVaryingWeights[4];
    // there can be only triangles or only quads in fully tessellated mesh
    int mVertexIndex[4];
    float mVertexWeights[4];
};

//----------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray

#endif /* GEOM_MESH_HAS_BEEN_INCLUDED */

