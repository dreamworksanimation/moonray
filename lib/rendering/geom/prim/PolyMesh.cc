// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PolyMesh.h
/// $Id$
///

#include "PolyMesh.h"

#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/MeshTessellationUtil.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/geom/BakedAttribute.h>
#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace scene_rdl2::math;
using namespace mcrt_common;
using namespace shading;

//------------------------------------------------------------------------------

size_t
PolyMesh::getMemory() const
{
    size_t mem = sizeof(PolyMesh) - sizeof(Mesh) + Mesh::getMemory();
    mem += scene_rdl2::util::getVectorElementsMemory(mBaseIndices);
    // get memory for index buffer
    mem += scene_rdl2::util::getVectorElementsMemory(mIndices);
    // get memory for vertex buffer
    mem += mVertices.get_memory_usage();
    mem += scene_rdl2::util::getVectorElementsMemory(mTessellatedToBaseFace);
    mem += scene_rdl2::util::getVectorElementsMemory(mFaceVaryingUv);
    return mem;
}

void
PolyMesh::tessellate(const TessellationParams& tessellationParams)
{
    if (mIsMeshFinalized) {
        return;
    }

    MeshIndexType baseFaceType = getBaseFaceType();
    size_t baseFaceVertexCount = baseFaceType == MeshIndexType::QUAD ?
        sQuadVertexCount : sTriangleVertexCount;
    auto& primitiveAttributeTable = mPolyMeshData->mPrimitiveAttributeTable;
    // make input mesh all quads or all triangles
    splitNGons(mPolyMeshData->mEstiFaceCount,
               mVertices,
               mFaceToPart,
               mPolyMeshData->mFaceVertexCount,
               mIndices,
               mLayerAssignmentId,
               primitiveAttributeTable);

    MNRY_ASSERT_REQUIRE(!mIndices.empty() &&
        mIndices.size() % baseFaceVertexCount == 0);
    MNRY_ASSERT_REQUIRE(mVertices.size() > 0);
    size_t baseFaceCount = mIndices.size() / baseFaceVertexCount;
    // varying and vertex are the same thing for polygon mesh if
    // there is no displacement tessellation involved, otherwise
    // vertex rate means "per tessellated vertex" and
    // varying rate means "per base vertex"
    // we make the distinction here that all per vertex
    // input primitive attributes are classified as RATE_VARYING
    // and make RATE_VERTEX storing primitive attributes that will be
    // tessellated to "per tessellated vertex"
    // Currently the only primitive attribute that is classified as
    // RATE_VERTEX is shading normal (StandardAttributes::sNormal)
    for (auto& kv : primitiveAttributeTable) {
        for (auto& attribute : kv.second) {
            if (attribute->getRate() == RATE_VERTEX) {
                attribute->setRate(RATE_VARYING);
            }
        }
    }
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::VARYING) {
        MNRY_ASSERT_REQUIRE(
            mLayerAssignmentId.getVaryingId().size() == baseFaceCount);
    }
    size_t baseVertexCount = mVertices.size();
    const scene_rdl2::rdl2::Layer* pRdlLayer = tessellationParams.mRdlLayer;
    if (shouldTessellate(tessellationParams.mEnableDisplacement, pRdlLayer)) {
        PolyTopologyIdLookup topologyIdLookup(mVertices.size(),
            baseFaceVertexCount, mIndices);
        // calculate edge tessellation factor based on either user specified
        // resolution (uniform) or camera frustum info (adaptive)
        std::vector<PolyTessellationFactor> tessellationFactors =
            computeTessellationFactor(pRdlLayer, tessellationParams.mFrustums, topologyIdLookup);
        std::vector<PolyFaceTopology> faceTopologies =
            generatePolyFaceTopology(topologyIdLookup);
        MNRY_ASSERT_REQUIRE(
            tessellationFactors.size() == faceTopologies.size());
        PolyTessellatedVertexLookup tessellatedVertexLookup(faceTopologies,
            topologyIdLookup, tessellationFactors, baseFaceVertexCount);
        // generate the tessellated index buffer and
        // sample points for generating tessellated vertex buffer
        std::vector<PolyMesh::SurfaceSample> surfaceSamples;
        size_t estimatedFaceCount =
            tessellatedVertexLookup.getEstimatedFaceCount();
        PolygonMesh::IndexBuffer tessellatedIndices;
        tessellatedIndices.reserve(sQuadVertexCount * estimatedFaceCount);
        std::vector<int> tessellatedToBaseFace;
        tessellatedToBaseFace.reserve(estimatedFaceCount);
        std::vector<Vec2f> faceVaryingUv;
        faceVaryingUv.reserve(sQuadVertexCount * estimatedFaceCount);
        generateIndexBufferAndSurfaceSamples(faceTopologies,
            tessellatedVertexLookup, tessellatedIndices,
            surfaceSamples, tessellatedToBaseFace, &faceVaryingUv);
        MNRY_ASSERT_REQUIRE(
            tessellatedIndices.size() == faceVaryingUv.size());
        mTessellatedToBaseFace = std::move(tessellatedToBaseFace);
        mFaceVaryingUv = std::move(faceVaryingUv);
        // generate tessellated vertex buffer
        PolygonMesh::VertexBuffer tessellatedVertices =
            generateVertexBuffer(mVertices, mIndices, surfaceSamples);

        mBaseIndices = std::move(mIndices);
        mIndices = std::move(tessellatedIndices);
        std::swap(mVertices, tessellatedVertices);
        mIsTessellated = true;
    } else {
        mIsTessellated = false;
    }
    initAttributesAndDisplace(pRdlLayer, baseFaceCount,
        baseVertexCount, tessellationParams.mEnableDisplacement,
        tessellationParams.mFastGeomUpdate, tessellationParams.mIsBaking,
        tessellationParams.mWorld2Render);

    // reverse normals reverses orientation and negates normals
    if (mIsNormalReversed ^ mIsOrientationReversed) {
        size_t faceVertexCount = mIsTessellated ?
            sQuadVertexCount : baseFaceVertexCount;
        reverseOrientation(faceVertexCount, mIndices, mAttributes);
    }

    if (mIsNormalReversed) mAttributes->negateNormal();

    // For the baked volume shader grid, we want to set the transform
    // to world2render if the primitive is shared.
    if (getIsReference()) {
        mPrimToRender = scene_rdl2::math::toFloat(tessellationParams.mWorld2Render);
    }

    mIsMeshFinalized = true;
}

size_t
PolyMesh::getTessellatedMeshFaceCount() const
{
    if (mIsTessellated || getBaseFaceType() == MeshIndexType::QUAD) {
        return mIndices.size() / sQuadVertexCount;
    } else {
        return mIndices.size() / sTriangleVertexCount;
    }
}

void
PolyMesh::getTessellatedMesh(TessellatedMesh& tessMesh) const
{
    if (mIsTessellated) {
        // we always tessellate poly mesh to quads
        tessMesh.mIndexBufferType = MeshIndexType::QUAD;
        tessMesh.mIndexBufferDesc.mStride =
            sQuadVertexCount * sizeof(geom::Primitive::IndexType);
    } else {
        MeshIndexType meshIndexType = getBaseFaceType();
        tessMesh.mIndexBufferType = meshIndexType;
        tessMesh.mIndexBufferDesc.mStride =
            meshIndexType == MeshIndexType::QUAD ?
            sQuadVertexCount * sizeof(geom::Primitive::IndexType) :
            sTriangleVertexCount * sizeof(geom::Primitive::IndexType);
    }
    tessMesh.mFaceCount = getTessellatedMeshFaceCount();
    tessMesh.mIndexBufferDesc.mData = static_cast<const void*>(mIndices.data());
    tessMesh.mIndexBufferDesc.mOffset = 0;
    tessMesh.mVertexCount = getVertexCount();
    size_t motionSampleCount = getMotionSamplesCount();
    size_t vertexSize = sizeof(geom::PolygonMesh::VertexBuffer::value_type);
    size_t vertexStride = motionSampleCount * vertexSize;
    const void* data = mVertices.data();
    for (size_t t = 0; t < motionSampleCount; ++t) {
        size_t offset = t * vertexSize;
        tessMesh.mVertexBufferDesc.emplace_back(data, offset, vertexStride);
    }
}

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
PolyMesh::getBakedAttributeData(const TypedAttributeKey<T>& key,
                                size_t& numElements,
                                AttributeRate &newRate) const
{
    int vertsPerFace = (mIsTessellated || getBaseFaceType() == MeshIndexType::QUAD) ?
                        sQuadVertexCount : sTriangleVertexCount;
    size_t faceCount = getTessellatedMeshFaceCount();

    Attributes *attributes = getAttributes();
    size_t timeSamples = attributes->getTimeSampleCount(key);

    void* data;
    newRate = attributes->getRate(key);

    switch (attributes->getRate(key)) {
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
    case RATE_PART:
    {
        numElements = faceCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t f = 0, dstIdx = 0; f < faceCount; f++) {
            for (size_t t = 0; t < timeSamples; t++) {
                int baseFaceIdx = mIsTessellated ? mTessellatedToBaseFace[f] : f;
                int part = mFaceToPart[baseFaceIdx];
                tdata[dstIdx++] = attributes->getPart(key, part, t);
            }
        }
        break;
    }
    case RATE_UNIFORM:
    {
        numElements = faceCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t f = 0, dstIdx = 0; f < faceCount; f++) {
            for (size_t t = 0; t < timeSamples; t++) {
                int baseFaceIdx = mIsTessellated ? mTessellatedToBaseFace[f] : f;
                tdata[dstIdx++] = attributes->getUniform(key, baseFaceIdx, t);
            }
        }
        break;
    }
    case RATE_FACE_VARYING:
    {
        numElements = faceCount * vertsPerFace * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t f = 0, dstIdx = 0; f < faceCount; f++) {
            int baseFaceIdx = mIsTessellated ? mTessellatedToBaseFace[f] : f;
            for (int v = 0; v < vertsPerFace; v++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    if (mIsTessellated) {
                        Vec2f uv = mFaceVaryingUv[f * sQuadVertexCount + v];
                        T a1 = attributes->getFaceVarying(key, baseFaceIdx, 0, t);
                        T a2 = attributes->getFaceVarying(key, baseFaceIdx, 1, t);
                        T a3 = attributes->getFaceVarying(key, baseFaceIdx, 2, t);
                        T a4 = attributes->getFaceVarying(key, baseFaceIdx, 3, t);
                        tdata[dstIdx++] = bilinearInterpolate(uv, a1, a2, a3, a4);
                    } else {
                        tdata[dstIdx++] = attributes->getFaceVarying(key, baseFaceIdx, v, t);
                    }
                }
            }
        }
        break;
    }
    case RATE_VARYING:
    {
        numElements = getVertexCount() * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        if (mIsTessellated) {
            // for tessellated mesh we need to do some tricky interpolation as the
            // varying attrs are on the base mesh's vertices

            // for each tessellated face's verts we interpolate the vertex value
            // on the base face and then write the value to the array of baked vertex data
            // note that this will visit the same vertex multiple times but with the same value

            for (size_t tessFaceIdx = 0; tessFaceIdx < faceCount; tessFaceIdx++) {
                size_t tessFaceOffset = sQuadVertexCount * tessFaceIdx;

                // find the base face
                int baseFaceIdx = mTessellatedToBaseFace[tessFaceIdx];
                size_t baseFaceOffset = sQuadVertexCount * baseFaceIdx;

                for (size_t t = 0; t < timeSamples; t++) {
                    // get the varying values of the base face's four verts
                    T a1 = attributes->getVarying(key, mBaseIndices[baseFaceOffset + 0], t);
                    T a2 = attributes->getVarying(key, mBaseIndices[baseFaceOffset + 1], t);
                    T a3 = attributes->getVarying(key, mBaseIndices[baseFaceOffset + 2], t);
                    T a4 = attributes->getVarying(key, mBaseIndices[baseFaceOffset + 3], t);

                    // for each tessellated face vertex
                    for (size_t vert = 0; vert < sQuadVertexCount; vert++) {
                        // get the UVs on the base face for the current tessellated face vertex
                        Vec2f uv = mFaceVaryingUv[tessFaceOffset + vert];
                        // interpolate the point on the base face that corresponds to the
                        //  tessellated face vertex
                        T result = bilinearInterpolate(uv, a1, a2, a3, a4);
                        // update the value of the tessellated vertex
                        size_t tessVtxIdx = mIndices[tessFaceOffset + vert];
                        tdata[tessVtxIdx * timeSamples + t] = result;
                    }
                }
            }
        } else {
            // for untessellated mesh we just look up the varying value with the vertex index
            for (size_t i = 0, dstIdx = 0; i < getVertexCount(); i++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    tdata[dstIdx++] = attributes->getVarying(key, i, t);
                }
            }
        }
        break;
    }
    case RATE_VERTEX:
    {
        if (key == StandardAttributes::sNormal) {
            // reformat to FACE_VARYING if the normals are vertex rate
            numElements = mIndices.size() * timeSamples;
            T* tdata = new T[numElements];
            data = tdata;
            for (size_t i = 0, dstIdx = 0; i < mIndices.size(); i++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    tdata[dstIdx++] = attributes->getVertex(key, mIndices[i], t);
                }
            }
            newRate = RATE_FACE_VARYING;
        } else {
            numElements = getVertexCount() * timeSamples;
            T* tdata = new T[numElements];
            data = tdata;
            for (size_t i = 0, dstIdx = 0; i < getVertexCount(); i++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    tdata[dstIdx++] = attributes->getVertex(key, i, t);
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
PolyMesh::getBakedAttributeData(const TypedAttributeKey<std::string>& key,
                                size_t& numElements,
                                AttributeRate &newRate) const
{
    int vertsPerFace = (mIsTessellated || getBaseFaceType() == MeshIndexType::QUAD) ?
                        sQuadVertexCount : sTriangleVertexCount;
    size_t faceCount = getTessellatedMeshFaceCount();

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
    case RATE_PART:
    {
        numElements = faceCount * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t f = 0, dstIdx = 0; f < faceCount; f++) {
            for (size_t t = 0; t < timeSamples; t++) {
                int baseFaceIdx = mIsTessellated ? mTessellatedToBaseFace[f] : f;
                int part = mFaceToPart[baseFaceIdx];
                tdata[dstIdx++] = attributes->getPart(key, part, t);
            }
        }
        break;
    }
    case RATE_UNIFORM:
    {
        numElements = faceCount * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t f = 0, dstIdx = 0; f < faceCount; f++) {
            for (size_t t = 0; t < timeSamples; t++) {
                int baseFaceIdx = mIsTessellated ? mTessellatedToBaseFace[f] : f;
                tdata[dstIdx++] = attributes->getUniform(key, baseFaceIdx, t);
            }
        }
        break;
    }
    case RATE_FACE_VARYING:
    {
        numElements = faceCount * vertsPerFace * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t f = 0, dstIdx = 0; f < faceCount; f++) {
            int baseFaceIdx = mIsTessellated ? mTessellatedToBaseFace[f] : f;
            for (int v = 0; v < vertsPerFace; v++) {
                for (size_t t = 0; t < timeSamples; t++) {
                    if (mIsTessellated) {
                        // can't bilinearly interpolate strings like the general case,
                        // result is an empty string
                    } else {
                        tdata[dstIdx++] = attributes->getFaceVarying(key, baseFaceIdx, v, t);
                    }
                }
            }
        }
        break;
    }
    case RATE_VARYING:
    {
        numElements = getVertexCount() * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t i = 0, dstIdx = 0; i < getVertexCount(); i++) {
            for (size_t t = 0; t < timeSamples; t++) {
                if (mIsTessellated) {
                    // can't bilinearly interpolate strings like the general case,
                    // result is an empty string
                } else {
                    tdata[dstIdx++] = attributes->getVarying(key, i, t);
                }
            }
        }
        break;
    }
    case RATE_VERTEX:
    {
        numElements = getVertexCount() * timeSamples;
        std::string* tdata = new std::string[numElements];
        data = tdata;
        for (size_t i = 0, dstIdx = 0; i < getVertexCount(); i++) {
            for (size_t t = 0; t < timeSamples; t++) {
                tdata[dstIdx++] = attributes->getVertex(key, i, t);
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
PolyMesh::getBakedAttribute(const AttributeKey& key) const
{
    int vertsPerFace = (mIsTessellated || getBaseFaceType() == MeshIndexType::QUAD) ?
                        sQuadVertexCount : sTriangleVertexCount;

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
PolyMesh::getBakedMesh(BakedMesh& bakedMesh) const
{
    bakedMesh.mName = mName;

    int vertsPerFace = (mIsTessellated || getBaseFaceType() == MeshIndexType::QUAD) ?
                        sQuadVertexCount : sTriangleVertexCount;
    bakedMesh.mVertsPerFace = vertsPerFace;
    bakedMesh.mIndexBuffer.resize(mIndices.size());
    for (size_t i = 0; i < mIndices.size(); i++) {
        bakedMesh.mIndexBuffer[i] = mIndices[i];
    }

    bakedMesh.mMotionSampleCount = getMotionSamplesCount();

    bakedMesh.mVertexCount = getVertexCount();
    bakedMesh.mVertexBuffer.resize(bakedMesh.mVertexCount * bakedMesh.mMotionSampleCount);
    for (size_t v = 0; v < bakedMesh.mVertexCount; v++) {
        for (size_t t = 0; t < bakedMesh.mMotionSampleCount; t++) {
            bakedMesh.mVertexBuffer[v * bakedMesh.mMotionSampleCount + t] = mVertices(v, t);
        }
    }

    bakedMesh.mTessellatedToBaseFace = mTessellatedToBaseFace;

    bakedMesh.mFaceToPart.resize(mFaceToPart.size());
    for (size_t i = 0; i < mFaceToPart.size(); i++) {
        bakedMesh.mFaceToPart[i] = static_cast<int>(mFaceToPart[i]);
    }

    Attributes *attributes = getAttributes();
    const PrimitiveAttributeTable* patable = getPrimitiveAttributeTable();
    for (const auto& entry : *patable) {
        const AttributeKey& key = entry.first;
        if (attributes->hasAttribute(key)) {
            bakedMesh.mAttrs.push_back(getBakedAttribute(key));
        }
    }
}

int
PolyMesh::getIntersectionAssignmentId(int primID) const
{
    return getFaceAssignmentId(primID);
}

void
PolyMesh::updateVertexData(const std::vector<float>& vertexData,
        const XformSamples& prim2render)
{
    // TODO the updateVertexData mechanics is an adhoc solution
    // and need to be updated, it doesn't support motion blur
    // VertexBuffer so we can only update the first time sample,
    // this function does not guaranteed work before we revise
    // the whole update pipeline
    size_t vertexCount = mVertices.size();
    if (vertexData.size() != 3 * vertexCount) {
        size_t updateVertexCount = vertexData.size() / 3;
        scene_rdl2::logging::Logger::error("PolygonMesh ", getName(), " contains ",
           vertexCount, " vertices"
           " while update data contains ", updateVertexCount,
           " vertices. Mesh topology should remain unchanged"
           " during geometry update.");
        vertexCount = std::min(vertexCount, updateVertexCount);
    }
    for (size_t i = 0; i < vertexCount; ++i) {
        Vec3f p(vertexData[3 * i],
                vertexData[3 * i + 1],
                vertexData[3 * i + 2]);
        mVertices(i) = Vec3fa(transformPoint(prim2render[0], p), 0.f);
    }
}

void
PolyMesh::setupRecomputeVertexNormals(bool fixInvalid)
{
    if (mIsTessellated) {
        mPolyMeshCalcNv = std::unique_ptr<QuadMeshCalcNv>(
            new QuadMeshCalcNv(fixInvalid));
    } else {
        if (getBaseFaceType() == MeshIndexType::QUAD) {
            mPolyMeshCalcNv = std::unique_ptr<QuadMeshCalcNv>(
                new QuadMeshCalcNv(fixInvalid));
        } else {
            mPolyMeshCalcNv = std::unique_ptr<TriMeshCalcNv>(
                new TriMeshCalcNv(fixInvalid));
        }
    }
    // set new vertex/face info and create vn
    mPolyMeshCalcNv->set(mVertices, mIndices);
    Attributes* primitiveAttributes = getAttributes();
    size_t timeSampleCount = primitiveAttributes->getMaxTimeSampleCount();
    for (size_t vId = 0; vId < mVertices.size(); ++vId) {
        for (size_t t = 0; t < timeSampleCount; ++t) {
            primitiveAttributes->setVertex(StandardAttributes::sNormal,
                *(const Vec3f*)mPolyMeshCalcNv->getVn(vId, t), vId, t);
        }
    }
}

void
PolyMesh::recomputeVertexNormals()
{
    // recompute vn based on new vtx
    mPolyMeshCalcNv->update(mVertices);
    Attributes* primitiveAttributes = getAttributes();
    size_t timeSampleCount = primitiveAttributes->getMaxTimeSampleCount();
    for (size_t vId = 0; vId < mVertices.size(); ++vId) {
        for (size_t t = 0; t < timeSampleCount; ++t) {
            primitiveAttributes->setVertex(StandardAttributes::sNormal,
                *(const Vec3f*)mPolyMeshCalcNv->getVn(vId, t), vId, t);
        }
    }
}

BBox3f
PolyMesh::computeAABB() const
{
    if (mVertices.empty()) {
        return BBox3f(scene_rdl2::math::zero);
    }
    BBox3f result(mVertices(0).asVec3f());
    size_t motionSampleCount = getMotionSamplesCount();
    for (size_t v = 0; v < mVertices.size(); ++v) {
        for (size_t t = 0; t < motionSampleCount; ++t) {
            result.extend(mVertices(v, t));
        }
    }
    return result;
}

BBox3f
PolyMesh::computeAABBAtTimeStep(int timeStep) const
{
    if (mVertices.empty()) {
        return BBox3f(scene_rdl2::math::zero);
    }
    MNRY_ASSERT(timeStep >= 0 && timeStep < static_cast<int>(getMotionSamplesCount()), "timeStep out of range");
    BBox3f result(mVertices(0, timeStep).asVec3f());
    for (size_t v = 1; v < mVertices.size(); ++v) {
        result.extend(mVertices(v, timeStep));
    }
    return result;
}

bool
PolyMesh::shouldTessellate(bool enableDisplacement, const scene_rdl2::rdl2::Layer* pRdlLayer) const
{
    return enableDisplacement && mMeshResolution > 1 && hasDisplacementAssignment(pRdlLayer);
}

std::vector<PolyTessellationFactor>
PolyMesh::computeTessellationFactor(const scene_rdl2::rdl2::Layer *pRdlLayer,
        const std::vector<mcrt_common::Frustum>& frustums,
        const PolyTopologyIdLookup& topologyIdLookup) const
{
    size_t faceVertexCount = getBaseFaceType() == MeshIndexType::QUAD ?
        sQuadVertexCount : sTriangleVertexCount;
    size_t baseFaceCount = mIndices.size() / faceVertexCount;

    const PolygonMesh::VertexBuffer& vertices = mVertices;
    const PolygonMesh::IndexBuffer& indices = mIndices;
    std::vector<PolyTessellationFactor> tessellationFactors;
    tessellationFactors.reserve(baseFaceCount);
    // only do adaptive tessellation when adaptiveError > 0
    if (mAdaptiveError > scene_rdl2::math::sEpsilon && !frustums.empty()) {
        float pixelsPerScreenHeight =
            frustums[0].mViewport[3] - frustums[0].mViewport[1] + 1;
        float pixelsPerEdge = mAdaptiveError;
        float edgesPerScreenHeight = pixelsPerScreenHeight / pixelsPerEdge;

        std::unordered_map<int, int> edgeTessellationFactor;
        size_t motionSampleCount = vertices.get_time_steps();
        int indexOffset = 0;
        for (size_t f = 0; f < baseFaceCount; ++f) {
            scene_rdl2::math::BBox3f bbox(vertices(indices[indexOffset]).asVec3f());
            for (size_t v = 0; v < faceVertexCount; ++v) {
                for (size_t t = 0; t < motionSampleCount; ++t) {
                    bbox.extend(vertices(indices[indexOffset + v], t));
                }
            }
            const scene_rdl2::rdl2::Displacement* displacement =
                pRdlLayer->lookupDisplacement(getFaceAssignmentId(f));
            PolyTessellationFactor factor;
            if (displacement == nullptr) {
                // set tessellation factor to 0 if no displacement
                // it doesn't make sense to tessellate polygon mesh if
                // we are not doing displacement
                for (size_t v = 0; v < faceVertexCount; ++v) {
                    int vid0 = indices[indexOffset + v];
                    int vid1 = indices[indexOffset + (v + 1) % faceVertexCount];
                    int eid = topologyIdLookup.getEdgeId(vid0, vid1);
                    factor.mEdgeId[v] = eid;
                    edgeTessellationFactor[eid] = scene_rdl2::math::max(
                        0, edgeTessellationFactor[eid]);
                }
            } else {
                // enlarge the bounding box with its diag length and
                // optional user provide displacement bound padding to avoid
                // case that undisplaced face got culled out unintionally
                Vec3f pCenter = scene_rdl2::math::center(bbox);
                float padding = displacement->get(
                    scene_rdl2::rdl2::Displacement::sBoundPadding);
                if (padding < 0.0f) {
                    padding = 0.0f;
                }
                Vec3f radius(0.5f * scene_rdl2::math::length(bbox.size()) + padding);
                bbox.lower = pCenter - radius;
                bbox.upper = pCenter + radius;
                // frustum culling test
                bool inFrustum = false;
                for (size_t i = 0; i < frustums.size(); ++i) {
                    if (frustums[i].testBBoxOverlaps(bbox)) {
                        inFrustum = true;
                        break;
                    }
                }
                for (size_t v = 0; v < faceVertexCount; ++v) {
                    int vid0 = indices[indexOffset + v];
                    int vid1 = indices[indexOffset + (v + 1) % faceVertexCount];
                    int eid = topologyIdLookup.getEdgeId(vid0, vid1);
                    factor.mEdgeId[v] = eid;
                    for (size_t t = 0; t < motionSampleCount; ++t) {
                        const Vec3f& v0 = vertices(vid0, t).asVec3f();
                        const Vec3f& v1 = vertices(vid1, t).asVec3f();
                        Vec3f vMid = 0.5f * (v0 + v1);
                        int edgeFactor = 0;
                        if (inFrustum) {
                            edgeFactor = computeEdgeVertexCount(v0, v1,
                                edgesPerScreenHeight, frustums[0].mC2S);
                        }
                        edgeTessellationFactor[eid] = scene_rdl2::math::max(
                            edgeFactor, edgeTessellationFactor[eid]);
                    }
                }
            }
            tessellationFactors.push_back(factor);
            indexOffset += faceVertexCount;
        }
        // Clamp the maximum tessellation factor based on user specified
        // mesh resolution. Otherwise the tessellation factor can get out
        // of control when the edge is extremely close to camera near plane
        int maxEdgeVertexCount = mMeshResolution - 1;
        for (size_t i = 0; i < tessellationFactors.size(); ++i) {
            for (size_t e = 0; e < faceVertexCount; ++e) {
                int eid = tessellationFactors[i].mEdgeId[e];
                tessellationFactors[i].mEdgeFactor[e] = scene_rdl2::math::clamp(
                    edgeTessellationFactor[eid], 0, maxEdgeVertexCount);
            }
        }
    } else {
        // uniform tessellation
        int edgeVertexCount = scene_rdl2::math::max(0, mMeshResolution - 1);
        for (size_t f = 0; f < baseFaceCount; ++f) {
            PolyTessellationFactor factor;
            for (size_t v = 0; v < faceVertexCount; ++v) {
                factor.mEdgeFactor[v] = edgeVertexCount;
            }
            tessellationFactors.push_back(factor);
        }
    }
    return tessellationFactors;
}

std::vector<PolyFaceTopology>
PolyMesh::generatePolyFaceTopology(
        const PolyTopologyIdLookup& topologyIdLookup) const
{
    size_t faceVertexCount = getBaseFaceType() == MeshIndexType::QUAD ?
        sQuadVertexCount : sTriangleVertexCount;
    size_t baseFaceCount = mIndices.size() / faceVertexCount;
    std::vector<PolyFaceTopology> faceTopologies;
    faceTopologies.reserve(baseFaceCount);
    size_t indexOffset = 0;
    for (size_t f = 0; f < baseFaceCount; ++f) {
        PolyFaceTopology faceTopology;
        for (size_t v = 0; v < faceVertexCount; ++v) {
            int vid0 = mIndices[indexOffset + v];
            int vid1 = mIndices[indexOffset + (v + 1) % faceVertexCount];
            faceTopology.mCornerVertexId[v] = vid0;
            faceTopology.mEdgeId[v] = topologyIdLookup.getEdgeId(vid0, vid1);
        }
        faceTopologies.push_back(faceTopology);
        indexOffset += faceVertexCount;
    }
    return faceTopologies;
}

void
PolyMesh::initAttributesAndDisplace(const scene_rdl2::rdl2::Layer *pRdlLayer,
        size_t baseFaceCount, size_t varyingsCount, bool enableDisplacement,
        bool realtimeMode, bool isBaking,
        const scene_rdl2::math::Mat4d& world2render)
{
    auto& primitiveAttributeTable = mPolyMeshData->mPrimitiveAttributeTable;
    MeshIndexType baseFaceType = getBaseFaceType();
    size_t faceVertexCount = baseFaceType == MeshIndexType::QUAD ?
        sQuadVertexCount : sTriangleVertexCount;
    std::vector<size_t> faceVaryingsCount(baseFaceCount, faceVertexCount);
    size_t verticesCount = mVertices.size();
    // figure out whether we need to calculate smooth shading normal
    bool calculateSmoothNormal = false;
    bool hasDisplacement = false;
    if (realtimeMode) {
        // for real time mode we always calculate smooth normal,
        // and we don't apply displacement in this mode now
        calculateSmoothNormal = true;
    } else {
        // if there is displacement, we need to calculate smooth shading normal
        hasDisplacement = enableDisplacement && hasDisplacementAssignment(pRdlLayer);
        calculateSmoothNormal = hasDisplacement;
        if (getSmoothNormal() &&
            !primitiveAttributeTable.hasAttribute(StandardAttributes::sNormal)) {
            calculateSmoothNormal = true;
        }
    }
    if (calculateSmoothNormal) {
        // when we decide to calculate the smooth shading normal,
        // the attribute rate need to be explicit to RATE_VERTEX
        // due to the normal computation utility logic
        if (primitiveAttributeTable.hasAttribute(StandardAttributes::sNormal)) {
            primitiveAttributeTable.erase(StandardAttributes::sNormal);
        }
        // Create empty shading normal buffer that we will calculate during tessellation stage.
        // (Note: we only support up to 2 motion samples at present)
        size_t motionSampleCount = std::min((int)getMotionSamplesCount(), 2);
        std::vector<std::vector<Vec3f>> shadingNormal(motionSampleCount);
        for (size_t t = 0; t < motionSampleCount; ++t) {
            shadingNormal[t].resize(verticesCount);
        }
        primitiveAttributeTable.addAttribute(StandardAttributes::sNormal,
            RATE_VERTEX, std::move(shadingNormal));
    }
    // interleave PrimitiveAttributeTable
    setAttributes(Attributes::interleave(primitiveAttributeTable,
        mPartCount, baseFaceCount, varyingsCount,
        faceVaryingsCount, verticesCount));

    getAttributes()->transformAttributes(mPolyMeshData->mXforms,
                                         mPolyMeshData->mShutterOpenDelta,
                                         mPolyMeshData->mShutterCloseDelta,
                                         {{StandardAttributes::sNormal, Vec3Type::NORMAL},
                                         {StandardAttributes::sdPds, Vec3Type::VECTOR},
                                         {StandardAttributes::sdPdt, Vec3Type::VECTOR}});

    if (calculateSmoothNormal) {
        setupRecomputeVertexNormals(!realtimeMode);
    }

    if (hasDisplacement) {
        displaceMesh(pRdlLayer, world2render);
    }
    // if it's not real time mode, we don't need the smooth normal utility
    // after shading normal computation is done
    if (!realtimeMode) {
        mPolyMeshCalcNv.reset();
    }

    if (!isBaking) {
        // Don't need this data anymore after primitive attributes are initialized,
        // except if we are baking geometry.
        mPolyMeshData.reset();
    }
}

void
PolyMesh::displaceMesh(const scene_rdl2::rdl2::Layer* pRdlLayer,
                       const scene_rdl2::math::Mat4d& world2render)
{
    struct VertexToDisplace {
        VertexToDisplace(): mAssignmentId(-1), mFaceId(0), mVIndex(0)
        {}

        int mAssignmentId;
        uint32_t mFaceId;
        uint32_t mVIndex;
    };

    MeshIndexType baseFaceType = getBaseFaceType();
    size_t faceVertexCount =
        (mIsTessellated || baseFaceType == MeshIndexType::QUAD) ?
        sQuadVertexCount : sTriangleVertexCount;

    size_t vertexCount = mVertices.size();
    std::vector<VertexToDisplace> toDisplace(vertexCount);
    for (size_t f = 0; f < getTessellatedMeshFaceCount(); ++f) {
        int assignmentId = getFaceAssignmentId(f);
        if (assignmentId == -1 ||
            pRdlLayer->lookupDisplacement(assignmentId) == nullptr) {
            continue;
        }
        for (size_t v = 0; v < faceVertexCount; ++v) {
            uint32_t vid = mIndices[f * faceVertexCount + v];
            if (toDisplace[vid].mAssignmentId == -1) {
                toDisplace[vid].mAssignmentId = assignmentId;
                toDisplace[vid].mFaceId = f;
                toDisplace[vid].mVIndex = v;
            }
        }
    }

    size_t motionSampleCount = getMotionSamplesCount();
    PolygonMesh::VertexBuffer displacementResult(vertexCount, motionSampleCount);

    tbb::blocked_range<size_t> range =
        tbb::blocked_range<size_t>(0, vertexCount);

    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
        shading::TLState *shadingTls = MNRY_VERIFY(tls->mShadingTls.get());
        Intersection isect;
        for (size_t v = r.begin(); v < r.end(); ++v) {
            int assignmentId = toDisplace[v].mAssignmentId;
            if (assignmentId == -1) {
                continue;
            }
            const scene_rdl2::rdl2::Displacement* displacement =
                pRdlLayer->lookupDisplacement(assignmentId);
            const scene_rdl2::rdl2::Geometry* geometry =
                pRdlLayer->lookupGeomAndPart(assignmentId).first;

            // get primitive attribute table
            const AttributeTable* table = nullptr;
            if (displacement->hasExtension()) {
                const shading::RootShader& rootShader =
                    displacement->get<shading::RootShader>();
                table = rootShader.getAttributeTable();
            }

            int tessFaceId = toDisplace[v].mFaceId;
            int baseFaceId = getBaseFaceId(tessFaceId);
            int vIndex = toDisplace[v].mVIndex;
            int vid, vid1, vid2, vid3;
            Vec2f st, st1, st2, st3;
            // need to pick one of the triangles on quad to calculate
            // dpds/dpdt through positions, st coordinates on three vertices
            getNeighborVertices(baseFaceId, tessFaceId, vIndex,
                vid, vid1, vid2, vid3, st, st1, st2, st3);

            for (size_t t = 0; t < motionSampleCount; ++t) {
                Vec3f position = mVertices(vid, t);
                Vec3f p1 = mVertices(vid1, t);
                Vec3f p2 = mVertices(vid2, t);
                Vec3f p3 = mVertices(vid3, t);


                Vec3f normal = scene_rdl2::math::normalize(scene_rdl2::math::cross(p2 - p1, p3 - p1));
                // degenerated surface, assign a valid but meaningless
                // value as fallback
                if (!scene_rdl2::math::isFinite(normal)) {
                    normal = Vec3f(0, 0, 1);
                }
                Vec3f dpdst[2];
                if (!computeTrianglePartialDerivatives(
                    p1, p2, p3, st1, st2, st3, dpdst)) {
                    scene_rdl2::math::ReferenceFrame frame(normal);
                    dpdst[0] = frame.getX();
                    dpdst[1] = frame.getY();
                }
                Vec2f dSt0 = 0.5f * (st2 - st1);
                Vec2f dSt1 = 0.5f * (st3 - st1);

                if (getIsReference()) {
                    // If this primitive is referenced by another geometry (i.e. instancing)
                    // then these attributes are given to us in local/object space.   Our
                    // shaders expect all State attributes to be given in render space
                    // so we transform them here by world2render.   The local2world transform
                    // is assumed to be identity for references.
                    position = scene_rdl2::math::transformPoint(world2render, position);
                    dpdst[0] = scene_rdl2::math::transformVector(world2render, dpdst[0]);
                    dpdst[1] = scene_rdl2::math::transformVector(world2render, dpdst[1]);
                }

                isect.initDisplacement(tls, table, geometry,
                    pRdlLayer, assignmentId, position, normal,
                    dpdst[0], dpdst[1], st, dSt0[0], dSt1[0], dSt0[1], dSt1[1]);
                fillDisplacementAttributes(tessFaceId, vIndex, isect);

                Vec3f displace;
                shading::displace(displacement, shadingTls, shading::State(&isect), &displace);
                // can't add this result directly to vertex buffer here...
                // otherwise it will cause wrong dpds/dpdt calculation since
                // displacement shaders expect pre-displaced dpds/dpdt values
                displacementResult(vid, t) = Vec3fa(displace, 0.f);
            }
        }
    });
    // now add the displacement result back to vertex buffer
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t v = r.begin(); v < r.end(); ++v) {
            int assignmentId = toDisplace[v].mAssignmentId;
            if (assignmentId == -1) {
                continue;
            }
            uint32_t faceId = toDisplace[v].mFaceId;
            uint32_t vIndex = toDisplace[v].mVIndex;
            uint32_t vid = mIndices[faceVertexCount * faceId + vIndex];
            for (size_t t = 0; t < motionSampleCount; ++t) {
                mVertices(vid, t) += displacementResult(vid, t);
            }
        }
    });

    // recompute the shading normal after displacement
    if (mPolyMeshCalcNv) {
        recomputeVertexNormals();
    } else {
        bool fixInvalid = true;
        setupRecomputeVertexNormals(fixInvalid);
    }
}


//------------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray


