// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Mesh.h"

#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/Util.h>

#include <moonray/rendering/bvh/shading/Attributes.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <openvdb/tools/Composite.h>
#include <openvdb/math/Proximity.h>
#include <openvdb/tools/MeshToVolume.h>

namespace moonray {
namespace geom {
namespace internal {

openvdb::VectorGrid::Ptr
Mesh::createTriMeshVelocityGrid(float interiorBandwidth,
                                const TessellatedMesh& mesh,
                                size_t motionSample,
                                float invFps)
{
    // Convert this mesh to a vdb grid. Inputs are the mesh, the world to index space transform, the exterior bandwidth
    // -- which is aribitrarily set to 3 because that is the default setting in openvdb --, the interior bandwith --
    // which is the number of voxels wide / 2, and flags.

    const size_t faceVertexCount = 3;
    const size_t faceCount = mesh.mFaceCount;

    std::vector<openvdb::Vec3s> vertices(mesh.mVertexCount);
    std::vector<openvdb::Vec3I> indices(faceCount);

    // Grab vertex buffer of this motion sample.
    const float* vertexBuffer = reinterpret_cast<const float *>(mesh.mVertexBufferDesc[motionSample].mData) +
        mesh.mVertexBufferDesc[motionSample].mOffset / sizeof(float);
    const int* indexBuffer = (const int* )mesh.mIndexBufferDesc.mData;

    // Create openvdb index buffer
    for (size_t f = 0; f < faceCount; ++f) {
        indices[f] = openvdb::Vec3I(indexBuffer[f * faceVertexCount],
                                    indexBuffer[f * faceVertexCount + 1],
                                    indexBuffer[f * faceVertexCount + 2]);
    }

    // Create openvdb vertex buffer
    size_t vert = 0;
    for(size_t v = 0; v < mesh.mVertexCount; ++v) {
        vertices[v] =
            openvdb::Vec3s(vertexBuffer[vert], vertexBuffer[vert+1], vertexBuffer[vert+2]);
        vert += mesh.mVertexBufferDesc[motionSample].mStride / sizeof(float);
    }

    // Set up openvdb mesh.
    openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec3I> meshAdapter(vertices, indices);
    // The primitiveIndexGrid stores the index of the primitive (face) that is closest to the voxel.
    openvdb::Int32Grid::Ptr primitiveIndexGrid(new openvdb::Int32Grid(0));
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform();
    primitiveIndexGrid->setTransform(transform);
    openvdb::tools::meshToVolume<openvdb::FloatGrid>(meshAdapter,
                                                     *transform,
                                                     3.0f,
                                                     interiorBandwidth,
                                                     0,
                                                     primitiveIndexGrid.get());

    openvdb::VectorGrid::Ptr velocityGrid = openvdb::Vec3SGrid::create();

    // If a shared primitive, the velocity grid is baked in local space.
    const scene_rdl2::math::Mat4f primToRender = getIsReference() ? scene_rdl2::math::Mat4f(scene_rdl2::math::one) : mPrimToRender;

    // Fill in velocity grid
    openvdb::tools::transformValues(primitiveIndexGrid->cbeginValueOn(), *velocityGrid,
        [&](const openvdb::Int32Grid::ValueOnCIter& it, openvdb::Vec3SGrid::Accessor& accessor) {

            int faceid = it.getValue();

            const openvdb::math::Coord& coord = it.getCoord();
            const openvdb::Vec3d voxelCenter(coord.x(), coord.y(), coord.z());

            const openvdb::Vec3d p0(vertices[indices[faceid].x()]);
            const openvdb::Vec3d p1(vertices[indices[faceid].y()]);
            const openvdb::Vec3d p2(vertices[indices[faceid].z()]);
            openvdb::Vec3d uvw;
            openvdb::math::closestPointOnTriangleToPoint(p0, p1, p2, voxelCenter, uvw);

            // Assume only one velocity buffer at t = 0.
            const scene_rdl2::math::Vec3f v0 = scene_rdl2::math::transformVector(primToRender, getVelocity(faceid, 0, float(motionSample)));
            const scene_rdl2::math::Vec3f v1 = scene_rdl2::math::transformVector(primToRender, getVelocity(faceid, 1, float(motionSample)));
            const scene_rdl2::math::Vec3f v2 = scene_rdl2::math::transformVector(primToRender, getVelocity(faceid, 2, float(motionSample)));

            const scene_rdl2::math::Vec3f v = invFps * (v0 * float(uvw[0]) + v1 * float(uvw[1]) + v2 * float(uvw[2]));
            accessor.setValue(coord, openvdb::Vec3d(v.x, v.y, v.z));

    });
    velocityGrid->pruneGrid();

    return velocityGrid;
}

openvdb::VectorGrid::Ptr
Mesh::createQuadMeshVelocityGrid(float interiorBandwidth,
                                 const TessellatedMesh& mesh,
                                 size_t motionSample,
                                 float invFps)
{
    // Convert this mesh to a vdb grid. Inputs are the mesh, the world to index space transform, the exterior bandwidth
    // -- which is aribitrarily set to 3 because that is the default setting in openvdb --, the interior bandwith --
    // which is the number of voxels wide / 2, and flags.

    const size_t faceVertexCount = 4;
    const size_t faceCount = mesh.mFaceCount;

    std::vector<openvdb::Vec3s> vertices(mesh.mVertexCount);
    std::vector<openvdb::Vec4I> indices(faceCount);

    // Grab vertex buffer of this motion sample.
    const float* vertexBuffer = reinterpret_cast<const float *>(mesh.mVertexBufferDesc[motionSample].mData) +
        mesh.mVertexBufferDesc[motionSample].mOffset / sizeof(float);
    const int* indexBuffer = (const int* )mesh.mIndexBufferDesc.mData;

    // Create openvdb index buffer
    for (size_t f = 0; f < faceCount; ++f) {
        indices[f] = openvdb::Vec4I(indexBuffer[f * faceVertexCount],
                                    indexBuffer[f * faceVertexCount + 1],
                                    indexBuffer[f * faceVertexCount + 2],
                                    indexBuffer[f * faceVertexCount + 3]);
    }

    // Create openvdb vertex buffer
    size_t vert = 0;
    for(size_t v = 0; v < mesh.mVertexCount; ++v) {
        vertices[v] =
            openvdb::Vec3s(vertexBuffer[vert], vertexBuffer[vert+1], vertexBuffer[vert+2]);
        vert += mesh.mVertexBufferDesc[motionSample].mStride / sizeof(float);
    }

    // Set up openvdb mesh.
    openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> meshAdapter(vertices, indices);
    // The primitiveIndexGrid stores the index of the primitive (face) that is closest to the voxel.
    openvdb::Int32Grid::Ptr primitiveIndexGrid(new openvdb::Int32Grid(0));
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform();
    primitiveIndexGrid->setTransform(transform);
    openvdb::tools::meshToVolume<openvdb::FloatGrid>(meshAdapter,
                                                     *transform,
                                                     3.f,
                                                     interiorBandwidth,
                                                     0,
                                                     primitiveIndexGrid.get());

    openvdb::VectorGrid::Ptr velocityGrid = openvdb::Vec3SGrid::create();

    // If a shared primitive, the velocity grid is baked in local space.
    // In that case mPrimToRender is the worldToRender matrix used for shading,
    // which we do not want to apply.
    const scene_rdl2::math::Mat4f primToRender = getIsReference() ? scene_rdl2::math::Mat4f(scene_rdl2::math::one) : mPrimToRender;

    // Fill in velocity grid
    openvdb::tools::transformValues(primitiveIndexGrid->cbeginValueOn(), *velocityGrid,
        [&](const openvdb::Int32Grid::ValueOnCIter& it, openvdb::Vec3SGrid::Accessor& accessor) {

            int faceid = it.getValue();

            const openvdb::math::Coord& coord = it.getCoord();
            const openvdb::Vec3d voxelCenter(coord.x(), coord.y(), coord.z());

            // first triangle
            const openvdb::Vec3d p0(vertices[indices[faceid].x()]);
            const openvdb::Vec3d p1(vertices[indices[faceid].y()]);
            const openvdb::Vec3d p2(vertices[indices[faceid].z()]);
            openvdb::Vec3d uvw;
            const openvdb::Vec3d closestP1 =
                openvdb::math::closestPointOnTriangleToPoint(p0, p1, p2, voxelCenter, uvw);
            float dist1 = (voxelCenter - closestP1).lengthSqr();
            int index0 = 0;
            int index1 = 1;
            int index2 = 2;

            // second triangle
            const openvdb::Vec3d p3(vertices[indices[faceid].w()]);
            openvdb::Vec3d uvw2;
            const openvdb::Vec3d closestP2 =
                openvdb::math::closestPointOnTriangleToPoint(p0, p3, p2, voxelCenter, uvw2);
            float dist2 = (voxelCenter - closestP2).lengthSqr();

            if (dist2 < dist1) {
                // use second triangle
                index1 = 3;
                uvw = uvw2;
            }

            // Assume only one velocity buffer at t = 0.
            const scene_rdl2::math::Vec3f v0 = scene_rdl2::math::transformVector(primToRender, getVelocity(faceid, index0,
                float(motionSample)));
            const scene_rdl2::math::Vec3f v1 = scene_rdl2::math::transformVector(primToRender, getVelocity(faceid, index1,
                float(motionSample)));
            const scene_rdl2::math::Vec3f v2 = scene_rdl2::math::transformVector(primToRender, getVelocity(faceid, index2,
                float(motionSample)));

            const scene_rdl2::math::Vec3f v = invFps * (v0 * float(uvw[0]) + v1 * float(uvw[1]) + v2 * float(uvw[2]));
            accessor.setValue(coord, openvdb::Vec3d(v.x, v.y, v.z));
    });
    velocityGrid->pruneGrid();

    return velocityGrid;
}

void
Mesh::createVelocityGrid(const float interiorBandwidth,
                         const MotionBlurParams& motionBlurParams,
                         const std::vector<int>& volumeIds)
{
    if(!getAttributes()->isSupported(shading::StandardAttributes::sVelocity)) {
        // don't bake velocity grid if velocity does not exist
        return;
    }

    size_t threadCount = mcrt_common::getMaxNumTLS();

    TessellatedMesh mesh;
    getTessellatedMesh(mesh);

    // Meshes that are bound to volume shaders and exhibit motion blur should have 2 motion samples. If this is the
    // case we must bake a velocity grid whose bounds contain the full motion of the mesh. We do this by baking
    // two velocity grids, one for each motion sample, and then compositing them by taking the max value of
    // the grids.
    size_t motionSampleCount = mesh.mVertexBufferDesc.size();

    // The input velcity on the mesh is in units of ditsance / second. We must divide out the seconds to make
    // it in units of distance / frame.
    float invFps = motionBlurParams.getInvFps();

    openvdb::VectorGrid::Ptr velocityGrid;
    if (mesh.mIndexBufferType == MeshIndexType::TRIANGLE) {
        // first motion sample
        velocityGrid = createTriMeshVelocityGrid(interiorBandwidth, mesh, 0, invFps);
        if (motionSampleCount > 1)
        {
            // For volumes, there should only be two motion samples because motion blur is created
            // by velocity.
            MNRY_ASSERT(motionSampleCount == 2);
            // second motion sample
            openvdb::VectorGrid::Ptr secondVelocityGrid = createTriMeshVelocityGrid(interiorBandwidth, mesh,
                motionSampleCount - 1, invFps);
            // combine velocity grids by taking max value.
            openvdb::tools::compMax(*velocityGrid, *secondVelocityGrid);
        }
    } else {
        // first motion sample
        velocityGrid = createQuadMeshVelocityGrid(interiorBandwidth, mesh, 0, invFps);
        if (motionSampleCount > 1)
        {
            // For volumes, there should only be two motion samples because motion blur is created
            // by velocity.
            MNRY_ASSERT(motionSampleCount == 2);
            // second motion sample
            openvdb::VectorGrid::Ptr secondVelocityGrid = createQuadMeshVelocityGrid(interiorBandwidth, mesh,
                motionSampleCount - 1, invFps);
            // combine velocity grids by taking max value.
            openvdb::tools::compMax(*velocityGrid, *secondVelocityGrid);
        }
    }

    mVdbVelocity->setShutterValues(motionBlurParams.getShutterOpen(),
        motionBlurParams.getShutterClose() - motionBlurParams.getShutterOpen());
    mVdbVelocity->setVelocityGrid(velocityGrid, volumeIds);
}

bool
Mesh::faceHasAssignment(uint faceId)
{
    // check if the part this face belongs to has the material assigned
    int assignmentId =
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
        mLayerAssignmentId.getConstId() :
        mLayerAssignmentId.getVaryingId()[faceId];
    return assignmentId != -1;
}

// get the memory in byte
size_t
Mesh::getMemory() const
{
    return sizeof(Mesh) - sizeof(NamedPrimitive) + NamedPrimitive::getMemory();
}

const scene_rdl2::rdl2::Material *
Mesh::getIntersectionMaterial(const scene_rdl2::rdl2::Layer *pRdlLayer,
        const mcrt_common::Ray &ray) const
{
    int layerAssignmentId = getIntersectionAssignmentId(ray.primID);
    const scene_rdl2::rdl2::Material *pMaterial = MNRY_VERIFY(pRdlLayer->lookupMaterial(layerAssignmentId));
    return pMaterial;
}

void
Mesh::reverseOrientation(const size_t faceVertexCount,
                         std::vector<uint32_t>& indices,
                         std::unique_ptr<shading::Attributes>& attributes)
{
    size_t faceCount = indices.size() / faceVertexCount;
    size_t indexOffset = 0;
    for (size_t i = 0; i < faceCount; i++) {
        std::reverse(indices.begin() + indexOffset,
            indices.begin() + indexOffset + faceVertexCount);
        indexOffset += faceVertexCount;
    }

    attributes->reverseFaceVaryingAttributes();
}

void
Mesh::reverseOrientation(const std::vector<uint32_t>& faceVertexCount,
                     std::vector<uint32_t>& indices,
                     std::unique_ptr<shading::Attributes>& attributes)
{
    size_t faceCount = faceVertexCount.size();
    size_t indexOffset = 0;
    for (size_t i = 0; i < faceCount; i++) {
        MNRY_ASSERT(faceVertexCount[i] == 3 || faceVertexCount[i] == 4);
        std::reverse(indices.begin() + indexOffset, indices.begin() + indexOffset + faceVertexCount[i]);
        indexOffset += faceVertexCount[i];
    }

    attributes->reverseFaceVaryingAttributes();
}


Vec3f
Mesh::computeMotion(const VertexBuffer<Vec3fa, InterleavedTraits> &vertBuf,
                    uint32_t id1, uint32_t id2, uint32_t id3,
                    float vertexW1, float vertexW2, float vertexW3,
                    const mcrt_common::Ray &ray) const
{
    Vec3f pos0, pos1;
    const Vec3f *pos1Ptr = nullptr;
    if (isMotionBlurOn()) {
        const Vec3f pShutterOpen =
            vertexW1 * vertBuf(id1, 0) + vertexW2 * vertBuf(id2, 0) + vertexW3 * vertBuf(id3, 0);
        const Vec3f pShutterClose =
            vertexW1 * vertBuf(id1, 1) + vertexW2 * vertBuf(id2, 1) + vertexW3 * vertBuf(id3, 1);
        pos0 = lerp(pShutterOpen, pShutterClose, ray.getTime() - sHalfDt);
        pos1 = lerp(pShutterOpen, pShutterClose, ray.getTime() + sHalfDt);
        pos1Ptr = &pos1;
    } else {
        pos0 = vertexW1 * vertBuf(id1, 0) + vertexW2 * vertBuf(id2, 0) + vertexW3 * vertBuf(id3, 0);
        pos1Ptr = nullptr;
    }

    // Motion vectors only support a single instance level, hence we only care
    // about ray.instance0.
    const Instance *instance = (ray.isInstanceHit())?
        static_cast<const Instance *>(ray.ext.instance0OrLight) : nullptr;
    return computePrimitiveMotion(pos0, pos1Ptr, ray.getTime(), instance);
}

//----------------------------------------------------------------------------

static void
interpolatePosInTri(const Vec2f &p, const Vec2f &a, const Vec2f &b, const Vec2f &c,
                    const Vec3f &posA, const Vec3f &posB, const Vec3f &posC,
                    const Vec3f *nrmA, const Vec3f *nrmB, const Vec3f *nrmC,
                    Vec3fa *posResult, Vec3f *nrmResult)
{
    // p is a pos in st space, a, b, and c are st coordinates of a triangle,
    // which are probably in ccw order, but this code does not assume that.
    // we first determine if p is inside the triangle using 3 half-plane
    // tests.  if p is on the inside of all three half planes
    // defined by the edges of the triangle, then p is inside the triangle.
    // once we determine that p is inside the triangle, we compute
    // barycentric coordinates for p and use them to create an interpolated
    // result.  if p is outside the triange, we leave the results untouched.

    // A little drawing for reference
    //
    //                     a
    //                     *
    //                 -      -
    //             -    *p        -
    //          *-  -  -  -  -  -  - *
    //          b                    c

    bool isNegAB = dot(Vec2f(a.y - b.y, b.x - a.x), p - (a + b) * 0.5f) < 0.f;
    bool isNegBC = dot(Vec2f(b.y - c.y, c.x - b.x), p - (b + c) * 0.5f) < 0.f;
    bool isNegCA = dot(Vec2f(c.y - a.y, a.x - c.x), p - (a + c) * 0.5f) < 0.f;

    if (isNegAB == isNegBC && isNegBC == isNegCA) {
        // point is inside triangle, compute barycentric coordinates
        const float areaABC = scene_rdl2::math::abs((c.x - b.x) * (a.y - b.y) - (a.x - b.x) * (c.y - b.y)); // actually 2x
        if (areaABC > 0) { // else degenerate
            const float areaPBC = scene_rdl2::math::abs((p.x - b.x) * (c.y - b.y) - (c.x - b.x) * (p.y - b.y));
            const float areaAPC = scene_rdl2::math::abs((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y));
            const float invAreaABC = 1.f / areaABC;
            float u = areaPBC * invAreaABC; // weight of a
            float v = areaAPC * invAreaABC; // weight of b
            float w = 1.f - u - v; // weight of c
            MNRY_ASSERT(
                u >= -1e-5f && u <= 1.0f + 1e-5f &&
                v >= -1e-5f && v <= 1.0f + 1e-5f &&
                w >= -1e-5f && w <= 1.0f + 1e-5f);
            // clamp to avoid float point precision error, the uvw coordinate
            // should be all within [0, 1] range mathematically
            u = scene_rdl2::math::clamp(u);
            v = scene_rdl2::math::clamp(v);
            w = scene_rdl2::math::clamp(w);
            // now compute the interpolated position result
            MNRY_ASSERT(posResult);
            *posResult = Vec3fa(posA * u + posB * v + posC * w, 0.f);
            // set alpha to 1.f to indicate that this pixel has
            // valid position data
            posResult->w = 1.f;

            // and the optional normal vector result
            if (nrmResult) {
                MNRY_ASSERT(nrmA && nrmB && nrmC);
                *nrmResult = *nrmA * u + *nrmB * v + *nrmC * w;
            }
        }
    }
}

void
Mesh::rasterizeTrianglePos(const scene_rdl2::math::BBox2f &roiST, int width, int height,
                           const Vec2f &a, const Vec2f &b, const Vec2f &c,
                           const Vec3f &posA, const Vec3f &posB, const Vec3f &posC,
                           const Vec3f *nrmA, const Vec3f *nrmB, const Vec3f *nrmC,
                           Vec3fa *posResult, Vec3f *nrmResult)
{
    const scene_rdl2::math::BBox2i roiXY(scene_rdl2::math::Vec2i(floor(roiST.lower.x * (width - 1)),
                                         floor(roiST.lower.y * (height - 1))),
                             scene_rdl2::math::Vec2i(ceil(roiST.upper.x * (width - 1)),
                                         ceil(roiST.upper.y * (height - 1))));
    for (int y = roiXY.lower.y; y <= roiXY.upper.y; ++y) {
        const float t = float(y) / (height - 1);
        for (int x = roiXY.lower.x; x <= roiXY.upper.x; ++x) {
            const Vec2f st(float(x) / (width - 1), t);
            MNRY_ASSERT(posResult);
            Vec3fa *pos = posResult + y * width + x;
            Vec3f *nrm = nrmResult? nrmResult + y * width + x : nullptr;
            interpolatePosInTri(st, a, b, c, posA, posB, posC, nrmA, nrmB, nrmC, pos, nrm);
        }
    }
}

bool
Mesh::udimxform(int udim, Vec2f &st)
{
    const int minS = (udim - 1001) % 10;
    const int minT = (udim - 1001) / 10;
    if (st.x >= minS && st.x <= minS + 1 &&
        st.y >= minT && st.y <= minT + 1) {
        // accept it
        st.x -= minS;
        st.y -= minT;
        st.x = scene_rdl2::math::clamp(st.x);
        st.y = scene_rdl2::math::clamp(st.y);
        return true;
    }
    return false;
}

//----------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray


