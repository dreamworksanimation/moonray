// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VdbVolume.cc
///

#include "VdbVolume.h"

#include <moonray/rendering/geom/prim/VolumeSampleInfo.h>
#include <moonray/rendering/geom/prim/VolumeTransition.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/stdmemory.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/io/Stream.h>
#include <openvdb/math/Ray.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/tools/ValueTransformer.h> // for tools::foreach()

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

using scene_rdl2::logging::Logger;

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;
using namespace scene_rdl2::math;

namespace {

template<typename GridType>
std::vector<float>
getGridLuminance(const VDBSampler<openvdb::VectorGrid>& emissionSampler,
                 const GridType& emissionGrid)
{
    std::vector<float> values;
    for (auto it = emissionGrid.cbeginValueOn(); it; ++it) {
        const openvdb::math::Coord& xyz = it.getCoord();
        const auto rgba = emissionSampler.mSamplers[0].mAccessor.getValue(it.getCoord());
        values.push_back(luminance(Color(rgba.x(), rgba.y(), rgba.z())));
    }
    return values;
}

} // anonymous namespace

// The algorithm openvdb::tools::VolumeRayIntersector uses to collect all
// volume intersection intervals is 3D DDA traversal. Since this utility tool
// doesn't support voxel grid with non-uniform voxels,
// we need to reinvent the wheel to:
// 1. create a render space bounding box to enclose the VDB grid
// 2. do 3D DDA traversal ourselves during ray tracing
//
// TODO currently we use coarser resolution grid to record whether
// a particular grid entry contains any active voxel
// This grid can be further used to store meta data like maximum/minimum sigmaT
// in the grid entry, which will be helpful for other advanced volume rendering
// techniques like residual/ratio tracking and decomposition tracking
class DDAIntersector
{
public:
    DDAIntersector(const scene_rdl2::math::Mat4f primToRender[2], const scene_rdl2::math::Mat4f renderToPrim[2],
            const openvdb::FloatGrid::Ptr& grid,
            const std::vector<openvdb::FloatGrid::ConstAccessor>& topologyAccessors,
            bool isMotionBlurOn):
        mHasActiveVoxel(nullptr),
        mIsMotionBlurOn(isMotionBlurOn)
    {
        // Copy transformations internally
        mPrimToRender[0] = primToRender[0];
        mPrimToRender[1] = primToRender[1];
        mRenderToPrim[0] = renderToPrim[0];
        mRenderToPrim[1] = renderToPrim[1];

        // index space bbox
        openvdb::math::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
        // Need to expand the bbox by one grid cell in all directions for 2 reasons:
        // 1) The bbox coords are inclusive, so subtracting max-min gives a
        //    size that's one unit too small.  Thus we need to add 1 cell to
        //    the max coord.
        // 2) There seems to be a half cell offset with the sampling so we need
        //    to pad the bbox by -1 cell in the min direction.  (i.e. the integer
        //    cell coord means the middle of the cell, not the corner.)
        bbox.expand(1);

        const auto& gridXform = grid->transform();
        // primitive space bbox
        auto aabb = gridXform.indexToWorld(bbox);
        mAABB = BBox3f(Vec3f(aabb.min().x(), aabb.min().y(), aabb.min().z()),
                       Vec3f(aabb.max().x(), aabb.max().y(), aabb.max().z()));
        Vec3f dim = mAABB.size();
        // TODO may worth experimenting other heuristic to figure out
        // coarse grid resolution
        float unitWidth = scene_rdl2::math::max(dim.x, scene_rdl2::math::max(dim.y, dim.z)) /
            sMaxResolution;
        for (int axis = 0; axis < 3; ++axis) {
            mRes[axis] = static_cast<int>(round(dim[axis] / unitWidth));
            mRes[axis] = scene_rdl2::math::clamp(mRes[axis], 1, sMaxResolution);
            mUnitWidth[axis] = dim[axis] / mRes[axis];
            mInvUnitWIdth[axis] = (mUnitWidth[axis] == 0.0f) ?
                0.0f : 1.0f / mUnitWidth[axis];
        }
        int nTotal = mRes[0] * mRes[1] * mRes[2];
        mHasActiveVoxel.reset(new bool[nTotal]);
        mMemory = nTotal * sizeof(bool);

        tbb::blocked_range<size_t> range =
            tbb::blocked_range<size_t>(0, nTotal);
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t offset = r.begin(); offset < r.end(); ++offset) {
                int w = offset / mRes[0];
                int x = offset - w * mRes[0];
                int z = w / mRes[1];
                int y = w - z * mRes[1];
                // Compute the bounding box of this mHasActiveVoxel cell and
                // transform to vdb index space.
                Vec3f bMin = mAABB.lower + Vec3f(x,y,z) * mUnitWidth;
                Vec3f bMax = bMin + mUnitWidth;
                openvdb::CoordBBox subBBox = gridXform.worldToIndexCellCentered(openvdb::BBoxd(
                    openvdb::Vec3d(bMin.x, bMin.y, bMin.z),
                    openvdb::Vec3d(bMax.x, bMax.y, bMax.z)));
                // The above tests that the transformed bounding box encloses the cell
                // CENTERS in index space.  (I checked the OpenVDB source code.)
                // This can miss cells that have their centers outside the bbox but
                // still overlap.  We want to know if any part of any cell in vdb index
                // space overlaps this mHasActiveVoxel cell.  An easy way to compensate for this is
                // to pad the bbox by one cell in index space to guarantee we enclose
                // partially overlapped cells.
                subBBox.expand(1);

                mHasActiveVoxel[offset] = false;
                if (!bbox.hasOverlap(subBBox)) {
                    continue;
                }
                int threadIdx = mcrt_common::getFrameUpdateTLS()->mThreadIdx;
                for (openvdb::CoordBBox::Iterator<true> ijk(subBBox); ijk; ++ijk) {
                    if (topologyAccessors[threadIdx].isValueOn(*ijk)) {
                        mHasActiveVoxel[offset] = true;
                        break;
                    }
                }
            }
        }); // end of parallel_for

    }

    bool intersect(const Primitive* primitive, const Vec3f& rayOrg, const Vec3f& rayDir, float tNear,
            int volumeId, VolumeRayState& volumeRayState, float time) const
    {
        Vec3f org = transformPoint( mRenderToPrim, rayOrg, time, mIsMotionBlurOn);
        Vec3f dir = transformVector(mRenderToPrim, rayDir, time, mIsMotionBlurOn);

        float tStart;
        float tEnd = volumeRayState.getTEnd();
        // figure out the starting point of DDA traversal
        Vec3f pStart = org + tNear * dir;
        if (pStart[0] >= mAABB.lower[0] && pStart[0] <= mAABB.upper[0] &&
            pStart[1] >= mAABB.lower[1] && pStart[1] <= mAABB.upper[1] &&
            pStart[2] >= mAABB.lower[2] && pStart[2] <= mAABB.upper[2]) {
            // the ray starts inside bounding box
            tStart = tNear;
        } else {
            // bounding box intersection test
            float t0 = tNear;
            float t1 = tEnd;
            for (int axis = 0; axis < 3; ++axis) {
                // this assumes that the architecture being used suports IEEE
                // float point arithmetic:
                // for all v > 0, v / 0 =  INF and
                // for all v < 0, v / 0 = -INF
                // where INF is special value such that any positive number
                // multiplied by INF gives INF, any negative number multiplied
                // by INF gives -INF, and so on
                float invDir = 1.0f / dir[axis];
                float tNear = (mAABB.lower[axis] - org[axis]) * invDir;
                float tFar  = (mAABB.upper[axis] - org[axis]) * invDir;
                if (tNear > tFar) {
                    std::swap(tNear, tFar);
                }
                t0 = tNear > t0 ? tNear : t0;
                t1 = tFar < t1 ? tFar : t1;
                // the ray doesn't hit bounding box
                if (t0 > t1) {
                    return false;
                }
            }
            tStart = t0;
            pStart = org + tStart * dir;
        }
        // For detail reference see
        // "A Fast Voxel Traversal Algorithm for Ray Tracing"
        // John Amanatides and Andrew Woo
        float nextT[3];
        float deltaT[3];
        int step[3];
        int out[3];
        int pos[3];
        for (int axis = 0; axis < 3; ++axis) {
            pos[axis] = gridIndex(pStart, axis);
            if (dir[axis] >= 0) {
                nextT[axis] = tStart +
                    (gridPosition(pos[axis] + 1, axis) - pStart[axis]) /
                    dir[axis];
                deltaT[axis] = mUnitWidth[axis] / dir[axis];
                step[axis] = 1;
                out[axis] = mRes[axis];
            } else {
                nextT[axis] = tStart +
                    (gridPosition(pos[axis], axis) - pStart[axis]) /
                    dir[axis];
                deltaT[axis] = -mUnitWidth[axis] / dir[axis];
                step[axis] = -1;
                out[axis] = -1;
            }
        }
        // before we start the DDA, the ray is outside of this volume
        bool prevState = false;
        float tCurrent = tStart;
        int intervalCount = 0;
        // this should be sufficient to hold the worst traversal scenario
        // (ray passes the bounding box in diagonal and every neighbor voxel
        // in grid has different state)
        float intervals[2 * sMaxResolution];
        bool transitions[2 * sMaxResolution];
        while (true) {
            bool currentState = mHasActiveVoxel[offset(pos[0], pos[1], pos[2])];
            if (currentState != prevState) {
                intervals[intervalCount] = tCurrent;
                transitions[intervalCount] = currentState;
                intervalCount++;
            }
            prevState = currentState;
            // figure out which axis we are stepping forward
            // use bit shifting tricks to avoid branching and lookup
            // the idea is finding the axis with smallest nextT
            // careful that this may not work if we are goint to port
            // this to ISPC because of the 0x00000001 vs 0xFFFFFFFF
            // truth value thing in ISPC land
            int stepAxis =
                ((nextT[1] < nextT[0]) | (nextT[2] < nextT[0])) <<
                (nextT[2] < nextT[1]);
            tCurrent = nextT[stepAxis];
            if (tEnd < tCurrent) {
                intervals[intervalCount] = tEnd;
                transitions[intervalCount] = false;
                intervalCount++;
                break;
            }
            pos[stepAxis] += step[stepAxis];
            // Ray is exiting the bounding box. Note that we should only add this intersection point to the list
            // if the final voxel is occupied (indicated by currentState), since otherwise we're just going from
            // empty space to empty space. This fixes a bug which was exposed by MOONRAY-4292.
            if (pos[stepAxis] == out[stepAxis]) {
                if (currentState) {
                    intervals[intervalCount] = tCurrent;
                    transitions[intervalCount] = false;
                    intervalCount++;
                }
                break;
            }
            nextT[stepAxis] += deltaT[stepAxis];
        }
        // a single exit event means we doesn't hit any active voxel
        // a single enter event should not happen in theory but may be
        // introduced by float point precision issue
        // both cases should be considered as the ray didn't hit the volume
        if (intervalCount < 2) {
            return false;
        }

        int counter = 0;
        bool hitValidInterval = false;
        while (counter < intervalCount - 1) {
            // the extremely small enter/exit interval can
            // generate subtle sorting error in later interval compile stage
            if (scene_rdl2::math::isEqual(intervals[counter], intervals[counter + 1]) &&
                transitions[counter] != transitions[counter + 1]) {
                counter += 2;
            } else {
                volumeRayState.addInterval(primitive, intervals[counter], volumeId,
                    transitions[counter]);
                hitValidInterval = true;
                counter++;
            }
        }
        if (counter == (intervalCount - 1)) {
            volumeRayState.addInterval(primitive, intervals[counter], volumeId,
                transitions[counter]);
            hitValidInterval = true;
        }
        return hitValidInterval;
    }

    const BBox3f& getAABB() const
    {
        return mAABB;
    }

    size_t getMemory() const
    {
        return sizeof(DDAIntersector) + mMemory;
    }

private:
    int gridIndex(const Vec3f& p, int axis) const
    {
        int index = static_cast<int>(scene_rdl2::math::floor(
            (p[axis] - mAABB.lower[axis]) * mInvUnitWIdth[axis]));
        return scene_rdl2::math::clamp(index, 0, mRes[axis] - 1);
    }

    float gridPosition(int index, int axis) const
    {
        return mAABB.lower[axis] + index * mUnitWidth[axis];
    }

    finline int offset(int x, int y, int z) const
    {
        return (z * mRes[1]  + y) * mRes[0] + x;
    }

    BBox3f mAABB;
    int mRes[3];
    Vec3f mUnitWidth;
    Vec3f mInvUnitWIdth;
    std::unique_ptr<bool []> mHasActiveVoxel;
    static constexpr int sMaxResolution = 64;
    scene_rdl2::math::Mat4f mPrimToRender[2];
    scene_rdl2::math::Mat4f mRenderToPrim[2];
    bool mIsMotionBlurOn;

    // Allocated memory for the mHasActiveVoxel grid
    size_t mMemory;
};

constexpr int DDAIntersector::sMaxResolution;

VdbVolume::~VdbVolume() = default;

VdbVolume::VdbVolume(const std::string& vdbFilePath,
        const std::string& densityGridName,
        const std::string& emissionGridName,
        const std::string& velocityGridName,
        const MotionBlurParams& motionBlurParams,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
    NamedPrimitive(std::move(layerAssignmentId)),
    mHasUniformVoxels(false), mHasEmissionField(false),
    mInterpolationMode(Interpolation::BOX),
    mIsEmpty(true),
    mIsMotionBlurOn(motionBlurParams.isMotionBlurOn())
{
    MNRY_ASSERT_REQUIRE(mLayerAssignmentId.getType() ==
        LayerAssignmentId::Type::CONSTANT);
    mVdbVolumeData.reset(new VdbVolumeData(
        vdbFilePath,
        densityGridName,
        emissionGridName,
        velocityGridName,
        motionBlurParams,
        std::move(primitiveAttributeTable)));
}

size_t
VdbVolume::getMemory() const
{
    size_t mem = sizeof(VdbVolume) - sizeof(NamedPrimitive) +
        NamedPrimitive::getMemory();
    mem += scene_rdl2::util::getVectorElementsMemory(mTopologyAccessors) +
           scene_rdl2::util::getVectorElementsMemory(mTopologyIntersectors);


    // The first topology intersector stores a grid mask.
    // All other topology intersectors reference this grid mask.
    if (!mTopologyIntersectors.empty()) {
        mem += mTopologyIntersectors[0].tree().memUsage();
    }

    if (mLinearTransform[0]) {
        mem += sizeof(LinearGridTransform);
    }
    if (mLinearTransform[1]) {
        mem += sizeof(LinearGridTransform);
    }

    // the DDAIntersector stores a grid mask
    if (mDDAIntersector) {
        mem += mDDAIntersector->getMemory();
    }

    if (mVdbVolumeData) {
        mem += sizeof(VdbVolumeData);
    }

    if (mTopologyGrid) {
        mem += mTopologyGrid->memUsage();
    }

    // DensitySampler shares a grid with mTopologyGrid
    // so we don't count it here.
    mem += mDensitySampler.getMemory();

    mem += mEmissionSampler.getMemory();

    return mem;
}

void
VdbVolume::transformPrimitive(const scene_rdl2::math::Mat4f& primToRender)
{
    // primToRender will be identity in case this is a shared primitive
    mVdbVolumeData->mXform = primToRender;
}

void
VdbVolume::tessellate(const TessellationParams& tessellationParams)
{
    // If we are a shared primitive, our primToRender is currently identity
    // We'll need to store off the worldToRender matrix
    if (getIsReference()) {
        mVdbVolumeData->mWorldToRender = scene_rdl2::math::toFloat(tessellationParams.mWorld2Render);
    }

    bool isInitialized = initialize(*getRdlGeometry(),
                                    tessellationParams.mRdlLayer,
                                    tessellationParams.mVolumeAssignmentTable);
    mIsEmpty = !isInitialized;
    if (!isInitialized) {
        return;
    }

    // Construct a bounding box (non axis aligned in render space) which
    // encloses all the active voxels in this vdb grid.
    // We use this bounding box as the proxy in BVH ray intersection test,
    // and forward the actual volume interval collecting work to either
    // vdb's VolumeRayIntersector (for uniform voxels case) or
    // our DDAIntersector (for non-uniform case).

    if (mTopologyGrid) {
        // Set up local AABB, and the transformation that takes it to render space (with 2 time steps)
        Vec3f pMin, pMax;
        const scene_rdl2::math::Mat4f* xform[2];
        if (hasUniformVoxels()) {
            // Uniform case
            openvdb::math::CoordBBox bbox = mTopologyGrid->evalActiveVoxelBoundingBox();
            pMin = Vec3f(bbox.min().x(), bbox.min().y(), bbox.min().z());
            pMax = Vec3f(bbox.max().x(), bbox.max().y(), bbox.max().z());

            // In the uniform voxel case, we expect to have linear transforms
            MNRY_ASSERT(mLinearTransform[0] && mLinearTransform[1]);

            xform[0] = &mLinearTransform[0]->mIndexToRender;
            xform[1] = &mLinearTransform[1]->mIndexToRender;
        } else {
            // Non-uniform case
            mDDAIntersector.reset(new DDAIntersector(mPrimToRender, mRenderToPrim, mTopologyGrid,
                                                     mTopologyAccessors, mIsMotionBlurOn));
            const BBox3f& aabb = mDDAIntersector->getAABB();
            pMin = aabb.lower;
            pMax = aabb.upper;
            xform[0] = &mPrimToRender[0];
            xform[1] = &mPrimToRender[1];
        }

        // Transform bounding box vertices to render space
        for (int i = 0; i < 2; i++) {
            mBBoxVertices[i + 0 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMin.x, pMin.y, pMin.z)), 0.f);
            mBBoxVertices[i + 1 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMin.x, pMin.y, pMax.z)), 0.f);
            mBBoxVertices[i + 2 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMin.x, pMax.y, pMin.z)), 0.f);
            mBBoxVertices[i + 3 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMin.x, pMax.y, pMax.z)), 0.f);
            mBBoxVertices[i + 4 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMax.x, pMin.y, pMin.z)), 0.f);
            mBBoxVertices[i + 5 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMax.x, pMin.y, pMax.z)), 0.f);
            mBBoxVertices[i + 6 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMax.x, pMax.y, pMin.z)), 0.f);
            mBBoxVertices[i + 7 * 2] = Vec3fa(scene_rdl2::math::transformPoint(*xform[i], Vec3f(pMax.x, pMax.y, pMax.z)), 0.f);
        }
    } else {
        // If we don't have a grid for intersection, create an empty bbox
        for (int i = 0; i < 8 * 2; i++ ) {
            mBBoxVertices[i] = Vec3fa(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Indices for bounding box mesh(es). If 2 time samples, both use same set of indices.
    // front
    mBBoxIndices[4 * 0 + 0] = 3;
    mBBoxIndices[4 * 0 + 1] = 1;
    mBBoxIndices[4 * 0 + 2] = 5;
    mBBoxIndices[4 * 0 + 3] = 7;
    // right
    mBBoxIndices[4 * 1 + 0] = 7;
    mBBoxIndices[4 * 1 + 1] = 5;
    mBBoxIndices[4 * 1 + 2] = 4;
    mBBoxIndices[4 * 1 + 3] = 6;
    // back
    mBBoxIndices[4 * 2 + 0] = 6;
    mBBoxIndices[4 * 2 + 1] = 4;
    mBBoxIndices[4 * 2 + 2] = 0;
    mBBoxIndices[4 * 2 + 3] = 2;
    // left
    mBBoxIndices[4 * 3 + 0] = 2;
    mBBoxIndices[4 * 3 + 1] = 0;
    mBBoxIndices[4 * 3 + 2] = 1;
    mBBoxIndices[4 * 3 + 3] = 3;
    // up
    mBBoxIndices[4 * 4 + 0] = 2;
    mBBoxIndices[4 * 4 + 1] = 3;
    mBBoxIndices[4 * 4 + 2] = 7;
    mBBoxIndices[4 * 4 + 3] = 6;
    // down
    mBBoxIndices[4 * 5 + 0] = 1;
    mBBoxIndices[4 * 5 + 1] = 0;
    mBBoxIndices[4 * 5 + 2] = 4;
    mBBoxIndices[4 * 5 + 3] = 5;
}

int
VdbVolume::getIntersectionAssignmentId(int /*primID*/) const
{
    MNRY_ASSERT(mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT,
        "Volume assignments must be constant");
    int assignmentId = mLayerAssignmentId.getConstId();
    MNRY_ASSERT(assignmentId != -1, "unassigned part");
    return assignmentId;
}

void
VdbVolume::getTessellatedMesh(BufferDesc * vertexBufferDesc,
        BufferDesc& indexBufferDesc,
        size_t& vertexCount, size_t& faceCount, size_t& timeSteps) const
{
    timeSteps = mIsMotionBlurOn ? 2 : 1;
    faceCount = 6;
    indexBufferDesc.mData = mBBoxIndices;
    indexBufferDesc.mOffset = 0;
    indexBufferDesc.mStride = 4 * sizeof(int);
    vertexCount = 8;
    for (size_t i = 0; i < timeSteps; i++) {
        vertexBufferDesc[i].mData = mBBoxVertices + i;
        vertexBufferDesc[i].mOffset = 0;
        vertexBufferDesc[i].mStride = 2 * sizeof(Vec3fa);
    }
}

void
VdbVolume::postIntersect(mcrt_common::ThreadLocalState& tls,
        const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
        Intersection& intersection) const
{
    int assignmentId = mLayerAssignmentId.getConstId();
    intersection.setLayerAssignments(assignmentId, pRdlLayer);

    const scene_rdl2::rdl2::Material* material = intersection.getMaterial();
    const AttributeTable *table =
        material->get<shading::RootShader>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    intersection.setIds(ray.primID, 0, 0);
    overrideInstanceAttrs(ray, intersection);

    Vec3f Ng = normalize(ray.Ng);
    intersection.setDifferentialGeometry(Ng, Ng, scene_rdl2::math::one,
        scene_rdl2::math::zero, scene_rdl2::math::zero, false);

    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    intersection.setEpsilonHint( geometry->getRayEpsilon() );

    // wireframe AOV is blank
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 0);
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
            static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_POLYGON));

    }
}

BBox3f
VdbVolume::computeAABB() const
{
    BBox3f result(mBBoxVertices[0]);
    for (int i = 1; i < 2 * 8; i++) {
        result.extend(mBBoxVertices[i]);
    }
    return result;
}

BBox3f
VdbVolume::computeAABBAtTimeStep(int timeStep) const
{
    MNRY_ASSERT(timeStep >= 0 && timeStep < static_cast<int>(getMotionSamplesCount()), "timeStep out of range");
    BBox3f result(mBBoxVertices[timeStep]);
    for (int i = 1; i < 8; i++) {
        result.extend(mBBoxVertices[timeStep + 2 * i]);
    }
    return result;
}

std::unique_ptr<EmissionDistribution>
VdbVolume::computeEmissionDistribution(const scene_rdl2::rdl2::VolumeShader* volumeShader) const
{
    if (!mHasEmissionField) {
        // There is no volume data but there is a emission from a volume shader.
        return Primitive::computeEmissionDistribution(volumeShader);
    }

    MNRY_ASSERT_REQUIRE(mEmissionGrid);
    MNRY_ASSERT_REQUIRE(mEmissionGrid->transform().isLinear());

    std::vector<float> values = getGridLuminance(mEmissionSampler,
        *static_cast<const openvdb::VectorGrid*>(mEmissionGrid.get()));

    return computeEmissionDistributionImpl(getRdlGeometry(),
                                           *static_cast<const openvdb::VectorGrid*>(mEmissionGrid.get()),
                                           mRenderToPrim,
                                           values,
                                           volumeShader);
}

const scene_rdl2::rdl2::Material *
VdbVolume::getIntersectionMaterial(const scene_rdl2::rdl2::Layer *pRdlLayer,
        const mcrt_common::Ray &ray) const
{
    int layerAssignmentId = getIntersectionAssignmentId(ray.primID);
    const scene_rdl2::rdl2::Material *pMaterial = MNRY_VERIFY(pRdlLayer->lookupMaterial(layerAssignmentId));
    return pMaterial;
}

scene_rdl2::math::Vec3f
VdbVolume::evalVolumeSamplePosition(mcrt_common::ThreadLocalState* tls,
                                    uint32_t volumeId,
                                    const Vec3f& pSample,
                                    float time) const
{
    const openvdb::Vec3d p = mVdbVelocity->getEvalPosition(tls, volumeId, pSample, time);
    return scene_rdl2::math::Vec3f(p.x(), p.y(), p.z());
}

 scene_rdl2::math::Vec3f
 VdbVolume::transformVolumeSamplePosition(const Vec3f& pSample, float time) const
{
    return transformPoint(mPrimToRender, pSample, time, mIsMotionBlurOn);
}

scene_rdl2::math::Color
VdbVolume::sampleBakedDensity(mcrt_common::ThreadLocalState* tls,
                              uint32_t volumeId,
                              const openvdb::Vec3d& p) const
{
    if (mBakedDensitySampler.mIsValid) {
        const openvdb::Vec3f density = mBakedDensitySampler.eval(tls,
                                                                 volumeId,
                                                                 p,
                                                                 geom::internal::Interpolation::POINT);
        return scene_rdl2::math::Color(density.x(), density.y(), density.z());
    } else {
        return mDensityColor;
    }
}

Color
VdbVolume::evalDensity(mcrt_common::ThreadLocalState* tls,
                       uint32_t volumeId,
                       const Vec3f& pSample, float /*rayVolumeDepth*/,
                       const scene_rdl2::rdl2::VolumeShader* const /*volumeShader*/) const
{
    const openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
    const Color density = Color(mDensitySampler.eval(tls, volumeId, p, Interpolation::POINT));
    return density * sampleBakedDensity(tls, volumeId, p);
}

void
VdbVolume::evalVolumeCoefficients(mcrt_common::ThreadLocalState* tls,
                                  uint32_t volumeId,
                                  const Vec3f& pSample,
                                  Color* extinction,
                                  Color* albedo,
                                  Color* temperature,
                                  bool highQuality,
                                  float /*rayVolumeDepth*/,
                                  const scene_rdl2::rdl2::VolumeShader* const /*volumeShader*/) const
{
    const openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
    Interpolation mode = highQuality ? mInterpolationMode : Interpolation::POINT;
    *extinction = Color(mDensitySampler.eval(tls, volumeId, p, mode));
    *albedo = Color(1.0f);
    if (temperature) {
        const auto colorVector = mEmissionSampler.eval(tls, volumeId, p, Interpolation::POINT);
        *temperature = Color(colorVector.x(), colorVector.y(), colorVector.z());
    }

    *extinction *= sampleBakedDensity(tls, volumeId, p);
}

Color
VdbVolume::evalTemperature(mcrt_common::ThreadLocalState* tls,
                           uint32_t volumeId,
                           const Vec3f& pSample) const
{
    const openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
    const auto colorVector = mEmissionSampler.eval(tls, volumeId, p, Interpolation::POINT);
    return Color(colorVector.x(), colorVector.y(), colorVector.z());
}

void
VdbVolume::initVolumeSampleInfo(VolumeSampleInfo* info,
        const Vec3f& rayOrg, const Vec3f& rayDir, const float time,
        const scene_rdl2::rdl2::VolumeShader* volumeShader,
        int volumeId) const
{
    // if we have an instance feature id for this volume id, use it
    const float featureSize = getInstanceFeatureSize(volumeId);

    // TODO: the amorphous sampler evaluates the voxel value in
    // the VDB's world space, while our shading convention currently
    // initializes the shading state in render space. Moreover, the amorphous
    // core library actually does one more space transformation to evaluate the voxel
    // in index space. We should probably modify the amorphous library to have
    // index space evaluation and redesign the VolumeShader interface
    // for API consistency and performance in the long run.

    // A note about instancing:  If the VdbVolume is a shared primitive
    // then it is assumed that rayOrg and rayDir are already in the
    // primitive space, as they should have been transformed there by the
    // instance intersection callbacks.  In that case, mRenderToPrim should be
    // the identity and we can avoid the xform.
    if (getIsReference()) {
        info->initialize(volumeShader,
            rayOrg, rayDir, featureSize,
            (mRdlGeometry->getVisibilityMask() & scene_rdl2::rdl2::SHADOW) != 0,
            /* isVDB = */ true);
    } else {
        info->initialize(volumeShader,
            transformPoint( mRenderToPrim, rayOrg, time, mIsMotionBlurOn),
            transformVector(mRenderToPrim, rayDir, time, mIsMotionBlurOn), featureSize,
            (mRdlGeometry->getVisibilityMask() & scene_rdl2::rdl2::SHADOW) != 0,
            /* isVDB = */ true);
    }
}

bool
VdbVolume::queryIntersections(const Vec3f& rayOrg, const Vec3f& rayDir,
                              float tNear, float time, int threadIdx, int volumeId,
                              VolumeRayState& volumeRayState, bool computeRenderSpaceDistance)
{

    // A note about instancing:  If the VdbVolume is a shared primitive
    // then it is assumed that rayOrg and rayDir are already in the
    // primitive space, as they should have been transformed there by the
    // instance intersection callbacks.  In that case, mRenderToPrim should be
    // the identity.  However we still need to take the prim to index transform
    // into account.

    if (hasUniformVoxels()) {
        // For the uniform case, delegate the intersection query duty to VDB's VolumeRayIntersector
        typedef openvdb::math::Ray<double> RayT;
        typedef RayT::Vec3Type Vec3T;

        const scene_rdl2::math::Mat4f& r2i0 = mLinearTransform[0]->mRenderToIndex;
        const scene_rdl2::math::Mat4f& r2i1 = mLinearTransform[1]->mRenderToIndex;

        const scene_rdl2::math::Mat4f& i2r0 = mLinearTransform[0]->mIndexToRender;
        const scene_rdl2::math::Mat4f& i2r1 = mLinearTransform[1]->mIndexToRender;

        Vec3f org, dir;
        if (mIsMotionBlurOn) {
            Vec3f org0 = scene_rdl2::math::transformPoint(r2i0, rayOrg);
            Vec3f org1 = scene_rdl2::math::transformPoint(r2i1, rayOrg);
            org = lerp(org0, org1, time);

            Vec3f dir0 = scene_rdl2::math::transformVector(r2i0, rayDir);
            Vec3f dir1 = scene_rdl2::math::transformVector(r2i1, rayDir);
            dir = lerp(dir0, dir1, time);
        } else {
            org = scene_rdl2::math::transformPoint (r2i0, rayOrg);
            dir = scene_rdl2::math::transformVector(r2i0, rayDir);
        }

        auto renderSpaceDistance = [&](Vec3f &iOrg, Vec3f &iDir, float ft) -> float {
            Vec3f iP = iOrg + iDir * ft;
            Vec3f rP;
            if (mIsMotionBlurOn) {
                Vec3f rP0 = scene_rdl2::math::transformPoint(i2r0, iP);
                Vec3f rP1 = scene_rdl2::math::transformPoint(i2r1, iP);
                rP = lerp(rP0, rP1, time);
            } else {
                rP = scene_rdl2::math::transformPoint(i2r0, iP);
            }
            return distance(rP, rayOrg);
        };

        RayT indexRay(Vec3T(org.x, org.y, org.z), Vec3T(dir.x, dir.y, dir.z));

        bool intersectVoxel = false;
        if (!mTopologyIntersectors[threadIdx].setIndexRay(indexRay)) {
            return intersectVoxel;
        }
        double t0, t1;
        float tEnd = volumeRayState.getTEnd();
        while (mTopologyIntersectors[threadIdx].march(t0, t1)) {
            float ft0 = static_cast<float>(t0);
            float ft1 = static_cast<float>(t1);
            // when the interval is extremely small, the casting from
            // double to float can result in a zero length interval, which can
            // generate subtle sorting errors later in the interval compile stage
            if (scene_rdl2::math::isEqual(ft0, ft1)) {
                continue;
            }
            intersectVoxel = true;
            if (ft0 >= tEnd) {
                break;
            }
            if (computeRenderSpaceDistance) {
                float d[2];
                d[0] = renderSpaceDistance(org, dir, ft0);
                d[1] = renderSpaceDistance(org, dir, ft1);
                volumeRayState.addInterval(this, ft0, volumeId, true, d);
            } else {
                volumeRayState.addInterval(this, ft0, volumeId, true);
            }
            if (ft1 >= tEnd) {
                break;
            }
            volumeRayState.addInterval(this, ft1, volumeId, false);
        }
        return intersectVoxel;
    } else {
        // Call our custom intersect() function for the non-uniform case
        return mDDAIntersector->intersect(this, rayOrg, rayDir, tNear, volumeId,
            volumeRayState, time);
    }
}

bool
VdbVolume::isInActiveField(uint32_t threadIdx, const Vec3f& p, float time) const
{
    // A note about instancing:  If the VdbVolume is a shared primitive
    // then it is assumed that p is already in the
    // primitive space.  In that case, mRenderToPrim should be
    // the identity.

    if (mIsMotionBlurOn) {
        if (mLinearTransform[0] && mLinearTransform[1]) {
            Vec3f pIndex0 = scene_rdl2::math::transformPoint(mLinearTransform[0]->mRenderToIndex, p);
            Vec3f pIndex1 = scene_rdl2::math::transformPoint(mLinearTransform[1]->mRenderToIndex, p);
            Vec3f pIndex  = lerp(pIndex0, pIndex1, time);
            return mTopologyAccessors[threadIdx].isValueOn(
                openvdb::Coord(pIndex.x, pIndex.y, pIndex.z));
        } else {
            Vec3f pLocal;
            if (getIsReference()) {
                // VDB instances don't support blurred matrices
                pLocal = p;
            } else {
                Vec3f pLocal0 = scene_rdl2::math::transformPoint(mRenderToPrim[0], p);
                Vec3f pLocal1 = scene_rdl2::math::transformPoint(mRenderToPrim[1], p);
                pLocal  = lerp(pLocal0, pLocal1, time);
            }
            return mTopologyAccessors[threadIdx].isValueOn(
                mTopologyGrid->transform().worldToIndexCellCentered(openvdb::Vec3d(
                pLocal.x, pLocal.y, pLocal.z)));
        }
    } else {
        if (mLinearTransform[0]) {
            Vec3f pIndex = scene_rdl2::math::transformPoint(mLinearTransform[0]->mRenderToIndex, p);
            return mTopologyAccessors[threadIdx].isValueOn(
                openvdb::Coord(pIndex.x, pIndex.y, pIndex.z));
        } else {
            Vec3f pLocal = scene_rdl2::math::transformPoint(mRenderToPrim[0], p);
            return mTopologyAccessors[threadIdx].isValueOn(
                mTopologyGrid->transform().worldToIndexCellCentered(openvdb::Vec3d(
                pLocal.x, pLocal.y, pLocal.z)));
        }
    }
}

void
VdbVolume::initMotionBlurBoundary(openvdb::FloatGrid::Ptr& topologyGrid,
        openvdb::VectorGrid::Ptr velocityGrid,
        float tShutterOpen, float tShutterClose)
{
    // temporary per thread local storage for padding operation
    struct PaddOpTls {
        PaddOpTls() : mIsInitialized(false) {}

        void initialize(openvdb::FloatGrid::Ptr& topologyGrid,
            openvdb::VectorGrid::Ptr velocityGrid)
        {
            mVelocityAccessor.reset(new openvdb::VectorGrid::ConstAccessor(
                velocityGrid->getConstAccessor()));
            mPaddedGrid = openvdb::FloatGrid::create();
            mPaddedGrid->setTransform(topologyGrid->transformPtr());
            mPaddedGridAccessor.reset(new openvdb::FloatGrid::Accessor(
                mPaddedGrid->getAccessor()));
            mPaddedVelocityGrid = openvdb::VectorGrid::create();
            mPaddedVelocityGrid->setTransform(velocityGrid->transformPtr());
            mIsInitialized = true;
        }

        std::unique_ptr<openvdb::VectorGrid::ConstAccessor> mVelocityAccessor;
        openvdb::FloatGrid::Ptr mPaddedGrid;
        std::unique_ptr<openvdb::FloatGrid::Accessor> mPaddedGridAccessor;
        openvdb::VectorGrid::Ptr mPaddedVelocityGrid;
        bool mIsInitialized;
    };

    tbb::enumerable_thread_specific<PaddOpTls> mblurTls;
    typedef typename openvdb::tree::IteratorRange<
        openvdb::FloatGrid::TreeType::LeafCIter> IterRange;
    IterRange range(topologyGrid->tree().cbeginLeaf());
    float voxelLength = topologyGrid->voxelSize().z();
    tbb::parallel_for(range, [&](IterRange& range) {
        tbb::enumerable_thread_specific<PaddOpTls>::reference localTmp =
            mblurTls.local();
        if (!localTmp.mIsInitialized) {
            localTmp.initialize(topologyGrid, velocityGrid);
        }
        const openvdb::VectorGrid::ConstAccessor& vAccessor =
            *(localTmp.mVelocityAccessor);
        openvdb::FloatGrid::Ptr& paddedGrid = localTmp.mPaddedGrid;
        openvdb::FloatGrid::Accessor& paddedGridAccessor =
            *(localTmp.mPaddedGridAccessor);
        openvdb::VectorGrid::Ptr paddedVGrid = localTmp.mPaddedVelocityGrid;
        for (; range; ++range) {
            const auto& leafIter = range.iterator();
            for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
                openvdb::Vec3s velocity;
                const openvdb::math::Coord& coord = iter.getCoord();
                const openvdb::Vec3d p =
                    topologyGrid->indexToWorld(coord);
                openvdb::math::Coord vcoord =
                    openvdb::math::Coord::round(
                    velocityGrid->worldToIndex(p));
                if (!vAccessor.probeValue(vcoord, velocity)) {
                    continue;
                }
                float dt = voxelLength / velocity.length();

                // The advection code in VDBVelocity.h looks both forwards and backwards
                // in time by up to the maximum value of shutter open and close.
                // Thus we must ensure the velocity is defined across the entire time range.
                // This is also what the "Kulla, Farjardo 12 approach" does in the comment below.
                float maxT = scene_rdl2::math::max(scene_rdl2::math::abs(tShutterOpen), scene_rdl2::math::abs(tShutterClose));
                for (float t = -maxT; t <= maxT; t += dt) {
                    const openvdb::Vec3d pPrime = p + velocity * t;
                    // activate voxel velocity passes through
                    paddedGridAccessor.setValueOn(
                        openvdb::math::Coord::round(
                        paddedGrid->worldToIndex(pPrime)));

                    vcoord = openvdb::math::Coord::round(
                        velocityGrid->worldToIndex(pPrime));
                    if (!vAccessor.isValueOn(vcoord)) {
                        openvdb::tools::setValueOnMax(
                            paddedVGrid->tree(), vcoord, velocity);
                    }
                }
            }
        }
    });
    // merge back temporary padding grids to input grids
    for (auto it = mblurTls.begin(); it != mblurTls.end(); ++it) {
        topologyGrid->topologyUnion(*(it->mPaddedGrid));
        openvdb::tools::compMax(*velocityGrid, *(it->mPaddedVelocityGrid));
    }
    // dilate the padded grid by one voxel in case our padding scheme above
    // "scratches" some voxels by cornors
    openvdb::tools::dilateActiveValues(topologyGrid->tree(), 1,
        openvdb::tools::NN_FACE, openvdb::tools::EXPAND_TILES);

    // This is the Kulla, Farjardo 12 approach
    // "we expand the bounding box of non-zero blocks by the length of
    // the longest velocity vector"
//    float maxVelocity = openvdb::tools::extrema(
//        velocityGrid->cbeginValueOn(),
//        /*threading*/true).max();
//    float maxT = scene_rdl2::math::max(scene_rdl2::math::abs(tShutterOpen), scene_rdl2::math::abs(tShutterClose));
//    int nAdvect = scene_rdl2::math::ceil(maxVelocity * maxT / voxelLength);
//    openvdb::tools::dilateActiveValues(topologyGrid->tree(), nAdvect,
//        openvdb::tools::NN_FACE, openvdb::tools::EXPAND_TILES);

//    // TODO remove this debugging write
//    openvdb::io::File file("debug.vdb");
//    openvdb::GridPtrVec grids;
//    grids.push_back(topologyGrid);
//    file.write(grids);
//    file.close();
}

// Returns nth grid that has name "name"
const openvdb::GridBase::Ptr
findNthGridByName(const openvdb::GridPtrVec& grids, const std::string& name)
{
    // get grid index from name
    int gridIndex = 0;
    const std::string uniqueName = openvdb::io::GridDescriptor::stringAsUniqueName(name);
    const std::string strippedName = openvdb::io::GridDescriptor::stripSuffix(uniqueName);
    if (uniqueName != strippedName) {
        // ASCII "record separator" character
        static constexpr char delimiter = 30;
        // This name has a suffix. Get it.
        gridIndex = std::stoi(uniqueName.substr(uniqueName.find(delimiter) + 1, uniqueName.size()));
    }

    // get nth grid with name
    int index = 0;
    for (auto it = grids.begin(); it != grids.end(); ++it) {
        const openvdb::GridBase::Ptr grid = *it;
        if (grid != nullptr && grid->getName() == strippedName) {
            if (gridIndex == index++) {
                return grid;
            }
        }
    }
    return openvdb::GridBase::Ptr();
}

bool
VdbVolume::initializePhase1(const std::string& vdbFilePath,
                            openvdb::io::File& file,
                            const scene_rdl2::rdl2::Geometry& rdlGeometry,
                            const scene_rdl2::rdl2::Layer* layer,
                            const VolumeAssignmentTable* volumeAssignmentTable,
                            std::vector<int>& volumeIds,
                            openvdb::GridPtrVecPtr& grids,
                            openvdb::VectorGrid::Ptr& velocityGrid)
{
    // openvdb::io::File::open() can throw an excpetion. That excpetion is caught by
    // moonray::rt::GeometryManager::tessellate.
    file.open();
    // Read in just the metadata for all grids.
    // We verify metadata before loading the voxel data.
    grids = file.readAllGridMetadata();
    if ((!grids) || (grids->size() == 0)) {
        rdlGeometry.warn("VDB file \"", vdbFilePath,
            "\" contains no grids.");
        return false;
    }

    // load in velocity grid if motion blur is enabled
    float velocityScale = mVdbVolumeData->mVelocityScale;
    if (mVdbVolumeData->mIsMotionBlurOn &&
        !isZero(velocityScale)) {
        const auto sceneContext =
            rdlGeometry.getSceneClass().getSceneContext();
        float fps = sceneContext->getSceneVariables().get(
            scene_rdl2::rdl2::SceneVariables::sFpsKey);
        float dt = 1.0f / fps;
        // information to remap ray's time sample (0-1)
        // to the actual time value we use to perturb the ray
        float tShutterOpen = dt * mVdbVolumeData->mShutterOpen;
        float tShutterRange = dt * mVdbVolumeData->mShutterClose -
            tShutterOpen;
        mVdbVelocity->setShutterValues(tShutterOpen, tShutterRange);
        openvdb::GridBase::Ptr velocityGridTmp = findNthGridByName(
            *grids, mVdbVolumeData->mVelocityGridName);
        // validate grid from metadata
        if (velocityGridTmp && velocityGridTmp->isType<openvdb::VectorGrid>()) {
            // read voxel data
            velocityGridTmp = file.readGrid(mVdbVolumeData->mVelocityGridName);
            if (!velocityGridTmp->empty()) {
                velocityGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(
                    velocityGridTmp);

                // scale the velocity
                openvdb::tools::foreach(velocityGrid->beginValueOn(),
                    [&](const openvdb::VectorGrid::ValueOnIter& iter) {
                    iter.setValue(*iter * velocityScale);
                });
            }
        }
    }

    // Create the grid samplers. There is a different sampler for each volume id.
    // If this primitive is a reference geometry, it will have multiple volume ids.
    std::unordered_set<int> assignmentIds;
    getUniqueAssignmentIds(assignmentIds);
    auto it = assignmentIds.begin();
    MNRY_ASSERT(assignmentIds.size() == 1);
    while (it != assignmentIds.end() && layer->lookupVolumeShader(*it) == nullptr) {
        it++;
    }
    volumeIds = volumeAssignmentTable->getVolumeIds(*it);

    return true;
}

void
VdbVolume::initializePhase2(const scene_rdl2::rdl2::Geometry& rdlGeometry,
                            const scene_rdl2::rdl2::Layer* layer,
                            const VolumeAssignmentTable* volumeAssignmentTable,
                            const std::vector<int>& volumeIds,
                            openvdb::VectorGrid::Ptr& velocityGrid)
{
    mHasUniformVoxels = mTopologyGrid->hasUniformVoxels();
    if (mTopologyGrid->transform().isLinear()) {
        openvdb::math::MapBase::ConstPtr map =
            mTopologyGrid->transform().baseMap();
        openvdb::math::Mat4d M = map->getAffineMap()->getMat4().asPointer();
        // Loop over 2 time samples
        for (int i = 0; i < 2; i++) {
            mLinearTransform[i].reset(new LinearGridTransform(Mat4f(
                M[0][0], M[0][1], M[0][2], M[0][3],
                M[1][0], M[1][1], M[1][2], M[1][3],
                M[2][0], M[2][1], M[2][2], M[2][3],
                M[3][0], M[3][1], M[3][2], M[3][3])));
            mLinearTransform[i]->appendXform(mPrimToRender[i]);
        }
        Mat4f indexToRenderMid = lerp(mLinearTransform[0]->mIndexToRender,
                                            mLinearTransform[1]->mIndexToRender, 0.5f);
        mFeatureSize = cbrt(indexToRenderMid.det());
        // vdb ray intersector does not support grids with non-uniform voxels
        if (mHasUniformVoxels) {
            size_t threadCount = mcrt_common::getMaxNumTLS();
            mTopologyIntersectors.reserve(threadCount);
            // construct VolumeRayIntersector for the first thread
            mTopologyIntersectors.emplace_back(*mTopologyGrid);
            for (size_t i = 1; i < threadCount; ++i) {
                // shallow copy the VolumeRayIntersector for subsequent threads
                mTopologyIntersectors.emplace_back(mTopologyIntersectors[0]);
            }
        }
    } else {
        // vdb grid uses non-linear transform
        mLinearTransform[0].reset();
        mLinearTransform[1].reset();
        // use the voxel size in z axis as marching step size, which will be
        // consistant across frustum VDB regardless of its distance to camera
        Mat4f primToRenderMid = lerp(mPrimToRender[0], mPrimToRender[1], 0.5f);
        mFeatureSize = mTopologyGrid->voxelSize().z() * cbrt(primToRenderMid.det());
    }

    // Set feature sizes for each instance of this volume.
    volumeAssignmentTable->setFeatureSizes(this, volumeIds, mFeatureSize);

    if (getIsReference()) {
        MNRY_ASSERT(isEqual(mPrimToRender[0], Mat4f(one)));
        MNRY_ASSERT(isEqual(mRenderToPrim[0], Mat4f(one)));
        if (mIsMotionBlurOn) {
            MNRY_ASSERT(isEqual(mPrimToRender[1], Mat4f(one)));
            MNRY_ASSERT(isEqual(mRenderToPrim[1], Mat4f(one)));
        }
        // We want mRenderToPrim to be the indentity, but we
        // want mPrimToRender to be the world2render xform
        // TODO: fix the dual usage of VdbVolume's mPrimToRender matrices. Even with the comment here, it's a bad state
        // of affairs because a developer working in another part of the code base will see mPrimToRender and assume
        // (logically) that it holds the prim-to-render matrices. This has already led to at least one hard-to-find bug.
        mPrimToRender[0] = mVdbVolumeData->mWorldToRender;
        mPrimToRender[1] = mVdbVolumeData->mWorldToRender;
    }

    if (velocityGrid) {
        float velocitySampleRate = clamp(
            mVdbVolumeData->mVelocitySampleRate);
        if (velocitySampleRate > 0.0f && velocitySampleRate < 1.0f) {
            // down sample velocity
            openvdb::VectorGrid::Ptr decimatedGrid(
                new openvdb::VectorGrid());
            decimatedGrid->setTransform(velocityGrid->transform().copy());
            decimatedGrid->transform().preScale(1.0f / velocitySampleRate);
            openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(
                *velocityGrid, *decimatedGrid);
            velocityGrid = decimatedGrid;
        }
        mVdbVelocity->setVelocityGrid(velocityGrid, volumeIds);
    }

    // set up per thread accessors for render time usage
    size_t threadCount = mcrt_common::getMaxNumTLS();
    mTopologyAccessors.reserve(threadCount);
    for (size_t i = 0; i < threadCount; ++i) {
        mTopologyAccessors.emplace_back(mTopologyGrid->getConstAccessor());
    }

    // we are done with grids/samplers initialization,
    // the temporary data is no longer needed
    mVdbVolumeData.reset();
}

bool
VdbVolume::initialize(const scene_rdl2::rdl2::Geometry& rdlGeometry,
                      const scene_rdl2::rdl2::Layer* layer,
                      const VolumeAssignmentTable* volumeAssignmentTable)
{
    const std::string& vdbFilePath = mVdbVolumeData->mFilePath;
    if (vdbFilePath.empty()) {
        return false;
    }

    openvdb::io::File file(vdbFilePath);
    openvdb::GridPtrVecPtr grids;
    openvdb::VectorGrid::Ptr velocityGrid;
    std::vector<int> volumeIds;

    if (!initializePhase1(vdbFilePath,
                          file,
                          rdlGeometry,
                          layer,
                          volumeAssignmentTable,
                          volumeIds,
                          grids,
                          velocityGrid)) {
        return false;
    }

    if (!initDensitySampler(file,
                            grids,
                            mVdbVolumeData->mDensityGridName,
                            rdlGeometry,
                            velocityGrid,
                            volumeIds)) {
        return false;
    }

    if (!mVdbVolumeData->mEmissionGridName.empty()) {
        if (!initEmissionSampler(file,
                                 grids,
                                 mVdbVolumeData->mEmissionGridName,
                                 rdlGeometry,
                                 velocityGrid,
                                 volumeIds)) {
            mHasEmissionField = false;
            return false;
        } else {
            mHasEmissionField = !mEmissionSampler.mGrid->empty();
        }
    }

    initializePhase2(rdlGeometry,
                     layer,
                     volumeAssignmentTable,
                     volumeIds,
                     velocityGrid);

    return true;
}


bool
VdbVolume::initDensitySampler(openvdb::io::File& file,
                              openvdb::GridPtrVecPtr grids,
                              const std::string& densityGridName,
                              const scene_rdl2::rdl2::Geometry& rdlGeometry,
                              openvdb::VectorGrid::Ptr velocityGrid,
                              const std::vector<int>& volumeIds)
{
    mTopologyGrid.reset();
    // read metadata of grid with name densityGridName.
    openvdb::GridBase::Ptr densityGrid = findNthGridByName(*grids, densityGridName);
    // check that it is a valid grid. If not, throw error.
    if (densityGrid) {
        if (!densityGrid->isType<openvdb::FloatGrid>()) {
            rdlGeometry.error("Density grid: \"", densityGridName, "\" is not a float grid.");
            return false;
        }
        // read voxel data
        densityGrid = file.readGrid(densityGridName);
        if (densityGrid->empty()) {
            rdlGeometry.error("Density grid: \"", densityGridName, "\" is empty.");
            return false;
        }

        mTopologyGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(densityGrid);
    } else {
        rdlGeometry.error("Density grid: \"", densityGridName, "\" does not exist.");
        return false;
    }

    // padding extra voxels for motion blur usage
    if (velocityGrid) {
        float shutterOpen, shutterClose;
        mVdbVelocity->getShutterOpenAndClose(shutterOpen, shutterClose);
        initMotionBlurBoundary(mTopologyGrid, velocityGrid,
            shutterOpen, shutterClose);
    }
    mDensitySampler.initialize(mTopologyGrid, volumeIds, STATS_DENSITY_GRID_SAMPLES);
    return true;
}

bool
VdbVolume::initEmissionSampler(openvdb::io::File& file,
                               openvdb::GridPtrVecPtr grids,
                               const std::string& emissionGridName,
                               const scene_rdl2::rdl2::Geometry& rdlGeometry,
                               openvdb::VectorGrid::Ptr velocityGrid,
                               const std::vector<int>& volumeIds)
{
    mEmissionGrid.reset();
    // read metadata of grid with name densityGridName.
    openvdb::GridBase::Ptr emissionGrid = findNthGridByName(*grids, emissionGridName);
    // check that it is a valid grid. If not, throw error.
    if (emissionGrid) {
        if (!emissionGrid->isType<openvdb::VectorGrid>()) {
            rdlGeometry.error("Emission grid: \"", emissionGridName, "\" is not an RGB grid.");
            return false;
        }
        // read voxel data
        emissionGrid = file.readGrid(emissionGridName);
        if (emissionGrid->empty()) {
            rdlGeometry.error("Emission grid: \"", emissionGridName, "\" is empty.");
            return false;
        }

        mEmissionGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(emissionGrid);
    } else {
        rdlGeometry.error("Emission grid: \"", emissionGridName, "\" does not exist.");
        return false;
    }

    mEmissionSampler.initialize(mEmissionGrid, volumeIds, STATS_EMISSION_GRID_SAMPLES);
    return true;
}

void
VdbVolume::bakeVolumeShaderDensityMap(const scene_rdl2::rdl2::VolumeShader* volumeShader,
                                      const scene_rdl2::math::Mat4f& primToRender,
                                      const MotionBlurParams& /* motionBlurParams */,
                                      const VolumeAssignmentTable* volumeAssignmentTable,
                                      const int assignmentId)
{
    if(!volumeShader->hasExtinctionMapBinding()) {
        // If no map binding, don't bake. Store uniform color instead.
        shading::Intersection isect;
        shading::TLState *tls = mcrt_common::getFrameUpdateTLS()->mShadingTls.get();
        mDensityColor = volumeShader->extinct(tls, shading::State(&isect), 
                                              scene_rdl2::math::Color(1.f), 
                                              /*rayVolumeDepth*/ -1);
        return;
    }

    // Get resolution of grid
    openvdb::Vec3d rez(1.f);
    openvdb::math::CoordBBox bbox = mTopologyGrid->evalActiveVoxelBoundingBox();
    openvdb::math::Transform::Ptr newTransform;
    newTransform.reset(new openvdb::math::Transform(mTopologyGrid->transform()));

    int bakeResolutionMode = volumeShader->getBakeResolutionMode();
    switch (bakeResolutionMode) {
    case 1:
    {
        // Divisions
        int divisions = volumeShader->getBakeDivisions();
        float maxDivisions = scene_rdl2::math::max(bbox.max().x() - bbox.min().x(), scene_rdl2::math::max(bbox.max().y() - bbox.min().y(),
            bbox.max().z() - bbox.min().z()));
        rez = openvdb::Vec3d(divisions / maxDivisions);
        break;
    }
    case 2:
    {
        // Voxel size
        float voxelSize = volumeShader->getBakeVoxelSize();
        openvdb::Vec3d currentVoxelSize = newTransform->voxelSize();
        rez = currentVoxelSize / voxelSize;
        break;
    }
    }

    if (rez.x() <= 0.0 || rez.y() <= 0.0 || rez.z() <= 0.0) {
        // bad user input. Don't bake grid.
        mDensityColor = scene_rdl2::math::Color(1.0);
        return;
    }

    // initialize grid
    mBakedDensityGrid = openvdb::Vec3SGrid::create();

    // need this scale if we are uprezing or downrezing grid
    newTransform->preScale(1.f / rez);
    mBakedDensityGrid->setTransform(newTransform);

    // fill
    bbox.reset(openvdb::math::Coord(bbox.min().x() * rez.x(), bbox.min().y() * rez.y(), bbox.min().z() * rez.z()),
               openvdb::math::Coord(bbox.max().x() * rez.x(), bbox.max().y() * rez.y(), bbox.max().z() * rez.z()));
    mBakedDensityGrid->denseFill(bbox, openvdb::Vec3f(1.f, 1.f, 1.f), true);

    // bake map shader
    openvdb::tools::foreach(mBakedDensityGrid->beginValueOn(),
        [&](const openvdb::Vec3SGrid::ValueOnIter& it) {
            // get xyz position of grid in primitive space == world space
            const openvdb::Vec3d pd = mBakedDensityGrid->indexToWorld(it.getCoord());
            const scene_rdl2::math::Vec3f p(pd.x(), pd.y(), pd.z());
            // sample volume shader
            shading::Intersection isect;
            isect.setP(scene_rdl2::math::transformPoint(primToRender, p));
            shading::TLState *tls = mcrt_common::getFrameUpdateTLS()->mShadingTls.get();
            const scene_rdl2::math::Color result = volumeShader->extinct(tls, shading::State(&isect), 
                                                                         scene_rdl2::math::Color(1.0f), 
                                                                         /*rayVolumeDepth*/ -1);
            // set result to new grid
            it.setValue(openvdb::Vec3f(result.r, result.g, result.b));
        }
    );

    mBakedDensityGrid->pruneGrid();

    // Get volume ids. There is a separate density sampler for each volume id.
    const std::vector<int>& volumeIds = volumeAssignmentTable->getVolumeIds(assignmentId);
    mBakedDensitySampler.initialize(mBakedDensityGrid,
                                    volumeIds,
                                    STATS_BAKED_DENSITY_GRID_SAMPLES);
}

} // namespace internal
} // namespace geom
} // namespace moonray


