// Copyright 2023-2024 DreamWorks Animation LLC and Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/geom/prim/Primitive.h>

#include "EmbreeAccelerator.h"

#include "AcceleratorUtils.h"
#include "IntersectionFilters.h"

#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>
#include <moonray/rendering/geom/prim/BVHHandle.h>
#include <moonray/rendering/geom/prim/CubicSpline.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/OpenSubdivMesh.h>
#include <moonray/rendering/geom/prim/Points.h>
#include <moonray/rendering/geom/prim/PolyMesh.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/geom/prim/VdbVolume.h>

#include <scene_rdl2/common/rec_time/RecTime.h>
#include <scene_rdl2/render/util/stdmemory.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

#include <embree4/rtcore.h>

#include <tbb/concurrent_unordered_map.h>

namespace scene_rdl2 {
using namespace math;
using namespace util;
}

namespace moonray {
namespace rt {


typedef tbb::concurrent_unordered_map<std::shared_ptr<geom::SharedPrimitive>,
        tbb::atomic<bool>, geom::SharedPtrHash> SharedSceneMap;


class BVHBuilder : public geom::PrimitiveVisitor
{
public:
    typedef geom::internal::BVHUserData::IntersectionFilterManager IntersectionFilterManager;

    BVHBuilder(const scene_rdl2::rdl2::Layer* layer, const scene_rdl2::rdl2::Geometry* geometry,
            RTCDevice& device, RTCScene& parentScene,
            SharedSceneMap& sharedSceneMap, BVHUserDataList& userData,
            bool getAssignments):
        mLayer(layer), mGeometry(geometry),
        mDevice(device), mParentScene(parentScene),
        mSharedSceneMap(sharedSceneMap), mBVHUserData(userData),
        mGetAssignments(getAssignments),
        mHasVolumeAssignment(false),
        mHasSurfaceAssignment(false) {}

    virtual void visitCurves(geom::Curves& c) override {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&c);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::CURVES);
        auto pCurves = static_cast<geom::internal::Curves*>(pImpl);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pCurves->isBVHInitialized()) {
            pCurves->setBVHHandle(createCurvesInBVH(*pCurves,
                c.getCurvesType(), c.getCurvesSubType(), c.getTessellationRate(), getGeomFlag()));
        } else {
            pCurves->updateBVHHandle();
        }
    }

    virtual void visitPoints(geom::Points& p) override {
        auto pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::QUADRIC);
        auto pPoints =
            static_cast<geom::internal::NamedPrimitive*>(pImpl);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pPoints->isBVHInitialized()) {
            pPoints->setBVHHandle(createQuadricInBVH(*pPoints, getGeomFlag()));
        } else {
            pPoints->updateBVHHandle();
        }
    }

    virtual void visitPolygonMesh(geom::PolygonMesh& p) override {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::POLYMESH);
        auto pMesh = static_cast<geom::internal::Mesh*>(pImpl);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pMesh->isBVHInitialized()) {
            pMesh->setBVHHandle(createPolyMeshInBVH(*pMesh, getGeomFlag()));
        } else {
            pMesh->updateBVHHandle();
        }
    }

    virtual void visitSphere(geom::Sphere& s) override {
        auto pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::QUADRIC);
        auto pSphere =
            static_cast<geom::internal::NamedPrimitive*>(pImpl);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pSphere->isBVHInitialized()) {
            pSphere->setBVHHandle(createQuadricInBVH(*pSphere, getGeomFlag()));
        } else {
            pSphere->updateBVHHandle();
        }
    }

    virtual void visitBox(geom::Box& b) override {
        auto pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&b);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::QUADRIC);
        auto pBox =
            static_cast<geom::internal::NamedPrimitive*>(pImpl);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pBox->isBVHInitialized()) {
            pBox->setBVHHandle(createQuadricInBVH(*pBox, getGeomFlag()));
        } else {
            pBox->updateBVHHandle();
        }
    }

    virtual void visitSubdivisionMesh(geom::SubdivisionMesh& s) override {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s);

        auto pMesh = static_cast<geom::internal::Mesh*>(pImpl);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pMesh->isBVHInitialized()) {
            pMesh->setBVHHandle(createPolyMeshInBVH(*pMesh, getGeomFlag()));
        } else {
            pMesh->updateBVHHandle();
        }
    }

    virtual void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override {
        // recursively travel down scene graph (we may have primitive group,
        // instances, regular primitives inside each primitive group)

        // when outer parallel loop involves lock/wait depends on
        // inner parrel loop, tbb may introduce deadlock due to its
        // job stealing mechanics. (inner loop done its tasks, stealing outer
        // loop tasks before unlock mutex). Disable parallel here explicitely
        // to avoid this issue. For related discussion, see:
        // https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/285550
        bool isParallel = false;
        pg.forEachPrimitive(*this, isParallel);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t) override {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitInstance(geom::Instance& i) override {
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedSceneMap.insert(std::make_pair(ref, false)).second) {
            RTCScene sharedScene = rtcNewScene(mDevice);
            rtcSetSceneBuildQuality(sharedScene, mGeometry->isStatic() ?
                RTC_BUILD_QUALITY_HIGH : RTC_BUILD_QUALITY_LOW);
            geom::internal::PrimitivePrivateAccess::setBVHScene(*ref,
                static_cast<void*>(sharedScene));
            BVHBuilder builder(mLayer, mGeometry, mDevice, sharedScene,
                mSharedSceneMap, mBVHUserData, mGetAssignments);
            ref->getPrimitive()->accept(builder);
            rtcCommitScene(sharedScene);
            // store if the reference contains volumes or surfaces
            if (mGetAssignments) {
                ref->setHasSurfaceAssignment(builder.getHasSurfaceAssignment());
                ref->setHasVolumeAssignment(builder.getHasVolumeAssignment());
            }
            // mark the BVH representation of referenced primitive (group)
            // has been correctly constructed so that all the instances
            // reference it can start accessing it
            mSharedSceneMap[ref] = true;
        }
        // wait for the first visited instance to construct the shared scene
        SharedSceneMap::const_iterator it = mSharedSceneMap.find(ref);
        while (it == mSharedSceneMap.end() || !it->second) {
            it = mSharedSceneMap.find(ref);
        }
        auto pImpl = geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&i);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::INSTANCE);
        auto pInstance = static_cast<geom::internal::Instance*>(pImpl);
        // bind the BVH representation to corresponding Primitive
        // or update the BVH representation if Primitive got deformed
        // (real time frame update case)
        if (mGeometry->isStatic() || !pInstance->isBVHInitialized()) {
            pInstance->setBVHHandle(createInstanceInBVH(*pInstance, getGeomFlag()));
        } else {
            pInstance->updateBVHHandle();
        }
    }

    virtual void visitVdbVolume(geom::VdbVolume& v) override {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&v);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::VDB_VOLUME);
        auto pVolume = static_cast<geom::internal::VdbVolume*>(pImpl);
        if (!pVolume->isEmpty()) {
            // bind the BVH representation to corresponding Primitive
            // or update the BVH representation if Primitive got deformed
            // (real time frame update case)
            if (mGeometry->isStatic() || !pVolume->isBVHInitialized()) {
                pVolume->setBVHHandle(createVolumeInBVH(*pVolume, getGeomFlag()));
            } else {
                pVolume->updateBVHHandle();
            }
        }
    }

    bool getHasSurfaceAssignment() const
    {
        MNRY_ASSERT(mGetAssignments);
        return mHasSurfaceAssignment;
    }

    bool getHasVolumeAssignment() const
    {
        MNRY_ASSERT(mGetAssignments);
        return mHasVolumeAssignment;
    }

private:

    std::unique_ptr<geom::internal::BVHHandle> createPolyMeshInBVH(
        const geom::internal::Mesh& geomMesh, const RTCBuildQuality flag) {

        geom::internal::Mesh::TessellatedMesh mesh;
        geomMesh.getTessellatedMesh(mesh);

        RTCGeometry rtcGeom;
        RTCFormat indexBufferFormat;
        switch (mesh.mIndexBufferType) {
        case geom::internal::MeshIndexType::TRIANGLE:
        {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_TRIANGLE);
            indexBufferFormat = RTC_FORMAT_UINT3;
            break;
        }
        case geom::internal::MeshIndexType::QUAD:
        {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_QUAD);
            indexBufferFormat = RTC_FORMAT_UINT4;
            break;
        }
        default:
            MNRY_ASSERT_REQUIRE(false);
            break;
        }
        size_t mbSteps = mesh.mVertexBufferDesc.size();
        rtcSetGeometryTimeStepCount(rtcGeom, mbSteps);

        // Set up mesh index buffer
        rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_INDEX, 0,
                                   indexBufferFormat, const_cast<void*>(
                                       (const void*)mesh.mIndexBufferDesc.mData),
                                   mesh.mIndexBufferDesc.mOffset,
                                   mesh.mIndexBufferDesc.mStride,
                                   mesh.mFaceCount);
        // Set up the polygon mesh vertex buffers, one for each motion step
        for (size_t i = 0; i < mbSteps; i++) {
            rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_VERTEX, i,
                RTC_FORMAT_FLOAT3, // xyz
                const_cast<void*>((const void*)mesh.mVertexBufferDesc[i].mData),
                mesh.mVertexBufferDesc[i].mOffset,
                mesh.mVertexBufferDesc[i].mStride,
                mesh.mVertexCount);
        }

        rtcSetGeometryMask(rtcGeom, resolveVisibilityMask(geomMesh));

        // set intersection filters
        IntersectionFilterManager* filterManager =
            new IntersectionFilterManager();
        bool hasVolumeAssignment = geomMesh.hasVolumeAssignment(mLayer);
        // force volume primitive to be two sided for odd-even test
        if (geomMesh.getIsSingleSided() && !hasVolumeAssignment) {
            filterManager->addIntersectionFilter(&backFaceCullingFilter);
            filterManager->addOcclusionFilter(&backFaceCullingFilter);
        }
        filterManager->addIntersectionFilter(&bssrdfTraceSetIntersectionFilter);
        if (hasVolumeAssignment) {
            filterManager->addIntersectionFilter(&manifoldVolumeIntervalFilter);
        }

        filterManager->addOcclusionFilter(&skipOcclusionFilter);
        installFilterCallbacks(rtcGeom, filterManager);

        // set user data
        geom::internal::BVHUserData* userData =
            new geom::internal::BVHUserData(mLayer, &geomMesh, filterManager);
        mBVHUserData.emplace_back(userData);
        rtcSetGeometryUserData(rtcGeom, (void*)userData);
        uint32_t geomID = rtcAttachGeometry(mParentScene, rtcGeom);
        rtcCommitGeometry(rtcGeom);
        return fauxstd::make_unique<geom::internal::BVHHandle>(
            mParentScene, geomID);
    }

    std::unique_ptr<geom::internal::BVHHandle> createQuadricInBVH(
        const geom::internal::NamedPrimitive& quadric,
        const RTCBuildQuality flag) {

        RTCGeometry rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryTimeStepCount(rtcGeom, 1);
        rtcSetGeometryBuildQuality(rtcGeom, flag);
        rtcSetGeometryUserPrimitiveCount(rtcGeom,
            quadric.getSubPrimitiveCount());

        // Set up bounds/intersection/occlusion kernel functions
        rtcSetGeometryBoundsFunction(rtcGeom,
            quadric.getBoundsFunction(), nullptr);
        rtcSetGeometryIntersectFunction(rtcGeom,
            quadric.getIntersectFunction());
        rtcSetGeometryOccludedFunction(rtcGeom,
            quadric.getOccludedFunction());

        rtcSetGeometryMask(rtcGeom, resolveVisibilityMask(quadric));

        // set intersection filter
        IntersectionFilterManager* filterManager =
            new IntersectionFilterManager();
        filterManager->addIntersectionFilter(&bssrdfTraceSetIntersectionFilter);
        if (quadric.hasVolumeAssignment(mLayer)) {
            filterManager->addIntersectionFilter(
                &manifoldVolumeIntervalFilter);
        }
        filterManager->addOcclusionFilter(&skipOcclusionFilter);
        installFilterCallbacks(rtcGeom, filterManager);

        // set user data
        geom::internal::BVHUserData* userData =
            new geom::internal::BVHUserData(mLayer, &quadric, filterManager);
        mBVHUserData.emplace_back(userData);
        rtcSetGeometryUserData(rtcGeom, (void*)userData);
        uint32_t geomID = rtcAttachGeometry(mParentScene, rtcGeom);
        rtcCommitGeometry(rtcGeom);
        return fauxstd::make_unique<geom::internal::BVHHandle>(
            mParentScene, geomID);
    }

    std::unique_ptr<geom::internal::BVHHandle> createCurvesInBVH(
        const geom::internal::Curves& geomCurves,
        const geom::Curves::Type curvesType,
        const geom::Curves::SubType curvesSubType,
        const int tessellationRate,
        const RTCBuildQuality flag) {

        rtcGetDeviceError(mDevice);  // clear error code

        geom::internal::Curves::Spans spans;
        geomCurves.getTessellatedSpans(spans);

        RTCGeometry rtcGeom;
        if (curvesType == geom::Curves::Type::LINEAR && curvesSubType == geom::Curves::SubType::RAY_FACING) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_FLAT_LINEAR_CURVE);
        } else if (curvesType == geom::Curves::Type::BEZIER && curvesSubType == geom::Curves::SubType::RAY_FACING) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_FLAT_BEZIER_CURVE);
        } else if (curvesType == geom::Curves::Type::BSPLINE && curvesSubType == geom::Curves::SubType::RAY_FACING) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_FLAT_BSPLINE_CURVE);
        } else if (curvesType == geom::Curves::Type::LINEAR && curvesSubType == geom::Curves::SubType::ROUND) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE);
        } else if (curvesType == geom::Curves::Type::BEZIER && curvesSubType == geom::Curves::SubType::ROUND) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_ROUND_BEZIER_CURVE);
        } else if (curvesType == geom::Curves::Type::BSPLINE && curvesSubType == geom::Curves::SubType::ROUND) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE);
        } else if (curvesType == geom::Curves::Type::BEZIER && curvesSubType == geom::Curves::SubType::NORMAL_ORIENTED) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_BEZIER_CURVE);
        } else if (curvesType == geom::Curves::Type::BSPLINE && curvesSubType == geom::Curves::SubType::NORMAL_ORIENTED) {
            rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_BSPLINE_CURVE);
        } else {
            MNRY_ASSERT_REQUIRE(false);
        }
        MNRY_ASSERT_REQUIRE(rtcGeom != NULL);

        size_t mbSteps = spans.mVertexBufferDesc.size();
        rtcSetGeometryTimeStepCount(rtcGeom, mbSteps);
        rtcSetGeometryBuildQuality(rtcGeom, flag);

        // Set up span index buffer
        MNRY_ASSERT_REQUIRE(spans.mIndexBufferDesc.mData != nullptr);
        rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_INDEX, 0,
            RTC_FORMAT_UINT,
            const_cast<void*>((const void*)spans.mIndexBufferDesc.mData),
            spans.mIndexBufferDesc.mOffset,
            spans.mIndexBufferDesc.mStride,
            spans.mSpanCount);

        // Set up the control vertex data buffers, one for each motion step
        for (size_t i = 0; i < mbSteps; i++) {
            MNRY_ASSERT_REQUIRE(spans.mVertexBufferDesc[i].mData != nullptr);
            rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_VERTEX, i,
                RTC_FORMAT_FLOAT4, // xyzr
                const_cast<void*>((const void*)spans.mVertexBufferDesc[i].mData),
                spans.mVertexBufferDesc[i].mOffset,
                spans.mVertexBufferDesc[i].mStride,
                spans.mVertexCount);
        }

        // Set up the normal data buffer for normal oriented curves
        if (curvesSubType == geom::Curves::SubType::NORMAL_ORIENTED) {
            rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_NORMAL, 0,
                RTC_FORMAT_FLOAT3,
                const_cast<void*>((const void*)spans.mNormalBufferDesc.mData),
                spans.mNormalBufferDesc.mOffset,
                spans.mNormalBufferDesc.mStride,
                spans.mVertexCount);
        }

        rtcSetGeometryMask(rtcGeom, resolveVisibilityMask(geomCurves));

        // set intersection filter
        IntersectionFilterManager* filterManager =
            new IntersectionFilterManager();
        filterManager->addIntersectionFilter(&bssrdfTraceSetIntersectionFilter);
        filterManager->addOcclusionFilter(&skipOcclusionFilter);
        installFilterCallbacks(rtcGeom, filterManager);

        rtcSetGeometryTessellationRate(rtcGeom, tessellationRate);

        // set user data
        geom::internal::BVHUserData* userData =
            new geom::internal::BVHUserData(mLayer, &geomCurves, filterManager);
        mBVHUserData.emplace_back(userData);
        rtcSetGeometryUserData(rtcGeom, (void*)userData);
        uint32_t geomID = rtcAttachGeometry(mParentScene, rtcGeom);
        rtcCommitGeometry(rtcGeom);

        // catch any embree errors in this function
        MNRY_ASSERT_REQUIRE(rtcGetDeviceError(mDevice) == RTC_ERROR_NONE);

        return fauxstd::make_unique<geom::internal::BVHHandle>(
            mParentScene, geomID);
    }

    std::unique_ptr<geom::internal::BVHHandle> createInstanceInBVH(
        const geom::internal::Instance& instance, const RTCBuildQuality flag) {

        RTCGeometry rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(rtcGeom, 1);
        rtcSetGeometryBuildQuality(rtcGeom, flag);
        // instancing kernel handle motion blur through matrix decomposition
        // so we don't need to feed in multiple buffers for motion blur case
        rtcSetGeometryTimeStepCount(rtcGeom, 1);
        // Set up bounds/intersection/occlusion kernel functions
        rtcSetGeometryBoundsFunction(rtcGeom,
            instance.getBoundsFunction(), nullptr);
        rtcSetGeometryIntersectFunction(rtcGeom,
            instance.getIntersectFunction());
        rtcSetGeometryOccludedFunction(rtcGeom,
            instance.getOccludedFunction());

        // The visibility mask needs to take volume assignments into account.
        // We use the visibility flags of the instance geometry, but do the
        // volume/surface shift based on the reference's assignments.
        const std::shared_ptr<geom::SharedPrimitive> &ref = instance.getReference();
        bool hasSurfaceAssignment = ref->getHasSurfaceAssignment();
        bool hasVolumeAssignment = ref->getHasVolumeAssignment();
        if (!hasSurfaceAssignment && !hasVolumeAssignment) {
            // In the case of multi-level instancing we'll need to
            // recurse through the primitives to check for volume assignments
            GetAssignments ga(hasVolumeAssignment, hasSurfaceAssignment);
            ref->getPrimitive()->accept(ga);
        }
        unsigned int mask = 0;
        if (hasSurfaceAssignment) {
            mask |= mGeometry->getVisibilityMask();
        }
        if (hasVolumeAssignment) {
            mask |= mGeometry->getVisibilityMask() << scene_rdl2::rdl2::sNumVisibilityTypes;
        }
        rtcSetGeometryMask(rtcGeom, mask);

        // set user data
        geom::internal::BVHUserData* userData =
            new geom::internal::BVHUserData(mLayer, &instance, nullptr);
        mBVHUserData.emplace_back(userData);
        rtcSetGeometryUserData(rtcGeom, (void*)userData);

        uint32_t geomID = rtcAttachGeometry(mParentScene, rtcGeom);
        rtcCommitGeometry(rtcGeom);
        return fauxstd::make_unique<geom::internal::BVHHandle>(
            mParentScene, geomID);
    }

    std::unique_ptr<geom::internal::BVHHandle> createVolumeInBVH(
        const geom::internal::VdbVolume& geomVolume,
        const RTCBuildQuality flag) {

        geom::internal::BufferDesc vertexBufferDesc[2];
        geom::internal::BufferDesc indexBufferDesc;
        size_t vertexCount, faceCount, mbSteps;
        geomVolume.getTessellatedMesh(vertexBufferDesc, indexBufferDesc,
            vertexCount, faceCount, mbSteps);

        RTCGeometry rtcGeom = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_QUAD);
        // the proxy bounding box for volume is always static,
        // we use velocity data for volume motion blur
        rtcSetGeometryTimeStepCount(rtcGeom, mbSteps);
        rtcSetGeometryBuildQuality(rtcGeom, flag);

        // Set up the quad mesh index buffer
        rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_INDEX, 0,
            RTC_FORMAT_UINT4,
            const_cast<void*>((const void*)indexBufferDesc.mData),
            indexBufferDesc.mOffset,
            indexBufferDesc.mStride,
            faceCount);
        // Set up the quad mesh vertex buffer, one for each motion step
        for (size_t i = 0; i < mbSteps; i++) {
            rtcSetSharedGeometryBuffer(
                rtcGeom, RTC_BUFFER_TYPE_VERTEX, i, RTC_FORMAT_FLOAT3,
                const_cast<void*>(static_cast<const void*>(vertexBufferDesc[i].mData)),
                vertexBufferDesc[i].mOffset,
                vertexBufferDesc[i].mStride,
                vertexCount);
        }

        rtcSetGeometryMask(rtcGeom, resolveVisibilityMask(geomVolume));

        // set intersection filters
        IntersectionFilterManager* filterManager =
            new IntersectionFilterManager();
        if (geomVolume.hasVolumeAssignment(mLayer)) {
            filterManager->addIntersectionFilter(&vdbVolumeIntervalFilter);
        }
        filterManager->addOcclusionFilter(&skipOcclusionFilter);
        installFilterCallbacks(rtcGeom, filterManager);

        // set user data
        geom::internal::BVHUserData* userData =
            new geom::internal::BVHUserData(mLayer, &geomVolume, filterManager);
        mBVHUserData.emplace_back(userData);
        rtcSetGeometryUserData(rtcGeom, (void*)userData);
        uint32_t geomID = rtcAttachGeometry(mParentScene, rtcGeom);
        rtcCommitGeometry(rtcGeom);
        return fauxstd::make_unique<geom::internal::BVHHandle>(
            mParentScene, geomID);
    }

    RTCBuildQuality getGeomFlag() const {
        return mGeometry->isStatic() ?
            RTC_BUILD_QUALITY_HIGH : RTC_BUILD_QUALITY_REFIT;
    }

    unsigned int resolveVisibilityMask(
            const geom::internal::NamedPrimitive& np) const
    {
        unsigned int mask = 0;
        if (np.hasSurfaceAssignment(mLayer)) {
            // visibility for surface ray query
            mask |= mGeometry->getVisibilityMask();
        }
        if (np.hasVolumeAssignment(mLayer)) {
            // visibility for volume ray query
            mask |= mGeometry->getVisibilityMask() << scene_rdl2::rdl2::sNumVisibilityTypes;
        }
        return mask;
    }

    void installFilterCallbacks(RTCGeometry rtcGeom,
            const IntersectionFilterManager* filterManager)
    {
        if (!filterManager->mIntersectionFilters.empty()) {
            rtcSetGeometryIntersectFilterFunction(rtcGeom,
                IntersectionFilterManager::intersectionFilter);
        }
        if (!filterManager->mOcclusionFilters.empty()) {
            rtcSetGeometryOccludedFilterFunction(rtcGeom,
                IntersectionFilterManager::occlusionFilter);
        }
    }

private:
    const scene_rdl2::rdl2::Layer* mLayer;
    const scene_rdl2::rdl2::Geometry* mGeometry;
    RTCDevice& mDevice;
    RTCScene mParentScene;

    SharedSceneMap& mSharedSceneMap;
    // Reference to container owned by EmbreeAccelerator
    // This container allow for safe allocation and
    // deletion of additional data needed for intersection filters
    BVHUserDataList& mBVHUserData;

    // When building scenes for shared primitives we need to know if they are
    // bound to volumes or materials in order to properly set the geometry
    // mask for instance primitives.
    bool mGetAssignments;
    bool mHasVolumeAssignment;
    bool mHasSurfaceAssignment;
};

static bool memoryMonitor(void* userPtr, const ssize_t bytes, const bool post)
{
    ((EmbreeAccelerator*)userPtr)->addMemoryUsage(bytes);
    return true;
}

EmbreeAccelerator::EmbreeAccelerator(const AcceleratorOptions& options):
    mBvhBuildProceduralTime(0.0),
    mRtcCommitTime(0.0),
    mRootScene(nullptr), mDevice(nullptr), mBVHMemory(0)
{
    std::string cfg = "threads=" + std::to_string(options.maxThreads);
    if (options.verbose) {
        cfg += ",verbose=2";
    }

    mDevice = rtcNewDevice(cfg.c_str());
    // monitor memory usage
    rtcSetDeviceMemoryMonitorFunction(mDevice, memoryMonitor, this);

    mRootScene = rtcNewScene(mDevice);
}

EmbreeAccelerator::~EmbreeAccelerator()
{
    // reset the root scene
    if (mRootScene != nullptr) {
        rtcReleaseScene(mRootScene);
        mRootScene = nullptr;
    }
    rtcReleaseDevice(mDevice);
}

void
buildBVHBottomUp(const scene_rdl2::rdl2::Layer* layer, scene_rdl2::rdl2::Geometry* geometry,
        RTCDevice& rtcDevice, RTCScene& rootScene,
        SharedSceneMap& visitedBVHScene,
        std::unordered_set<scene_rdl2::rdl2::Geometry*>& visitedGeometry,
        BVHUserDataList& bvhUserData)
{
    geom::Procedural* procedural = geometry->getProcedural();
    // All parts in a procedural are unassigned in the layer
    if (!procedural) {
        return;
    }
    // This BVH part for this particular rdl geometry has been constructed
    if (visitedGeometry.find(geometry) != visitedGeometry.end()) {
        return;
    }
    // Do a bottom up traversal so that referenced BVH got built first
    const scene_rdl2::rdl2::SceneObjectVector& references =
        geometry->get(scene_rdl2::rdl2::Geometry::sReferenceGeometries);
    for (const auto& ref : references) {
        if (!ref->isA<scene_rdl2::rdl2::Geometry>()) {
            continue;
        }
        scene_rdl2::rdl2::Geometry* referencedGeometry = ref->asA<scene_rdl2::rdl2::Geometry>();
        buildBVHBottomUp(layer, referencedGeometry, rtcDevice, rootScene,
            visitedBVHScene, visitedGeometry, bvhUserData);
    }
    // We disable the parallel here to solve the non-deterministic
    // issue for some hair/fur related scenes.
    // Also, testing shows no speed gain is achieved when
    // parallelize this for loop.
    bool doParallel = false;
    if (procedural->isReference()) {
        const std::shared_ptr<geom::SharedPrimitive>& ref =
            procedural->getReference();
        if (visitedBVHScene.insert(std::make_pair(ref, false)).second) {
            RTCScene sharedScene = rtcNewScene(rtcDevice);
            rtcSetSceneBuildQuality(sharedScene, geometry->isStatic()?
                RTC_BUILD_QUALITY_HIGH : RTC_BUILD_QUALITY_LOW);
            geom::internal::PrimitivePrivateAccess::setBVHScene(*ref,
                static_cast<void*>(sharedScene));
            BVHBuilder builder(layer, geometry, rtcDevice, sharedScene,
                visitedBVHScene, bvhUserData, /* get assignments = */ true);
            ref->getPrimitive()->accept(builder);
            rtcCommitScene(sharedScene);
            // mark the BVH representation of referenced primitive (group)
            // has been correctly constructed so that all the instances
            // reference it can start accessing it
            visitedBVHScene[ref] = true;
        }
    } else {
        BVHBuilder bvhBuilder(layer, geometry, rtcDevice, rootScene,
            visitedBVHScene, bvhUserData, /* get assignments = */ false);
        procedural->forEachPrimitive(bvhBuilder, doParallel);
    }
    visitedGeometry.insert(geometry);
}

void
EmbreeAccelerator::build(OptimizationTarget accelMode, ChangeFlag changeFlag,
        const scene_rdl2::rdl2::Layer *layer,
        const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
        const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s)
{
    if (changeFlag == ChangeFlag::ALL) {
        if (accelMode == OptimizationTarget::HIGH_QUALITY_BVH_BUILD) {
            rtcSetSceneBuildQuality(mRootScene, RTC_BUILD_QUALITY_HIGH);
            rtcSetSceneFlags(mRootScene, RTC_SCENE_FLAG_NONE);
        } else {
            rtcSetSceneBuildQuality(mRootScene, RTC_BUILD_QUALITY_LOW);
            rtcSetSceneFlags(mRootScene, RTC_SCENE_FLAG_DYNAMIC);
        }
    }
    scene_rdl2::rec_time::RecTime recTime;

    recTime.start();
    SharedSceneMap visitedBVHScene;
    std::unordered_set<scene_rdl2::rdl2::Geometry*> visitedGeometry;
    for (const auto& geometrySet : geometrySets) {
        const scene_rdl2::rdl2::SceneObjectIndexable& geometries = geometrySet->getGeometries();
        for (auto& sceneObject : geometries) {
            scene_rdl2::rdl2::Geometry* geometry = sceneObject->asA<scene_rdl2::rdl2::Geometry>();
            if (g2s != nullptr && g2s->find(geometry) == g2s->end()) {
                continue;
            }
            buildBVHBottomUp(layer, geometry, mDevice, mRootScene,
                visitedBVHScene, visitedGeometry, mBVHUserData);
        }
    }
    mBvhBuildProceduralTime = recTime.end();

    // now build the root scene
    recTime.start();
    rtcCommitScene(mRootScene);
    mRtcCommitTime = recTime.end();
}

// For debugging purpose
// Verify whether the input ray is valid (contains no nan value)
bool
isValidRay(const mcrt_common::Ray& ray)
{
    return isFinite(ray.org) && isFinite(ray.dir) &&
        !std::isnan(ray.tnear) && !std::isnan(ray.tfar);
}

void
EmbreeAccelerator::intersect(mcrt_common::Ray& ray) const
{
    MNRY_ASSERT(isValidRay(ray));

    // Carry per ray data that we need to process after rtcOccluded1...
    // The IntersectContext contains an embree RTCRayQueryContext and a 
    // MoonRay RayExtension object.
    // The RTCRayQueryContext object is the first member of the IntersectContext
    // struct so we do a hack where we pass the RTCRayQueryContext pointer through 
    // the rtcIntersect/rtcOccluded funcs and then cast that pointer back to a 
    // IntersectContext in the intersection filters and instance intersect/occluded
    // functions, which then lets us access the RayExtension.

    ray.id = 0;
    mcrt_common::IntersectContext context;
    context.mRayExtension = &ray.ext;

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &context.mRtcContext;

    rtcIntersect1(mRootScene, (RTCRayHit*)&ray, &args);

    if (ray.geomID != RTC_INVALID_GEOMETRY_ID &&
        ray.instID == RTC_INVALID_GEOMETRY_ID) {
        // intersect a regular primitive (if the intersection is an instance,
        // its userData has been filled in instance intersection kernel
        ray.ext.userData = rtcGetGeometryUserData(
            rtcGetGeometry(mRootScene, ray.geomID));
    }
}

bool
EmbreeAccelerator::occluded(mcrt_common::Ray& ray) const
{
    MNRY_ASSERT(isValidRay(ray));

    // Carry per ray data that we need to process after rtcOccluded1...
    // The IntersectContext contains an embree RTCRayQueryContext and a 
    // MoonRay RayExtension object.
    // The RTCRayQueryContext object is the first member of the IntersectContext
    // struct so we do a hack where we pass the RTCRayQueryContext pointer through 
    // the rtcIntersect/rtcOccluded funcs and then cast that pointer back to a 
    // IntersectContext in the intersection filters and instance intersect/occluded
    // functions, which then lets us access the RayExtension.

    ray.id = 0;
    mcrt_common::IntersectContext context;
    context.mRayExtension = &ray.ext;

    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.context = &context.mRtcContext;

    rtcOccluded1(mRootScene, (RTCRay*)&ray, &args);

    return ray.tfar < 0.0f;
}

scene_rdl2::math::BBox3f
EmbreeAccelerator::getBounds() const
{
    RTCBounds bounds;
    rtcGetSceneBounds(mRootScene, &bounds);

    if (bounds.lower_x > bounds.upper_x ||
        bounds.lower_y > bounds.upper_y ||
        bounds.lower_z > bounds.upper_z) {
        // There is probably no geometry in the scene. Return a default bounding
        // box centered at the origin.
        return scene_rdl2::math::BBox3f(scene_rdl2::math::Vec3f(-1.f), scene_rdl2::math::Vec3f(1.f));
    }

    return scene_rdl2::math::BBox3f(*((scene_rdl2::math::Vec3f *)&bounds.lower_x), *((scene_rdl2::math::Vec3f *)&bounds.upper_x));
}

} // namespace rt
} // namespace moonray

