// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "MetalGPUAccelerator.h"
#include "Math2MetalGPUMath.h"

#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/prim/Box.h>
#include <moonray/rendering/geom/prim/BVHHandle.h>
#include <moonray/rendering/geom/prim/CubicSpline.h>
#include <moonray/rendering/geom/prim/Curves.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/OpenSubdivMesh.h>
#include <moonray/rendering/geom/prim/Points.h>
#include <moonray/rendering/geom/prim/PolyMesh.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/geom/prim/Sphere.h>
#include <moonray/rendering/geom/prim/VdbVolume.h>
#include <moonray/rendering/rt/AcceleratorUtils.h>
#include <scene_rdl2/render/util/BitUtils.h>
#include <scene_rdl2/render/util/GetEnv.h>

#include <Metal/Metal.h>

using namespace moonray::geom;

namespace moonray {
namespace rt {


// Optix will call this callback for information / error messages.
static void optixMessageCallback(unsigned int level,
                                 const char *tag,
                                 const char *message,
                                 void *)
{
    scene_rdl2::logging::Logger::info("GPU: ", message);
}


class MetalGPUBVHBuilder : public geom::PrimitiveVisitor
{
public:
    // Extremely similar to EmbreeAccelerator.cc BVHBuilder.  This is intentional so
    // it behaves the same and is easy to maintain and debug.

    MetalGPUBVHBuilder(id<MTLDevice> context,
                  bool allowUnsupportedFeatures,
                  const scene_rdl2::rdl2::Layer* layer,
                  const scene_rdl2::rdl2::Geometry* geometry,
                  MetalGPUPrimitiveGroup* parentGroup,
                  SharedGroupMap& groups) :
        mContext(context),
        mAllowUnsupportedFeatures(allowUnsupportedFeatures),
        mFailed(false),
        mLayer(layer),
        mGeometry(geometry),
        mParentGroup(parentGroup),
        mSharedGroups(groups) {}

    virtual void visitCurves(geom::Curves& c) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&c);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::CURVES);
        const auto pCurves = static_cast<geom::internal::Curves*>(pImpl);

        createRoundCurves(*pCurves, c.getCurvesType(), c.getCurvesSubType());

    }

    virtual void visitPoints(geom::Points& p) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::QUADRIC);
        const auto pPoints = static_cast<const geom::internal::Points*>(pImpl);

        createPoints(*pPoints);
    }

    virtual void visitPolygonMesh(geom::PolygonMesh& p) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::POLYMESH);
        const auto pMesh = static_cast<geom::internal::Mesh*>(pImpl);

        createPolyMesh(*pMesh);
    }

    virtual void visitSphere(geom::Sphere& s) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::QUADRIC);
        const auto pSphere = static_cast<geom::internal::Sphere*>(pImpl);

        createSphere(*pSphere);
    }

    virtual void visitBox(geom::Box& b) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&b);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::QUADRIC);
        const auto pBox = static_cast<const geom::internal::Box*>(pImpl);

        createBox(*pBox);
    }

    virtual void visitSubdivisionMesh(geom::SubdivisionMesh& s) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::POLYMESH);
        const auto pMesh = static_cast<geom::internal::Mesh*>(pImpl);

        createPolyMesh(*pMesh);
    }

    virtual void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override
    {
        if (mFailed) {
            return;
        }

        // Recursively travel down scene graph (we may have primitive groups,
        // instances, regular primitives inside each primitive group).

        // Like EmbreeAccelerator.cc::visitPrimitiveGroup(), we do not run parallel here.
        // See that function for explanation.
        bool isParallel = false;
        pg.forEachPrimitive(*this, isParallel);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t) override
    {
        if (mFailed) {
            return;
        }

        t.getPrimitive()->accept(*this);
    }

    virtual void visitInstance(geom::Instance& i) override
    {
        if (mFailed) {
            return;
        }

        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedGroups.insert(std::make_pair(ref, nullptr)).second) {
            MetalGPUPrimitiveGroup *group = new MetalGPUPrimitiveGroup(mContext);
            MetalGPUBVHBuilder builder(mContext, mAllowUnsupportedFeatures, mLayer, mGeometry, group, mSharedGroups);
            ref->getPrimitive()->accept(builder);
            // mark the BVH representation of referenced primitive (group)
            // has been correctly constructed so that all the instances
            // reference it can start accessing it
            mSharedGroups[ref] = group;
        }
        // wait for the first visited instance to construct the shared scene
        SharedGroupMap::const_iterator it = mSharedGroups.find(ref);
        while (it == mSharedGroups.end() || !it->second) {
            it = mSharedGroups.find(ref);
        }
        auto pImpl = geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&i);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::INSTANCE);
        auto pInstance = static_cast<geom::internal::Instance*>(pImpl);

        createInstance(*pInstance, mSharedGroups[ref]);
    }

    virtual void visitVdbVolume(geom::VdbVolume& v) override
    {
        if (mFailed) {
            return;
        }

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&v);
        MNRY_ASSERT_REQUIRE(pImpl != nullptr);
        MNRY_ASSERT_REQUIRE(pImpl->getType() == geom::internal::Primitive::VDB_VOLUME);
        const auto pVolume = static_cast<geom::internal::VdbVolume*>(pImpl);

        if (!pVolume->isEmpty()) {
            // Volumes are only visible for regular rays intersected in Scene::intersectVolumes()
            // so we don't need to consider them because that calls scalar Embree intersect()
            // which isn't vectorized.
        }
    }

    const std::vector<std::string>& warningMsgs() const { return mWarningMsgs; }

    bool hasFailed() const { return mFailed; }
    std::string whyFailed() const { return mWhyFailed; }

private:
    void logWarningMsg(const std::string& msg);

    void createBox(const geom::internal::Box& geomBox);

    void createCurves(const geom::internal::Curves& geomCurves,
                      const geom::Curves::Type curvesType,
                      const int tessellationRate);

    void createRoundCurves(const geom::internal::Curves& geomCurves,
                           const geom::Curves::Type curvesType,
                           const geom::Curves::SubType curvesSubType);

    void createPoints(const geom::internal::Points& geomPoints);

    void createPolyMesh(const geom::internal::Mesh& geomMesh);

    void createSphere(const geom::internal::Sphere& geomSphere);

    void createInstance(const geom::internal::Instance& instance,
                        MetalGPUPrimitiveGroup* group);

    unsigned int resolveVisibilityMask(const geom::internal::NamedPrimitive& np) const;

    bool getShadowLinkingLights(const geom::internal::NamedPrimitive& np,
                                MetalGPUBuffer<ShadowLinkLight>& lightsBuf) const;

    bool getShadowLinkingReceivers(const geom::internal::NamedPrimitive& np,
                                   MetalGPUBuffer<ShadowLinkReceiver>& receiversBuf) const;

    bool mAllowUnsupportedFeatures;
    std::vector<std::string> mWarningMsgs;
    id<MTLDevice> mContext;
    bool mFailed;
    std::string mWhyFailed;
    const scene_rdl2::rdl2::Layer* mLayer;
    const scene_rdl2::rdl2::Geometry* mGeometry;
    MetalGPUPrimitiveGroup* mParentGroup;
    SharedGroupMap& mSharedGroups;
};

void
MetalGPUBVHBuilder::logWarningMsg(const std::string& msg)
{
    for (auto& prevMsg : mWarningMsgs) {
        if (msg == prevMsg) {
            // only log a particular warning message once
            return;
        }
    }
    mWarningMsgs.push_back(msg);
}

void
MetalGPUBVHBuilder::createBox(const geom::internal::Box& geomBox)
{
    MetalGPUBox* gpuBox = new MetalGPUBox(mContext);
    mParentGroup->mCustomPrimitives.push_back(gpuBox);

    // Boxes only have one motion sample

    gpuBox->mInputFlags = 0;
    gpuBox->mIsSingleSided = geomBox.getIsSingleSided();
    gpuBox->mIsNormalReversed = geomBox.getIsNormalReversed();
    gpuBox->mVisibleShadow = resolveVisibilityMask(geomBox) & scene_rdl2::rdl2::SHADOW;

    if (!getShadowLinkingLights(geomBox,
                                gpuBox->mShadowLinkLights)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking light IDs to the GPU";
        return;
    }

    if (!getShadowLinkingReceivers(geomBox,
                                   gpuBox->mShadowLinkReceivers)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking receiver IDs to the GPU";
        return;
    }

    std::vector<int> assignmentIds(1);
    assignmentIds[0] = geomBox.getIntersectionAssignmentId(0);
    if (gpuBox->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
        return;
    }
    [gpuBox->mAssignmentIds.deviceptr() setLabel:@"GPUBox: AssigmentIDs Buf"];

    gpuBox->mL2P = mat43ToMetalGPUXform(geomBox.getL2P());
    gpuBox->mP2L = mat43ToMetalGPUXform(geomBox.getP2L());
    gpuBox->mLength = geomBox.getLength();
    gpuBox->mWidth = geomBox.getWidth();
    gpuBox->mHeight = geomBox.getHeight();
}

void
MetalGPUBVHBuilder::createCurves(const geom::internal::Curves& geomCurves,
                            const geom::Curves::Type curvesType,
                            const int tessellationRate)
{
    geom::internal::Curves::Spans spans;
    geomCurves.getTessellatedSpans(spans);

    MetalGPUCurve* gpuCurve = new MetalGPUCurve(mContext);
    mParentGroup->mCustomPrimitives.push_back(gpuCurve);

    gpuCurve->mInputFlags = 0;
    gpuCurve->mIsSingleSided = false;
    gpuCurve->mIsNormalReversed = false;
    gpuCurve->mVisibleShadow = resolveVisibilityMask(geomCurves) & scene_rdl2::rdl2::SHADOW;

    if (!getShadowLinkingLights(geomCurves,
                                gpuCurve->mShadowLinkLights)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking light IDs to the GPU";
        return;
    }

    if (!getShadowLinkingReceivers(geomCurves,
                                   gpuCurve->mShadowLinkReceivers)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking receiver IDs to the GPU";
        return;
    }

    gpuCurve->mMotionSamplesCount = geomCurves.getMotionSamplesCount();
    gpuCurve->mHostIndices.resize(spans.mSpanCount);
    gpuCurve->mNumControlPoints = spans.mVertexCount;
    gpuCurve->mHostControlPoints.resize(spans.mVertexCount * gpuCurve->mMotionSamplesCount);

    switch (curvesType) {
    case geom::Curves::Type::LINEAR:
        gpuCurve->mBasis = LINEAR;
    break;
    case geom::Curves::Type::BEZIER:
        gpuCurve->mBasis = BEZIER; // the rdlcurves default
    break;
    case geom::Curves::Type::BSPLINE:
        gpuCurve->mBasis = BSPLINE; // the furdeform/willow default
    break;
    default:
        MNRY_ASSERT_REQUIRE(false);
    }

    const unsigned int* indices = reinterpret_cast<const unsigned int*>(spans.mIndexBufferDesc.mData);
    std::vector<int> assignmentIds(spans.mSpanCount);

    for (size_t i = 0; i < spans.mSpanCount; i++) {
        // We only want every third element of the index buffer to get the vertex index.
        // See: geom/prim/Curves.h struct IndexData
        gpuCurve->mHostIndices[i] = indices[i * 3];
        assignmentIds[i] = geomCurves.getIntersectionAssignmentId(i);
    }

    const geom::Curves::VertexBuffer& verts = geomCurves.getVertexBuffer();

    for (int ms = 0; ms < gpuCurve->mMotionSamplesCount; ms++) {
        for (size_t i = 0; i < spans.mVertexCount; i++) {
            const scene_rdl2::math::Vec3fa& cp = verts(i, ms);
            gpuCurve->mHostControlPoints[ms * spans.mVertexCount + i] =
                {cp.x, cp.y, cp.z, cp.w};
        }
    }

    if (gpuCurve->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
        return;
    }
    [gpuCurve->mAssignmentIds.deviceptr() setLabel:@"GPUCurve: AssigmentIDs Buf"];

    if (gpuCurve->mIndices.allocAndUpload(gpuCurve->mHostIndices) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve indices to the GPU";
        return;
    }
    [gpuCurve->mIndices.deviceptr() setLabel:@"GPUCurve: Indices Buf"];

    if (gpuCurve->mControlPoints.allocAndUpload(gpuCurve->mHostControlPoints)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve control points to the GPU";
        return;
    }
    [gpuCurve->mControlPoints.deviceptr() setLabel:@"GPUCurve: Control Points Buf"];

    gpuCurve->mSegmentsPerCurve = tessellationRate;  // for bezier and bspline, embree default
}

void
MetalGPUBVHBuilder::createRoundCurves(const geom::internal::Curves& geomCurves,
                                 const geom::Curves::Type curvesType,
                                 const geom::Curves::SubType curvesSubType)
{

    // This code assumes that there is a max of 2 motion samples.
    // TODO: add support for round curves with more motion samples.
    bool motionBlur = geomCurves.getMotionSamplesCount() > 1;
    if (geomCurves.getMotionSamplesCount() > 2) {
        mFailed = true;
        mWhyFailed = "Round curves with more than 2 motion samples are currently unsupported in XPU mode";
        return;
    }

    geom::internal::Curves::Spans spans;
    geomCurves.getTessellatedSpans(spans);

    MetalGPURoundCurves* gpuCurve = new MetalGPURoundCurves(mContext);
    if (!motionBlur) {
        mParentGroup->mRoundCurves.push_back(gpuCurve);
    } else {
        mParentGroup->mRoundCurvesMB.push_back(gpuCurve);
    }

    gpuCurve->mInputFlags = 0;
    gpuCurve->mIsSingleSided = false;
    gpuCurve->mIsNormalReversed = false;
    gpuCurve->mVisibleShadow = resolveVisibilityMask(geomCurves) & scene_rdl2::rdl2::SHADOW;

    if (!getShadowLinkingLights(geomCurves,
                                gpuCurve->mShadowLinkLights)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking light IDs to the GPU";
        return;
    }

    if (!getShadowLinkingReceivers(geomCurves,
                                   gpuCurve->mShadowLinkReceivers)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking receiver IDs to the GPU";
        return;
    }

    gpuCurve->mMotionSamplesCount = geomCurves.getMotionSamplesCount();

    switch (curvesSubType) {
       case geom::Curves::SubType::RAY_FACING:
       gpuCurve->mSubType = MTLCurveTypeFlat;
    break;
       case geom::Curves::SubType::ROUND:
       gpuCurve->mSubType = MTLCurveTypeRound;
    break;
    default:
       MNRY_ASSERT_REQUIRE(false);
    }

    switch (curvesType) {
    case geom::Curves::Type::LINEAR:
       gpuCurve->mType = MTLCurveBasisLinear;
    break;
    case geom::Curves::Type::BSPLINE:
       gpuCurve->mType = MTLCurveBasisBSpline;
    break;
    case geom::Curves::Type::BEZIER:
      gpuCurve->mType = MTLCurveBasisBezier;
    break;
    default:
       MNRY_ASSERT_REQUIRE(false);
    }

    gpuCurve->mNumControlPoints = spans.mVertexCount;
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(spans.mIndexBufferDesc.mData);

    std::vector<int> assignmentIds(spans.mSpanCount);
    std::vector<unsigned int> hostIndices(spans.mSpanCount);
    std::vector<float3> hostVertices(spans.mVertexCount * gpuCurve->mMotionSamplesCount);
    std::vector<float> hostWidths(spans.mVertexCount * gpuCurve->mMotionSamplesCount);

    for (size_t i = 0; i < spans.mSpanCount; i++) {
        // We only want every third element of the index buffer to get the vertex index.
        // See: geom/prim/Curves.h struct IndexData
        hostIndices[i] = indices[i * 3];
        assignmentIds[i] = geomCurves.getIntersectionAssignmentId(i);
    }

    const geom::Curves::VertexBuffer& verts = geomCurves.getVertexBuffer();

    for (int ms = 0; ms < gpuCurve->mMotionSamplesCount; ms++) {
        for (size_t i = 0; i < spans.mVertexCount; i++) {
            const scene_rdl2::math::Vec3fa& cp = verts(i, ms);
            hostVertices[ms * spans.mVertexCount + i] = {cp.x, cp.y, cp.z};
            hostWidths[ms * spans.mVertexCount + i] = cp.w;
        }
    }

    if (gpuCurve->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
        return;
    }
    [gpuCurve->mAssignmentIds.deviceptr() setLabel:@"GPUCurve: AssigmentIDs Buf"];

    if (gpuCurve->mIndices.allocAndUpload(hostIndices) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve indices to the GPU";
        return;
    }
    [gpuCurve->mIndices.deviceptr() setLabel:@"GPUCurve: Indices Buf"];

    if (gpuCurve->mVertices.allocAndUpload(hostVertices) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve vertices to the GPU";
        return;
    }
    [gpuCurve->mVertices.deviceptr() setLabel:@"GPUCurve: Vertices Buf"];

    if (gpuCurve->mWidths.allocAndUpload(hostWidths) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve widths to the GPU";
        return;
    }
    [gpuCurve->mWidths.deviceptr() setLabel:@"GPUCurve: Widths Buf"];

    // Get the vertex/width buffer pointers.  For curves without motion blur, the
    // second pointer is null.
    gpuCurve->mVerticesPtrs[0] = 0;
    gpuCurve->mWidthsPtrs[0] = 0;
    if (gpuCurve->mMotionSamplesCount == 2) {
        gpuCurve->mVerticesPtrs[1] = gpuCurve->mNumControlPoints * sizeof(float3);
        gpuCurve->mWidthsPtrs[1] = gpuCurve->mNumControlPoints * sizeof(float);
    } else {
        gpuCurve->mVerticesPtrs[1] = 0;
        gpuCurve->mWidthsPtrs[1] = 0;
    }
}

void
MetalGPUBVHBuilder::createPoints(const geom::internal::Points& geomPoints)
{
    MetalGPUPoints* gpuPoints = new MetalGPUPoints(mContext);
    mParentGroup->mCustomPrimitives.push_back(gpuPoints);

    gpuPoints->mInputFlags = 0;
    gpuPoints->mIsSingleSided = false;
    gpuPoints->mIsNormalReversed = false;
    gpuPoints->mVisibleShadow = resolveVisibilityMask(geomPoints) & scene_rdl2::rdl2::SHADOW;

    if (!getShadowLinkingLights(geomPoints,
                                gpuPoints->mShadowLinkLights)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking light IDs to the GPU";
        return;
    }

    if (!getShadowLinkingReceivers(geomPoints,
                                   gpuPoints->mShadowLinkReceivers)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking receiver IDs to the GPU";
        return;
    }

    gpuPoints->mMotionSamplesCount = geomPoints.getMotionSamplesCount();

    int numPoints = geomPoints.getSubPrimitiveCount();
    std::vector<int> assignmentIds(numPoints);
    gpuPoints->mHostPoints.resize(numPoints * gpuPoints->mMotionSamplesCount);

    const geom::Points::VertexBuffer& verts = geomPoints.getVertexBuffer();
    const geom::Points::RadiusBuffer& radii = geomPoints.getRadiusBuffer();

    for (int i = 0; i < numPoints; i++) {
        const float radius = radii[i]; // radius doesn't have motion samples
        for (int ms = 0; ms < gpuPoints->mMotionSamplesCount; ms++) {
            const scene_rdl2::math::Vec3f& vert = verts(i, ms);
            gpuPoints->mHostPoints[i * gpuPoints->mMotionSamplesCount + ms] =
                {vert.x, vert.y, vert.z, radius};
        }
        assignmentIds[i] = geomPoints.getIntersectionAssignmentId(i);
    }

    if (gpuPoints->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
        return;
    }
    [gpuPoints->mAssignmentIds.deviceptr() setLabel:@"GPUPoints: AssigmentIDs Buf"];

    if (gpuPoints->mPoints.allocAndUpload(gpuPoints->mHostPoints) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the points to the GPU";
        return;
    }
    [gpuPoints->mPoints.deviceptr() setLabel:@"GPUPoints: Points Buf"];
}

void
MetalGPUBVHBuilder::createPolyMesh(const geom::internal::Mesh& geomMesh)
{
    geom::internal::Mesh::TessellatedMesh mesh;
    geomMesh.getTessellatedMesh(mesh);

    // This code assumes that there is a max of 2 motion samples
    // TODO: add support for mesh with more motion samples.
    const size_t mbSamples = mesh.mVertexBufferDesc.size();
    const bool enableMotionBlur = mbSamples  > 1;
    if (mbSamples  > 2) {
        mFailed = true;
        mWhyFailed = "Meshes with more than 2 motion samples are currently unsupported in XPU mode";
        return;
    }

    MetalGPUTriMesh* gpuMesh = new MetalGPUTriMesh(mContext);
    gpuMesh->mName = geomMesh.getName();
    if (enableMotionBlur) {
        mParentGroup->mTriMeshesMB.push_back(gpuMesh);
    } else {
        mParentGroup->mTriMeshes.push_back(gpuMesh);
    }

    bool hasVolumeAssignment = geomMesh.hasVolumeAssignment(mLayer);

    gpuMesh->mInputFlags = 0;
    gpuMesh->mIsSingleSided = geomMesh.getIsSingleSided() && !hasVolumeAssignment;
    gpuMesh->mIsNormalReversed = geomMesh.getIsNormalReversed();
    gpuMesh->mVisibleShadow = resolveVisibilityMask(geomMesh) & scene_rdl2::rdl2::SHADOW;
    gpuMesh->mEnableMotionBlur = enableMotionBlur;

    if (!getShadowLinkingLights(geomMesh,
                                gpuMesh->mShadowLinkLights)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking light IDs to the GPU";
        return;
    }

    if (!getShadowLinkingReceivers(geomMesh,
                                   gpuMesh->mShadowLinkReceivers)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking receiver IDs to the GPU";
        return;
    }

    unsigned int vertsPerFace;
    switch (mesh.mIndexBufferType) {
    case geom::internal::MeshIndexType::TRIANGLE:
    {
        vertsPerFace = 3;
        break;
    }
    case geom::internal::MeshIndexType::QUAD:
    {
        vertsPerFace = 4;
        break;
    }
    default:
        MNRY_ASSERT_REQUIRE(false);
        break;
    }

    // Record the bounding box of all vertices
    scene_rdl2::math::Vec3fa vMin(std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
    scene_rdl2::math::Vec3fa vMax(std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest());

    if (vertsPerFace == 3) {
        gpuMesh->mNumVertices = mesh.mVertexCount;
        gpuMesh->mNumFaces = mesh.mFaceCount;

        // When a mesh uses motion blur, the vertices of each motion key are stored consecutively in a single buffer.
        // According to Nvidia, GAS construction is most efficient when each motion buffer is 16-byte aligned. This
        // isn't a problem for the first motion key buffer, but the second motion key buffer may not be correctly aligned.
        // We handle that case here. Note that motionKeyOffsets is {0, 0} if there is no motion. This way, if some mesh
        // don't support motion blur, their mVerticesPtrs will both point to the same buffer. Optix needs both pointers
        // to point to a valid buffer.
        const std::array<size_t, 2> motionKeyOffsets = {0, enableMotionBlur ? scene_rdl2::util::alignUp<size_t>(mesh.mVertexCount, 4) : 0};
        const size_t gpuVerticesSize = motionKeyOffsets[1] + mesh.mVertexCount;
        std::vector<float3> gpuVertices(gpuVerticesSize);

        std::vector<unsigned int> gpuIndices(mesh.mFaceCount * 3);
        const unsigned int* indices = reinterpret_cast<const unsigned int*>(mesh.mIndexBufferDesc.mData);
        std::vector<int> assignmentIds(mesh.mFaceCount);

        // First set all assignmentIds and indices
        for (size_t i = 0; i < mesh.mFaceCount; i++) {
            assignmentIds[i] = geomMesh.getIntersectionAssignmentId(i);
            gpuIndices[i * 3 + 0] = indices[i * 3 + 0];
            gpuIndices[i * 3 + 1] = indices[i * 3 + 1];
            gpuIndices[i * 3 + 2] = indices[i * 3 + 2];

            assert(gpuIndices[i * 3 + 0] < gpuMesh->mNumVertices);
            assert(gpuIndices[i * 3 + 1] < gpuMesh->mNumVertices);
            assert(gpuIndices[i * 3 + 2] < gpuMesh->mNumVertices);
        }

        for (size_t ms = 0; ms < mbSamples; ms++) {
            // Because the offset and stride members of mVertexBufferDesc are in bytes, we want to perform pointer
            // arithmetic on pointers to char (or anything byte sized).
            const char* vertices = reinterpret_cast<const char*>(mesh.mVertexBufferDesc[ms].mData);

            const unsigned int offset = mesh.mVertexBufferDesc[ms].mOffset;
            const unsigned int stride = mesh.mVertexBufferDesc[ms].mStride;
            const size_t motionKeyOffset = motionKeyOffsets[ms];

            for (size_t i = 0; i < mesh.mVertexCount; i++) {
                scene_rdl2::math::Vec3fa vtx = *reinterpret_cast<const scene_rdl2::math::Vec3fa*>(&vertices[offset + stride * i]);
                gpuVertices[motionKeyOffset + i] = {vtx.x, vtx.y, vtx.z};

                if (vtx.x < vMin.x) vMin.x = vtx.x;
                if (vtx.y < vMin.y) vMin.y = vtx.y;
                if (vtx.z < vMin.z) vMin.z = vtx.z;

                if (vtx.x > vMax.x) vMax.x = vtx.x;
                if (vtx.y > vMax.y) vMax.y = vtx.y;
                if (vtx.z > vMax.z) vMax.z = vtx.z;
            }
        }

        gpuMesh->mBounds[0].x = vMin.x;
        gpuMesh->mBounds[0].y = vMin.y;
        gpuMesh->mBounds[0].z = vMin.z;
        gpuMesh->mBounds[1].x = vMax.x;
        gpuMesh->mBounds[1].y = vMax.y;
        gpuMesh->mBounds[1].z = vMax.z;

        if (gpuMesh->mVertices.allocAndUpload(gpuVertices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the vertices to the GPU";
            return;
        }
        [gpuMesh->mVertices.deviceptr() setLabel:@"GPUMesh: Vertices Buffer"];
        
        if (gpuMesh->mIndices.allocAndUpload(gpuIndices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the indices to the GPU";
            return;
        }
        [gpuMesh->mIndices.deviceptr() setLabel:@"GPUMesh: Indices Buffer"];
        
        if (gpuMesh->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
            return;
        }
        [gpuMesh->mAssignmentIds.deviceptr() setLabel:@"GPUMesh: AssigmentIDs Buf"];

        gpuMesh->mVerticesPtrs[0] = 0;
        gpuMesh->mVerticesPtrs[1] = motionKeyOffsets[1] * sizeof(float3);

    } else if (vertsPerFace == 4) {
        // convert quads to tris

        gpuMesh->mNumVertices = mesh.mVertexCount;
        gpuMesh->mNumFaces = mesh.mFaceCount * 2;

        // See above for reason this code exists:
        const std::array<size_t, 2> motionKeyOffsets = {0, enableMotionBlur ? scene_rdl2::util::alignUp<size_t>(mesh.mVertexCount, 4) : 0};
        const size_t gpuVerticesSize = motionKeyOffsets[1] + mesh.mVertexCount;
        std::vector<float3> gpuVertices(gpuVerticesSize);

        std::vector<unsigned int> gpuIndices(mesh.mFaceCount * 6);
        const unsigned int* indices = reinterpret_cast<const unsigned int*>(mesh.mIndexBufferDesc.mData);
        std::vector<int> assignmentIds(mesh.mFaceCount * 2);

        // First set all assignmentIds:
        for (size_t i = 0; i < mesh.mFaceCount; i++) {
            assignmentIds[i * 2 + 0] = geomMesh.getIntersectionAssignmentId(i);
            assignmentIds[i * 2 + 1] = assignmentIds[i * 2];
            // quad -> two tris
            unsigned int idx0 = indices[i * 4 + 0];
            unsigned int idx1 = indices[i * 4 + 1];
            unsigned int idx2 = indices[i * 4 + 2];
            unsigned int idx3 = indices[i * 4 + 3];
            // first tri 0-1-3
            gpuIndices[i * 6 + 0] = idx0;
            gpuIndices[i * 6 + 1] = idx1;
            gpuIndices[i * 6 + 2] = idx3;
            // second tri 1-2-3
            gpuIndices[i * 6 + 3] = idx1;
            gpuIndices[i * 6 + 4] = idx2;
            gpuIndices[i * 6 + 5] = idx3;
        }

        for (size_t ms = 0; ms < mbSamples; ms++) {
            // Because the offset and stride members of mVertexBufferDesc are in bytes, we want to perform pointer
            // arithmetic on pointers to char (or anything byte sized).
            const char* vertices = reinterpret_cast<const char*>(mesh.mVertexBufferDesc[ms].mData);

            const unsigned int offset = mesh.mVertexBufferDesc[ms].mOffset;
            const unsigned int stride = mesh.mVertexBufferDesc[ms].mStride;
            const size_t motionKeyOffset = motionKeyOffsets[ms];

            for (size_t i = 0; i < mesh.mVertexCount; i++) {
                scene_rdl2::math::Vec3fa vtx = *reinterpret_cast<const scene_rdl2::math::Vec3fa*>(&vertices[offset + stride * i]);
                gpuVertices[motionKeyOffset + i] = {vtx.x, vtx.y, vtx.z};

                if (vtx.x < vMin.x) vMin.x = vtx.x;
                if (vtx.y < vMin.y) vMin.y = vtx.y;
                if (vtx.z < vMin.z) vMin.z = vtx.z;

                if (vtx.x > vMax.x) vMax.x = vtx.x;
                if (vtx.y > vMax.y) vMax.y = vtx.y;
                if (vtx.z > vMax.z) vMax.z = vtx.z;
            }
        }

        if (gpuMesh->mVertices.allocAndUpload(gpuVertices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the vertices to the GPU";
            return;
        }
        [gpuMesh->mVertices.deviceptr() setLabel:@"GPUMesh: Vertices Buffer"];
        
        if (gpuMesh->mIndices.allocAndUpload(gpuIndices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the indices to the GPU";
            return;
        }
        [gpuMesh->mIndices.deviceptr() setLabel:@"GPUMesh: Indices Buffer"];
        
        if (gpuMesh->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
            return;
        }
        [gpuMesh->mAssignmentIds.deviceptr() setLabel:@"GPUMesh: AssigmentIDs Buf"];

        gpuMesh->mVerticesPtrs[0] = 0;
        gpuMesh->mVerticesPtrs[1] = motionKeyOffsets[1] * sizeof(float3);
    }
}

void
MetalGPUBVHBuilder::createSphere(const geom::internal::Sphere& geomSphere)
{
    MetalGPUSphere* gpuSphere = new MetalGPUSphere(mContext);
    mParentGroup->mCustomPrimitives.push_back(gpuSphere);

    // Spheres only have one motion sample

    gpuSphere->mInputFlags = 0;
    gpuSphere->mIsSingleSided = geomSphere.getIsSingleSided();
    gpuSphere->mIsNormalReversed = geomSphere.getIsNormalReversed();
    gpuSphere->mVisibleShadow = resolveVisibilityMask(geomSphere) & scene_rdl2::rdl2::SHADOW;


    if (!getShadowLinkingLights(geomSphere,
                                gpuSphere->mShadowLinkLights)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking light IDs to the GPU";
        return;
    }

    if (!getShadowLinkingReceivers(geomSphere,
                                   gpuSphere->mShadowLinkReceivers)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the shadow linking receiver IDs to the GPU";
        return;
    }

    std::vector<int> assignmentIds(1);
    assignmentIds[0] = geomSphere.getIntersectionAssignmentId(0);
    if (gpuSphere->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
        return;
    }
    [gpuSphere->mAssignmentIds.deviceptr() setLabel:@"GPUSphere: AssigmentIDs Buf"];

    gpuSphere->mL2P = mat43ToMetalGPUXform(geomSphere.getL2P());
    gpuSphere->mP2L = mat43ToMetalGPUXform(geomSphere.getP2L());
    gpuSphere->mRadius = geomSphere.getRadius();
    gpuSphere->mPhiMax = geomSphere.getPhiMax();
    gpuSphere->mZMin = geomSphere.getZMin();
    gpuSphere->mZMax = geomSphere.getZMax();
}

void
MetalGPUBVHBuilder::createInstance(const geom::internal::Instance& instance,
                              MetalGPUPrimitiveGroup* group)
{
    const geom::internal::MotionTransform& l2pXform = instance.getLocal2Parent();

    scene_rdl2::math::Xform3f xforms[MetalGPUInstance::sNumMotionKeys];
    bool hasMotionBlur;
    if (!l2pXform.isMotion()) {
        xforms[0] = l2pXform.getStaticXform();
        hasMotionBlur = false;
    } else {
        // eval() the l2pXform at sNumMotionKeys discrete timesteps between time [0, 1].
        // Optix can't slerp() between matrices so we need to generate smaller
        // timesteps it can lerp() between.
        for (int i = 0; i < MetalGPUInstance::sNumMotionKeys; i++) {
            float t = i / static_cast<float>(MetalGPUInstance::sNumMotionKeys - 1);
            xforms[i] = l2pXform.eval(t);
        }
        hasMotionBlur = true;
    }

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

    bool visibleShadow = mask & scene_rdl2::rdl2::SHADOW;
    if (!visibleShadow) {
        // Instance doesn't cast a shadow so it doesn't exist for occlusion
        // queries.  Just skip it.  We will need to add some more logic when
        // regular rays are supported by XPU.
        return;
    }

    MetalGPUInstance* gpuInstance = new MetalGPUInstance();
    mParentGroup->mInstances.push_back(gpuInstance);
    gpuInstance->mGroup = group;

    if (hasMotionBlur) {
        for (int i = 0; i < MetalGPUInstance::sNumMotionKeys; i++) {
            gpuInstance->mXforms[i] = mat43ToMetalGPUXform(xforms[i]);
        }
    } else {
        gpuInstance->mXforms[0] = mat43ToMetalGPUXform(xforms[0]);
    }

    gpuInstance->mHasMotionBlur = hasMotionBlur;
}

unsigned int
MetalGPUBVHBuilder::resolveVisibilityMask(const geom::internal::NamedPrimitive& np) const
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

bool
MetalGPUBVHBuilder::getShadowLinkingLights(const geom::internal::NamedPrimitive& np,
                                           MetalGPUBuffer<ShadowLinkLight>& lightsBuf) const
{
    std::vector<ShadowLinkLight> lights;

    for (const auto& element : np.getShadowLinkings()) {
        int casterId = element.first;
        const geom::internal::ShadowLinking *linking = element.second;

        if (linking == nullptr) {
            // skip any empty
            continue;
        }

        for (const auto& light : linking->getLights()) {
            ShadowLinkLight lgt;
            lgt.mCasterId = casterId;
            lgt.mLightId = reinterpret_cast<intptr_t>(light);
            lights.push_back(lgt);
        }
    }
    if (lightsBuf.allocAndUpload(lights) != cudaSuccess) {
        return false;
    }
    return true;
}

bool
MetalGPUBVHBuilder::getShadowLinkingReceivers(const geom::internal::NamedPrimitive& np,
                                              MetalGPUBuffer<ShadowLinkReceiver>& receiversBuf) const
{
    std::vector<ShadowLinkReceiver> receivers;

    for (const auto& element : np.getShadowLinkings()) {
        int casterId = element.first;
        const geom::internal::ShadowLinking *linking = element.second;

        if (linking == nullptr) {
            // skip any empty
            continue;
        }

        for (const auto& receiverId : linking->getReceivers()) {
            ShadowLinkReceiver rcv;
            rcv.mCasterId = casterId;
            rcv.mReceiverId = receiverId;
            // the same for all entries for this caster
            rcv.mIsComplemented = linking->getIsComplemented();
            receivers.push_back(rcv);
        }
    }
    if (receiversBuf.allocAndUpload(receivers) != cudaSuccess) {
        return false;
    }
    return true;
}



MetalGPUAccelerator::MetalGPUAccelerator(bool allowUnsupportedFeatures,
                                        const uint32_t numCPUThreads,
                                       const scene_rdl2::rdl2::Layer *layer,
                                       const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                                       const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                                       std::vector<std::string>& warningMsgs,
                                       std::string* errorMsg) :
    mAllowUnsupportedFeatures {allowUnsupportedFeatures},
    mContext(createMetalContext()),
    mModule {nullptr},
    mRootGroup {nullptr},
    mHitGroupRecordBuf(mContext),
    mHasMotionBlur(GPU_MOTION_NONE),
    mDebugBuf(mContext)
{
    // The constructor fully initializes the GPU.  We are ready to trace rays afterwards.

    scene_rdl2::logging::Logger::info("GPU: Creating accelerator");

    if (!validateMetalContext(mContext,
                              &mGPUDeviceName,
                              errorMsg)) {
        return;
    }

    mEncoderStates.resize(numCPUThreads);

    const std::string moonrayRootPath = scene_rdl2::util::getenv<std::string>("REZ_MOONRAY_ROOT");
    const std::string metalLibPath = moonrayRootPath + "/shaders/default.metallib";
    scene_rdl2::logging::Logger::info("GPU: Loading .metallib: ", metalLibPath);

    if (!createMetalLibrary(mContext,
                            metalLibPath,
                            //pipelineCompileOptions,
                            &mModule,
                            errorMsg)) {
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating intersection functions");

    if (!createIntersectionFunctions(errorMsg)) {
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating pipeline");

    const char* groupNames[] = {
        "__raygen__base",
        "__raygen__im",
        "__raygen__pm",
        "__raygen__im_pm",
    };
    static_assert(sizeof(groupNames) / sizeof(groupNames[0]) == PSO_NUM_GROUPS);
    for (int group = 0; group < PSO_NUM_GROUPS; group++) {
        PSOGroup &pso = mPSOs[group];
        if (!createMetalRaygenKernel(mContext,
                                     mModule,
                                     pso.linkedFunctions,
                                     groupNames[group],
                                     &pso.rayGenPSO,
                                     errorMsg)) {
            return;
        }

        if (!createIntersectionFunctionTables(pso.linkedFunctions,
                                              pso.rayGenPSO,
                                              &pso.intersectFuncTable,
                                              errorMsg)) {
            return;
        }
    }

    scene_rdl2::logging::Logger::info("GPU: Creating traversables");

    std::string buildErrorMsg;
    if (!build(layer, geometrySets, g2s, warningMsgs, &buildErrorMsg)) {
        *errorMsg = "GPU: Accel creation failed: " + buildErrorMsg;
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating shader binding table");

    if (!createShaderBindingTable(&buildErrorMsg)) {
        *errorMsg = "GPU: Shader binding table creation failed: " + buildErrorMsg;
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Allocating ray, occlusion and params buffers");

    mRaysBuf.reserve(numCPUThreads);
    mIsOccludedBuf.reserve(numCPUThreads);
    mParamsBuf.reserve(numCPUThreads);
    for (int i = 0; i < numCPUThreads; i++) {
        mRaysBuf.push_back(MetalGPUBuffer<GPURay>(mContext));
        if (mRaysBuf[i].alloc(mRaysBufSize) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating rays buffer";
            return;
        }
        [mRaysBuf[i].deviceptr() setLabel:[NSString stringWithFormat:@"GPU Occl Ray [%d]", i]];
        mIsOccludedBuf.push_back(MetalGPUBuffer<unsigned char>(mContext));
        if (mIsOccludedBuf[i].alloc(mRaysBufSize) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating occlusion buffer";
            return;
        }
        [mIsOccludedBuf[i].deviceptr() setLabel:[NSString stringWithFormat:@"Is it Occluded?  [%d]", i]];
        mParamsBuf.push_back(MetalGPUBuffer<MetalGPUParams>(mContext));
        if (mParamsBuf[i].alloc(1) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating params buffer";
            return;
        }
        [mParamsBuf[i].deviceptr() setLabel:[NSString stringWithFormat:@"GPU Params [%d]", i]];
    }

    scene_rdl2::logging::Logger::info("GPU: Setup complete");
}

MetalGPUAccelerator::~MetalGPUAccelerator()
{
    scene_rdl2::logging::Logger::info("GPU: Freeing accelerator");

    for (auto& encoderState : mEncoderStates) {
        if (encoderState.cpuBuffer) {
            [encoderState.cpuBuffer release];
            encoderState.cpuBuffer = nil;
        }
        if (encoderState.encoder) {
            [encoderState.encoder endEncoding];
            encoderState.encoder = nil;
        }
        if (encoderState.mQueue) {
            [encoderState.mQueue release];
            encoderState.mQueue = nil;
        }
    }
    mEncoderStates.clear();

    for (const auto& groupEntry : mSharedGroups) {
        delete groupEntry.second;
    }
    delete mRootGroup;

    if (mModule != nullptr) {
        [mModule release];
        mModule = nil;
    }

    if (mContext != nullptr) {
        [mContext release];
        mContext = nil;
    }
}

std::string
MetalGPUAccelerator::getGPUDeviceName() const
{
    return mGPUDeviceName;
}

bool
buildGPUBVHBottomUp(bool allowUnsupportedFeatures,
                    id<MTLDevice> context,
                    const scene_rdl2::rdl2::Layer* layer,
                    scene_rdl2::rdl2::Geometry* geometry,
                    MetalGPUPrimitiveGroup* rootGroup,
                    SharedGroupMap& groups,
                    std::unordered_set<scene_rdl2::rdl2::Geometry*>& visitedGeometry,
                    std::vector<std::string>& warningMsgs,
                    std::string* errorMsg)
{

    // Extremely similar to EmbreeAccelerator.cc buildBVHBottomUp().  This is intentional so
    // it behaves the same and is easy to maintain and debug.

    geom::Procedural* procedural = geometry->getProcedural();
    // All parts in a procedural are unassigned in the layer
    if (!procedural) {
        return true;
    }
    // This BVH part for this particular rdl geometry has been constructed
    if (visitedGeometry.find(geometry) != visitedGeometry.end()) {
        return true;
    }
    // Do a bottom up traversal so that referenced BVH is built first
    const scene_rdl2::rdl2::SceneObjectVector& references =
        geometry->get(scene_rdl2::rdl2::Geometry::sReferenceGeometries);

    for (const auto& ref : references) {
        if (!ref->isA<scene_rdl2::rdl2::Geometry>()) {
            continue;
        }
        scene_rdl2::rdl2::Geometry* referencedGeometry = ref->asA<scene_rdl2::rdl2::Geometry>();
        buildGPUBVHBottomUp(allowUnsupportedFeatures,
                            context,
                            layer,
                            referencedGeometry,
                            rootGroup,
                            groups,
                            visitedGeometry,
                            warningMsgs,
                            errorMsg);
    }

    // We disable parallelism here to solve the non-deterministic
    // issue for some hair/fur related scenes.
    // Also, testing shows no speed gain is achieved when this for() loop is parallelized.
    bool doParallel = false;
    if (procedural->isReference()) {
        const std::shared_ptr<geom::SharedPrimitive>& ref =
            procedural->getReference();
        if (groups.insert(std::make_pair(ref, nullptr)).second) {
            MetalGPUPrimitiveGroup *group = new MetalGPUPrimitiveGroup(context);
            MetalGPUBVHBuilder builder(context, allowUnsupportedFeatures, layer, geometry, group, groups);
            ref->getPrimitive()->accept(builder);
            // mark the BVH representation of referenced primitive (group)
            // has been correctly constructed so that all the instances
            // reference it can start accessing it
            groups[ref] = group;
        }
    } else {
        MetalGPUPrimitiveGroup *aGroup = rootGroup;

        MetalGPUBVHBuilder geomBuilder(context,
                                  allowUnsupportedFeatures,
                                  layer,
                                  geometry,
                                  aGroup,
                                  groups);

        unsigned int aCount = procedural->getPrimitivesCount();

        procedural->forEachPrimitive(geomBuilder, doParallel);

        warningMsgs.insert(warningMsgs.end(),
                           geomBuilder.warningMsgs().begin(),
                           geomBuilder.warningMsgs().end());

        if (geomBuilder.hasFailed()) {
            *errorMsg = geomBuilder.whyFailed();
            return false;
        }
    }
    visitedGeometry.insert(geometry);

    return true;
}

bool
MetalGPUAccelerator::build(const scene_rdl2::rdl2::Layer *layer,
                          const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                          const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                          std::vector<std::string>& warningMsgs,
                          std::string* errorMsg)
{
    // See embree EmbreeAccelerator::build()

    mRootGroup = new MetalGPUPrimitiveGroup(mContext);

    std::unordered_set<scene_rdl2::rdl2::Geometry*> visitedGeometry;
    for (const auto& geometrySet : geometrySets) {
        const scene_rdl2::rdl2::SceneObjectIndexable& geometries = geometrySet->getGeometries();
        for (auto& sceneObject : geometries) {
            scene_rdl2::rdl2::Geometry* geometry = sceneObject->asA<scene_rdl2::rdl2::Geometry>();
            if (g2s != nullptr && g2s->find(geometry) == g2s->end()) {
                continue;
            }
            if (!buildGPUBVHBottomUp(mAllowUnsupportedFeatures,
                                     mContext,
                                     layer,
                                     geometry,
                                     mRootGroup,
                                     mSharedGroups,
                                     visitedGeometry,
                                     warningMsgs,
                                     errorMsg)) {
                return false;
            }
        }
    }

    mHasMotionBlur = GPU_MOTION_NONE;
    unsigned int baseID = 0;
    mRootGroup->setAccelStructBaseID(baseID);
    mRootGroup->hasMotionBlur(&mHasMotionBlur);

    for (auto& groupEntry : mSharedGroups) {
        MetalGPUPrimitiveGroup *group = groupEntry.second;
        group->setAccelStructBaseID(baseID);
        group->hasMotionBlur(&mHasMotionBlur);
    }


    // now build the root group and referenced/instanced groups
    id<MTLCommandQueue> queue = [mContext newCommandQueue];
    if (!mRootGroup->build(mContext, queue, mHasMotionBlur, &mBottomLevelAS, errorMsg)) {
        [queue release];
        return false;
    }
    [queue release];

    return true;
}

bool
MetalGPUAccelerator::createIntersectionFunctions(std::string* errorMsg)
{
    const char* functionNames[] = {
        "__intersection__triangle",
        "__intersection__box",
        "__intersection__points",
        "__intersection__sphere"
    };
    const char* group_ext[] = {
        "",
        "_im",
        "_pm",
        "_im_pm"
    };
    const int numFunctions = sizeof(functionNames) / sizeof(functionNames[0]);

    static_assert(sizeof(group_ext) / sizeof(group_ext[0]) == PSO_NUM_GROUPS);

    // See MetalGPUPrograms.metal for the implementations of the programs.
    MTLFunctionDescriptor *desc = [MTLIntersectionFunctionDescriptor functionDescriptor];
    for (int group = 0; group < PSO_NUM_GROUPS; group++) {
        mPSOs[group].linkedFunctions = [NSMutableArray arrayWithCapacity:numFunctions];

        for (int i = 0; i < numFunctions; i++) {
            std::string full_name = std::string(functionNames[i]) + group_ext[group];
            const char *name = full_name.c_str();
            NSError *error = NULL;

            desc.name = [@(name) copy];
            id<MTLFunction> function = [mModule newFunctionWithDescriptor:desc error:&error];

            if (function == nil) {
                NSString *err = [error localizedDescription];
                std::string errors = [err UTF8String];

                *errorMsg = "Error fetching intersection function \"" + std::string(name) + ": " + errors;
                return false;
            }

            function.label = [@(name) copy];
            [mPSOs[group].linkedFunctions addObject:function];
        }
    }
    return true;
}

bool
MetalGPUAccelerator::createShaderBindingTable(std::string* errorMsg)
{
    {
        std::vector<HitGroupRecord> hitgroupRecs;
        mRootGroup->getSBTRecords(hitgroupRecs, mUsedIndirectResources);
        for (auto& groupEntry : mSharedGroups) {
            MetalGPUPrimitiveGroup *group = groupEntry.second;
            group->getSBTRecords(hitgroupRecs, mUsedIndirectResources);
        }

        mUsedIndirectResources.erase(
                    std::remove_if(begin(mUsedIndirectResources), end(mUsedIndirectResources),
                                   [](id<MTLBuffer> b) { return b == nil; }),
                    end(mUsedIndirectResources));

        // Upload the hitgroup records to the GPU
        if (mHitGroupRecordBuf.allocAndUpload(hitgroupRecs) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating HitGroup SBT record buffer";
            return false;
        }
        
        [mHitGroupRecordBuf.deviceptr() setLabel:@"HitGroupRec Buffer"];
    }

    return true;
}

void
MetalGPUAccelerator::prepareEncoder(const uint32_t queueIdx) const
{
    MNRY_ASSERT_REQUIRE(queueIdx < mEncoderStates.size());

    int psoIdx = mHasMotionBlur;

    if (!mEncoderStates[queueIdx].mQueue) {
        mEncoderStates[queueIdx].mQueue = [mContext newCommandQueue];
    }
    id<MTLCommandBuffer> commandBuffer = [mEncoderStates[queueIdx].mQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

    // Setup the parameters
    [encoder setComputePipelineState:mPSOs[psoIdx].rayGenPSO];
    [encoder setBuffer:mParamsBuf[queueIdx].deviceptr()
                offset:0
               atIndex:0];
    [encoder setAccelerationStructure:mRootGroup->mTopLevelIAS
                        atBufferIndex:1];
    [encoder setIntersectionFunctionTable:mPSOs[psoIdx].intersectFuncTable
                            atBufferIndex:2];
    [encoder setBuffer:mHitGroupRecordBuf.deviceptr()
                offset:0
               atIndex:3];
    [encoder setBuffer:mDebugBuf.deviceptr()
                offset:0
               atIndex:4];

    // Buffer bindings to intersect functions
    [mPSOs[psoIdx].intersectFuncTable setBuffer:mHitGroupRecordBuf.deviceptr()
                                         offset:0
                                        atIndex:0];

    // Mark all the resources that are used
    [encoder useResource:mRaysBuf[queueIdx].deviceptr()
                   usage:MTLResourceUsageRead];
    [encoder useResource:mIsOccludedBuf[queueIdx].deviceptr()
                   usage:MTLResourceUsageWrite];
    [encoder useResource:mEncoderStates[queueIdx].cpuBuffer
                   usage:MTLResourceUsageRead];
    [encoder useResources:mBottomLevelAS.data()
                    count:mBottomLevelAS.size()
                    usage:MTLResourceUsageRead];
    [encoder useResources:mUsedIndirectResources.data()
                    count:mUsedIndirectResources.size()
                    usage:MTLResourceUsageRead];

    mEncoderStates[queueIdx].commandBuffer = commandBuffer;
    mEncoderStates[queueIdx].encoder = encoder;
}

void*
MetalGPUAccelerator::getBundledOcclRaysBufUMA(const uint32_t queueIdx,
                                              const uint32_t numRays,
                                              const size_t stride) const
{
    MNRY_ASSERT_REQUIRE(queueIdx < mEncoderStates.size());

    if (!mEncoderStates[queueIdx].cpuBuffer) {
        mEncoderStates[queueIdx].cpuBuffer =
            [mContext newBufferWithLength:numRays * stride
                                  options:MTLResourceStorageModeShared];
    }
    
    [mEncoderStates[queueIdx].cpuBuffer setLabel:[NSString stringWithFormat:@"CPU Occl Ray  [%u]", queueIdx]];
    
    return [mEncoderStates[queueIdx].cpuBuffer contents];
}

void
MetalGPUAccelerator::intersect(const uint32_t queueIdx,
                               const uint32_t numRays,
                               const GPURay* rays) const
{
    // not yet implemented
}

void
MetalGPUAccelerator::occluded(const uint32_t queueIdx,
                              const uint32_t numRays,
                              const GPURay* rays,
                              const void* cpuRays,
                              const size_t cpuRayStride) const
{
    // std::cout << "occluded(): " << numRays << std::endl;

    MNRY_ASSERT_REQUIRE(queueIdx < mEncoderStates.size());
    MNRY_ASSERT_REQUIRE(numRays <= mRaysBufSize);
    // Ensure getBundledOcclRaysBufUMA was called to allocate the cpuRays pointer for this queue
    MNRY_ASSERT_REQUIRE(cpuRays == [mEncoderStates[queueIdx].cpuBuffer contents]);

    // Setup the global GPU parameters
    MetalGPUParams &params(*mParamsBuf[queueIdx].cpu_ptr());
    params.mCPURays = [mEncoderStates[queueIdx].cpuBuffer gpuAddress];
    params.mCPURayStride = cpuRayStride;
    params.mNumRays = numRays;
    params.mRaysBuf = mRaysBuf[queueIdx].ptr();
    params.mIsOccludedBuf = mIsOccludedBuf[queueIdx].ptr();

    if (!mEncoderStates[queueIdx].encoder) {
        prepareEncoder(queueIdx);
    }

    // Dispatch the kernel
    const int executionWidth = 128;
    MTLSize threadsPerThreadgroup = MTLSizeMake(executionWidth, 1, 1);
    MTLSize threadgroupsPerDispatch = MTLSizeMake((numRays + executionWidth - 1) / executionWidth, 1, 1);
    [mEncoderStates[queueIdx].encoder dispatchThreadgroups:threadgroupsPerDispatch
                                    threadsPerThreadgroup:threadsPerThreadgroup];

    // Commit the GPU work
    id<MTLCommandBuffer> commandBuffer = mEncoderStates[queueIdx].commandBuffer;
    [mEncoderStates[queueIdx].encoder endEncoding];
    [commandBuffer commit];

    // Before waiting for the GPU to complete, use the time to set up the next
    // encoder
    prepareEncoder(queueIdx);

    // Now block until the GPU is done
    [commandBuffer waitUntilCompleted];
}

void*
MetalGPUAccelerator::instanceIdToInstancePtr(unsigned int /*instanceId*/) const
{
    // not yet implemented
    return nullptr;
}

} // namespace rt
} // namespace moonray

