// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "OptixGPUAccelerator.h"
#include "Math2OptixGPUMath.h"

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


class OptixGPUBVHBuilder : public geom::PrimitiveVisitor
{
public:
    // Extremely similar to EmbreeAccelerator.cc BVHBuilder.  This is intentional so
    // it behaves the same and is easy to maintain and debug.

    OptixGPUBVHBuilder(bool allowUnsupportedFeatures,
                       const scene_rdl2::rdl2::Layer* layer,
                       const scene_rdl2::rdl2::Geometry* geometry,
                       OptixGPUPrimitiveGroup* parentGroup,
                       SharedGroupMap& groups) :
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

        switch (pCurves->getSubType()) {
        case geom::internal::Curves::SubType::RAY_FACING:
            createCurves(*pCurves, c.getCurvesType(), c.getTessellationRate());
        break;
        case geom::internal::Curves::SubType::ROUND:
            createRoundCurves(*pCurves, c.getCurvesType());
        break;
        case geom::internal::Curves::SubType::NORMAL_ORIENTED:
            if (mAllowUnsupportedFeatures) {
                logWarningMsg(
                    "Normal-oriented curves are not supported in XPU mode.  "
                    "Using ray-facing curves instead.");
                createCurves(*pCurves, c.getCurvesType(), c.getTessellationRate());
            } else {
                mFailed = true;
                mWhyFailed = "Normal-oriented curves are not supported in XPU mode";
                return;
            }
        break;
        }
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
            OptixGPUPrimitiveGroup *group = new OptixGPUPrimitiveGroup();
            OptixGPUBVHBuilder builder(mAllowUnsupportedFeatures, mLayer, mGeometry, group, mSharedGroups);
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
                           const geom::Curves::Type curvesType);

    void createPoints(const geom::internal::Points& geomPoints);

    void createPolyMesh(const geom::internal::Mesh& geomMesh);

    void createSphere(const geom::internal::Sphere& geomSphere);

    void createInstance(const geom::internal::Instance& instance,
                        OptixGPUPrimitiveGroup* group);

    unsigned int resolveVisibilityMask(const geom::internal::NamedPrimitive& np) const;

    bool getShadowLinkingLights(const geom::internal::NamedPrimitive& np,
                                OptixGPUBuffer<ShadowLinkLight>& lightsBuf) const;

    bool getShadowLinkingReceivers(const geom::internal::NamedPrimitive& np,
                                   OptixGPUBuffer<ShadowLinkReceiver>& receiversBuf) const;

    bool mAllowUnsupportedFeatures;
    std::vector<std::string> mWarningMsgs;
    bool mFailed;
    std::string mWhyFailed;
    const scene_rdl2::rdl2::Layer* mLayer;
    const scene_rdl2::rdl2::Geometry* mGeometry;
    OptixGPUPrimitiveGroup* mParentGroup;
    SharedGroupMap& mSharedGroups;
};

void
OptixGPUBVHBuilder::logWarningMsg(const std::string& msg)
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
OptixGPUBVHBuilder::createBox(const geom::internal::Box& geomBox)
{
    OptixGPUBox* gpuBox = new OptixGPUBox();
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

    gpuBox->mL2P = mat43ToOptixGPUXform(geomBox.getL2P());
    gpuBox->mP2L = mat43ToOptixGPUXform(geomBox.getP2L());
    gpuBox->mLength = geomBox.getLength();
    gpuBox->mWidth = geomBox.getWidth();
    gpuBox->mHeight = geomBox.getHeight();
}

void
OptixGPUBVHBuilder::createCurves(const geom::internal::Curves& geomCurves,
                            const geom::Curves::Type curvesType,
                            const int tessellationRate)
{
    geom::internal::Curves::Spans spans;
    geomCurves.getTessellatedSpans(spans);

    OptixGPUCurve* gpuCurve = new OptixGPUCurve();
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

    if (gpuCurve->mIndices.allocAndUpload(gpuCurve->mHostIndices) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve indices to the GPU";
        return;
    }

    if (gpuCurve->mControlPoints.allocAndUpload(gpuCurve->mHostControlPoints)) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve control points to the GPU";
        return;
    }

    gpuCurve->mSegmentsPerCurve = tessellationRate;  // for bezier and bspline, embree default
}

void
OptixGPUBVHBuilder::createRoundCurves(const geom::internal::Curves& geomCurves,
                                      const geom::Curves::Type curvesType)
{
    if (curvesType == geom::Curves::Type::BEZIER) {
        // todo: this is supported in Optix 7.7
        if (mAllowUnsupportedFeatures) {
            logWarningMsg(
                "Round bezier curves are currently unsupported in XPU mode.  "
                "Ignoring geometry.");
            return;
        } else {
            mFailed = true;
            mWhyFailed = "Round bezier curves are currently unsupported in XPU mode";
            return;
        }
    }

    // This code assumes that there is a max of 2 motion samples.
    // TODO: add support for round curves with more motion samples.
    bool motionBlur = geomCurves.getMotionSamplesCount() > 1;
    if (geomCurves.getMotionSamplesCount() > 2) {
        if (mAllowUnsupportedFeatures) {
            logWarningMsg(
                "Round curves with more than 2 motion samples are currently unsupported in XPU mode.  "
                "Only using first 2 samples.");
        } else {
            mFailed = true;
            mWhyFailed = "Round curves with more than 2 motion samples are currently unsupported in XPU mode";
            return;
        }
    }

    geom::internal::Curves::Spans spans;
    geomCurves.getTessellatedSpans(spans);

    OptixGPURoundCurves* gpuCurve = new OptixGPURoundCurves();
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
    if (gpuCurve->mMotionSamplesCount > 2) {
        gpuCurve->mMotionSamplesCount = 2;
    }

    switch (curvesType) {
    case geom::Curves::Type::LINEAR:
        gpuCurve->mType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
    break;
    case geom::Curves::Type::BSPLINE:
        gpuCurve->mType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
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

    if (gpuCurve->mIndices.allocAndUpload(hostIndices) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve indices to the GPU";
        return;
    }

    if (gpuCurve->mVertices.allocAndUpload(hostVertices) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve vertices to the GPU";
        return;
    }

    if (gpuCurve->mWidths.allocAndUpload(hostWidths) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the curve widths to the GPU";
        return;
    }

    // Get the vertex/width buffer pointers.  For curves without motion blur, the
    // second pointer is null.
    gpuCurve->mVerticesPtrs[0] = gpuCurve->mVertices.deviceptr();
    gpuCurve->mWidthsPtrs[0] = gpuCurve->mWidths.deviceptr();
    if (gpuCurve->mMotionSamplesCount == 2) {
        gpuCurve->mVerticesPtrs[1] = gpuCurve->mVertices.deviceptr() +
                                     gpuCurve->mNumControlPoints * sizeof(float3);
        gpuCurve->mWidthsPtrs[1] = gpuCurve->mWidths.deviceptr() +
                                   gpuCurve->mNumControlPoints * sizeof(float);
    } else {
        gpuCurve->mVerticesPtrs[1] = 0;
        gpuCurve->mWidthsPtrs[1] = 0;
    }
}

void
OptixGPUBVHBuilder::createPoints(const geom::internal::Points& geomPoints)
{
    OptixGPUPoints* gpuPoints = new OptixGPUPoints();
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

    if (gpuPoints->mPoints.allocAndUpload(gpuPoints->mHostPoints) != cudaSuccess) {
        mFailed = true;
        mWhyFailed = "There was a problem uploading the points to the GPU";
        return;
    }
}

void
OptixGPUBVHBuilder::createPolyMesh(const geom::internal::Mesh& geomMesh)
{
    geom::internal::Mesh::TessellatedMesh mesh;
    geomMesh.getTessellatedMesh(mesh);

    // This code assumes that there is a max of 2 motion samples
    // TODO: add support for mesh with more motion samples.
    size_t mbSamples = mesh.mVertexBufferDesc.size();
    const bool enableMotionBlur = mbSamples > 1;
    if (mbSamples > 2) {
      if (mAllowUnsupportedFeatures) {
            logWarningMsg(
                "Meshes with more than 2 motion samples are currently unsupported in XPU mode.  "
                "Only using first 2 samples.");
            mbSamples = 2;
        } else {
            mFailed = true;
            mWhyFailed = "Meshes with more than 2 motion samples are currently unsupported in XPU mode";
            return;
        }
    }

    OptixGPUTriMesh* gpuMesh = new OptixGPUTriMesh();
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
            }
        }

        if (gpuMesh->mVertices.allocAndUpload(gpuVertices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the vertices to the GPU";
            return;
        }
        if (gpuMesh->mIndices.allocAndUpload(gpuIndices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the indices to the GPU";
            return;
        }
        if (gpuMesh->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
            return;
        }

        gpuMesh->mVerticesPtrs[0] = gpuMesh->mVertices.deviceptr();
        gpuMesh->mVerticesPtrs[1] = gpuMesh->mVertices.deviceptr() + motionKeyOffsets[1] * sizeof(float3);

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
            }
        }

        if (gpuMesh->mVertices.allocAndUpload(gpuVertices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the vertices to the GPU";
            return;
        }
        if (gpuMesh->mIndices.allocAndUpload(gpuIndices) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the indices to the GPU";
            return;
        }
        if (gpuMesh->mAssignmentIds.allocAndUpload(assignmentIds) != cudaSuccess) {
            mFailed = true;
            mWhyFailed = "There was a problem uploading the assignment IDs to the GPU";
            return;
        }

        gpuMesh->mVerticesPtrs[0] = gpuMesh->mVertices.deviceptr();
        gpuMesh->mVerticesPtrs[1] = gpuMesh->mVertices.deviceptr() + motionKeyOffsets[1] * sizeof(float3);
    }
}

void
OptixGPUBVHBuilder::createSphere(const geom::internal::Sphere& geomSphere)
{
    OptixGPUSphere* gpuSphere = new OptixGPUSphere();
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

    gpuSphere->mL2P = mat43ToOptixGPUXform(geomSphere.getL2P());
    gpuSphere->mP2L = mat43ToOptixGPUXform(geomSphere.getP2L());
    gpuSphere->mRadius = geomSphere.getRadius();
    gpuSphere->mPhiMax = geomSphere.getPhiMax();
    gpuSphere->mZMin = geomSphere.getZMin();
    gpuSphere->mZMax = geomSphere.getZMax();
}

void
OptixGPUBVHBuilder::createInstance(const geom::internal::Instance& instance,
                              OptixGPUPrimitiveGroup* group)
{
    const geom::internal::MotionTransform& l2pXform = instance.getLocal2Parent();

    scene_rdl2::math::Xform3f xforms[OptixGPUInstance::sNumMotionKeys];
    bool hasMotionBlur;
    if (!l2pXform.isMotion()) {
        xforms[0] = l2pXform.getStaticXform();
        hasMotionBlur = false;
    } else {
        // eval() the l2pXform at sNumMotionKeys discrete timesteps between time [0, 1].
        // Optix can't slerp() between matrices so we need to generate smaller
        // timesteps it can lerp() between.
        for (int i = 0; i < OptixGPUInstance::sNumMotionKeys; i++) {
            float t = i / static_cast<float>(OptixGPUInstance::sNumMotionKeys - 1);
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

    OptixGPUInstance* gpuInstance = new OptixGPUInstance();
    mParentGroup->mInstances.push_back(gpuInstance);
    gpuInstance->mGroup = group;

    if (hasMotionBlur) {
        for (int i = 0; i < OptixGPUInstance::sNumMotionKeys; i++) {
            gpuInstance->mXforms[i] = mat43ToOptixGPUXform(xforms[i]);
        }
    } else {
        gpuInstance->mXforms[0] = mat43ToOptixGPUXform(xforms[0]);
    }

    gpuInstance->mHasMotionBlur = hasMotionBlur;
}

unsigned int
OptixGPUBVHBuilder::resolveVisibilityMask(const geom::internal::NamedPrimitive& np) const
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
OptixGPUBVHBuilder::getShadowLinkingLights(const geom::internal::NamedPrimitive& np,
                                           OptixGPUBuffer<ShadowLinkLight>& lightsBuf) const
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
OptixGPUBVHBuilder::getShadowLinkingReceivers(const geom::internal::NamedPrimitive& np,
                                              OptixGPUBuffer<ShadowLinkReceiver>& receiversBuf) const
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


OptixGPUAccelerator::OptixGPUAccelerator(bool allowUnsupportedFeatures,
                                         const scene_rdl2::rdl2::Layer *layer,
                                         const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                                         const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                                         std::vector<std::string>& warningMsgs,
                                         std::string* errorMsg) :
    mAllowUnsupportedFeatures {allowUnsupportedFeatures},
    mContext {nullptr},
    mModule {nullptr},
    mRoundLinearCurvesModule {nullptr},
    mRoundLinearCurvesMBModule {nullptr},
    mRoundCubicBsplineCurvesModule {nullptr},
    mRoundCubicBsplineCurvesMBModule {nullptr},
    mPipeline {nullptr},
    mRootGroup {nullptr},
    mOutputOcclusionBuf {nullptr}
{
    // The constructor fully initializes the GPU.  We are ready to trace rays afterwards.

    scene_rdl2::logging::Logger::info("GPU: Creating accelerator");

    if (!createOptixContext(optixMessageCallback,
                            &mCudaStream,
                            &mContext,
                            &mGPUDeviceName,
                            errorMsg)) {
        return;
    }

    // All modules in a pipeline need to use the same values for the pipeline compile options.
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = true;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    //pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = static_cast<unsigned int>(
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE);

    const std::string moonrayRootPath = scene_rdl2::util::getenv<std::string>("REZ_MOONRAY_ROOT");
    const std::string ptxPath = moonrayRootPath + "/shaders/OptixGPUPrograms.ptx";
    scene_rdl2::logging::Logger::info("GPU: Loading .ptx: ", ptxPath);

    OptixModuleCompileOptions moduleCompileOptions = {
        0, // maxRegisterCount
        //OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        //OPTIX_COMPILE_DEBUG_LEVEL_FULL},
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE
    };

    if (!createOptixModule(mContext,
                           ptxPath,
                           moduleCompileOptions,
                           pipelineCompileOptions,
                           &mModule,
                           errorMsg)) {
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Loading built in modules");

    if (!getBuiltinISModule(mContext,
                            moduleCompileOptions,
                            pipelineCompileOptions,
                            OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
                            false,
                            &mRoundLinearCurvesModule,
                            errorMsg)) {
        return;
    }

    if (!getBuiltinISModule(mContext,
                            moduleCompileOptions,
                            pipelineCompileOptions,
                            OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
                            true,
                            &mRoundLinearCurvesMBModule,
                            errorMsg)) {
        return;
    }

    if (!getBuiltinISModule(mContext,
                            moduleCompileOptions,
                            pipelineCompileOptions,
                            OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
                            false,
                            &mRoundCubicBsplineCurvesModule,
                            errorMsg)) {
        return;
    }

    if (!getBuiltinISModule(mContext,
                            moduleCompileOptions,
                            pipelineCompileOptions,
                            OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
                            true,
                            &mRoundCubicBsplineCurvesMBModule,
                            errorMsg)) {
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating program groups");

    if (!createProgramGroups(errorMsg)) {
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating pipeline");

    if (!createOptixPipeline(mContext,
                             pipelineCompileOptions,
                             { 1, // maxTraceDepth
                               OPTIX_COMPILE_DEBUG_LEVEL_NONE },
                             mProgramGroups,
                             &mPipeline,
                             errorMsg)) {
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating traversables");

    std::string buildErrorMsg;
    if (!build(mCudaStream, mContext, layer, geometrySets, g2s, warningMsgs, &buildErrorMsg)) {
        *errorMsg = "GPU: Accel creation failed: " + buildErrorMsg;
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Creating shader binding table");

    if (!createShaderBindingTable(&buildErrorMsg)) {
        *errorMsg = "GPU: Shader binding table creation failed: " + buildErrorMsg;
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Allocating rays buffer");

    if (mRaysBuf.alloc(mRaysBufSize) != cudaSuccess) {
        *errorMsg = "GPU: Error allocating rays buffer";
        return;
    }

    if (cudaMallocHost(&mOutputOcclusionBuf, sizeof(unsigned char) * mRaysBufSize) != cudaSuccess) {
        *errorMsg = "GPU: Error allocating output occlusion buffer";
        return;
    }

    if (mIsOccludedBuf.alloc(mRaysBufSize) != cudaSuccess) {
        *errorMsg = "GPU: Error allocating occlusion buffer";
        return;
    }

    if (mParamsBuf.alloc(1) != cudaSuccess) {
        *errorMsg = "GPU: Error allocating params buffer";
        return;
    }

    scene_rdl2::logging::Logger::info("GPU: Setup complete");
}

OptixGPUAccelerator::~OptixGPUAccelerator()
{
    scene_rdl2::logging::Logger::info("GPU: Freeing accelerator");

    for (const auto& groupEntry : mSharedGroups) {
        delete groupEntry.second;
    }
    delete mRootGroup;

    // delete in the opposite order of creation
    if (mPipeline != nullptr) {
        optixPipelineDestroy(mPipeline);
    }

    for (const auto& pgEntry : mProgramGroups) {
        OptixProgramGroup pg = pgEntry.second;
        optixProgramGroupDestroy(pg);
    }
    mProgramGroups.clear();

    if (mModule != nullptr) {
        optixModuleDestroy(mModule);
    }

    if (mRoundLinearCurvesModule != nullptr) {
        optixModuleDestroy(mRoundLinearCurvesModule);
    }

    if (mRoundLinearCurvesMBModule != nullptr) {
        optixModuleDestroy(mRoundLinearCurvesMBModule);
    }

    if (mRoundCubicBsplineCurvesModule != nullptr) {
        optixModuleDestroy(mRoundCubicBsplineCurvesModule);
    }

    if (mRoundCubicBsplineCurvesMBModule != nullptr) {
        optixModuleDestroy(mRoundCubicBsplineCurvesMBModule);
    }

    if (mContext != nullptr) {
        optixDeviceContextDestroy(mContext);
    }

    cudaFreeHost(mOutputOcclusionBuf);
}

std::string
OptixGPUAccelerator::getGPUDeviceName() const
{
    return mGPUDeviceName;
}

bool
buildGPUBVHBottomUp(bool allowUnsupportedFeatures,
                    const scene_rdl2::rdl2::Layer* layer,
                    scene_rdl2::rdl2::Geometry* geometry,
                    OptixGPUPrimitiveGroup* rootGroup,
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
            OptixGPUPrimitiveGroup *group = new OptixGPUPrimitiveGroup();
            OptixGPUBVHBuilder builder(allowUnsupportedFeatures, layer, geometry, group, groups);
            ref->getPrimitive()->accept(builder);
            // mark the BVH representation of referenced primitive (group)
            // has been correctly constructed so that all the instances
            // reference it can start accessing it
            groups[ref] = group;
        }
    } else {
        OptixGPUBVHBuilder geomBuilder(allowUnsupportedFeatures,
                                       layer,
                                       geometry,
                                       rootGroup,
                                       groups);
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
OptixGPUAccelerator::build(CUstream cudaStream,
                           OptixDeviceContext context,
                           const scene_rdl2::rdl2::Layer *layer,
                           const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                           const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                           std::vector<std::string>& warningMsgs,
                           std::string* errorMsg)
{
    // See embree EmbreeAccelerator::build()

    mRootGroup = new OptixGPUPrimitiveGroup();

    std::unordered_set<scene_rdl2::rdl2::Geometry*> visitedGeometry;
    for (const auto& geometrySet : geometrySets) {
        const scene_rdl2::rdl2::SceneObjectIndexable& geometries = geometrySet->getGeometries();
        for (auto& sceneObject : geometries) {
            scene_rdl2::rdl2::Geometry* geometry = sceneObject->asA<scene_rdl2::rdl2::Geometry>();
            if (g2s != nullptr && g2s->find(geometry) == g2s->end()) {
                continue;
            }
            if (!buildGPUBVHBottomUp(mAllowUnsupportedFeatures,
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

    unsigned int sbtOffset = 0;
    mRootGroup->setSBTOffset(sbtOffset);
    for (auto& groupEntry : mSharedGroups) {
        OptixGPUPrimitiveGroup *group = groupEntry.second;
        group->setSBTOffset(sbtOffset);
    }

    // now build the root group and referenced/instanced groups
    if (!mRootGroup->build(cudaStream, context, errorMsg)) {
        return false;
    }

    return true;
}

bool
OptixGPUAccelerator::createProgramGroups(std::string* errorMsg)
{
    // See OptixGPUPrograms.cu for the implementations of the programs.

    OptixProgramGroup pg;
    if (!createOptixRaygenProgramGroup(mContext,
                                       mModule,
                                       "__raygen__",
                                       &pg,
                                       errorMsg)) {
        return false;
    }
    mProgramGroups["raygen"] = pg;

    if (!createOptixMissProgramGroup(mContext,
                                     mModule,
                                     "__miss__",
                                     &pg,
                                     errorMsg)) {
        return false;
    }
    mProgramGroups["miss"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mModule,
                                         "__intersection__box",
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["boxHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mModule,
                                         "__intersection__flat_bezier_curve",
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["flatBezierCurveHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mModule,
                                         "__intersection__flat_bspline_curve",
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["flatBsplineCurveHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mModule,
                                         "__intersection__flat_linear_curve",
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["flatLinearCurveHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mRoundLinearCurvesModule,
                                         nullptr,
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["roundLinearCurvesHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mRoundLinearCurvesMBModule,
                                         nullptr,
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["roundLinearCurvesMBHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mRoundCubicBsplineCurvesModule,
                                         nullptr,
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["roundCubicBsplineCurvesHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mRoundCubicBsplineCurvesMBModule,
                                         nullptr,
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["roundCubicBsplineCurvesMBHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         nullptr,
                                         nullptr,
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["triMeshHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mModule,
                                         "__intersection__points",
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["pointsHG"] = pg;

    if (!createOptixHitGroupProgramGroup(mContext,
                                         mModule,
                                         "__anyhit__",
                                         mModule,
                                         "__closesthit__",
                                         mModule,
                                         "__intersection__sphere",
                                         &pg,
                                         errorMsg)) {
        return false;
    }
    mProgramGroups["sphereHG"] = pg;

    return true;
}

bool
OptixGPUAccelerator::createShaderBindingTable(std::string* errorMsg)
{
    mSBT = {}; // zero initialize

    {
        // Tells Optix to use the mRayGenPG program group for ray generation
        RaygenRecord rec = {};
        optixSbtRecordPackHeader(mProgramGroups["raygen"], &rec);
        if (mRaygenRecordBuf.allocAndUpload(&rec) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating raygen SBT record buffer";
            return false;
        }
        mSBT.raygenRecord = mRaygenRecordBuf.deviceptr();
    }
    {
        // Tells Optix to use the mMissPG program group for ray misses
        MissRecord rec = {};
        optixSbtRecordPackHeader(mProgramGroups["miss"], &rec);
        if (mMissRecordBuf.allocAndUpload(&rec) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating miss SBT record buffer";
            return false;
        }
        mSBT.missRecordBase = mMissRecordBuf.deviceptr();
        mSBT.missRecordStrideInBytes = sizeof(MissRecord);
        mSBT.missRecordCount = 1;
    }
    {
        std::vector<HitGroupRecord> hitgroupRecs;
        mRootGroup->getSBTRecords(mProgramGroups, hitgroupRecs);
        for (auto& groupEntry : mSharedGroups) {
            OptixGPUPrimitiveGroup *group = groupEntry.second;
            group->getSBTRecords(mProgramGroups, hitgroupRecs);
        }

        // Upload the hitgroup records to the GPU
        if (mHitGroupRecordBuf.allocAndUpload(hitgroupRecs) != cudaSuccess) {
            *errorMsg = "GPU: Error allocating HitGroup SBT record buffer";
            return false;
        }

        // Point the shader binding table to the HitGroup records on the GPU
        mSBT.hitgroupRecordBase = mHitGroupRecordBuf.deviceptr();
        mSBT.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        mSBT.hitgroupRecordCount = (unsigned int)hitgroupRecs.size();
    }

    return true;
}

void
OptixGPUAccelerator::intersect(const unsigned /*numRays*/, const GPURay* /*rays*/) const
{
    // TODO
}

void
OptixGPUAccelerator::occluded(const unsigned numRays, const GPURay* rays) const
{
    // std::cout << "occluded(): " << numRays << std::endl;

    MNRY_ASSERT_REQUIRE(numRays <= mRaysBufSize);

    // Setup the global GPU parameters
    OptixGPUParams params;
    params.mAccel = mRootGroup->mTopLevelIAS;
    params.mNumRays = numRays;
    params.mRaysBuf = mRaysBuf.ptr();
    params.mIsOccludedBuf = mIsOccludedBuf.ptr();
    params.mIsectBuf = nullptr;
    mParamsBuf.upload(&params);

    // Upload the rays to the GPU
    mRaysBuf.upload(rays, numRays);

    if (optixLaunch(mPipeline,
                    mCudaStream,
                    mParamsBuf.deviceptr(),
                    mParamsBuf.sizeInBytes(),
                    &mSBT,
                    numRays, 1, 1) != OPTIX_SUCCESS) {
        // There isn't any feasible way to recover from this, but it shouldn't
        // happen unless the code is broken.  Log the error and try to keep going.
        scene_rdl2::logging::Logger::error("GPU: optixLaunch() failure");
    }

    // Wait for the GPU to finish processing the rays
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // There isn't any feasible way to recover from this, but it shouldn't
        // happen unless the code is broken.  Log the error and try to keep going.
        scene_rdl2::logging::Logger::error("GPU: optixLaunch() cudaDeviceSynchronize() error: ",
                            cudaGetErrorString(error));
    }

    // Download the intersection results from the GPU
    mIsOccludedBuf.download(mOutputOcclusionBuf, numRays);
}

} // namespace rt
} // namespace moonray

