// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "MeshLight.h"
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/mcrt_common/SOAUtil.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/pbr/light/MeshLight_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/rt/IntersectionFilters.h>

#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

#define SHADING_BRACKET_TIMING_ENABLED

#ifdef SHADING_BRACKET_TIMING_ENABLED
#include <moonray/common/time/Ticker.h>
#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#endif

using namespace scene_rdl2;
// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.
using scene_rdl2::logging::Logger;

namespace moonray {
namespace pbr {


bool                                     MeshLight::sAttributeKeyInitialized;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>           MeshLight::sNormalizedKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>           MeshLight::sApplySceneScaleKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject *>  MeshLight::sGeometryKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::StringVector>   MeshLight::sPartsKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject *>  MeshLight::sMapShaderKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>          MeshLight::sClearRadiusKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>          MeshLight::sClearRadiusFalloffDistanceKey;
scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>            MeshLight::sClearRadiusInterpolationKey;

// TODO for Motion blur:
/*
 * convert geometry node xform, geometry vertex buffer, and light node xform into
 * 2 vertex buffers
 * In BVH, each node has 2 bounding regions: bbox and bounding cone
 * When building the bvh, we must extend the bbox and bounding cones from t0 to t1
 * then compute the buckets and cost on those. The centroid is the centroid of the
 * extended bounds
 *
 * For intersection and sampling, traverse bvh on Interpolate(node, time).
 * So we get the importance values from Interpolate(node, time) instead of node.
 *
 * If we just do transformational motion blur. We don't need to modify the bvh.
 * The Interpolate(node, time) function will just apply the transform at time
 * t to the bbox and bcone. We could potentially do that first and then rewrite
 * Interpolate(node, time) later to include deformational motion blur.
 *
 * Example of motion blur to pbrt bvh from the web:
 * https://github.com/popDuorz/pbrt-v3-bvh-motionblur/blob/master/accelerators/bvh.cpp
 */

// The number of buckets used to build the Sampling BVH.
static constexpr int32_t sBucketsCount = 8;

// thetaE describes the angle of influence of a light's emmission.
// For a flat polygon in a mesh, thetaE is pi / 2. If there are multiple
// polygons in a light, thetaE remains pi / 2.
static constexpr float sThetaE = scene_rdl2::math::sHalfPi;
// thetaO describes the angular spread of a light's orientation.
// For a flat triangle in a mesh, thetaO is 0. If there are multiple
// triangles in a light, thetaO is the angular spread of the normals of the
// triangles.
static constexpr float sThetaO = 0.0f;

//----------------------------------------------------------------------------

// Get uv coordinates of a point p on a quad. The first triangle of the quad is defined
// by p0, p1, and p3. The second triangle is defined by p2, p3, and p1. If u + v <= 1,
// then the point lies on the first tringle. If u + v > 1, then it lies on the second
// triangle. If the quad is degenerate, or p lies outside of the quad, then we fall back
// to a dummy uv of (0,0).
scene_rdl2::math::Vec2f getQuadUVCoordinates(const scene_rdl2::math::Vec3f& p0, const scene_rdl2::math::Vec3f& p1, const scene_rdl2::math::Vec3f& p2,
    const scene_rdl2::math::Vec3f& p3, const scene_rdl2::math::Vec3f& p)
{
    // Which triangle in the quad do we use to compute the uv coordinates for p?
    // Figure out which side of the triangle's diagonal the point is on.
    // The diagonal is p3 - p1.
    const scene_rdl2::math::Vec3f diagonal = p3 - p1;
    const scene_rdl2::math::Vec3f pvec = p - p1;
    const scene_rdl2::math::Vec3f pvecxdiag = math::cross(pvec, diagonal);
    // edge on first triangle
    const scene_rdl2::math::Vec3f e1 = p0 - p1;
    // edge on second triangle
    const scene_rdl2::math::Vec3f e2 = p2 - p1;
    // normal of first triangle
    const scene_rdl2::math::Vec3f n1 = math::cross(e1, diagonal);
    // normal of second triangle (pointing in opposite direction)
    const scene_rdl2::math::Vec3f n2 = math::cross(e2, diagonal);
    float tri1 = math::dot(pvecxdiag, n1);
    float tri2 = math::dot(pvecxdiag, n2);
    // 99% of the time, one dot product will be negative and the other will be positive.
    // But if there is a precision error, pick the larger one.
    bool insideTri1 = tri1 > tri2;

    scene_rdl2::math::Vec2f uv;
    if (insideTri1) {
        float recipDenom = 1.0f / dot(n1, n1);
        // w is weight applied to p0
        float w = tri1 * recipDenom;
        // v is weight applied to p3
        float v = math::dot(math::cross(e1, pvec), n1) * recipDenom;
        // u is weight applied to p1
        float u = 1.f - v - w;
        uv.x = u;
        uv.y = v;
    } else {
        float recipDenom = 1.0f / dot(n2, n2);
        // w is weight applied to p2
        float w = tri2 * recipDenom;
        // v is weight applied to p3
        float v = math::dot(math::cross(e2, pvec), n2) * recipDenom;
        // u is weight applied to p1
        float u = 1.f - v - w;
        // second triangle must flip the uv coordinates
        uv.x = 1.f - u;
        uv.y = 1.f - v;
    }

    // Check that uv is valid. Reasons to have an invalid uv are:
    // Zero area quad or the point p is outside the quad.
    // In that case, make a dummy uv.
    if (!math::isFinite(uv) ||
        uv.x > 1.f || uv.y > 1.f ||
        uv.x < 0.f || uv.y < 0.f) {

        uv.x = 0.f;
        uv.y = 0.f;
    }

    return uv;
}

// Given three points that define a triangle, find the cross product of
// those axes, which is the unnormalized normal of the triangle.
scene_rdl2::math::Vec3f getNormal(const scene_rdl2::math::Vec3f& p1, const scene_rdl2::math::Vec3f& p2, const scene_rdl2::math::Vec3f& p3)
{
    const scene_rdl2::math::Vec3f axis1 = p2 - p1;
    const scene_rdl2::math::Vec3f axis2 = p3 - p1;
    return cross(axis1, axis2);
}

// Given three points that define a plane, find the normal of that plane.
scene_rdl2::math::Vec3f getUnitNormal(const scene_rdl2::math::Vec3f& p1, const scene_rdl2::math::Vec3f& p2, const scene_rdl2::math::Vec3f& p3)
{
    const scene_rdl2::math::Vec3f axis1 = p2 - p1;
    const scene_rdl2::math::Vec3f axis2 = p3 - p1;
    return normalize(cross(axis1, axis2));
}

// Given n normalized vectors and a central axis vector, find the maximum angle
// between each vector and the axis. This is used to find thetaO of a face that
// has motion blur. ThetaO spans the motion of the normal of that face from
// t = 0 to t = 1. If the face is a quad, then the normals from both subtriangles
// must be considered.
float getMaxAngle(const scene_rdl2::math::Vec3f& axis, const scene_rdl2::math::Vec3f vectors[], size_t n)
{
    float minCosAngle = 1.f;
    for (size_t i = 0; i < n; ++i) {
        float cosAngle = dot(axis, vectors[i]);
        if (cosAngle < minCosAngle) {
            minCosAngle = cosAngle;
        }
    }

    return scene_rdl2::math::acos(scene_rdl2::math::clamp(minCosAngle, -1.f, 1.f));
}

//----------------------------------------------------------------------------

// A Cone describes the parameters needed to compute the orientation heuristics
// when building and the bvh and computing the importance of a node in the bvh.
// It is the orientation analogue to a bounding box.
struct Cone {
    scene_rdl2::math::Vec3f mAxis;
    float mThetaO;
    // for the mesh light, we do not need to store thetaE because it is always
    // pi / 2. For a many light solution we would need to store thetaE for each
    // bounding cone.
    //float mThetaE;

    void print() const {
        std::cout << "axis: " << mAxis << std::endl;
        std::cout << "thetaO: " << mThetaO << std::endl;
        //std::cout << "thetaE: " << mThetaE << std::endl;
    }
};

// Taken from Algorithm 1 in Kulla (2017) p. 5
// Given two cones what is their combined axis and orientation angle?
Cone
combineBoundingCones(const Cone* a, const Cone* b)
{
    if (b->mThetaO > a->mThetaO) {
        std::swap(a, b);
    }

    // cone a is a sphere. Just return it.
    if (a->mThetaO > scene_rdl2::math::sPi) {
        return *a;
    }

    float cosThetaD = scene_rdl2::math::clamp(dot(a->mAxis, b->mAxis), -1.0f, 1.0f);
    float thetaD = scene_rdl2::math::acos(cosThetaD);

    // For mesh lights, we do not need to compute thetaE because it is always
    // pi / 2. For a many light solution we need to compute thetaE. I am leaving
    // this computation as a reminder in case we ever want to use this for a many
    // lights solution.
    //float thetaE = max(a.mThetaE, b.mThetaE);

    // The bound of cone a already covers b, return cone a
    if (thetaD + b->mThetaO <= a->mThetaO) {
        return *a;
    }

    // construct new bounding cone
    float thetaO = (a->mThetaO + thetaD + b->mThetaO) / 2.0f;
    // thetaO cannot be greater than pi for a sphere.
    if (scene_rdl2::math::sPi <= thetaO) {
        return {a->mAxis, scene_rdl2::math::sPi};
    }

    // rotate a's axis towards b's axis so thetaO covers both
    scene_rdl2::math::Vec3f rotAxis = cross(a->mAxis, b->mAxis);
    if (scene_rdl2::math::isZero(length(rotAxis))) {
        // cone becomes sphere because axes are antiparallel
        return { scene_rdl2::math::Vec3f(1.f, 0.f, 0.f), scene_rdl2::math::sPi };
    }
    rotAxis = rotAxis.normalize();
    float thetaR = thetaO - a->mThetaO;
    const scene_rdl2::math::Vec3f axis = scene_rdl2::math::cos(thetaR) * a->mAxis + scene_rdl2::math::sin(thetaR) * cross(rotAxis,
                                                                                                        a->mAxis);
    return {axis, thetaO};
}

// A Face is a polygon in the mesh. It behaves as a flat polygonal light.
struct Face
{
    const int* mIndices; // vertex buffer indices
    scene_rdl2::math::Vec3f mCenter; // center of  the face
    float mInvArea; // 1 / area of the face
    scene_rdl2::math::Vec3f mNormal; // face normal
    int mPrimID; // primitive id of the face in the mesh
    int mGeomID; // id of mesh in the mesh light
    float mEnergy; // energy of the face
    float mThetaO; // for quad faces, the angle between the two normals

    void print() const
    {
        std::cout << "primID: " << mPrimID << std::endl;
        std::cout << "geomID: " << mGeomID << std::endl;
        std::cout << "center: " << mCenter << std::endl;
        std::cout << "inv area: " << mInvArea << std::endl;
        std::cout << "normal: " << mNormal << std::endl;
        std::cout << "energy: " << mEnergy << std::endl;
        std::cout << "thetaO: " << mThetaO << std::endl;
    }
};

// From Estevez and Kulla (2017)
// This is the node struct of their SAOH sampling BVH.
// It contains the bounding box, the bounding cone, the light energy and
// the information needed to do forward and backward traversal of the BVH.
struct Node
{
    float mEnergy; // total energy in this node
    int mRightChildIndex; // index of right child in BVH
    int mParentIndex; // index of parent in BVH
    Face* mFace; // pointer to the face in the mFaces buffer
    Cone mBcone; // bounding cone of the node
    scene_rdl2::math::BBox3f mBbox; // bounding box of the node

    bool isLeaf() const
    {
        return  mFace != nullptr;
    }

    void print() const
    {
        if (isLeaf()) {
            mFace->print();
        } else {
            std::cout << "mBcone: \n";
            mBcone.print();
            std::cout << "mBbox: " << mBbox << std::endl;
            std::cout << "rightChildIndex: " << mRightChildIndex << std::endl;
            std::cout << "center: " << getCenter() << std::endl;
        }
    }

    scene_rdl2::math::Vec3f getCenter() const
    {
        if (isLeaf()) {
            return mFace->mCenter;
        } else {
            return 0.5f * (mBbox.lower + mBbox.upper);
        }
    }
};

//----------------------------------------------------------------------------

// We spatially bin the primitives of the bvh into buckets.
struct Bucket {
    Bucket() : mEnergy(0.0f), mBbox(scene_rdl2::util::empty), mBcone({scene_rdl2::math::Vec3f(0.f), 0.f}) {}
    float mEnergy;
    scene_rdl2::math::BBox3f mBbox;
    Cone mBcone;
};

// The SplitCost computes the cost of splitting the bvh node at a particular
// location. The cost for splitting a node N into nodes NL and NR is
// c * (EL * SAL * OL + ER * SAR * OR) / (SA * O).
// c is the cost of splitting along axis x, y or z.
// E (and EL and ER) is the cumulative energy of all the primitives in the node.
// SA (and SAL and SAR) is the bounding box surface area of the node.
// O (and OR and OL) is the solid angle of light emission influence of the node.
struct SplitCost {
    SplitCost() : mLeftEnergy(0.0f), mRightEnergy(0.0f),
        mLeftBbox(scene_rdl2::util::empty), mRightBbox(scene_rdl2::util::empty),
        mLeftBcone({scene_rdl2::math::Vec3f(0.f), 0.f}), mRightBcone({scene_rdl2::math::Vec3f(0.f), 0.f})
    {}

    // Compute the cost of the bounding cone
    float orientationHeuristic(float thetaO) const
    {
        // Kulla (2017) p. 6 Eq. 1.
        float thetaW = scene_rdl2::math::min(thetaO + sThetaE, scene_rdl2::math::sPi);
        return (scene_rdl2::math::sPi/2) * (4 - 3 * scene_rdl2::math::cos(thetaO) + 2 * (thetaW - thetaO) *
            scene_rdl2::math::sin(thetaO) - scene_rdl2::math::cos(thetaO - 2 * thetaW));
    }

    inline float boxArea(const scene_rdl2::math::Vec3f& dim) const
    {
        return dim[0] * dim[1] + dim[1] * dim[2] + dim[2] * dim[0];
    };

    float eval(float splitCoefficient) const
    {
        // get cost of orientation
        float leftOrientation = orientationHeuristic(mLeftBcone.mThetaO);
        float rightOrientation = orientationHeuristic(mRightBcone.mThetaO);
        Cone combinedBcone = combineBoundingCones(&mLeftBcone, &mRightBcone);
        float combinedOrientation = orientationHeuristic(combinedBcone.mThetaO);

        // get cost of surface area
        const scene_rdl2::math::Vec3f leftDim = mLeftBbox.size();
        const scene_rdl2::math::Vec3f rightDim = mRightBbox.size();

        float leftBoundSA = boxArea(leftDim);
        float rightBoundSA = boxArea(rightDim);

        scene_rdl2::math::BBox3f combinedBound = mLeftBbox;
        combinedBound.extend(mRightBbox);
        const scene_rdl2::math::Vec3f totalDim = combinedBound.size();
        float combinedBoundSA = boxArea(totalDim);

        // SAOH
        return (mLeftEnergy * leftBoundSA * leftOrientation
            + mRightEnergy * rightBoundSA * rightOrientation)
            * splitCoefficient / (combinedBoundSA * combinedOrientation);
    };

    float mLeftEnergy;
    float mRightEnergy;
    scene_rdl2::math::BBox3f mLeftBbox;
    scene_rdl2::math::BBox3f mRightBbox;
    Cone mLeftBcone;
    Cone mRightBcone;
};

//----------------------------------------------------------------------------

HUD_VALIDATOR(MeshLight);

MeshLight::MeshLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling) :
    LocalParamLight(rdlLight),
    mBVHPtr(nullptr),
    mBVHSize(0),
    mMbSteps(0),
    mDeformationMb(false),
    mVerticesPtr(nullptr),
    mVertexOffsetPtr(nullptr),
    mFaceOffsetPtr(nullptr),
    mFaceVertexCountPtr(nullptr),
    mPrimIDToNodeIDPtr(nullptr),
    mRtcScene(nullptr),
    mMapShader(nullptr),
    mRdlGeometry(nullptr),
    mLayer(nullptr),
    mAttributeTable(nullptr)
{
    initAttributeKeys(rdlLight->getSceneClass());

    ispc::MeshLight_init(this->asIspc(), uniformSampling);
}

MeshLight::~MeshLight()
{
    reset();
}

bool
MeshLight::update(const scene_rdl2::math::Mat4d& world2render)
{
    MNRY_ASSERT(mRdlLight);

    // reset everything
    reset();

    // Store the scene's render to world transform, which is equivalent to the
    // render to light transform. Please see the explanation about this transform
    // in MeshLight.h
    mRender2Light = world2render.inverse();

    // check if light is on and the reference geometry is valid
    mOn = mRdlLight->get(scene_rdl2::rdl2::Light::sOnKey) && getReferenceGeometry();
    if (!mOn) {
        return false;
    }

    // Compute a dummy radiance using just the color, intensity and exposure.
    // There is no point in continuing the update if this evaluates to zero.
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
         scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
         sApplySceneScaleKey, 1.f /* dummy inverse area */);

    if (isBlack(mRadiance)) {
        mOn = false;
        return false;
    }

    // Update light transform
    // Accommodating 2 motion steps for now. However we do not have light motion
    // blur working yet so only the first motion step is used.
    const scene_rdl2::math::Mat4d& l2w0 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 0.f);
    const scene_rdl2::math::Mat4d& l2w1 = mRdlLight->get(scene_rdl2::rdl2::Node::sNodeXformKey, /* rayTime = */ 1.f);
    const scene_rdl2::math::Mat4f local2Render0 = toFloat(l2w0 * world2render);
    const scene_rdl2::math::Mat4f local2Render1 = toFloat(l2w1 * world2render);
    if (!updateParamAndTransforms(local2Render0, local2Render1, 1.0f, 1.0f)) {
        return false;
    }

    // Update miscellaneous
    updateVisibilityFlags();
    updatePresenceShadows();
    updateRayTermination();
    updateMaxShadowDistance();

    // update rdl geometry
    scene_rdl2::rdl2::SceneObject* geomSo = mRdlLight->get(sGeometryKey);
    if (geomSo) {
        mRdlGeometry = geomSo->asA<scene_rdl2::rdl2::Geometry>();
    }

    // update image map
    scene_rdl2::rdl2::SceneObject* mapSo = mRdlLight->get(sMapShaderKey);
    if (mapSo) {
        // ispc ptr is int64
        mMapShader = (const int64_t *) mapSo->asA<scene_rdl2::rdl2::Map>();
    }

    UPDATE_ATTRS_CLEAR_RADIUS

    return true;
}

void
MeshLight::reset() {
    mArea = 0;

    // reset offsets
    mVertexOffset.clear();
    mVertexOffsetPtr = nullptr;
    mFaceOffset.clear();
    mFaceOffsetPtr = nullptr;

    // clear vertices
    mVertices.clear();
    mVerticesPtr = nullptr;

    // clear primIDToNodeID
    mPrimIDToNodeID.clear();
    mPrimIDToNodeIDPtr = nullptr;

    // clear faces
    mFaces.clear();
    mFaceCount = 0;

    // clear BVH
    mBVH.clear();
    mBVHPtr = nullptr;
    mBVHSize = 0;

    // clear face vertex count
    mFaceVertexCount.clear();
    mFaceVertexCountPtr = nullptr;

    // reset other members
    mDeformationMb = false;
    mMbSteps = 0;
    mMapShader = nullptr;
    mRdlGeometry = nullptr;
    mLayer = nullptr;
    mAttributeTable = nullptr;

    // clear embree scene
    if (mRtcScene != nullptr) {
        size_t meshCount = mGeomMeshes.size();
        for (size_t geomID = 0; geomID < meshCount; ++geomID) {
            rtcDetachGeometry(mRtcScene, geomID);
            rtcReleaseGeometry(rtcGetGeometry(mRtcScene, geomID));
        }
        rtcReleaseScene(mRtcScene);
        mRtcScene = nullptr;
    }
    mGeomMeshes.clear();

    mRenderSpaceBounds = scene_rdl2::math::BBox3f(scene_rdl2::util::empty);
    mLocalSpaceBounds = scene_rdl2::math::BBox3f(scene_rdl2::util::empty);
}

bool
MeshLight::canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time, float radius,
    const LightFilterList* lightFilterList) const
{
    MNRY_ASSERT(mOn);
    // TODO: Consider a bounding solid angle
    if (lightFilterList) {
        float lightRadius = math::length(mBVH[0].mBbox.upper - mBVH[0].mBbox.lower) / 2;

        return canIlluminateLightFilterList(lightFilterList,
            { xformPointLocal2Render(mBVH[0].getCenter(), time),
              xformLocal2RenderScale(lightRadius, time),
              p, getXformRender2Local(time, lightFilterList->needsLightXform()),
              radius, time
            });
    }

    return true;
}

bool
MeshLight::isBounded() const
{
    return true;
}

bool
MeshLight::isDistant() const
{
    return false;
}

bool
MeshLight::isEnv() const
{
    return false;
}

scene_rdl2::math::BBox3f
MeshLight::getBounds() const
{
    return mRenderSpaceBounds;
}

void
MeshLight::setMesh(geom::internal::Primitive* prim)
{
    // If there is no reference mesh, then the light is off
    if (prim == nullptr) {
        return;
    }

    MNRY_ASSERT(prim->getType() == geom::internal::Primitive::POLYMESH);

    // Check parts
    bool validPart = false;
    const scene_rdl2::rdl2::StringVector& parts = getPartsList();
    for (const auto& part : parts) {
        if (mLayer->getAssignmentId(mRdlGeometry, part) != -1) {
            validPart = true;
            break;
        }
    }
    // load in all meshes if parts list is empty
    validPart |= parts.empty();

    if (!validPart) {
        return;
    }

    geom::internal::Mesh* geomMesh = (geom::internal::Mesh*)prim;
    geom::internal::Mesh::TessellatedMesh mesh;
    geomMesh->getTessellatedMesh(mesh);

    size_t faceCount = mesh.mFaceCount;

    // If the mesh is empty, early return.
    if (faceCount == 0) {
        return;
    }

    // Is this mesh a triangle mesh or a quad mesh?
    size_t faceVertexCount = 0;
    if (mesh.mIndexBufferType == geom::internal::MeshIndexType::TRIANGLE) {
        faceVertexCount = 3;
    } else if (mesh.mIndexBufferType == geom::internal::MeshIndexType::QUAD) {
        faceVertexCount = 4;
    } else {
        MNRY_ASSERT(0, "MeshLight: face vertex count is bad");
    }

    // The id of this mesh
    int geomID = mGeomMeshes.size();
    mGeomMeshes.push_back(geomMesh);

    // Does this mesh have deformation motion blur (ie, multiple vertex buffers?)
    mDeformationMb = (mesh.mVertexBufferDesc.size() > 1);

    // All meshes in the mesh light must have same number of motion steps.
    // setMesh is called multiple times on the same rdlGeometry, once per
    // mesh in the procedural. The first mesh determines the number of motion
    // blur steps. Each subsequent mesh checks its number of motion blur steps
    // against the first one.
    if (mMbSteps == 0) {
        mMbSteps = mesh.mVertexBufferDesc.size();
    } else {
        MNRY_ASSERT(mMbSteps == mesh.mVertexBufferDesc.size());
    }

    // index buffer
    const int* indices = (const int* )mesh.mIndexBufferDesc.mData;
    const size_t indexCount = faceCount * faceVertexCount;

    // get unique indices
    std::set<int> tmpOrderedIndices(indices, indices + indexCount);
    std::vector<int> orderedIndices(tmpOrderedIndices.begin(), tmpOrderedIndices.end());

    float dt = mMbSteps == 1 ? 1.f : 1.f / float(mMbSteps - 1);

    // vertex buffer
    size_t motionSampleCount = mesh.mVertexBufferDesc.size();
    size_t vertexCount = motionSampleCount * mesh.mVertexCount;
    size_t vertexOffset = mVertices.size();
    mVertexOffset.push_back(vertexOffset);
    mVertices.resize(vertexOffset + vertexCount);
    for (int index : orderedIndices) {
        for (size_t t = 0; t < motionSampleCount; ++t) {
            // byte offset of vertex
            size_t byteOffset = mesh.mVertexBufferDesc[t].mOffset + mesh.mVertexBufferDesc[t].mStride * index;

            // x component of vertex
            const float* f = reinterpret_cast<const float *>((const char *)mesh.mVertexBufferDesc[t].mData + byteOffset);

            // get and transform vertex
            const scene_rdl2::math::Vec3f vertex(*f, *(f+1), *(f+2));
            MNRY_ASSERT(math::isFinite(vertex));
            const scene_rdl2::math::Vec3f localSpaceVertex = transformPoint(mRender2Light, vertex);
            mVertices[vertexOffset + index * motionSampleCount + t] = localSpaceVertex;

            // extend bounding boxes in local space and render space
            mLocalSpaceBounds.extend(localSpaceVertex);
            float time = float(t) * dt;
            mRenderSpaceBounds.extend(xformPointLocal2Render(localSpaceVertex, time));

            // If we have transform motion blur but only one vertex buffer, we must expand the bounds at time point 1.0f
            if (mMb && motionSampleCount == 1) {
                // If there is rotational motion blur, we should have many motion steps. 10 is the default number of
                // motion steps for curved motion blur on geometries. If it is linear or scale motion blur then 2 motion
                // steps is enough.
                const size_t motionSteps = mMb & LIGHT_MB_ROTATION ? 10 : 2;
                const float dt = 1.f / float(motionSteps - 1);
                for (size_t t = 1; t < motionSteps; ++t) {
                    float time = float(t) * dt;
                    mRenderSpaceBounds.extend(xformPointLocal2Render(localSpaceVertex, time));
                }
            }
        }
    }

    // centroid time is for determining what face centers, areas, normals, and
    // energies will be used for the importance sampling when we have motion blur.
    float centroidTime = mDeformationMb ? 0.5f : 0.0f;

    // Face buffer
    std::vector<Face> faces(faceCount);
    for (size_t f = 0; f < faceCount; ++f) {
        // get IDs
        faces[f].mGeomID = geomID;
        faces[f].mPrimID = f;

        // get indices
        faces[f].mIndices = &indices[f * faceVertexCount];

        // get center
        faces[f].mCenter = getFaceVertex(faces[f], 0, centroidTime) +
            getFaceVertex(faces[f], 1, centroidTime) +
            getFaceVertex(faces[f], 2, centroidTime);
        if (faceVertexCount == 4) {
            faces[f].mCenter += getFaceVertex(faces[f], 3, centroidTime);
            faces[f].mCenter *= 0.25f;
        } else {
            faces[f].mCenter *= (1.0f / 3.0f);
        }

        float area = 0.f;
        if (faceVertexCount == 3 || getFaceVertex(faces[f], 2, centroidTime) ==
            getFaceVertex(faces[f], 3, centroidTime)) {
            // triangle or degenerate quad case

            // get area
            const scene_rdl2::math::Vec3f normal = getNormal(
                getFaceVertex(faces[f], 0, centroidTime),
                getFaceVertex(faces[f], 1, centroidTime),
                getFaceVertex(faces[f], 2, centroidTime));

            area = 0.5f * length(normal);

            // get normal
            faces[f].mNormal = normalize(normal);

            // get thetaO
            if (mDeformationMb) {
                // normal at t = 0
                const scene_rdl2::math::Vec3f normal0 = getUnitNormal(getFaceVertex(faces[f], 0, 0.f),
                    getFaceVertex(faces[f], 1, 0.f), getFaceVertex(faces[f], 2, 0.f));

                // normal at t = 1
                const scene_rdl2::math::Vec3f normal1 = getUnitNormal(getFaceVertex(faces[f], 0, 1.f),
                    getFaceVertex(faces[f], 1, 1.f), getFaceVertex(faces[f], 2, 1.f));

                // Accommodate motion of normals in thetaO. ThetaO is the
                // maximum angle between the normal at the centroid time point
                // and the normals at the first and last time point.
                const scene_rdl2::math::Vec3f normals[] = { normal0, normal1 };
                faces[f].mThetaO = getMaxAngle(faces[f].mNormal, normals, 2);
            } else {
                faces[f].mThetaO = sThetaO;
            }
        } else {
            // quad case

            // get area
            const scene_rdl2::math::Vec3f cross1 = getNormal(
                getFaceVertex(faces[f], 0, centroidTime),
                getFaceVertex(faces[f], 1, centroidTime),
                getFaceVertex(faces[f], 3, centroidTime));
            area = 0.5 * length(cross1);
            const scene_rdl2::math::Vec3f cross2 = getNormal(
                getFaceVertex(faces[f], 2, centroidTime),
                getFaceVertex(faces[f], 3, centroidTime),
                getFaceVertex(faces[f], 1, centroidTime));
            area += 0.5f * length(cross2);

            // get normal
            // A quad is not necessarily flat, so we average the two normals.
            faces[f].mNormal = normalize(cross1 + cross2);

            // get thetaO
            if (mDeformationMb) {
                // normal of first triangle at t = 0
                const scene_rdl2::math::Vec3f normal00 = getUnitNormal(getFaceVertex(faces[f], 0, 0.f),
                    getFaceVertex(faces[f], 1, 0.f), getFaceVertex(faces[f], 3, 0.f));

                // normal of second triangle at t = 0
                const scene_rdl2::math::Vec3f normal10 = getUnitNormal(getFaceVertex(faces[f], 2, 0.f),
                    getFaceVertex(faces[f], 3, 0.f), getFaceVertex(faces[f], 1, 0.f));

                // normal of first triangle at t = 1
                const scene_rdl2::math::Vec3f normal01 = getUnitNormal(getFaceVertex(faces[f], 0, 1.f),
                    getFaceVertex(faces[f], 1, 1.f), getFaceVertex(faces[f], 3, 1.f));

                // normal of second triangle at t = 1
                const scene_rdl2::math::Vec3f normal11 = getUnitNormal(getFaceVertex(faces[f], 2, 1.f),
                    getFaceVertex(faces[f], 3, 1.f), getFaceVertex(faces[f], 1, 1.f));

                // Accommodate motion of normals in thetaO. ThetaO is the
                // maximum angle between the normal at the centroid time point
                // and the normals at the first and last time point.
                const scene_rdl2::math::Vec3f normals[] = {normal00, normal10, normal01, normal11};
                faces[f].mThetaO = getMaxAngle(faces[f].mNormal, normals, 4);
            } else {
                // If there is no deformation mb, thetaO is the larger angle
                // between the two triangle normals and the face normal.
                const scene_rdl2::math::Vec3f normals[] = { normalize(cross1), normalize(cross2) };
                faces[f].mThetaO = getMaxAngle(faces[f].mNormal, normals, 2);
            }
        }
        mArea += area;
        faces[f].mInvArea = 1 / area;

        // get energy
        // TODO: more fine detail on texture sampling? Take average instead of max?
        // Currently we sample the texture from each corner of the mesh and from its
        // center. But it might be better to take many more samples than that.
        if (mMapShader) {
            float energy = 0.0f;
            mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();

            // In order to approximate the energy of a face, we sample at each
            // corner of the face and its center and take the maximum color value.
            // Due to the location of the samples, it is trivial to compute most
            // of the uv coordinates.

            // sample texture at first corner of the face
            const scene_rdl2::math::Color& firstColor = sampleMapShader(tls, geomID, f,
                getFaceVertex(faces[f], 0, centroidTime), faces[f].mNormal, scene_rdl2::math::Vec2f(0.f, 0.f));
            energy = scene_rdl2::math::max(energy, luminance(firstColor));

            // sample texture at second corner of the face
            const scene_rdl2::math::Color& secondColor = sampleMapShader(tls, geomID, f,
                getFaceVertex(faces[f], 1, centroidTime), faces[f].mNormal, scene_rdl2::math::Vec2f(1.f, 0.f));
            energy = scene_rdl2::math::max(energy, luminance(secondColor));

            // sample texture at third corner of the face
            scene_rdl2::math::Vec2f thirdUV;
            if (faceVertexCount == 3) {
                // triangle
                thirdUV = scene_rdl2::math::Vec2f(0.f, 1.f);
            } else {
                // quad
                thirdUV = scene_rdl2::math::Vec2f(1.f, 1.f);
            }
            const scene_rdl2::math::Color& thirdColor = sampleMapShader(tls, geomID, f,
                getFaceVertex(faces[f], 2, centroidTime), faces[f].mNormal, thirdUV);
            energy = scene_rdl2::math::max(energy, luminance(thirdColor));

            // sample texture at fourth corner if the face is a quad
            if (faceVertexCount == 4) {
                const scene_rdl2::math::Color& fourthColor = sampleMapShader(tls, geomID, f,
                    getFaceVertex(faces[f], 3, centroidTime), faces[f].mNormal, scene_rdl2::math::Vec2f(0.f, 1.f));
                energy = scene_rdl2::math::max(energy, luminance(fourthColor));
            }

            // sample texture in the middle of the face
            scene_rdl2::math::Vec2f centerUV;
            if (faceVertexCount == 3) {
                // the uv is trivial if the face is a triangle
                centerUV = scene_rdl2::math::Vec2f(1.f / 3.f, 1.f / 3.f);
            } else {
                // if the face is a quad, we need to compute the uv coordinate
                const scene_rdl2::math::Vec3f& p0 = getFaceVertex(faces[f], 0, centroidTime);
                const scene_rdl2::math::Vec3f& p1 = getFaceVertex(faces[f], 1, centroidTime);
                const scene_rdl2::math::Vec3f& p2 = getFaceVertex(faces[f], 2, centroidTime);
                const scene_rdl2::math::Vec3f& p3 = getFaceVertex(faces[f], 3, centroidTime);
                const scene_rdl2::math::Vec3f& center = faces[f].mCenter;

                centerUV = getQuadUVCoordinates(p0, p1, p2, p3, center);
            }
            const scene_rdl2::math::Color& centerColor = sampleMapShader(tls, geomID, f,
                faces[f].mCenter, faces[f].mNormal, centerUV);
            energy = scene_rdl2::math::max(energy, luminance(centerColor));
            faces[f].mEnergy = energy * area;
        } else {
            faces[f].mEnergy = 1.0f * area;
        }
    }

    // Append the faces from this mesh to the faces list. This vector will be submitted to the bvh.
    mFaceOffset.push_back(mFaces.size());
    mFaces.insert(mFaces.end(), faces.begin(), faces.end());
    mFaceVertexCount.push_back(faceVertexCount);
}

void
MeshLight::finalize() {
    // first check if there are any faces. If not, mOn is false.
    if (mFaces.empty()) {
        mOn = false;
        return;
    }

    if (scene_rdl2::math::isZero(mArea)) {
        mOn = false;
        return;
    }

    mInvArea = 1.0f / mArea;
    mFaceCount = mFaces.size();
    mPrimIDToNodeID.resize(mFaceCount);

    // Compute radiance
    mRadiance = computeLightRadiance(mRdlLight, scene_rdl2::rdl2::Light::sColorKey,
         scene_rdl2::rdl2::Light::sIntensityKey, scene_rdl2::rdl2::Light::sExposureKey, sNormalizedKey,
         sApplySceneScaleKey, mInvArea);

    if (isBlack(mRadiance)) {
        mOn = false;
        return;
    }

    // build the mesh light bvh
    mBVH.clear();
    buildBVHRecurse(mLocalSpaceBounds, mFaces, 0, mFaceCount, 0);
    MNRY_ASSERT(verifyBuild(0));

    // Fill in relevant HUD data
    mBVHPtr = mBVH.data();
    mBVHSize = mBVH.size();
    mVerticesPtr = mVertices.data();
    mVertexOffsetPtr = mVertexOffset.data();
    mFaceOffsetPtr = mFaceOffset.data();
    mFaceVertexCountPtr = mFaceVertexCount.data();
    mPrimIDToNodeIDPtr = mPrimIDToNodeID.data();
}


void
MeshLight::setEmbreeAccelerator(RTCDevice rtcDevice)
{
    if (mRtcScene == nullptr) {
        mRtcScene = rtcNewScene(rtcDevice);
        rtcSetSceneBuildQuality(mRtcScene, RTC_BUILD_QUALITY_HIGH);

        // iterate over all meshes
        size_t meshCount = mGeomMeshes.size();
        for (size_t i = 0; i < meshCount; ++i) {
            geom::internal::Mesh::TessellatedMesh mesh;
            mGeomMeshes[i]->getTessellatedMesh(mesh);

            // Create Embree geometry
            RTCGeometry rtcGeom;
            RTCFormat indexBufferFormat;
            if (mFaceVertexCount[i] == 3) {
                rtcGeom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE);
                indexBufferFormat = RTC_FORMAT_UINT3;
            } else if (mFaceVertexCount[i] == 4) {
                rtcGeom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_QUAD);
                indexBufferFormat = RTC_FORMAT_UINT4;
            } else {
                MNRY_ASSERT(0);
                return;
            }

            // set motion steps.
            rtcSetGeometryTimeStepCount(rtcGeom, mMbSteps);

            // Set up the mesh index buffer.
            rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_INDEX, 0,
                indexBufferFormat,
                mesh.mIndexBufferDesc.mData,
                mesh.mIndexBufferDesc.mOffset,
                mesh.mIndexBufferDesc.mStride,
                mesh.mFaceCount);

            // Set up the mesh vertex buffers
            for (size_t t = 0; t < mMbSteps; ++t) {
                rtcSetSharedGeometryBuffer(rtcGeom, RTC_BUFFER_TYPE_VERTEX, t,
                    RTC_FORMAT_FLOAT3, // xyz
                    (void*) mVerticesPtr,
                    (mVertexOffset[i] + t) * sizeof(scene_rdl2::math::Vec3f), /*offset*/
                    mMbSteps * sizeof(scene_rdl2::math::Vec3f), /*stride*/
                    mesh.mVertexCount);
            }

            // geometry mask
            unsigned mask = scene_rdl2::rdl2::ALL_VISIBLE;
            rtcSetGeometryMask(rtcGeom, mask);

            rtcSetGeometryIntersectFilterFunction(rtcGeom, &rt::backFaceCullingFilter);
            rtcSetGeometryOccludedFilterFunction(rtcGeom, &rt::backFaceCullingFilter);

            unsigned geomID = rtcAttachGeometry(mRtcScene, rtcGeom);
            // If this ever fails, we need to create a mapping from embree geomID
            // to the index into the mGeomMeshes vector.
            MNRY_ASSERT_REQUIRE(geomID == i);
            rtcCommitGeometry(rtcGeom);
        }
        rtcCommitScene(mRtcScene);
    }
}

unsigned int
MeshLight::getFaceVertexCount(const Face& face) const
{
    return mFaceVertexCount[face.mGeomID];
}

const scene_rdl2::math::Vec3f
MeshLight::getFaceVertex(const Face& face, size_t index, float time) const
{
    size_t vertexOffset = mVertexOffset[face.mGeomID] + face.mIndices[index] * mMbSteps;

    if (!mDeformationMb) {
        return mVertices[vertexOffset];
    }

    // Special case for time == 1.0f, because otherwise there will be a segfault
    // when trying to access v2 below. time == 1.0f when setting the mesh and
    // building the bvh, but it shouldn't occur during mcrt.
    if (time == 1.0f) {
        return mVertices[vertexOffset + mMbSteps - 1];
    }

    float invDt = mMbSteps - 1;
    float d = time * invDt;
    float step = scene_rdl2::math::floor(d);
    float t = d - step;

    const scene_rdl2::math::Vec3f& v1 = mVertices[vertexOffset + step];
    const scene_rdl2::math::Vec3f& v2 = mVertices[vertexOffset + step + 1];
    return math::lerp(v1, v2, t);
}

int
MeshLight::naiveSplit(std::vector<Face>& faces,
                      int start, int end, unsigned splitAxis,
                      scene_rdl2::math::BBox3f& leftBound, scene_rdl2::math::BBox3f& rightBound) const
{
    // We won't split a face
    MNRY_ASSERT(end - start > 1);

    leftBound = scene_rdl2::math::BBox3f(scene_rdl2::util::empty);
    rightBound = scene_rdl2::math::BBox3f(scene_rdl2::util::empty);

    // put half of the faces in the left node and half in the right
    int mid = (start + end) / 2;

    // partially sort each face according to the center of the face
    std::nth_element(faces.begin() + start, faces.begin() + mid, faces.begin() + end,
        [&](const Face &face1, const Face &face2) {
            return face1.mCenter[splitAxis] < face2.mCenter[splitAxis];
        }
    );

    // number of motion blur steps
    float dt = mDeformationMb ? 1.0f / (mMbSteps - 1) : 0.f;

    for (int i = start; i < mid; ++i) {
        unsigned fvc = getFaceVertexCount(faces[i]);
        for (unsigned j = 0; j < fvc; ++j) {
            for (unsigned t = 0; t < mMbSteps; ++t) {
                float time = dt * t;
                leftBound.extend(getFaceVertex(faces[i], j, time));
            }
        }
    }

    for (int i = mid; i < end; ++i) {
        unsigned fvc = getFaceVertexCount(faces[i]);
        for (unsigned j = 0; j < fvc; ++j) {
            for (unsigned t = 0; t < mMbSteps; ++t) {
                float time = dt * t;
                rightBound.extend(getFaceVertex(faces[i], j, time));
            }
        }
    }

    return mid;
}

int
MeshLight::buildBVHRecurse(const scene_rdl2::math::BBox3f& bbox, std::vector<Face>& faces,
    int start, int end, int parentIndex)
{
    if ((end - start) < sBucketsCount) {

        //
        // If we only have a few primitives in the node, fallback to a naive node
        // partition
        //

        int nodeIndex = mBVH.size();
        Node node;
        mBVH.push_back(node);
        if ((end - start) > 1) {
            // compute energy of node
            float energy = 0.0f;
            for (int i = start; i < end; ++i) {
                energy += faces[i].mEnergy;
            }

            // compute bounding cone of the node
            Cone cone = {faces[start].mNormal, faces[start].mThetaO};
            for (int i = start + 1; i < end; ++i) {
                const Cone faceCone = {faces[i].mNormal, faces[i].mThetaO};
                cone = combineBoundingCones(&cone, &faceCone);
            }

            // fill in node parameters
            mBVH[nodeIndex].mParentIndex = parentIndex;
            mBVH[nodeIndex].mBbox = bbox;
            mBVH[nodeIndex].mBcone = cone;
            mBVH[nodeIndex].mEnergy = energy;
            mBVH[nodeIndex].mFace = nullptr;

            // the longest length of the bounding box
            int splitAxis = maxDim(bbox.size());

            // split node into left and right nodes
            scene_rdl2::math::BBox3f leftBound;
            scene_rdl2::math::BBox3f rightBound;
            int mid = naiveSplit(faces, start, end, splitAxis, leftBound, rightBound);

            // recursively build left branch
            buildBVHRecurse(leftBound, faces, start, mid, nodeIndex);
            // recursively build right branch

            int rightChildIndex = buildBVHRecurse(rightBound, faces, mid, end, nodeIndex);
            mBVH[nodeIndex].mRightChildIndex = rightChildIndex;

        } else {
            // leaf node
            mBVH[nodeIndex].mParentIndex = parentIndex;
            mBVH[nodeIndex].mEnergy = faces[start].mEnergy;
            mBVH[nodeIndex].mFace = &faces[start];
            mBVH[nodeIndex].mBbox = bbox;
            mBVH[nodeIndex].mBcone = {faces[start].mNormal, faces[start].mThetaO};
            // right child nodeIndex is a dummy value that is never used for leaf nodes
            mBVH[nodeIndex].mRightChildIndex = -1;
            // Map the leaf node ids to the bvh node index
            unsigned faceOffset = mFaceOffset[faces[start].mGeomID] + faces[start].mPrimID;
            mPrimIDToNodeID[faceOffset] = nodeIndex;
        }

        return nodeIndex;
    }
    //
    // Split According to the Surface Area Orientation Heuristic
    //

    // Getting a bound around the centroids of the faces was a recommendation from
    // "On fast Construction of SAH-based Bounding Volume Hierarchies"
    // by Ingo Wald (2007).
    scene_rdl2::math::BBox3f centroidBbox(scene_rdl2::util::empty);
    for (int i = start; i < end; ++i) {
        centroidBbox.extend(faces[i].mCenter);
    }
    const scene_rdl2::math::Vec3f& centroidBboxSize = centroidBbox.size();
    const scene_rdl2::math::Vec3f& pMin = centroidBbox.lower;

    // Initialize the split index of the list of primitives as the start index
    int mid = start;

    // We iterate over the x, y, and z axes and compute the split cost for each
    // So we must keep track of the split cost and the left / right bounding
    // regions for each axis.
    float minCosts[3];
    std::pair<scene_rdl2::math::BBox3f, scene_rdl2::math::BBox3f> minBboxes[3];
    std::pair<Cone, Cone> minBcones[3];

    // iterate over all 3 coordinate axes
    for (int splitAxis = 0; splitAxis < 3; splitAxis++) {
        if (centroidBboxSize[splitAxis] == 0.f) {
            // The centroids of all the faces are perfectly aligned along
            // this axis. We cannot attempt to split along this axis.
            // Give the axis infinite cost and continue.
            minCosts[splitAxis] = math::pos_inf;
            continue;
        }

        // initialize the minBboxes to empty
        minBboxes[splitAxis] = {scene_rdl2::util::empty, scene_rdl2::util::empty};

        // Spatially bin primitives into buckets
        Bucket buckets[sBucketsCount];
        float stride = centroidBboxSize[splitAxis] / sBucketsCount;
        for (int i = start; i < end; ++i) {
            const Face &face = faces[i];
            int32_t bucketIndex = scene_rdl2::math::clamp(std::floor((face.mCenter[splitAxis] - pMin[splitAxis]) /
                stride), 0.0f, (float) (sBucketsCount - 1));

            buckets[bucketIndex].mEnergy += face.mEnergy;
            if (buckets[bucketIndex].mBbox.empty()) {
                // if this bucket is empty, initialize with the first bounding cone
                buckets[bucketIndex].mBcone = {face.mNormal, faces[i].mThetaO};
            } else {
                const Cone faceCone = {faces[i].mNormal, faces[i].mThetaO};
                buckets[bucketIndex].mBcone = combineBoundingCones(&buckets[bucketIndex].mBcone,
                    &faceCone);
            }
            buckets[bucketIndex].mBbox.extend(face.mCenter);
        }

        // Some buckets maybe empty. Remove those.
        Bucket finalBuckets[sBucketsCount];
        int bucketsCount = 0;
        for (int i = 0; i < sBucketsCount; ++i) {
            if (!buckets[i].mBbox.empty()) {
                finalBuckets[bucketsCount++] = buckets[i];
            }
        }

        // there should be at least 2 buckets at this point
        MNRY_ASSERT(bucketsCount >= 2);

        // the splitCoefficienct is a regularizing term that penalizes splitting
        // across axes with smaller lengths.
        float splitCoefficient = centroidBboxSize[maxDim(centroidBboxSize)] /
            centroidBboxSize[splitAxis];

        // Then split the node into left and right nodes along the splitAxis.
        if (bucketsCount == 2) {
            // only two buckets, must split between them
            minBboxes[splitAxis] = {finalBuckets[0].mBbox, finalBuckets[1].mBbox};
            minBcones[splitAxis] = {finalBuckets[0].mBcone, finalBuckets[1].mBcone};

            // compute split cost
            SplitCost cost;
            cost.mLeftEnergy = finalBuckets[0].mEnergy;
            cost.mLeftBbox = finalBuckets[0].mBbox;
            cost.mLeftBcone = finalBuckets[0].mBcone;
            cost.mRightEnergy = finalBuckets[1].mEnergy;
            cost.mRightBbox = finalBuckets[1].mBbox;
            cost.mRightBcone = finalBuckets[1].mBcone;
            minCosts[splitAxis] = cost.eval(splitCoefficient);

        } else {

            // The number of candidate splits is splitsCount. We compute the
            // cost of each candidate split and keep the minimum cost.
            int splitsCount = bucketsCount - 1;
            SplitCost costs[splitsCount];

            // Fill in the splitCost structs
            costs[0].mLeftEnergy = finalBuckets[0].mEnergy;
            costs[0].mLeftBbox = finalBuckets[0].mBbox;
            costs[0].mLeftBcone = finalBuckets[0].mBcone;
            costs[splitsCount - 1].mRightEnergy = finalBuckets[bucketsCount - 1].mEnergy;
            costs[splitsCount - 1].mRightBbox = finalBuckets[bucketsCount - 1].mBbox;
            costs[splitsCount - 1].mRightBcone = finalBuckets[bucketsCount - 1].mBcone;

            for (int i = 1; i < splitsCount; ++i) {
                costs[i].mLeftEnergy = finalBuckets[i].mEnergy + costs[i - 1].mLeftEnergy;
                costs[i].mLeftBbox = finalBuckets[i].mBbox;
                costs[i].mLeftBbox.extend(costs[i - 1].mLeftBbox);
                costs[i].mLeftBcone = combineBoundingCones(&finalBuckets[i].mBcone,
                    &costs[i - 1].mLeftBcone);
            }

            for (int i = splitsCount - 2; i >= 0; --i) {
                costs[i].mRightEnergy = finalBuckets[i + 1].mEnergy + costs[i + 1].mRightEnergy;
                costs[i].mRightBbox = finalBuckets[i + 1].mBbox;
                costs[i].mRightBbox.extend(costs[i + 1].mRightBbox);
                costs[i].mRightBcone = combineBoundingCones(&finalBuckets[i + 1].mBcone,
                    &costs[i + 1].mRightBcone);
            }

            // compute min split cost
            float minCost = costs[0].eval(splitCoefficient);
            int minCostSplit = 0;
            for (int i = 1; i < splitsCount; ++i) {
                float splitCost = costs[i].eval(splitCoefficient);
                if (splitCost < minCost) {
                    minCost = splitCost;
                    minCostSplit = i;
                }
            }

            // store the cost and bounds for the minimum cost split
            minCosts[splitAxis] = minCost;
            minBboxes[splitAxis] = {costs[minCostSplit].mLeftBbox,
                                    costs[minCostSplit].mRightBbox};
            minBcones[splitAxis] = {costs[minCostSplit].mLeftBcone,
                                    costs[minCostSplit].mRightBcone};
        }
    }

    // Get the axis with the minimum split cost
    int splitAxis = 0;
    float minCost = minCosts[splitAxis];
    for (int i = 1; i < 3; ++i) {
        if (minCosts[i] < minCost) {
            splitAxis = i;
            minCost = minCosts[i];
        }
    }

    scene_rdl2::math::BBox3f leftBound = minBboxes[splitAxis].first;
    scene_rdl2::math::BBox3f rightBound = minBboxes[splitAxis].second;

    // partition the faces into left and right nodes
    const auto& pMid = std::partition(faces.begin() + start, faces.begin() + end,
        [&](const Face &face) {
            return face.mCenter[splitAxis] <= leftBound.upper[splitAxis];
        }
    );

    mid = pMid - faces.begin();

    // If we did not successfully partition the faces, fall back to a naive partition.
    // There is no known scenario where the partition would be unsuccessful, so there
    // is an assert here. However, it is best to have a fallback just in case there
    // is a splitting scenario not already accounted for.
    Cone cone;
    if (mid == start || mid == end) {
        MNRY_ASSERT(0);
        splitAxis = maxDim(centroidBboxSize);
        mid = naiveSplit(faces, start, end, splitAxis, leftBound, rightBound);
        // compute bounding cone of this node
        cone = {faces[start].mNormal, faces[start].mThetaO};
        for (int i = start + 1; i < end; ++i) {
            const Cone faceCone = {faces[i].mNormal, faces[i].mThetaO};
            cone = combineBoundingCones(&cone, &faceCone);
        }
    } else {
        // We partitioned the centers of the faces into left and right bboxes
        // We must extend the bboxes to include the corners of the face.

        // number of motion blur steps
        float dt = mDeformationMb ? 1.0f / (mMbSteps - 1) : 0.f;

        for (int i = start; i < mid; ++i) {
            unsigned fvc = getFaceVertexCount(faces[i]);
            for (unsigned j = 0; j < fvc; ++j) {
                for (unsigned t = 0; t < mMbSteps; ++t) {
                    float time = dt * t;
                    leftBound.extend(getFaceVertex(faces[i], j, time));
                }
            }
        }

        for (int i = mid; i < end; ++i) {
            unsigned fvc = getFaceVertexCount(faces[i]);
            for (unsigned j = 0; j < fvc; ++j) {
                for (unsigned t = 0; t < mMbSteps; ++t) {
                    float time = dt * t;
                    rightBound.extend(getFaceVertex(faces[i], j, time));
                }
            }
        }
        // compute bounding cone of this node
        cone = combineBoundingCones(&minBcones[splitAxis].first, &minBcones[splitAxis].second);
    }

    // create node
    Node node;
    node.mParentIndex = parentIndex;
    node.mBbox = bbox;
    node.mBcone = cone;
    node.mFace = nullptr;
    node.mEnergy = 0.0f;
    for (int i = start; i < end; ++i) {
        node.mEnergy += faces[i].mEnergy;
    }

    int32_t nodeIndex = mBVH.size();
    mBVH.push_back(node);

    // recursively build left branch
    buildBVHRecurse(leftBound, faces, start, mid, nodeIndex);
    // recursively build right branch
    int rightChildIndex = buildBVHRecurse(rightBound, faces, mid, end, nodeIndex);
    mBVH[nodeIndex].mRightChildIndex = rightChildIndex;

    return nodeIndex;
}

bool
MeshLight::verifyBuild(int index) const
{
    // TODO: Is there anything else that should be verified?
    const Node& node = mBVH[index];
    if (node.isLeaf()) {
        return true;
    } else {
        const scene_rdl2::math::BBox3f& bbox = node.mBbox;
        const scene_rdl2::math::BBox3f& leftBbox = mBVH[index + 1].mBbox;
        const scene_rdl2::math::BBox3f& rightBbox = mBVH[node.mRightChildIndex].mBbox;
        bool leftIsInside = (bbox.lower.x <= leftBbox.lower.x &&
                             bbox.lower.y <= leftBbox.lower.y &&
                             bbox.lower.z <= leftBbox.lower.z &&
                             bbox.upper.x >= leftBbox.upper.x &&
                             bbox.upper.y >= leftBbox.upper.y &&
                             bbox.upper.z >= leftBbox.upper.z);
        bool rightIsInside = (bbox.lower.x <= rightBbox.lower.x &&
                              bbox.lower.y <= rightBbox.lower.y &&
                              bbox.lower.z <= rightBbox.lower.z &&
                              bbox.upper.x >= rightBbox.upper.x &&
                              bbox.upper.y >= rightBbox.upper.y &&
                              bbox.upper.z >= rightBbox.upper.z);

        // the energy of each node should be the sum of its children's energies
        bool correctEnergy = (scene_rdl2::math::isEqual(node.mEnergy, mBVH[index + 1].mEnergy +
            mBVH[node.mRightChildIndex].mEnergy, node.mEnergy * 1e-5f));

        return leftIsInside && rightIsInside && correctEnergy &&
            verifyBuild(index + 1) && verifyBuild(node.mRightChildIndex);
    }
}

float
MeshLight::importance(const scene_rdl2::math::Vec3f& shadingPoint, const scene_rdl2::math::Vec3f* shadingNormal, const Node& node) const
{
    // Early out if this node has no energy. This would happen if the MapShader
    // is black for all the faces contained in this node.
    if (node.mEnergy == 0.0f) {
        return 0.0f;
    }

    scene_rdl2::math::Vec3f center = node.getCenter();
    // the vector from the shadingPoint to the center of the node
    scene_rdl2::math::Vec3f point2center = center - shadingPoint;
    float distance2 = lengthSqr(point2center);

    // orientation metric
    // For more deatils see Kulla (2017) p. 8 Fig. 7 and Eq. 3.
    float orientation = 1.0f;
    if (!scene_rdl2::math::isEqual(node.mBcone.mThetaO, scene_rdl2::math::sPi) && distance2 != 0) {
        float distance = scene_rdl2::math::sqrt(distance2);
        point2center /= distance;
        // thetaU is the uncertainty angle from the center of the box to its edge
        float thetaU = 0.0f;
        // radius of the sphere that circumscribes the bounding box
        float radius = length(node.mBbox.upper - center);

        if (distance >= radius) {
            // the shading point is outside the bounding sphere
            thetaU = scene_rdl2::math::asin(radius / distance);
        } else {
            // shading point is inside the bounding sphere
            thetaU = scene_rdl2::math::sPi;
        }
        // thetaI is the angle between the axis and the shading normal
        float thetaI = 0.0f;
        float shadingNormalContribution = 1.0f;
        if (shadingNormal) {
            const scene_rdl2::math::Vec3f& normal = *shadingNormal;
            thetaI = scene_rdl2::math::acos(scene_rdl2::math::clamp(dot(point2center, normal), -1.0f, 1.0f));
            shadingNormalContribution = scene_rdl2::math::abs(scene_rdl2::math::cos(scene_rdl2::math::max(thetaI - thetaU, 0.0f)));
        }
        // theta is the angle between the bounding cone axis and the center to shading point axis
        float theta = scene_rdl2::math::acos(-dot(node.mBcone.mAxis, point2center));
        float thetaPrime = scene_rdl2::math::max(theta - node.mBcone.mThetaO - thetaU, 0.0f);
        // For the mesh light, thetaE is always the same. For the many lights
        // solution we would need to compare against node.mBcone.mThetaE
        bool useThetaPrime = thetaPrime < sThetaE;
        orientation = shadingNormalContribution * (useThetaPrime ? scene_rdl2::math::cos(thetaPrime) : 0.0f);
    }

    return node.mEnergy * orientation / distance2;
}

int
MeshLight::drawSampleRecurse(const scene_rdl2::math::Vec3f& shadingPoint, const scene_rdl2::math::Vec3f* shadingNormal,
    int currentNodeIndex, float u, float& pdf) const
{
    const Node& node = mBVH[currentNodeIndex];
    if (node.isLeaf()) {
        // The face is selected. Multiply by the inverse area of the face to get
        // the final pdf = pdf(face) * pdf(point | face)
        pdf *= node.mFace->mInvArea;
        return currentNodeIndex;
    }

    int32_t leftIndex = currentNodeIndex + 1;
    const Node& leftChild = mBVH[leftIndex];
    int32_t rightIndex = node.mRightChildIndex;
    const Node& rightChild = mBVH[rightIndex];

    float leftImportance = importance(shadingPoint, shadingNormal, leftChild);
    float rightImportance = importance(shadingPoint, shadingNormal, rightChild);

    MNRY_ASSERT(leftImportance >= 0.0f);
    MNRY_ASSERT(rightImportance >= 0.0f);

    float pdfL;
    if (leftImportance == 0 && rightImportance == 0) {
        // edge case
        pdfL = 0.5;
    } else {
        pdfL = leftImportance / (leftImportance + rightImportance);
    }

    float pdfR = 1.0f - pdfL;

    // During the traversal of the BVH, we my encounter situations where
    // u, pdfL, or pdfR equals exactly 1 or exactly 0.
    // If u == 1 and pdfL == 1, we want to traverse the left branch.
    // But if u == 0 and pdfL == 0, we want to traverse the right branch.
    if (u <= pdfL && pdfL != 0) {
        // traverse left branch
        float uRemapped = u / pdfL;
        pdf *= pdfL;
        return drawSampleRecurse(shadingPoint, shadingNormal, leftIndex,
            uRemapped, pdf);
    } else {
        // traverse right branch
        float uRemapped = (u - pdfL) / pdfR;
        pdf *= pdfR;
        return drawSampleRecurse(shadingPoint, shadingNormal, rightIndex,
            uRemapped, pdf);
    }
}

float
MeshLight::getPdfOfFace(size_t nodeID, const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f* n) const
{
    // compute pdf by traversing backwards through the bvh and computing
    // the importance of each traversed node and its sibling.
    MNRY_ASSERT(mBVH[nodeID].isLeaf());
    float pdf = 1.0f;
    unsigned currNodeIndex = nodeID;
    while (currNodeIndex > 0) {
        Node currNode = mBVH[currNodeIndex];
        MNRY_ASSERT(currNode.mParentIndex < (int)mBVH.size() ||
            (currNode.mParentIndex == -1 && currNodeIndex == 0));
        Node siblingNode;
        const Node& parentNode = mBVH[currNode.mParentIndex];
        if (parentNode.mRightChildIndex == currNodeIndex) {
            // currNode is the right child, sibling is the left child
            siblingNode = mBVH[currNode.mParentIndex + 1];
        } else {
            // currNode is the left child, sibling is the right child
            MNRY_ASSERT(currNodeIndex == currNode.mParentIndex + 1);
            siblingNode =  mBVH[parentNode.mRightChildIndex];
        }

        float currImportance = importance(p, n, currNode);
        float siblingImportance = importance(p, n, siblingNode);

        MNRY_ASSERT(currImportance >= 0.0f);
        MNRY_ASSERT(siblingImportance >= 0.0f);

        if (currImportance == 0.0f && siblingImportance == 0.0f) {
            pdf *= 0.5;
        } else {
            pdf *= currImportance / (currImportance + siblingImportance);
        }

        currNodeIndex = currNode.mParentIndex;
    }

    return pdf;
}

bool
MeshLight::intersect(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, const scene_rdl2::math::Vec3f &wi, float time, float maxDistance,
    LightIntersection &isect) const
{
    RTCRayHit rayHit;

    // transform ray origin and direction from render space to the light's local
    // space because the vertex buffer is in the light's local space
    const scene_rdl2::math::Vec3f transformedP = xformPointRender2Local(p, time);
    const scene_rdl2::math::Vec3f transformedWi = xformVectorRender2LocalRot(wi, time);

    rayHit.ray.org_x = transformedP.x;
    rayHit.ray.org_y = transformedP.y;
    rayHit.ray.org_z = transformedP.z;
    rayHit.ray.tnear = 0.0f;
    rayHit.ray.dir_x = transformedWi.x;
    rayHit.ray.dir_y = transformedWi.y;
    rayHit.ray.dir_z = transformedWi.z;
    rayHit.ray.time  = time;
    rayHit.ray.tfar  = maxDistance;
    rayHit.ray.mask  = ~0u;
    rayHit.ray.id = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // Call Embree intersection test
    LightIntersectContext context;

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &context.mRtcContext;

    rtcIntersect1(mRtcScene, &rayHit, &args);

    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }

    // compute pdf of this intersection.
    // pdf = pdf (point | face) * pdf(face)
    int faceOffset = mFaceOffset[rayHit.hit.geomID] + rayHit.hit.primID;
    int nodeIndex = mPrimIDToNodeID[faceOffset];
    float pdf = 1.0f;
    if (n) {
        // transform shading point's normal from render space to the light's
        // local space.
        const scene_rdl2::math::Vec3f transformedN = xformNormalRender2LocalRot(*n, time);
        pdf = mBVH[nodeIndex].mFace->mInvArea * getPdfOfFace(nodeIndex, transformedP, &transformedN);
    } else {
        pdf = mBVH[nodeIndex].mFace->mInvArea * getPdfOfFace(nodeIndex, transformedP, nullptr);
    }

    // Fill in isect members
    isect.N = xformNormalLocal2RenderRot(mBVH[nodeIndex].mFace->mNormal, time);
    isect.uv = scene_rdl2::math::Vec2f(rayHit.hit.u, rayHit.hit.v);
    isect.distance = xformLocal2RenderScale(rayHit.ray.tfar, time);
    isect.pdf = pdf;
    isect.primID = rayHit.hit.primID;
    isect.geomID = rayHit.hit.geomID;

    return true;
}

bool
MeshLight::sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time, const scene_rdl2::math::Vec3f& r,
    scene_rdl2::math::Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const
{
    MNRY_ASSERT(mOn);

    const float r1 = r[0];
    const float r2 = r[1];
    const float r3 = r[2];

    float pdf = 1.0f;
    int faceIndex;
    // transform shading point's position and normal from render space to the
    // light's local space, because the vertex buffer is in the light's local space.
    const scene_rdl2::math::Vec3f transformedP = xformPointRender2Local(p, time);
    if (n) {
        const scene_rdl2::math::Vec3f transformedN = xformNormalRender2LocalRot(*n, time);
        faceIndex = drawSampleRecurse(transformedP, &transformedN, 0, r3, pdf);
    } else {
        faceIndex = drawSampleRecurse(transformedP, nullptr, 0, r3, pdf);
    }

    MNRY_ASSERT(scene_rdl2::math::isfinite(pdf) && pdf >= 0.0f);
    MNRY_ASSERT(mBVH[faceIndex].isLeaf());
    const Face& face = *mBVH[faceIndex].mFace;

    scene_rdl2::math::Vec2f uv;
    scene_rdl2::math::Vec3f hit;
    scene_rdl2::math::Vec3f normal;

    if (getFaceVertexCount(face) == 3 || getFaceVertex(face, 2, time) == getFaceVertex(face, 3, time)) {
        // triangle or degenerate quad case
        scene_rdl2::math::Vec3f p1 = getFaceVertex(face, 0, time);
        scene_rdl2::math::Vec3f p2 = getFaceVertex(face, 1, time);
        scene_rdl2::math::Vec3f p3 = getFaceVertex(face, 2, time);

        // the random numbers r1 and r2 can be used as uv coordinates
        float u = r1;
        float v = r2;
        float w = 1 - u - v;
        if (w < 0) {
            // if outside the triangle, flip the uvs to be inside the triangle
            u = 1 - u;
            v = 1 - v;
            w = -w;
        }
        hit = w*p1 + u*p2 + v*p3;

        if (mDeformationMb) {
            // We cannot directly use face.mNormal here because it is the normal at
            // time = centroidTime. Therefore we compute it here.
            normal = normalize(cross(p2 - p1, p3 - p1));
        } else {
            normal = face.mNormal;
        }

        if (mMapShader) {
            // get barycentric coordinate for hit
            uv.x = u;
            uv.y = v;
        }
    } else {
        // quad case
        MNRY_ASSERT(getFaceVertexCount(face) == 4);

        scene_rdl2::math::Vec3f v0 = getFaceVertex(face, 0, time);
        scene_rdl2::math::Vec3f v1 = getFaceVertex(face, 1, time);
        scene_rdl2::math::Vec3f v2 = getFaceVertex(face, 2, time);
        scene_rdl2::math::Vec3f v3 = getFaceVertex(face, 3, time);

        scene_rdl2::math::Vec3f normal013 = cross(v1 - v0, v3 - v0);
        scene_rdl2::math::Vec3f normal231 = cross(v3 - v2, v1 - v2);
        float area013 = length(normal013);
        float area231 = length(normal231);
        float areaSum = area013 + area231;

        if (areaSum == 0) {
            return false;
        }
        
        float rSplit = area013 / areaSum;

        scene_rdl2::math::Vec3f p1;
        scene_rdl2::math::Vec3f p2;
        scene_rdl2::math::Vec3f p3;
        float u;

        // Choose which triangle according to the area of the triangles
        if (r1 < rSplit) {
            // first triangle
            u = r1 / rSplit;
            p1 = v0;
            p2 = v1;
            p3 = v3;
            normal = normal013 / area013;
        } else {
            // second triangle
            u = (r1 - rSplit) / (1 - rSplit);
            p1 = v2;
            p2 = v3;
            p3 = v1;
            normal = normal231 / area231;
        }

        float v = r2;
        float w = 1 - u - v;
        if (w < 0) {
            // if outside the triangle, flip the uvs to be inside the triangle
            u = 1 - u;
            v = 1 - v;
            w = -w;
        }
        hit = w*p1 + u*p2 + v*p3;

        if (mMapShader) {
            if (r1 < rSplit) {
                // first triangle
                uv.x = u;
                uv.y = v;
            } else {
                // second triangle
                uv.x = 1 - u;
                uv.y = 1 - v;
            }
        }
    }

    MNRY_ASSERT(isFinite(hit));

    // compute wi and hit distance
    wi = xformVectorLocal2Render(hit - transformedP, time);
    isect.distance = length(wi);
    if (isect.distance > scene_rdl2::math::sEpsilon) {
        wi /= isect.distance;
    }

    // light is on other side of shading point
    if (n && dot(*n, wi) < scene_rdl2::math::sEpsilon) {
        return false;
    }

    isect.N = xformNormalLocal2RenderRot(normal, time);
    isect.pdf = pdf;
    isect.primID = face.mPrimID;
    isect.geomID = face.mGeomID;
    isect.uv = uv;

    return true;
}

scene_rdl2::math::Color
MeshLight::eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &p, const LightFilterRandomValues& filterR, float time,
    const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList, float rayDirFootprint,
    float *pdf) const
{
    MNRY_ASSERT(mOn);

    scene_rdl2::math::Color radiance = mRadiance;

    if (lightFilterList) {
        evalLightFilterList(lightFilterList,
                            { tls, &isect, getPosition(time),
                              getDirection(time), p,
                              filterR, time,
                              getXformRender2Local(time, lightFilterList->needsLightXform()),
                              wi
                            },
                            radiance);
    }

    if (pdf) {
        *pdf = isect.pdf * areaToSolidAngleScale(wi, isect.N, isect.distance);
        MNRY_ASSERT(scene_rdl2::math::isfinite(*pdf));
        if (*pdf == 0.f) {
            // we don't need to sample the map shader if the pdf is 0.
            return radiance;
        }
    }

    // sample the texture
    // TODO: Use texture derivatives
    if (mMapShader) {
        radiance *= sampleMapShader(tls, isect.geomID, isect.primID, p, isect.N, isect.uv);
    }

    MNRY_ASSERT(isFinite(radiance));

    return radiance;
}

// TODO: implement this
scene_rdl2::math::Vec3f
MeshLight::getEquiAngularPivot(const scene_rdl2::math::Vec3f& r, float time) const
{
    return mBVH[0].getCenter();
}

void
MeshLight::initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc)
{
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sNormalizedKey      = sc.getAttributeKey<scene_rdl2::rdl2::Bool>         ("normalized");
    sApplySceneScaleKey = sc.getAttributeKey<scene_rdl2::rdl2::Bool>         ("apply_scene_scale");
    sGeometryKey        = sc.getAttributeKey<scene_rdl2::rdl2::SceneObject *>("geometry");
    sPartsKey           = sc.getAttributeKey<scene_rdl2::rdl2::StringVector> ("parts");
    sMapShaderKey       = sc.getAttributeKey<scene_rdl2::rdl2::SceneObject *>("map_shader");
    INIT_ATTR_KEYS_CLEAR_RADIUS

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

scene_rdl2::math::Color
MeshLight::sampleMapShader(mcrt_common::ThreadLocalState* tls, int geomID, int primID,
    const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n, const scene_rdl2::math::Vec2f& uv) const
{
    shading::Intersection isect;
    scene_rdl2::math::Vec2f st;
    mGeomMeshes[geomID]->getST(primID, uv.x, uv.y, st);
    int assignmentId = mGeomMeshes[geomID]->getFaceAssignmentId(primID);
    isect.initMapEvaluation(&tls->mArena, mAttributeTable, mRdlGeometry, mLayer,
        // All texture derivatives are set to 0.
        assignmentId, p, n, 0, 0, st, 0, 0, 0, 0);
    float u = uv.x;
    float v = uv.y;
    float w = 1.0f - uv.x - uv.y;
    bool isFirst = w >= 0.f;
    if (!isFirst) {
        u = 1.0f - u;
        v = 1.0f - v;
        w = -w;
    }
    mGeomMeshes[geomID]->setRequiredAttributes(primID, 0.f, u, v, w, isFirst, isect);
    scene_rdl2::math::Color result;
    const scene_rdl2::rdl2::Map * map = reinterpret_cast<const scene_rdl2::rdl2::Map *>(mMapShader);
    {
        // time shader
#ifdef SHADING_BRACKET_TIMING_ENABLED
        auto threadIndex = tls->mThreadIdx;
        time::RAIIInclusiveExclusiveTicker<int64> ticker(
            MNRY_VERIFY(map->getThreadLocalObjectState())[threadIndex].mShaderCallStat,
            MNRY_VERIFY(mLayer->lookupMaterial(assignmentId)->getThreadLocalObjectState())[threadIndex].mShaderCallStat);
#endif
        map->sample(tls->mShadingTls.get(), shading::State(&isect), &result);
    }

    return result;
}

extern "C" void
CPP_initShadingState(const MeshLight* light, shading::TLState* tls, int* geomID, int* primID,
    const float* p, const float* n, const float* uv, int32_t lanemask, shading::Intersectionv& isectv)
{
    // set primitive attribute offsets
    // needed only for ispc evaluation of attribute maps
    const shading::AttributeTable* attributeTable = light->getAttributeTable();
    if (attributeTable) {
        tls->mAttributeOffsets = attributeTable->getKeyOffsets();
    }

    shading::Intersection isects[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        // Don't init isects for invalid lanes.
        if (!isActive(lanemask, i)) {
            continue;
        }
        int geomId = geomID[i];
        int primId = primID[i];
        if (geomId < 0 || primId < 0) {
            // invalid intersection
            continue;
        }

        const geom::internal::Mesh* geomMesh = light->getGeomMesh(geomId);

        scene_rdl2::math::Vec2f st;
        float u = uv[i];
        float v = uv[VLEN + i];
        geomMesh->getST(primId, u, v, st);
        int assignmentId = geomMesh->getFaceAssignmentId(primId);
        const scene_rdl2::math::Vec3f P(p[i], p[VLEN + i], p[2*VLEN + i]);
        const scene_rdl2::math::Vec3f N(n[i], n[VLEN + i], n[2*VLEN + i]);
        isects[i].initMapEvaluation(tls->mArena, attributeTable, light->getRdlGeometry(),
            // All texture derivatives are set to 0.
            light->getLayer(), assignmentId, P, N, 0, 0, st, 0, 0, 0, 0);

        float w = 1.0f - u - v;
        bool isFirst = w >= 0.f;
        if (!isFirst) {
            u = 1.0f - u;
            v = 1.0f - v;
            w = -w;
        }
        geomMesh->setRequiredAttributes(primId, 0.f, u, v, w, isFirst, isects[i]);
    }

    // transpose AOS to SOA
    {
        ACCUMULATOR_PROFILE(tls, ACCUM_AOS_TO_SOA_INTERSECTIONS);

#if (VLEN == 16u)
    mcrt_common::convertAOSToSOA_AVX512
#elif (VLEN == 8u)
    mcrt_common::convertAOSToSOA_AVX
#else
    #error Requires at least AVX to build.
#endif
        <sizeof(shading::Intersection), sizeof(shading::Intersection),
         sizeof(shading::Intersectionv), 0>
            (VLEN, (const uint32_t *)isects, (uint32_t *)&isectv);
    }
}
//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

