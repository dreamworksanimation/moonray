// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightAccelerator.h"
#include "MeshLight.h"

#include <moonray/rendering/pbr/light/LightAccelerator_ispc_stubs.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>

#include <scene_rdl2/common/math/ispc/Typesv.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

// Define the types which depend on the vector width
#if (VLEN == 8u)
    const auto& rtcIntersectv = rtcIntersect8;
    typedef RTCRayHit8 RTCRayHitv;
    typedef RTCRay8 RTCRayv;
    static constexpr __align(32) int sAllValidMask[8] = {
        -1, -1, -1, -1, -1, -1, -1, -1};
#elif (VLEN == 16u)
    const auto& rtcIntersectv = rtcIntersect16;
    typedef RTCRayHit16 RTCRayHitv;
    typedef RTCRay16 RTCRayv;
    static constexpr __align(64) int sAllValidMask[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
#endif

LightAccelerator::LightAccelerator() :
    mRtcScene(nullptr),
    mLights(nullptr),
    mLightCount(0),
    mBoundedLights(nullptr),
    mBoundedLightCount(0),
    mUnboundedLights(nullptr),
    mUnboundedLightCount(0),
    mSamplingTree(0.f, 0.f)
{
}


// dtor

LightAccelerator::~LightAccelerator()
{
    if (mRtcScene) {
        rtcReleaseScene(mRtcScene);
    }
}


// bounds callback required by Embree.
// Accesses the light's bounding volume and copies it to bounds_o.
static void
boundsCallback(const struct RTCBoundsFunctionArguments* args)
{
    const Light*const* lights = (const Light*const*)args->geometryUserPtr;
    const BBox3f bbox = lights[args->primID]->getBounds();
    args->bounds_o->lower_x = bbox.lower.x;
    args->bounds_o->lower_y = bbox.lower.y;
    args->bounds_o->lower_z = bbox.lower.z;
    args->bounds_o->upper_x = bbox.upper.x;
    args->bounds_o->upper_y = bbox.upper.y;
    args->bounds_o->upper_z = bbox.upper.z;
}

// intersect callback required by Embree.
static void
intersectCallback(const RTCIntersectFunctionNArguments* args)
{
    int* valid = (int*)args->valid;
    unsigned int N = args->N;
    unsigned int primID = args->primID;
    RTCRayHitN* rayhit = (RTCRayHitN*)args->rayhit;
    RTCRayN* rays = RTCRayHitN_RayN(rayhit, N);
    RTCHitN* hits = RTCRayHitN_HitN(rayhit, N);
    const Light*const* lights = (const Light*const*)args->geometryUserPtr;
    // get light
    const Light* light = lights[primID];
    LightIntersectContext* context = (LightIntersectContext*)args->context;
    for (unsigned int index = 0; index < N; ++index) {
        // If the lane is not valid in vector mode or if the light in the accelerator
        // does not have a corresponding light in the LightSet, then we can skip this
        // intersection test.
        if (valid[index] == 0 || context->mLightIdMap[primID] == -1) {
            continue;
        }

        if (!(RTCRayN_mask(rays, N, index) & light->getVisibilityMask())) {
            // skip light if it is masked
            continue;
        }

        int rayId = RTCRayN_id(rays, N, index);
        bool includeRayTerminationLights = context->mIncludeRayTerminationLights[rayId];
        if (!includeRayTerminationLights && light->getIsRayTerminator()) {
            // Skip any ray termination lights if we were told not to include them
            continue;
        }

        const Vec3f p(
            RTCRayN_org_x(rays, N, index),
            RTCRayN_org_y(rays, N, index),
            RTCRayN_org_z(rays, N, index));
        const Vec3f wi(
            RTCRayN_dir_x(rays, N, index),
            RTCRayN_dir_y(rays, N, index),
            RTCRayN_dir_z(rays, N, index));
        float maxDistance = RTCRayN_tfar(rays, N, index);

        const Vec3f* n = nullptr;
        if (context->mShadingNormal) {
            // is mShadingNormal == nullptr, it is an invalid shading normal
            // in c++ code derived from a subsurface material
            n = &context->mShadingNormal[index];
            if (n->x > 1.f) {
                // invalid normal coming from ispc
                n = nullptr;
            }
        }

        LightIntersection isect;
        bool hit = light->intersect(p, n, wi, RTCRayN_time(rays, N, index),
            maxDistance, isect);
        if (!hit) {
            continue;
        }

        context->mNumHits[rayId]++;

        if (chooseThisLight(context->mSamples[rayId], context->mDepth[rayId], context->mNumHits[rayId])) {
            RTCHitN_instID(hits, N, index, 0) = context->mRtcContext.instID[0];
            RTCHitN_geomID(hits, N, index) = 0;
            RTCHitN_primID(hits, N, index) = primID;
            RTCHitN_u(hits, N, index) = isect.uv[0];
            RTCHitN_v(hits, N, index) = isect.uv[1];
            RTCHitN_Ng_x(hits, N, index) = isect.N.x;
            RTCHitN_Ng_y(hits, N, index) = isect.N.y;
            RTCHitN_Ng_z(hits, N, index) = isect.N.z;
            context->mData0[rayId] = isect.data[0];
            context->mData1[rayId] = isect.data[1];
            // We cannot set the RTCHit distance parameter to the isect.distance. If we do, the ray
            // traversal will stop at the closest ray hit. Therefore we store the distance of the chosen
            // light in the context instead.
            context->mDistance[rayId] = isect.distance;
            context->mPdf[rayId] = isect.pdf;
            context->mMeshGeomID[rayId] = isect.geomID;
            context->mMeshPrimID[rayId] = isect.primID;
        }
    }
}

// Initialise LightAccelerator using the given light list. We also pass in the Embree device
// created by the geometry manager so that we don't need to create our own.
// This function sets all of the callbacks required by Embree.

void
LightAccelerator::init(const Light*const* lights, int lightCount, const RTCDevice& rtcDevice, 
                       float sceneDiameter, float samplingThreshold)
{
    // Deal with boundary cases
    if (lights == nullptr) {
        mRtcScene            = nullptr;
        mLights              = nullptr;
        mLightCount          = 0;
        mBoundedLights       = nullptr;
        mBoundedLightCount   = 0;
        mUnboundedLights     = nullptr;
        mUnboundedLightCount = 0;
        return;
    }

    // Bounded lights should have been sorted to the front - count them
    int boundedLightCount = 0;
    for (int l=0; l<lightCount; l++) {
        if (!lights[l]->isBounded()) {
            break;
        }
        boundedLightCount++;
    }

    // Check that the rest are all unbounded
    for (int l=boundedLightCount; l<lightCount; l++) {
        MNRY_ASSERT(!lights[l]->isBounded());
    }  

    mLights = lights;
    mLightCount = lightCount;

    mBoundedLights = lights;
    mBoundedLightCount = boundedLightCount;

    mUnboundedLights = lights + boundedLightCount;
    mUnboundedLightCount = lightCount - boundedLightCount;

    // Create an Embree scene as long as there are enough lights to put in it.
    // We check the SCALAR_THRESHOLD_COUNT because even in vector mode,
    // we fall into scalar code when computing subsurface radiance.
    // All bounded lights are stored in embree as one UserGeometry
    if (boundedLightCount >= SCALAR_THRESHOLD_COUNT) {
        mRtcScene = rtcNewScene(rtcDevice);
        RTCGeometry rtcGeom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryBuildQuality(rtcGeom, RTC_BUILD_QUALITY_MEDIUM);
        rtcSetGeometryUserPrimitiveCount(rtcGeom, boundedLightCount);
        rtcSetGeometryTimeStepCount(rtcGeom, 1);
        rtcSetGeometryUserData(rtcGeom, (void *)const_cast<Light**>(mBoundedLights));
        rtcSetGeometryBoundsFunction(rtcGeom, boundsCallback, nullptr);
        rtcSetGeometryIntersectFunction(rtcGeom, intersectCallback);
        // The default mask is 0x00000001.  We need to set it to all 1s because we use
        //  the mask to mask light types instead of geometry.  If we don't, geometry 
        //  will incorrectly be ignored for certain light types.  We want all geometry
        //  to be tested for intersection regardless of the light type mask.
        rtcSetGeometryMask(rtcGeom, 0xffffffff);
        rtcAttachGeometry(mRtcScene, rtcGeom);
        rtcCommitGeometry(rtcGeom);
        rtcReleaseGeometry(rtcGeom);
        rtcCommitScene(mRtcScene);
    } else {
        mRtcScene = nullptr;
    }

    mSamplingTree.setSceneDiameter(sceneDiameter);
    mSamplingTree.setSamplingThreshold(samplingThreshold);
}


// The main intersection function for the Embree light acceleration structure.
// The intention is to avoid a brute-force search through the array of lights to find a randomly hit light.
// Only the bounded lights (cylinder, disk, rect, sphere, spot) are put in the acceleration structure.
// Unbounded lights by definition do not have a bounding volume so they must still be iterated through
// in search of a hit.

int
LightAccelerator::intersect(const Vec3f &P,  const Vec3f* N, const Vec3f &wi,
        float time, float maxDistance, bool includeRayTerminationLights, int visibilityMask,
        IntegratorSample1D &samples, int depth, LightIntersection &isect, int &numHits,
        const int* lightIdMap) const
{
    MNRY_ASSERT(mBoundedLightCount >= SCALAR_THRESHOLD_COUNT);
    MNRY_ASSERT(mRtcScene != nullptr);
    if (mBoundedLightCount < SCALAR_THRESHOLD_COUNT  ||  mRtcScene == nullptr) {
        return -1;
    }

    LightIntersection boundedIsect;
    int boundedLightIdx = intersectBounded(P, N, wi, time, maxDistance, includeRayTerminationLights, 
        visibilityMask, samples, depth, boundedIsect, numHits, lightIdMap);

    LightIntersection unboundedIsect;
    int unboundedLightIdx = intersectUnbounded(P, wi, time, maxDistance, includeRayTerminationLights,
        visibilityMask, samples, depth, unboundedIsect, numHits, lightIdMap);

    if (unboundedLightIdx >= 0) {
        isect = unboundedIsect;
        return unboundedLightIdx;
    }

    isect = boundedIsect;
    return boundedLightIdx;
}


// A simple wrapper for the Embree intersection test against the
// light acceleration structure. Its main role is to convert between
// the Moonray-style parameters and the Embree Ray struct.
int
LightAccelerator::intersectBounded(const Vec3f &P,  const Vec3f* N, const Vec3f &wi,
        float time, float maxDistance, bool includeRayTerminationLights, int visibilityMask,
        IntegratorSample1D &samples, int depth, LightIntersection &isect, int &numHits, const int* lightIdMap) const
{
    RTCRayHit rayHit;
    rayHit.ray.org_x = P.x;
    rayHit.ray.org_y = P.y;
    rayHit.ray.org_z = P.z;
    rayHit.ray.tnear = 0.0f;
    rayHit.ray.dir_x = wi.x;
    rayHit.ray.dir_y = wi.y;
    rayHit.ray.dir_z = wi.z;
    rayHit.ray.time  = time;
    rayHit.ray.tfar  = maxDistance;
    rayHit.ray.mask  = visibilityMask;
    rayHit.ray.id = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    LightIntersectContext context;
    int iIncludeRayTerminationLights = includeRayTerminationLights;
    float distance = maxDistance;
    context.mIncludeRayTerminationLights = &iIncludeRayTerminationLights;
    context.mData0 = &isect.data[0];
    context.mData1 = &isect.data[1];
    context.mDistance = &distance;
    context.mPdf = &isect.pdf;
    context.mMeshGeomID = &isect.geomID;
    context.mMeshPrimID = &isect.primID;
    context.mShadingNormal = N;
    context.mSamples = &samples;
    context.mNumHits = &numHits;
    context.mDepth = &depth;
    context.mLightIdMap = lightIdMap;

    // Call Embree intersection test
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &context.mRtcContext;

    rtcIntersect1(mRtcScene, &rayHit, &args);

    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        isect.distance = maxDistance;
        return -1;
    }

    isect.N.x      = rayHit.hit.Ng_x;
    isect.N.y      = rayHit.hit.Ng_y;
    isect.N.z      = rayHit.hit.Ng_z;
    isect.distance = *context.mDistance;
    isect.uv.x     = rayHit.hit.u;
    isect.uv.y     = rayHit.hit.v;

    return rayHit.hit.primID;
}


// Randomly intersect a ray against the LightAccelerator's list of unbounded lights.

int
LightAccelerator::intersectUnbounded(const Vec3f &P, const Vec3f &wi, float time,
        float maxDistance, bool includeRayTerminationLights, int visibilityMask, IntegratorSample1D &samples,
        int depth, LightIntersection &isect, int &numHits, const int* lightIdMap) const
{
    isect.distance = maxDistance;

    int chosenLightIdx = -1;
    for (int idx = 0; idx < mUnboundedLightCount; idx++) {
        // skip lights that do not exist in LightSet
        if (lightIdMap[idx + mBoundedLightCount] == -1) {
            continue;
        }
        const Light *light = mUnboundedLights[idx];
        LightIntersection currentIsect;

        if (!(visibilityMask & light->getVisibilityMask())) {
            // skip light if it is masked
            continue;
        }

        if (!includeRayTerminationLights && light->getIsRayTerminator()) {
            // Skip any ray termination lights if we were told not to include them
            continue;
        }

        if (light->intersect(P, nullptr /*unbounded lights do not depend on the shade point normal*/,
                wi, time, maxDistance, currentIsect)) {

            numHits++;

            if (chooseThisLight(samples, depth, numHits)) {
                chosenLightIdx = idx + mBoundedLightCount;
                isect = currentIsect;
            }
        }
    }

    return chosenLightIdx;
}


// Wrappers for the vector versions of rtcIntersect. These are called from the ISPC intersection code.
// We don't currently use the Embree ISPC API due to build system issues: Embree's ISPC build uses CMake
// to generate the various vector versions of the library; since we don't use CMake this would entail
// porting over this mechanism to our SCons-based build system, which would involve considerable CM work.
// For now, the mechanism used here offers the path of least resistance.
// See VLEN switch above for vector type information.

extern "C" void CPP_lightIntersect(RTCScene scene, RTCRayHitv& rayHitv,
    int* includeRayTerminationLights, float* isectData0, float* isectData1, SequenceID* sequenceID,
    uint32_t* totalSamples, uint32_t* sampleNumber, int* depth, float* isectDistance, int* numHits,
    float* pdf, int* meshGeomId, int* meshPrimId, const scene_rdl2::math::Vec3fv* shadingNormalv,
    const int* lightIdMap)
{
    // Create IntegratorSample1D from ispc array data
    IntegratorSample1D samples[VLEN];
    for (int i = 0; i < VLEN; ++i) {
        samples[i] = IntegratorSample1D(sequenceID[i], totalSamples[i], sampleNumber[i]);
    }

    LightIntersectContext context;
    context.mNumHits = numHits;
    context.mDepth = depth;
    context.mSamples = (const IntegratorSample1D *)&samples;
    context.mIncludeRayTerminationLights = includeRayTerminationLights;
    context.mLightIdMap = lightIdMap;

    // light type specific data passed to eval from intersect or sample
    context.mData0 = isectData0;
    context.mData1 = isectData1;
    // we get the distance to the chosen light here
    context.mDistance = isectDistance;
    // mesh light specific parameters
    context.mPdf = pdf;
    context.mMeshGeomID = meshGeomId;
    context.mMeshPrimID = meshPrimId;
    // create shading normals
    Vec3f shadingNormals[VLEN];
    for (int i = 0; i < VLEN; ++i) {
        shadingNormals[i] = Vec3f(shadingNormalv->x[i], shadingNormalv->y[i], shadingNormalv->z[i]);
    }
    context.mShadingNormal = shadingNormals;

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &context.mRtcContext;

    rtcIntersectv(sAllValidMask, scene, &rayHitv, &args);

    // need to update the sample number for the ispc SampleIntegrator1D struct
    for (int i = 0; i < VLEN; ++i) {
        sampleNumber[i] = samples[i].getSampleNumber();
    }
}

void LightAccelerator::buildSamplingTree() 
{
    mSamplingTree.build(mBoundedLights, mBoundedLightCount, mUnboundedLights, mUnboundedLightCount);
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

