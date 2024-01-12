// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "MeshLight.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/shading/ispc/Shading.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

#include <embree4/rtcore.isph>
#include <embree4/rtcore_ray.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(MeshLight);

export uniform uint32_t
MeshLight_hudValidation(uniform bool verbose)
{
    MESH_LIGHT_VALIDATION;
}

//----------------------------------------------------------------------------

uniform bool
Node_isLeaf(uniform Node& node)
{
    return  node.mFace != nullptr;
}

varying bool
Node_isLeaf(const uniform Node * varying node)
{
    return  node->mFace != nullptr;
}

uniform Vec3f
Node_getCenter(uniform Node& node)
{
    if (Node_isLeaf(node)) {
        return node.mFace->mCenter;
    } else {
        return 0.5f * (node.mBbox.lower + node.mBbox.upper);
    }
}

varying Vec3f
Node_getCenter(const uniform Node * varying node)
{
    if (Node_isLeaf(node)) {
        return node->mFace->mCenter;
    } else {
        return 0.5f * (node->mBbox.lower + node->mBbox.upper);
    }
}

// thetaE describes the angle of influence of a light's emission.
// For a flat polygon in a mesh, thetaE is pi / 2. If there are multiple
// polygons in a light, thetaE remains pi / 2.
static const uniform float sThetaE = sHalfPi;
// thetaO describes the angular spread of a light's orientation.
// For a flat triangle in a mesh, thetaO is 0. If there are multiple
// triangles in a light, thetaO is the angular spread of the normals of the
// triangles.
static const uniform float sThetaO = 0.0f;


//----------------------------------------------------------------------------

varying unsigned int
MeshLight_getFaceVertexCount(const uniform MeshLight * uniform light, const uniform Face * varying face)
{
    return light->mFaceVertexCountPtr[face->mGeomID];
}

const varying Vec3f
MeshLight_getFaceVertex(const uniform MeshLight * uniform light,
                        const uniform Face * varying face,
                        uniform size_t index,
                        varying float time)
{
    size_t vertexOffset = light->mVertexOffsetPtr[face->mGeomID] + face->mIndices[index] * light->mMbSteps;

    if (!light->mDeformationMb) {
        return light->mVerticesPtr[vertexOffset];
    }

    // Special case for time == 1.0f, because otherwise there will be a segfault
    // when trying to access v2 below. Technically time should never == 1.0f during
    // mcrt.
    MNRY_ASSERT(time < 1.0f);
    if (time == 1.0f) {
        return light->mVerticesPtr[vertexOffset + light->mMbSteps - 1];
    }

    uniform float invDt = light->mMbSteps - 1;
    varying float d = time * invDt;
    varying float step = floor(d);
    varying float t = d - step;

    const varying Vec3f v1 = light->mVerticesPtr[vertexOffset + step];
    const varying Vec3f v2 = light->mVerticesPtr[vertexOffset + step + 1];

    return lerp(v1, v2, t);
}

varying bool
MeshLight_canIlluminate(const uniform Light * uniform li,
                        const varying Vec3f &p,
                        const varying Vec3f &cullingNormal,
                        varying float time,
                        varying float radius,
                        const uniform LightFilterList * uniform lightFilterList)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform MeshLight * uniform light = (const uniform MeshLight * uniform)li;
    MNRY_ASSERT(li->mOn);

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = LocalParamLight_xformPointLocal2Render(lpl, Node_getCenter(light->mBVHPtr[0]), time);
        float lightRadius = length(light->mBVHPtr[0].mBbox.upper - light->mBVHPtr[0].mBbox.lower) / 2;
        cid.lightRadius = LocalParamLight_xformLocal2RenderScale(lpl, lightRadius, time);
        cid.shadingPointPosition = p;
        cid.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        cid.shadingPointRadius = radius;
        cid.time = time;
        return LightFilterList_canIlluminate(lightFilterList, cid);
    }

    return true;
}

varying float
MeshLight_importance(const varying Vec3f& shadingPoint,
                     const varying Vec3f * uniform shadingNormal,
                     const uniform Node * varying node)
{
    // Early out if this node has no energy. This would happen if the MapShader
    // is black for all the faces contained in this node.
    if (node->mEnergy == 0.0f) {
        return 0.0f;
    }

    Vec3f center = Node_getCenter(node);
    // the vector from the shadingPoint to the center of the node
    Vec3f point2center = center - shadingPoint;
    float distance2 = lengthSqr(point2center);

    // orientation metric
    // For more deatils see Kulla (2017) p. 8 Fig. 7 and Eq. 3.
    float orientation = 1.0f;
    if (!isEqual(node->mBcone.mThetaO, sPi) && distance2 != 0) {
        float distance = sqrt(distance2);
        point2center = point2center / distance;
        // thetaU is the uncertainty angle from the center of the box to its edge
        float thetaU = 0.0f;
        // radius of the sphere that circumscribes the bounding box
        float radius = length(node->mBbox.upper - center);

        if (distance >= radius) {
            // the shading point is outside the bounding sphere
            thetaU = asin(radius / distance);
        } else {
            // shading point is inside the bounding sphere
            thetaU = sPi;
        }
        // thetaI is the angle between the axis and the shading normal
        float thetaI = 0.0f;
        float shadingNormalContribution = 1.0f;
        if (shadingNormal) {
            const varying Vec3f normal = *shadingNormal;
            thetaI = dw_acos(clamp(dot(point2center, normal), -1.0f, 1.0f));
            shadingNormalContribution = abs(cos(max(thetaI - thetaU, 0.0f)));
        }
        // theta is the angle between the bounding cone axis and the center to shading point axis
        float theta = dw_acos(-dot(node->mBcone.mAxis, point2center));
        // thetaPrime is the minimum possible angle between edge of the bounding cone and the
        // center to shading point axis
        float thetaPrime = max(theta - node->mBcone.mThetaO - thetaU, 0.0f);
        // For the mesh light, thetaE is always the same. For the many lights
        // solution we would need to compare against node->mBcone.mThetaE
        bool useThetaPrime = thetaPrime < sThetaE;
        orientation = shadingNormalContribution * (useThetaPrime ? cos(thetaPrime) : 0.0f);
    }

    return node->mEnergy * orientation / distance2;
}

varying int
MeshLight_drawSampleRecurse(const uniform MeshLight * uniform light,
                            const varying Vec3f& shadingPoint,
                            const varying Vec3f * uniform shadingNormal,
                            varying int currentNodeIndex,
                            varying float u,
                            varying float& pdf)
{
    //print("%\n", u);
    const uniform Node * varying node = &light->mBVHPtr[currentNodeIndex];
    if (Node_isLeaf(node)) {
        // The face is selected. Multiply by the inverse area of the face to get
        // the final pdf = pdf(face) * pdf(point | face)
        pdf *= node->mFace->mInvArea;
        return currentNodeIndex;
    }

    int32_t leftIndex = currentNodeIndex + 1;
    const uniform Node * varying leftChild = &light->mBVHPtr[leftIndex];
    int32_t rightIndex = node->mRightChildIndex;
    const uniform Node * varying rightChild = &light->mBVHPtr[rightIndex];

    float leftImportance = MeshLight_importance(shadingPoint, shadingNormal, leftChild);
    float rightImportance = MeshLight_importance(shadingPoint, shadingNormal, rightChild);

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
        // traverse left branch. Rescale random number u.
        const varying float uRemapped = u / pdfL;
        pdf *= pdfL;
        return MeshLight_drawSampleRecurse(light, shadingPoint, shadingNormal, leftIndex, uRemapped, pdf);
    } else {
        // traverse right branch. Rescale random number u.
        const varying float uRemapped = (u - pdfL) / pdfR;
        pdf *= pdfR;
        return MeshLight_drawSampleRecurse(light, shadingPoint, shadingNormal, rightIndex, uRemapped, pdf);
    }
}

varying float
MeshLight_getPdfOfFace(const uniform MeshLight * uniform light,
                       varying int32_t nodeID,
                       const varying Vec3f& p,
                       const varying Vec3f * uniform n)
{
    // compute pdf by traversing backwards through the bvh and computing
    // the importance of each traversed node and its sibling.
    MNRY_ASSERT(Node_isLeaf(&light->mBVHPtr[nodeID]));
    float pdf = 1.0f;
    uint32_t currNodeIndex = nodeID;
    while (currNodeIndex > 0) {
        const uniform Node * varying currNode = &light->mBVHPtr[currNodeIndex];
        MNRY_ASSERT(currNode->mParentIndex < light->mBVHSize ||
                  (currNode->mParentIndex == -1 && currNodeIndex == 0));
        uniform Node * varying siblingNode;
        const uniform Node * varying parentNode = &light->mBVHPtr[currNode->mParentIndex];
        if (parentNode->mRightChildIndex == currNodeIndex) {
            // currNode is the right child, sibling is the left child
            siblingNode = &light->mBVHPtr[currNode->mParentIndex + 1];
        } else {
            // currNode is the left child, sibling is the right child
            MNRY_ASSERT(currNodeIndex == currNode->mParentIndex + 1);
            siblingNode = &light->mBVHPtr[parentNode->mRightChildIndex];
        }

        float currImportance = MeshLight_importance(p, n, currNode);
        float siblingImportance = MeshLight_importance(p, n, siblingNode);

        MNRY_ASSERT(currImportance >= 0.0f);
        MNRY_ASSERT(siblingImportance >= 0.0f);

        if (currImportance == 0.0f && siblingImportance == 0.0f) {
            // edge case
            pdf *= 0.5;
        } else {
            pdf *= currImportance / (currImportance + siblingImportance);
        }

        currNodeIndex = currNode->mParentIndex;
    }

    return pdf;
}

varying bool
MeshLight_intersect(const uniform Light * uniform li,
                    const varying Vec3f &p,
                    const varying Vec3f& cullingNormal,
                    const varying Vec3f &wi,
                    varying float time,
                    varying float maxDistance,
                    varying LightIntersection &isect)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;
    const uniform MeshLight * uniform light = (const uniform MeshLight * uniform)li;

    RTCRayHit rayHit;

    // transform ray origin and direction from render space to the light's local
    // space because the vertex buffer is in the light's local space
    const varying Vec3f transformedP = LocalParamLight_xformPointRender2Local(lpl, p, time);
    const varying Vec3f transformedWi = LocalParamLight_xformVectorRender2LocalRot(lpl, wi, time);

    rayHit.ray.org_x = transformedP.x;
    rayHit.ray.org_y = transformedP.y;
    rayHit.ray.org_z = transformedP.z;
    rayHit.ray.tnear = 0.0f;
    rayHit.ray.dir_x = transformedWi.x;
    rayHit.ray.dir_y = transformedWi.y;
    rayHit.ray.dir_z = transformedWi.z;
    rayHit.ray.time  = time;
    rayHit.ray.tfar  = maxDistance;
    rayHit.ray.mask  = (unsigned int32_t)(-1);
    rayHit.ray.id = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    uniform RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;

    rtcIntersectV(light->mRtcScene, &rayHit, &args);

    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }

    // compute pdf of this intersection.
    // pdf = pdf(point | face) * pdf(face)
    int faceOffset = light->mFaceOffsetPtr[rayHit.hit.geomID] + rayHit.hit.primID;
    int nodeIndex = light->mPrimIDToNodeIDPtr[faceOffset];
    float pdf = 1.0f;
    if (isValidCullingNormal(cullingNormal)) {
        // transform shading point's normal from render space to the light's local space.
        const Vec3f transformedN = LocalParamLight_xformNormalRender2LocalRot(lpl, cullingNormal, time);
        pdf = light->mBVHPtr[nodeIndex].mFace->mInvArea *
            MeshLight_getPdfOfFace(light, nodeIndex, transformedP, &transformedN);
    } else {
        pdf = light->mBVHPtr[nodeIndex].mFace->mInvArea *
            MeshLight_getPdfOfFace(light, nodeIndex, transformedP, nullptr);
    }

    // Fill in isect members
    Vec3f normal = light->mBVHPtr[nodeIndex].mFace->mNormal;
    isect.N = LocalParamLight_xformNormalLocal2RenderRot(lpl, normal, time);
    isect.uv = Vec2f_ctor(rayHit.hit.u, rayHit.hit.v);
    isect.distance = LocalParamLight_xformLocal2RenderScale(lpl, rayHit.ray.tfar, time);
    isect.pdf = pdf;
    isect.primID = rayHit.hit.primID;
    isect.geomID = rayHit.hit.geomID;

    return true;
}

varying bool
MeshLight_sample(const uniform Light * uniform li,
                 const varying Vec3f &p,
                 const varying Vec3f &cullingNormal,
                 varying float time,
                 const varying Vec3f &r,
                 varying Vec3f &wi,
                 varying LightIntersection &isect,
                 varying float rayDirFootprint)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform MeshLight * uniform light = (const uniform MeshLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    // TODO: get good 3D sample and use 3rd D to traverse BVH. Currently we use r1 to traverse BVH and
    // then reuse it to sample the face.
    const varying float r1 = r.x;
    const varying float r2 = r.y;
    const varying float r3 = r.z;

    float pdf = 1.0f;
    int faceIndex;
    // transform shading point's position and normal from render space to the
    // light's local space, because the vertex buffer is in the light's local space.
    const Vec3f transformedP = LocalParamLight_xformPointRender2Local(lpl, p, time);

    if (isValidCullingNormal(cullingNormal)) {
        const Vec3f transformedN = LocalParamLight_xformNormalRender2LocalRot(lpl, cullingNormal, time);
        faceIndex = MeshLight_drawSampleRecurse(light, transformedP, &transformedN, 0, r3, pdf);
    } else {
        faceIndex = MeshLight_drawSampleRecurse(light, transformedP, nullptr, 0, r3, pdf);
    }

    MNRY_ASSERT(isfinite(pdf) && pdf >= 0.0f);
    MNRY_ASSERT(Node_isLeaf(&light->mBVHPtr[faceIndex]));

    const uniform Face * varying face = light->mBVHPtr[faceIndex].mFace;

    Vec2f uv;
    Vec3f hit;
    Vec3f normal;

    if (MeshLight_getFaceVertexCount(light, face) == 3) {
        // triangle or degenerate quad case
        Vec3f p1 = MeshLight_getFaceVertex(light, face, 0, time);
        Vec3f p2 = MeshLight_getFaceVertex(light, face, 1, time);
        Vec3f p3 = MeshLight_getFaceVertex(light, face, 2, time);

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

        if (light->mDeformationMb) {
            // We cannot directly use face.mNormal here because it is the normal at
            // time = centroidTime. Therefore we compute it here.
            normal = normalize(cross(p2 - p1, p3 - p1));
        } else {
            normal = face->mNormal;
        }

        if (light->mMapShader) {
            // get barycentric coordinate for hit
            uv.x = u;
            uv.y = v;
        }
    } else {
        // quad case
        MNRY_ASSERT(MeshLight_getFaceVertexCount(light, face) == 4);

        Vec3f v0 = MeshLight_getFaceVertex(light, face, 0, time);
        Vec3f v1 = MeshLight_getFaceVertex(light, face, 1, time);
        Vec3f v2 = MeshLight_getFaceVertex(light, face, 2, time);
        Vec3f v3 = MeshLight_getFaceVertex(light, face, 3, time);

        Vec3f normal013 = cross(v1 - v0, v3 - v0);
        Vec3f normal231 = cross(v3 - v2, v1 - v2);
        float area013 = length(normal013);
        float area231 = length(normal231);
        float areaSum = area013 + area231;
        float rSplit = area013 / areaSum;

        Vec3f p1;
        Vec3f p2;
        Vec3f p3;
        float u;

        // Choose which triangle according to the area of the triangles
        if (r1 < rSplit) {
            // first triangle
            MNRY_ASSERT(area013 > 0);
            u = r1 / rSplit;
            p1 = v0;
            p2 = v1;
            p3 = v3;
            normal = normal013 / area013;
        } else {
            // second triangle
            MNRY_ASSERT(area231 > 0);
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

        if (light->mMapShader) {
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

    MNRY_ASSERT(isfinite(hit.x) && isfinite(hit.y) && isfinite(hit.z));

    // compute wi and hit distance
    wi = LocalParamLight_xformVectorLocal2Render(lpl, hit - transformedP, time);
    isect.distance = length(wi);
    if (isect.distance > sEpsilon) {
        wi = wi * rcp(isect.distance);
    }

    // light is on other side of shading point
    if (isValidCullingNormal(cullingNormal) && dot(cullingNormal, wi) < sEpsilon) {
        return false;
    }

    isect.N = LocalParamLight_xformNormalLocal2RenderRot(lpl, normal, time);
    isect.pdf = pdf;
    isect.primID = face->mPrimID;
    isect.geomID = face->mGeomID;
    isect.uv = uv;

    return true;
}


varying Color
MeshLight_eval(const uniform Light * uniform li,
               uniform ShadingTLState * uniform tls,
               const varying Vec3f &wi,
               const varying Vec3f &p,
               const varying LightFilterRandomValues &filterR,
               varying float time,
               const varying LightIntersection &isect,
               uniform bool fromCamera,
               const uniform LightFilterList * uniform lightFilterList,
               varying float rayDirFootprint,
               varying float * uniform pdf)
{
    const uniform MeshLight * uniform light = (const uniform MeshLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(li->mOn);

    Color radiance = light->mRadiance;

    if (lightFilterList) {
        EvalData evalData;
        evalData.tls = tls;
        evalData.isect = &isect;
        evalData.lightPosition = Light_getPosition(li, time);
        evalData.lightDirection = Light_getDirection(li, time);
        evalData.shadingPointPosition = p;
        evalData.randVar = filterR;
        evalData.time = time;
        evalData.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        evalData.wi = wi;

        LightFilterList_eval(lightFilterList, evalData, &radiance);
    }

    if (pdf) {
        *pdf = isect.pdf;
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
        if (*pdf == 0.f) {
            // we don't need to sample the map shader if the pdf is 0.
            return radiance;
        }
    }

    // sample the texture
    // TODO: Use texture derivatives
    if (light->mMapShader) {
        // Initialize shading state
        varying State state;
        CPP_initShadingState(light, tls, &isect.geomID, &isect.primID, &p.x, &isect.N.x, &isect.uv.x, lanemask(),
            state);

        // we have a binding, find the sample function and evaluate it
        int64 * uniform ptr =
            (int64 * uniform) *(light->mMapShader + (MAP_SAMPLEFUNCV_OFFSET / 8));

        // if the function pointer is non-null, evaluate the function
        if (ptr != NULL) {
            // call and time the sample function.
#define SHADING_BRACKET_TIMING_ENABLED
#ifdef SHADING_BRACKET_TIMING_ENABLED
            const uniform int64 start = getInclusiveTime(light->mMapShader, tls);
#endif
            SamplePtr sampleFn = (SamplePtr) ptr;
            varying Color result;
            sampleFn((const Map * uniform) light->mMapShader, tls, &state, &result);

#ifdef SHADING_BRACKET_TIMING_ENABLED
            const uniform int64 end = getInclusiveTime(light->mMapShader, tls);
            subtractTimeInChild((const SceneObject * uniform)light->mMapShader, tls, start, end);
#endif
            radiance = radiance * result;
        }

        // reset mAttributeOffsets so that geometry can use the offet for the attribute table on the main Layer
        tls->mAttributeOffsets = nullptr;
    }

    return radiance;
}

// TODO: update with correct values once MeshLight is supported with adaptive light sampling
inline uniform float MeshLight_getThetaO(const uniform Light * uniform li) { return 0.f; }
inline uniform float MeshLight_getThetaE(const uniform Light * uniform li) { return 0.f; }

//----------------------------------------------------------------------------

export void
MeshLight_init(uniform MeshLight * uniform light, uniform bool uniformSampling)
{
    Light_init((uniform Light * uniform)light, MeshLight_canIlluminate,
        MeshLight_intersect, MeshLight_sample, MeshLight_eval,
        MeshLight_getThetaO, MeshLight_getThetaE);
}


//----------------------------------------------------------------------------

