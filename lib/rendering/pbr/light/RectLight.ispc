// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include <moonray/rendering/pbr/core/Util.isph>

#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(RectLight);

export uniform uint32_t
RectLight_hudValidation(uniform bool verbose)
{
    RECT_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

static void
RectLight_computeCorners(const uniform RectLight * uniform light,
                         varying Vec3f * uniform corners, varying float time)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) light;

    corners[0] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(light->mHalfWidth,  light->mHalfHeight, 0.f), time);
    corners[1] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(light->mHalfWidth, -light->mHalfHeight, 0.f), time);
    corners[2] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(-light->mHalfWidth, light->mHalfHeight, 0.f), time);
    corners[3] = LocalParamLight_xformPointLocal2Render(lpl, (varying Vec3f) Vec3f_ctor(-light->mHalfWidth, -light->mHalfHeight, 0.f), time);
}

//----------------------------------------------------------------------------

static varying float
planeDistance(const uniform RectLight * uniform light,
              const varying Vec3f &p, varying float time)
{
    const uniform Light * uniform li = (const uniform Light * uniform) light;
    Plane renderPlane;
    if (light->mMb) {
        Plane_init(&renderPlane, Light_getPosition(li, time), Light_getDirection(li, time));
    } else {
        renderPlane = light->mRenderPlane;
    }

    float distance = Plane_getDistance(&renderPlane, p);
    if (light->mSidedness == LIGHT_SIDEDNESS_REVERSE) {
        distance = -distance;
    } else if (light->mSidedness == LIGHT_SIDEDNESS_2_SIDED) {
        distance = abs(distance);
    }

    return distance;
}

// Get region of overlap between RectLight and square region of influence.
// The square is centered at localP and has a side length 2 * localLength. 
// We want to get the center of the rectangular bound and its width and height.
// Returns true if the region overlaps with the light.
bool
RectLight_getOverlapBounds(const uniform RectLight * uniform light, const varying Vec2f& localP, 
        varying float localLength, varying Vec2f& center, varying float& width, varying float& height)
{
    float minX = max(localP.x - localLength, -light->mHalfWidth);
    float maxX = min(localP.x + localLength, light->mHalfWidth);
    float minY = max(localP.y - localLength, -light->mHalfHeight);
    float maxY = min(localP.y + localLength, light->mHalfHeight);

    if (maxX <= minX || maxY <= minY) {
        // region does not overlap with RectLight
        return false;
    }

    center.x = 0.5f * (maxX + minX);
    center.y = 0.5f * (maxY + minY);
    width = maxX - minX;
    height = maxY - minY;

    return true;
}

varying bool
RectLight_canIlluminate(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        varying float radius, const uniform LightFilterList * uniform lightFilterList)
{
    const uniform RectLight * uniform light = (const uniform RectLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(li->mOn);

    // Cull points which are completely on the backside of the light, taking sidedness into account
    const float threshold = sEpsilon - radius;
    if ((light->mSidedness != LIGHT_SIDEDNESS_2_SIDED) && (planeDistance(light, p, time) < threshold)) {
        return false;
    }

    // Cull lights which are completely on the backside of the point.
    if (isValidCullingNormal(cullingNormal)) {
        Plane pl;
        Plane_init(&pl, p, cullingNormal);
        varying float d0, d1, d2, d3;
        if (!light->mMb) {
            d0 = Plane_getDistance(&pl, light->mRenderCorners[0]);
            d1 = Plane_getDistance(&pl, light->mRenderCorners[1]);
            d2 = Plane_getDistance(&pl, light->mRenderCorners[2]);
            d3 = Plane_getDistance(&pl, light->mRenderCorners[3]);
        } else {
            varying Vec3f corners[4];
            RectLight_computeCorners(light, corners, time);
            d0 = Plane_getDistance(&pl, corners[0]);
            d1 = Plane_getDistance(&pl, corners[1]);
            d2 = Plane_getDistance(&pl, corners[2]);
            d3 = Plane_getDistance(&pl, corners[3]);
        }
        bool canIllum = d0 > threshold || d1 > threshold ||
                        d2 > threshold || d3 > threshold;
        if (!canIllum) return false;
    }

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = Light_getPosition(li, time);
        cid.lightRadius = LocalParamLight_xformLocal2RenderScale(lpl, sqrt(light->mWidth * light->mWidth +
            light->mHeight * light->mHeight) / 2, time);
        cid.shadingPointPosition = p;
        cid.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        cid.shadingPointRadius = radius;
        cid.time = time;
        return LightFilterList_canIlluminate(lightFilterList, cid);
    }

    return true;
}

varying bool
RectLight_intersect(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f& cullingNormal,
        const varying Vec3f &wi, varying float time,
        varying float maxDistance, varying LightIntersection &isect)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform RectLight * uniform light = (const uniform RectLight * uniform)li;

    // Get local direction and position
    Vec3f localWi = LocalParamLight_xformVectorRender2LocalRot(lpl, wi, time);
    Vec3f localP = LocalParamLight_xformVectorRender2Local(lpl, p - Light_getPosition(li, time), time);

    // Trivial rejection of pos/dir/sidedness cases
    if (Light_rejectPosDirSidedness(li, localP.z, localWi.z)) return false;

    // Reject if intersection with plane is beyond maxDistance
    float localDistance = -localP.z / localWi.z;
    float renderDistance = LocalParamLight_xformLocal2RenderScale(lpl, localDistance, time);
    if (renderDistance > maxDistance) {
        return false;
    }

    // Reject if intersection is outside rectangle
    Vec3f localHit = localP + localWi * localDistance;
    if (abs(localHit.x) >= light->mHalfWidth || abs(localHit.y) >= light->mHalfHeight) {
        return false;
    }

    if (light->mSpread < 1.f) {
        // Reject if interseciton point is outside of spread
        Vec3f absLocalWi = abs(localWi);
        if (max(absLocalWi.x, absLocalWi.y) > absLocalWi.z * light->mTanSpreadTheta) {
            return false;
        }

        // compute pdf
        // get half side length of square spread region
        float spreadLength = localP.z * light->mTanSpreadTheta;
        Vec2f center;
        float width, height;
        RectLight_getOverlapBounds(light, Vec2f_ctor(localP.x, localP.y), spreadLength, center, width, height);
        // store pdf if not rejected
        isect.pdf = 1.f / (LocalParamLight_xformLocal2RenderScale(lpl, width, time) *
                           LocalParamLight_xformLocal2RenderScale(lpl, height, time));
    }

    // Fill in isect members.
    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. The correct normal is obtained by ensuring the ray is travelling against the normal,
    // i.e. the ray's direction has negative local z-coordinate.
    Vec3f normal = Light_getDirection(li, time);
    isect.N = (localWi.z < 0.0f) ? normal : neg(normal);
    isect.uv = (light->mDistribution  ?  asAVec2f(localHit) * light->mUvScale + light->mUvOffset
                                       :  Vec2f_ctor(0.0f));
    isect.distance = renderDistance;

    return true;
}

varying bool
RectLight_sample(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time, const varying Vec3f &r,
        varying Vec3f &wi, varying LightIntersection &isect, varying float rayDirFootprint)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform)li;
    const uniform RectLight * uniform light = (const uniform RectLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    // We don't cast light on points behind the light. Also we can't always
    // rely on canIlluminate() to test for that case, as it's an optional
    // culling function. canIlluminate() can also be passed a non-zero radius,
    // defining a p-region partially in front and partially behind the light.
    // So we have to test for it per sample, here too.
    float distance = planeDistance(light, p, time);
    if ((light->mSidedness != LIGHT_SIDEDNESS_2_SIDED) && (distance < sEpsilon)) {
        return false;
    }

    Vec3f localHit;
    
    // There are 3 possible sampling modes:
    // 1) If there is an image distribution, sample that.
    // 2) If there is no image distribution, and the spread parameter is less than 1 and
    //    we are in a good region to sample the spread, then sample that region.
    // 3) Otherwise, sample the shape of the light.
    if (light->mDistribution) {
        // Sample the image distribution if any
        ImageDistribution_sample(light->mDistribution, r.x, r.y, 0, &isect.uv, nullptr, light->mTextureFilter);
        localHit = Vec3f_ctor((isect.uv.x - light->mUvOffset.x) / light->mUvScale.x,
                         (isect.uv.y - light->mUvOffset.y) / light->mUvScale.y,
                         0.0f);

        // reject samples that are outside of the spread region.
        if (light->mSpread < 1.f) {
            Vec3f localP = LocalParamLight_xformPointRender2Local(lpl, p, time);
            Vec3f localWi = localHit - localP;
            Vec3f absLocalWi = abs(localWi);
            if (max(absLocalWi.x, absLocalWi.y) > absLocalWi.z * light->mTanSpreadTheta) {
                return false;
            }
        }

    } else if (light->mSpread < 1.f) {
        // Uniformly sample the rectangular region of overlap between the spread region and the RectLight.
        
        // get p in local space
        Vec3f localP = LocalParamLight_xformPointRender2Local(lpl, p, time);

        // transform unit circle to the spread region in the light's local space
        float spreadLength = distance * light->mTanSpreadTheta;
        float localSpreadLength = LocalParamLight_xformRender2LocalScale(lpl, spreadLength, time);

        Vec2f center;
        float width, height;
        if (!RectLight_getOverlapBounds(light, Vec2f_ctor(localP.x, localP.y), localSpreadLength, 
            center, width, height)) {
            return false;
        }

        // sample rectangular region of influence
        localHit = Vec3f_ctor((0.5f - r.x) * width + center.x,
                         (0.5f - r.y) * height + center.y,
                         0.0f);

        isect.uv = Vec2f_ctor(0.f);
        // store pdf
        isect.pdf = 1.f / (LocalParamLight_xformLocal2RenderScale(lpl, width, time) *
                           LocalParamLight_xformLocal2RenderScale(lpl, height, time));

    } else {
        // Otherwise uniformly distribute samples on rectangle area
        isect.uv = Vec2f_ctor(0.0f);
        localHit = Vec3f_ctor((0.5f - r.x) * light->mWidth,
                         (0.5f - r.y) * light->mHeight,
                         0.0f);
    }

    Vec3f renderHit = LocalParamLight_xformPointLocal2Render(lpl, localHit, time);

    // Compute wi and d
    wi = renderHit - p;
    isect.distance = length(wi);
    if (isect.distance < sEpsilon) {
        return false;
    }
    wi = wi * rcp(isect.distance);
    if (isValidCullingNormal(cullingNormal)  &&  dot(cullingNormal, wi) < sEpsilon) {
        return false;
    }

    // We will only reach this far if p is on the illuminated side of the light's plane (either side, for 2-sided).
    // The light's mDirection is its local z-axis, which is only guaranteed to concide with the normal for lights
    // with regular sidedness. The correct normal is obtained by ensuring the ray is travelling against the normal,
    // i.e. the ray's direction has negative dot product with it.
    Vec3f normal = Light_getDirection(li, time);
    isect.N = (dot(normal, wi) < 0.0f) ? normal : neg(normal);

    return true;
}

varying Color
RectLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls,
        const varying Vec3f &wi, const varying Vec3f &p, const varying LightFilterRandomValues& filterR, varying float time,
        const varying LightIntersection &isect, uniform bool fromCamera,
        const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
        varying float * uniform pdf)
{
    const uniform RectLight * uniform light = (const uniform RectLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(li->mOn);

    Color radiance = light->mRadiance;
    
    if (light->mSpread < 1.f) {
        // modify radiance as if it were physically blocked by an eggcrate light filter.
        Vec3f localWi = LocalParamLight_xformVectorRender2LocalRot(lpl, wi, time);
        radiance = radiance * max(1.f - abs(localWi.x / (localWi.z * light->mTanSpreadTheta)), 0.f) * 
                              max(1.f - abs(localWi.y / (localWi.z * light->mTanSpreadTheta)), 0.f);
    }

    if (light->mDistribution) {
        // Point sample the texture if any
        // TODO: Use proper filtering with ray differentials and mip-mapping.
        radiance = radiance * ImageDistribution_eval(light->mDistribution,
                isect.uv.x, isect.uv.y, 0, light->mTextureFilter);
    }

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
        if (light->mDistribution) {
            *pdf = light->mInvArea * ImageDistribution_pdf(light->mDistribution,
                                                           isect.uv.x, isect.uv.y, 0, light->mTextureFilter);
        } else if (light->mSpread < 1.f) {
            // we sampled the spread region, grab the stored pdf.
            *pdf = isect.pdf;
        } else {
            // we uniformly sampled the rectangle
            *pdf = light->mInvArea;
        }
        *pdf *= areaToSolidAngleScale(wi, isect.N, isect.distance);
    }

    return radiance;
}

inline uniform float RectLight_getThetaO(const uniform Light * uniform li)
{
    const uniform RectLight * uniform light = (const uniform RectLight * uniform)li;
    return sPi * 0.5f * light->mSpread; 
}
inline uniform float RectLight_getThetaE(const uniform Light * uniform li) { return sPi * 0.5f; }

//----------------------------------------------------------------------------

export void
RectLight_init(uniform RectLight * uniform light, uniform bool uniformSampling)
{
    Light_init((uniform Light * uniform)light, RectLight_canIlluminate,
                RectLight_intersect, RectLight_sample, RectLight_eval,
                RectLight_getThetaO, RectLight_getThetaE);
}

//----------------------------------------------------------------------------

