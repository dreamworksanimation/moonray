// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Light.isph"
#include <moonray/rendering/pbr/core/Util.isph>

#include <moonray/rendering/shading/ispc/Util.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(SphereLight);

export uniform uint32_t
SphereLight_hudValidation(uniform bool verbose)
{
    SPHERE_LIGHT_VALIDATION;
}


//----------------------------------------------------------------------------

inline uniform float
getRadius(const uniform SphereLight * uniform light)
{
    // TODO: mb scale radiance
    return light->mLocal2RenderScale[0];
}

inline uniform float
getInvRadius(const uniform SphereLight * uniform light)
{
    // TODO: mb scale radiance
    return light->mRender2LocalScale[0];
}


inline varying Vec2f
local2uv(const varying Vec3f &local)
{
    MNRY_ASSERT(isNormalized(local));

    Vec2f uv;
    if (abs(local.y) >= 1.0f) {
        // Handle singularities at poles, to keep function consistent between
        // sample and intersect
        uv.x = 0.0f;
        uv.y = (local.y > 0.0f) ? 0.0f : 1.0f;
    } else {
        // Convert to spherical polar coords with y up.
        // Note that phi is intentionally off by pi, so that adding 0.5 to
        // uv.x brings it into the range [0,1] without using a conditional.
        float theta = dw_acos(local.y);
        float phi = atan2(-local.z, -local.x);

        // Generate (u,v)
        uv.x = phi * sOneOverTwoPi + 0.5f;
        uv.y = theta * sOneOverPi;
    }

    return uv;
}


inline Vec3f vectorFromPolarSinCosVersion(float sinTheta, float cosTheta, float phi)
{
    float sinPhi, cosPhi;
    sincos(phi, &sinPhi, &cosPhi);
    return Vec3f_ctor(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}


inline Vec3f vectorFromPolarCosVersion(float cosTheta, float phi)
{
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    return vectorFromPolarSinCosVersion(sinTheta, cosTheta, phi);
}


//----------------------------------------------------------------------------

varying bool
SphereLight_canIlluminate(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        varying float radius, const uniform LightFilterList * uniform lightFilterList)
{
    const uniform SphereLight * uniform light = (const uniform SphereLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(li->mOn);

    // Reject if sphere is completely on the backside of the point's plane
    if (isValidCullingNormal(cullingNormal)) {
        Plane pl;
        Plane_init(&pl, p, cullingNormal);
        float planeDist = Plane_getDistance(&pl, Light_getPosition(li, time));
        // mlee: we are losing a small amount of energy here by adding small value.
        float planeThreshold = sEpsilon - light->mRadius - radius;
        if (planeDist < planeThreshold) {
            return false;
        }
    }

    // Reject when completely inside an outward-facing sphere light
    if (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) {
        // mlee: we are losing a small amount of energy here by adding small value.
        float sphereThreshold = light->mRadius + sEpsilon - radius;
        Vec3f dp = p - Light_getPosition(li, time);
        if (lengthSqr(dp) < sphereThreshold * sphereThreshold) {
            return false;
        }
    }

    if (lightFilterList) {
        CanIlluminateData cid;
        cid.lightPosition = Light_getPosition(li, time);
        cid.lightRadius = getRadius(light);
        cid.shadingPointPosition = p;
        cid.lightRender2LocalXform = LocalParamLight_getXformRender2Local(
            lpl, time, LightFilterList_getNeedsLightXform(lightFilterList));
        cid.shadingPointRadius = radius;
        cid.time = time;
        return LightFilterList_canIlluminate(lightFilterList, cid);
    }

    return true;
}


// For algorithm details, see SphereLight::Intersect() in SphereLight.cc

varying bool
SphereLight_intersect(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f& cullingNormal,
        const varying Vec3f &wi, varying float time,
        varying float maxDistance, varying LightIntersection &isect)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;
    const uniform SphereLight * uniform light = (const uniform SphereLight * uniform)li;

    const Vec3f v = Light_getPosition(li, time) - p;
    const float b = dot(wi, v);
    const float v2 = dot(v, v);
    const float c = v2 - light->mRadiusSqr;

    // Do trivial rejection based on these values, according to sphere's sidedness...
    // regular sidedness: reject if ray is travelling away from sphere center OR  ray origin is INSIDE sphere
    // reverse sidedness: reject if ray is travelling away from sphere center AND ray origin is OUTSIDE sphere
    if ((c > 0.0f) ? (b <= 0.0f) : (light->mSidedness == LIGHT_SIDEDNESS_REGULAR)) {
        return false;
    }

    // Reject if discriminant is negative (ray misses sphere)
    const float D = light->mRadiusSqr - lengthSqr(v - b * wi); // More accurate version of discriminant than b^2-c
    if (D <= 0.0f) {
        return false;
    }

    // Calculate distance to intersection, reject if too big
    const float s = sqrt(D);
    // Choose near or far intersection depending on sidedness
    const float t = (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) ? b - s : b + s;
    if (t > maxDistance) {
        return false;
    }

    // Fill in isect members.
    const Vec3f normal = (t*wi - v) * getInvRadius(light);  // Outward-facing normal
    // We don't need to worry about 2-sided here because that's not supported for SphereLight
    isect.N = (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) ? normal : neg(normal);
    isect.distance = t;

    // Optionally compute uvs for texturing
    isect.uv = light->mDistribution ?
        local2uv(LocalParamLight_xformVectorRender2LocalRot(lpl, normal, time)) :
        Vec2f_ctor(0.0f);


    // Store values which eval() will later use to compute the pdf.
    // See sample() function for details about each case.
    if (c < 0.0f) {
        // Due to earlier rejection, this code only executes for reverse sidedness
        isect.data[1] = -1.0f;  // Signal uniform sampling over the solid angle
        if (isValidCullingNormal(cullingNormal)) {
            isect.data[0] = sOneOverTwoPi;
        } else {
            isect.data[0] = sOneOverFourPi;
        }
    } else {
        // This is the nastiest part of all! We need to match the pdf of the sampling scheme used in the sample()
        // function, where the samples were generated on the front-facing visible cap of the sphere and then
        // projected along the ray to the back surface in the case of reverse sidedness. So here, in the reverse
        // sidedness case, we must generate the values of t, the normal vector, and the trig expressions as if
        // the intersection point were on the front surface of the sphere.
        const float sinThetaMaxSq = c / v2;
        float cosTheta;
        if (light->mSidedness == LIGHT_SIDEDNESS_REVERSE) {
            const float t_front = b - s;
            const Vec3f normal_front = (t_front * wi - v) * light->mRcpRadius;
            cosTheta = -dot(v, normal_front) / sqrt(v2);
            isect.data[1] = t_front;
        } else {
            cosTheta = -dot(v, normal) / sqrt(v2);
            isect.data[1] = t;
        }
        isect.data[0] = cosTheta / sinThetaMaxSq;
    }

    return true;
}


varying bool
SphereLight_sample(const uniform Light * uniform li,
        const varying Vec3f &p, const varying Vec3f &cullingNormal, varying float time,
        const varying Vec3f &r, varying Vec3f &wi, varying LightIntersection &isect, varying float rayDirFootprint)
{
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;
    const uniform SphereLight * uniform light = (const uniform SphereLight * uniform)li;

    MNRY_ASSERT(li->mOn);

    const Vec3f v = Light_getPosition(li, time) - p;
    const float v2 = dot(v, v);
    const float c = v2 - light->mRadiusSqr;

    float t;
    Vec3f normal;

    // Sampling strategy is different for inside vs outside the sphere
    if (c < 0.0f) {

        // Inside sphere
        if (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) {
            // Trivial rejection when inside an outward-facing sphere light
            return false;
        }

        // Signal uniform sampling over solid angle
        isect.data[1] = -1.0f;

        if (isValidCullingNormal(cullingNormal)) {
            // Use surface normal to sample only the visible hemisphere
            Vec3f localWi = vectorFromPolarCosVersion(r.x, sTwoPi * r.y);
            ReferenceFrame frame;
            ReferenceFrame_init(frame, cullingNormal);
            wi = localToGlobal(frame, localWi);
            isect.data[0] = sOneOverTwoPi;     // pdf = 1 / (solid angle of hemisphere)
        } else {
            // No normal supplied; sample the entire sphere
            wi = vectorFromPolarCosVersion(2.0f * r.x - 1.0f, sTwoPi * r.y);
            isect.data[0] = sOneOverFourPi;    // pdf = 1 / (solid angle of full sphere)
        }

        // See comments for intersect()
        const float b = dot(wi, v);
        const float D = b*b - c;    // Precision doesn't need to be improved here because radius is large compared to v
        const float s = sqrt(D);
        t = b + s;                  // Pick far intersection because we're inside the sphere
        normal = (t*wi - v) * light->mRcpRadius;    // Outward-facing normal

    } else {

        // See Dutre's "Global Illumination Compendium" section 35, page 20.
        const float sinThetaMaxSq = c / v2;
        const float sinThetaSq = r.x * sinThetaMaxSq;
        const float sinTheta = sqrt(sinThetaSq);
        const float cosTheta = sqrt(1.0f - sinThetaSq);

        ReferenceFrame frame;
        ReferenceFrame_init(frame, normalize(neg(v)));
        const Vec3f localNormal = vectorFromPolarSinCosVersion(sinTheta, cosTheta, sTwoPi * r.y);
        normal = localToGlobal(frame, localNormal);     // Outward-facing normal

        const Vec3f rayDir = v + light->mRadius * normal;
        const float t2 = lengthSqr(rayDir);
        t = sqrt(t2);
        if (t < sEpsilon) {
            return false;
        }

        wi = rayDir / t;
        if (isValidCullingNormal(cullingNormal) && dot(cullingNormal, wi) < sEpsilon) {
            return false;
        }

        // Store values which eval() will later use to compute the pdf.
        isect.data[0] = cosTheta / sinThetaMaxSq;
        isect.data[1] = t;

        // We don't need to worry about 2-sided here because that's not supported for SphereLight
        if (light->mSidedness == LIGHT_SIDEDNESS_REVERSE) {
            // This is nasty. We use the direction of the generated sample, but continue the ray to where
            // it hits the far side of the sphere. It's done this way so the sampling scheme doesn't change,
            // because otherwise it could yield a different image when sample clamping is active. Although
            // no prior production shots used SphereLights with reverse sidedness, we want the result to match
            // that of a SphereLight with regular sidedness when there is no geometry inside the SphereLight.
            // Also, it is important to adjust the values of t and normal AFTER stashing the values in isect.data
            // which will be used by eval() to compute the pdf. This is because the pdf comes from a distribution
            // of points on the front-facing surface of the sphere.
            const float b = dot(wi, v);
            const float D = light->mRadiusSqr - lengthSqr(v - b * wi); // More accurate version of discrim than b^2-c
            t = b + sqrt(max(D, 0.0f));                 // Rounding can cause D to go slight negative, hence the max()
            normal = (t*wi - v) * light->mRcpRadius;    // Outward-facing normal
        }
    }

    isect.N = (light->mSidedness == LIGHT_SIDEDNESS_REGULAR) ? normal : neg(normal);
    isect.distance = t;

    // Optionally compute uvs for texturing
    isect.uv = light->mDistribution ?
        local2uv(LocalParamLight_xformVectorRender2LocalRot(lpl, normal, time)) :
        Vec2f_ctor(0.0f);

    return true;
}

varying Color
SphereLight_eval(const uniform Light * uniform li, uniform ShadingTLState * uniform tls,
        const varying Vec3f &wi, const varying Vec3f &p, const varying LightFilterRandomValues& filterR, varying float time,
        const varying LightIntersection &isect, uniform bool fromCamera,
        const uniform LightFilterList * uniform lightFilterList, varying float rayDirFootprint,
        varying float * uniform pdf)
{
    const uniform SphereLight * uniform light = (const uniform SphereLight * uniform)li;
    const uniform LocalParamLight * uniform lpl = (const uniform LocalParamLight * uniform) li;

    MNRY_ASSERT(light->mOn);

    // TODO: Use proper filtering with ray differentials and mip-mapping.
    Color radiance = light->mRadiance;

    if (light->mDistribution) {
        // Note: the distribution doesn't contribute to the pdf, since we ignore it when sampling
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

    // To evaluate the pdf, we need information about which sampling scheme was used (or, in the case of the
    // intersect() function, which one would have been used for the given geometric config) -
    // (a) point outside sphere, cosine-weighted sampling over front-facing surface
    // (b) point inside sphere, uniform sampling over hemisphere above local horizon
    // (c) point inside sphere, uniform sampling over full sphere
    // Cases (b) and (c) only apply to reverse sidedness. In case (a), whether normal or reverse sidedness, 
    // we use isect.data[0] and [1] to pass a pair of intermediate values through to this point for computing
    // the pdf. In cases (b) and (c), we pass the pdf directly in isect.data[0], and use a negative value in
    // isect.data[1] to signal this.
    if (pdf) {
        float t = isect.data[1];
        if (t >= 0.0f) {
            // See Dutre's "Global Illumination Compendium" section 35, page 20.
            float costThetaOverSinThetaSq = isect.data[0];
            MNRY_ASSERT(isfinite(costThetaOverSinThetaSq));
            *pdf = costThetaOverSinThetaSq / (sPi * light->mRadiusSqr);
            *pdf *= areaToSolidAngleScale(wi, isect.N, t);
            MNRY_ASSERT(isfinite(*pdf));
        } else {
            // t<0 is used to signal that isect.data[0] stores the pdf directly
            *pdf = isect.data[0];
        }
    }

    return radiance;
}

inline uniform float SphereLight_getThetaO(const uniform Light * uniform li) { return sPi; }
inline uniform float SphereLight_getThetaE(const uniform Light * uniform li) { return sPi * 0.5f; }

//----------------------------------------------------------------------------

export void
SphereLight_init(uniform SphereLight * uniform light, uniform bool uniformSampling)
{
    Light_init((uniform Light * uniform)light, SphereLight_canIlluminate,
                SphereLight_intersect, SphereLight_sample, SphereLight_eval,
                SphereLight_getThetaO, SphereLight_getThetaE);
}

//----------------------------------------------------------------------------

