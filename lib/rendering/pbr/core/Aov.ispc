// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Aov.ispc

#include "Aov.isph"

#include <moonray/rendering/pbr/integrator/BsdfSampler.isph>
#include "Util.isph"

#include <moonray/rendering/bvh/shading/ispc/Intersection.isph>
#include <moonray/rendering/bvh/shading/ispc/State.isph>
#include <moonray/rendering/lpe/StateMachine.isph>
#include <moonray/rendering/shading/ispc/AovLabels.isph>
#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/bssrdf/Bssrdf.isph>

#include <scene_rdl2/common/math/ispc/Mat4.isph>
#include <scene_rdl2/common/math/ispc/ReferenceFrame.isph>
#include <scene_rdl2/common/math/ispc/Vec3.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

// ---------------------------------------------------------------------------

// utility function to place a color into a uniform float array
inline void
storeResult(const varying Color &result,
            uniform float * uniform dest)
{
    // pack result
    varying float * uniform f = (varying float * uniform) dest;
    *f++ = result.r;
    *f++ = result.g;
    *f   = result.b;
}

// utility function to place a vec3f into a uniform float array
inline void
storeResult(const varying Vec3f &result,
            uniform float * uniform dest)
{
    // pack result
    varying float * uniform f = (varying float * uniform) dest;
    *f++ = result.x;
    *f++ = result.y;
    *f   = result.z;
}

// utility function to place a vec2f into a uniform float array
inline void
storeResult(const varying Vec2f &result,
            uniform float * uniform dest)
{
    // pack result
    varying float * uniform f = (varying float * uniform) dest;
    *f++ = result.x;
    *f   = result.y;
}

// utility function to place a float into a uniform float array
inline void
storeResult(const varying float result,
            uniform float * uniform dest)
{
    // pack result
    varying float * uniform f = (varying float * uniform) dest;
    *f = result;
}

#define APPLY_PIXEL_WEIGHT_BODY(UN_WEIGHTED_RESULT, PIXEL_WEIGHT)       \
    /* apply pixel weight */                                            \
    varying float pw = *((varying float * uniform) (PIXEL_WEIGHT));     \
    return (UN_WEIGHTED_RESULT) * pw;

inline varying Color
applyPixelWeight(const varying Color &unWeightedResult,
                 const uniform float * uniform pixelWeight)
{
    APPLY_PIXEL_WEIGHT_BODY(unWeightedResult, pixelWeight);
}

inline varying Vec3f
applyPixelWeight(const varying Vec3f &unWeightedResult,
                 const uniform float * uniform pixelWeight)
{
    APPLY_PIXEL_WEIGHT_BODY(unWeightedResult, pixelWeight);
}

inline varying Vec2f
applyPixelWeight(const varying Vec2f &unWeightedResult,
                 const uniform float * uniform pixelWeight)
{
    APPLY_PIXEL_WEIGHT_BODY(unWeightedResult, pixelWeight);
}

inline varying float
applyPixelWeight(const varying float unWeightedResult,
                 const uniform float * uniform pixelWeight)
{
    APPLY_PIXEL_WEIGHT_BODY(unWeightedResult, pixelWeight);
}


#define LABEL_MATCH_BODY(LABEL_ID, ENTRY_LABEL_INDICES)                 \
    const uniform int numEntryLabelIds =                                \
        ((const uniform int * uniform) (ENTRY_LABEL_INDICES)->mEnd) -   \
        ((const uniform int * uniform) (ENTRY_LABEL_INDICES)->mBegin);  \
                                                                        \
    const uniform int * uniform const entryLabelIds =                   \
        (const uniform int * uniform) (ENTRY_LABEL_INDICES)->mBegin;    \
                                                                        \
    /* if no labels exist on the aov entry */                           \
    /* then the labelId is irrelvant */                                 \
                                                                        \
    /* entry doesn't have labels */                                     \
    if (numEntryLabelIds == 0) return true;                             \
                                                                        \
    /* entry has labels, one must match labelId */                      \
    for (uniform int i = 0; i < numEntryLabelIds; ++i) {                \
        if (entryLabelIds[i] == (LABEL_ID)) {                           \
            return true;                                                \
        }                                                               \
    }                                                                   \
                                                                        \
    /* entry has labels, but labelId does not match */                  \
    return false;

inline uniform bool
labelMatch(uniform int labelId, const uniform std_vector * uniform entryLabelIndices)
{
    LABEL_MATCH_BODY(labelId, entryLabelIndices);
}

inline varying bool
labelMatch(varying int labelId, const uniform std_vector * uniform entryLabelIndices)
{
    LABEL_MATCH_BODY(labelId, entryLabelIndices);
}


// ---------------------------------------------------------------------------
extern "C" void exit(uniform int);

export void
computeAlbedo(const uniform int flags,
              const uniform int subsurface,
              const uniform std_vector * uniform entryLabelIndices,
              const uniform std_vector * uniform entryMaterialLabelIndices,
              const uniform std_vector * uniform entryGeomLabelIndices,
#pragma ignore warning(all)
              const varying Bsdf * uniform bsdf,
#pragma ignore warning(all)
              const varying BsdfSampler * uniform bSampler,
#pragma ignore warning(all)
              const varying BsdfSample * uniform bsmps,
#pragma ignore warning(all)
              const varying BsdfSlice * uniform bsdfSlice,
              const uniform float * uniform pixelWeight,
#pragma ignore warning(all)
              const varying Color * uniform ssAov,
              uniform float * uniform dest,
              const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Color result = sBlack;
    unsigned int numContributions = 0;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        // lobe albedo
        if (flags != BSDF_LOBE_TYPE_NONE) {
            if (bSampler && bsmps) {
                // mimics drawBsdfSamples
                const uniform int lobeCount = BsdfSampler_getLobeCount(bSampler);
                for (uniform int lobeIdx = 0; lobeIdx < lobeCount; ++lobeIdx) {
                    const varying BsdfLobe * uniform lobe = BsdfSampler_getLobe(bSampler, lobeIdx);

                    const varying int lobeSampleCount = BsdfSampler_getLobeSampleCount(bSampler, lobeIdx);
                    const varying float invLobeSampleCount = BsdfSampler_getInvLobeSampleCount(bSampler, lobeIdx);

                    const uniform int lobeLabel = BsdfLobe_getLabel(lobe);
                    if (isActive(lobe) &&
                        BsdfLobe_matchesFlags(lobe, flags) &&
                        labelMatch(aovDecodeLabel(lobeLabel), entryLabelIndices)) {
                        ++numContributions;
                        uniform int s = bSampler->mLobeToSampleIndex[lobeIdx];
                        for (int i = 0; i < lobeSampleCount; ++i, ++s) {
                            const varying BsdfSample &bsmp = bsmps[s];
                            if (BsdfSample_isValid(&bsmp)) {
                                MNRY_ASSERT(!isSampleInvalid(bsmp.f, bsmp.pdf));
                                result = result + invLobeSampleCount * bsmp.f * rcp(bsmp.pdf);
                            }
                        }

                    }
                }

                // apply pixel weight
                result = applyPixelWeight(result, pixelWeight);
            } else if (bsdfSlice) {
                // Not using the bsdf sampler, which means we don't have bsdf sample objects.
                // We'll fallback to the cheaper lobe->albedo() method.  This might actually
                // be a more useful result since the lobe albedo method is used directly by the
                // bsdf one sampler to drive sampling decisions.
                for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                    const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                    const uniform int lobeLabel = BsdfLobe_getLabel(lobe);
                    if (isActive(lobe) &&
                        BsdfLobe_matchesFlags(lobe, flags) &&
                        labelMatch(aovDecodeLabel(lobeLabel), entryLabelIndices)) {
                        ++numContributions;
                        result = result + BsdfLobe_albedo(lobe, *bsdfSlice);
                    }
                }
            }
        }

        // subsurface albedo
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices)) {
                ++numContributions;
                // ssAov already takes pixel weight into account
                result = result + *ssAov;
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices)) {
                ++numContributions;
                // ssAov already takes pixel weight into account
                result = result + *ssAov;
            }
        }
    }

    // don't double count the albedo
    if (numContributions > 1) {
        // take the average
        result = result / numContributions;
    }

    storeResult(result, dest);
}

// ---------------------------------------------------------------------------

export void
#pragma ignore warning(all)
computeColor(const varying Bsdf * uniform bsdf,
             const uniform int flags,
             const uniform int subsurface,
             const uniform std_vector * uniform entryLabelIndices,
             const uniform std_vector * uniform entryMaterialLabelIndices,
             const uniform std_vector * uniform entryGeomLabelIndices,
             const uniform float * uniform pixelWeight,
             uniform float * uniform dest,
             const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Color result = sBlack;

    bool diffuseLobeContributed = false;
    bool subsurfaceContributed = false;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        // lobes
        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {
                    diffuseLobeContributed = true;
                    if (BsdfLobe_hasProperty(lobe, BSDF_LOBE_PROPERTY_COLOR)) {
                        Color c;
                        BsdfLobe_getProperty(lobe, BSDF_LOBE_PROPERTY_COLOR, (varying float * uniform) &c);
                        result = result + c;
                    }
                }
            }
        }

        // subsurface
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices) &&
                Bssrdf_hasProperty(bssrdf, SUBSURFACE_PROPERTY_COLOR)) {
                subsurfaceContributed = true;
                Color c;
                Bssrdf_getProperty(bssrdf, SUBSURFACE_PROPERTY_COLOR, (varying float * uniform) &c);
                result = result + c;
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices) &&
                VolumeSubsurface_hasProperty(vs, SUBSURFACE_PROPERTY_COLOR)) {
                subsurfaceContributed = true;
                Color c;
                VolumeSubsurface_getProperty(vs, SUBSURFACE_PROPERTY_COLOR, (varying float * uniform) &c);
                result = result + c;
            }
        }

        // pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    // don't double count the albedo
    if (diffuseLobeContributed && subsurfaceContributed) {
        result = result * 0.5f;
    }

    storeResult(result, dest);
}

// ---------------------------------------------------------------------------
export void
#pragma ignore warning(all)
computeEmission(const varying Bsdf * uniform bsdf,
                const uniform std_vector * uniform entryMaterialLabelIndices,
                const uniform std_vector * uniform entryGeomLabelIndices,
                const uniform float * uniform pixelWeight,
                uniform float * uniform dest,
                const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Color result = sBlack;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        if (!((1 << programIndex) & bsdf->mEarlyTerminationMask)) {
            // get the emission
            result = Bsdf_getSelfEmission(bsdf);

            // apply pixel weight
            result = applyPixelWeight(result, pixelWeight);
        }

    }

    storeResult(result, dest);
}

export void
computeMatte(const uniform int flags,
             const uniform int subsurface,
             const uniform std_vector * uniform entryLabelIndices,
             const uniform std_vector * uniform entryMaterialLabelIndices,
             const uniform std_vector * uniform entryGeomLabelIndices,
#pragma ignore warning(all)
             const varying Bsdf * uniform bsdf,
             const uniform float * uniform pixelWeight,
             uniform float * uniform dest,
             const uniform uint32_t laneMask) {

    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    varying float result = 0.f;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {

                    // pack result
                    varying float * uniform f = (varying float * uniform) pixelWeight;
                    result = *f;
                    break;
                }
            }
        }

        // subsurface
        if (result == 0.f && subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) && labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices)) {
                varying float * uniform f = (varying float * uniform) pixelWeight;
                result = *f;
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) && labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices)) {
                varying float * uniform f = (varying float * uniform) pixelWeight;
                result = *f;
            }
        }

        // if our bsdf has no lobes or sub-surface, and our entry flags are empty
        // we want to output the matte aov
        if (!Bsdf_getLobeCount(bsdf) && !Bsdf_hasSubsurface(bsdf) && flags == BSDF_LOBE_TYPE_ALL) {
            varying float * uniform f = (varying float * uniform) pixelWeight;
            result = *f;
        }
    }

    storeResult(result, dest);
}

// we need to rely on a c++ function to retrieve
// interpolated camera matrices
typedef void(*GetRender2CameraFn)(uniform intptr_t, uniform float, uniform Mat4f * uniform);

inline void
gatherVector(varying Vec4f &target, const uniform Vec4f &source, uniform int lane)
{
    target.x = insert(target.x, lane, source.x);
    target.y = insert(target.y, lane, source.y);
    target.z = insert(target.z, lane, source.z);
    target.w = insert(target.w, lane, source.w);
}

inline void
gatherMatrix(varying Mat4f &target, const uniform Mat4f &source, uniform int lane)
{
    gatherVector(target.vx, source.vx, lane);
    gatherVector(target.vy, source.vy, lane);
    gatherVector(target.vz, source.vz, lane);
    gatherVector(target.vw, source.vw, lane);
}

static varying float
computeDepthFromIsect(const varying Intersection &isect, const varying RayDifferential &ray,
                      uniform intptr_t scene, GetRender2CameraFn getRender2Camera)
{
    varying float rayTime = Ray_getTime(ray);
    // Use reduce_equal to determine if the varying inputs ray time
    // are identical across all lanes.
    if (reduce_equal(rayTime)) {
        // Great!
        // Pick an active lane and get the time.
        // Choose the smallest lane number from the set of active lanes
        uniform int activeLanes = lanemask();
        uniform int laneBit = activeLanes & ~(activeLanes - 1);
        uniform int lane = 0;
        while (laneBit >> (lane + 1)) ++lane;
        MNRY_ASSERT((1 << lane) & lanemask());
        uniform float t = extract(Ray_getTime(ray), lane);
        uniform Mat4f r2c;
        getRender2Camera(scene, t, &r2c);

        varying Vec3f cp = transformPoint(r2c, getP(isect));
        return -cp.z; // might want to clamp this to 0.f
    } else {
        varying Mat4f r2c;
        foreach_active(lane) {
            uniform float laneT = extract(rayTime, lane);
            uniform Mat4f laneR2c;
            getRender2Camera(scene, laneT, (uniform Mat4f * uniform) &laneR2c);
            gatherMatrix(r2c, laneR2c, lane);
        }

        varying Vec3f cp = transformPoint(r2c, getP(isect));
        return -cp.z; // might want to clamp this to 0.f
    }
}

export void
computeAndStoreDepthFromIsect(const uniform intptr_t isectv,
                              const uniform intptr_t rayv,
                              const uniform intptr_t scene,
                              uniform intptr_t getRender2CameraFn,
#pragma ignore warning(all)
                              varying float * uniform depth)
{
    varying Intersection &isect = *((varying Intersection * uniform) isectv);
    varying RayDifferential &ray = *((varying RayDifferential * uniform) rayv);
    GetRender2CameraFn getRender2Camera = (GetRender2CameraFn) getRender2CameraFn;

    *depth = computeDepthFromIsect(isect, ray, scene, getRender2Camera);
}

// Need to define limited double types to support
// the world space position aov.  This is not in the general math
// library as that lib is exclusively 32-bit float
struct Vec3d {
    varying double x;
    varying double y;
    varying double z;
};

struct Vec4d {
    varying double x;
    varying double y;
    varying double z;
    varying double w;
};

struct Mat4d {
    Vec4d vx;
    Vec4d vy;
    Vec4d vz;
    Vec4d vw;
};

varying Vec3d
Vec3d_ctor(varying double x, varying double y, varying double z)
{
    varying Vec3d v;
    v.x = x;
    v.y = y;
    v.z = z;

    return v;
}

uniform Vec4d
Vec4d_ctor(uniform double x, uniform double y, uniform double z, uniform double w)
{
    uniform Vec4d v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;

    return v;
}

uniform Mat4d
Mat4d_ctor(const uniform double * uniform d)
{
    uniform Mat4d m;
    m.vx = Vec4d_ctor(d[0], d[1], d[2], d[3]);
    m.vy = Vec4d_ctor(d[4], d[5], d[6], d[7]);
    m.vz = Vec4d_ctor(d[8], d[9], d[10], d[11]);
    m.vw = Vec4d_ctor(d[12], d[13], d[14], d[15]);

    return m;
}

varying Vec3d
transformPoint(const uniform Mat4d &r2w, const varying Vec3d &v)
{
    Vec3d result = Vec3d_ctor(v.x * r2w.vx.x + v.y * r2w.vy.x + v.z * r2w.vz.x + r2w.vw.x,
                              v.x * r2w.vx.y + v.y * r2w.vy.y + v.z * r2w.vz.y + r2w.vw.y,
                              v.x * r2w.vx.z + v.y * r2w.vy.z + v.z * r2w.vz.z + r2w.vw.z);

    return result;
}

varying Vec3f
toVec3f(const varying Vec3d &v)
{
    return Vec3f_ctor((varying float) v.x, (varying float) v.y, (varying float) v.z);
}

// ---------------------------------------------------------------------------
// Motion Vectors

// cpp functions on data relied on for motion computations
struct MotionVectorFns {
    intptr_t computeMotionFunction;
    int32_t motionKey;
};

typedef void(* uniform ComputeMotionFn)(uniform intptr_t scene, uniform float time,
    const uniform Vec3f * uniform p, const uniform Vec3f * uniform dp,
    uniform Vec2f * uniform result);

static varying Vec2f
computeMotion(const varying Intersection &isect, const varying RayDifferential &ray,
              uniform intptr_t scene, const uniform MotionVectorFns * uniform mvFns)
{

    varying Vec3f isectP = getP(isect);
    varying float rayTime = Ray_getTime(ray);
    const uniform uint8_t * varying isectDpPtr =
        getAttributeLocation(isect, mvFns->motionKey);
    varying int8_t providesDp = isectDpPtr != nullptr;
    varying Vec3f isectDp;
    if (providesDp) {
        isectDp = *((const uniform Vec3f * varying) isectDpPtr);
    }
    ComputeMotionFn computeMotionFn = (ComputeMotionFn) mvFns->computeMotionFunction;
    
    varying Vec2f motion = Vec2f_ctor(0.0f, 0.0f);
    foreach_active(lane) {
        uniform Vec3f p;
        p.x = extract(isectP.x, lane);
        p.y = extract(isectP.y, lane);
        p.z = extract(isectP.z, lane);
        uniform float time = extract(rayTime, lane);
        const uniform Vec3f * uniform dpPtr = nullptr;
        uniform Vec3f dp;
        if (extract(providesDp, lane)) {
            dp.x = extract(isectDp.x, lane);
            dp.y = extract(isectDp.y, lane);
            dp.z = extract(isectDp.z, lane);
            dpPtr = &dp;
        }
        uniform Vec2f result;
        computeMotionFn(scene, time, &p, dpPtr, &result);
        motion.x = insert(motion.x, lane, result.x);
        motion.y = insert(motion.y, lane, result.y);
    }

    return motion;
}

// ---------------------------------------------------------------------------

static void
getStateVar(const uniform int stateVar, const varying Intersection &isect,
            const varying RayDifferential &ray, uniform intptr_t scene,
            GetRender2CameraFn getRender2CameraFn, const uniform double * uniform r2w,
            const uniform MotionVectorFns * uniform mvFns,
            const uniform float * uniform pixelWeight, uniform float * uniform dest)
{
    switch (stateVar) {
    case AOV_STATE_VAR_P:
        {
            const varying Vec3f p = applyPixelWeight(getP(isect), pixelWeight);
            storeResult(p, dest);
        }
        break;
    case AOV_STATE_VAR_NG:
        {
            const varying Vec3f ng = applyPixelWeight(getNg(isect), pixelWeight);
            storeResult(ng, dest);
        }
        break;
    case AOV_STATE_VAR_N:
        {
            const varying Vec3f n = applyPixelWeight(getN(isect), pixelWeight);
            storeResult(n, dest);
        }
        break;
    case AOV_STATE_VAR_ST:
        {
            const varying Vec2f st = applyPixelWeight(getSt(isect), pixelWeight);
            storeResult(st, dest);
        }
        break;
    case AOV_STATE_VAR_DPDS:
        {
            const varying Vec3f dpds = applyPixelWeight(getdPds(isect), pixelWeight);
            storeResult(dpds, dest);
        }
        break;
    case AOV_STATE_VAR_DPDT:
        {
            const varying Vec3f dpdt = applyPixelWeight(getdPdt(isect), pixelWeight);
            storeResult(dpdt, dest);
        }
        break;
    case AOV_STATE_VAR_DSDX:
        {
            const float dsdx = applyPixelWeight(getdSdx(isect), pixelWeight);
            storeResult(dsdx, dest);
        }
        break;
    case AOV_STATE_VAR_DSDY:
        {
            const float dsdy = applyPixelWeight(getdSdy(isect), pixelWeight);
            storeResult(dsdy, dest);
        }
        break;
    case AOV_STATE_VAR_DTDX:
        {
            const float dtdx = applyPixelWeight(getdTdx(isect), pixelWeight);
            storeResult(dtdx, dest);
        }
        break;
    case AOV_STATE_VAR_DTDY:
        {
            const float dtdy = applyPixelWeight(getdTdy(isect), pixelWeight);
            storeResult(dtdy, dest);
        }
        break;
    case AOV_STATE_VAR_WP:
        {
            // precision loss is unavoidable as our output is necessarily 32 bits
            const varying Vec3f renderPos = getP(isect);
            const varying Vec3d renderPos64 = Vec3d_ctor(renderPos.x, renderPos.y, renderPos.z);
            const varying Vec3d worldPos64 = transformPoint(Mat4d_ctor(r2w), renderPos64);
            const varying Vec3f worldPos = applyPixelWeight(toVec3f(worldPos64), pixelWeight);
            storeResult(worldPos, dest);
        }
        break;
    case AOV_STATE_VAR_DEPTH:
        {
            varying float depth = computeDepthFromIsect(isect, ray, scene, getRender2CameraFn);
            depth = applyPixelWeight(depth, pixelWeight);
            storeResult(depth, dest);
        }
        break;
    case AOV_STATE_VAR_MOTION:
        {
            varying Vec2f v = computeMotion(isect, ray, scene, mvFns);
            v = applyPixelWeight(v, pixelWeight);
            storeResult(v, dest);
        }
        break;
    default:
        MNRY_ASSERT(0);
    }
}

// label matching is identical for state and primitive attributes
static varying bool
labelMatch(const uniform int flags, const uniform int subsurface,
           const uniform std_vector * uniform entryLabelIndices,
           const uniform std_vector * uniform entryMaterialLabelIndices,
           const uniform std_vector * uniform entryGeomLabelIndices,
           const varying Bsdf * uniform bsdf)
{
    varying bool matched = false;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        // This initial case looks complicated, but all it really says is if
        // the user specified a default (empty) lobe selector then
        // match an empty material - unless the empty material is a cutout.
        const uniform bool selectorIsDefault = (flags == BSDF_LOBE_TYPE_ALL && subsurface != 0);
        if (selectorIsDefault) {
            // Not as simple as scalar, we need to check per lane for active
            // lobes, subsurface, and early terminators.
            bool isEmpty = true;
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                if (isActive(Bsdf_getLobe(bsdf, lobeIdx))) {
                    isEmpty = false;
                    break;
                }
            }
            if (isActive(Bsdf_getBssrdf(bsdf))) {
                isEmpty = false;
            }
            if (isEmpty) {
                matched = !((1 << programIndex) & bsdf->mEarlyTerminationMask);
            }
        }

        // lobes
        if (!matched && flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {
                    matched = true;
                    break;
                }
            }
        }

        // subsurface
        if (!matched && subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) && labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices)) {
                matched = true;
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) && labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices)) {
                matched = true;
            }
        }
    }

    return matched;
}

// fill in a miss value for state and primitive attributes
static void
setMissValue(const uniform float missValue, const uniform int numFloats, uniform float * uniform dest)
{
    // assumes the missValue is 0, +inf, or -inf, so no need to apply pixel weight
    MNRY_ASSERT(missValue == 0.0f || isinf(missValue));
    if (numFloats == 1) {
        const varying float v = missValue;
        storeResult(v, dest);
    } else if (numFloats == 2) {
        const varying Vec2f v2 = Vec2f_ctor(missValue, missValue);
        storeResult(v2, dest);
    } else if (numFloats == 3) {
        const varying Vec3f v3 = Vec3f_ctor(missValue, missValue, missValue);
        storeResult(v3, dest);
    }
}

export void
computeStateAov(const uniform int flags,
                const uniform int stateVar,
                const uniform float missValue,
                const uniform int numFloats,
                const uniform int subsurface,
                const uniform std_vector * uniform entryLabelIndices,
                const uniform std_vector * uniform entryMaterialLabelIndices,
                const uniform std_vector * uniform entryGeomLabelIndices,
                const uniform intptr_t isectv,
                const uniform intptr_t rayv,
                const uniform intptr_t scene,
#pragma ignore warning(all)
                const varying Bsdf * uniform bsdf,
                const uniform intptr_t getRender2CameraFn,
                const uniform double * uniform r2w,
                const uniform MotionVectorFns * uniform mvFns,
                const uniform float * uniform pixelWeight,
                uniform float * uniform dest,
                const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    varying bool matched = labelMatch(flags, subsurface, entryLabelIndices,
                                      entryMaterialLabelIndices,
                                      entryGeomLabelIndices, bsdf);

    if (matched) {
        // if we matched, we compute the state aov
        varying Intersection &isect = *((varying Intersection * uniform) isectv);
        varying RayDifferential &ray = *((varying RayDifferential * uniform) rayv);
        GetRender2CameraFn getRender2Camera = (GetRender2CameraFn) getRender2CameraFn;

        getStateVar(stateVar, isect, ray, scene, getRender2Camera, r2w, mvFns, pixelWeight, dest);
    } else {
        // if not, need to fill in dest with an appropriate miss value
        setMissValue(missValue, numFloats, dest);
    }
}

export void
computePrimitiveAttribute(const uniform int flags,
                          const uniform int geomKey,
                          const uniform float missValue,
                          const uniform int numFloats,
                          const uniform int subsurface,
                          const uniform std_vector * uniform entryLabelIndices,
                          const uniform std_vector * uniform entryMaterialLabelIndices,
                          const uniform std_vector * uniform entryGeomLabelIndices,
                          const uniform intptr_t isectv,
#pragma ignore warning(all)
                          const varying Bsdf * uniform bsdf,
                          const uniform float * uniform pixelWeight,
                          uniform float * uniform dest,
                          const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    varying bool matched = labelMatch(flags, subsurface, entryLabelIndices,
                                      entryMaterialLabelIndices,
                                      entryGeomLabelIndices, bsdf);

    if (matched) {
        varying Intersection &isect = *((varying Intersection * uniform) isectv);
        const uniform uint8_t * varying ptr = getAttributeLocation(isect, geomKey);
        if (ptr) {
            if (numFloats == 1) {
                varying float v = *((const uniform float * varying) ptr);
                storeResult(applyPixelWeight(v, pixelWeight), dest);
            } else if (numFloats == 2) {
                varying Vec2f v2 = *((const uniform Vec2f * varying) ptr);
                storeResult(applyPixelWeight(v2, pixelWeight), dest);
            } else if (numFloats == 3) {
                // could be Rgb or Vec3
                varying Vec3f v3 = *((const uniform Vec3f * varying) ptr);
                storeResult(applyPixelWeight(v3, pixelWeight), dest);
            } else {
                MNRY_ASSERT(0);
            }
        } else {
            // intersection did not have this primitive attribute, use the miss value
            setMissValue(missValue, numFloats, dest);
        }
    } else {
        // if not, need to fill in dest with an appropriate miss value
        setMissValue(missValue, numFloats, dest);
    }
}

// ---------------------------------------------------------------------------
export void
computeRoughness(const uniform int flags,
                 const uniform std_vector * uniform entryLabelIndices,
                 const uniform std_vector * uniform entryMaterialLabelIndices,
                 const uniform std_vector * uniform entryGeomLabelIndices,
#pragma ignore warning(all)
                 const varying Bsdf * uniform bsdf,
                 const uniform float * uniform pixelWeight,
                 uniform float * uniform dest,
                 const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    varying Vec2f result = Vec2f_ctor(0.f, 0.f);
    varying int matchingLobes = 0;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    BsdfLobe_hasProperty(lobe, BSDF_LOBE_PROPERTY_ROUGHNESS) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {

                    varying Vec2f sr;
                    BsdfLobe_getProperty(lobe,
                                         BSDF_LOBE_PROPERTY_ROUGHNESS,
                                         (varying float * uniform) &sr);
                    result = result + sr;
                    ++matchingLobes;
                }
            }
        }

        if (matchingLobes) result = result * rcp((varying float) matchingLobes);

        // apply pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);
}

// ---------------------------------------------------------------------------
export void
computeNormal(const uniform int flags,
              const uniform int subsurface,
              const uniform std_vector * uniform entryLabelIndices,
              const uniform std_vector * uniform entryMaterialLabelIndices,
              const uniform std_vector * uniform entryGeomLabelIndices,
#pragma ignore warning(all)
              const varying Bsdf * uniform bsdf,
              const uniform float * uniform pixelWeight,
              uniform float * uniform dest,
              const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    varying Vec3f result = Vec3f_ctor(0.f, 0.f, 0.f);
    varying int matchingLobes = 0;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        // lobes
        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    BsdfLobe_hasProperty(lobe, BSDF_LOBE_PROPERTY_NORMAL) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {
                    varying Vec3f N;
                    BsdfLobe_getProperty(lobe,
                                         BSDF_LOBE_PROPERTY_NORMAL,
                                         (varying float * uniform) &N);
                    result = result + N;
                    ++matchingLobes;
                }
            }
        }

        // subsurface
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices)) {
                const varying ReferenceFrame * uniform frame = Bssrdf_getFrame(bssrdf);
                result = result + getN(*frame);
                ++matchingLobes;
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices)) {
                const varying Vec3f N = VolumeSubsurface_getNormal(vs);
                result = result + N;
                ++matchingLobes;
            }
        }

        // average
        if (matchingLobes) result = result * rcp((varying float) matchingLobes);

        // apply pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);
}

// ---------------------------------------------------------------------------
export void
#pragma ignore warning(all)
computeRadius(const varying Bsdf * uniform bsdf,
              const uniform std_vector * uniform entryLabelIndices,
              const uniform std_vector * uniform entryMaterialLabelIndices,
              const uniform std_vector * uniform entryGeomLabelIndices,
              const uniform float * uniform pixelWeight,
              uniform float * uniform dest,
              const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Color result = sBlack;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
        if (isActive(bssrdf) &&
            labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices) &&
            Bssrdf_hasProperty(bssrdf, SUBSURFACE_PROPERTY_RADIUS)) {
            Bssrdf_getProperty(bssrdf, SUBSURFACE_PROPERTY_RADIUS, (varying float * uniform) &result);
        }
        const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
        if (isActive(vs) &&
            labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices) &&
            VolumeSubsurface_hasProperty(vs, SUBSURFACE_PROPERTY_RADIUS)) {
            VolumeSubsurface_getProperty(vs, SUBSURFACE_PROPERTY_RADIUS, (varying float * uniform) &result);
        }
        // apply pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);
}

// ---------------------------------------------------------------------------
export void
#pragma ignore warning(all)
computeFresnelColor(const varying Bsdf * uniform bsdf,
                    const uniform int flags,
                    const uniform int subsurface,
                    const uniform std_vector * uniform entryLabelIndices,
                    const uniform std_vector * uniform entryMaterialLabelIndices,
                    const uniform std_vector * uniform entryGeomLabelIndices,
                    const uniform float * uniform pixelWeight,
                    uniform float * uniform dest,
                    const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Color result = sBlack;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        // lobe fresnel color
        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {

                    const varying Fresnel * uniform fresnel = BsdfLobe_getFresnel(lobe);
                    if (isActive(fresnel) &&
                        Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_COLOR)) {
                        varying Color c;
                        Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_COLOR, (varying float * uniform) &c);
                        result = result + c;
                    }
                }
            }
        }

        // subsurface fresnel
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices) &&
                isFresnelActive(bssrdf)) {
                const varying Fresnel * uniform fresnel = Bssrdf_getTransmissionFresnel(bssrdf);
                if (isActive(fresnel) && Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_COLOR)) {
                    varying Color c;
                    Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_COLOR, (varying float * uniform) &c);
                    result = result + c;
                }
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices) &&
                isFresnelActive(vs)) {
                const varying Fresnel * uniform fresnel = VolumeSubsurface_getTransmissionFresnel(vs);
                if (isActive(fresnel) && Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_COLOR)) {
                    varying Color c;
                    Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_COLOR, (varying float * uniform) &c);
                    result = result + c;
                }
            }
        }

        // pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);

}

// ---------------------------------------------------------------------------
export void
#pragma ignore warning(all)
computeFresnelFactor(const varying Bsdf * uniform bsdf,
                     const uniform int flags,
                     const uniform int subsurface,
                     const uniform std_vector * uniform entryLabelIndices,
                     const uniform std_vector * uniform entryMaterialLabelIndices,
                     const uniform std_vector * uniform entryGeomLabelIndices,
                     const uniform float * uniform pixelWeight,
                     uniform float * uniform dest,
                     const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    float result = 0;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        varying int s = 0;

        // lobes
        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {

                    const varying Fresnel * uniform fresnel = BsdfLobe_getFresnel(lobe);
                    if (isActive(fresnel) &&
                        Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_FACTOR)) {
                        float f;
                        Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_FACTOR, &f);
                        result = result + f;
                        ++s;
                    }
                }
            }
        }

        // subsurface
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices) &&
                isFresnelActive(bssrdf)) {
                const varying Fresnel * uniform fresnel = Bssrdf_getTransmissionFresnel(bssrdf);
                if (isActive(fresnel) && Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_FACTOR)) {
                    float f;
                    Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_FACTOR, &f);
                    result = result + f;
                    ++s;
                }
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices) &&
                isFresnelActive(vs)) {
                const varying Fresnel * uniform fresnel = VolumeSubsurface_getTransmissionFresnel(vs);
                if (isActive(fresnel) && Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_FACTOR)) {
                    float f;
                    Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_FACTOR, &f);
                    result = result + f;
                    ++s;
                }
            }
        }

        // average
        if (s > 0) result = result * rcp((varying float) s);

        // pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);

}

// ---------------------------------------------------------------------------
export void
#pragma ignore warning(all)
computeFresnelRoughness(const varying Bsdf * uniform bsdf,
                        const uniform int flags,
                        const uniform int subsurface,
                        const uniform std_vector * uniform entryLabelIndices,
                        const uniform std_vector * uniform entryMaterialLabelIndices,
                        const uniform std_vector * uniform entryGeomLabelIndices,
                        const uniform float * uniform pixelWeight,
                        uniform float * uniform dest,
                        const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Vec2f result = Vec2f_ctor(0.f, 0.f);

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        varying int matchingLobes = 0;

        // lobes
        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int i = 0; i < Bsdf_getLobeCount(bsdf); ++i) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, i);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {

                    const varying Fresnel * uniform fresnel = BsdfLobe_getFresnel(lobe);
                    if (isActive(fresnel) &&
                        Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_ROUGHNESS)) {
                        varying Vec2f sr;
                        Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_ROUGHNESS, (varying float * uniform) &sr);
                        result = result + sr;
                        ++matchingLobes;
                    }
                }
            }
        }

        // subsurface
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices) &&
                isFresnelActive(bssrdf)) {
                const varying Fresnel * uniform fresnel = Bssrdf_getTransmissionFresnel(bssrdf);
                if (isActive(fresnel) && Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_ROUGHNESS)) {
                    varying Vec2f sr;
                    Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_ROUGHNESS, (varying float * uniform) &sr);
                    result = result + sr;
                    ++matchingLobes;
                }
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices) &&
                isFresnelActive(vs)) {
                const varying Fresnel * uniform fresnel = VolumeSubsurface_getTransmissionFresnel(vs);
                if (isActive(fresnel) && Fresnel_hasProperty(fresnel, FRESNEL_PROPERTY_ROUGHNESS)) {
                    varying Vec2f sr;
                    Fresnel_getProperty(fresnel, FRESNEL_PROPERTY_ROUGHNESS, (varying float * uniform) &sr);
                    result = result + sr;
                    ++matchingLobes;
                }
            }
        }

        if (matchingLobes) result = result * rcp((varying float) matchingLobes);

        // apply pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);

}

// ---------------------------------------------------------------------------

export void
#pragma ignore warning(all)
computePbrValidity(const varying Bsdf * uniform bsdf,
                   const uniform int flags,
                   const uniform int subsurface,
                   const uniform std_vector * uniform entryLabelIndices,
                   const uniform std_vector * uniform entryMaterialLabelIndices,
                   const uniform std_vector * uniform entryGeomLabelIndices,
                   const uniform float * uniform pixelWeight,
                   uniform float * uniform dest,
                   const uniform uint32_t laneMask)
{
    // coming back to ispc from c++, we need to re-establish our lane mask
    if (!((1 << programIndex) & laneMask)) return;

    Color result = sBlack;

    if (labelMatch(Bsdf_getGeomLabelId(bsdf), entryGeomLabelIndices) &&
        labelMatch(Bsdf_getMaterialLabelId(bsdf), entryMaterialLabelIndices)) {

        // lobes
        if (flags != BSDF_LOBE_TYPE_NONE) {
            for (uniform int lobeIdx = 0; lobeIdx < Bsdf_getLobeCount(bsdf); ++lobeIdx) {
                const varying BsdfLobe * uniform lobe = Bsdf_getLobe(bsdf, lobeIdx);
                if (isActive(lobe) &&
                    BsdfLobe_matchesFlags(lobe, flags) &&
                    labelMatch(aovDecodeLabel(BsdfLobe_getLabel(lobe)), entryLabelIndices)) {

                    if (BsdfLobe_hasProperty(lobe, BSDF_LOBE_PROPERTY_PBR_VALIDITY)) {
                        Color c;
                        BsdfLobe_getProperty(lobe, BSDF_LOBE_PROPERTY_PBR_VALIDITY, (varying float * uniform) &c);
                        result = result + c;
                    }
                }
            }
        }

        // subsurface
        if (subsurface) {
            const varying Bssrdf * uniform bssrdf = Bsdf_getBssrdf(bsdf);
            if (isActive(bssrdf) &&
                labelMatch(aovDecodeLabel(Bssrdf_getLabel(bssrdf)), entryLabelIndices) &&
                Bssrdf_hasProperty(bssrdf, SUBSURFACE_PROPERTY_PBR_VALIDITY)) {
                Color c;
                Bssrdf_getProperty(bssrdf, SUBSURFACE_PROPERTY_PBR_VALIDITY, (varying float * uniform) &c);
                result = result + c;
            }
            const varying VolumeSubsurface * uniform vs = Bsdf_getVolumeSubsurface(bsdf);
            if (isActive(vs) &&
                labelMatch(aovDecodeLabel(VolumeSubsurface_getLabel(vs)), entryLabelIndices) &&
                VolumeSubsurface_hasProperty(vs, SUBSURFACE_PROPERTY_PBR_VALIDITY)) {
                Color c;
                VolumeSubsurface_getProperty(vs, SUBSURFACE_PROPERTY_PBR_VALIDITY, (varying float * uniform) &c);
                result = result + c;
            }
        }

        // pixel weight
        result = applyPixelWeight(result, pixelWeight);
    }

    storeResult(result, dest);
}

// ---------------------------------------------------------------------------

extern "C" void
CPP_aovSetMaterialAovs(uniform PbrTLState * uniform pbrTls,
                       const uniform AovSchema &aovSchema,
                       const uniform LightAovs &lightAovs,
                       const uniform MaterialAovs &materialAovs,
                       const varying Intersection &isect,
                       const varying RayDifferential &ray,
                       uniform intptr_t scene,
                       const varying Bsdf &bsdf,
                       const varying Color &ssAov,
                       const varying BsdfSampler * uniform bSampler,
                       const varying BsdfSample * uniform bsmps,
                       const varying BsdfSlice * uniform bsdfSlice,
                       const uniform float * uniform pixelWeight,
                       const uniform uint32_t * uniform pixel,
                       const uniform uint32_t * uniform deepDataHandle,
                       const uniform int * uniform lpeStateId,
                       const uniform uint32_t * uniform isPrimaryRay,
                       uniform uint32_t laneMask);

extern "C" void
CPP_aovAccumLightAovs(uniform PbrTLState * uniform pbrTls,
                      const uniform AovSchema &aovSchema,
                      const uniform LightAovs &lightAovs,
                      const uniform Color &value,
                      uniform int lpeStateId,
                      uniform uint32_t pixel,
                      uniform uint32_t deepDataHandle);

extern "C" void
CPP_aovAccumPostScatterExtraAov(uniform PbrTLState * uniform pbrTls,
                                const uniform AovSchema &aovSchema,
                                const uniform LightAovs &lightAovs,
                                const uniform Color &value,
                                const uniform Color &sampleValue,
                                uniform int lpeStateId,
                                uniform int labelId,
                                uniform uint32_t pixel,
                                uniform uint32_t deepDataHandle);

void
aovAccumMaterialAovs(uniform PbrTLState * uniform pbrTls,
                     const uniform AovSchema &aovSchema,
                     const uniform LightAovs &lightAovs,
                     const uniform MaterialAovs &materialAovs,
                     const varying Intersection &isect,
                     const varying RayDifferential &ray,
                     uniform intptr_t scene,
                     const varying Bsdf &bsdf,
                     const varying Color &ssAov,
                     const varying BsdfSampler * uniform bSampler,
                     const varying BsdfSample * uniform bsmps,
                     varying float pixelWeight,
                     varying uint32_t pixel,
                     varying uint32_t deepDataHandle,
                     varying int lpeStateId)
{
    // It's easier to figure this out here per-lane, rather than in CPP_aovSetMaterialAovs().
    varying uint32_t isPrimaryRay = (Ray_getDepth(ray) == 0) ? 1 : 0;

    CPP_aovSetMaterialAovs(pbrTls,
                           aovSchema,
                           lightAovs,
                           materialAovs,
                           isect,
                           ray,
                           scene,
                           bsdf,
                           ssAov,
                           bSampler,
                           bsmps,
                           /* bsdfSlice = */ nullptr,
                           (uniform float * uniform) &pixelWeight,
                           (uniform uint32_t * uniform) &pixel,
                           (uniform uint32_t * uniform) &deepDataHandle,
                           (uniform int * uniform) &lpeStateId,
                           (uniform uint32_t * uniform) &isPrimaryRay,
                           lanemask());
}

void
aovAccumMaterialAovs(uniform PbrTLState * uniform pbrTls,
                     const uniform AovSchema &aovSchema,
                     const uniform LightAovs &lightAovs,
                     const uniform MaterialAovs &materialAovs,
                     const varying Intersection &isect,
                     const varying RayDifferential &ray,
                     uniform intptr_t scene,
                     const varying Bsdf &bsdf,
                     const varying Color &ssAov,
                     const varying BsdfSlice &bsdfSlice,
                     varying float pixelWeight,
                     varying uint32_t pixel,
                     varying uint32_t deepDataHandle,
                     varying int lpeStateId)
{
    // It's easier to figure this out here per-lane, rather than in CPP_aovSetMaterialAovs().
    varying uint32_t isPrimaryRay = (Ray_getDepth(ray) == 0) ? 1 : 0;

    CPP_aovSetMaterialAovs(pbrTls,
                           aovSchema,
                           lightAovs,
                           materialAovs,
                           isect,
                           ray,
                           scene,
                           bsdf,
                           ssAov,
                           /* bSampler = */ nullptr,
                           /* bsmps = */ nullptr,
                           &bsdfSlice,
                           (uniform float * uniform) &pixelWeight,
                           (uniform uint32_t * uniform) &pixel,
                           (uniform uint32_t * uniform) &deepDataHandle,
                           (uniform int * uniform) &lpeStateId,
                           (uniform uint32_t * uniform) &isPrimaryRay,
                           lanemask());
}

void
aovAccumLightAovs(uniform PbrTLState * uniform pbrTls,
                  const uniform AovSchema &aovSchema,
                  const uniform LightAovs &lightAovs,
                  const varying Color &value,
                  varying int lpeStateId,
                  varying uint32_t pixel,
                  varying uint32_t deepDataHandle)
{
    foreach_active(lane) {
        const uniform Color uValue = { extract(value.r, lane), extract(value.g, lane), extract(value.b, lane) };
        const uniform int uStateId = extract(lpeStateId, lane);
        const uniform uint32_t uPixel = extract(pixel, lane);
        const uniform uint32_t uDeepDataHandle = extract(deepDataHandle, lane);

        CPP_aovAccumLightAovs(pbrTls, aovSchema, lightAovs, uValue, uStateId, uPixel, 
                              uDeepDataHandle);
    }
}

void
aovAccumPostScatterExtraAovs(uniform PbrTLState * uniform pbrTls,
                             const uniform FrameState &fs,
                             const varying RayState * uniform rayState,
                             const varying Bsdf &bsdf)
{
    const uniform BsdfExtraAovs &extraAovs = Bsdf_getPostScatterExtraAovs(&bsdf);
    const uniform int numExtraAovs = extraAovs.mNum;
    const PathVertex &pv = rayState->mPathVertex;
    if (numExtraAovs > 0 && pv.lpeStateId != -1) {
        const uniform int * uniform labelIds = extraAovs.mLabelIds;
        const varying Color * uniform colors = extraAovs.mColors;
        for (uniform int i = 0; i < numExtraAovs; ++i) {
            const varying Color sampleValue = colors[i];
            const varying Color value = colors[i] * pv.pathThroughput;
            const uniform AovSchema &aovSchema = *((const uniform AovSchema * uniform) fs.mAovSchema); 
            const uniform LightAovs &lightAovs = *((const uniform LightAovs * uniform) fs.mLightAovs);
            foreach_active(lane) {
                const uniform Color uSampleValue = {
                    extract(sampleValue.r, lane),
                    extract(sampleValue.g, lane),
                    extract(sampleValue.b, lane)
                };
                const uniform Color uValue = {
                    extract(value.r, lane),
                    extract(value.g, lane),
                    extract(value.b, lane)
                };
                const uniform int uStateId = extract(pv.lpeStateId, lane);
                const uniform uint32_t uPixel = extract(rayState->mSubpixel.mPixel, lane);
                const uniform uint32_t uDeepDataHandle = extract(rayState->mDeepDataHandle, lane);

                CPP_aovAccumPostScatterExtraAov(pbrTls,
                                                aovSchema,
                                                lightAovs,
                                                uValue,
                                                uSampleValue,
                                                uStateId,
                                                labelIds[i],
                                                uPixel,
                                                uDeepDataHandle);
            }
        }
    }
}

// ---------------------------------------------------------------------------
uniform bool
LightAovs_hasEntries(const uniform LightAovs * uniform lightAovs)
{
    // Need to keep in sync with Aov.h
    static const uniform int AOV_SCHEMA_ID_VISIBILITY_AOV = 900;
    static const uniform int AOV_SCHEMA_ID_LIGHT_AOV = 1000;

    return lightAovs->mNextLightAovSchemaId > AOV_SCHEMA_ID_LIGHT_AOV ||
           lightAovs->mNextVisibilityAovSchemaId > AOV_SCHEMA_ID_VISIBILITY_AOV;
}

uniform bool
LightAovs_hasVisibilityEntries(const uniform LightAovs * uniform lightAovs)
{
    // Need to keep in sync with Aov.h
    static const uniform int AOV_SCHEMA_ID_VISIBILITY_AOV = 900;

    return lightAovs->mNextVisibilityAovSchemaId > AOV_SCHEMA_ID_VISIBILITY_AOV;
}

varying int
LightAovs_scatterEventTransition(const uniform LightAovs * uniform lightAovs,
                                 varying int lpeStateId,
                                 const varying Bsdf &bsdf,
                                 const varying BsdfLobe &lobe)
{
    MNRY_ASSERT(lightAovs->mFinalized);
    if (!LightAovs_hasEntries(lightAovs)) return -1;
    if (lpeStateId < 0) return lpeStateId;

    const uniform int lobeType = BsdfLobe_getType(&lobe);

    // Construct an event and event scattering type based on the lobe type
    uniform LpeEventType ev = LPE_EVENT_TYPE_NONE;
    if (lobeType & BSDF_LOBE_TYPE_REFLECTION) ev = LPE_EVENT_TYPE_REFLECTION;
    else if (lobeType & BSDF_LOBE_TYPE_TRANSMISSION) ev = LPE_EVENT_TYPE_TRANSMISSION;

    uniform LpeEventScatteringType evs = LPE_EVENT_SCATTERING_TYPE_NONE;
    if (lobeType & BSDF_LOBE_TYPE_DIFFUSE) evs = LPE_EVENT_SCATTERING_TYPE_DIFFUSE;
    else if (lobeType & BSDF_LOBE_TYPE_GLOSSY) evs = LPE_EVENT_SCATTERING_TYPE_GLOSSY;
    else if (lobeType & BSDF_LOBE_TYPE_MIRROR) evs = LPE_EVENT_SCATTERING_TYPE_MIRROR;

    MNRY_ASSERT(ev != LPE_EVENT_TYPE_NONE);
    MNRY_ASSERT(evs != LPE_EVENT_SCATTERING_TYPE_NONE);

    // labels
    uniform int labelId = LPE_noLabel;
    const uniform int lobeLabelId = BsdfLobe_getLabel(&lobe);
    if (aovLabelIsTransformed(lobeLabelId)) {
        labelId = aovDecodeLpeLabel(lobeLabelId);
    } else {
        // no lobe label assigned, will use the material label if assigned
        labelId = Bsdf_getLpeMaterialLabelId(&bsdf);
    }

    return LpeStateMachine_transition(&lightAovs->mLpeStateMachine, lpeStateId, ev, evs, labelId);
}

varying int
LightAovs_lightEventTransition(const uniform LightAovs * uniform lightAovs,
                               varying int lpeStateId,
                               const uniform Light * varying light)
{
    MNRY_ASSERT(lightAovs->mFinalized);
    if (!LightAovs_hasEntries(lightAovs)) return -1;
    if (lpeStateId < 0) return lpeStateId;
    if (!light) return -1;

    // labels
    const int labelId = Light_getLabelId(light);
    return LpeStateMachine_transition(&lightAovs->mLpeStateMachine, lpeStateId, LPE_EVENT_TYPE_LIGHT,
                                      LPE_EVENT_SCATTERING_TYPE_NONE, labelId);
}

varying int
LightAovs_emissionEventTransition(const uniform LightAovs * uniform lightAovs,
                                  varying int lpeStateId,
                                  const varying Bsdf &bsdf)
{
    MNRY_ASSERT(lightAovs->mFinalized);
    if (!LightAovs_hasEntries(lightAovs)) return -1;
    if (lpeStateId < 0) return lpeStateId;

    // labels
    // no lobe labels for emission, only consider the material label, if
    // assigned
    const uniform int labelId = Bsdf_getLpeMaterialLabelId(&bsdf);

    return LpeStateMachine_transition(&lightAovs->mLpeStateMachine, lpeStateId, LPE_EVENT_TYPE_EMISSION,
                                      LPE_EVENT_SCATTERING_TYPE_NONE, labelId);
}

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(AovSchema);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(MaterialAovs);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(LightAovs);

export uniform uint32_t
AovSchema_hudValidation(uniform bool verbose)
{
    AOV_SCHEMA_VALIDATION;
}

export uniform uint32_t
MaterialAovs_hudValidation(uniform bool verbose)
{
    MATERIAL_AOVS_VALIDATION;
}

export uniform uint32_t
LightAovs_hudValidation(uniform bool verbose)
{
    LIGHT_AOVS_VALIDATION;
}

//----------------------------------------------------------------------------

