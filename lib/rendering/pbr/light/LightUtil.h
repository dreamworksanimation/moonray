// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "LightUtil.hh"
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/platform/HybridUniformData.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <embree4/rtcore.h>


// Forward declaration of the ISPC types
namespace ispc {
    struct Plane;
    struct FalloffCurve;
    struct OldFalloffCurve;
}

//----------------------------------------------------------------------------

namespace moonray {

namespace shading {
    class Bsdf;
    class Intersection;
}

namespace pbr {

struct LightIntersection {
    scene_rdl2::math::Vec3f N;              ///< normal at hit point on light surface
    float distance;             ///< distance traveled along ray to the hit point
    scene_rdl2::math::Vec2f uv;             ///< used for texture mapped lights
    float mipLevel;             ///< used for texture mapped lights
    float data[2];              ///< light type specific data passed to eval from
                                ///< intersect or sample
    float pdf;                  ///< pdf of sample
    int   primID;               ///< primID of face on mesh light
    int   geomID;               ///< geomID of geometry on mesh light
};

class Scene;
class Light;
typedef std::vector<Light *> LightPtrList;
class LightFilterList;
typedef std::vector<LightFilterList *> LightFilterLists;
typedef std::vector<std::unique_ptr<LightFilterList>> LightFilterListsUniquePtrs;

class LightSet;


enum VisibleInCamera
{
    VISIBLE_IN_CAMERA_OFF,          ///< Never render light geometry for this light.
    VISIBLE_IN_CAMERA_ON,           ///< Always render light geometry for this light.
    VISIBLE_IN_CAMERA_USE_GLOBAL,   ///< Use "debug visualize lights" from scene vars.
};

enum PresenceShadows {
    PRESENCE_SHADOWS_OFF,           ///< Presence shadows off for this light.
    PRESENCE_SHADOWS_ON,            ///< Presence shadows on for this light.
    PRESENCE_SHADOWS_USE_GLOBAL,    ///< Use "enable presence shadows" from scene vars.
};

//----------------------------------------------------------------------------

// TODO: flesh this out and move it into common/math?
struct Plane
{
    Plane() {}
    Plane(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n) :
        mN(n),
        mOffset(-dot(p, mN))
    {
    }

    // Specify verts in counter-clockwise order relative to the desired normal.
    Plane(const scene_rdl2::math::Vec3f &p1, const scene_rdl2::math::Vec3f &p2, const scene_rdl2::math::Vec3f &p3) :
        mN(normalize(cross(p2-p1, p3-p1))),
        mOffset(-dot(p1, mN))
    {
    }

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        PLANE_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(Plane);


    /// Returns the signed distance from a point to a line.
    float getDistance(const scene_rdl2::math::Vec3f &p) const
    {
        return dot(p, mN) + mOffset;
    }

    /// The plane is formed by the equation: dot(mN, p) + mOffset = 0.
    PLANE_MEMBERS;
};

//----------------------------------------------------------------------------

enum FalloffCurveType
{
    FALLOFF_CURVE_TYPE_ENUM
};


class FalloffCurve
{
public:
    FalloffCurve() :
        mType(FALLOFF_CURVE_TYPE_LINEAR) {}

    FalloffCurve(FalloffCurveType curveType)
    {
        init(curveType);
    }

    void init(FalloffCurveType curveType)
    {
        mType = (FalloffCurveType)scene_rdl2::math::min(curveType, FALLOFF_CURVE_TYPE_NUM_TYPES - 1);
    }

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose);
    HUD_AS_ISPC_METHODS(FalloffCurve);


    float eval(float t) const
    {
        t = scene_rdl2::math::saturate(t);

        switch (mType) {
        case FALLOFF_CURVE_TYPE_NONE:
            return (t > 0.0f) ? 1.0f : 0.0f;
        case FALLOFF_CURVE_TYPE_LINEAR:
            return t;
        case FALLOFF_CURVE_TYPE_EASEIN:
            return t * t;
        case FALLOFF_CURVE_TYPE_EASEOUT:
            return t * (2.0f - t);
        case FALLOFF_CURVE_TYPE_EASEINOUT:
            return t * t * (3.0f - 2.0f * t);
        default:
            return t;
        }
    }

protected:
    FALLOFF_CURVE_MEMBERS;

};

//----------------------------------------------------------------------------

enum OldFalloffCurveType
{
    OLD_FALLOFF_CURVE_TYPE_ENUM
};


class OldFalloffCurve
{
public:
    OldFalloffCurve() :
        mType(OLD_FALLOFF_CURVE_TYPE_NONE), mExp(1.0f), mG1(0.0f), mInvH0(0.0f) {}

    OldFalloffCurve(OldFalloffCurveType curveType, float exp)
    {
        init(curveType, exp);
    }

    void init(OldFalloffCurveType curveType, float exp)
    {
        mType = (OldFalloffCurveType)scene_rdl2::math::min(curveType, OLD_FALLOFF_CURVE_TYPE_NUM_TYPES - 1);
        mExp = exp;
        mG1 = g(1.0f);  // initialize this before mInvH0!
        mInvH0 = 1.0f / h(0.0f);
    }

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose);
    HUD_AS_ISPC_METHODS(OldFalloffCurve);


    float eval(float t) const
    {
        t = scene_rdl2::math::saturate(t);

        switch (mType) {
        case OLD_FALLOFF_CURVE_TYPE_NONE:
            return t;
        case OLD_FALLOFF_CURVE_TYPE_EASEOUT:
            return scene_rdl2::math::pow(3.0f * t * t - 2.0f * t * t * t, mExp);
        case OLD_FALLOFF_CURVE_TYPE_GAUSSIAN:
            return scene_rdl2::math::pow((scene_rdl2::math::cos((1.0f - t) * scene_rdl2::math::sPi) + 1.0f) * 0.5f,
                                         mExp);
        case OLD_FALLOFF_CURVE_TYPE_LINEAR:
            return scene_rdl2::math::pow(t, mExp);
        case OLD_FALLOFF_CURVE_TYPE_SQUARED:
            return t * t;
        case OLD_FALLOFF_CURVE_TYPE_NATURAL:
            return naturalOldFalloff(t);
        }
        return t;
    }

protected:
    OLD_FALLOFF_CURVE_MEMBERS;

private:
    //
    // We use the following function in the [0..1] interval:
    //   natural_falloff(x) = h(x) / h(0)
    //   h(x) = g(x) - g(1)
    //   g(x) = f(tail * x + 1)
    //   f(x) = 1.0 / (x ** power)
    // Here tail and g(x) are chosen so that
    //   g'(0) is a large negative value (high slope)
    //   g'(1) is close to 0 (small slope)
    // In gnuplot:
    //   tail = 7
    //   power = 2
    //   plot [x=0:1][0:1] ((1.0 / ((tail*x+1)**power)) - (1.0 / ((tail*1+1)**power))) / ((1.0 / ((tail*0+1)**power)) -
    //                     (1.0 / ((tail*1+1)**power)))
    //
    #define OLD_FALLOFF_HEAD    1.0f
    #define OLD_FALLOFF_TAIL    7.0f

    float naturalOldFalloff(float x) const { return 1.0f - (h(x) * mInvH0); }

    float f(float x) const  { return 1.0f / scene_rdl2::math::pow(x, mExp); }
    float g(float x) const  { return f(OLD_FALLOFF_TAIL * x + OLD_FALLOFF_HEAD); }
    float h(float x) const  { return g(x) - mG1; }
};

//----------------------------------------------------------------------------

finline scene_rdl2::math::Color
computeLightRadiance(const scene_rdl2::rdl2::Light *light,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> &colorKey,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> &intensityKey,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> &exposureKey,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> &normalizedKey,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> &applySceneScaleKey,
                     float invSurfaceArea)
{
    // Compute color, intensity and exposure contributions
    scene_rdl2::math::Color color = light->get<scene_rdl2::rdl2::Rgb>(colorKey);
    float exposure = light->get<scene_rdl2::rdl2::Float>(exposureKey);
    color *= light->get<scene_rdl2::rdl2::Float>(intensityKey) * powf(2.0f, exposure);

    // When normalized, we can change the size of the emitter without
    // changing the amount of total energy cast into the scene.
    bool normalized = light->get<scene_rdl2::rdl2::Bool>(normalizedKey);

    if (!normalized) {
        // Interpret as radiance.
        return color;
    }

    // Interpret as flux
    // For backwards compatibility, we can multiply the surface area
    // by the scene scale and pi for purposes of flux computation.
    // Our goal is to eliminate this behavior (or at least default it to false)
    // so we match other renderer's implemntation of normalized lights.
    if (light->get<scene_rdl2::rdl2::Bool>(applySceneScaleKey)) {
        const scene_rdl2::rdl2::SceneVariables &sv = light->getSceneClass().getSceneContext()->getSceneVariables();
        float sceneScale = sv.get(scene_rdl2::rdl2::SceneVariables::sSceneScaleKey);
        invSurfaceArea *= 1.0f / (sceneScale * sceneScale);
        invSurfaceArea *= scene_rdl2::math::sOneOverPi;
    }

    return color * invSurfaceArea;
}

// Use this variant for lights that only work in radiance units
finline scene_rdl2::math::Color
computeLightRadiance(const scene_rdl2::rdl2::Light *light,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> &colorKey,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> &intensityKey,
                     const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> &exposureKey)
{
    // Compute color, intensity and exposure contributions
    scene_rdl2::math::Color color = light->get<scene_rdl2::rdl2::Rgb>(colorKey);
    float exposure = light->get<scene_rdl2::rdl2::Float>(exposureKey);
    color *= light->get<scene_rdl2::rdl2::Float>(intensityKey) * powf(2.0f, exposure);

    return color;
}

// Approximation which breaks down the closer p gets to the light but should
// be sufficiently accurate as the light gets more distant.
finline float
getPlanarApproxSubtendedSolidAngle(const scene_rdl2::math::Vec3f &p,  // point we are illuminating
                                   const scene_rdl2::math::Vec3f &lightPos,
                                   const scene_rdl2::math::Vec3f &lightDir,
                                   float lightArea)
{
    scene_rdl2::math::Vec3f negWi = p - lightPos;
    float distSq = negWi.lengthSqr();
    if (distSq < scene_rdl2::math::sEpsilon) {
        return scene_rdl2::math::sTwoPi;
    }
    float dist = scene_rdl2::math::sqrt(distSq);
    float solidAngle = dot(negWi, lightDir) * lightArea / (distSq * dist);
    return scene_rdl2::math::clamp(solidAngle, 0.0f, scene_rdl2::math::sTwoPi);
}

// Generate the set of active lights for a LightPtrList and position/normal
// in space.
void
computeActiveLights(scene_rdl2::alloc::Arena *arena,
                    const Scene *scene,
                    const shading::Intersection &isect,
                    const scene_rdl2::math::Vec3f *normal,
                    const shading::Bsdf &bsdf,
                    float rayTime,
                    // outputs
                    LightSet &lightSet,
                    bool &hasRayTerminatorLights);

// Randomly choose a light based on how many lights have been hit so far
bool chooseThisLight(const IntegratorSample1D &samples, int depth, unsigned int numHits);

//----------------------------------------------------------------------------

// per ray data for light intersection test
struct LightIntersectContext {
    LightIntersectContext():
        mData0(nullptr),
        mData1(nullptr),
        mDistance(nullptr),
        mFromCamera(nullptr),
        mIncludeRayTerminationLights(nullptr),
        mPdf(nullptr),
        mMeshGeomID(nullptr),
        mMeshPrimID(nullptr),
        mShadingNormal(nullptr),
        mSamples(nullptr),
        mDepth(nullptr),
        mNumHits(nullptr),
        mLightIdMap(nullptr)
    {
        rtcInitRayQueryContext(&mRtcContext);
    }

    RTCRayQueryContext mRtcContext;
    float* mData0;
    float* mData1;
    float* mDistance;
    int* mFromCamera;
    int* mIncludeRayTerminationLights;
    float* mPdf;
    int* mMeshGeomID;
    int* mMeshPrimID;
    const scene_rdl2::math::Vec3f* mShadingNormal;
    const IntegratorSample1D* mSamples;
    int* mDepth;
    int* mNumHits;
    const int* mLightIdMap;
};

struct LightOccludeContext {
    LightOccludeContext(): mSelf(nullptr)
    {
        rtcInitRayQueryContext(&mRtcContext);
    }

    RTCRayQueryContext mRtcContext;
    const Light* mSelf;
};

} // namespace pbr
} // namespace moonray

