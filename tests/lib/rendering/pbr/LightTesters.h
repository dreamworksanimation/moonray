// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// @file LightTesters.h
//

#pragma once
#include <moonray/rendering/pbr/light/CylinderLight.h>
#include <moonray/rendering/pbr/light/DiskLight.h>
#include <moonray/rendering/pbr/light/EnvLight.h>
#include <moonray/rendering/pbr/light/MeshLight.h>
#include <moonray/rendering/pbr/light/RectLight.h>
#include <moonray/rendering/pbr/light/SphereLight.h>
#include <moonray/rendering/pbr/light/SpotLight.h>
#include <moonray/rendering/pbr/light/DistantLight.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/shading/Util.h>

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

//
// The general approach to testing lights is to write a simple version of each
// light, one per real light, which does simple uniform sampling over the entire
// surface area. This will give a reference value which the optimized lights
// will need to match for them to pass the tests. For cases where we are texture
// mapping the lights, the approach still works. In this case the test version
// of each light will do the uniform sampling as before but will lookup the
// texture at the sample point instead of returning a constant color.
// The associated pdf is still uniform.
//
// The various light testers have intimate knowledge of the light types they are
// testing. In fact, each real light type declares the corresponding light
// tester as a friend. This is done to avoid reimplementing the entire
// functionality from scratch, e.g. we can reuse the cached data and various
// transform in each light. The main things tested here are that the importance
// sampling functions are unbiased, and the sampling and intersection functions
// are consistent with each other.
//

class LightTester
{
public:
    struct Intersection
    {
        scene_rdl2::math::Vec3f   mIllumPoint;    // original point we want to illuminate
        scene_rdl2::math::Vec3f   mWi;            // direction from point to light
        scene_rdl2::math::Vec3f   mHitPoint;      // hit point on light surface
        scene_rdl2::math::Vec3f   mNormal;        // normal at hit point on light surface
        float   mDistance;      // distance traveled to get to mHitPoint

        void init(const scene_rdl2::math::Vec3f &illumPoint, const scene_rdl2::math::Vec3f &hitPoint, const scene_rdl2::math::Vec3f &normal)
        {
            mIllumPoint = illumPoint;
            mWi = hitPoint - illumPoint;
            mHitPoint = hitPoint;
            mNormal = normal;
            mDistance = mWi.length();

            if (mDistance > 0.f) {
                mWi /= mDistance;
            }
        }
    };

    LightTester(std::shared_ptr<const Light> light, const char *desc,
                bool infinite, unsigned isectDataFieldsUsed) :
        mLight(light),
        mDesc(desc),
        mInfinite(infinite),
        mIsectDataFieldsUsed(isectDataFieldsUsed) {}

    virtual ~LightTester() {}

    const char *getDesc() const             { return mDesc;}
    bool isInfinite() const                 { return mInfinite; }
    unsigned getIsectDataFieldsUsed() const { return mIsectDataFieldsUsed;}

    // Returns the total world space area of the light.
    // Used by TestPDF.
    float getSurfaceArea() const
    {
        return getLight()->mArea;
    }

    // Returns the world space surface area on the light which is visible to point p.
    // Used by TestPDF.
    virtual float getVisibleSurfaceArea(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n) const
    {
        return getLight()->mArea;
    }

    /// Derived classes must override this if they want to run the testLightRadiance
    /// unit test.
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const 
    {
        return false;
    }

    scene_rdl2::math::Color eval(const Intersection &isect, float *pdf) const
    {
        MNRY_ASSERT(mLight && pdf);

        // Area to solid angle conversion if needed
        *pdf = mInfinite  ?  1.0f  :
                absAreaToSolidAngleScale(isect.mWi, isect.mNormal, isect.mDistance);
        *pdf /= getSurfaceArea();

        bool backSided = mInfinite  ?  false  :
                dot(isect.mWi, isect.mNormal) > 0.f;
        scene_rdl2::math::Color result = scene_rdl2::math::zero;

        if (!backSided && *pdf != 0.f) {
            // call into the real lights eval function to retrieve the textured
            // color value at this intersection point, but keep the pdf we've
            // already computed
            LightIntersection lightIsect;
            LightFilterRandomValues filterR = {scene_rdl2::math::Vec2f(0.f, 0.f), scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)};
            if (getLight()->intersect(isect.mIllumPoint, nullptr, isect.mWi, 0.f, isect.mDistance + 1.f, lightIsect)) {
                result = getLight()->eval(nullptr, isect.mWi, isect.mIllumPoint, filterR, 0.f, lightIsect, false,
                                          nullptr,  0.0f);
            }
        }

        return result;
    }

    virtual const Light *getLight() const   {  return mLight.get();  }
    virtual const char *getLightTypeName() const = 0;

protected:
    std::shared_ptr<const Light> mLight;
    const char *mDesc;
    bool mInfinite;     ///< Set to true for conceptually "infinitely distant"
                        ///< lights such as EnvLight and DistantLight.
    unsigned mIsectDataFieldsUsed;
};

//----------------------------------------------------------------------------

class RectLightTester : public LightTester
{
public:
    RectLightTester(std::shared_ptr<const RectLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, false, 0) {}

    // simple uniform sampling
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const override
    {
        const RectLight *light = getLight();

        float halfWidth  = light->mHalfWidth;
        float halfHeight = light->mHalfHeight;

        scene_rdl2::math::Vec3f hit;
        hit.x = scene_rdl2::math::lerp(halfWidth, -halfWidth, r0);
        hit.y = scene_rdl2::math::lerp(halfHeight, -halfHeight, r1);
        hit.z = 0.f;

        isect->init(p, light->xformPointLocal2Render(hit, 0.f), light->getDirection(0.f));

        return true;
    }

    virtual const RectLight *getLight() const override
    {
        return static_cast<const RectLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "RectLight";
    }
};

//----------------------------------------------------------------------------

class CylinderLightTester : public LightTester
{
public:
    CylinderLightTester(std::shared_ptr<const CylinderLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, false, 0) {}

    // simple uniform sampling
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const override
    {
        const CylinderLight *light = getLight();

        const scene_rdl2::math::Vec3f localP = light->uv2local(scene_rdl2::math::Vec2f(r0, r1));
        const scene_rdl2::math::Vec3f localN = scene_rdl2::math::Vec3f(localP.x, 0.0f, localP.z);

        const scene_rdl2::math::Vec3f renderP = light->xformPointLocal2Render(localP, 0.f);
        const scene_rdl2::math::Vec3f renderN = normalize(light->xformNormalLocal2Render(localN, 0.f));

        // We don't cast light from points not visible from p.
        scene_rdl2::math::Vec3f wi = normalize(renderP - p);
        if (dot(renderN, wi) > -scene_rdl2::math::sEpsilon) {
            return false;
        }

        isect->init(p, renderP, renderN);
        return true;
    }

    virtual const CylinderLight *getLight() const override
    {
        return static_cast<const CylinderLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "CylinderLight";
    }
};

//----------------------------------------------------------------------------

class DiskLightTester : public LightTester
{
public:
    DiskLightTester(std::shared_ptr<const DiskLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, false, 0) {}

    // simple uniform sampling
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const override
    {
        const DiskLight *light = getLight();

        scene_rdl2::math::Vec3f hit;
        squareSampleToCircle(r0, r1, &hit.x, &hit.y);
        hit.z = 0.f;

        isect->init(p, light->xformPointLocal2Render(hit, 0.f), light->getDirection(0.f));

        return true;
    }

    virtual const DiskLight *getLight() const override
    {
        return static_cast<const DiskLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "DiskLight";
    }
};

//----------------------------------------------------------------------------

class SphereLightTester : public LightTester
{
public:
    SphereLightTester(std::shared_ptr<const SphereLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, false, 1) {}

    virtual float getVisibleSurfaceArea(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n) const override
    {
        const SphereLight *light = getLight();

        // Transform point into unit sphere space. We are taking advantage of sphere
        // symmetry here.
        scene_rdl2::math::Vec3f localP = (p - light->getPosition(0.f)) * light->getInvRadius();

        // Compute spherical cap on unit sphere and rescale based on world space
        // sphere radius.
        float distToLightSq = localP.lengthSqr();
        float cosTheta = 1.0f / scene_rdl2::math::sqrt(distToLightSq);
        return (1.0f - cosTheta) * scene_rdl2::math::sTwoPi * light->mRadiusSqr;
    }

    // simple uniform sampling
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const override
    {
        const SphereLight *light = getLight();

        scene_rdl2::math::Vec3f localP = shading::sampleSphereUniform(r0, r1);

        const scene_rdl2::math::Vec3f renderP = light->xformPointLocal2Render(localP, 0.f);
        const scene_rdl2::math::Vec3f renderN = light->xformVectorLocal2RenderRot(localP, 0.f);

        // We don't cast light from points not visible from p.
        scene_rdl2::math::Vec3f wi = normalize(renderP - p);
        if (dot(renderN, wi) > -scene_rdl2::math::sEpsilon) {
            return false;
        }

        isect->init(p, renderP, renderN);

        return true;
    }

    virtual const SphereLight *getLight() const override
    {
        return static_cast<const SphereLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "SphereLight";
    }
};

//----------------------------------------------------------------------------

class SpotLightTester : public LightTester
{
public:
    SpotLightTester(std::shared_ptr<const SpotLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, false, 0) {}

    // simple uniform sampling
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const override
    {
        const SpotLight *light = getLight();

        scene_rdl2::math::Vec3f localSample;
        squareSampleToCircle(r0, r1, &localSample.x, &localSample.y);
        localSample.x *= light->mLensRadius;
        localSample.y *= light->mLensRadiusY;
        localSample.z = 0.0f;

        isect->init(p, 
                    light->xformPointLocal2Render(localSample, 0.f),
                    light->getDirection(0.f));

        return true;
    }

    virtual const SpotLight *getLight() const override
    {
        return static_cast<const SpotLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "SpotLight";
    }
};

//----------------------------------------------------------------------------

// We currently skip running the testLightRadiance on this light type, but run
// the other tests.
class DistantLightTester : public LightTester
{
public:
    DistantLightTester(std::shared_ptr<const DistantLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, true, 0) {}

    virtual const DistantLight *getLight() const override
    {
        return static_cast<const DistantLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "DistantLight";
    }
};

//----------------------------------------------------------------------------

class EnvLightTester : public LightTester
{
public:
    EnvLightTester(std::shared_ptr<const EnvLight> light, const char *desc) :
        LightTester(std::static_pointer_cast<const Light>(light), desc, true, 0) {}

    // simple uniform sampling
    virtual bool sample(const scene_rdl2::math::Vec3f &p, float r0, float r1, Intersection *isect) const override
    {
        scene_rdl2::math::Vec3f wi;
        if (getLight()->mHemispherical) {
            wi = shading::sampleLocalHemisphereUniform(r0, r1);
            wi = getLight()->mFrame.localToGlobal(wi);
        } else {
            wi = shading::sampleSphereUniform(r0, r1);
        }
        isect->init(scene_rdl2::math::zero, wi, -wi);
        isect->mDistance = FLT_MAX;
        return true;
    }

    virtual const EnvLight *getLight() const override
    {
        return static_cast<const EnvLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "EnvLight";
    }
};

class MeshLightTester : public LightTester
{
public:
    MeshLightTester(std::shared_ptr<const MeshLight> light, const char *desc) :
       LightTester(std::static_pointer_cast<const Light>(light), desc, true, 0)
    {
    }

    virtual ~MeshLightTester() override {

    }

    virtual const MeshLight *getLight() const override
    {
        return static_cast<const MeshLight *>(mLight.get());
    }

    virtual const char *getLightTypeName() const override
    {
        return "MeshLight";
    }

    float getIntegratedPdf(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &n) const {
        const MeshLight* meshLight = getLight();
        size_t faceCount = meshLight->mFaceCount;
        float pdf = 0.f;
        for (unsigned face = 0; face < faceCount; ++face) {
            int nodeIndex = meshLight->mPrimIDToNodeID[face];
            pdf += meshLight->getPdfOfFace(nodeIndex, p, &n);
        }

        return pdf;
    }
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

