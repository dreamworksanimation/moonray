// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Light.hh"
#include "LightUtil.h"
#include <moonray/rendering/pbr/lightfilter/LightFilter.h>
#include <moonray/rendering/pbr/core/Distribution.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/HybridUniformData.h>


// Forward declaration of the ISPC types
namespace ispc {
    struct Light;
    struct LocalParamLight;
}



namespace moonray {
namespace pbr {

// Infinity, but not quite so we can distinguish from no light hit
static const float sInfiniteLightDistance = scene_rdl2::math::sMaxValue - 1e32f;
static const float sEnvLightDistance = sInfiniteLightDistance * 0.9f;
static const float sDistantLightDistance = sInfiniteLightDistance * 0.8f;

//----------------------------------------------------------------------------


///
/// @class Light Light.h <pbr/Light.h>
/// @brief Base class that defines the light interface. All lights are assumed
///  to be area lights. All lights operate in render space.
/// 
class Light
{
    friend class LightTester;

public:
    /// Constructor / Destructor
    Light(const scene_rdl2::rdl2::Light* rdlLight);
    virtual ~Light()  {
        delete mDistribution;
    }

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(Light);


    /// Is this light active.
    bool isOn() const { return mOn; }

    /// Is this light two-sided
    bool isTwoSided() const { return mSidedness == LIGHT_SIDEDNESS_2_SIDED; }

    /// Is this light visible in camera, and if so is it opaque in the
    /// alpha channel ?
    bool getIsVisibleInCamera() const { return mIsVisibleInCamera; }
    bool getIsOpaqueInAlpha() const { return mIsOpaqueInAlpha; }

    /// Return the rdlLight
    const scene_rdl2::rdl2::Light* getRdlLight() const { return mRdlLight; }

    /// Updates the light if the light has changed.
    /// Returns false if there was a problem updating the light.
    /// If false is returned, the light is set to off.
    virtual bool update(const scene_rdl2::math::Mat4d& world2render) = 0;

    /// Render space position and direction
    finline scene_rdl2::math::Vec3f getPosition(float time) const
    {
        if (!isMb()) return mPosition[0];
        return lerpPosition(time);
    }
    finline scene_rdl2::math::Vec3f getDirection(float time) const
    {
        if (!isMb()) return mDirection;
        return slerpDirection(time);
    }

    // Attributes for light sampling acceleration

    // Get the spread of possible light directions
    virtual float getThetaO() const = 0;

    // Get the spread of possible emission directions
    virtual float getThetaE() const = 0;

    /// Get image distribution and color correction. the update() method should be
    /// called first for these to be valid.
    const ImageDistribution* getDistribution() const { return mDistribution; }
    Distribution2D::Mapping getDistributionMapping() const { return mDistributionMapping; }

    bool getPresenceShadows() const { return mPresenceShadows; }

    /// Intersection and Sampling API
    /// IMPORTANT: The API below operates entirely in render space. All positions
    /// and directions passed in / returned are in render space.

    /// Can this light illuminate point p with surface normal n (ignoring
    /// occlusion consideration)?
    /// This function is used to perform early culling checks to avoid
    /// integrating lights when they can't affect the result.
    /// Passing in the the normal is optional. We can do a better job on culling
    /// if n is present for some light types.
    /// The radius defines a spherical region around p in which case
    /// the light should not be culled if any portion of that spherical region
    /// can be illuminated by this light (this is necessary so the culling
    /// calculations are accurate with sub-surface scattering).
    virtual bool canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time,
            float radius, const LightFilterList* lightFilterList) const = 0;

    /// Compute an approximation to the total power for this light, so integrators
    /// can spend more samples towards lights that contribute more.
    // TODO: Implement this later
    /*
    virtual scene_rdl2::math::Color power() const = 0;
    */

    /// Predicate used to determine if a light can be bounded by a bounding box,
    /// and can therefore be put in an acceleration structure. Some infinite
    /// distance/area lights are not able to be bounded.
    virtual bool isBounded() const = 0;
    virtual bool isDistant() const = 0;
    virtual bool isEnv()     const = 0;
    // Specify if light is a MeshLight. If so, a few extra steps are taken to
    // initialize the mesh light.
    virtual bool isMesh() const {
        MNRY_ASSERT(mRdlLight);
        const std::string &className =  mRdlLight->getSceneClass().getName();
        return (className == "MeshLight");
    }

    /// Returns the light's axis-aligned bounding box in render space.
    ///
    /// Needs to be defined if isBounded() returns true;
    virtual scene_rdl2::math::BBox3f getBounds() const;

    /// Compute whether the ray (p, wi) intersects the light, within the given
    /// maxDistance. If the actual distance is equal to maxDistance, then we 
    /// count that as a hit. If the light is hit, the function also returns the
    /// corresponding intersection.
    ///
    /// Intersection tests are performed first against the bounded lights (cylinder,
    /// disk, rect, sphere, spot) using an Embree acceleration structure. If no
    /// such intersections are found, it will proceed to iterate over the unbounded
    /// lights (distant, env) until the first one hit.
    virtual bool intersect(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n,
                           const scene_rdl2::math::Vec3f &wi, float time,
                           float maxDistance, LightIntersection &isect) const = 0;

    /// Sample a position on the light and return true if the sample can
    /// contribute radiance to the point p, and false otherwise (i.e. if the
    /// light sample is facing away from point p or if the light was unable to
    /// draw a sample).
    ///
    /// When it returns true, the method also returns the direction wi from
    /// the point p on the surface, to the sampled position as well as the
    /// resulting light intersection.
    ///
    /// The n parameter (also optional as in canIlluminate) defines the
    /// hemisphere of directions that can be seen from point p. If non-null,
    /// Lights can test that the sampled direction is within that hemisphere
    /// and return true / false to respectively accept / cull the sample.
    ///
    /// This result assumes visibility: occlusion is only tested later if the
    /// contribution is actually significant.
    /// TODO: later we may need a 3D sample value and a component in the
    /// intersection, when dealing with geometry lights.
    virtual bool sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time,
                        const scene_rdl2::math::Vec3f& r,
                        scene_rdl2::math::Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const = 0;

    /// Evaluate the light's emitted radiance from the intersection position
    /// in the direction -wi (make sure to pass-in the same wi that was passed-in
    /// the intersect() call or returned from the sample() call above).
    /// Also compute the probability density of sampling this direction from the
    /// point p (density measured wrt. solid angle).
    /// See intersect() for more info on the fromCamera toggle.
    /// @@@ TODO: Expand interface to take texture differentials.
    ///
    /// The fromCamera toggle is set to true when visualizing lights visible
    /// in camera. Some lights require a specific implementation in this context.
    /// Most light types are unaffected. SpotLights are the current exception
    /// due to not being able to see any radiance when outside of their outer
    /// field of view.
    virtual scene_rdl2::math::Color eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi,
                                         const scene_rdl2::math::Vec3f &p, const LightFilterRandomValues& filterR,
                                         float time, const LightIntersection &isect, bool fromCamera,
                                         const LightFilterList *lightFilterList, float rayDirFootprint,
                                         float *pdf = nullptr) const = 0;

    /// Query a position from light that will be used as pivot point for
    /// equi-angular sampling (part of volume scattering integration)
    /// For most of the lights this is a uniform area sampling,
    /// but for sphere light we intentionally use sphere center as pivot
    /// For detail reference see:
    /// "Importance Sampling Techniques for Path Tracing in Participating Media"
    /// EGSR2012 Christopher Kulla and Marcos Fajardo
    virtual scene_rdl2::math::Vec3f getEquiAngularPivot(const scene_rdl2::math::Vec3f& r, float time) const = 0;

    /// TODO: Sampling from the light outwards is currently un-supported (the second
    /// Sample_L() in pbrt).

    /// Set the light aov labelId
    void setLabelId(int32_t labelId);
    /// Get the light aov labelId
    int32_t getLabelId() const { return mLabelId; }

    /// Set the global hash of the light. This is a hash value of its rdl name.
    /// It is used to create the sequence ID of the light sample. To keep the
    /// noise pattern deterministic, this value does not change from run to run.
    void setHash(uint32_t hash) { mHash = hash; }
    /// Get the global index
    uint32_t getHash() const { return mHash; }

    /// Gets the visibility mask of the light
    int getVisibilityMask() const { return mVisibilityMask; }

    /// Get the flag which identifies this light as a ray termination light
    bool getIsRayTerminator() const { return mIsRayTerminator; }

    /// Get the clear radius value of the light
    float getClearRadius() const { return mClearRadius; }

    /// Get the clear radius falloff distance
    float getClearRadiusFalloffDistance() const { return mClearRadiusFalloffDistance; }
    
    /// Get the type of shadow falloff interpolation
    int getClearRadiusInterpolationType() const { return mClearRadiusInterpolation; }

    /// Get the maximum shadow distance
    float getMaxShadowDistance() const { return mMaxShadowDistance; }

    /// Get radiance
    scene_rdl2::math::Color getRadiance() const { return mRadiance; }

    // Reject dir/pos combos depending on sidedness.
    // (Note: a 'true' return value means 'reject'.)
    finline bool rejectPosDirSidedness(float pz, float wz) const
    {
        if ((pz < 0.0f) == (wz < 0.0f)) return true;
        return (pz < 0.0f) ? mSidedness == LIGHT_SIDEDNESS_REGULAR
                           : mSidedness == LIGHT_SIDEDNESS_REVERSE;
    }

protected:

    /// Derived classes should call this method in their update() function.
    /// Update whether or not the light is visible in camera,
    /// diffuse/glossy/mirror reflection/transmission lobes
    void updateVisibilityFlags();

    /// Derived classes should call this method in their update() function.
    /// This will, if needed, (re)-load the texture (.exr, .map) and (re)-compute
    /// the sampling CDF in the mDistribution.
    /// Returns false if there was an error loading the map and true otherwise.
    /// Also returns true if there is no map to load.
    bool updateImageMap(Distribution2D::Mapping distributionMapping);

    void updatePresenceShadows();
    void updateRayTermination();
    void updateTextureFilter();
    void updateSidedness();
    void updateMaxShadowDistance();

    /// See Light.hh for details
    finline bool isMb() const { return mMb; }

    static const scene_rdl2::math::Mat4f sIdentity;
    static const scene_rdl2::math::Mat4f sRotateX180;

    LIGHT_MEMBERS;

private:
    /// Copy is disabled
    Light(const Light &other);
    const Light &operator=(const Light &other);

    scene_rdl2::math::Vec3f lerpPosition(float time) const;
    scene_rdl2::math::Vec3f slerpDirection(float time) const;
};


//----------------------------------------------------------------------------

///
/// @class LocalParamLight Light.h <pbr/Light.h>
/// @brief Intermediate class that defines a local parameterization and local to
/// render transformation.
///
class LocalParamLight : public Light
{
public:
    /// Constructor / Destructor
    LocalParamLight(const scene_rdl2::rdl2::Light* rdlLight);
    virtual ~LocalParamLight()  { }

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LOCAL_PARAM_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(LocalParamLight);


    /// Initialize the transform and parametrization members below,
    /// as well as mPosition, mOrientation, mDirection, and mMb in the base class
    ///
    /// General comment and warning!  The term "local" when used by
    /// lights and the various transformation functions does not necessarily mean
    /// the usual rdl local space of the light object.  Rather, it refers to
    /// a convenient local light space used by the light computation code.  It
    /// it probably more properly named "lightLocal" space - but that is a bit
    /// much verbosity.
    bool updateParamAndTransforms(const scene_rdl2::math::Mat4f &local2Render0, ///< at rayTime == 0
                                  const scene_rdl2::math::Mat4f &local2Render1,  ///< at rayTime == 1
                                  float halfWidth,
                                  float halfHeight);

protected:
    /// point transformations
    /// scale, orientation, and translation
    finline scene_rdl2::math::Vec3f xformPointLocal2Render(const scene_rdl2::math::Vec3f &p, float time) const
    {
        if (!isMb()) return transformPoint(mLocal2Render[0], p);

        return slerpPointLocal2Render(p, time);
    }
    finline scene_rdl2::math::Vec3f xformPointRender2Local(const scene_rdl2::math::Vec3f &p, float time) const
    {
        if (!isMb()) return transformPoint(mRender2Local[0], p);

        return slerpPointRender2Local(p, time);
    }
    finline scene_rdl2::math::Xform3f getXformRender2Local(float time, bool needed = true) const
    {
        if (!needed) {
            return scene_rdl2::math::Xform3f();
        }

        if (!isMb()) {
            return mRender2LocalRot[0];
        }

        const scene_rdl2::math::Vec3f trans = (mMb & LIGHT_MB_TRANSLATION) ?
            lerp(mRender2LocalRot[0].p, mRender2LocalRot[1].p, time) :
            mRender2LocalRot[0].p;
        const scene_rdl2::math::Mat3f rot = (mMb & LIGHT_MB_ROTATION)?
            scene_rdl2::math::Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() : // inverse is transpose
            mRender2LocalRot[0].l;  // same as Mat3f(mOrientation[0]).transpose()

        return scene_rdl2::math::Xform3f(rot, trans);
    }

    /// vector transformations
    /// scale and orientation
    finline scene_rdl2::math::Vec3f xformVectorLocal2Render(const scene_rdl2::math::Vec3f &v, float time) const
    {
        if (!isMb()) return transformVector(mLocal2Render[0], v);

        return slerpVectorLocal2Render(v, time);
    }
    finline scene_rdl2::math::Vec3f xformVectorRender2Local(const scene_rdl2::math::Vec3f &v, float time) const
    {
        if (!isMb()) return transformVector(mRender2Local[0], v);

        return slerpVectorRender2Local(v, time);
    }
    /// orientation only
    finline scene_rdl2::math::Vec3f xformVectorLocal2RenderRot(const scene_rdl2::math::Vec3f &v, float time) const
    {
        if (!isMb()) return transformVector(mLocal2RenderRot[0], v);

        return slerpVectorLocal2RenderRot(v, time);
    }
    finline scene_rdl2::math::Vec3f xformVectorRender2LocalRot(const scene_rdl2::math::Vec3f &v, float time) const
    {
        if (!isMb()) return transformVector(mRender2LocalRot[0], v);

        return slerpVectorRender2LocalRot(v, time);
    }

    /// normal transformations
    /// scale and orientation
    finline scene_rdl2::math::Vec3f xformNormalLocal2Render(const scene_rdl2::math::Vec3f &n, float time) const
    {
        if (!isMb()) return transformNormal(mRender2Local[0], n);

        return slerpNormalLocal2Render(n, time);
    }
    // orientation only
    finline scene_rdl2::math::Vec3f xformNormalLocal2RenderRot(const scene_rdl2::math::Vec3f &n, float time) const
    {
        if (!isMb()) return transformNormal(mRender2LocalRot[0], n);

        return slerpNormalLocal2RenderRot(n, time);
    }
    finline scene_rdl2::math::Vec3f xformNormalRender2LocalRot(const scene_rdl2::math::Vec3f &n, float time) const
    {
        if (!isMb()) return transformNormal(mLocal2RenderRot[0], n);

        return slerpNormalRender2LocalRot(n, time);
    }

    /// uniform scale transformations
    finline float xformLocal2RenderScale(float s, float time) const
    {
        if (!isMb()) return mLocal2RenderScale[0] * s;

        return lerpLocal2RenderScale(s, time);
    }
    finline float xformRender2LocalScale(float s, float time) const
    {
        if (!isMb()) return mRender2LocalScale[0] * s;

        return lerpRender2LocalScale(s, time);
    }

    LOCAL_PARAM_LIGHT_MEMBERS;

protected:
    virtual bool updateTransforms(const scene_rdl2::math::Mat4f &local2Render, int ti);

private:

    scene_rdl2::math::Vec3f slerpPointLocal2Render(const scene_rdl2::math::Vec3f &p, float time) const;
    scene_rdl2::math::Vec3f slerpPointRender2Local(const scene_rdl2::math::Vec3f &p, float time) const;

    scene_rdl2::math::Vec3f slerpVectorLocal2Render(const scene_rdl2::math::Vec3f &v, float time) const;
    scene_rdl2::math::Vec3f slerpVectorRender2Local(const scene_rdl2::math::Vec3f &v, float time) const;
    scene_rdl2::math::Vec3f slerpVectorLocal2RenderRot(const scene_rdl2::math::Vec3f &v, float time) const;
    scene_rdl2::math::Vec3f slerpVectorRender2LocalRot(const scene_rdl2::math::Vec3f &v, float time) const;

    scene_rdl2::math::Vec3f slerpNormalLocal2Render(const scene_rdl2::math::Vec3f &n, float time) const;
    scene_rdl2::math::Vec3f slerpNormalLocal2RenderRot(const scene_rdl2::math::Vec3f &n, float time) const;
    scene_rdl2::math::Vec3f slerpNormalRender2LocalRot(const scene_rdl2::math::Vec3f &n, float time) const;

    float lerpLocal2RenderScale(float s, float time) const;
    float lerpRender2LocalScale(float s, float time) const;
};

} // namespace pbr
} // namespace moonray

