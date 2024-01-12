// Copyright 2023-2024 DreamWorks Animation LLC and Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Ray.hh"
#include "Types.h"
#include "Util.h"

#include <moonray/common/mcrt_macros/moonray_nonvirt_baseclass.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Xform.h>

#include <embree4/rtcore.h>

namespace scene_rdl2 {

namespace rdl2 {
class Light;
}
}

namespace moonray {

namespace mcrt_common {

struct RayExtension
{
    MCRT_COMMON_RAY_EXTENSION_MEMBERS;

    RayExtension():
        materialID(-1),
        depth(0),
        userData(nullptr),
        geomTls(nullptr),
        priorityMaterial0(nullptr),
        priorityMaterial1(nullptr),
        priorityMaterial2(nullptr),
        priorityMaterial3(nullptr),
        priorityMaterial0Count(0),
        priorityMaterial1Count(0),
        priorityMaterial2Count(0),
        priorityMaterial3Count(0),
        instance0OrLight(nullptr),
        instance1(nullptr),
        instance2(nullptr),
        instance3(nullptr),
        instanceAttributesDepth(0),
        volumeInstanceState(-1),
        shadowReceiverId(-1)
    {
    }

    RayExtension(int inDepth):
        materialID(-1),
        depth(inDepth),
        userData(nullptr),
        geomTls(nullptr),
        priorityMaterial0(nullptr),
        priorityMaterial1(nullptr),
        priorityMaterial2(nullptr),
        priorityMaterial3(nullptr),
        priorityMaterial0Count(0),
        priorityMaterial1Count(0),
        priorityMaterial2Count(0),
        priorityMaterial3Count(0),
        instance0OrLight(nullptr),
        instance1(nullptr),
        instance2(nullptr),
        instance3(nullptr),
        instanceAttributesDepth(0),
        volumeInstanceState(-1),
        shadowReceiverId(-1)
    {
    }

    static uint32_t hvdValidation(bool verbose)
    {
        MCRT_COMMON_RAY_EXTENSION_VALIDATION(VLEN);
    }
};

// This is meant to be a simple POD type class. If you want to inherit
// from it be aware that it doesn't have a virtual destructor.
// The implication is you shouldn't be holding onto a dynamically allocated
// instance of the derived class using a pointer to this base class as
// this will result in the derived class' destructor(s) not getting
// called on deletion.

/*! Ray structure. Contains all information about a ray including
 *  precomputed reciprocal direction. */
MOONRAY_START_INHERIT_FROM_NONVIRTUAL_BASECLASS
struct ALIGN(16) Ray
{
    /*! Default construction does nothing. */
    __forceinline Ray()
        : tnear(0.0f),
        time(0),
        tfar(FLT_MAX),
        mask(-1),
        id(0),
        primID(-1),
        geomID(-1),
        instID(-1)
    {
    }

    /*! Constructs a ray from origin, direction, and ray segment. Near
     *  has to be smaller than far. */
    __forceinline Ray(const scene_rdl2::math::Vec3f& inOrg, const scene_rdl2::math::Vec3f& inDir,
            float inTnear = scene_rdl2::math::zero, float inTfar = scene_rdl2::math::sMaxValue,
            float inTime = scene_rdl2::math::zero, int inDepth = scene_rdl2::math::zero,
            int inMask = -1) :
        org(inOrg),
        tnear(inTnear),
        dir(inDir),
        time(inTime),
        tfar(inTfar),
        mask(inMask),
        id(0),
        primID(-1),
        geomID(-1),
        instID(-1),
        ext(inDepth)
    {
    }

    __forceinline Ray(const Ray &inParent,
                      float inTnear, float inTfar = scene_rdl2::math::sMaxValue) :
        Ray(inParent.org, inParent.dir,
            inTnear, inTfar,
            inParent.time, inParent.ext.depth + 1,
            inParent.mask)
    {
    }

    finline const scene_rdl2::math::Vec3f& getOrigin() const {
        return org;
    }

    finline const scene_rdl2::math::Vec3f& getDirection() const {
        return dir;
    }

    finline void setDirection(const scene_rdl2::math::Vec3f &direction) {
        dir = direction;
    }

    finline float getTime() const { return time; }

    finline float getStart() const { return tnear; }

    finline float getEnd() const { return tfar; }

    /// Camera rays have a depth of 0
    finline int getDepth() const          {  return ext.depth;  }
    finline void setDepth(int rayDepth)   {  ext.depth = rayDepth; }

    finline void setMask(const int m) { mask = m; }

    finline int getMask() const { return mask; }
    /// Sets the geometric normal.
    finline void setNg(const scene_rdl2::math::Vec3f& vec) { Ng = vec; }

    finline void setNg(const float x, const float y, const float z) {
        Ng = scene_rdl2::math::Vec3f(x, y, z);
    }

    /// Gets the geometric normal.
    finline const scene_rdl2::math::Vec3f& getNg() const {
        return Ng;
    }

    // query whether the ray hit an instancing object
    finline bool isInstanceHit() const {
        return instID == geomID;
    }

    /// Debugging
    std::ostream& print(std::ostream& cout) const;

    bool isValid() const;

public:
    MCRT_COMMON_RAY_MEMBERS;
};

//----------------------------------------------------------------------------

///
/// @class RayDifferential RayDifferential.h <shading/RayDifferential.h>
/// @brief Tracks a ray and its differentials through reflections/refractions.
///
/// This class computes and tracks ray differentials along the ray
/// as it travels through the scene, reflects and refracts. It is
/// used to compute a ray's position, direction and footprint wrt.
/// neighboring rays initiating from neighboring pixels.
/// Note: We always keep track of differentials assuming a single ray per
/// pixel, or a single ray over a scattering domain. Ray differetnials are
/// scaled down by the renderer when they are used, according to ray splitting.
///
/// WARNING: Using the "set" API of the parent class invalidates the
/// ray differentials in the child class. Ray differentials are valid
/// only if the RayDifferential is traced using the methods of this class.
///
/// For details see:
///     "Tracing ray differentials - Homan Igehy,
///      Computer Graphics #33 (Annual Conference Series 1999)"
///
/// For additional details on ray differentials for path tracing, see:
///     "Path Differentials and Applications - F. Suykens & Y.D. Willems,
///      (Make sure to look at the tech-report, which has additional derivations)"
///
class RayDifferential : public Ray
{
public:
    // Bits 8-15 inclusive are allocated for RayDifferential.
    enum
    {
        HAS_DIFFERENTIALS   = 1 << 8,
    };

    /// Constructor / Destructor

    // Needs a default constructor due to being contained in a RayState.
    RayDifferential()
    {
    }

    RayDifferential(const scene_rdl2::math::Vec3f &origin, const scene_rdl2::math::Vec3f &direction,
                    const scene_rdl2::math::Vec3f &originX, const scene_rdl2::math::Vec3f &directionX,
                    const scene_rdl2::math::Vec3f &originY, const scene_rdl2::math::Vec3f &directionY,
                    float start, float end = scene_rdl2::math::sMaxValue,
                    float rayTime = 0, int rayDepth = 0) :
        Ray(origin, direction, start, end, rayTime, rayDepth),
        mOriginX(originX),
        mDirX(directionX),
        mOriginY(originY),
        mDirY(directionY),
        mOrigTfar(end)
    {
        mFlags.set(HAS_DIFFERENTIALS);
    }

    /// Create a n-ary ray differential by making a copy of a parent ray
    RayDifferential(const RayDifferential &parent, float start, float end = scene_rdl2::math::sMaxValue) :
        Ray(parent, start, end),
        mOriginX(parent.mOriginX),
        mDirX(parent.mDirX),
        mOriginY(parent.mOriginY),
        mDirY(parent.mDirY),
        mOrigTfar(parent.mOrigTfar)
    {
        mFlags.set(HAS_DIFFERENTIALS, parent.hasDifferentials());
    }

    ///
    /// Apply ray operations.
    /// Position, direction and derivatives are updated simultaneously.
    ///

    /// Advances aux ray origins to the plane described by the hit point and normal.
    /// See "Tracing Ray Differentials" - Igehy
    void transfer(const scene_rdl2::math::Vec3f &hitPoint, const scene_rdl2::math::Vec3f &hitNormal)
    {
        org = hitPoint;

        if (hasDifferentials()) {
            float denomX = dot(hitNormal, mDirX);
            float denomY = dot(hitNormal, mDirY);

            // TODO: do something more expensive here to avoid denormals?
            if (denomX != 0.0f && denomY != 0.0f) {
                float planeD = -dot(hitNormal, hitPoint);
                float distX = -(dot(mOriginX, hitNormal) + planeD) / denomX;
                float distY = -(dot(mOriginY, hitNormal) + planeD) / denomY;

                mOriginX += mDirX * distX;
                mOriginY += mDirY * distY;
            } else {
                mFlags.clear(HAS_DIFFERENTIALS);
            }
        }
    }

    ///
    /// Query ray properties
    ///
    bool hasDifferentials() const   { return mFlags.get(HAS_DIFFERENTIALS); }
    const scene_rdl2::math::Vec3f &getOriginX() const { return mOriginX; }
    const scene_rdl2::math::Vec3f &getDirX() const    { return mDirX; }
    const scene_rdl2::math::Vec3f &getOriginY() const { return mOriginY; }
    const scene_rdl2::math::Vec3f &getDirY() const    { return mDirY; }

    /// Return a float value for the footprint established by the direction differentials.
    /// TODO: there are multiple options for this calculation, so investigate which one(s) give the best results. e.g.
    /// min, max, arithmetic mean, projected area of parallelogram, etc. Here we take the geometric mean.
    ///  (We take the log2() at the end since it is used in a mip mapping calculation)
    float getDirFootprint() const
    {
        if (!hasDifferentials()) {
            return scene_rdl2::math::neg_inf;
        }
        float lx = scene_rdl2::math::length(getdDdx());
        float ly = scene_rdl2::math::length(getdDdy());
        return 0.5f * scene_rdl2::math::log2(lx * ly);  // = log2(sqrt(lx * ly))
    }

    /// Derivative of ray position wrt. pixel increments along x and y
    scene_rdl2::math::Vec3f getdPdx() const   { return mOriginX - getOrigin(); }
    scene_rdl2::math::Vec3f getdPdy() const   { return mOriginY - getOrigin(); }

    /// Derivative of  ray normalized direction wrt. pixel increments
    /// along x and y
    scene_rdl2::math::Vec3f getdDdx() const   { return mDirX - getDirection();  }
    scene_rdl2::math::Vec3f getdDdy() const   { return mDirY - getDirection();  }

    void scaleDifferentials(float scale)
    {
        scene_rdl2::math::Vec3f const &origin = getOrigin();
        scene_rdl2::math::Vec3f const &direction = getDirection();

        mOriginX = origin + (mOriginX - origin) * scale;
        mOriginY = origin + (mOriginY - origin) * scale;
        mDirX = normalize(direction + (mDirX - direction) * scale);
        mDirY = normalize(direction + (mDirY - direction) * scale);
    }

    ///
    /// Set ray properties
    ///
    void clearDifferentials()       { return mFlags.clear(HAS_DIFFERENTIALS); }
    void setOriginX(const scene_rdl2::math::Vec3f &originX) { mOriginX = originX; }
    void setDirX(const scene_rdl2::math::Vec3f &dirX) { mDirX = dirX; }
    void setOriginY(const scene_rdl2::math::Vec3f &originY) { mOriginY = originY; }
    void setDirY(const scene_rdl2::math::Vec3f &dirY) { mDirY = dirY; }

    /// The original tfar of the ray.  ray.tfar is modified during the intersections
    ///  so this keeps track of the original value.
    void setOrigTfar(float origTfar) { mOrigTfar = origTfar; }
    float getOrigTfar() const { return mOrigTfar; }

    /// Debugging
    std::ostream& print(std::ostream& cout) const;

    bool isValid() const;

    // HVD validation.
    static uint32_t hvdValidation(bool verbose) { MCRT_COMMON_RAY_DIFFERENTIAL_VALIDATION(VLEN); }

private:
    MCRT_COMMON_RAY_DIFFERENTIAL_MEMBERS;
};

MOONRAY_FINISH_INHERIT_FROM_NONVIRTUAL_BASECLASS

template<int N>
struct RayPacket : public RTCRayHitNt<N>
{
    RayExtension ext[N];
};

/*! Outputs ray to stream. */
finline std::ostream& operator<<(std::ostream& cout, const Ray& ray)
{
    return cout << "{ " <<
        "org = " << ray.org
        << ", dir = " << ray.dir
        << ", near = " << ray.tnear
        << ", far = " << ray.tfar
        << ", time = " << ray.time
        << ", mask = " << ray.mask
        << ", Ng = " << ray.Ng
        << ", u = " << ray.u
        << ", v = " << ray.v
        << ", geomID = " << ray.geomID
        << ", primID = " << ray.primID
        << ", instID = " << ray.instID
        << ", depth = " << ray.ext.depth
        << " }";
}

finline std::ostream& operator<<(std::ostream& outs, const RayDifferential& rd)
{
    return rd.print(outs);
}

struct CACHE_ALIGN RayDifferentialv
{
    uint8_t mPlaceholder[sizeof(RayDifferential) * VLEN];
};

MNRY_STATIC_ASSERT(sizeof(RayDifferential) * VLEN == sizeof(RayDifferentialv));
struct IntersectContext
{
    IntersectContext(): mRayExtension(nullptr)
    {
        rtcInitRayQueryContext(&mRtcContext);
    }

    RTCRayQueryContext mRtcContext; // this should always be the first member!
    RayExtension* mRayExtension;
};

} // namespace mcrt_common
} // namespace moonray

