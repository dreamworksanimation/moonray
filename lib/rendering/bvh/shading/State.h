// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file State.h
/// $Id$
///

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <scene_rdl2/scene/rdl2/Types.h>

// API Overview
// This class is part of the public API for use in shaders.
// It's main purpose is to remove the inclusion of the State
// class and all of it's parents.   This class therefore wraps
// all of methods of State that are used in shaders.

namespace moonray {
namespace shading {

class Intersection;

class State
{
public:
    State(const Intersection *intersection):
        mIntersection(intersection)
    {}

    const scene_rdl2::math::Vec3f &getP()            const;
    const scene_rdl2::math::Vec3f &getNg()           const;
    const scene_rdl2::math::Vec3f &getN()            const;
    const scene_rdl2::math::Vec3f &getdPds()         const;
    const scene_rdl2::math::Vec3f &getdPdt()         const;
    const scene_rdl2::math::Vec3f &getdNds()         const;
    const scene_rdl2::math::Vec3f &getdNdt()         const;
    const scene_rdl2::math::Vec3f &getWo()           const;
    const scene_rdl2::math::Vec2f &getSt()           const;
    const scene_rdl2::math::Vec2f &getMinRoughness() const;
    const scene_rdl2::rdl2::Geometry* getGeometryObject()   const;

    scene_rdl2::math::Vec3f getdPdx()                const;
    scene_rdl2::math::Vec3f getdPdy()                const;

    // Returns "ref_P" primitive attribute data
    bool getRefP(scene_rdl2::math::Vec3f &refP) const;

    // Returns "ref_N" primitive attribute data
    bool getRefN(scene_rdl2::math::Vec3f &refN) const;

    // Returns derivative wrt s for a Vec3f primitive attribute
    bool getdVec3fAttrds(int key,
                         scene_rdl2::math::Vec3f& dVec3fAttrds) const;

    // Returns derivative wrt t for a Vec3f primitive attribute
    bool getdVec3fAttrdt(int key,
                         scene_rdl2::math::Vec3f& dVec3fAttrdt) const;

    // Returns derivative wrt x for a Vec3f primitive attribute
    bool getdVec3fAttrdx(int key,
                         scene_rdl2::math::Vec3f& dVec3fAttrdx) const;

    // Returns derivative wrt y for a Vec3f primitive attribute
    bool getdVec3fAttrdy(int key,
                         scene_rdl2::math::Vec3f& dVec3fAttrdy) const;

    float getdSdx() const;
    float getdSdy() const;
    float getdTdx() const;
    float getdTdy() const;

    bool isProvided(shading::AttributeKey key)   const;
    bool isdsProvided(shading::AttributeKey key) const;
    bool isdtProvided(shading::AttributeKey key) const;

    bool isDisplacement()       const;
    bool isEntering()           const;
    bool isCausticPath()        const;
    bool isSubsurfaceAllowed()  const;
    bool isIndirect()           const;
    bool isHifi()               const;

    float getMediumIor() const;

    scene_rdl2::math::Vec3f adaptNormal(const scene_rdl2::math::Vec3f &Ns) const;
    // adaptToonNormal differs by checking if the integrator has
    // indicated that no light culling should be performed. If so,
    // the normal is not adapted. This is useful for certain NPR effects
    // that use an arbitrary normal.
    scene_rdl2::math::Vec3f adaptToonNormal(const scene_rdl2::math::Vec3f &Ns) const;

    template <typename T> const T &getAttribute(shading::TypedAttributeKey<T> key) const;
    template <typename T> T getdAttributedx(shading::TypedAttributeKey<T> key)     const;
    template <typename T> T getdAttributedy(shading::TypedAttributeKey<T> key)     const;
    template <typename T> T getdAttributeds(shading::TypedAttributeKey<T> key)     const;
    template <typename T> T getdAttributedt(shading::TypedAttributeKey<T> key)     const;

    const Intersection* getIntersection() const { return mIntersection; }

private:
    const Intersection* mIntersection;
};

} // namespace shading
} // namespace moonray

