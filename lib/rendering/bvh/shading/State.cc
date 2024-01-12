// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file State.h
/// $Id$
///

#include "State.h"
#include <moonray/rendering/bvh/shading/Intersection.h>

using namespace scene_rdl2;

namespace moonray {
namespace shading {

const math::Vec3f& State::getP()                 const { return mIntersection->getP();              }
const math::Vec3f& State::getNg()                const { return mIntersection->getNg();             }
const math::Vec3f& State::getN()                 const { return mIntersection->getN();              }
const math::Vec3f& State::getdPds()              const { return mIntersection->getdPds();           }
const math::Vec3f& State::getdPdt()              const { return mIntersection->getdPdt();           }
const math::Vec3f& State::getdNds()              const { return mIntersection->getdNds();           }
const math::Vec3f& State::getdNdt()              const { return mIntersection->getdNdt();           }
const math::Vec3f& State::getWo()                const { return mIntersection->getWo();             }
const math::Vec2f& State::getSt()                const { return mIntersection->getSt();             }
const math::Vec2f& State::getMinRoughness()      const { return mIntersection->getMinRoughness();   }
const rdl2::Geometry* State::getGeometryObject() const { return mIntersection->getGeometryObject(); }

float State::getdSdx() const { return mIntersection->getdSdx(); }
float State::getdSdy() const { return mIntersection->getdSdy(); }
float State::getdTdx() const { return mIntersection->getdTdx(); }
float State::getdTdy() const { return mIntersection->getdTdy(); }

bool State::isProvided(shading::AttributeKey key)   const { return mIntersection->isProvided(key);   }
bool State::isdsProvided(shading::AttributeKey key) const { return mIntersection->isdsProvided(key); }
bool State::isdtProvided(shading::AttributeKey key) const { return mIntersection->isdtProvided(key); }

bool State::isDisplacement()      const { return mIntersection->isDisplacement();      }
bool State::isEntering()          const { return mIntersection->isEntering();          }
bool State::isCausticPath()       const { return mIntersection->isCausticPath();       }
bool State::isSubsurfaceAllowed() const { return mIntersection->isSubsurfaceAllowed(); }
bool State::isIndirect()          const { return mIntersection->isIndirect();          }
bool State::isHifi()              const { return mIntersection->isHifi();              }

float State::getMediumIor() const { return mIntersection->getMediumIor(); }

math::Vec3f State::adaptNormal(const math::Vec3f &Ns) const { return mIntersection->adaptNormal(Ns); }
math::Vec3f State::adaptToonNormal(const math::Vec3f &Ns) const { return mIntersection->adaptToonNormal(Ns); }

// Macro for compact template specialization definitions
#define INSTANTIATE_GET_ATTRIBUTE(T) \
template <>\
const T &State::getAttribute(shading::TypedAttributeKey<T> key) const\
{\
    return mIntersection->getAttribute(key);\
}

INSTANTIATE_GET_ATTRIBUTE(rdl2::Bool);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Int);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Long);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Float);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Double);
INSTANTIATE_GET_ATTRIBUTE(rdl2::String);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Rgb);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Rgba);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec2f);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec2d);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec3f);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec3d);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec4f);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec4d);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Mat4f);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Mat4d);
INSTANTIATE_GET_ATTRIBUTE(rdl2::BoolVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::IntVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::LongVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::FloatVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::DoubleVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::StringVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::RgbVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::RgbaVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec2fVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec2dVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec3fVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec3dVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec4fVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Vec4dVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Mat4fVector);
INSTANTIATE_GET_ATTRIBUTE(rdl2::Mat4dVector);

#define INSTANTIATE_GET_D_ATTRIBUTE(T) \
template <>\
T State::getdAttributedx(shading::TypedAttributeKey<T> key) const\
{\
    return mIntersection->getdAttributedx(key);\
}\
\
template <>\
T State::getdAttributedy(shading::TypedAttributeKey<T> key) const\
{\
    return mIntersection->getdAttributedy(key);\
}\
\
template <>\
T State::getdAttributeds(shading::TypedAttributeKey<T> key) const\
{\
    return mIntersection->getdAttributeds(key);\
}\
\
template <>\
T State::getdAttributedt(shading::TypedAttributeKey<T> key) const\
{\
    return mIntersection->getdAttributedt(key);\
}

// Derivatives are only supported for the types below
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Int);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Long);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Float);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Double);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Rgb);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Rgba);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Vec2f);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Vec3f);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Vec4f);
INSTANTIATE_GET_D_ATTRIBUTE(rdl2::Mat4f);

bool
State::getRefP(math::Vec3f &refP) const
{
    bool retVal = false;
    if (isProvided(shading::StandardAttributes::sRefP)) {
        refP = getAttribute(shading::StandardAttributes::sRefP);
        retVal = true;
    } else {
        refP = math::zero;
    }
    return retVal;
}

bool
State::getRefN(math::Vec3f &refN) const
{
    refN = math::zero;

    if (isProvided(shading::StandardAttributes::sRefN)) {
        refN = getAttribute(shading::StandardAttributes::sRefN);
    } else if (isProvided(shading::StandardAttributes::sRefP)) {
        // Compute refN from refP partials
        math::Vec3f dPds, dPdt;
        getdVec3fAttrds(shading::StandardAttributes::sRefP, dPds);
        getdVec3fAttrdt(shading::StandardAttributes::sRefP, dPdt);
        if (math::isZero(math::lengthSqr(dPds))) {
            // Invalid zero vector. Avoid divide by zero and NaNs
            // that would result from math::normalize().
            return false;
        }
        dPds = math::normalize(dPds);
        // For curves, since dPdt is zero, we use the camera direction
        // Wo and make it orthogonal to dPds
        if (math::isZero(math::lengthSqr(dPdt))) {
            dPdt = getWo();
            dPdt = math::cross(dPds, math::cross(dPds, math::normalize(dPdt)));

        }
        refN = math::cross(dPds, math::normalize(dPdt));
        if (math::isZero(math::lengthSqr(refN))) {
            // dPds and dPdt are identical so their cross product is a zero vector.
            return false;
        }
    } else {
        return false;
    }

    if (isEntering() == false) {
        // Flip reference space normals on exiting a surface
        refN = -refN;
    }

    return true;
}

math::Vec3f
State::getdPdx() const
{
    const math::Vec3f dpds = getdPds();
    const math::Vec3f dpdt = getdPdt();
    const float dsdx = getdSdx();
    const float dtdx = getdTdx();

    const math::Vec3f dpdx = dpds * dsdx + dpdt * dtdx;

    return dpdx;
}

math::Vec3f
State::getdPdy() const
{
    const math::Vec3f dpds = getdPds();
    const math::Vec3f dpdt = getdPdt();
    const float dsdy = getdSdy();
    const float dtdy = getdTdy();

    const math::Vec3f dpdy = dpds * dsdy + dpdt * dtdy;

    return dpdy;
}

bool
State::getdVec3fAttrds(int key,
                       math::Vec3f& dVec3fAttrds) const
{
    if (isdsProvided(key)) {
        dVec3fAttrds = getdAttributeds(static_cast<shading::TypedAttributeKey<math::Vec3f>>(key));
        return true;
    } else {
        dVec3fAttrds = math::zero;
        return false;
    }
}

bool
State::getdVec3fAttrdt(int key,
                       math::Vec3f& dVec3fAttrdt) const
{
    if (isdtProvided(key)) {
        dVec3fAttrdt = getdAttributedt(static_cast<shading::TypedAttributeKey<math::Vec3f>>(key));
        return true;
    } else {
        dVec3fAttrdt = math::zero;
        return false;
    }
}

bool
State::getdVec3fAttrdx(int key,
                       math::Vec3f& dVec3fAttrdx) const
{
    if (isdsProvided(key) && isdtProvided(key)) {
        dVec3fAttrdx = getdAttributedx(static_cast<shading::TypedAttributeKey<math::Vec3f>>(key));
        return true;
    } else {
        dVec3fAttrdx = math::zero;
        return false;
    }
}

bool
State::getdVec3fAttrdy(int key,
                       math::Vec3f& dVec3fAttrdy) const
{
    if (isdsProvided(key) && isdtProvided(key)) {
        dVec3fAttrdy = getdAttributedy(static_cast<shading::TypedAttributeKey<math::Vec3f>>(key));
        return true;
    } else {
        dVec3fAttrdy = math::zero;
        return false;
    }
}

} // namespace shading
} // namespace moonray

