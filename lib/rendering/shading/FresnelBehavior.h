// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/platform/Platform.h> // finline

#include <typeinfo>

// ===========================================================================
// MATERIAL PUBLIC SHADING API
// ===========================================================================

namespace moonray {
namespace shading {

// Abstract base class. Objects of this type serve to describe the Fresnel
// behavior of a lobe, and to hold the construction arguments for the
// underlying Fresnel implementation
class FresnelBehavior
{
public:
    // Describes the class of Fresnel behavior.  We can apply different
    // behaviors when combining lobes based on knowing if a FresnelBehavior
    // is dielectric or conductor.  Dielectrics transmit light, whereas
    // conductors do not.
    enum class Class
    {
        Dielectric,
        Conductor,
    };

    FresnelBehavior() =default;

    virtual ~FresnelBehavior() =0;

    virtual Class getClass() const =0;

    template <typename T>
    bool isType() const { return (typeid(T) == typeid(*this)); }

    FresnelBehavior(const FresnelBehavior& other) =delete;
    FresnelBehavior& operator=(const FresnelBehavior& other) =delete;
};

// Represents physical dielectric Fresnel behavior
class Dielectric : public FresnelBehavior
{
public:
    Dielectric(float etaI, float etaT) :
        mEtaI(etaI),
        mEtaT(etaT) {}

    ~Dielectric() override {}

    Class getClass() const override { return Class::Dielectric; }

    finline float getEtaI() const { return mEtaI; }
    finline float getEtaT() const { return mEtaT; }

    void setEtaI(float etaI) { mEtaI = etaI; }
    void setEtaT(float etaT) { mEtaI = etaT; }

private:
    float mEtaI;
    float mEtaT;
};

// Represents physical conductor Fresnel behavior. Uses the "Artist
// Friendly Metallic Fresnel" parameterization proposed by Ole Gulbrandsen
// in Journal of Computer Graphics Techniques Vol. 3, No. 4, 2014
// http://jcgt.org
class Conductor : public FresnelBehavior
{
public:
    Conductor(const scene_rdl2::math::Color& reflectivity,
              const scene_rdl2::math::Color& edgeTint) :
        mReflectivity(reflectivity),
        mEdgeTint(edgeTint) {}

    ~Conductor() override {}

    Class getClass() const override { return Class::Conductor; }

    finline const scene_rdl2::math::Color& getReflectivity() const { return mReflectivity; }
    finline const scene_rdl2::math::Color& getEdgeTint()     const { return mEdgeTint; }

private:
    math::Color mReflectivity;
    math::Color mEdgeTint;
};

} // end namespace shading
} // end namespace moonray

