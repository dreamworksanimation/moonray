// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bsdf/Bsdf.h"
#include "bsdf/Fresnel.h"
#include "bsdf/under/BsdfUnderClearcoat.h"
#include "bsdf/under/BsdfUnderClearcoatTransmission.h"
#include "bssrdf/Bssrdf.h"
#include "bssrdf/VolumeSubsurface.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/platform/Platform.h> // finline
#include <scene_rdl2/render/util/Arena.h>

namespace moonray {
namespace shading {

// Abstract base class. Objects of this type serve to
// handle the attenuation of a BsdfLobe or Bssrdf or VolumeSubsurface
class LobeAttenuator
{
public:
    LobeAttenuator():
        mN(0.0f, 0.0f, 1.0f),
        mOneMinusFresnel(nullptr)
    {}

    LobeAttenuator(const scene_rdl2::math::Vec3f& N):
        mN(N),
        mOneMinusFresnel(nullptr)
    {}

    // Make this an abstract class not be instantiated
    virtual ~LobeAttenuator() =0;

    // Default Attenuation Operators
    // Can be Overridden by Child Attenuators
    virtual BsdfLobe*
    operator() (scene_rdl2::alloc::Arena * arena,
                BsdfLobe* lobe) const
    {
        if (mOneMinusFresnel == nullptr)
            return lobe;

        // Default BSDF Attenuator
        lobe = arena->allocWithArgs<UnderBsdfLobe>(lobe, mN);
        lobe->setFresnel(mOneMinusFresnel);
        return lobe;
    }

    virtual void
    operator() (scene_rdl2::alloc::Arena * arena,
                Bssrdf* bssrdf) const
    {
        if (mOneMinusFresnel == nullptr)
            return;

        // Default BSSRDF Attenuator
        auto fresnel =
            static_cast<MultipleTransmissionFresnel*>(bssrdf->getTransmissionFresnel());
        // Assumes all BSSRDF's have a MultipleTransmissionFresnel attached.
        MNRY_ASSERT(fresnel);
        fresnel->add(mOneMinusFresnel);
    }

    virtual void
    operator() (scene_rdl2::alloc::Arena * arena,
                VolumeSubsurface* vs) const
    {
        if (mOneMinusFresnel == nullptr)
            return;

        // Default VolumeSubsurface Attenuator
        auto fresnel =
            static_cast<MultipleTransmissionFresnel*>(vs->getTransmissionFresnel());
        // Assumes all VS's have a MultipleTransmissionFresnel attached.
        MNRY_ASSERT(fresnel);
        fresnel->add(mOneMinusFresnel);
    }

    LobeAttenuator(const LobeAttenuator& other) =delete;
    LobeAttenuator& operator=(const LobeAttenuator& other) =delete;

protected:
    scene_rdl2::math::Vec3f mN;
    Fresnel*    mOneMinusFresnel;
};

// Compulsary definition of a pure virtual destructor to abstract the base class
LobeAttenuator::~LobeAttenuator() {}

class SimpleAttenuator : public LobeAttenuator
{
public:
    SimpleAttenuator(scene_rdl2::alloc::Arena *arena,
                     const scene_rdl2::math::Vec3f& N,
                     float roughness,
                     Fresnel * fresnel) :
        LobeAttenuator(N)
    {
        if (scene_rdl2::math::isZero(roughness)) {
            mOneMinusFresnel = arena->allocWithArgs<OneMinusFresnel>(fresnel);
        } else {
            mOneMinusFresnel = arena->allocWithArgs<OneMinusRoughFresnel>(fresnel, roughness);
        }
    }

    SimpleAttenuator(const SimpleAttenuator& other) =delete;
    SimpleAttenuator& operator=(const SimpleAttenuator& other) =delete;

    ~SimpleAttenuator() override {}

    // Use Standard Attenuation Operations
    using LobeAttenuator::operator();
};

class HairAttenuator : public LobeAttenuator
{
public:
    HairAttenuator(scene_rdl2::alloc::Arena *arena,
                   Fresnel * fresnel) :
        LobeAttenuator()
    {
        if (fresnel != nullptr) {
            mOneMinusFresnel = arena->allocWithArgs<OneMinusFresnel>(fresnel);
        }
    }

    HairAttenuator(const HairAttenuator& other) =delete;
    HairAttenuator& operator=(const HairAttenuator& other) =delete;

    ~HairAttenuator() override {}

    // Use Standard Attenuation Operations
    using LobeAttenuator::operator();

    // Override BSDF Attenuation
    // Does not use UnderBsdfLobe
    BsdfLobe*
    operator() (scene_rdl2::alloc::Arena * arena, BsdfLobe* lobe) const override
    {
        lobe->setFresnel(mOneMinusFresnel);
        return lobe;
    }
};

class VelvetAttenuator : public LobeAttenuator
{
public:
    VelvetAttenuator(scene_rdl2::alloc::Arena *arena,
                     const VelvetBRDF& velvet,
                     float weight) :
        LobeAttenuator(velvet.getN())
    {
        if (velvet.getUseAbsorbingFibers() == false) {
            // Lets more energy through based on the velvet color
            const scene_rdl2::math::Color& c = velvet.getColor();
            weight *= scene_rdl2::math::max(c.r,
                      scene_rdl2::math::max(c.g,
                                c.b));
        }
        mOneMinusFresnel =
            arena->allocWithArgs<OneMinusVelvetFresnel>(velvet.getRoughness());

        mOneMinusFresnel->setWeight(weight);
    }

    VelvetAttenuator(const VelvetAttenuator& other) =delete;
    VelvetAttenuator& operator=(const VelvetAttenuator& other) =delete;

    ~VelvetAttenuator() override {}

    // Use Standard Attenuation Operations
    using LobeAttenuator::operator();
};

class ClearcoatAttenuator : public LobeAttenuator
{
public:
    ClearcoatAttenuator(
            scene_rdl2::alloc::Arena *arena,
            const scene_rdl2::math::Vec3f& N,
            float roughness,
            float etaI,
            float etaT,
            bool refracts,
            float thickness,
            const scene_rdl2::math::Color& attenuationColor,
            float weight,
            Fresnel * fresnel) :
        LobeAttenuator(N),
        mEtaI(etaI),
        mEtaT(etaT),
        mRefracts(refracts),
        mThickness(thickness),
        mAttenuationColor(attenuationColor),
        mAttenuationWeight(weight)
    {
        if (scene_rdl2::math::isZero(roughness)) {
            mOneMinusFresnel = arena->allocWithArgs<OneMinusFresnel>(fresnel);
        } else {
            mOneMinusFresnel =
                arena->allocWithArgs<OneMinusRoughFresnel>(fresnel, roughness);
        }
    }

    ClearcoatAttenuator(const ClearcoatAttenuator& other) =delete;
    ClearcoatAttenuator& operator=(const ClearcoatAttenuator& other) =delete;

    ~ClearcoatAttenuator() override {}

    // Use Standard Attenuation Operations
    using LobeAttenuator::operator();

    // Overrride BSDF Attenuation
    BsdfLobe*
    operator() (scene_rdl2::alloc::Arena * arena, BsdfLobe* lobe) const override
    {
        if (mRefracts && !scene_rdl2::math::isEqual(mEtaI, mEtaT)) {
            const bool isReflectionLobe = lobe->matchesFlag(BsdfLobe::REFLECTION);
            if (isReflectionLobe) {
                lobe = arena->allocWithArgs<UnderClearcoatBsdfLobe>(
                        lobe, arena, mN,
                        mEtaI, mEtaT,
                        mThickness, mAttenuationColor, mAttenuationWeight);
            } else {
                // we are wrapping a transmission lobe
                lobe = arena->allocWithArgs<UnderClearcoatTransmissionBsdfLobe>(
                        lobe, arena, mN,
                        mEtaI, mEtaT,
                        mThickness, mAttenuationColor, mAttenuationWeight);
            }
        } else {
            // no refraction
            lobe = arena->allocWithArgs<UnderBsdfLobe>(lobe,
                                                       mN,
                                                       mThickness,
                                                       mAttenuationColor,
                                                       mAttenuationWeight);
        }

        lobe->setFresnel(mOneMinusFresnel);

        return lobe;
    }

    // We don't currently support doing proper clearcoat refraction/absorption
    // with Subsurface - but for now we at least use standard attenuation

private:
    float                       mEtaI;
    float                       mEtaT;
    bool                        mRefracts;
    float                       mThickness;
    scene_rdl2::math::Color     mAttenuationColor;
    float                       mAttenuationWeight;
};

} // end namespace shading
} // end namespace moonray

