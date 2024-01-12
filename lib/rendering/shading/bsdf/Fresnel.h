// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Fresnel.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/ispc/bsdf/Fresnel.hh>
#include <moonray/rendering/shading/PbrValidity.h>
#include "fabric/VelvetAlbedo.h"

#include <moonray/rendering/shading/ispc/bsdf/Fresnel_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

#include <vector>

namespace scene_rdl2 {
namespace alloc { class Arena; }
}
namespace moonray {
namespace shading {

ISPC_UTIL_TYPEDEF_STRUCT(Fresnel, Fresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(SchlickFresnel, SchlickFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(OneMinusRoughSchlickFresnel, OneMinusRoughSchlickFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(DielectricFresnel, DielectricFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(ConductorFresnel, ConductorFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(OneMinusFresnel, OneMinusFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(OneMinusVelvetFresnel, OneMinusVelvetFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(OneMinusRoughFresnel, OneMinusRoughFresnelv);
ISPC_UTIL_TYPEDEF_STRUCT(MultipleTransmissionFresnel, MultipleTransmissionFresnelv);

typedef ispc::FresnelType FresnelType;
class Fresnel;

// Factory function to create an AOS Fresnel object from an SOA Fresnel object.
// The fresnelv parameter may be nullptr in which case nullptr is returned.
Fresnel *createFresnel(scene_rdl2::alloc::Arena* arena,
                       const Fresnelv *fresnelv,
                       int lane);

//----------------------------------------------------------------------------

///
/// @class Fresnel Fresnel.h <shading/Fresnel.h>
/// @brief Base class defining the Fresnel reflectance interface.
/// 
class Fresnel
{
public:
    /// Optional properties that a Fresnel may expose
    enum Property {
        PROPERTY_NONE         = 0,
        PROPERTY_COLOR        = 1 << 0,
        PROPERTY_FACTOR       = 1 << 1,
        PROPERTY_ROUGHNESS    = 1 << 2
    };

    explicit Fresnel(int pflags) :
        mPflags(pflags), mWeight(1.0f) {  }

    Fresnel(scene_rdl2::alloc::Arena *arena, const Fresnelv *fresnelv, int lane)
    {
        MNRY_ASSERT(fresnelv && fresnelv->mMask & (1 << lane));
        mPflags = fresnelv->mPflags;
        mWeight = fresnelv->mWeight[lane];
    }

    virtual ~Fresnel()  {  }

    /// Pass in the dot product between light direction and half-vector
    virtual scene_rdl2::math::Color eval(float cosThetaI) const = 0;

    bool hasProperty(Property property) const { return mPflags & property; }
    int getPropertyFlags() const { return mPflags; }
    virtual void getProperty(Property prop, float *dest) const = 0;

    /// Override this function if you need to compute PBR Validity based on the
    /// derived class' members
    virtual scene_rdl2::math::Color computePbrValidity() const { return scene_rdl2::math::sBlack; }

    float getWeight() const { return mWeight; }
    void  setWeight(float weight) { mWeight = weight; }

    // prints out a description of this object with the provided indentation
    // prepended.
    virtual void show(std::ostream& os, const std::string& indent) const = 0;

protected:

    int mPflags;
    float mWeight;

private:
    Fresnel &operator=(const Fresnel &other);
};


///
/// @class SchlickFresnel Fresnel.h <shading/Fresnel.h>
/// @brief Schlick approximation to fresnel reflectance.
/// Pass in neta, the ratio of the ior for incident / transmitted material.
///
class SchlickFresnel : public Fresnel
{
public:
    SchlickFresnel(scene_rdl2::math::Color spec, float factor, float neta = 1.0f) :
        Fresnel(PROPERTY_COLOR | PROPERTY_FACTOR),
        mSpec(spec),
        mFactor(factor),
        mNeta(neta)  {  }

    SchlickFresnel(scene_rdl2::alloc::Arena *arena, const SchlickFresnelv *fresnelv, int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane)
    {
        mSpec.r = fresnelv->mSpec.r[lane];
        mSpec.g = fresnelv->mSpec.g[lane];
        mSpec.b = fresnelv->mSpec.b[lane];
        mFactor = fresnelv->mFactor[lane];
        mNeta = fresnelv->mNeta[lane];
    }


    finline scene_rdl2::math::Color eval(float hDotWi) const override
    {
        if (mFactor == 0.0f) {
            return mSpec * mWeight;
        }

        // Deal with total internal reflection if neta is > 1.0f
        float tmp;
        if (mNeta > 1.0f) {
            const float rcpNeta = scene_rdl2::math::rcp(mNeta);
            const float cosSqrCriticalAngle = 1.0f - rcpNeta * rcpNeta;
            const float cosSqrTheta =  hDotWi * hDotWi;
            if (cosSqrTheta < cosSqrCriticalAngle) {
                // Total internal reflection!
                return scene_rdl2::math::sWhite;
            }
            // Remap the curve to peak at the critical angle
            tmp = (1.0f - hDotWi) * scene_rdl2::math::rcp(1.0f - scene_rdl2::math::sqrt(cosSqrCriticalAngle));
        } else {
            tmp = 1.0f - hDotWi;
        }

        float power = tmp * tmp;
        power = tmp * power * power;

        return (mSpec + mFactor * (scene_rdl2::math::sWhite - mSpec) * power) * mWeight;
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        switch (prop)
        {
        case PROPERTY_COLOR:
            *dest       = mSpec.r * mWeight;
            *(dest + 1) = mSpec.g * mWeight;
            *(dest + 2) = mSpec.b * mWeight;
            break;
        case PROPERTY_FACTOR:
            *dest = mFactor * mWeight;
            break;
        default:
            MNRY_ASSERT(0 && "unexpected property");
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{SchlickFresnel}" << " " << this << " weight: " << mWeight << "\n";
        os << indent << "    " << "reflectivity at 0: "
            << mSpec.r << " "
            << mSpec.g << " "
            << mSpec.b << "\n";
    }

private:
    friend class OneMinusRoughSchlickFresnel;

    SCHLICK_FRESNEL_MEMBERS;
};

///
/// @class OneMinusFresnel Fresnel.h <shading/Fresnel.h>
/// @brief Uses 1 - some other fresnel term.
///
class OneMinusFresnel : public Fresnel
{
public:
    OneMinusFresnel(const Fresnel *other) :
        Fresnel(other->getPropertyFlags()), mTopFresnel(other)  {  }

    OneMinusFresnel(scene_rdl2::alloc::Arena *arena, const OneMinusFresnelv *fresnelv, int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane),
        mTopFresnel(nullptr)
    {
        if (fresnelv->mTopFresnel) {
            mTopFresnel = createFresnel(arena, (const Fresnelv *)fresnelv->mTopFresnel, lane);
        }
    }

    finline scene_rdl2::math::Color eval(float hDotWi) const override
    {
        return scene_rdl2::math::Color(1.0f, 1.0f, 1.0f) - mTopFresnel->eval(hDotWi);
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        mTopFresnel->getProperty(prop, dest);
        // if the property is color, we'll 1 - dest, other
        // properties will remain as is.
        if (prop == PROPERTY_COLOR) {
            *dest       = 1.f - *dest;
            *(dest + 1) = 1.f - *(dest + 1);
            *(dest + 2) = 1.f - *(dest + 2);
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{OneMinusFresnel}" << " " << this << " weight: " << mWeight << "\n";
        mTopFresnel->show(os, indent + "    ");
    }

private:
    ONE_MINUS_FRESNEL_MEMBERS;
};

///
/// @class OneMinusRoughFresnel Fresnel.h <shading/Fresnel.h>
/// @brief Approximation used to attenuate lobes under a more-or-less rough
///        specular lobe and enforce energy conservation.
///
class OneMinusRoughFresnel : public Fresnel
{
public:
    OneMinusRoughFresnel(const Fresnel *fresnel,
                         float specRoughness) :
        Fresnel(fresnel->getPropertyFlags() | PROPERTY_ROUGHNESS),
        mTopFresnel(fresnel),
        // Base Fresnel Behavior at Normal Incidence
        mOneMinusFresnelAt0(scene_rdl2::math::sWhite - fresnel->eval(1.0f)),
        // Apply roughness squaring to linearize roughness response
        // like all roughness-based lobes do.
        mSpecRoughness(specRoughness * specRoughness)
    {
        // Interpolating non-linearly based on roughness, using:
        // t = 1 - (1 - roughness)^3
        const float tmp = 1.0f - mSpecRoughness;
        mInterpolator = scene_rdl2::math::clamp(1.0f - (tmp * tmp * tmp), 0.0f, 1.0f);
    }

    OneMinusRoughFresnel(scene_rdl2::alloc::Arena *arena, const OneMinusRoughFresnelv *fresnelv, int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane),
        mTopFresnel(nullptr)
    {
        if (fresnelv->mTopFresnel) {
            mTopFresnel = createFresnel(arena, (const Fresnelv *)fresnelv->mTopFresnel, lane);
        }

        mOneMinusFresnelAt0.r = fresnelv->mOneMinusFresnelAt0.r[lane];
        mOneMinusFresnelAt0.g = fresnelv->mOneMinusFresnelAt0.g[lane];
        mOneMinusFresnelAt0.b = fresnelv->mOneMinusFresnelAt0.b[lane];
        mSpecRoughness = fresnelv->mSpecRoughness[lane];
        mInterpolator = fresnelv->mInterpolator[lane];
    }

    /// This approximation works well when passing nDotWo instead of hDotWi
    finline scene_rdl2::math::Color eval(float nDotWo) const override
    {
        return scene_rdl2::math::lerp(scene_rdl2::math::sWhite - mTopFresnel->eval(nDotWo),
                          mOneMinusFresnelAt0,
                          mInterpolator);
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        switch (prop)
        {
        case PROPERTY_ROUGHNESS: {
                // this is a VEC2F property for compatibility with bsdf lobes
                const float inputRoughness = scene_rdl2::math::sqrt(mSpecRoughness);
                *dest       = inputRoughness;
                *(dest + 1) = inputRoughness;
                break;
            }
        default:
            mTopFresnel->getProperty(prop, dest);
            // if the property is color, we'll 1 - dest, other
            // properties will remain as is.
            if (prop == PROPERTY_COLOR) {
                *dest       = 1.f - *dest;
                *(dest + 1) = 1.f - *(dest + 1);
                *(dest + 2) = 1.f - *(dest + 2);
            }
            break;
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{OneMinusRoughFresnel}" << " " << this << " weight: " << mWeight << "\n";
        os << indent << "    " << "1 - Fresnel at 0: "
            << mOneMinusFresnelAt0.r << " "
            << mOneMinusFresnelAt0.g << " "
            << mOneMinusFresnelAt0.b << "\n";
        os << indent << "    " << "roughness^2: " << mSpecRoughness << "\n";
        os << indent << "    " << "interpolator: " << mInterpolator << "\n";
        mTopFresnel->show(os, indent + "    ");
    }

private:
    ONE_MINUS_ROUGH_FRESNEL_MEMBERS;
};


///
/// @class OneMinusRoughSchlickFresnel Fresnel.h <shading/Fresnel.h>
/// @brief Approximation used to attenuate lobes under a more-or-less rough
///        specular lobe and enforce energy conservation.
///
class OneMinusRoughSchlickFresnel : public Fresnel
{
public:
    OneMinusRoughSchlickFresnel(const SchlickFresnel *schlick,
                                float specRoughness) :
        Fresnel(schlick->getPropertyFlags() | PROPERTY_ROUGHNESS),
        mTopFresnel(schlick),
        // Apply roughness squaring to linearize roughness response
        // like all roughness-based lobes do.
        mSpecRoughness(specRoughness * specRoughness)    {  }

    OneMinusRoughSchlickFresnel(scene_rdl2::alloc::Arena *arena, const OneMinusRoughSchlickFresnelv *fresnelv, int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane),
        mTopFresnel(nullptr)
    {
        if (fresnelv->mTopFresnel) {
            mTopFresnel = (const SchlickFresnel *)createFresnel(arena, (const Fresnelv *)fresnelv->mTopFresnel, lane);
        }

        mSpecRoughness = fresnelv->mSpecRoughness[lane];
    }

    /// This approximation works well when passing nDotWo instead of hDotWi
    finline scene_rdl2::math::Color eval(float nDotWo) const override
    {
        // Interpolating non-linearly based on roughness, using:
        // t = 1 - (1 - roughness)^3
        const float tmp = 1.0f - mSpecRoughness;
        const float t = 1.0f - (tmp * tmp * tmp);
        return scene_rdl2::math::lerp(scene_rdl2::math::Color(1.0f) - mTopFresnel->eval(nDotWo),
                          scene_rdl2::math::Color(1.0f) - mTopFresnel->mSpec,
                          scene_rdl2::math::clamp(t));
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        switch (prop)
        {
        case PROPERTY_ROUGHNESS: {
                // this is a VEC2F property for compatibility with bsdf lobes
                const float inputRoughness = scene_rdl2::math::sqrt(mSpecRoughness);
                *dest       = inputRoughness;
                *(dest + 1) = inputRoughness;
                break;
            }
        default:
            mTopFresnel->getProperty(prop, dest);
            // if the property is color, we'll 1 - dest, other
            // properties will remain as is.
            if (prop == PROPERTY_COLOR) {
                *dest       = 1.f - *dest;
                *(dest + 1) = 1.f - *(dest + 1);
                *(dest + 2) = 1.f - *(dest + 2);
            }
            break;
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{OneMinusRoughSchlickFresnel}" << " " << this << " weight: " << mWeight << "\n";
        mTopFresnel->show(os, indent + "    ");
    }

private:
    ONE_MINUS_ROUGH_SCHLICK_FRESNEL_MEMBERS;
};

///
/// @class OneMinusVelvetFresnel Fresnel.h <shading/Fresnel.h>
/// @brief Uses velvet albedo tables to distribute energy
/// Based on this paper:
/// "A Microfacet Based Coupled Specular-Matte BRDF Model"
/// https://pdfs.semanticscholar.org/658b/a4e43402545e5478ea5b8b2cdea3ebe59675.pdf
class OneMinusVelvetFresnel : public Fresnel
{
public:
    OneMinusVelvetFresnel(const float roughness) :
        Fresnel(PROPERTY_ROUGHNESS),
        mRoughness(roughness)
        {  }

    OneMinusVelvetFresnel(scene_rdl2::alloc::Arena *arena,
                          const OneMinusVelvetFresnelv *fresnelv,
                          int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane)
    {
        mRoughness = fresnelv->mRoughness[lane];
    }

    finline scene_rdl2::math::Color eval(float cosThetaWo) const override
    {
        float matteComponent = 1.0f - VelvetAlbedo::at(cosThetaWo, mRoughness) * mWeight;
        /* The paper includes the following terms for bidirectionality.
         * However, for the current multi-sample BSDF sampling approach, coupled with the
         * one-minus-fresnel architecture, a simple reduction in energy based on the
         * *average reflectance of the outgoing vector* is suitable for our needs, similar to the
         * other oneMinus lobes. */
        //matteComponent /= (1.0f - VelvetAlbedo::avg(mRoughness));
        //matteComponent *= (1.0f - VelvetAlbedo::at(cosThetaWi, mRoughness) * mAlbedo);
        return (scene_rdl2::math::sWhite * matteComponent);
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        switch (prop)
        {
        case PROPERTY_ROUGHNESS:
            *dest = *(dest + 1) = mRoughness;
            break;
        default:
            // SHOULD REALLY HANDLE non-PROPERTY_ROUGHNESS
            break;
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{OneMinusVelvetFresnel}" << " " << this << " weight: " << mWeight << "\n";
        os << indent << "    " << "roughness: " << mRoughness << "\n";
    }

private:
    ONE_MINUS_VELVET_FRESNEL_MEMBERS;
};


///
/// @class DielectricFresnel Fresnel.h <shading/Fresnel.h>
/// @brief A dielectric fresnel reflectance object.
///
class DielectricFresnel : public Fresnel
{
public:
    DielectricFresnel(float etaI, float etaT) :
        Fresnel(PROPERTY_COLOR), mEtaI(etaI), mEtaT(etaT)  {  }

    DielectricFresnel(scene_rdl2::alloc::Arena *arena, const DielectricFresnelv *fresnelv, int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane)
    {
        mEtaI = fresnelv->mEtaI[lane];
        mEtaT = fresnelv->mEtaT[lane];
    }

    finline scene_rdl2::math::Color eval(float hDotWi) const override
    {
        return eval(hDotWi, mEtaI, mEtaT, mWeight);
    }

    static scene_rdl2::math::Color eval(float hDotWi,
                            float etaI, float etaT,
                            float weight)
    {
        // Make sure we don't nan out down below
        hDotWi = scene_rdl2::math::max(hDotWi, -1.0f);
        hDotWi = scene_rdl2::math::min(hDotWi, 1.0f);

        // Compute Snell law
        const float eta = etaI * scene_rdl2::math::rcp(etaT);
        const float sinThetaTSqr = eta * eta * (1.0f - hDotWi * hDotWi);
        if (sinThetaTSqr >= 1.0f) {
            // Total internal reflection
            return scene_rdl2::math::sWhite * weight;

        } else {

            const float cosThetaT = scene_rdl2::math::sqrt(1.0f - sinThetaTSqr);
            const float cosThetaI = scene_rdl2::math::abs(hDotWi);

            const float parallel = evalParallel(etaI, etaT, cosThetaI, cosThetaT);
            const float perp = evalPerp(etaI, etaT, cosThetaI, cosThetaT);

            const float fr = 0.5f * (parallel + perp);
            return scene_rdl2::math::Color(fr, fr, fr) * weight;
        }
    }

    finline scene_rdl2::math::Color computePbrValidity() const override
    {
        scene_rdl2::math::Color res = scene_rdl2::math::sBlack;

        if (mEtaT < sPbrValidityDielectricValidLowBegin || mEtaT > sPbrValidityDielectricValidHighEnd) {
            res = sPbrValidityInvalidColor;
        } else if (mEtaT > sPbrValidityDielectricValidLowEnd && mEtaT < sPbrValidityDielectricValidHighBegin) {
            res = sPbrValidityValidColor;
        } else if (mEtaT > sPbrValidityDielectricValidHighBegin) {
            const float gradient = (mEtaT - sPbrValidityDielectricValidHighBegin) /
                    (sPbrValidityDielectricValidHighEnd - sPbrValidityDielectricValidHighBegin);
            res = scene_rdl2::math::lerp(sPbrValidityValidColor, sPbrValidityInvalidColor, gradient);
        } else { // mEtaT < sPbrValidityDielectricValidLowEnd
            const float gradient = (mEtaT - sPbrValidityDielectricValidLowBegin) /
                    (sPbrValidityDielectricValidLowEnd - sPbrValidityDielectricValidLowBegin);
            res = scene_rdl2::math::lerp(sPbrValidityInvalidColor, sPbrValidityValidColor, gradient);
        }
        return res;
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        switch (prop)
        {
        case PROPERTY_COLOR:
            *dest = *(dest + 1) = *(dest + 2) = mWeight;
            break;
        default:
            MNRY_ASSERT(0 && "unknown fresnel property");
        }
    }
    void show(std::ostream& os,
              const std::string& indent) const override
    {
        os << indent << "{DielectricFresnel}" << " " << this << " weight: " << mWeight << "\n";
        os << indent << "    " << "etaI: " << mEtaI << "\n";
        os << indent << "    " << "etaT: " << mEtaT << "\n";
    }
private:
    DIELECTRIC_FRESNEL_MEMBERS;

protected:
    // S Polarized
    static float evalPerp(float etaI,
                          float etaT,
                          float cosThetaI,
                          float cosThetaT)
    {
        const float etaIcosThetaI = etaI * cosThetaI;
        const float etaTcosThetaT = etaT  * cosThetaT;
        const float perp = ((etaIcosThetaI) - (etaTcosThetaT)) *
            scene_rdl2::math::rcp((etaIcosThetaI) + (etaTcosThetaT));
        return perp*perp;
    }

    // P Polarized
    static float evalParallel(float etaI,
                              float etaT,
                              float cosThetaI,
                              float cosThetaT)
    {
        const float etaTcosThetaI = etaT * cosThetaI;
        const float etaIcosThetaT = etaI * cosThetaT;
        const float parallel = ((etaTcosThetaI) - (etaIcosThetaT)) *
                scene_rdl2::math::rcp((etaTcosThetaI) + (etaIcosThetaT));
        return parallel*parallel;
    }

};

///
/// @class LayeredDielectricFresnel Fresnel.h <pbr/Fresnel.h>
/// @brief A layered dielectric fresnel reflectance object.
/// From "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
/// Eqn (6) and (7)
/// Models multiple layers of dielectric slabs with air outside both sides of each layer.
/// These properties increase reflectance compared to Fresnel reflectance from
/// a dielectric interface.
class LayeredDielectricFresnel : public DielectricFresnel
{
public:
    LayeredDielectricFresnel(float etaI, float etaT, float layerThickness) :
        DielectricFresnel(etaI, etaT)
    {
        // Layers vary in between [0.5, 1.5]
        // See Table 2
        mNumLayers = scene_rdl2::math::lerp(0.5f, 1.5f,
                                scene_rdl2::math::saturate(layerThickness));
    }

    finline Fresnel *copy() const   {  return new LayeredDielectricFresnel(*this);  }

    finline scene_rdl2::math::Color eval(float cosThetaI) const override
    {
        // Use the same refractive index for the
        // parallel and perpendicular components of
        // the dielectric fresnel equation by default
        return mWeight*evalFresnel(mEtaI, mEtaT,
                                   mEtaT, mEtaT,
                                   cosThetaI,
                                   mNumLayers);
    }

    // Layered Fresnel Evaluation with distinct IORs for
    // perpendicular and parallel components of the dielectric
    // fresnel equation.
    // See Eqn (6) & (7) in
    // "Physically-Accurate Fur Reflectance: Modeling, Measurement and Rendering"
    static scene_rdl2::math::Color evalFresnel(float etaI,
                                   float etaT,
                                   float etaPerp,
                                   float etaParallel,
                                   float cosThetaI,
                                   float numLayers)
    {
        // Make sure we don't nan out down below
        cosThetaI = scene_rdl2::math::max(cosThetaI, -1.0f);
        cosThetaI = scene_rdl2::math::min(cosThetaI, 1.0f);

        // Compute Snell law
        const float eta = 1.0f * scene_rdl2::math::rcp(etaT);
        const float sinThetaTSqr = eta * eta * (1.0f - cosThetaI*cosThetaI);

        if (sinThetaTSqr >= 1.0f) {
            // Total internal reflection
            return scene_rdl2::math::sWhite;
        }

        const float cosThetaT = scene_rdl2::math::sqrt(1.0f - sinThetaTSqr);
        cosThetaI = scene_rdl2::math::abs(cosThetaI);

        // Eqn (6)
        const float F = evalSingleLayer(etaI,
                                        etaPerp,
                                        etaParallel,
                                        cosThetaI,
                                        cosThetaT);

        // Eqn (7)
        const float n = numLayers * F;
        const float d = 1.0f + (numLayers - 1.0f) * F;

        const float layeredRefl = scene_rdl2::math::clamp(n/d, 0.0f, 1.0f);
        return scene_rdl2::math::Color(layeredRefl);
    }

    void show(std::ostream& os,
              const std::string& indent) const override
    {
        os << indent << "{LayeredDielectricFresnel}" << " "
                << this << " weight: " << mWeight << "\n";
        os << indent << "    " << "etaI: " << mEtaI << "\n";
        os << indent << "    " << "etaT: " << mEtaT << "\n";
        os << indent << "    " << "num layers: " << mNumLayers << "\n";
    }

private:
    DIELECTRIC_FRESNEL_MEMBERS;

    float mNumLayers;

    // Reflectance from a single layer
    // Eqn (6)
    static float evalSingleLayer(float etaI,
                                 float etaPerp,
                                 float etaParallel,
                                 float cosThetaI,
                                 float cosThetaT)
    {
        float Fs = evalPerp(etaI, etaPerp, cosThetaI, cosThetaT);
        float Fp = evalParallel(etaI, etaParallel, cosThetaI, cosThetaT);

        float t1 = Fs + (1.0f - Fs)*(1.0f - Fs) * Fs / (1.0f - Fs*Fs);
        float t2 = Fp + (1.0f - Fp)*(1.0f - Fp) * Fp / (1.0f - Fp*Fp);

        return (0.5f*t1 + 0.5f*t2);
    }
};

///
/// @class ConductorFresnel Fresnel.h <shading/Fresnel.h>
/// @brief A conductor fresnel reflectance object.
///
class ConductorFresnel : public Fresnel
{
public:

    // Initialize Conductor Fresnel object using real and imaginary components
    // of a complex IOR
    ConductorFresnel(const scene_rdl2::math::Color& eta, const scene_rdl2::math::Color& absorption) :
        Fresnel(PROPERTY_COLOR), mEta(eta),
        mAbsorption(absorption) {}

    ConductorFresnel(scene_rdl2::alloc::Arena *arena, const ConductorFresnelv *fresnelv, int lane)
      : Fresnel(arena, (const Fresnelv *)fresnelv, lane)
    {
        mEta.r = fresnelv->mEta.r[lane];
        mEta.g = fresnelv->mEta.g[lane];
        mEta.b = fresnelv->mEta.b[lane];
        mAbsorption.r = fresnelv->mAbsorption.r[lane];
        mAbsorption.g = fresnelv->mAbsorption.g[lane];
        mAbsorption.b = fresnelv->mAbsorption.b[lane];
    }

    finline scene_rdl2::math::Color eval(float hDotWi) const override
    {
        const float hDotWiSqrd = hDotWi * hDotWi;
        const scene_rdl2::math::Color factor = mEta * mEta + mAbsorption * mAbsorption;
        const scene_rdl2::math::Color tmp = factor * hDotWiSqrd;
        const scene_rdl2::math::Color etaHDotWi = mEta * hDotWi;

        const scene_rdl2::math::Color parallel = (tmp - (2.0f * etaHDotWi) + scene_rdl2::math::Color(1.0f)) /
                           (tmp + (2.0f * etaHDotWi) + scene_rdl2::math::Color(1.0f));

        const scene_rdl2::math::Color perp = (factor - (2.0f * etaHDotWi) + scene_rdl2::math::Color(hDotWiSqrd)) /
                       (factor + (2.0f * etaHDotWi) + scene_rdl2::math::Color(hDotWiSqrd));

        return 0.5f * (parallel + perp) * mWeight;
    }

    finline scene_rdl2::math::Color computePbrValidity() const override
    {
        scene_rdl2::math::Color res = scene_rdl2::math::sBlack;
        const scene_rdl2::math::Color reflectivity = eval(1.f);
        const float value = scene_rdl2::math::max(reflectivity.r, reflectivity.g, reflectivity.b);

        if (value < sPbrValidityConductorInvalid || value > 1.0f) {
            res = sPbrValidityInvalidColor;
        } else if (value > sPbrValidityConductorValid) {
            res = sPbrValidityValidColor;
        } else {
            const float gradient = (value - sPbrValidityConductorInvalid) /
                    (sPbrValidityConductorValid - sPbrValidityConductorInvalid);
            res = scene_rdl2::math::lerp(sPbrValidityInvalidColor, sPbrValidityValidColor, gradient);
        }
        return res;
    }

    finline void getProperty(Property prop, float *dest) const override
    {
        switch (prop)
        {
        case PROPERTY_COLOR:
            *dest       = (1.f - mAbsorption.r) * mWeight;
            *(dest + 1) = (1.f - mAbsorption.g) * mWeight;
            *(dest + 2) = (1.f - mAbsorption.b) * mWeight;
            break;
        default:
            MNRY_ASSERT(0 && "unknown fresnel property");
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{ConductorFresnel}" << " " << this << " weight: " << mWeight << "\n";
        os << indent << "    " << "Eta: " << mEta.r << " " << mEta.g << " " << mEta.b << "\n";
        os << indent << "    " << "Absorption: " << mAbsorption.r << " " << mAbsorption.g << " " << mAbsorption.b << "\n";
    }

private:
    CONDUCTOR_FRESNEL_MEMBERS;
};


///
/// @class MultipleTransmissionFresnel Fresnel.h <shading/Fresnel.h>
/// @brief Packages up to sixteen 'transmission' (OneMinus*) Fresnel objects.
/// Use the add() function to add these fresnel objects
///

class MultipleTransmissionFresnel : public Fresnel
{
public:
    MultipleTransmissionFresnel() :
        Fresnel(PROPERTY_NONE),
        mNumFresnels(0)
    {
    }

    MultipleTransmissionFresnel(scene_rdl2::alloc::Arena *arena,
                                const MultipleTransmissionFresnelv *fresnelv,
                                int lane)
      : Fresnel(arena,
                (const Fresnelv *)fresnelv,
                lane),
        mNumFresnels(0)
    {
        for (int i = 0; i < fresnelv->mNumFresnels; ++i) {
            Fresnel *fresnel = createFresnel(arena, MNRY_VERIFY(fresnelv->mFresnels[i]), lane);
            if (fresnel) {
                mFresnels[mNumFresnels++] = fresnel;
            }
        }
    }

    finline void add(const Fresnel* fresnel)
    {
        MNRY_ASSERT(mNumFresnels < 16);
        MNRY_ASSERT(fresnel);
        mFresnels[mNumFresnels++] = fresnel;
        mPflags |= fresnel->getPropertyFlags();
    }

    // compute transmission through all Fresnel objects.
    finline scene_rdl2::math::Color eval(float nDotWo) const override
    {
        scene_rdl2::math::Color result = scene_rdl2::math::sWhite;
        for (int i = 0; i < mNumFresnels; ++i) {
            result *= mFresnels[i]->eval(nDotWo);
        }
        return result;
    }

    // compute average property value of all Fresnel objects
    finline void getProperty(Property prop, float *dest) const override
    {
        MNRY_ASSERT(mNumFresnels > 0);
        switch (prop)
        {
        case PROPERTY_COLOR:
            computeColorProperty(dest);
            break;
        case PROPERTY_FACTOR:
            computeFactorProperty(dest);
            break;
        case PROPERTY_ROUGHNESS:
            computeRoughnessProperty(dest);
            break;
        default:
            // What should this case compute?
            break;
        }
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        os << indent << "{MultipleTransmissionFresnel}" << " " << this << " weight: " << mWeight << "\n";
        for (int i = 0; i < mNumFresnels; ++i) {
            mFresnels[i]->show(os, indent + "    ");
        }
    }

private:
    finline void computeColorProperty(float *dest) const
    {
        // compute average
        scene_rdl2::math::Color avg = scene_rdl2::math::sBlack;
        scene_rdl2::math::Color tmp;
        for (int i = 0; i < mNumFresnels; ++i) {
            if (mFresnels[i]->hasProperty(PROPERTY_COLOR)) {
                mFresnels[i]->getProperty(PROPERTY_COLOR, &tmp[0]);
                avg += tmp;
            }
        }
        avg *= scene_rdl2::math::rcp(static_cast<float>(mNumFresnels));
        *dest       = avg[0];
        *(dest + 1) = avg[1];
        *(dest + 2) = avg[2];
    }

    finline void computeFactorProperty(float *dest) const
    {
        // compute average
        float avg = 0.0f;
        float tmp;
        for (int i = 0; i < mNumFresnels; ++i) {
            if (mFresnels[i]->hasProperty(PROPERTY_FACTOR)) {
                mFresnels[i]->getProperty(PROPERTY_FACTOR, &tmp);
                avg += tmp;
            }
        }
        avg *= scene_rdl2::math::rcp(static_cast<float>(mNumFresnels));
        *dest = avg;
    }

    finline void computeRoughnessProperty(float *dest) const
    {
        // compute average
        scene_rdl2::math::Vec2f avg(0.0f, 0.0f);
        scene_rdl2::math::Vec2f tmp;
        for (int i = 0; i < mNumFresnels; ++i) {
            if (mFresnels[i]->hasProperty(PROPERTY_ROUGHNESS)) {
                mFresnels[i]->getProperty(PROPERTY_ROUGHNESS, &tmp[0]);
                avg += tmp;
            }
        }
        avg *= scene_rdl2::math::rcp(static_cast<float>(mNumFresnels));
        *dest       = avg[0];
        *(dest + 1) = avg[1];
    }

    MULTIPLE_TRANSMISSION_FRESNEL_MEMBERS;
};

// Utility function for diffuse reflectance fresnel used in the dipole model
finline float
diffuseFresnelReflectance(float eta)
{
    if (eta >= 1.0f) {
        return -1.4399f / (eta * eta) + 0.7099f / eta + 0.6681f + 0.0636f * eta;
    } else {
        float etaSqr = eta * eta;
        return -0.4399f + 0.7099f / eta - 0.3319f / etaSqr + 0.0636f /
                (etaSqr * eta);
    }
}

// Average Dielectric Fresnel Reflectance
// Ref:"Revisiting Physically Based Shading", Kulla'17
finline float
averageFresnelReflectance(float neta)
{
    MNRY_ASSERT(neta >= 1.0f);
    float favg =
            (neta - 1.0f) / (4.08567f + 1.00071f*neta);
    return favg;
}

// Average Dielectric Fresnel Reflectance
// Ref:"Revisiting Physically Based Shading", Kulla'17
finline float
averageInvFresnelReflectance(float neta)
{
    MNRY_ASSERT(neta < 1.0f);
    float favgInv = 0.997118f +
              0.1014f * neta -
              0.965241f * neta * neta -
              0.130607f * neta * neta * neta;
    return favgInv;
}

// Average Conductor Fresnel Reflectance
// Ref:"Revisiting Physically Based Shading", Kulla'17
finline scene_rdl2::math::Color
averageFresnelReflectance(const scene_rdl2::math::Color& reflectivity,
                          const scene_rdl2::math::Color& edgeTint)
{
    const scene_rdl2::math::Color r2 = reflectivity * reflectivity;
    const scene_rdl2::math::Color e2 = edgeTint*edgeTint;
    scene_rdl2::math::Color favg = scene_rdl2::math::Color(0.087237f) + 0.0230685f * edgeTint -
            0.0864902f * e2 +
            0.0774594f * e2 * edgeTint +
            0.782654f * reflectivity -
            0.136432f * r2 +
            0.278708f * r2 * reflectivity +
            0.19744f * edgeTint * reflectivity +
            0.0360605f * e2 * reflectivity -
            0.2586f * edgeTint * r2;
    return favg;
}

// Average Dielectric Fresnel Reflectance
// Ref:"Revisiting Physically Based Shading", Kulla'17
finline void
averageFresnelReflectance(float neta,
                          float& favg,
                          float& favgInv)
{
    if (neta > 1.0f) {
        favg = averageFresnelReflectance(neta);
        favgInv = averageInvFresnelReflectance(1.0f/neta);
    } else if (neta < 1.0f) {
        favg = averageFresnelReflectance(1.0f/neta);
        favgInv = averageInvFresnelReflectance(neta);
    } else {
        favg = favgInv = 0.0f;
    }
}


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

