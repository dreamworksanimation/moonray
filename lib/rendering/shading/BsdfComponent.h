// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/platform/Platform.h> // finline
#include <scene_rdl2/scene/rdl2/TraceSet.h>

#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/RampControl_ispc_stubs.h>

// ===========================================================================
// MATERIAL PUBLIC SHADING API
// ===========================================================================

namespace moonray {

namespace shading {

class Iridescence;

// Abstract base class. Objects of this type serve to describe a component
// of a Bsdf.  This description includes constructor arguments for one or
// more particular lobe types, as well as any associated Fresnel objects,
// and any associated BsdfComponentModifiers (see below).
class BsdfComponent
{
public:
    explicit BsdfComponent(const Iridescence* const iridescence = nullptr) :
        mIridescence(iridescence)
    {}

    virtual ~BsdfComponent() =0;

    finline const Iridescence* getIridescence() const { return mIridescence; }

    BsdfComponent(const BsdfComponent& other) =delete;
    BsdfComponent& operator=(const BsdfComponent& other) =delete;

private:
    const Iridescence* const mIridescence;
};

// #####################################################################
// BsdfComponents
// #####################################################################

class MicrofacetAnisotropicClearcoat : public BsdfComponent
{
public:
    MicrofacetAnisotropicClearcoat(
            const scene_rdl2::math::Vec3f& N,
            const float eta,
            float roughnessU,
            float roughnessV,
            const scene_rdl2::math::Vec3f& shadingTangent,
            const float thickness,
            const scene_rdl2::math::Color& attenuationColor,
            const bool refracts,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mRoughnessU(roughnessU),
        mRoughnessV(roughnessV),
        mShadingTangent(shadingTangent),
        mThickness(thickness),
        mAttenuationColor(attenuationColor),
        mRefracts(refracts),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric)
    {}

    ~MicrofacetAnisotropicClearcoat() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline float                               getEta()                    const { return mEta; }
    finline float                               getRoughnessU()             const { return mRoughnessU; }
    finline float                               getRoughnessV()             const { return mRoughnessV; }
    finline const scene_rdl2::math::Vec3f&      getShadingTangent()         const { return mShadingTangent; }
    finline float                               getThickness()              const { return mThickness; }
    finline const scene_rdl2::math::Color&      getAttenuationColor()       const { return mAttenuationColor; }
    finline bool                                getRefracts()               const { return mRefracts; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }

private:
    scene_rdl2::math::Vec3f mN;
    float mEta;
    float mRoughnessU;
    float mRoughnessV;
    scene_rdl2::math::Vec3f mShadingTangent;
    float mThickness;
    scene_rdl2::math::Color mAttenuationColor;
    bool mRefracts;
    ispc::MicrofacetDistribution mMicrofacetDistribution;
    ispc::MicrofacetGeometric mMicrofacetGeometric;
};

class MicrofacetIsotropicClearcoat : public BsdfComponent
{
public:
    MicrofacetIsotropicClearcoat(
            const scene_rdl2::math::Vec3f& N,
            const float eta,
            float roughness,
            const float thickness,
            const scene_rdl2::math::Color& attenuationColor,
            const bool refracts,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mRoughness(roughness),
        mThickness(thickness),
        mAttenuationColor(attenuationColor),
        mRefracts(refracts),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric)
    {}

    ~MicrofacetIsotropicClearcoat() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline float                               getEta()                    const { return mEta; }
    finline float                               getRoughness()              const { return mRoughness; }
    finline float                               getThickness()              const { return mThickness; }
    finline const scene_rdl2::math::Color&      getAttenuationColor()       const { return mAttenuationColor; }
    finline bool                                getRefracts()               const { return mRefracts; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }


private:
    scene_rdl2::math::Vec3f mN;
    float mEta;
    float mRoughness;
    float mThickness;
    scene_rdl2::math::Color mAttenuationColor;
    bool mRefracts;
    ispc::MicrofacetDistribution mMicrofacetDistribution;
    ispc::MicrofacetGeometric mMicrofacetGeometric;
};

class MirrorClearcoat : public BsdfComponent
{
public:
    MirrorClearcoat(
            const scene_rdl2::math::Vec3f& N,
            const float eta,
            const float thickness,
            const scene_rdl2::math::Color& attenuationColor,
            const bool refracts,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mThickness(thickness),
        mAttenuationColor(attenuationColor),
        mRefracts(refracts)
    {}

    ~MirrorClearcoat() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline float                               getEta()                    const { return mEta; }
    finline float                               getThickness()              const { return mThickness; }
    finline const scene_rdl2::math::Color&      getAttenuationColor()       const { return mAttenuationColor; }
    finline bool                                getRefracts()               const { return mRefracts; }

private:
    scene_rdl2::math::Vec3f mN;
    float mEta;
    float mThickness;
    scene_rdl2::math::Color mAttenuationColor;
    bool mRefracts;
};

class LambertianBRDF : public BsdfComponent
{
public:
    LambertianBRDF(const scene_rdl2::math::Vec3f& N,
                   const scene_rdl2::math::Color& albedo,
                   const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mAlbedo(albedo) {}

    ~LambertianBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()        const { return mN; }
    finline const scene_rdl2::math::Color& getAlbedo()   const { return mAlbedo; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Color mAlbedo;
};

class LambertianBTDF : public BsdfComponent
{
public:
    // Caller's responsibility to pass negated normal
    LambertianBTDF(const scene_rdl2::math::Vec3f& N,
                   const scene_rdl2::math::Color& tint) :
        BsdfComponent(nullptr),
        mN(N),
        mTint(tint) {}

    ~LambertianBTDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()       const { return mN; }
    finline const scene_rdl2::math::Color& getTint()    const { return mTint; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Color mTint;
};

class OrenNayarBRDF : public BsdfComponent
{
public:
    OrenNayarBRDF(const scene_rdl2::math::Vec3f& N,
                  const scene_rdl2::math::Color& albedo,
                  const float roughness,
                  const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mAlbedo(albedo),
        mRoughness(roughness) {}

    ~OrenNayarBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()        const { return mN; }
    finline const scene_rdl2::math::Color& getAlbedo()   const { return mAlbedo; }
    finline float getRoughness()                         const { return mRoughness; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Color mAlbedo;
    float mRoughness;
};

class FlatDiffuseBRDF : public OrenNayarBRDF
{
public:
    FlatDiffuseBRDF(const scene_rdl2::math::Vec3f& N,
                    const scene_rdl2::math::Color& albedo,
                    const float roughness,
                    const float terminatorShift,
                    const float flatness,
                    const float flatnessFalloff,
                    const Iridescence* const iridescence = nullptr) :
        OrenNayarBRDF(N, albedo, roughness, iridescence),
        mTerminatorShift(terminatorShift),
        mFlatness(flatness),
        mFlatnessFalloff(flatnessFalloff) {}

    ~FlatDiffuseBRDF() override {}

    finline float getTerminatorShift()    const { return mTerminatorShift; }
    finline float getFlatness()           const { return mFlatness; }
    finline float getFlatnessFalloff()    const { return mFlatnessFalloff; }

private:
    float mTerminatorShift;
    float mFlatness;
    float mFlatnessFalloff;
};

// Represents a stand-alone mirror-smooth reflection,
// without a corresponding transmission. Can represent
// both dielectric and conductor fresnel behaviors.
class MirrorBRDF : public BsdfComponent
{
public:
    // dielectric ctor
    MirrorBRDF(const scene_rdl2::math::Vec3f &N,
               float eta,
               const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(scene_rdl2::math::Color(eta)),
        mK(scene_rdl2::math::sBlack),
        mIsConductor(false)
    {}

    // conductor ctor
    MirrorBRDF(const scene_rdl2::math::Color& eta,
               const scene_rdl2::math::Color& k,
               const scene_rdl2::math::Vec3f N,
               const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mK(k),
        mIsConductor(true)
    {}

    // artist-friendly conductor ctor
    MirrorBRDF(const scene_rdl2::math::Vec3f &N,
               const scene_rdl2::math::Color& reflectivity,
               const scene_rdl2::math::Color& edgeTint,
               const Iridescence* const iridescence = nullptr);

    ~MirrorBRDF() override {}

    finline const scene_rdl2::math::Vec3f&  getN()        const { return mN; }
    finline const scene_rdl2::math::Color&  getEta()      const { return mEta; }
    finline const scene_rdl2::math::Color&  getK()        const { return mK; }
    finline bool                            isConductor() const { return mIsConductor; }

private:
    scene_rdl2::math::Vec3f     mN;
    scene_rdl2::math::Color     mEta;
    scene_rdl2::math::Color     mK;
    bool            mIsConductor;
};

// Represents a stand-alone mirror-smooth transmission,
// without a corresponding reflection.
class MirrorBTDF : public BsdfComponent
{
public:
    MirrorBTDF(const scene_rdl2::math::Vec3f &N,
               float eta,
               const scene_rdl2::math::Color &tint,
               float abbeNumber) :
        BsdfComponent(nullptr),
        mN(N),
        mEta(eta),
        mTint(tint),
        mAbbeNumber(abbeNumber)
    {}

    ~MirrorBTDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()          const { return mN; }
    finline float                               getEta()        const { return mEta; }
    finline const scene_rdl2::math::Color&      getTint()       const { return mTint; }
    finline float                               getAbbeNumber() const { return mAbbeNumber; }

private:
    scene_rdl2::math::Vec3f mN;
    float                   mEta;
    scene_rdl2::math::Color mTint;
    float                   mAbbeNumber;
};

// Represents a dielectric mirror-smooth reflection with
// corresponsing transmission, using correctly coupled
// Fresnel behavior.
class MirrorBSDF : public BsdfComponent
{
public:
    MirrorBSDF(const scene_rdl2::math::Vec3f &N,
               float eta,
               const scene_rdl2::math::Color& tint,
               float abbeNumber,
               float refractionEta,
               float reflectionWeight,
               float transmissionWeight,
               const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mTint(tint),
        mAbbeNumber(abbeNumber),
        mRefractionEta(refractionEta),
        mReflectionWeight(reflectionWeight),
        mTransmissionWeight(transmissionWeight)
    {}

    ~MirrorBSDF() override {}

    finline const scene_rdl2::math::Vec3f&  getN()                  const { return mN; }
    finline float                           getEta()                const { return mEta; }
    finline const scene_rdl2::math::Color&  getTint()               const { return mTint; }
    finline float                           getAbbeNumber()         const { return mAbbeNumber; }
    finline float                           getRefractionEta()      const { return mRefractionEta; }
    finline float                           getReflectionWeight()   const { return mReflectionWeight; }
    finline float                           getTransmissionWeight() const { return mTransmissionWeight; }

private:
    scene_rdl2::math::Vec3f mN;
    float                   mEta;
    scene_rdl2::math::Color mTint;
    float                   mAbbeNumber;
    float                   mRefractionEta;
    float                   mReflectionWeight;
    float                   mTransmissionWeight;
};

// Represents a stand-alone anisotropic microfacet reflection,
// without a corresponding transmission. Can represent
// both dielectric and conductor fresnel behaviors.
class MicrofacetAnisotropicBRDF : public BsdfComponent
{
public:
    // dielectric ctor
    MicrofacetAnisotropicBRDF(
            const scene_rdl2::math::Vec3f& N,
            float eta,
            float roughnessU,
            float roughnessV,
            const scene_rdl2::math::Vec3f& shadingTangent,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr);

    // conductor ctor
    MicrofacetAnisotropicBRDF(
            const scene_rdl2::math::Color& eta,
            const scene_rdl2::math::Color& k,
            const scene_rdl2::math::Vec3f& N,
            float roughnessU,
            float roughnessV,
            const scene_rdl2::math::Vec3f& shadingTangent,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mK(k),
        mRoughnessU(roughnessU),
        mRoughnessV(roughnessV),
        mShadingTangent(shadingTangent),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric),
        mIsConductor(true),
        mFavg(scene_rdl2::math::sWhite)
    {
        // mFavg is the Averge Fresnel Reflectance.
        // Ref:"Revisiting Physically Based Shading", Kulla'17
        // TODO - support for anisotropy
    }

    // artist-friendly conductor ctor
    MicrofacetAnisotropicBRDF(
            const scene_rdl2::math::Vec3f& N,
            const scene_rdl2::math::Color& reflectivity,
            const scene_rdl2::math::Color& edgeTint,
            float roughnessU,
            float roughnessV,
            const scene_rdl2::math::Vec3f& shadingTangent,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr);

    ~MicrofacetAnisotropicBRDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline const scene_rdl2::math::Color&      getEta()                    const { return mEta; }
    finline const scene_rdl2::math::Color&      getK()                      const { return mK; }
    finline float                               getRoughnessU()             const { return mRoughnessU; }
    finline float                               getRoughnessV()             const { return mRoughnessV; }
    finline const scene_rdl2::math::Vec3f&      getShadingTangent()         const { return mShadingTangent; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }
    finline bool                                isConductor()               const { return mIsConductor; }
    finline scene_rdl2::math::Color             getFavg()                   const { return mFavg; }

private:
    scene_rdl2::math::Vec3f                     mN;
    scene_rdl2::math::Color                     mEta;
    scene_rdl2::math::Color                     mK;
    float                                       mRoughnessU;
    float                                       mRoughnessV;
    scene_rdl2::math::Vec3f                     mShadingTangent;
    ispc::MicrofacetDistribution                mMicrofacetDistribution;
    ispc::MicrofacetGeometric                   mMicrofacetGeometric;
    bool                                        mIsConductor;
    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    scene_rdl2::math::Color                     mFavg;
};

// Represents a stand-alone isotropic microfacet reflection,
// without a corresponding transmission. Can represent
// both dielectric and conductor fresnel behaviors.
class MicrofacetIsotropicBRDF : public BsdfComponent
{
public:
    // dielectric ctor
    MicrofacetIsotropicBRDF(
            const scene_rdl2::math::Vec3f& N,
            float eta,
            float roughness,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr);
    // conductor ctor
    MicrofacetIsotropicBRDF(
            const scene_rdl2::math::Color& eta,
            const scene_rdl2::math::Color& k,
            const scene_rdl2::math::Vec3f& N,
            float roughness,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mK(k),
        mRoughness(roughness),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric),
        mIsConductor(true),
        mFavg(scene_rdl2::math::sWhite)
    {
        // TODO: what's the average reflectance here (mFavg)?
        // This code path is currently not used in our materials,
        // but when it is, we'll need a solution for this value.
    }

    // artist-friendly conductor ctor
    MicrofacetIsotropicBRDF(
            const scene_rdl2::math::Vec3f& N,
            const scene_rdl2::math::Color& reflectivity,
            const scene_rdl2::math::Color& edgeTint,
            float roughness,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const Iridescence* const iridescence = nullptr);

    ~MicrofacetIsotropicBRDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline const scene_rdl2::math::Color&      getEta()                    const { return mEta; }
    finline const scene_rdl2::math::Color&      getK()                      const { return mK; }
    finline float                               getRoughness()              const { return mRoughness; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }
    finline bool                                isConductor()               const { return mIsConductor; }
    finline scene_rdl2::math::Color             getFavg()                   const { return mFavg; }

private:
    scene_rdl2::math::Vec3f             mN;
    scene_rdl2::math::Color             mEta;
    scene_rdl2::math::Color             mK;
    float                   mRoughness;
    ispc::MicrofacetDistribution  mMicrofacetDistribution;
    ispc::MicrofacetGeometric     mMicrofacetGeometric;
    bool                    mIsConductor;
    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    scene_rdl2::math::Color             mFavg;
};

// Represents a stand-alone isotropic microfacet transmission,
// without a corresponding reflection.
// NOTE: At this time the underlying implementation does not
// support anisotropic BTDF and will fall back to isotropic using
// the average of roughnessU and roughnessV. This component
// is provided for completeness in the API. Additionally,
// only Beckmann is supported for the BTDF regardless of which
// distribution is requested.
class MicrofacetAnisotropicBTDF : public BsdfComponent
{
public:
    MicrofacetAnisotropicBTDF(
            const scene_rdl2::math::Vec3f& N,
            float eta,
            float roughnessU,
            float roughnessV,
            const scene_rdl2::math::Vec3f& shadingTangent,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const scene_rdl2::math::Color &tint,
            float abbeNumber) :
        BsdfComponent(nullptr),
        mN(N),
        mEta(eta),
        mRoughnessU(roughnessU),
        mRoughnessV(roughnessV),
        mShadingTangent(shadingTangent),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric),
        mTint(tint),
        mAbbeNumber(abbeNumber)
    {}

    ~MicrofacetAnisotropicBTDF() override {}

    finline const scene_rdl2::math::Vec3f&          getN()                      const { return mN; }
    finline float                       getEta()                    const { return mEta; }
    finline float                       getRoughnessU()             const { return mRoughnessU; }
    finline float                       getRoughnessV()             const { return mRoughnessV; }
    finline const scene_rdl2::math::Vec3f&          getShadingTangent()         const { return mShadingTangent; }
    finline ispc::MicrofacetDistribution getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric   getMicrofacetGeometric()    const { return mMicrofacetGeometric; }
    finline const scene_rdl2::math::Color&          getTint()                   const { return mTint; }
    finline float                       getAbbeNumber()             const { return mAbbeNumber; }

private:
    scene_rdl2::math::Vec3f             mN;
    float                               mEta;
    float                               mRoughnessU;
    float                               mRoughnessV;
    scene_rdl2::math::Vec3f             mShadingTangent;
    ispc::MicrofacetDistribution        mMicrofacetDistribution;
    ispc::MicrofacetGeometric           mMicrofacetGeometric;
    scene_rdl2::math::Color             mTint;
    float                               mAbbeNumber;
};

// Represents a stand-alone isotropic microfacet transmission,
// without a corresponding reflection.
class MicrofacetIsotropicBTDF : public BsdfComponent
{
public:
    MicrofacetIsotropicBTDF(
            const scene_rdl2::math::Vec3f& N,
            float eta,
            float roughness,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const scene_rdl2::math::Color &tint,
            float abbeNumber) :
        BsdfComponent(nullptr),
        mN(N),
        mEta(eta),
        mRoughness(roughness),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric),
        mTint(tint),
        mAbbeNumber(abbeNumber)
    {}

    ~MicrofacetIsotropicBTDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline float                               getEta()                    const { return mEta; }
    finline float                               getRoughness()              const { return mRoughness; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }
    finline const scene_rdl2::math::Color&      getTint()                   const { return mTint; }
    finline float                               getAbbeNumber()             const { return mAbbeNumber; }

private:
    scene_rdl2::math::Vec3f         mN;
    float                           mEta;
    float                           mRoughness;
    ispc::MicrofacetDistribution    mMicrofacetDistribution;
    ispc::MicrofacetGeometric       mMicrofacetGeometric;
    scene_rdl2::math::Color         mTint;
    float                           mAbbeNumber;
};

// Represents a dielectric isotropic microfacet reflection with
// corresponsing transmission, using correctly coupled
// Fresnel behavior.
// NOTE: At this time the underlying implementation does not
// support anisotropic BTDF and will fall back to isotropic using
// the average of roughnessU and roughnessV. This component
// is provided for completeness in the API. Additionally,
// only Beckmann is supported for the BTDF regardless of which
// distribution is requested.
class MicrofacetAnisotropicBSDF : public BsdfComponent
{
public:
    MicrofacetAnisotropicBSDF(
            const scene_rdl2::math::Vec3f& N,
            float eta,
            float roughnessU,
            float roughnessV,
            const scene_rdl2::math::Vec3f& shadingTangent,
            ispc::MicrofacetDistribution microfacetDistribution,
            ispc::MicrofacetGeometric microfacetGeometric,
            const scene_rdl2::math::Color& tint,
            float abbeNumber,
            float refractionEta,
            float reflectionWeight,
            float transmissionWeight,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mEta(eta),
        mRoughnessU(roughnessU),
        mRoughnessV(roughnessV),
        mShadingTangent(shadingTangent),
        mMicrofacetDistribution(microfacetDistribution),
        mMicrofacetGeometric(microfacetGeometric),
        mTint(tint),
        mAbbeNumber(abbeNumber),
        mRefractionEta(refractionEta),
        mReflectionWeight(reflectionWeight),
        mTransmissionWeight(transmissionWeight)
    {}

    ~MicrofacetAnisotropicBSDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline float                               getEta()                    const { return mEta; }
    finline float                               getRoughnessU()             const { return mRoughnessU; }
    finline float                               getRoughnessV()             const { return mRoughnessV; }
    finline const scene_rdl2::math::Vec3f&      getShadingTangent()         const { return mShadingTangent; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }
    finline const scene_rdl2::math::Color&      getTint()                   const { return mTint; }
    finline float                               getAbbeNumber()             const { return mAbbeNumber; }
    finline float                               getRefractionEta()          const { return mRefractionEta; }
    finline float                               getReflectionWeight()       const { return mReflectionWeight; }
    finline float                               getTransmissionWeight()     const { return mTransmissionWeight; }

private:
    scene_rdl2::math::Vec3f             mN;
    float                               mEta;
    float                               mRoughnessU;
    float                               mRoughnessV;
    scene_rdl2::math::Vec3f             mShadingTangent;
    ispc::MicrofacetDistribution        mMicrofacetDistribution;
    ispc::MicrofacetGeometric           mMicrofacetGeometric;
    scene_rdl2::math::Color             mTint;
    float                               mAbbeNumber;
    float                               mRefractionEta;
    float                               mReflectionWeight;
    float                               mTransmissionWeight;
};

// Represents a dielectric isotropic microfacet reflection with
// corresponsing transmission, using correctly coupled
// Fresnel behavior.
class MicrofacetIsotropicBSDF : public BsdfComponent
{
public:
    MicrofacetIsotropicBSDF(const scene_rdl2::math::Vec3f& N,
                            float eta,
                            float roughness,
                            ispc::MicrofacetDistribution microfacetDistribution,
                            ispc::MicrofacetGeometric microfacetGeometric,
                            const scene_rdl2::math::Color& tint,
                            float abbeNumber,
                            float refractionEta,
                            float reflectionWeight,
                            float transmissionWeight,
                            const Iridescence* const iridescence = nullptr);

    ~MicrofacetIsotropicBSDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()                      const { return mN; }
    finline float                               getEta()                    const { return mEta; }
    finline float                               getRoughness()              const { return mRoughness; }
    finline ispc::MicrofacetDistribution        getMicrofacetDistribution() const { return mMicrofacetDistribution; }
    finline ispc::MicrofacetGeometric           getMicrofacetGeometric()    const { return mMicrofacetGeometric; }
    finline const scene_rdl2::math::Color&      getTint()                   const { return mTint; }
    finline float                               getAbbeNumber()             const { return mAbbeNumber; }
    finline float                               getRefractionEta()          const { return mRefractionEta; }
    finline float                               getReflectionWeight()       const { return mReflectionWeight; }
    finline float                               getTransmissionWeight()     const { return mTransmissionWeight; }

private:
    scene_rdl2::math::Vec3f             mN;
    float                               mEta;
    float                               mRoughness;
    ispc::MicrofacetDistribution        mMicrofacetDistribution;
    ispc::MicrofacetGeometric           mMicrofacetGeometric;
    scene_rdl2::math::Color             mTint;
    float                               mAbbeNumber;
    float                               mRefractionEta;
    float                               mReflectionWeight;
    float                               mTransmissionWeight;
};

class DipoleDiffusion : public BsdfComponent
{
public:
    DipoleDiffusion(
            const scene_rdl2::math::Vec3f &N,
            const scene_rdl2::math::Color &albedo,
            const scene_rdl2::math::Color &radius,
            const scene_rdl2::rdl2::Material* const material,
            const scene_rdl2::rdl2::TraceSet* const traceSet,
            const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);

    ~DipoleDiffusion() override {}

    finline const scene_rdl2::math::Vec3f& getN()                const { return mN; }
    finline const scene_rdl2::math::Color& getAlbedo()           const { return mAlbedo; }
    finline const scene_rdl2::math::Color& getRadius()           const { return mRadius; }
    finline const scene_rdl2::rdl2::Material* getMaterial()      const { return mMaterial; }
    finline const scene_rdl2::rdl2::TraceSet* getTraceSet()      const { return mTraceSet; }
    finline scene_rdl2::rdl2::EvalNormalFunc getEvalNormalFn()   const { return mEvalNormalFn; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Color mAlbedo;
    scene_rdl2::math::Color mRadius;
    const scene_rdl2::rdl2::Material* const mMaterial;
    const scene_rdl2::rdl2::TraceSet* const mTraceSet;
    scene_rdl2::rdl2::EvalNormalFunc mEvalNormalFn;

};

class NormalizedDiffusion : public BsdfComponent
{
public:
    // Constructor / Destructor
    NormalizedDiffusion(
            const scene_rdl2::math::Vec3f &N,
            const scene_rdl2::math::Color &albedo,
            const scene_rdl2::math::Color &radius,
            const scene_rdl2::rdl2::Material* const material,
            const scene_rdl2::rdl2::TraceSet* const traceSet,
            const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);

    ~NormalizedDiffusion() override {}

    finline const scene_rdl2::math::Vec3f& getN()               const { return mN; }
    finline const scene_rdl2::math::Color& getAlbedo()          const { return mAlbedo; }
    finline const scene_rdl2::math::Color& getRadius()          const { return mRadius; }
    finline const scene_rdl2::rdl2::Material* getMaterial()     const { return mMaterial; }
    finline const scene_rdl2::rdl2::TraceSet* getTraceSet()     const { return mTraceSet; }
    finline scene_rdl2::rdl2::EvalNormalFunc getEvalNormalFn()  const { return mEvalNormalFn; }

private:
    const scene_rdl2::math::Vec3f &mN;
    const scene_rdl2::math::Color &mAlbedo;
    const scene_rdl2::math::Color &mRadius;
    const scene_rdl2::rdl2::Material* const mMaterial;
    const scene_rdl2::rdl2::TraceSet* const mTraceSet;
    scene_rdl2::rdl2::EvalNormalFunc mEvalNormalFn;
};

class RandomWalkSubsurface : public BsdfComponent
{
public:
    // Constructor / Destructor
    RandomWalkSubsurface(
            const scene_rdl2::math::Vec3f &N,
            const scene_rdl2::math::Color &albedo,
            const scene_rdl2::math::Color &radius,
            const bool resolveSelfIntersections,
            const scene_rdl2::rdl2::Material* const material,
            const scene_rdl2::rdl2::TraceSet* const traceSet,
            const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);

    ~RandomWalkSubsurface() override {}

    finline const scene_rdl2::math::Vec3f& getN()               const { return mN; }
    finline const scene_rdl2::math::Color& getAlbedo()          const { return mAlbedo; }
    finline const scene_rdl2::math::Color& getRadius()          const { return mRadius; }
    finline bool getResolveSelfIntersections()                  const { return mResolveSelfIntersections; }
    finline const scene_rdl2::rdl2::Material* getMaterial()     const { return mMaterial; }
    finline const scene_rdl2::rdl2::TraceSet* getTraceSet()     const { return mTraceSet; }
    finline scene_rdl2::rdl2::EvalNormalFunc getEvalNormalFn()  const { return mEvalNormalFn; }

private:
    const scene_rdl2::math::Vec3f &mN;
    const scene_rdl2::math::Color &mAlbedo;
    const scene_rdl2::math::Color &mRadius;
    const bool mResolveSelfIntersections;
    const scene_rdl2::rdl2::Material* const mMaterial;
    const scene_rdl2::rdl2::TraceSet* const mTraceSet;
    scene_rdl2::rdl2::EvalNormalFunc mEvalNormalFn;
};

class FabricBRDF : public BsdfComponent
{
public:
    FabricBRDF(const scene_rdl2::math::Vec3f &N,
               const scene_rdl2::math::Vec3f &T,
               const scene_rdl2::math::Vec3f &threadDirection,
               float threadElevation,
               float roughness,
               const scene_rdl2::math::Color& fabricColor,
               const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N), mT(T),
        mThreadDirection(threadDirection),
        mThreadElevation(threadElevation),
        mRoughness(roughness),
        mFabricColor(fabricColor)
    {}

    ~FabricBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()                const { return mN; }
    finline const scene_rdl2::math::Vec3f& getT()                const { return mT; }
    finline const scene_rdl2::math::Vec3f& getThreadDirection()  const { return mThreadDirection; }
    finline float                          getThreadElevation()  const { return mThreadElevation; }
    finline float                          getRoughness()        const { return mRoughness; }
    finline const scene_rdl2::math::Color& getFabricColor()      const { return mFabricColor; }
private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Vec3f mT;
    scene_rdl2::math::Vec3f mThreadDirection;
    float mThreadElevation;
    float mRoughness;
    scene_rdl2::math::Color mFabricColor;
};

class VelvetBRDF : public BsdfComponent
{
public:
    VelvetBRDF(const scene_rdl2::math::Vec3f &N,
               float roughness,
               const scene_rdl2::math::Color& color,
               bool useAbsorbingFibers,
               const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mRoughness(roughness),
        mColor(color),
        mUseAbsorbingFibers(useAbsorbingFibers)
    {}

    ~VelvetBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()       const { return mN; };
    finline float              getRoughness()           const { return mRoughness; }
    finline const scene_rdl2::math::Color& getColor()   const { return mColor; }
    finline bool               getUseAbsorbingFibers()  const { return mUseAbsorbingFibers; }

private:
    scene_rdl2::math::Vec3f     mN;
    float           mRoughness;
    scene_rdl2::math::Color     mColor;
    bool            mUseAbsorbingFibers;
};

class EyeCausticBRDF : public BsdfComponent
{
public:
    EyeCausticBRDF(const scene_rdl2::math::Vec3f& N,
                   const scene_rdl2::math::Vec3f& irisN,
                   const scene_rdl2::math::Color& causticColor,
                   const float exponent) :
        BsdfComponent(nullptr),
        mN(N),
        mIrisNormal(irisN),
        mCausticColor(causticColor),
        mExponent(exponent)
    {}

    ~EyeCausticBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()                const { return mN; }
    finline const scene_rdl2::math::Vec3f& getIrisNormal()       const { return mIrisNormal; }
    finline const scene_rdl2::math::Color& getCausticColor()     const { return mCausticColor; }
    finline float getExponent()                                  const { return mExponent; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Vec3f mIrisNormal;
    scene_rdl2::math::Color mCausticColor;
    float mExponent;
};

class HairDiffuseBSDF : public BsdfComponent
{
public:
    HairDiffuseBSDF(const scene_rdl2::math::Vec3f& hairDir,
                const scene_rdl2::math::Color& reflectionColor,
                const scene_rdl2::math::Color& transmissionColor) :
        mHairDir(hairDir),
        mReflectionColor(reflectionColor),
        mTransmissionColor(transmissionColor)
    {}

    ~HairDiffuseBSDF() override {}

    finline const scene_rdl2::math::Vec3f& getHairDir()             const { return mHairDir; }
    finline const scene_rdl2::math::Color& getReflectionColor()     const { return mReflectionColor; }
    finline const scene_rdl2::math::Color& getTransmissionColor()   const { return mTransmissionColor; }

private:
    scene_rdl2::math::Vec3f mHairDir;
    scene_rdl2::math::Color mReflectionColor;
    scene_rdl2::math::Color mTransmissionColor;
};

class HairBSDF : public BsdfComponent
 {
 public:
     HairBSDF(const scene_rdl2::math::Vec3f& hairDir,
              const scene_rdl2::math::Vec2f& hairUV,
              const float ior,
              const ispc::HairFresnelType fresnelType,
              const float cuticleLayerThickness,
              bool  showR,
              float shiftR,
              float roughnessR,
              const scene_rdl2::math::Color& tintR,
              bool  showTT,
              float shiftTT,
              float roughnessTT,
              float azimuthalRoughnessTT,
              const scene_rdl2::math::Color& tintTT,
              float saturationTT,
              bool  showTRT,
              float shiftTRT,
              float roughnessTRT,
              const scene_rdl2::math::Color& tintTRT,
              bool  showGlint,
              float roughnessGlint,
              float eccentricityGlint,
              float saturationGlint,
              float hairRotation,
              const scene_rdl2::math::Vec3f& hairNormal,
              bool  showTRRT,
              const scene_rdl2::math::Color &hairColor):
         mHairDir(hairDir),
         mHairUV(hairUV),
         mIOR(ior),
         mFresnelType(fresnelType),
         mCuticleLayerThickness(cuticleLayerThickness),
         mShowR(showR),
         mShiftR(shiftR),
         mRoughnessR(roughnessR),
         mTintR(tintR),
         mShowTT(showTT),
         mShiftTT(shiftTT),
         mRoughnessTT(roughnessTT),
         mAziRoughnessTT(azimuthalRoughnessTT),
         mTintTT(tintTT),
         mSaturationTT(saturationTT),
         mShowTRT(showTRT),
         mShiftTRT(shiftTRT),
         mRoughnessTRT(roughnessTRT),
         mTintTRT(tintTRT),
         mShowGlint(showGlint),
         mRoughnessGlint(roughnessGlint),
         mEccentricityGlint(eccentricityGlint),
         mSaturationGlint(saturationGlint),
         mHairRotation(hairRotation),
         mHairNormal(hairNormal),
         mShowTRRT(showTRRT),
         mHairColor(hairColor)
     {}

     ~HairBSDF() override {}

     finline const scene_rdl2::math::Vec3f& getHairDir()   const { return mHairDir; }
     finline const scene_rdl2::math::Vec2f& getHairUV()    const { return mHairUV; }
     finline float              getIOR()                   const { return mIOR; }
     finline ispc::HairFresnelType getFresnelType()        const { return mFresnelType; }
     finline float              getCuticleLayerThickness() const { return mCuticleLayerThickness; }

     finline bool               getShowR()                 const { return mShowR; }
     finline float              getShiftR()                const { return mShiftR; }
     finline float              getRoughnessR()            const { return mRoughnessR; }
     finline const scene_rdl2::math::Color& getTintR()     const { return mTintR; }

     finline bool               getShowTT()                const { return mShowTT; }
     finline float              getShiftTT()               const { return mShiftTT; }
     finline float              getRoughnessTT()           const { return mRoughnessTT; }
     finline float              getAziRoughnessTT()        const { return mAziRoughnessTT; }
     finline const scene_rdl2::math::Color& getTintTT()    const { return mTintTT; }
     finline float              getSaturationTT()          const { return mSaturationTT; }

     finline bool               getShowTRT()               const { return mShowTRT; }
     finline float              getShiftTRT()              const { return mShiftTRT; }
     finline float              getRoughnessTRT()          const { return mRoughnessTRT; }
     finline const scene_rdl2::math::Color& getTintTRT()   const { return mTintTRT; }

     finline bool               getShowGlint()             const { return mShowGlint; }
     finline float              getRoughnessGlint()        const { return mRoughnessGlint; }
     finline float              getEccentricityGlint()     const { return mEccentricityGlint; }
     finline float              getSaturationGlint()       const { return mSaturationGlint; }
     finline float              getHairRotation()          const { return mHairRotation; }
     finline const scene_rdl2::math::Vec3f& getHairNormal() const { return mHairNormal; }

     finline bool               getShowTRRT()              const { return mShowTRRT; }

     finline const scene_rdl2::math::Color& getHairColor() const { return mHairColor; }

 private:
     scene_rdl2::math::Vec3f mHairDir;
     scene_rdl2::math::Vec2f mHairUV;

     float mIOR;

     ispc::HairFresnelType mFresnelType;
     float mCuticleLayerThickness;

     bool  mShowR;
     float mShiftR;
     float mRoughnessR;
     scene_rdl2::math::Color mTintR;

     bool  mShowTT;
     float mShiftTT;
     float mRoughnessTT;
     float mAziRoughnessTT;
     scene_rdl2::math::Color mTintTT;
     float mSaturationTT;

     bool  mShowTRT;
     float mShiftTRT;
     float mRoughnessTRT;
     scene_rdl2::math::Color mTintTRT;

     bool  mShowGlint;
     float mRoughnessGlint;
     float mEccentricityGlint;
     float mSaturationGlint;
     float mHairRotation;
     scene_rdl2::math::Vec3f mHairNormal;

     bool mShowTRRT;

     scene_rdl2::math::Color mHairColor;
 };

 class HairRBRDF : public BsdfComponent
 {
 public:
     HairRBRDF(const scene_rdl2::math::Vec3f& hairDir,
               const scene_rdl2::math::Vec2f& hairUV,
               const float ior,
               const ispc::HairFresnelType fresnelType,
               const float cuticleLayerThickness,
               float shift,
               float roughness,
               const scene_rdl2::math::Color& tint) :
         mHairDir(hairDir),
         mHairUV(hairUV),
         mIOR(ior),
         mFresnelType(fresnelType),
         mCuticleLayerThickness(cuticleLayerThickness),
         mShift(shift),
         mRoughness(roughness),
         mTint(tint)
     {}

     ~HairRBRDF() override {}

     finline const scene_rdl2::math::Vec3f& getHairDir()        const { return mHairDir; }
     finline const scene_rdl2::math::Vec2f& getHairUV()         const { return mHairUV; }
     finline float              getIOR()                        const { return mIOR; }
     finline ispc::HairFresnelType getFresnelType()             const { return mFresnelType; }
     finline float              getCuticleLayerThickness()      const { return mCuticleLayerThickness; }
     finline float              getShift()                      const { return mShift; }
     finline float              getRoughness()                  const { return mRoughness; }
     finline const scene_rdl2::math::Color& getTint()           const { return mTint; }

 private:
     scene_rdl2::math::Vec3f mHairDir;
     scene_rdl2::math::Vec2f mHairUV;
     float mIOR;

     ispc::HairFresnelType mFresnelType;
     float mCuticleLayerThickness;
     float mShift;
     float mRoughness;
     scene_rdl2::math::Color mTint;
 };

 class HairTRTBRDF : public BsdfComponent
 {
 public:
     HairTRTBRDF(const scene_rdl2::math::Vec3f& hairDir,
                 const scene_rdl2::math::Vec2f& hairUV,
                 const float ior,
                 const ispc::HairFresnelType fresnelType,
                 const float cuticleLayerThickness,
                 float shift,
                 float roughness,
                 float aziRoughness,
                 const scene_rdl2::math::Color &hairColor,
                 const scene_rdl2::math::Color& tint,
                 bool showGlint,
                 float roughnessGlint,
                 float eccentricityGlint,
                 float saturationGlint,
                 const float hairRotation,
                 const scene_rdl2::math::Vec3f& hairNormal) :
         mHairDir(hairDir),
         mHairUV(hairUV),
         mIOR(ior),
         mFresnelType(fresnelType),
         mCuticleLayerThickness(cuticleLayerThickness),
         mShift(shift),
         mRoughness(roughness),
         mAziRoughness(aziRoughness),
         mHairColor(hairColor),
         mTint(tint),
         mShowGlint(showGlint),
         mRoughnessGlint(roughnessGlint),
         mEccentricityGlint(eccentricityGlint),
         mSaturationGlint(saturationGlint),
         mHairRotation(hairRotation),
         mHairNormal(hairNormal)
     {}

     ~HairTRTBRDF() override {}

     finline const scene_rdl2::math::Vec3f&  getHairDir()   const { return mHairDir; }
     finline const scene_rdl2::math::Vec2f&  getHairUV()    const { return mHairUV; }
     finline float               getIOR()                   const { return mIOR; }
     finline ispc::HairFresnelType getFresnelType()         const { return mFresnelType; }
     finline float               getCuticleLayerThickness() const { return mCuticleLayerThickness; }
     finline float               getShift()                 const { return mShift; }
     finline float               getRoughness()             const { return mRoughness; }
     finline float               getAziRoughness()          const { return mAziRoughness; }
     finline const scene_rdl2::math::Color&  getHairColor() const { return mHairColor; }
     finline const scene_rdl2::math::Color&  getTint()      const { return mTint; }
     finline bool                getShowGlint()             const { return mShowGlint; }
     finline float               getRoughnessGlint()        const { return mRoughnessGlint; }
     finline float               getEccentricityGlint()     const { return mEccentricityGlint; }
     finline float               getSaturationGlint()       const { return mSaturationGlint; }
     finline float               getHairRotation()          const { return mHairRotation; }
     finline const scene_rdl2::math::Vec3f&  getHairNormal() const { return mHairNormal; }

 private:
     scene_rdl2::math::Vec3f mHairDir;
     scene_rdl2::math::Vec2f mHairUV;
     float mIOR;
     ispc::HairFresnelType mFresnelType;
     float mCuticleLayerThickness;
     float mShift;
     float mRoughness;
     float mAziRoughness;
     scene_rdl2::math::Color mHairColor;
     scene_rdl2::math::Color mTint;
     bool mShowGlint;
     float mRoughnessGlint;
     float mEccentricityGlint;
     float mSaturationGlint;
     float mHairRotation;
     scene_rdl2::math::Vec3f mHairNormal;
 };

 class HairTTBTDF : public BsdfComponent
 {
 public:
     HairTTBTDF(const scene_rdl2::math::Vec3f& hairDir,
                const scene_rdl2::math::Vec2f& hairUV,
                const float ior,
                const ispc::HairFresnelType fresnelType,
                const float cuticleLayerThickness,
                float shift,
                float roughness,
                float azimuthalRoughness,
                const scene_rdl2::math::Color &hairColor,
                const scene_rdl2::math::Color& tint,
                float saturation = 1.0f) :
         mHairDir(hairDir),
         mHairUV(hairUV),
         mIOR(ior),
         mFresnelType(fresnelType),
         mCuticleLayerThickness(cuticleLayerThickness),
         mShift(shift),
         mRoughness(roughness),
         mAziRoughness(azimuthalRoughness),
         mHairColor(hairColor),
         mTint(tint),
         mSaturation(saturation)
     {}

     ~HairTTBTDF() override {}

     finline const scene_rdl2::math::Vec3f&  getHairDir()   const { return mHairDir; }
     finline const scene_rdl2::math::Vec2f&  getHairUV()    const { return mHairUV; }
     finline float               getIOR()                   const { return mIOR; }
     finline ispc::HairFresnelType getFresnelType()         const { return mFresnelType; }
     finline float               getCuticleLayerThickness() const { return mCuticleLayerThickness; }
     finline float               getShift()                 const { return mShift; }
     finline float               getRoughness()             const { return mRoughness; }
     finline float               getAziRoughness()          const { return mAziRoughness; }
     finline const scene_rdl2::math::Color&  getHairColor() const { return mHairColor; }
     finline const scene_rdl2::math::Color&  getTint()      const { return mTint; }
     finline float               getSaturation()            const { return mSaturation; }

 private:
     scene_rdl2::math::Vec3f mHairDir;
     scene_rdl2::math::Vec2f mHairUV;
     float mIOR;
     ispc::HairFresnelType mFresnelType;
     float mCuticleLayerThickness;
     float mShift;
     float mRoughness;
     float mAziRoughness;
     scene_rdl2::math::Color mHairColor;
     scene_rdl2::math::Color mTint;
     float mSaturation;
 };

 class HairTRRTBRDF : public BsdfComponent
 {
 public:
     HairTRRTBRDF(const scene_rdl2::math::Vec3f& hairDir,
                  const scene_rdl2::math::Vec2f& hairUV,
                  const float ior,
                  const ispc::HairFresnelType fresnelType,
                  const float cuticleLayerThickness,
                  float roughness,
                  float aziRoughness,
                  const scene_rdl2::math::Color &hairColor,
                  const scene_rdl2::math::Color& tint) :
         mHairDir(hairDir),
         mHairUV(hairUV),
         mIOR(ior),
         mFresnelType(fresnelType),
         mCuticleLayerThickness(cuticleLayerThickness),
         mRoughness(roughness),
         mAziRoughness(aziRoughness),
         mHairColor(hairColor),
         mTint(tint)
     {}

     ~HairTRRTBRDF() override {}

     finline const scene_rdl2::math::Vec3f&  getHairDir()   const { return mHairDir; }
     finline const scene_rdl2::math::Vec2f&  getHairUV()    const { return mHairUV; }
     finline float               getIOR()                   const { return mIOR; }
     finline ispc::HairFresnelType getFresnelType()         const { return mFresnelType; }
     finline float               getCuticleLayerThickness() const { return mCuticleLayerThickness; }
     finline float               getRoughness()             const { return mRoughness; }
     finline float               getAziRoughness()          const { return mAziRoughness; }
     finline const scene_rdl2::math::Color&  getHairColor() const { return mHairColor; }
     finline const scene_rdl2::math::Color&  getTint()      const { return mTint; }

 private:
     scene_rdl2::math::Vec3f mHairDir;
     scene_rdl2::math::Vec2f mHairUV;
     float mIOR;
     ispc::HairFresnelType mFresnelType;
     float mCuticleLayerThickness;
     float mRoughness;
     float mAziRoughness;
     scene_rdl2::math::Color mHairColor;
     scene_rdl2::math::Color mTint;
 };

class GlitterFlakeBRDF : public BsdfComponent
{
public:
    GlitterFlakeBRDF(
            const scene_rdl2::math::Vec3f& N,
            const scene_rdl2::math::Vec3f& flakeN,
            const scene_rdl2::math::Color& reflectivity,
            const scene_rdl2::math::Color& edgeTint,
            float roughness,
            const Iridescence* const iridescence = nullptr);

    ~GlitterFlakeBRDF() override {}

    finline const scene_rdl2::math::Vec3f&      getN()          const { return mN; }
    finline const scene_rdl2::math::Vec3f&      getFlakeN()     const { return mFlakeN; }
    finline const scene_rdl2::math::Color&      getEta()        const { return mEta; }
    finline const scene_rdl2::math::Color&      getK()          const { return mK; }
    finline float                               getRoughness()  const { return mRoughness; }
    finline const scene_rdl2::math::Color&      getFavg()       const { return mFavg; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Vec3f mFlakeN;
    scene_rdl2::math::Color mEta;
    scene_rdl2::math::Color mK;
    float       mRoughness;
    scene_rdl2::math::Color mFavg;
};

class StochasticFlakesBRDF : public BsdfComponent
{
public:
    StochasticFlakesBRDF(
            const scene_rdl2::math::Vec3f& N,
            scene_rdl2::math::Vec3f* flakeNormals,
            scene_rdl2::math::Color* flakeColors,
            const size_t flakeCount,
            float flakeRoughness,
            float flakeRandomness,
            const Iridescence* const iridescence = nullptr) :
        BsdfComponent(iridescence),
        mN(N),
        mFlakeNormals(flakeNormals),
        mFlakeColors(flakeColors),
        mFlakeCount(flakeCount),
        mFlakeRoughness(flakeRoughness),
        mFlakeRandomness(flakeRandomness)
    {}

    ~StochasticFlakesBRDF() override {}

    finline const scene_rdl2::math::Vec3f&  getN()              const { return mN; }
    finline const scene_rdl2::math::Vec3f*  getFlakeNormals()   const { return mFlakeNormals; }
    finline const scene_rdl2::math::Color*  getFlakeColors()    const { return mFlakeColors; }
    finline size_t                          getFlakeCount()     const { return mFlakeCount; }
    finline float                           getFlakeRoughness() const { return mFlakeRoughness; }
    finline float                           getFlakeRandomness() const { return mFlakeRandomness; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Vec3f* mFlakeNormals;
    scene_rdl2::math::Color* mFlakeColors;
    size_t mFlakeCount;
    float mFlakeRoughness;
    float mFlakeRandomness;
};

class ToonBRDF : public BsdfComponent
{
public:
    ToonBRDF(const scene_rdl2::math::Vec3f& N,
             const scene_rdl2::math::Color& albedo,
             int numRampPoints,
             const float* rampPositions,
             const ispc::RampInterpolatorMode* rampInterpolators,
             const scene_rdl2::math::Color* rampColors,
             bool extendRamp,
             const Iridescence* const iridescence = nullptr) :
                 BsdfComponent(iridescence),
                 mN(N),
                 mAlbedo(albedo),
                 mRampNumPoints(numRampPoints),
                 mRampPositions(rampPositions),
                 mRampInterpolators(rampInterpolators),
                 mRampColors(rampColors),
                 mExtendRamp(extendRamp)
    {}

    ~ToonBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN()           const { return mN; }
    finline const scene_rdl2::math::Color& getAlbedo()      const { return mAlbedo; }
    finline int getRampNumPoints()                          const { return mRampNumPoints; }
    finline const float* getRampPositions()                 const { return mRampPositions; }
    finline const ispc::RampInterpolatorMode* getRampInterpolators() const { return mRampInterpolators; }
    finline const scene_rdl2::math::Color* getRampColors()  const { return mRampColors; }
    finline bool getExtendRamp()                            const { return mExtendRamp; }

private:
    scene_rdl2::math::Vec3f mN;
    scene_rdl2::math::Color mAlbedo;
    int mRampNumPoints;
    const float* mRampPositions;
    const ispc::RampInterpolatorMode* mRampInterpolators;
    const scene_rdl2::math::Color* mRampColors;
    bool mExtendRamp;
};

class HairToonSpecularBRDF : public BsdfComponent
{
public:
    HairToonSpecularBRDF(const scene_rdl2::math::Vec3f& N,
                         const scene_rdl2::math::Vec3f& hairDir,
                         const scene_rdl2::math::Vec2f& hairUV,
                         const float ior,
                         const ispc::HairFresnelType fresnelType,
                         const float cuticleLayerThickness,
                         float shift,
                         float roughness,
                         const scene_rdl2::math::Color& tint,
                         float intensity,
                         int numRampPoints,
                         const float* rampPositions,
                         const ispc::RampInterpolatorMode* rampInterpolators,
                         const float* rampValues,
                         bool enableIndirectReflections,
                         float indirectReflectionsIntensity,
                         float indirectReflectionsRoughness) :
        mN(N),
        mHairDir(hairDir),
        mHairUV(hairUV),
        mIOR(ior),
        mFresnelType(fresnelType),
        mCuticleLayerThickness(cuticleLayerThickness),
        mShift(shift),
        mRoughness(roughness),
        mTint(tint),
        mIntensity(intensity),
        mRampNumPoints(numRampPoints),
        mRampPositions(rampPositions),
        mRampInterpolators(rampInterpolators),
        mRampValues(rampValues),
        mEnableIndirectReflections(enableIndirectReflections),
        mIndirectReflectionsIntensity(indirectReflectionsIntensity),
        mIndirectReflectionsRoughness(indirectReflectionsRoughness)
    {}

    ~HairToonSpecularBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN() const { return mN; }
    finline const scene_rdl2::math::Vec3f& getHairDir() const { return mHairDir; }
    finline const scene_rdl2::math::Vec2f& getHairUV() const { return mHairUV; }
    finline float getIOR() const { return mIOR; }
    finline ispc::HairFresnelType getFresnelType() const { return mFresnelType; }
    finline float getCuticleLayerThickness() const { return mCuticleLayerThickness; }
    finline float getShift() const { return mShift; }
    finline float getRoughness() const { return mRoughness; }
    finline const scene_rdl2::math::Color& getTint() const { return mTint; }
    finline float getIntensity() const { return mIntensity; }
    finline int getRampNumPoints() const { return mRampNumPoints; }
    finline const float* getRampPositions() const { return mRampPositions; }
    finline const ispc::RampInterpolatorMode* getRampInterpolators() const { return mRampInterpolators; }
    finline const float* getRampValues() const { return mRampValues; }
    finline bool getEnableIndirectReflections() const { return mEnableIndirectReflections; }
    finline float getIndirectReflectionsIntensity() const { return mIndirectReflectionsIntensity; }
    finline float getIndirectReflectionsRoughness() const { return mIndirectReflectionsRoughness; }

private:
     scene_rdl2::math::Vec3f mN;

     scene_rdl2::math::Vec3f mHairDir;
     scene_rdl2::math::Vec2f mHairUV;

     float mIOR;

     ispc::HairFresnelType mFresnelType;
     float mCuticleLayerThickness;

     float mShift;
     float mRoughness;
     scene_rdl2::math::Color mTint;

    float mIntensity;

    int mRampNumPoints;
    const float* mRampPositions;
    const ispc::RampInterpolatorMode* mRampInterpolators;
    const float* mRampValues;

    bool mEnableIndirectReflections;
    float mIndirectReflectionsIntensity;
    float mIndirectReflectionsRoughness;
};

class ToonSpecularBRDF : public BsdfComponent
{
public:
    // dielectric ctor
    ToonSpecularBRDF(
            const scene_rdl2::math::Vec3f& N,
            float intensity,
            const scene_rdl2::math::Color& tint,
            const float rampInputScale,
            int numRampPoints,
            const float* rampPositions,
            const ispc::RampInterpolatorMode* rampInterpolators,
            const float* rampValues,
            float stretchU,
            float stretchV,
            const scene_rdl2::math::Vec3f& dPds,
            const scene_rdl2::math::Vec3f& dPdt,
            bool enableIndirectReflections,
            float indirectReflectionsIntensity,
            float indirectReflectionsRoughness) :
        mN(N),
        mIntensity(intensity),
        mTint(tint),
        mRampInputScale(rampInputScale),
        mRampNumPoints(numRampPoints),
        mRampPositions(rampPositions),
        mRampInterpolators(rampInterpolators),
        mRampValues(rampValues),
        mStretchU(stretchU),
        mStretchV(stretchV),
        mdPds(dPds),
        mdPdt(dPdt),
        mEnableIndirectReflections(enableIndirectReflections),
        mIndirectReflectionsIntensity(indirectReflectionsIntensity),
        mIndirectReflectionsRoughness(indirectReflectionsRoughness)
    {}

    ~ToonSpecularBRDF() override {}

    finline const scene_rdl2::math::Vec3f& getN() const { return mN; }
    finline float getIntensity() const { return mIntensity; }
    finline scene_rdl2::math::Color getTint() const { return mTint; }
    finline float getRampInputScale() const { return mRampInputScale; }
    finline int getRampNumPoints() const { return mRampNumPoints; }
    finline const float* getRampPositions() const { return mRampPositions; }
    finline const ispc::RampInterpolatorMode* getRampInterpolators() const { return mRampInterpolators; }
    finline const float* getRampValues() const { return mRampValues; }
    finline float getStretchU() const { return mStretchU; }
    finline float getStretchV() const { return mStretchV; }
    finline const scene_rdl2::math::Vec3f& getdPds() const { return mdPds; }
    finline const scene_rdl2::math::Vec3f& getdPdt() const { return mdPdt; }
    finline bool getEnableIndirectReflections() const { return mEnableIndirectReflections; }
    finline float getIndirectReflectionsIntensity() const { return mIndirectReflectionsIntensity; }
    finline float getIndirectReflectionsRoughness() const { return mIndirectReflectionsRoughness; }

private:
    scene_rdl2::math::Vec3f mN;
    float mIntensity;
    scene_rdl2::math::Color mTint;

    const float mRampInputScale;
    int mRampNumPoints;
    const float* mRampPositions;
    const ispc::RampInterpolatorMode* mRampInterpolators;
    const float* mRampValues;

    float mStretchU;
    float mStretchV;
    scene_rdl2::math::Vec3f mdPds;
    scene_rdl2::math::Vec3f mdPdt;

    bool mEnableIndirectReflections;
    float mIndirectReflectionsIntensity;
    float mIndirectReflectionsRoughness;
};

} // end namespace shading
} // end namespace moonray

