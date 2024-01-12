// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "BsdfComponent.h"
#include "bsdf/Fresnel.h"
#include "Ior.h"
#include "Iridescence.h"


namespace moonray {
    using namespace scene_rdl2::math;
namespace shading {

BsdfComponent::~BsdfComponent() {}

// dielectric ctor
MicrofacetAnisotropicBRDF::MicrofacetAnisotropicBRDF(
        const scene_rdl2::math::Vec3f& N,
        float eta,
        float roughnessU,
        float roughnessV,
        const scene_rdl2::math::Vec3f& shadingTangent,
        ispc::MicrofacetDistribution microfacetDistribution,
        ispc::MicrofacetGeometric microfacetGeometric,
        const Iridescence* const iridescence) :
    BsdfComponent(iridescence),
    mN(N),
    mEta(scene_rdl2::math::Color(eta)),
    mK(scene_rdl2::math::sBlack),
    mRoughnessU(roughnessU),
    mRoughnessV(roughnessV),
    mShadingTangent(shadingTangent),
    mMicrofacetDistribution(microfacetDistribution),
    mMicrofacetGeometric(microfacetGeometric),
    mIsConductor(false)
{
    // Average Dielectric Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    float favg = shading::averageFresnelReflectance(eta);
    mFavg = scene_rdl2::math::sWhite * favg;
}

MicrofacetAnisotropicBRDF::MicrofacetAnisotropicBRDF(
        const scene_rdl2::math::Vec3f& N,
        const scene_rdl2::math::Color& reflectivity,
        const scene_rdl2::math::Color& edgeTint,
        float roughnessU,
        float roughnessV,
        const scene_rdl2::math::Vec3f& shadingTangent,
        ispc::MicrofacetDistribution microfacetDistribution,
        ispc::MicrofacetGeometric microfacetGeometric,
        const Iridescence* const iridescence):
    BsdfComponent(iridescence),
    mN(N),
    mRoughnessU(roughnessU),
    mRoughnessV(roughnessV),
    mShadingTangent(shadingTangent),
    mMicrofacetDistribution(microfacetDistribution),
    mMicrofacetGeometric(microfacetGeometric),
    mIsConductor(true)
{
    auto ior = shading::ShaderComplexIor::createComplexIorFromColor(
            reflectivity, edgeTint);
    mEta = ior.getEta();
    mK = ior.getAbsorption();

    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    mFavg = shading::averageFresnelReflectance(reflectivity, edgeTint);
}

MicrofacetIsotropicBRDF::MicrofacetIsotropicBRDF(
        const scene_rdl2::math::Vec3f& N,
        float eta,
        float roughness,
        ispc::MicrofacetDistribution microfacetDistribution,
        ispc::MicrofacetGeometric microfacetGeometric,
        const Iridescence* const iridescence) :
    BsdfComponent(iridescence),
    mN(N),
    mEta(scene_rdl2::math::Color(eta)),
    mK(scene_rdl2::math::sBlack),
    mRoughness(roughness),
    mMicrofacetDistribution(microfacetDistribution),
    mMicrofacetGeometric(microfacetGeometric),
    mIsConductor(false)
{
    // Average Dielectric Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    float favg = shading::averageFresnelReflectance(eta);
    mFavg = scene_rdl2::math::sWhite * favg;
}

MicrofacetIsotropicBRDF::MicrofacetIsotropicBRDF(
        const scene_rdl2::math::Vec3f& N,
        const scene_rdl2::math::Color& reflectivity,
        const scene_rdl2::math::Color& edgeTint,
        float roughness,
        ispc::MicrofacetDistribution microfacetDistribution,
        ispc::MicrofacetGeometric microfacetGeometric,
        const Iridescence* const iridescence):
    BsdfComponent(iridescence),
    mN(N),
    mRoughness(roughness),
    mMicrofacetDistribution(microfacetDistribution),
    mMicrofacetGeometric(microfacetGeometric),
    mIsConductor(true)
{
    auto ior = shading::ShaderComplexIor::createComplexIorFromColor(
            reflectivity, edgeTint);
    mEta = ior.getEta();
    mK = ior.getAbsorption();

    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    mFavg = shading::averageFresnelReflectance(reflectivity, edgeTint);
}

MicrofacetIsotropicBSDF::MicrofacetIsotropicBSDF(
        const scene_rdl2::math::Vec3f& N,
        float eta,
        float roughness,
        ispc::MicrofacetDistribution microfacetDistribution,
        ispc::MicrofacetGeometric microfacetGeometric,
        const scene_rdl2::math::Color& tint,
        float abbeNumber,
        float refractionEta,
        float reflectionWeight,
        float transmissionWeight,
        const Iridescence* const iridescence) :
    BsdfComponent(iridescence),
    mN(N),
    mEta(eta),
    mRoughness(roughness),
    mMicrofacetDistribution(microfacetDistribution),
    mMicrofacetGeometric(microfacetGeometric),
    mTint(tint),
    mAbbeNumber(abbeNumber),
    mRefractionEta(refractionEta),
    mReflectionWeight(reflectionWeight),
    mTransmissionWeight(transmissionWeight)
{}

MirrorBRDF::MirrorBRDF(
        const scene_rdl2::math::Vec3f& N,
        const scene_rdl2::math::Color& reflectivity,
        const scene_rdl2::math::Color& edgeTint,
        const Iridescence* const iridescence):
    BsdfComponent(iridescence),
    mN(N),
    mIsConductor(true)
{
    auto ior = shading::ShaderComplexIor::createComplexIorFromColor(
            reflectivity, edgeTint);
    mEta = ior.getEta();
    mK = ior.getAbsorption();
}

GlitterFlakeBRDF::GlitterFlakeBRDF(
        const scene_rdl2::math::Vec3f& N,
        const scene_rdl2::math::Vec3f& flakeN,
        const scene_rdl2::math::Color& reflectivity,
        const scene_rdl2::math::Color& edgeTint,
        const float roughness,
        const Iridescence* const iridescence):
    BsdfComponent(iridescence),
    mN(N),
    mFlakeN(flakeN),
    mRoughness(roughness)
{
    auto ior = shading::ShaderComplexIor::createComplexIorFromColor(
            reflectivity, edgeTint);
    mEta = ior.getEta();
    mK = ior.getAbsorption();

    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    mFavg = shading::averageFresnelReflectance(reflectivity, edgeTint);
}

DipoleDiffusion::DipoleDiffusion(
        const scene_rdl2::math::Vec3f &N,
        const scene_rdl2::math::Color &albedo,
        const scene_rdl2::math::Color &radius,
        const scene_rdl2::rdl2::Material* const material,
        const scene_rdl2::rdl2::TraceSet* const traceSet,
        const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn) :
    BsdfComponent(nullptr),
    mN(N),
    mAlbedo(albedo),
    mRadius(radius),
    mMaterial(material),
    mTraceSet(traceSet),
    mEvalNormalFn(evalNormalFn) {}

NormalizedDiffusion::NormalizedDiffusion(
        const scene_rdl2::math::Vec3f &N,
        const scene_rdl2::math::Color &albedo,
        const scene_rdl2::math::Color &radius,
        const scene_rdl2::rdl2::Material* const material,
        const scene_rdl2::rdl2::TraceSet* const traceSet,
        const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn) :
    BsdfComponent(nullptr),
    mN(N),
    mAlbedo(albedo),
    mRadius(radius),
    mMaterial(material),
    mTraceSet(traceSet),
    mEvalNormalFn(evalNormalFn) {}

RandomWalkSubsurface::RandomWalkSubsurface(
        const scene_rdl2::math::Vec3f &N,
        const scene_rdl2::math::Color &albedo,
        const scene_rdl2::math::Color &radius,
        const bool resolveSelfIntersections,
        const scene_rdl2::rdl2::Material* const material,
        const scene_rdl2::rdl2::TraceSet* const traceSet,
        const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn) :
    BsdfComponent(nullptr),
    mN(N),
    mAlbedo(albedo),
    mRadius(radius),
    mResolveSelfIntersections(resolveSelfIntersections),
    mMaterial(material),
    mTraceSet(traceSet),
    mEvalNormalFn(evalNormalFn)
{
}


} // end namespace shading
} // end namespace moonray

