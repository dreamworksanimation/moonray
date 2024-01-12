// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfComponent.ispc

#include "BsdfComponent.isph"
#include "bsdf/Fresnel.isph"

#include <moonray/rendering/shading/ispc/Ior.isph>

// export types for use in c++
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(MicrofacetDistribution);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(MicrofacetGeometric);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(SubsurfaceType);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(HairFresnelType);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(SHADING_IridescenceColorMode);

void
MicrofacetAnisotropicClearcoat_init(
        varying MicrofacetAnisotropicClearcoat& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughnessU,
        const varying float roughnessV,
        const varying Vec3f& shadingTangent,
        const varying float thickness,
        const varying Color& attenuationColor,
        const uniform bool refracts,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * const uniform iridescence)
{
    me.mN = N;
    me.mEta = eta;
    me.mRoughnessU = roughnessU;
    me.mRoughnessV = roughnessV;
    me.mShadingTangent = shadingTangent;
    me.mThickness = thickness;
    me.mAttenuationColor = attenuationColor;
    me.mRefracts = refracts;
    me.mMicrofacetDistribution = microfacetDistribution;
    me.mMicrofacetGeometric = microfacetGeometric;
    me.mIridescence = iridescence;
}

void
MicrofacetIsotropicClearcoat_init(
        varying MicrofacetIsotropicClearcoat& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughness,
        const varying float thickness,
        const varying Color& attenuationColor,
        const uniform bool refracts,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * const uniform iridescence)
{
    me.mN = N;
    me.mEta = eta;
    me.mRoughness = roughness;
    me.mThickness = thickness;
    me.mAttenuationColor = attenuationColor;
    me.mRefracts = refracts;
    me.mMicrofacetDistribution = microfacetDistribution;
    me.mMicrofacetGeometric = microfacetGeometric;
    me.mIridescence = iridescence;
}

void
MirrorClearcoat_init(
        varying MirrorClearcoat& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float thickness,
        const varying Color& attenuationColor,
        const uniform bool refracts,
        const varying Iridescence * const uniform iridescence)
{
    me.mN = N;
    me.mEta = eta;
    me.mThickness = thickness;
    me.mAttenuationColor = attenuationColor;
    me.mRefracts = refracts;
    me.mIridescence = iridescence;
}

// dielectric initializer
void
MirrorBRDF_init(
        varying MirrorBRDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying Iridescence * const uniform iridescence)
{
    me.mN           = N;
    me.mEta         = Color_ctor(eta);
    me.mK           = sBlack;
    me.mIsConductor = false;
    me.mIridescence = iridescence;
}

// conductor initializer
void
MirrorBRDF_init(
        varying MirrorBRDF& me,
        const varying Color& eta,
        const varying Color& k,
        const varying Vec3f& N,
        const varying Iridescence * const uniform iridescence)
{
    me.mN           = N;
    me.mEta         = eta;
    me.mK           = k;
    me.mIsConductor = true;
    me.mIridescence = iridescence;
}

// artist-friendly conductor initializer
void
MirrorBRDF_init(
        varying MirrorBRDF& me,
        const varying Vec3f& N,
        const varying Color& reflectivity,
        const varying Color& edgeTint,
        const varying Iridescence * const uniform iridescence)
{
    varying ShaderComplexIor ior =
        ShaderComplexIor_createComplexIorFromColor(reflectivity, edgeTint);

    me.mN           = N;
    me.mEta         = ior.mEta;
    me.mK           = ior.mAbsorption;
    me.mIsConductor = true;
    me.mIridescence = iridescence;
}

void
MirrorBTDF_init(
        varying MirrorBTDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying Color& tint,
        const varying float abbeNumber)
{
    me.mN           = N;
    me.mEta         = eta;
    me.mTint        = tint;
    me.mAbbeNumber  = abbeNumber;
}

void
MirrorBSDF_init(
        varying MirrorBSDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying Color& tint,
        const varying float abbeNumber,
        const varying float refractionEta,
        const varying float reflectionWeight,
        const varying float transmissionWeight,
        const varying Iridescence * uniform iridescence)
{
    me.mN                   = N;
    me.mEta                 = eta;
    me.mTint                = tint;
    me.mAbbeNumber          = abbeNumber;
    me.mRefractionEta       = refractionEta;
    me.mReflectionWeight    = reflectionWeight;
    me.mTransmissionWeight  = transmissionWeight;
    me.mIridescence         = iridescence;
}

// dielectric initializer
void
MicrofacetAnisotropicBRDF_init(
        varying MicrofacetAnisotropicBRDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughnessU,
        const varying float roughnessV,
        const varying Vec3f& shadingTangent,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * uniform iridescence)
{
    me.mN                       = N;
    me.mEta                     = Color_ctor(eta);
    me.mK                       = sBlack;
    me.mRoughnessU              = roughnessU;
    me.mRoughnessV              = roughnessV;
    me.mShadingTangent          = shadingTangent;
    me.mIsConductor             = false;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    // TODO - support for anisotropy
    me.mFavg                    = sWhite;
}

// conductor initializer
void
MicrofacetAnisotropicBRDF_init(
        varying MicrofacetAnisotropicBRDF& me,
        const varying Color& eta,
        const varying Color& k,
        const varying Vec3f& N,
        const varying float roughnessU,
        const varying float roughnessV,
        const varying Vec3f& shadingTangent,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * uniform iridescence)
{
    me.mN                       = N;
    me.mEta                     = eta;
    me.mK                       = k;
    me.mRoughnessU              = roughnessU;
    me.mRoughnessV              = roughnessV;
    me.mShadingTangent          = shadingTangent;
    me.mIsConductor             = true;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
    // TODO: what's the average reflectance here?
    // This code path is currently not used in our materials,
    // but when it is, we'll need a solution for this value.
    me.mFavg = sWhite;
}

// artist-friendly conductor initializer
void
MicrofacetAnisotropicBRDF_init(
        varying MicrofacetAnisotropicBRDF& me,
        const varying Vec3f& N,
        const varying Color& reflectivity,
        const varying Color& edgeTint,
        const varying float roughnessU,
        const varying float roughnessV,
        const varying Vec3f& shadingTangent,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * uniform iridescence)
{
    varying ShaderComplexIor ior =
        ShaderComplexIor_createComplexIorFromColor(reflectivity, edgeTint);

    me.mN                       = N;
    me.mEta                     = ior.mEta;
    me.mK                       = ior.mAbsorption;
    me.mRoughnessU              = roughnessU;
    me.mRoughnessV              = roughnessV;
    me.mShadingTangent          = shadingTangent;
    me.mIsConductor             = true;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    me.mFavg = averageFresnelReflectance(reflectivity, edgeTint);
}

// dielectric initializer
void
MicrofacetIsotropicBRDF_init(
        varying MicrofacetIsotropicBRDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughness,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * uniform iridescence)
{
    me.mN                       = N;
    me.mEta                     = Color_ctor(eta);
    me.mK                       = sBlack;
    me.mRoughness               = roughness;
    me.mIsConductor             = false;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
    // Average Dielectric Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    float favg = averageFresnelReflectance(eta);
    me.mFavg = sWhite * favg;
}

// conductor initializer
void
MicrofacetIsotropicBRDF_init(
        varying MicrofacetIsotropicBRDF& me,
        const varying Color& eta,
        const varying Color& k,
        const varying Vec3f& N,
        const varying float roughness,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * uniform iridescence)
{
    me.mN                       = N;
    me.mEta                     = eta;
    me.mK                       = k;
    me.mRoughness               = roughness;
    me.mIsConductor             = true;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
}

// artist-friendly conductor initializer
void
MicrofacetIsotropicBRDF_init(
        varying MicrofacetIsotropicBRDF& me,
        const varying Vec3f& N,
        const varying Color& reflectivity,
        const varying Color& edgeTint,
        const varying float roughness,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Iridescence * uniform iridescence)
{
    varying ShaderComplexIor ior =
        ShaderComplexIor_createComplexIorFromColor(reflectivity, edgeTint);

    me.mN                       = N;
    me.mEta                     = ior.mEta;
    me.mK                       = ior.mAbsorption;
    me.mRoughness               = roughness;
    me.mIsConductor             = true;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    me.mFavg = averageFresnelReflectance(reflectivity, edgeTint);

}

void
MicrofacetAnisotropicBTDF_init(
        varying MicrofacetAnisotropicBTDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughnessU,
        const varying float roughnessV,
        const varying Vec3f& shadingTangent,
        const varying Color& tint,
        const varying float abbeNumber,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric)
{
    me.mN                       = N;
    me.mEta                     = eta;
    me.mRoughnessU              = roughnessU;
    me.mRoughnessV              = roughnessV;
    me.mShadingTangent          = shadingTangent;
    me.mTint                    = tint;
    me.mAbbeNumber              = abbeNumber;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
}

void
MicrofacetIsotropicBTDF_init(
        varying MicrofacetIsotropicBTDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughness,
        const varying Color& tint,
        const varying float abbeNumber,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric)
{
    me.mN                       = N;
    me.mEta                     = eta;
    me.mRoughness               = roughness;
    me.mTint                    = tint;
    me.mAbbeNumber              = abbeNumber;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
}

void
MicrofacetAnisotropicBSDF_init(
        varying MicrofacetAnisotropicBSDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughnessU,
        const varying float roughnessV,
        const varying Vec3f& shadingTangent,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Color& tint,
        const varying float abbeNumber,
        const varying float refractionEta,
        const varying float reflectionWeight,
        const varying float transmissionWeight,
        const varying Iridescence * uniform iridescence)
{
    me.mN                       = N;
    me.mEta                     = eta;
    me.mRoughnessU              = roughnessU;
    me.mRoughnessV              = roughnessV;
    me.mShadingTangent          = shadingTangent;
    me.mTint                    = tint;
    me.mAbbeNumber              = abbeNumber;
    me.mRefractionEta           = refractionEta;
    me.mReflectionWeight        = reflectionWeight;
    me.mTransmissionWeight      = transmissionWeight;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
}

void
MicrofacetIsotropicBSDF_init(
        varying MicrofacetIsotropicBSDF& me,
        const varying Vec3f& N,
        const varying float eta,
        const varying float roughness,
        const uniform MicrofacetDistribution microfacetDistribution,
        const uniform MicrofacetGeometric microfacetGeometric,
        const varying Color& tint,
        const varying float abbeNumber,
        const varying float refractionEta,
        const varying float reflectionWeight,
        const varying float transmissionWeight,
        const varying Iridescence * const uniform iridescence)
{
    me.mN                       = N;
    me.mEta                     = eta;
    me.mRoughness               = roughness;
    me.mTint                    = tint;
    me.mAbbeNumber              = abbeNumber;
    me.mRefractionEta           = refractionEta;
    me.mReflectionWeight        = reflectionWeight;
    me.mTransmissionWeight      = transmissionWeight;
    me.mMicrofacetDistribution  = microfacetDistribution;
    me.mMicrofacetGeometric     = microfacetGeometric;
    me.mIridescence             = iridescence;
}

void
ToonBRDF_init(
        varying ToonBRDF& me,
        const varying Vec3f& N,
        const varying Color &albedo,
        const varying int numRampPoints,
        const varying float* uniform rampPositions,
        const varying RampInterpolatorMode* uniform rampInterpolators,
        const varying Color* uniform rampColors,
        const varying bool extendRamp,
        const varying Iridescence * const uniform iridescence)
{
    me.mN           = N;
    me.mAlbedo      = albedo;
    me.mRampNumPoints = numRampPoints;
    me.mRampPositions = rampPositions;
    me.mRampInterpolators = rampInterpolators;
    me.mRampColors = rampColors;
    me.mExtendRamp = extendRamp;

    me.mIridescence = iridescence;
}

void
HairToonSpecularBRDF_init(
        varying HairToonSpecularBRDF& me,
        const varying Vec3f& N,
        const varying Vec3f& hairDir,
        const varying Vec2f& hairUV,
        const varying float ior,
        const varying HairFresnelType fresnelType,
        const varying float layers,
        const varying float shiftAmount,
        const varying float roughness,
        const varying Color& tint,
        const varying float intensity,
        const varying int numRampPoints,
        const varying float* uniform rampPositions,
        const varying RampInterpolatorMode* uniform rampInterpolators,
        const varying float* uniform rampValues,
        const uniform bool enableIndirectReflections,
        const varying float indirectReflectionsIntensity,
        const varying float indirectReflectionsRoughness)
{
    me.mN = N;
    me.mHairDir = hairDir;
    me.mHairUV = hairUV;
    me.mIOR = ior;
    me.mFresnelType = fresnelType;
    me.mCuticleLayerThickness = layers;
    me.mShift = shiftAmount;
    me.mRoughness = roughness;
    me.mTint = tint;
    me.mIntensity = intensity;
    me.mRampNumPoints = numRampPoints;
    me.mRampPositions = rampPositions;
    me.mRampInterpolators = rampInterpolators;
    me.mRampValues = rampValues;
    me.mEnableIndirectReflections = enableIndirectReflections;
    me.mIndirectReflectionsIntensity = indirectReflectionsIntensity;
    me.mIndirectReflectionsRoughness = indirectReflectionsRoughness;
}

void
ToonSpecularBRDF_init(
        varying ToonSpecularBRDF& me,
        const varying Vec3f& N,
        const varying float intensity,
        const varying Color& tint,
        const float rampInputScale,
        const varying int numRampPoints,
        const varying float* uniform rampPositions,
        const varying RampInterpolatorMode* uniform rampInterpolators,
        const varying float* uniform rampValues,
        const varying float stretchU,
        const varying float stretchV,
        const varying Vec3f& dPds,
        const varying Vec3f& dPdt,
        const uniform bool enableIndirectReflections,
        const varying float indirectReflectionsIntensity,
        const varying float indirectReflectionsRoughness)
{
    me.mN = N;
    me.mIntensity = intensity;
    me.mTint = tint;
    me.mRampInputScale = rampInputScale,
    me.mRampNumPoints = numRampPoints;
    me.mRampPositions = rampPositions;
    me.mRampInterpolators = rampInterpolators;
    me.mRampValues = rampValues;
    me.mStretchU = stretchU;
    me.mStretchV = stretchV;
    me.mdPds = dPds;
    me.mdPdt = dPdt;
    me.mEnableIndirectReflections = enableIndirectReflections;
    me.mIndirectReflectionsIntensity = indirectReflectionsIntensity;
    me.mIndirectReflectionsRoughness = indirectReflectionsRoughness;
}

void
LambertianBRDF_init(
        varying LambertianBRDF& me,
        const varying Vec3f& N,
        const varying Color& albedo,
        const varying Iridescence * const uniform iridescence)
{
    me.mN           = N;
    me.mAlbedo      = albedo;
    me.mIridescence = iridescence;
}

void
LambertianBTDF_init(
        varying LambertianBTDF& me,
        const varying Vec3f& N,
        const varying Color& tint)
{
    me.mN           = N;
    me.mTint        = tint;
}

void
FlatDiffuseBRDF_init(
        varying FlatDiffuseBRDF& me,
        const varying Vec3f& N,
        const varying Color& albedo,
        const varying float roughness,
        const varying float terminatorShift,
        const varying float flatness,
        const varying float flatnessFalloff,
        const varying Iridescence * const uniform iridescence)
{
    me.mN               = N;
    me.mAlbedo          = albedo;
    me.mRoughness       = roughness;
    me.mTerminatorShift = terminatorShift;
    me.mFlatness        = flatness;
    me.mFlatnessFalloff = flatnessFalloff;
}

void
OrenNayarBRDF_init(
        varying OrenNayarBRDF& me,
        const varying Vec3f& N,
        const varying Color& albedo,
        const varying float roughness,
        const varying Iridescence * const uniform iridescence)
{
    me.mN           = N;
    me.mAlbedo      = albedo;
    me.mRoughness   = roughness;
    me.mIridescence = iridescence;
}

void
DipoleDiffusion_init(
        varying DipoleDiffusion& me,
        const varying Vec3f& N,
        const varying Color& albedo,
        const varying Color& radius,
        const uniform Material * uniform material,
        const uniform TraceSet * uniform traceSet,
        const uniform intptr_t evalNormalFn)
{
    me.mN               = N;
    me.mAlbedo          = albedo;
    me.mRadius          = radius;
    me.mMaterial        = material;
    me.mTraceSet        = traceSet;
    me.mEvalNormalFn    = evalNormalFn;
}

void
NormalizedDiffusion_init(
        varying NormalizedDiffusion& me,
        const varying Vec3f& N,
        const varying Color& albedo,
        const varying Color& radius,
        const uniform Material * uniform material,
        const uniform TraceSet * uniform traceSet,
        const uniform intptr_t evalNormalFn)
{
    me.mN               = N;
    me.mAlbedo          = albedo;
    me.mRadius          = radius;
    me.mMaterial        = material;
    me.mTraceSet        = traceSet;
    me.mEvalNormalFn    = evalNormalFn;
}

void
RandomWalkSubsurface_init(
        varying RandomWalkSubsurface& me,
        const varying Vec3f& N,
        const varying Color& albedo,
        const varying Color& radius,
        const uniform bool resolveSelfIntersections,
        const uniform Material * uniform material,
        const uniform TraceSet * uniform traceSet,
        const uniform intptr_t evalNormalFn)
{
    me.mN                           = N;
    me.mAlbedo                      = albedo;
    me.mRadius                      = radius;
    me.mResolveSelfIntersections    = resolveSelfIntersections,
    me.mMaterial                    = material;
    me.mTraceSet                    = traceSet;
    me.mEvalNormalFn                = evalNormalFn;
}

void
FabricBRDF_init(
        varying FabricBRDF& me,
        const varying Vec3f& N,
        const varying Vec3f& shadingTangent,
        const varying Vec3f& threadDirection,
        const varying float threadElevation,
        const varying float roughness,
        const varying Color& threadColor,
        const varying Iridescence * uniform iridescence)
{
    me.mN               = N;
    me.mShadingTangent  = shadingTangent;
    me.mRoughness       = roughness;
    me.mThreadDirection = threadDirection;
    me.mThreadElevation = threadElevation;
    me.mThreadColor     = threadColor;
    me.mIridescence     = iridescence;
}

void
VelvetBRDF_init(
        varying VelvetBRDF& me,
        const varying Vec3f& N,
        const varying float roughness,
        const varying Color& color,
        const varying bool useAbsorbingFibers,
        const varying Iridescence * uniform iridescence)
{
    me.mN                   = N;
    me.mRoughness           = roughness;
    me.mColor               = color;
    me.mUseAbsorbingFibers  = useAbsorbingFibers;
    me.mIridescence         = iridescence;
}

void
EyeCausticBRDF_init(
        varying EyeCausticBRDF& me,
        const varying Vec3f& N,
        const varying Vec3f& irisNormal,
        const varying Color& causticColor,
        const varying float exponent)
{
    me.mN               = N;
    me.mIrisNormal      = irisNormal;
    me.mCausticColor    = causticColor;
    me.mExponent        = exponent;
}

void
HairDiffuseBSDF_init(
        varying HairDiffuseBSDF& me,
        const varying Vec3f& hairDirection,
        const varying Color& reflectionColor,
        const varying Color& transmissionColor)
{
    me.mHairDir             = hairDirection;
    me.mReflectionColor     = reflectionColor;
    me.mTransmissionColor   = transmissionColor;
}

void
HairBSDF_init(
        varying HairBSDF& me,
        const varying Vec3f& hairDirection,
        const varying Vec2f& hairUV,
        const varying float ior,
        const varying HairFresnelType fresnelType,
        const varying float layers,
        const varying bool showR,
        const varying float longShiftR,
        const varying float roughnessR,
        const varying Color& tintR,
        const varying bool showTT,
        const varying float longShiftTT,
        const varying float roughnessTT,
        const varying float aziRoughnessTT,
        const varying Color& tintTT,
        const varying float saturationTT,
        const varying bool showTRT,
        const varying float longShiftTRT,
        const varying float roughnessTRT,
        const varying Color& tintTRT,
        const varying bool showGlint,
        const varying float roughnessGlint,
        const varying float eccentricityGlint,
        const varying float saturationGlint,
        const varying float hairRotation,
        const varying Vec3f& hairNormal,
        const varying bool showTRRT,
        const varying Color& hairColor)
{
    me.mHairDir            = hairDirection;
    me.mHairUV             = hairUV;
    me.mIOR                = ior;
    me.mFresnelType        = fresnelType;
    me.mCuticleLayerThickness = layers;
    me.mShowR              = showR;
    me.mShiftR             = longShiftR;
    me.mRoughnessR         = roughnessR;
    me.mTintR              = tintR;
    me.mShowTT             = showTT;
    me.mShiftTT            = longShiftTT;
    me.mRoughnessTT        = roughnessTT;
    me.mAziRoughnessTT     = aziRoughnessTT;
    me.mTintTT             = tintTT;
    me.mSaturationTT       = saturationTT;
    me.mShowTRT            = showTRT;
    me.mShiftTRT           = longShiftTRT;
    me.mRoughnessTRT       = roughnessTRT;
    me.mTintTRT            = tintTRT;
    me.mShowGlint          = showGlint;
    me.mRoughnessGlint     = roughnessGlint;
    me.mEccentricityGlint  = eccentricityGlint;
    me.mSaturationGlint    = saturationGlint;
    me.mHairRotation       = hairRotation;
    me.mHairNormal         = hairNormal;
    me.mShowTRRT           = showTRRT;
    me.mHairColor          = hairColor;
}

void
HairRBRDF_init(
        varying HairRBRDF& me,
        const varying Vec3f& hairDirection,
        const varying Vec2f& hairUV,
        const varying float ior,
        const varying HairFresnelType fresnelType,
        const varying float layers,
        const varying float longShift,
        const varying float roughness,
        const varying Color& tint)
{
    me.mHairDir            = hairDirection;
    me.mHairUV             = hairUV;
    me.mIOR                = ior;
    me.mFresnelType        = fresnelType;
    me.mCuticleLayerThickness = layers;
    me.mShift              = longShift;
    me.mRoughness          = roughness;
    me.mTint               = tint;
}

void
HairTRTBRDF_init(
        varying HairTRTBRDF& me,
        const varying Vec3f& hairDirection,
        const varying Vec2f& hairUV,
        const varying float ior,
        const varying HairFresnelType fresnelType,
        const varying float layers,
        const varying float longShift,
        const varying float roughness,
        const varying float aziRoughness,
        const varying Color& hairColor,
        const varying Color& tint,
        const varying bool showGlint,
        const varying float roughnessGlint,
        const varying float eccentricityGlint,
        const varying float saturationGlint,
        const varying float hairRotation,
        const varying Vec3f& hairNormal)
{
    me.mHairDir            = hairDirection;
    me.mHairUV             = hairUV;
    me.mIOR                = ior;
    me.mFresnelType        = fresnelType;
    me.mCuticleLayerThickness = layers;
    me.mShift              = longShift;
    me.mRoughness          = roughness;
    me.mAziRoughness       = aziRoughness;
    me.mHairColor          = hairColor;
    me.mTint               = tint;
    me.mShowGlint          = showGlint;
    me.mRoughnessGlint     = roughnessGlint;
    me.mEccentricityGlint  = eccentricityGlint;
    me.mSaturationGlint    = saturationGlint;
    me.mHairRotation       = hairRotation;
    me.mHairNormal         = hairNormal;
}

void
HairTTBTDF_init(
        varying HairTTBTDF& me,
        const varying Vec3f& hairDirection,
        const varying Vec2f& hairUV,
        const varying float ior,
        const varying HairFresnelType fresnelType,
        const varying float layers,
        const varying float longShift,
        const varying float roughness,
        const varying float aziRoughness,
        const varying Color& hairColor,
        const varying Color& tint,
        const varying float saturation)
{
    me.mHairDir            = hairDirection;
    me.mHairUV             = hairUV;
    me.mIOR                = ior;
    me.mFresnelType        = fresnelType;
    me.mCuticleLayerThickness = layers;
    me.mShift              = longShift;
    me.mRoughness          = roughness;
    me.mAziRoughness       = aziRoughness;
    me.mHairColor          = hairColor;
    me.mTint               = tint;
    me.mSaturation         = saturation;
}

void
HairTRRTBRDF_init(
        varying HairTRRTBRDF& me,
        const varying Vec3f& hairDirection,
        const varying Vec2f& hairUV,
        const varying float ior,
        const varying HairFresnelType fresnelType,
        const varying float layers,
        const varying float roughness,
        const varying float aziRoughness,
        const varying Color& hairColor,
        const varying Color& tint)
{
    me.mHairDir            = hairDirection;
    me.mHairUV             = hairUV;
    me.mIOR                = ior;
    me.mFresnelType        = fresnelType;
    me.mCuticleLayerThickness = layers;
    me.mRoughness          = roughness;
    me.mAziRoughness       = aziRoughness;
    me.mHairColor          = hairColor;
    me.mTint               = tint;
}

void
GlitterFlakeBRDF_init(
        varying GlitterFlakeBRDF& me,
        const varying Vec3f& N,
        const varying Vec3f& flakeN,
        const varying Color& reflectivity,
        const varying Color& edgeColor,
        const varying float roughness,
        const varying Iridescence * uniform iridescence)
{
    ShaderComplexIor ior =
        ShaderComplexIor_createComplexIorFromColor(reflectivity,
                                                   edgeColor);
    me.mN           = N;
    me.mFlakeN      = flakeN;
    me.mEta         = ior.mEta;
    me.mK           = ior.mAbsorption;
    me.mRoughness   = roughness;
    me.mIridescence = iridescence;
    // Average Fresnel Reflectance
    // Ref:"Revisiting Physically Based Shading", Kulla'17
    me.mFavg = averageFresnelReflectance(reflectivity, edgeColor);
}

void
StochasticFlakesBRDF_init(
        varying StochasticFlakesBRDF& me,
        const varying Vec3f& N,
        const varying uint32_t flakeCount,
        const varying Vec3f * varying flakeNormals,
        const varying Color * varying flakeColors,
        const varying float flakeRoughness,
        const uniform float flakeRandomness,
        const varying Iridescence * uniform iridescence)
{
    me.mN               = N;
    me.mFlakeCount      = flakeCount;
    me.mFlakeNormals    = flakeNormals;
    me.mFlakeColors     = flakeColors;
    me.mFlakeRoughness  = flakeRoughness;
    me.mFlakeRandomness = flakeRandomness;
    me.mIridescence     = iridescence;
}

void
Iridescence_init(
        varying Iridescence& me,
        const varying Vec3f& N,
        const varying float strength,
        const varying Color& primary,
        const varying Color& secondary,
        const varying bool flipHue,
        const varying float thickness,
        const varying float exponent,
        const varying float strengthAt0,
        const varying float strengthAt90)
{
    me.mN                       = N;
    me.mStrength                = strength;
    me.mColorControl            = SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION;
    me.mPrimary                 = primary;
    me.mSecondary               = secondary;
    me.mFlipHue                 = flipHue;
    me.mRampNumPoints           = 0;
    me.mRampPositions           = nullptr;
    me.mRampInterpolators       = nullptr;
    me.mRampColors              = nullptr;
    me.mThickness               = thickness;
    me.mExponent                = exponent;
    me.mIridescenceAt0          = strengthAt0;
    me.mIridescenceAt90         = strengthAt90;
}

void
Iridescence_init(
        varying Iridescence& me,
        const varying Vec3f& N,
        const varying float strength,
        const varying ColorRampControlSpace rampInterpolationMode,
        const varying int numPoints,
        const varying float * uniform positions,
        const varying RampInterpolatorMode * uniform interpolators,
        const varying Color * uniform  colors,
        const varying float thickness,
        const varying float exponent,
        const varying float strengthAt0,
        const varying float strengthAt90)
{
    me.mN                       = N;
    me.mStrength                = strength;
    me.mColorControl            = SHADING_IRIDESCENCE_COLOR_USE_RAMP;
    me.mRampInterpolationMode   = rampInterpolationMode;
    me.mRampNumPoints           = numPoints;
    me.mRampPositions           = positions;
    me.mRampInterpolators       = interpolators;
    me.mRampColors              = colors;
    me.mThickness               = thickness;
    me.mExponent                = exponent;
    me.mIridescenceAt0          = strengthAt0;
    me.mIridescenceAt90         = strengthAt90;
}

