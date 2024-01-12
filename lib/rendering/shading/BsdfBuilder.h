// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/shading/ispc/BsdfBuilder_ispc_stubs.h>

#include <scene_rdl2/common/platform/IspcUtil.h>
#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace shading {

class Bsdf;
class State;
class TLState;

// shading components
class MicrofacetAnisotropicClearcoat;
class MicrofacetIsotropicClearcoat;
class MirrorClearcoat;
class MirrorBRDF;
class MirrorBTDF;
class MirrorBSDF;
class MicrofacetAnisotropicBRDF;
class MicrofacetIsotropicBRDF;
class MicrofacetAnisotropicBTDF;
class MicrofacetIsotropicBTDF;
class MicrofacetAnisotropicBSDF;
class MicrofacetIsotropicBSDF;
class LambertianBRDF;
class LambertianBTDF;
class FlatDiffuseBRDF;
class OrenNayarBRDF;
class DipoleDiffusion;
class NormalizedDiffusion;
class RandomWalkSubsurface;
class FabricBRDF;
class VelvetBRDF;
class EyeCausticBRDF;
class HairDiffuseBSDF;
class HairBSDF;
class HairRBRDF;
class HairTRTBRDF;
class HairTTBTDF;
class HairTRRTBRDF;
class GlitterFlakeBRDF;
class StochasticFlakesBRDF;
class ToonBRDF;
class ToonSpecularBRDF;
class HairToonSpecularBRDF;

class BsdfBuilder
{
public:

    BsdfBuilder(
            Bsdf& bsdf,
            shading::TLState *tls,
            const State& state);

    ~BsdfBuilder();

    void addMicrofacetAnisotropicClearcoat(
            const MicrofacetAnisotropicClearcoat& clearcoat,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMicrofacetIsotropicClearcoat(
            const MicrofacetIsotropicClearcoat& clearcoat,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMirrorClearcoat(
            const MirrorClearcoat& clearcoat,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMirrorBRDF(
            const MirrorBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMirrorBTDF(
            const MirrorBTDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMirrorBSDF(
            const MirrorBSDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int reflectionLabel,
            int transmissionLabel);

    void addMicrofacetAnisotropicBRDF(
            const MicrofacetAnisotropicBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMicrofacetIsotropicBRDF(
            const MicrofacetIsotropicBRDF& brdf,
            const float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            const int label);

    void addMicrofacetAnisotropicBTDF(
            const MicrofacetAnisotropicBTDF& btdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMicrofacetIsotropicBTDF(
            const MicrofacetIsotropicBTDF& btdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addMicrofacetAnisotropicBSDF(
            const MicrofacetAnisotropicBSDF& bsdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int reflectionLabel,
            int transmissionLabel);

    void addMicrofacetIsotropicBSDF(
            const MicrofacetIsotropicBSDF& bsdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int reflectionLabel,
            int transmissionLabel);

    void addLambertianBRDF(
            const LambertianBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addLambertianBTDF(
            const LambertianBTDF& btdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addFlatDiffuseBRDF(
            const FlatDiffuseBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addOrenNayarBRDF(
            const OrenNayarBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addDipoleDiffusion(
            const DipoleDiffusion& bssrdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addNormalizedDiffusion(
            const NormalizedDiffusion& bssrdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addRandomWalkSubsurface(
            const RandomWalkSubsurface& sss,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addFabricBRDF(
            const FabricBRDF& bssrdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addVelvetBRDF(
            const VelvetBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addEyeCausticBRDF(
            const EyeCausticBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addHairDiffuseBSDF(
            const HairDiffuseBSDF& bsdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addHairBSDF(
            const HairBSDF& bsdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addHairRBRDF(
            const HairRBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addHairTRTBRDF(
            const HairTRTBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addHairTTBTDF(
            const HairTTBTDF& btdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addHairTRRTBRDF(
            const HairTRRTBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addGlitterFlakeBRDF(
            const GlitterFlakeBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addStochasticFlakesBRDF(
            const StochasticFlakesBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addToonBRDF(
            const ToonBRDF& brdf,
            float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            int label);

    void addToonSpecularBRDF(
            const ToonSpecularBRDF& brdf,
            const float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            const int label);

    void addHairToonSpecularBRDF(
            const HairToonSpecularBRDF& brdf,
            const float weight,
            ispc::BsdfBuilderBehavior combineBehavior,
            const int label);

    // Tells the BsdfBuilder than the next few lobes to be added should be
    // considered "adjacent" to eachother in terms of energy distribution.
    // In other words, any lobes that are added while the BsdfBuilder is
    // in "adjacent mode" will not affect eachother.  They will however
    // still be attenuated by any previously added  lobes and they will
    // attenuate any lobes that are added after endAdjacentComponents()
    // is called.
    void startAdjacentComponents();
    void endAdjacentComponents();

    // Specifies a radiance value to be emitted from the surface.
    // This effectively turns the object into a light source.
    void addEmission(const scene_rdl2::math::Color& emission);

    // Set early termination - any primary ray paths are immediately
    // terminated on intersection.
    // This is used in the "CutoutMaterial", for example.
    void setEarlyTermination();

    // Set Thin Geometry
    void setThinGeo();

    // Signal to the integrator that no light culling should be performed on
    // any of the diffuse lobes.  This allows a material to use completely arbitrary
    // shading normals that may diverge significantly from the geometric normal.
    // This is useful for certain NPR effects
    void setPreventLightCulling(bool isPrevented);

    // Temporary function to allow certain legacy materials direct
    // access to the Bsdf. This is required for now because we
    // do not support Schlick Fresnel through our shading API. This function
    // should be considered deprecated and removed as part of MOONSHINE-999.
    [[deprecated]]
    const Bsdf* getBsdf() const;

    BsdfBuilder(const BsdfBuilder& other) =delete;
    BsdfBuilder& operator=(const BsdfBuilder& other) =delete;

private:
    class Impl;

    Impl* mImpl;
};

// ispc vector types
ISPC_UTIL_TYPEDEF_STRUCT(BsdfBuilder, BsdfBuilderv);

} // end namespace shading
} // end namespace moonray

