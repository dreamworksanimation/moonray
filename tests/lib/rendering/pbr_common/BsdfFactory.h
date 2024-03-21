// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfFactory.h
/// $Id$
///

#pragma once

#include "BsdfFactory_ispc_stubs.h"

#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/bsdf/ashikhmin_shirley/BsdfAshikhminShirley.h>
#include <moonray/rendering/shading/bsdf/cook_torrance/BsdfCookTorrance.h>
#include <moonray/rendering/shading/bsdf/BsdfIridescence.h>
#include <moonray/rendering/shading/bsdf/hair/BsdfHairDiffuse.h>
#include <moonray/rendering/shading/bsdf/hair/BsdfHairLobes.h>
#include <moonray/rendering/shading/bsdf/fabric/BsdfFabric.h>
#include <moonray/rendering/shading/bsdf/BsdfLambert.h>
#include <moonray/rendering/shading/bsdf/BsdfRetroreflection.h>
#include <moonray/rendering/shading/bsdf/BsdfEyeCaustic.h>
#include <moonray/rendering/shading/bsdf/under/BsdfUnder.h>
#include <moonray/rendering/shading/bsdf/under/BsdfUnderClearcoat.h>
#include <moonray/rendering/shading/bsdf/ward/BsdfWard.h>
#include <moonray/rendering/shading/bsdf/BsdfStochasticFlakes.h>
#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

class BsdfFactory {
public:
    virtual ~BsdfFactory()  {}
    virtual shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena,
                                      const scene_rdl2::math::ReferenceFrame &frame) const = 0;
    virtual shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena &arena,
                                     const scene_rdl2::math::ReferenceFrame &frame) const = 0;
protected:
    ispc::Arena *ispcArena(scene_rdl2::alloc::Arena &arena) const
    { return (ispc::Arena *) &arena; }
    const ispc::ReferenceFrame *ispcFrame(const scene_rdl2::math::ReferenceFrame &frame) const
    { return (const ispc::ReferenceFrame *) &frame; }
};


//----------------------------------------------------------------------------

class LambertBsdfFactory : public BsdfFactory {
public:
    LambertBsdfFactory()  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::LambertBsdfLobe>(frame.getN(), scene_rdl2::math::sWhite, true);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::LambertBsdfFactory(ispcArena(arena), ispcFrame(frame));

        return bsdfv;
    }
};


//----------------------------------------------------------------------------

class CookTorranceBsdfFactory : public BsdfFactory {
public:
    CookTorranceBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::CookTorranceBsdfLobe>(
                frame.getN(), mRoughness);
        lobe->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::CookTorranceBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);

        return bsdfv;
    }
private:
    float mRoughness;
};


class GGXCookTorranceBsdfFactory : public BsdfFactory {
public:
    GGXCookTorranceBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::GGXCookTorranceBsdfLobe>(
                frame.getN(), mRoughness);
        lobe->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::GGXCookTorranceBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);

        return bsdfv;
    }
private:
    float mRoughness;
};


class AnisoCookTorranceBsdfFactory : public BsdfFactory {
public:
    AnisoCookTorranceBsdfFactory(float roughnessU, float roughnessV) :
        mRoughnessU(roughnessU), mRoughnessV(roughnessV)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::AnisoCookTorranceBsdfLobe>(
                frame.getN(), frame.getX(), mRoughnessU, mRoughnessV);
        lobe->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::AnisoCookTorranceBsdfFactory(ispcArena(arena), ispcFrame(frame),
                                                                   mRoughnessU, mRoughnessV);

        return bsdfv;
    }
private:
    float mRoughnessU;
    float mRoughnessV;
};

// Testing IridescenceBsdfLobe's tinting only with CookTorrance child lobe here
class IridescenceBsdfFactory : public BsdfFactory {
public:
    IridescenceBsdfFactory(float roughness, float iridescence, ispc::SHADING_IridescenceColorMode colorControl,
                           const scene_rdl2::math::Color& primary, const scene_rdl2::math::Color& secondary, bool flipHue,
                           ispc::ColorRampControlSpace rampInterpolationMode, int numRampPoints, const float* positions,
                           const ispc::RampInterpolatorMode* interpolators, const scene_rdl2::math::Color* colors,
                           float thickness, float exponent, float iridescenceAt0, float iridescenceAt90) :
        mRoughness(roughness), mIridescence(iridescence), mColorControl(colorControl),
        mPrimary(primary), mSecondary(secondary), mFlipHue(flipHue),
        mRampInterpolationMode(rampInterpolationMode), mRampNumPoints(numRampPoints),
        mRampPositions(positions), mRampInterpolators(interpolators), mRampColors(colors),
        mThickness(thickness), mExponent(exponent),
        mIridescenceAt0(iridescenceAt0), mIridescenceAt90(iridescenceAt90) {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe* child = arena.allocWithArgs<shading::CookTorranceBsdfLobe>(frame.getN(), mRoughness);
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::IridescenceBsdfLobe>(
                child, frame.getN(), mIridescence, mColorControl, mPrimary, mSecondary, mFlipHue,
                mRampInterpolationMode, mRampNumPoints, mRampPositions, mRampInterpolators, mRampColors,
                mThickness, mExponent, mIridescenceAt0, mIridescenceAt90);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::IridescenceBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness, mIridescence,
                mColorControl, asIspc(mPrimary), asIspc(mSecondary), mFlipHue,
                mRampInterpolationMode, mRampNumPoints, mRampPositions,
                mRampInterpolators, asIspc(mRampColors),
                mThickness, mExponent, mIridescenceAt0, mIridescenceAt90);

        return bsdfv;
    }
private:
    float mRoughness;
    float mIridescence;
    ispc::SHADING_IridescenceColorMode mColorControl;
    scene_rdl2::math::Color mPrimary;
    scene_rdl2::math::Color mSecondary;
    bool mFlipHue;
    ispc::ColorRampControlSpace mRampInterpolationMode;
    size_t mRampNumPoints;
    const float* mRampPositions;
    const ispc::RampInterpolatorMode* mRampInterpolators;
    const scene_rdl2::math::Color* mRampColors;
    float mThickness;
    float mExponent;
    float mIridescenceAt0;
    float mIridescenceAt90;
};


class TransmissionCookTorranceBsdfFactory : public BsdfFactory {
public:
    TransmissionCookTorranceBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        // Energy Compensation Params
        float eta = 1.5f;
        float favg, favgInv;
        shading::averageFresnelReflectance(eta,
                                           favg, favgInv);
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::TransmissionCookTorranceBsdfLobe>(
                frame.getN(), mRoughness, 1.0f, eta, scene_rdl2::math::sWhite, favg, favgInv);
        lobe->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::TransmissionCookTorranceBsdfFactory(
                ispcArena(arena), ispcFrame(frame), mRoughness, 1.0f, 1.5f);

        return bsdfv;
    }
private:
    float mRoughness;
};


//----------------------------------------------------------------------------

class RetroreflectionBsdfFactory : public BsdfFactory {
public:
    RetroreflectionBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::RetroreflectionBsdfLobe>(
                frame.getN(), mRoughness);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::RetroreflectionBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);
        return bsdfv;
    }
private:
    float mRoughness;
};

//----------------------------------------------------------------------------

class EyeCausticBsdfFactory : public BsdfFactory {
public:
    EyeCausticBsdfFactory(float roughness)
    {
        //convert roughness to phong exponent
        mExponent = 2.0f * scene_rdl2::math::rcp(roughness * roughness);
    }
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe =
            arena.allocWithArgs<shading::EyeCausticBsdfLobe>(frame.getN(),
                                                             frame.getN(),
                                                             scene_rdl2::math::sWhite,
                                                             mExponent);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::EyeCausticBsdfFactory(ispcArena(arena), ispcFrame(frame), mExponent);
        return bsdfv;
    }
private:
    float mExponent;
};


//----------------------------------------------------------------------------

class DwaFabricBsdfFactory : public BsdfFactory {
public:
    DwaFabricBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::DwaFabricBsdfLobe>(frame.getN(),
                                                                frame.getT(),
                                                                scene_rdl2::math::Vec3f(1,1,0),    // thread direction
                                                                0.0,                   // thread rotation
                                                                mRoughness);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::FabricBsdfFactory(ispcArena(arena),
                                                        ispcFrame(frame),
                                                        mRoughness,
                                                        true);
        return bsdfv;
    }
private:
    float mRoughness;
};

//----------------------------------------------------------------------------

class KajiyaKayFabricBsdfFactory : public BsdfFactory {
public:
    KajiyaKayFabricBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::KajiyaKayFabricBsdfLobe>(frame.getN(),
                                                                      frame.getT(),
                                                                      scene_rdl2::math::Vec3f(1,1,0),    // thread direction
                                                                      0.0,                   // thread rotation
                                                                      mRoughness);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::FabricBsdfFactory(ispcArena(arena),
                                                        ispcFrame(frame),
                                                        mRoughness,
                                                        false);
        return bsdfv;
    }
private:
    float mRoughness;
};

//----------------------------------------------------------------------------

class AshikminhShirleyBsdfFactory : public BsdfFactory {
public:
    AshikminhShirleyBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *asGlossy = arena.allocWithArgs<shading::AshikhminShirleyGlossyBsdfLobe>(
                frame.getN(), frame.getX(), mRoughness, mRoughness);
        asGlossy->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(asGlossy);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::AshikminhShirleyBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);

        return bsdfv;
    }
private:
    float mRoughness;
};


class AshikminhShirleyFullBsdfFactory : public BsdfFactory {
public:
    AshikminhShirleyFullBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *asDiffuse = arena.allocWithArgs<shading::AshikhminShirleyDiffuseBsdfLobe>(
                frame.getN());
        shading::BsdfLobe *asGlossy = arena.allocWithArgs<shading::AshikhminShirleyGlossyBsdfLobe>(
                frame.getN(), frame.getX(), mRoughness, mRoughness);
        asDiffuse->setScale(scene_rdl2::math::Color(0.5f, 0.5f, 0.5f));
        asGlossy->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(0.5f, 0.5f, 0.5f), 1.0f));
        bsdf->addLobe(asDiffuse);
        bsdf->addLobe(asGlossy);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::AshikminhShirleyFullBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);
        return bsdfv;
    }
private:
    float mRoughness;
};


class WardCorrectedBsdfFactory : public BsdfFactory {
public:
    WardCorrectedBsdfFactory(float roughness) : mRoughness(roughness) {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::WardCorrectedBsdfLobe>(
                frame.getN(), frame.getX(), mRoughness, mRoughness, true);
        lobe->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::WardCorrectedBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);

        return bsdfv;
    }
private:
    float mRoughness;
};


class WardDuerBsdfFactory : public BsdfFactory {
public:
    WardDuerBsdfFactory(float roughness) : mRoughness(roughness)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::WardDuerBsdfLobe>(
                frame.getN(), frame.getX(), mRoughness, mRoughness, false);
        lobe->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(1.0f, 1.0f, 1.0f), 1.0f));
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::WardDuerBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);

        return bsdfv;
    }
private:
    float mRoughness;
};


//----------------------------------------------------------------------------

class HairDiffuseBsdfFactory : public BsdfFactory {
public:
    HairDiffuseBsdfFactory()  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithArgs<shading::Bsdf>();
        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::HairDiffuseLobe>(frame.getT(),
                scene_rdl2::math::sWhite, scene_rdl2::math::sWhite);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::HairDiffuseBsdfFactory(ispcArena(arena), ispcFrame(frame));

        return bsdfv;
    }
};

class HairRBsdfFactory : public BsdfFactory {
public:
    HairRBsdfFactory(float roughness,
                     float offset) :
            mRoughness(roughness),
            mOffset(offset)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();

        float shift = scene_rdl2::math::deg2rad(mOffset);
        const scene_rdl2::math::Vec2f uvs(0.25f, 0.1f);
        const float ior = 1.55f;
        auto lobe = arena.allocWithArgs<shading::HairRLobe>(frame.getT(),
                uvs,
                1.0f, //mediumIOR
                ior,
                ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                0.0f, //cuticle layer thickness
                shift, mRoughness);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv = ispc::HairRBsdfFactory(ispcArena(arena),
                                              ispcFrame(frame),
                                              mRoughness,
                                              mOffset);

        return bsdfv;
    }
private:
    const float mRoughness;
    const float mOffset;
};

class HairTTBsdfFactory : public BsdfFactory {
public:
    HairTTBsdfFactory(float roughness,
                      float aziRoughness,
                      float offset) :
            mRoughness(roughness),
            mAzimuthalRoughness(aziRoughness),
            mOffset(offset)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        float shift = scene_rdl2::math::deg2rad(mOffset);
        const scene_rdl2::math::Vec2f uvs(0.25f, 0.1f);
        const float ior = 1.55f;
        const scene_rdl2::math::Color hairColor = scene_rdl2::math::Color(0.65f, 0.54f, 0.1f);
        const scene_rdl2::math::Color hairSigmaA =
            shading::HairUtil::computeAbsorptionCoefficients(hairColor, mAzimuthalRoughness);

        auto lobe = arena.allocWithArgs<shading::HairTTLobe>(frame.getT(),
                uvs,
                1.0f, //mediumIOR
                ior,
                ispc::HAIR_FRESNEL_DIELECTRIC_CYLINDER,
                0.0f, //cuticle layer thickness
                shift, mRoughness, mAzimuthalRoughness, hairColor, hairSigmaA);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::HairTTBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness,
                                    mOffset, mAzimuthalRoughness);

        return bsdfv;
    }
private:
    const float mRoughness;
    const float mAzimuthalRoughness;
    const float mOffset;
};

//----------------------------------------------------------------------------

class TwoLobeBsdfFactory : public BsdfFactory {
public:
    TwoLobeBsdfFactory(float roughness) :
        mRoughness(roughness) {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();

        shading::BsdfLobe *lobe1 =
            arena.allocWithArgs<shading::LambertBsdfLobe>(frame.getN(), scene_rdl2::math::sWhite, true);

        shading::BsdfLobe *lobe2 =
            arena.allocWithArgs<shading::CookTorranceBsdfLobe>( frame.getN(), mRoughness);

        lobe1->setScale(scene_rdl2::math::Color(0.3f, 0.3f, 0.3f));
        lobe2->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(0.7f, 0.7f, 0.7f), 1.0f));
        bsdf->addLobe(lobe1);
        bsdf->addLobe(lobe2);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::TwoLobeBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness);

        return bsdfv;
    }
private:
    float mRoughness;
};


//----------------------------------------------------------------------------

class ThreeLobeBsdfFactory : public BsdfFactory {
public:
    ThreeLobeBsdfFactory(float roughness1, float roughness2) :
        mRoughness1(roughness1), mRoughness2(roughness2)  {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        // TODO: This bsdf is not energy preserving
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe1 = arena.allocWithArgs<shading::LambertBsdfLobe>(frame.getN(), scene_rdl2::math::sWhite, true);
        shading::BsdfLobe *lobe2 = arena.allocWithArgs<shading::CookTorranceBsdfLobe>(
                frame.getN(), mRoughness2);
        shading::BsdfLobe *lobe3 = arena.allocWithArgs<shading::CookTorranceBsdfLobe>(
                frame.getN(), mRoughness1);
        lobe1->setScale(scene_rdl2::math::Color(0.1f, 0.1f, 0.1f));
        lobe2->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(0.5f, 0.5f, 0.5f), 1.0f));
        lobe3->setFresnel(arena.allocWithArgs<shading::SchlickFresnel>(
                scene_rdl2::math::Color(0.3f, 0.3f, 0.3f), 1.0f));
        bsdf->addLobe(lobe1);
        bsdf->addLobe(lobe2);
        bsdf->addLobe(lobe3);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::ThreeLobeBsdfFactory(ispcArena(arena), ispcFrame(frame), mRoughness1, mRoughness2);
        return bsdfv;
    }
private:
    float mRoughness1;
    float mRoughness2;
};

//----------------------------------------------------------------------------

class StochasticFlakesBsdfFactory : public BsdfFactory {
public:
    StochasticFlakesBsdfFactory(const scene_rdl2::math::Vec3f* normals, const scene_rdl2::math::Color* colors, const size_t flakeCount,
                                float roughness, float inputFlakeRandomness) :
            mNormals(normals), mColors(colors), mFlakeCount(flakeCount), mRoughness(roughness),
            mFlakeRandomness(inputFlakeRandomness) {}
    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame& frame) const
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe *lobe =
            arena.allocWithArgs<shading::StochasticFlakesBsdfLobe>(frame.getN(), mNormals, mColors, mFlakeCount,
                                                                   mRoughness, mFlakeRandomness);
        bsdf->addLobe(lobe);
        return bsdf;
    }
    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        shading::Bsdfv *bsdfv =
            ispc::StochasticFlakesBsdfFactory(ispcArena(arena), ispcFrame(frame),
                                              scene_rdl2::math::asIspc(mNormals), scene_rdl2::math::asIspc(mColors),
                                              mFlakeCount, mRoughness, mFlakeRandomness);
        return bsdfv;
    }

private:
    const scene_rdl2::math::Vec3f* mNormals;
    const scene_rdl2::math::Color* mColors;
    size_t mFlakeCount;
    float mRoughness;
    float mFlakeRandomness;
};


//----------------------------------------------------------------------------
class UnderClearcoatBsdfFactory : public BsdfFactory {
public:
    enum class UnderlobeType {
        Lambert,
        CookTorrance,
    };

    UnderClearcoatBsdfFactory(
            const scene_rdl2::math::Vec3f &N,
            const float etaI,
            const float etaT,
            const float thickness,
            const scene_rdl2::math::Color &attenuationColor,
            const float attenuationWeight,
            const UnderlobeType underlobeType) :
        mN(N),
        mRoughness(0.5f),
        mEtaI(etaI),
        mEtaT(etaT),
        mThickness(thickness),
        mAttenuationColor(attenuationColor),
        mAttenuationWeight(attenuationWeight),
        mClearcoatTransmissionLobe(nullptr),
        mUnderlobeType(underlobeType) {}

    shading::Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const override
    {
        shading::Bsdf *bsdf = arena.allocWithCtor<shading::Bsdf>();
        shading::BsdfLobe* under = nullptr;
        switch (mUnderlobeType)
        {
        case UnderlobeType::Lambert :
            under = arena.allocWithArgs<shading::LambertBsdfLobe>(frame.getN(), scene_rdl2::math::sWhite, true);
            break;
        case UnderlobeType::CookTorrance :
            under = arena.allocWithArgs<shading::CookTorranceBsdfLobe>(frame.getN(), mRoughness);
            break;
        }

        shading::BsdfLobe *lobe = arena.allocWithArgs<shading::UnderClearcoatBsdfLobe>(
                under,
                &arena,
                frame.getN(),
                mEtaI,
                mEtaT,
                mThickness,
                mAttenuationColor,
                mAttenuationWeight,
                false); // Do not use the assumption employed in UnderClearcoat to pass through TIR samples
        bsdf->addLobe(lobe);
        return bsdf;
    }

    shading::Bsdfv *getBsdfv(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const override
    {
        // FIXME: implement !
        return nullptr;
    }

    void setRoughness(const float roughness) { mRoughness = roughness; }

private:
    scene_rdl2::math::Vec3f mN;
    float mRoughness;
    float mEtaI;
    float mEtaT;
    float mThickness;
    scene_rdl2::math::Color mAttenuationColor;
    float mAttenuationWeight;
    shading::BsdfLobe* mClearcoatTransmissionLobe;
    UnderlobeType mUnderlobeType;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

