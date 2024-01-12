// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "BsdfSampler.h"
#include "LightSetSampler.h"

#include <moonray/rendering/pbr/core/RayState.h>
#include <moonray/rendering/pbr/core/Scene.h>
#include <moonray/rendering/pbr/sampler/SequenceID.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/Material.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

namespace moonray {

namespace shading { class BsdfLobe; class Intersection; }

namespace mcrt_common {
class RayDifferential;
}

namespace pbr {


//----------------------------------------------------------------------------

/// This converts a lobe type to the corresponding embree compatible ray mask.
/// Compute the proper ray mask value by shift left the lobe category bit by
/// the surface side category bit - 1 position (0 or 1 left shift). Note that
/// a lobe can be both transmission and reflection.
/// Look in scene_rdl2/scene/rdl2/VisibilityFlags.h for the definition of the
/// mask. Look in rendering/pbr/bsdf/Bsdf.h for the definition of lobe type.
finline int
lobeTypeToRayMask(const int lobeType)
{
    int mask = 0;
    if (lobeType & shading::BsdfLobe::REFLECTION) {
        mask |= (lobeType & shading::BsdfLobe::ALL_LOBES);
    }
    if (lobeType & shading::BsdfLobe::TRANSMISSION) {
        mask |= (lobeType & shading::BsdfLobe::ALL_LOBES) << 1;
    }

    return mask;
}

finline scene_rdl2::rdl2::RaySwitchContext::RayType
lobeTypeToRayType(const int lobeType)
{
    if (lobeType == 0) {
        return scene_rdl2::rdl2::RaySwitchContext::RayType::CameraRay;
    }
    if (lobeType & shading::BsdfLobe::Type::DIFFUSE) {
        return scene_rdl2::rdl2::RaySwitchContext::RayType::IndirectDiffuseRay;
    }
    if (lobeType & shading::BsdfLobe::Type::MIRROR) {
        return scene_rdl2::rdl2::RaySwitchContext::RayType::IndirectMirrorRay;
    }
    if (lobeType & shading::BsdfLobe::Type::GLOSSY) {
        return scene_rdl2::rdl2::RaySwitchContext::RayType::IndirectGlossyRay;
    }
    return scene_rdl2::rdl2::RaySwitchContext::RayType::OtherRay;
}

// Query the bsdf reflection lobes that their spawned rays can intersect with
// geometries using specified ray mask
finline int
rayMaskToReflectionLobes(const int rayMask)
{
    return (rayMask & shading::BsdfLobe::ALL_LOBES) |
        ((rayMask >> scene_rdl2::rdl2::sNumVisibilityTypes) & shading::BsdfLobe::ALL_LOBES) |
        shading::BsdfLobe::REFLECTION;
}

// Query the bsdf transmission lobes that their spawned rays can intersect with
// geometries using specified ray mask
finline int
rayMaskToTransmissionLobes(const int rayMask)
{
    return ((rayMask >> 1) & shading::BsdfLobe::ALL_LOBES) |
        ((rayMask >> (scene_rdl2::rdl2::sNumVisibilityTypes + 1)) & shading::BsdfLobe::ALL_LOBES) |
        shading::BsdfLobe::TRANSMISSION;
}

finline bool
checkForNanSimple(const scene_rdl2::math::Color &radiance,
        const char *contributionName, const Subpixel &sp)
{
    if (!scene_rdl2::math::isFinite(radiance)) {
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] " , contributionName ,
                " produced NaN values at pixel (",
                uint32ToPixelX(sp.mPixel) , ", " , uint32ToPixelY(sp.mPixel),
                "), subpixel index (", sp.mSubpixelIndex, ")");
        return true;
    }
    return false;
}


finline bool
checkForNan(scene_rdl2::math::Color &radiance, const char *contributionName,
        const Subpixel &sp, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
        const shading::Intersection &isect)
{
#ifdef DEBUG
    if (!scene_rdl2::math::isFinite(radiance)) {
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] ", contributionName,
                " produced NaN values");
        // TODO: print more info from sp and pv
        Scene::print(std::cerr, ray, isect, pv.diffuseDepth, pv.glossyDepth);
        std::cerr.flush();
        radiance = scene_rdl2::math::sBlack;
        return true;
    }
#endif
    return false;
}

//----------------------------------------------------------------------------

inline void
getPriorityList(const mcrt_common::RayDifferential &ray,
                const scene_rdl2::rdl2::Material* dstList[4],
                int dstListCount[4])
{
    const void* const* srcList = &ray.ext.priorityMaterial0;
    const int* srcListCount = &ray.ext.priorityMaterial0Count;
    for (int i = 0; i < 4; i++) {
        dstList[i] = static_cast<const scene_rdl2::rdl2::Material*>(srcList[i]);
        dstListCount[i] = srcListCount[i];
    }
}

inline void
setPriorityList(mcrt_common::RayDifferential &ray,
                const scene_rdl2::rdl2::Material* srcList[4],
                int srcListCount[4])
{
    const void** dstList = &ray.ext.priorityMaterial0;
    int* dstListCount = &ray.ext.priorityMaterial0Count;
    for (int i = 0; i < 4; i++) {
        dstList[i] = srcList[i];
        dstListCount[i] = srcListCount[i];
    }
}

inline void
addPriorityMaterial(const scene_rdl2::rdl2::Material* material,
                    const scene_rdl2::rdl2::Material* dstList[4],
                    int dstListCount[4])
{
    // First see if the material is already in the list.  If so, increase its
    // count.
    for (int i = 0; i < 4; i++) {
        if (dstList[i] == material) {
            dstListCount[i]++;
            return;
        }
    }

    // Else put it in the first empty list slot
    for (int i = 0; i < 4; i++) {
        if (dstList[i] == nullptr) {
            dstList[i] = material;
            dstListCount[i] = 1;
            return;
        }
    }
}

inline void
removePriorityMaterial(const scene_rdl2::rdl2::Material* material,
                       const scene_rdl2::rdl2::Material* dstList[4],
                       int dstListCount[4])
{
    for (int i = 0; i < 4; i++) {
        if (dstList[i] == material) {
            dstListCount[i]--;
            if (dstListCount[i] == 0) {
                dstList[i] = nullptr;
            }
            return;
        }
    }
}

inline int
getPriorityMaterialCount(const scene_rdl2::rdl2::Material* material,
                         const scene_rdl2::rdl2::Material* dstList[4],
                         int dstListCount[4])
{
    for (int i = 0; i < 4; i++) {
        if (dstList[i] == material) {
            return dstListCount[i];
        }
    }
    return 0;
}

inline const scene_rdl2::rdl2::Material*
getHighestPriorityMaterial(mcrt_common::RayDifferential &ray, int &highestPriority)
{
    const scene_rdl2::rdl2::Material *hpMat = nullptr;
    highestPriority = 0;
    const void** list = &ray.ext.priorityMaterial0;
    for (int i = 0; i < 4; i++) {
        if (list[i]) {
            const scene_rdl2::rdl2::Material *mat = static_cast<const scene_rdl2::rdl2::Material*>(list[i]);
            int priority = mat->priority();
            if (hpMat == nullptr || priority < highestPriority) {
                hpMat = mat;
                highestPriority = priority;
            }
        }
    }
    return hpMat;
}

inline const scene_rdl2::rdl2::Material*
getHighestPriorityMaterial(const scene_rdl2::rdl2::Material* list[4], int &highestPriority)
{
    const scene_rdl2::rdl2::Material *hpMat = nullptr;
    highestPriority = 0;
    for (int i = 0; i < 4; i++) {
        if (list[i]) {
            const scene_rdl2::rdl2::Material *mat = list[i];
            int priority = mat->priority();
            if (hpMat == nullptr || priority < highestPriority) {
                hpMat = mat;
                highestPriority = priority;
            }
        }
    }
    return hpMat;
}

// Updates the material priority list for the given ray and returns the mediumIor. See 
// "Simple Nested Dielectrics in Ray Traced Images".
float updateMaterialPriorities(mcrt_common::RayDifferential& ray, const Scene* scene, 
        const scene_rdl2::rdl2::Camera* camera, shading::TLState* shadingTls, const shading::Intersection& isect, 
        const scene_rdl2::rdl2::Material* material, float* presence, int materialPriority, 
        const scene_rdl2::rdl2::Material** newPriorityList, int* newPriorityListCount, int presenceDepth);

void accumVisibilityAovsOccluded(float* aovs, pbr::TLState* pbrTls, const LightSetSampler& lSampler,
                                 const BsdfSampler& bSampler, const PathVertex& pv, const Light* const light,
                                 int missCount);
//----------------------------------------------------------------------------

void drawBsdfSamples(pbr::TLState *pbrTls, const BsdfSampler &bSampler, const LightSetSampler &lSampler,
        const Subpixel &sp, const PathVertex &pv, const scene_rdl2::math::Vec3f& P, const scene_rdl2::math::Vec3f *N,
        float time, unsigned sequenceID, BsdfSample *bsmp, int clampingDepth,
        float clampingValue, shading::BsdfLobe::Type indirectFlags, float rayDirFootprint);

void drawLightSetSamples(pbr::TLState *pbrTls, const LightSetSampler &lSampler, const BsdfSampler &bSampler,
        const Subpixel &sp, const PathVertex &pv, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f *N,
        float time, unsigned sequenceID, LightSample *lsmp, int clampingDepth, float clampingValue, 
        float rayDirFootprint, float* aovs, int lightIndex);

void applyRussianRoulette(const BsdfSampler &bSampler, BsdfSample *bsmp,
        const Subpixel &sp, const PathVertex &pv, unsigned sequenceID,
        float threshold, float invThreshold);

void applyRussianRoulette(const LightSetSampler &lSampler, LightSample *lsmp,
        const Subpixel &sp, const PathVertex &pv, unsigned sequenceID,
        float threshold, float invThreshold, IntegratorSample1D& rrSamples);

void accumulateRayPresence(pbr::TLState *pbrTls,
                           const Light* light,
                           const mcrt_common::Ray& shadowRay,
                           float rayEpsilon,
                           int maxDepth,
                           float& totalPresence);

void scatterAndScale(const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
                     const shading::BsdfLobe &lobe, const scene_rdl2::math::Vec3f &wo,
                     const scene_rdl2::math::Vec3f &wi, float scale, float r1, float r2,
                     mcrt_common::RayDifferential &rd);

void scatterAndScale(const shading::Intersection &isect, const shading::BsdfLobe &lobe,
                     const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi, float scale,
                     float r1, float r2,
                     mcrt_common::RayDifferential &rd);

//----------------------------------------------------------------------------
// Subsurface Integrator Utility Functions

// Selects an axis for projection
int bssrdfSelectAxisAndRemapSample(const scene_rdl2::math::ReferenceFrame &localF,
                                   float &rnd,
                                   scene_rdl2::math::Vec3f &directionProj);
// Converts the *local* BSSRDF Offset into *render* space using the
// appropriate local frame based on the projection axis
scene_rdl2::math::Vec3f bssrdfOffsetLocalToGlobal(const scene_rdl2::math::ReferenceFrame &localF,
                                      int axisIndex,
                                      const scene_rdl2::math::Vec3f &PiTangent);
// Calculates the MIS Weight Using Veachs' One-Sample Method
float bssrdfGetMISAxisWeight(const scene_rdl2::math::ReferenceFrame &localF,
                             const scene_rdl2::math::Vec3f &NiProj,
                             const scene_rdl2::math::Vec3f &offset,
                             const shading::Bssrdf &bssrdf);
// Calculates the Area-Compensation Term.
// We divide the analytically computed diffuse reflectance by the one
// computed via sampling. This gives us a measure of how much the underlying
// geometry deviates from the "semi-infinite, planar" assumption for diffusion based BSSRDFs.
// On a regular planar surface, this compensation term amount to 1.
scene_rdl2::math::Color bssrdfAreaCompensation(const scene_rdl2::math::Color &measuredDiffuseReflectance,
                                   const scene_rdl2::math::Color &analyticDiffuseReflectance);

// End of Subsurface Integrator Utility Functions
//----------------------------------------------------------------------------


//-------------------------------------------------------------------------------
// Shadow Falloff Utility Functions

scene_rdl2::math::Color calculateShadowFalloff(const Light *light, float distToLight, 
                                               const scene_rdl2::math::Color unoccludedColor);

// End of Shadow Falloff Utility Functions
//----------------------------------------------------------------------------

// These functions are called from the bundled ISPC code path.

// Function exposed to ISPC:
extern "C"
{

bool CPP_isIntegratorAccumulatorRunning(pbr::TLState *pbrTls);

bool CPP_isIspcAccumulatorRunning(pbr::TLState *pbrTls);

void CPP_computeRadianceSubsurface(const PathIntegrator * pathIntegrator,
                                   pbr::PbrTLState * pbrTls,
                                   const uint32_t * rayStateIndices,
                                   const shading::Bssrdfv * bssrdfv,
                                   const shading::VolumeSubsurfacev * volumeSubsurfacev,
                                   const LightSet * lightSet,
                                   const int materialLabelIds,
                                   const int lpeMaterialLabelIds,
                                   const int * geomLabelIds,
                                   const int * doIndirect,
                                   const float * rayEpsilon,
                                   const float * shadowRayEpsilon,
                                   uint32_t * sequenceID,
                                   float * results,        // VLEN rgb colors in SOA format
                                   float * ssAovResults,   // VLEN rgb colors in SOA format
                                   int32_t lanemask);

void
CPP_computeRadianceEmissiveRegionsBundled(const PathIntegrator *pathIntegrator,
    PbrTLState *pbrTls, const uint32_t *rayStateIndices, const float *rayEpsilons,
    const shading::Bsdfv *bsdfv, const shading::BsdfSlicev *slicev,
    float *results, int32_t lanemask);

void
CPP_applyVolumeTransmittance(const PathIntegrator *pathIntegrator,
    PbrTLState *pbrTls, const uint32_t *rayStateIndices, int32_t lanemask);

void CPP_addIncoherentRayQueueEntries(pbr::TLState *pbrTls, const RayStatev *rayStatesv,
                                      unsigned numRayStates, const unsigned *indices);

void CPP_addOcclusionQueueEntries(pbr::TLState *pbrTls, const BundledOcclRayv *occlRaysv,
                                  unsigned numOcclRays, const unsigned *indices);

void CPP_addPresenceShadowsQueueEntries(pbr::TLState *pbrTls, const BundledOcclRayv *occlRaysv,
                                        unsigned numOcclRays, const unsigned *indices);

void CPP_addRadianceQueueEntries(pbr::TLState *pbrTls, const BundledRadiancev *radiancesv,
                                 unsigned numRadiances, const unsigned *indices);

}

//----------------------------------------------------------------------------


} // namespace pbr
} // namespace moonray

