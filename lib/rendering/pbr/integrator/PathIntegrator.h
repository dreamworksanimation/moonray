// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PathIntegrator.h
/// $Id$
///

#pragma once

#include "PathIntegrator.hh"

#include "BsdfOneSampler.h"
#include "BsdfSampler.h"
#include "LightSetSampler.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/pbr/sampler/Sampler.h>
#include <moonray/rendering/rndr/rndr.h>
#include <moonray/rendering/rndr/Types.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/ispc/Shadingv.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/scene/rdl2/SceneVariables.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <vector>

namespace moonray {

namespace geom {
namespace internal {
class TLState;
class VolumeRayState;
class VolumeRegions;
}
}

namespace mcrt_common {
class Ray;
class RayDifferential;
}

namespace shading {
class BsdfSlice;
class Bssrdf;
class VolumeSubsurface;
}

namespace pbr {

// Forward Decl.
class Camera;
class CryptomatteBuffer;
class DeepBuffer;
struct FrameState;
class Light;
class PathGuide;
struct PathVertex;
struct RayState;
struct Sample;
class Sampler;
class Scene;
struct Subpixel;
class TLState;
class VolumePhase;
class VolumeProperties;
class VolumeScatteringSampler;
class VolumeTransmittance;

//----------------------------------------------------------------------------
enum class VolumeOverlapMode
{
    SUM = static_cast<int>(scene_rdl2::rdl2::VolumeOverlapMode::SUM),
    MAX = static_cast<int>(scene_rdl2::rdl2::VolumeOverlapMode::MAX),
    RND = static_cast<int>(scene_rdl2::rdl2::VolumeOverlapMode::RND),
    NUM_MODES
};

struct PathIntegratorParams
{
    int mIntegratorPixelSamplesSqrt;
    int mIntegratorLightSamplesSqrt;
    int mIntegratorBsdfSamplesSqrt;
    int mIntegratorBssrdfSamplesSqrt;
    int mIntegratorMaxDepth;
    int mIntegratorMaxDiffuseDepth;
    int mIntegratorMaxGlossyDepth;
    int mIntegratorMaxMirrorDepth;
    int mIntegratorMaxVolumeDepth;
    int mIntegratorMaxPresenceDepth;
    int mIntegratorMaxHairDepth;
    int mIntegratorMaxSubsurfacePerPath;
    float mIntegratorTransparencyThreshold;
    float mIntegratorPresenceThreshold;
    float mIntegratorRussianRouletteThreshold;
    float mSampleClampingValue;
    int mSampleClampingDepth;
    float mRoughnessClampingFactor;
    float mIntegratorVolumeQuality;
    float mIntegratorVolumeShadowQuality;
    int mIntegratorVolumeIlluminationSamples;
    float mIntegratorVolumeTransmittanceThreshold;
    float mIntegratorVolumeAttenuationFactor;
    float mIntegratorVolumeContributionFactor;
    float mIntegratorVolumePhaseAttenuationFactor;
    VolumeOverlapMode mIntegratorVolumeOverlapMode;
};

struct ComputeRadianceAovParams
{
    float mAlpha;
    float *mDepth;
    float *mAovs;
    float *mDeepAovs;        // temporary aov storage for the path integrator
    float *mDeepVolumeAovs;  // temporary aov storage for the volume integrator
};

//----------------------------------------------------------------------------

///
/// @class PathIntegrator PathIntegrator.h <pbr/PathIntegrator.h>
/// @brief A path tracer integrator.
///
class PathIntegrator
{
public:

    struct DeepParams
    {
        DeepBuffer *mDeepBuffer;
        int mPixelX;
        int mPixelY;
        float mSampleX;
        float mSampleY;
        int mPixelSamples;

        // temporary AOV storage for volume integrator
        float *mVolumeAovs;

        // intersection results
        bool mHitDeep;
        float mDeepIDs[7];  // Also see types.hh DEEP_DATA_MEMBERS
    };

    /// Constructor / Destructor
    /// Valid square-root sample counts are 1, 2, 4, 8 as these also map to
    /// powers of 2:
    /*
          n  -  n*n  -  2^n
        -------------------
          0  -    0  -    1
          1  -    1  -    2
          2  -    4  -    4
          3  -    9  -    8
          4  -   16  -   16
          5  -   25  -   32
          6  -   36  -   64
          7  -   49  -  128
          8  -   64  -  256
          9  -   81  -  512
         10  -  100  - 1024
     */
    /// TODO: implement getters and setters for all the settings
    PathIntegrator();
    ~PathIntegrator();

    void update(const FrameState &fs, const PathIntegratorParams& params);

    // A render might be broken down into a series of passes.
    // The render driver is responsible for ensuring that this method is called
    // only on one thread and that all other render threads are blocked.
    void passReset();

    /// Sample a path for the given:
    /// - pixel in viewport coordinates)
    /// - subpixel in the range [0,subpixelRate)
    /// - If we encountered a NaN during the execution of this function, a
    ///   a negative alpha value is returned and radiance is set to black.
    scene_rdl2::math::Color computeRadiance(pbr::TLState *pbrTls, int pixelX, int pixelY,
            int subpixelIndex, int pixelSamples, const Sample& sample,
            ComputeRadianceAovParams &aovParams, DeepBuffer *deepBuffer,
            CryptomatteBuffer *cryptomatteBuffer) const;

    /// Output intersection properties as color for fast progressive render mode
    scene_rdl2::math::Color computeColorFromIntersection(pbr::TLState *pbrTls, int pixelX, int pixelY,
            int subpixelIndex, int pixelSamples, const Sample& sample,
            ComputeRadianceAovParams &aovParams, rndr::FastRenderMode fastMode) const;

    scene_rdl2::math::Color computeRadianceDiffusionSubsurface(pbr::TLState *pbrTls, const shading::Bsdf &bsdf,
            const Subpixel &sp, const PathVertex &pv,
            const mcrt_common::RayDifferential &ray, const shading::Intersection &isect,
            const shading::BsdfSlice &slice, const shading::Bssrdf &bssrdf, const LightSet &lightSet,
            bool doIndirect, float rayEpsilon, float shadowRayEpsilon, unsigned &sequenceID,
            scene_rdl2::math::Color &ssAov, float *aovs) const;

    scene_rdl2::math::Color computeRadiancePathTraceSubsurface(pbr::TLState *pbrTls, const shading::Bsdf &bsdf,
            const Subpixel &sp, const PathVertex &pv,
            const mcrt_common::RayDifferential &ray, const shading::Intersection &isect,
            const shading::VolumeSubsurface& volumeSubsurface, const LightSet &lightSet,
            bool doIndirect, float rayEpsilon, float shadowRayEpsilon, unsigned &sequenceID,
            scene_rdl2::math::Color &ssAov, float *aovs) const;

    // compute the volume scattering contribution and add the result to
    // radiance, also compute the volume transmittance across
    // the whole ray segment (stored to tr). Return true when there is radiance
    // contribution along ray due to volume emission or in-scattering
    // RayState will be null if rendering scalar, otherwise it should be
    // non-null and is used to queue aov values.  The aovs buffer is only
    // used in scalar mode.
    bool computeRadianceVolume(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
            const Subpixel& sp, PathVertex& pv, const int lobeType,
            scene_rdl2::math::Color& radiance, unsigned sequenceID, VolumeTransmittance& vt,
            float* aovs, DeepParams* deepParams, const RayState *rs,
            float* surfaceT) const;

    // return the volume transmittance across the whole ray segment
    // estimateInScatter=true : light transmittance computation for volume
    //                   false : all other situations 
    scene_rdl2::math::Color transmittance(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
            uint32_t pixel, int subpixelIndex, unsigned sequenceID, const Light* light,
            float scaleFactor = 1.0f, bool estimateInScatter = false) const;

    // compute and return the emission contribution from volumes emitting energy
    // (something like fire and explosion) toward intersection point
    // with vector shading results.
    // aovs are sent directly to the aov queue.
    scene_rdl2::math::Color computeRadianceEmissiveRegionsBundled(pbr::TLState *pbrTls, const RayState &rs,
            const shading::Bsdfv &bsdfv, const shading::BsdfSlicev &slicev,
            float rayEpsilon, unsigned int lane) const;

    bool queuePrimaryRay(pbr::TLState *pbrTls, int pixelX, int pixelY,
            int subpixelIndex, int pixelSamples, const Sample& sample,
            RayState *rs) const;

    // Used for bundled case as a wrapper to call directly into ISPC.
    void integrateBundledv(pbr::TLState *pbrTls, shading::TLState *shadingTls, unsigned numEntries,
            RayStatev *rayStates, const shading::Intersectionv *isects,
            const shading::Bsdfv *bsdfs, const LightPtrList *lightList,
            const LightFilterLists *lightFilterLists,
            const LightAccelerator *lightAcc,
            const float *presences) const;

    bool getEnableShadowing() const { return mEnableShadowing; }
    bool getEnablePathGuide() const;

    // mLightSamples is the user parameter "light_sample_count" squared
    int getLightSampleCount() const { return mLightSamples; }

    const std::vector<int>& getDeepIDAttrIdxs() const { return mDeepIDAttrIdxs; }
    int getDeepMaxLayers() const { return mDeepMaxLayers; }
    float getDeepLayerBias() const { return mDeepLayerBias; }

    int getCryptoUVAttrIdx() const { return mCryptoUVAttrIdx; }

    // HUD validation.
    static uint32_t hudValidation(bool verbose) { PATH_INTEGRATOR_VALIDATION; }

    // Is this a path to calculate Subsurface Scattering for?
    // We use it in the material to replace Subsurface with
    // Lambertian diffuse which results in faster renders
    // with very similar final results.
    bool isSubsurfaceAllowed(int subsurfaceDepth) const
    {
        return (subsurfaceDepth < mMaxSubsurfacePerPath);
    }

private:
    enum IndirectRadianceType
    {
        NONE = 0,
        SURFACE = 1 << 0,
        VOLUME  = 1 << 1,
        ALL = SURFACE | VOLUME
    };


    struct CryptomatteParams
    {
        // intersection results
        bool mHit;
        float mId;
        scene_rdl2::math::Vec3f mPosition;
        scene_rdl2::math::Vec3f mNormal;
        scene_rdl2::math::Color4 mBeauty;
        scene_rdl2::math::Vec3f mRefP;
        scene_rdl2::math::Vec3f mRefN;
        scene_rdl2::math::Vec2f mUV;
        CryptomatteBuffer* mCryptomatteBuffer;

        void init(CryptomatteBuffer* buffer) {
            mHit = false;
            mId = 0.f;
            mPosition = scene_rdl2::math::Vec3f(0.f);
            mNormal = scene_rdl2::math::Vec3f(0.f);
            mBeauty = scene_rdl2::math::Color4(0.f);
            mRefP = scene_rdl2::math::Vec3f(0.f);
            mRefN = scene_rdl2::math::Vec3f(0.f);
            mUV = scene_rdl2::math::Vec2f(0.f);
            mCryptomatteBuffer = buffer;
        }
    };

    /// Copy is disabled
    PathIntegrator(const PathIntegrator&) = delete;
    PathIntegrator &operator=(const PathIntegrator&) = delete;

    finline void addDirectVisibleBsdfLobeSampleContribution(pbr::TLState *pbrTls,
            const Subpixel &sp, const PathVertex &pv,
            const BsdfSampler &bSampler, int lobeIndex, bool doIndirect, const BsdfSample &bsmp,
            const mcrt_common::RayDifferential &parentRay, float rayEpsilon, float shadowRayEpsilon,
            scene_rdl2::math::Color &radiance, unsigned& sequenceID, float *aovs,
            const shading::Intersection &isect) const;

    void addDirectVisibleBsdfSampleContributions(pbr::TLState *pbrTls,
            const Subpixel &sp, const PathVertex &pv,
            const BsdfSampler &bSampler, bool doIndirect, const BsdfSample *bsmp,
            const mcrt_common::RayDifferential &parentRay, float rayEpsilon, float shadowRayEpsilon,
            scene_rdl2::math::Color &radiance, unsigned& sequenceID, float *aovs,
            const shading::Intersection &isect) const;

    void addDirectVisibleLightSampleContributions(pbr::TLState *pbrTls,
            const Subpixel &sp, const PathVertex &pv,
            const LightSetSampler &lSampler, LightSample *lsmp,
            const BsdfSampler& bSampler, const scene_rdl2::math::Vec3f* cullingNormal,
            const mcrt_common::RayDifferential &parentRay, float rayEpsilon, float shadowRayEpsilon,
            scene_rdl2::math::Color &radiance, unsigned& sequenceID, float *aovs,
            const shading::Intersection &isect) const;

    void addIndirectOrDirectVisibleContributions(pbr::TLState *pbrTls,
            const Subpixel &sp, 
            const PathVertex &parentPv, const BsdfSampler &bSampler,
            const BsdfSample *bsmp, const mcrt_common::RayDifferential &parentRay,
            float rayEpsilon, float shadowRayEpsilon,
            const shading::Intersection &isect, shading::BsdfLobe::Type indirectFlags,
            const scene_rdl2::rdl2::Material* newPriorityList[4], int newPriorityListCount[4],
            scene_rdl2::math::Color &radiance, unsigned &sequenceID,
            float *aovs, CryptomatteParams *refractCryptomatteParams) const;

    // compute volume emission line integral along the ray
    scene_rdl2::math::Color computeEmissiveVolumeIntegral(pbr::TLState *pbrTls, mcrt_common::Ray& ray,
            int emissiveVolumeId, const Subpixel& sp,
            unsigned sequenceID) const;

    scene_rdl2::math::Color computeEmissiveVolumeIntegralSubInterval(pbr::TLState *pbrTls,
            int emissiveVolumeId, scene_rdl2::math::Color& transmittance, float t0, float t1,
            float time, int depth, const geom::internal::VolumeRayState& volumeRayState,
            int* volumeIds, const IntegratorSample1D& trSamples, const IntegratorSample1D& trSamples2,
            bool& reachTransmittanceThreshold) const;

    // compute integrated radiance, transparency and aovs from a multiple lobe bsdf using
    // a bsdf multi sampler strategy
    scene_rdl2::math::Color computeRadianceBsdfMultiSampler(pbr::TLState *pbrTls,
            const Subpixel &sp, const PathVertex &pv, const mcrt_common::RayDifferential &ray,
            const shading::Intersection &isect, const shading::Bsdf &bsdf, const shading::BsdfSlice &slice,
            bool doIndirect, shading::BsdfLobe::Type indirectFlags, const scene_rdl2::rdl2::Material *newPriorityList[4],
            int newPriorityListCount[4], const LightSet &activeLightSet, const scene_rdl2::math::Vec3f *cullingNormal,
            float rayEpsilon, float shadowRayEpsilon, const scene_rdl2::math::Color &ssAov, unsigned &sequenceID,
            float *aovs, CryptomatteParams *refractCryptomatteParams) const;

    // compute the emission contribution from volumes emitting energy
    // (something like fire and explosion) toward intersection point
    // with scalar shading results.
    scene_rdl2::math::Color computeRadianceEmissiveRegionsScalar(pbr::TLState *pbrTls,
            const Subpixel& sp, const PathVertex& pv, const mcrt_common::Ray& ray,
            const shading::Intersection& isect, shading::Bsdf& bsdf, const shading::BsdfSlice& slice,
            float rayEpsilon, unsigned sequenceID, float* aovs) const;

    // compute the emission contribution from volumes emitting energy
    // (something like fire and explosion) toward bssrdf projection samples
    scene_rdl2::math::Color computeRadianceEmissiveRegionsSSS(pbr::TLState *pbrTls,
            const Subpixel& sp, const PathVertex& pv, const mcrt_common::Ray& ray,
            const scene_rdl2::math::Color& pathThroughput, const shading::Fresnel* transmissionFresnel,
            const shading::Bsdf& bsdf, const shading::BsdfLobe &lobe, const shading::BsdfSlice &slice,
            const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n,
            int subsurfaceSplitFactor, int subsurfaceIndex,
            float rayEpsilon, unsigned sssSampleID, bool isLocal,
            float* aovs) const;

    // compute the emission contribution from volumes emitting energy
    // (something like fire and explosion) to volume in-scattering along the ray
    // If called from bundled mode, aovs are sent directly to the aov bundle handler
    // via info in the ray state.  If called from scalar code, rs will be null
    scene_rdl2::math::Color computeRadianceEmissiveRegionsVolumes(pbr::TLState *pbrTls,
            const Subpixel& sp, const PathVertex& pv, const mcrt_common::Ray& ray,
            const VolumeProperties* volumeProperties,
            const GuideDistribution1D& densityDistribution,
            unsigned sequenceID, float* aovs, const RayState *rs) const;

    // compute volume in-scattering integration estimator along the ray
    scene_rdl2::math::Color integrateVolumeScattering(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
            const VolumeProperties* volumeProperties,
            const GuideDistribution1D& densityDistribution,
            const Subpixel &sp, const PathVertex& pv,
            unsigned& sequenceID, float* aovs,
            DeepParams* deepParams, const RayState *rs) const;

    scene_rdl2::math::Color equiAngularVolumeScattering(pbr::TLState *pbrTls,
            const mcrt_common::Ray& ray, int lightIndex,
            float ue, const scene_rdl2::math::Vec3f& ul, const LightFilterRandomValues& ulFilter,
            float D, float thetaA, float thetaB, float offset,
            const VolumeProperties* volumeProperties,
            const GuideDistribution1D& densityDistribution,
            const Subpixel &sp, unsigned& sequenceID, bool doMIS) const;

    scene_rdl2::math::Color distanceVolumeScattering(pbr::TLState *pbrTls,
            const mcrt_common::Ray& ray, int lightIndex,
            float ud, const scene_rdl2::math::Vec3f& ul, const LightFilterRandomValues& ulFilter,
            float D, float thetaA, float thetaB, float offset,
            const VolumeProperties* volumeProperties,
            const GuideDistribution1D& densityDistribution,
            const Subpixel &sp, unsigned& sequenceID, bool doMIS,
            float& td, scene_rdl2::math::Color& radiance, scene_rdl2::math::Color& transmittance) const;

    scene_rdl2::math::Color approximateVolumeMultipleScattering(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
            const VolumeProperties* volumeProperties,
            const GuideDistribution1D& densityDistribution,
            const Subpixel &sp, const PathVertex& pv, const int rayMask,
            unsigned sequenceID, float* aovs, DeepParams* deepParams, const RayState *rs) const;

    // estimator for volume scattering direct lighting contribution
    // from specified light to scatterPoint
    scene_rdl2::math::Color estimateInScatteringSourceTerm(pbr::TLState *pbrTls, const mcrt_common::Ray& ray,
            const scene_rdl2::math::Vec3f& scatterPoint, const Light* light, int assignmentId,
            const VolumePhase& phaseFunction,
            const scene_rdl2::math::Vec3f& ul, const LightFilterRandomValues& ulFilter,
            const Subpixel &sp, unsigned sequenceID,
            float scaleFactor = 1.0f) const;

    scene_rdl2::math::Color transmittanceSubinterval(pbr::TLState *pbrTls,
            float t0, float t1,
            const geom::internal::VolumeRegions& volumeRegions,
            float time, int depth,
            const IntegratorSample1D& trSamples, float tauThreshold,
            const Light* light, float scaleFactor) const;

    // march through the volume regions between t0 and t1 and
    // accumulate volume transmittance, emission integration.
    // It also store the marching steps in an array of VolumeProperties
    // for later 1d distribution construction
    void decoupledRayMarching(pbr::TLState *pbrTls,
            VolumeTransmittance& vt, scene_rdl2::math::Color* perVolumeLVe,
            const size_t maxStepCount, size_t& marchingStepsCount,
            float t0, float t1, float time, int depth,
            const geom::internal::VolumeRayState& volumeRayState,
            int* volumeIds, VolumeProperties* volumeProperties,
            const IntegratorSample1D& trSamples, const IntegratorSample1D& trSamples2,
            bool& reachTransmittanceThreshold,
            DeepParams* deepParams) const;

    // return the type of indirect radiance contribution along ray
    // (it can be a surface intersection that contributes bounce lighting or
    // volume emission/in-scattering accumulated along the ray)
    IndirectRadianceType computeRadianceRecurse(pbr::TLState *pbrTls,
            mcrt_common::RayDifferential &ray,
            const Subpixel &sp, const PathVertex &pv, const shading::BsdfLobe *lobe,
            scene_rdl2::math::Color &radiance, float &transparency, VolumeTransmittance& vt,
            unsigned &sequenceID, float *aovs, float *depth,
            DeepParams* deepParams, CryptomatteParams *cryptomatteParams,
            CryptomatteParams *refractCryptomatteParams,
            bool ignoreVolumes, bool &hitVolume) const;

    scene_rdl2::math::Color computeRadianceSubsurfaceSample(pbr::TLState *pbrTls,
            const shading::Bsdf &bsdf, const Subpixel &sp,
            const PathVertex &pv, const mcrt_common::RayDifferential &parentRay,
            const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            const scene_rdl2::math::Color &pathThroughput, const shading::Fresnel *transmissionFresnel,
            const LightSet &lightSet, shading::BsdfLobe &lobe,
            const shading::BsdfSlice &slice, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &N,
            int subsurfaceSplitFactor, int computeRadianceSplitFactor,
            int subsurfaceIndex, bool doIndirect, float rayEpsilon, float shadowRayEpsilon,
            unsigned sssSampleID, unsigned &sequenceID, bool isLocal, float *aovs,
            const shading::Intersection &isect) const;

    scene_rdl2::math::Color computeDiffusionForwardScattering(pbr::TLState *pbrTls,
            const shading::Bsdf &bsdf, const Subpixel &sp, const PathVertex &pv,
            const mcrt_common::RayDifferential &ray, const shading::Intersection &isect,
            const shading::BsdfSlice &slice, const shading::Fresnel *transmissionFresnel,
            const scene_rdl2::math::Color& scaleFresnelWo, const LightSet &lightSet,
            const shading::Bssrdf &bssrdf, const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &N,
            const scene_rdl2::math::ReferenceFrame &localF, int subsurfaceSplitFactor,
            bool doIndirect, float rayEpsilon, float shadowRayEpsilon, unsigned sssSampleID,
            unsigned &sequenceID, scene_rdl2::math::Color &ssAov, float *aovs) const;

    bool initPrimaryRay(pbr::TLState *pbrTls, const Camera *camera,
            int pixelX, int pixelY, int subpixelIndex, int pixelSamples,
            const Sample& sample, mcrt_common::RayDifferential &ray,
            Subpixel &sp, PathVertex &pv) const;

    // Utility function handling occlusion/presence shadow ray query based on whether light has presence shadow enabled.
    // Return true when the ray is completely occluded
    // (either hits a fully opaque surface, or accumulated presence reaches 1).
    // receiverId is set to -1 in the general case because we only need to pass a valid Id when there is
    // shadow-linking information present. skipOcclusionFilter() makes use of this value to suppress occusion
    // of a specified shadow receiver by the specified shadow caster in the shadow-linking info.
    bool isRayOccluded(pbr::TLState *pbrTls, const Light* light, mcrt_common::Ray& shadowRay, float rayEpsilon,
                       float shadowRayEpsilon, float& presence, int receiverId, bool isVolume = false) const;

    PATH_INTEGRATOR_MEMBERS;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

