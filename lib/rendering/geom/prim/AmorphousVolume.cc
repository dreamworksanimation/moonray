// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file AmorphousVolume.cc
///

#include "AmorphousVolume.h"

#include <moonray/rendering/bvh/shading/State.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;
using namespace scene_rdl2::math;

namespace {

template<typename GridType>
std::vector<float>
getGridLuminance(const amorphous::SScatterSampler& sampler,
                 const GridType& emissionGrid)
{
    std::vector<float> values;
    for (auto it = emissionGrid.cbeginValueOn(); it; ++it) {
        const openvdb::math::Coord& xyz = it.getCoord();
        const auto rgba = sampler.incand(xyz);
        values.push_back(luminance(Color(rgba.r(), rgba.g(), rgba.b())));
    }
    return values;
}

} // anonymous namespace

AmorphousVolume::~AmorphousVolume() = default;

AmorphousVolume::AmorphousVolume(const std::string& vdbFilePath,
                                 const std::string& densityGridName,
                                 const std::string& velocityGridName,
                                 const MotionBlurParams& motionBlurParams,
                                 LayerAssignmentId&& layerAssignmentId,
                                 PrimitiveAttributeTable&& primitiveAttributeTable):
    VdbVolume(vdbFilePath,
              densityGridName,
              "", // Emission grid name is only used for non-amorphous
                  // vdb rendering so we leave this empty.  For Amorphous
                  // vdbs the emission grid name is stored in the file's
                  // meta-data.
              velocityGridName,
              motionBlurParams,
              std::move(layerAssignmentId),
              std::move(primitiveAttributeTable)),
    mUseAmorphousSampler(false)
{
}

size_t
AmorphousVolume::getMemory() const
{
    size_t mem = VdbVolume::getMemory();

    // AmorphousSampler loads a few grids into memory
    mem += mAmorphousSampler.getMemory();

    return mem;
}

std::unique_ptr<EmissionDistribution>
AmorphousVolume::computeEmissionDistribution(const scene_rdl2::rdl2::VolumeShader* volumeShader) const
{
    if (!mHasEmissionField || !mUseAmorphousSampler || !mAmorphousSampler.mMasterSampler) {
        // There is no amorphous data but there is a emission from a volume shader.
        return Primitive::computeEmissionDistribution(volumeShader);
    }

    openvdb::GridBase::ConstPtr emissionGrid = mAmorphousSampler.getIncandGrid();
    const amorphous::SScatterSampler& scatterSampler = *mAmorphousSampler.mMasterSampler.get();

    MNRY_ASSERT_REQUIRE(emissionGrid);
    MNRY_ASSERT_REQUIRE(emissionGrid->transform().isLinear());

    std::unique_ptr<EmissionDistribution> distribution;

    // TODO: fix the dual usage of VdbVolume's mPrimToRender matrices, then we can pass mPrimToRender into
    // computeEmissionDistributionImpl() in both the calls here. We can't at present because it may be being used to
    // store something different
    if (emissionGrid->isType<amorphous::ScalarGridType>()) {

        std::vector<float> values = getGridLuminance(scatterSampler,
            *static_cast<const amorphous::ScalarGridType*>(emissionGrid.get()));

        distribution = computeEmissionDistributionImpl(
            getRdlGeometry(),
            *static_cast<const amorphous::ScalarGridType*>(emissionGrid.get()),
            mRenderToPrim,
            values,
            volumeShader);

    } else if (emissionGrid->isType<amorphous::RGBGridType>()) {

        std::vector<float> values = getGridLuminance(scatterSampler,
            *static_cast<const amorphous::RGBGridType*>(emissionGrid.get()));

        distribution = computeEmissionDistributionImpl(
            getRdlGeometry(),
            *static_cast<const amorphous::RGBGridType*>(emissionGrid.get()),
            mRenderToPrim,
            values,
            volumeShader);
    }
    return distribution;
}

Color
AmorphousVolume::evalDensity(mcrt_common::ThreadLocalState* tls,
                             uint32_t volumeId,
                             const Vec3f& pSample,
                             float /*rayVolumeDepth*/,
                             const scene_rdl2::rdl2::VolumeShader* const /*volumeShader*/) const
{
    const openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
    if (mUseAmorphousSampler) {
        // for shadow attenuation, always use the cheapest interpolation
        const Color density = mAmorphousSampler.extinct(tls, volumeId, p, Interpolation::POINT);
        Color retVal = density * sampleBakedDensity(tls, volumeId, p);
        return retVal;
    } else {
        const Color density = Color(mDensitySampler.eval(tls, volumeId, p, Interpolation::POINT));
        Color retVal = density * sampleBakedDensity(tls, volumeId, p);
        return retVal;
    }
}

void
AmorphousVolume::evalVolumeCoefficients(mcrt_common::ThreadLocalState* tls,
                                        uint32_t volumeId,
                                        const Vec3f& pSample,
                                        Color* extinction,
                                        Color* albedo,
                                        Color* temperature,
                                        bool highQuality,
                                        float /*rayVolumeDepth*/,
                                        const scene_rdl2::rdl2::VolumeShader* const /*volumeShader*/) const
{
    const openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
    Interpolation mode = highQuality ? mInterpolationMode : Interpolation::POINT;
    if (mUseAmorphousSampler) {
        *extinction = mAmorphousSampler.extinct(tls, volumeId, p, mode);
        *albedo = mAmorphousSampler.albedo(tls, volumeId, p, mode);
        if (temperature) {
            *temperature = mAmorphousSampler.incand(tls, volumeId, p, mode);
        }
    } else {
        *extinction = Color(mDensitySampler.eval(tls, volumeId, p, mode));
        *albedo = Color(1.0f);
        if (temperature) {
            const auto colorVector = mEmissionSampler.eval(tls, volumeId, p, Interpolation::POINT);
            *temperature = Color(colorVector.x(), colorVector.y(), colorVector.z());
        }
    }

    *extinction *= sampleBakedDensity(tls, volumeId, p);
}

Color
AmorphousVolume::evalTemperature(mcrt_common::ThreadLocalState* tls,
                                 uint32_t volumeId,
                                 const Vec3f& pSample) const
{
    const openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
    if (mUseAmorphousSampler) {
        return mAmorphousSampler.incand(tls, volumeId, p, Interpolation::POINT);
    } else {
        const auto colorVector = mEmissionSampler.eval(tls, volumeId, p, Interpolation::POINT);
        return Color(colorVector.x(), colorVector.y(), colorVector.z());
    }
}

bool
AmorphousVolume::initAmorphousSampler(const std::string& vdbFilePath,
                                const std::string& densityGridName,
                                const scene_rdl2::rdl2::Geometry& rdlGeometry,
                                openvdb::VectorGrid::Ptr velocityGrid,
                                const std::vector<int>& volumeIds)
{
    const float emissionSampleRate = scene_rdl2::math::clamp(
        mVdbVolumeData->mEmissionSampleRate);

    if (!mAmorphousSampler.initialize(vdbFilePath,
                                      densityGridName,
                                      rdlGeometry,
                                      emissionSampleRate,
                                      volumeIds)) {
        return false;
    }

    mTopologyGrid = mAmorphousSampler.getDensityGrid();
    // padding extra voxels for motion blur usage
    if (velocityGrid) {
        float shutterOpen, shutterClose;
        mVdbVelocity->getShutterOpenAndClose(shutterOpen, shutterClose);
        initMotionBlurBoundary(mTopologyGrid, velocityGrid,
            shutterOpen, shutterClose);
    }

    mHasEmissionField = (mAmorphousSampler.getIncandGrid() != nullptr) &&
        (!mAmorphousSampler.getIncandGrid()->empty());

    return true;
}

bool
AmorphousVolume::initialize(const scene_rdl2::rdl2::Geometry& rdlGeometry,
                      const scene_rdl2::rdl2::Layer* layer,
                      const VolumeAssignmentTable* volumeAssignmentTable)
{
    const std::string& vdbFilePath = mVdbVolumeData->mFilePath;
    if (vdbFilePath.empty()) {
        return false;
    }

    openvdb::io::File file(vdbFilePath);
    openvdb::GridPtrVecPtr grids;
    openvdb::VectorGrid::Ptr velocityGrid;
    std::vector<int> volumeIds;

    if (!initializePhase1(vdbFilePath,
                          file,
                          rdlGeometry,
                          layer,
                          volumeAssignmentTable,
                          volumeIds,
                          grids,
                          velocityGrid)) {
        return false;
    }

    if (initAmorphousSampler(vdbFilePath,
                             mVdbVolumeData->mDensityGridName,
                             rdlGeometry,
                             velocityGrid,
                             volumeIds)) {
        mUseAmorphousSampler = true;
    } else {
        rdlGeometry.warn("fallback to default grid loading solution");
        mUseAmorphousSampler = false;
        if (!initDensitySampler(file,
                                grids,
                                mVdbVolumeData->mDensityGridName,
                                rdlGeometry,
                                velocityGrid,
                                volumeIds)) {
            return false;
        }

        if (!mVdbVolumeData->mEmissionGridName.empty()) {
            if (!initEmissionSampler(file,
                                     grids,
                                     mVdbVolumeData->mEmissionGridName,
                                     rdlGeometry,
                                     velocityGrid,
                                     volumeIds)) {
                mHasEmissionField = false;
                return false;
            } else {
                mHasEmissionField = !mEmissionSampler.mGrid->empty();
            }
        }
    }

    initializePhase2(rdlGeometry,
                     layer,
                     volumeAssignmentTable,
                     volumeIds,
                     velocityGrid);

    return true;
}

} // namespace internal
} // namespace geom
} // namespace moonray


