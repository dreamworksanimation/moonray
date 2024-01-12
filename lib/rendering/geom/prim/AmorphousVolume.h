// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file AmorphousVolume.h
///

#pragma once

#include "VdbVolume.h"

#include <amorphous_core/File.h>
#include <amorphous_core/ScatterSampler.h>
#include <amorphous_core/Rgba.h>

namespace moonray {
namespace geom {
namespace internal {

class AmorphousVolume : public VdbVolume
{
public:
    AmorphousVolume(const std::string& vdbFilePath,
            const std::string& densityGridName,
            const std::string& velocityGridName,
            const MotionBlurParams& motionBlurParams,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~AmorphousVolume();

    virtual size_t getMemory() const override;

    virtual std::unique_ptr<EmissionDistribution>
    computeEmissionDistribution(const scene_rdl2::rdl2::VolumeShader* volumeShader) const override;

    virtual scene_rdl2::math::Color evalDensity(mcrt_common::ThreadLocalState* tls,
                                    uint32_t volumeId,
                                    const Vec3f& pSample,
                                    float /*rayVolumeDepth*/,
                                    const scene_rdl2::rdl2::VolumeShader* const /*volumeShader*/) const override;

    virtual void evalVolumeCoefficients(mcrt_common::ThreadLocalState* tls,
                                        uint32_t volumeId,
                                        const Vec3f& pSample,
                                        scene_rdl2::math::Color* extinction,
                                        scene_rdl2::math::Color* albedo,
                                        scene_rdl2::math::Color* temperature,
                                        bool highQuality,
                                        float /*rayVolumeDepth*/,
                                        const scene_rdl2::rdl2::VolumeShader* const /*volumeShader*/) const override;

    virtual scene_rdl2::math::Color evalTemperature(mcrt_common::ThreadLocalState* tls,
                                        uint32_t volumeId,
                                        const Vec3f& pSample) const override;

protected:
    virtual bool initialize(const scene_rdl2::rdl2::Geometry& rdlGeometry,
                    const scene_rdl2::rdl2::Layer* layer,
                    const VolumeAssignmentTable* volumeAssignmentTable) override;

    // Thge amorphous library uses metadata to figure out which
    // grid in the vdb file needs to be loaded up.
    bool initAmorphousSampler(const std::string& vdbFilePath,
                              const std::string& densityGridName,
                              const scene_rdl2::rdl2::Geometry& rdlGeometry,
                              openvdb::VectorGrid::Ptr velocityGrid,
                              const std::vector<int>& volumeIds);

    struct AmorphousSampler
    {
        using ScatterParams = amorphous::SScatterParams;
        using ScatterSampler = amorphous::SScatterSampler;

        const ScatterParams& scatterParams() const { return mScatterParams; }

        // Initialize from scattering parameters.
        bool initialize(const ScatterParams& scatterParams,
            const std::vector<int>& volumeIds,
            const scene_rdl2::rdl2::Geometry* rdlGeometry = nullptr)
        {
            if (!scatterParams.extinctionGrid() ||
                scatterParams.extinctionGrid()->empty())
            {
                if (rdlGeometry) {
                    rdlGeometry->error("An extinction"
                        " (AKA opacity or shadowing) volume is required.");
                }
                return false;
            }

            mScatterParams = scatterParams;
            mMasterSampler.reset(new ScatterSampler(mScatterParams));
            mVolumeIdCount = volumeIds.size();
            for (unsigned samplerId = 0; samplerId < mVolumeIdCount; ++samplerId) {
                mVolumeIdToSamplerId[volumeIds[samplerId]] = samplerId;
            }
            unsigned samplerCount = mVolumeIdCount * mcrt_common::getNumTBBThreads();
            for (unsigned i = 0; i < samplerCount; ++i) {
                mThreadSamplers.push_back(*mMasterSampler);
            }
            return true;
        }

        // Initialize from a .vdb file.
        bool initialize(const std::string& vdbFilePath,
            const std::string& densityGridName,
            const scene_rdl2::rdl2::Geometry& rdlGeometry,
            const float emissionSampleRate,
            const std::vector<int>& volumeIds)
        {
            amorphous::File vdbFile(vdbFilePath);
            std::stringstream msg;
            if (!vdbFile.read(msg, densityGridName)) {
                rdlGeometry.error(msg.str());
                return false;
            }

            amorphous::SScatterParams& scatterParams = vdbFile.scatterParams();
            if (!scatterParams.extinctionGrid() ||
                scatterParams.extinctionGrid()->empty())
            {
                rdlGeometry.error("VDB", vdbFilePath,
                    " contains no extinction metadata. "
                    "Extinction (AKA opacity or shadowing) is required.");
                return false;
            }

            // downsample emission grid
            if (scatterParams.incandescenceGrid() != nullptr &&
                !scatterParams.incandescenceGrid()->empty() &&
                emissionSampleRate > 0.0f && emissionSampleRate < 1.0f) {
                scatterParams.downsampleIncand(emissionSampleRate);
            }

            return initialize(scatterParams, volumeIds);
        }

        scene_rdl2::math::Color extinct(mcrt_common::ThreadLocalState* tls,
                uint32_t volumeId,
                const openvdb::Vec3d& p,
                Interpolation mode) const
        {
            tls->mGeomTls->mStatistics.incCounter(STATS_DENSITY_GRID_SAMPLES);
            uint32_t threadIdx = tls->mThreadIdx;
            unsigned samplerIdx = threadIdx * mVolumeIdCount + (mVolumeIdToSamplerId.at(volumeId));
            return scene_rdl2::math::Color(mThreadSamplers[samplerIdx].extinc(
                p, (amorphous::InterpolationMethod)mode));
        }

        scene_rdl2::math::Color albedo(mcrt_common::ThreadLocalState* tls,
                uint32_t volumeId,
                const openvdb::Vec3d& p,
                Interpolation mode) const
        {
            tls->mGeomTls->mStatistics.incCounter(STATS_COLOR_GRID_SAMPLES);
            uint32_t threadIdx = tls->mThreadIdx;
            unsigned samplerIdx = threadIdx * mVolumeIdCount + (mVolumeIdToSamplerId.at(volumeId));
            amorphous::Rgba aVal = mThreadSamplers[samplerIdx].color(
                p, (amorphous::InterpolationMethod)mode);
            return scene_rdl2::math::Color(aVal.r(), aVal.g(), aVal.b());
        }

        scene_rdl2::math::Color incand(mcrt_common::ThreadLocalState* tls,
                uint32_t volumeId,
                const openvdb::Vec3d& p,
                Interpolation mode) const
        {
            tls->mGeomTls->mStatistics.incCounter(STATS_EMISSION_GRID_SAMPLES);
            uint32_t threadIdx = tls->mThreadIdx;
            unsigned samplerIdx = threadIdx * mVolumeIdCount + (mVolumeIdToSamplerId.at(volumeId));
            amorphous::Rgba aVal = mThreadSamplers[samplerIdx].incand(
                p, (amorphous::InterpolationMethod)mode);
            return scene_rdl2::math::Color(aVal.r(), aVal.g(), aVal.b());
        }

        openvdb::FloatGrid::Ptr getDensityGrid()
        {
            return openvdb::ConstPtrCast<openvdb::FloatGrid>(
                mScatterParams.extinctionGrid());
        }

        amorphous::BaseGridCPtr getIncandGrid() const
        {
            return mScatterParams.incandescenceGrid();
        }

        void loadVelocityGrid(const std::string& vdbFilePath,
                const std::string& velocityGridName,
                openvdb::VectorGrid::Ptr& velocityGrid)
        {
            std::stringstream msg;
            velocityGrid.reset();
            amorphous::File::readVelocityGrid(velocityGrid, velocityGridName, vdbFilePath, msg);
        }

        const ScatterSampler* getThreadSampler(uint32_t threadIdx)
        {
            return &mThreadSamplers[threadIdx];
        }

        size_t getMemory() const
        {
            size_t mem = 0;
            // The ExtincGrid is shared with the mTopology grid
            // We only need to count the color grid and incand grid,
            // because they are not stored anywhere else in VdbVolume.
            if (mScatterParams.hasColor()) {
                mem += mScatterParams.colorGrid()->memUsage();
            }
            if (mScatterParams.hasIncandescence()) {
                mem += mScatterParams.incandescenceGrid()->memUsage();
            }
            if (mMasterSampler) {
                mem += sizeof(amorphous::SScatterSampler);
            }

            mem += scene_rdl2::util::getVectorElementsMemory(mThreadSamplers);

            return mem;
        }

        ScatterParams mScatterParams;
        std::unique_ptr<ScatterSampler> mMasterSampler;
        std::vector<ScatterSampler> mThreadSamplers;
        std::unordered_map<uint32_t, uint32_t> mVolumeIdToSamplerId;
        unsigned mVolumeIdCount;
    };

    bool mUseAmorphousSampler;
    AmorphousSampler mAmorphousSampler;
};


} // namespace internal
} // namespace geom
} // namespace moonray


