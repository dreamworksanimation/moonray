// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeRayState.h
///

#pragma once

#include <moonray/rendering/geom/prim/VolumeAssignmentTable.h>
#include <moonray/rendering/geom/prim/VolumeRegions.h>
#include <moonray/rendering/geom/prim/VolumeSampleInfo.h>
#include <moonray/rendering/geom/prim/VolumeTransition.h>

namespace moonray {
namespace geom {
namespace internal {

/// This class stores all the volume intervals along a ray during the
/// BVH traversal stage. Each thread owns "one" VolumeRayState so
/// the data it stores is overwritten each time we do another ray intersection
/// test in this thread
class VolumeRayState
{
public:
    static constexpr int ORIGIN_VOLUME_INIT = -2; // We don't have volume along the ray
    static constexpr int ORIGIN_VOLUME_EMPTY = -1; // We have volume but ray origin is outside volume

    VolumeRayState() : mIntervalCount(0), mVolumeAssignmentTable(nullptr) {}

    finline void initializeVolumeLookup(
            const VolumeAssignmentTable* volumeAssignmentTable) {
        mVolumeAssignmentTable = volumeAssignmentTable;
        mVolumeSampleInfo.resize(mVolumeAssignmentTable->getVolumeCount());
    }

    finline void resetState(float tEnd, bool estimateInScatter) {
        mIntervalCount = 0;
        mVolumeRegions.reset();
        mVisitedVolumes.reset();
        mTEnd = tEnd;
        mOriginVolumeId = ORIGIN_VOLUME_INIT;
        mEstimateInScatter = estimateInScatter;
    }

    finline void addInterval(const Primitive* primitive, float t, int volumeId, bool isEntry, float *tRenderSpace = nullptr) {
        MNRY_ASSERT(mIntervalCount < sMaxIntervalCount,
            "volume intersections along the ray exceeds "
            "thread local storage capacity");
        mVolumeIntervals[mIntervalCount++] =
            VolumeTransition(primitive, t, volumeId, isEntry, tRenderSpace);
    }

    finline size_t getIntervalCount() const { return mIntervalCount; }

    finline const VolumeRegions& getCurrentVolumeRegions() const {
        return mVolumeRegions;
    }

    finline void turnOn(int volumeId, const Primitive* primitive) {
        mVolumeRegions.turnOn(volumeId, primitive);
    }

    finline void turnOff(int volumeId) {
        mVolumeRegions.turnOff(volumeId);
    }

    finline bool isOn(int volumeId) const {
        return mVolumeRegions.isOn(volumeId);
    }

    finline bool isOff(int volumeId) const {
        return !mVolumeRegions.isOn(volumeId);
    }

    finline bool isVisited(int volumeId) const {
        return mVisitedVolumes.isOn(volumeId);
    }

    finline void setVisited(int volumeId, const Primitive* primitive) {
        mVisitedVolumes.turnOn(volumeId, primitive);
    }

    finline int getVisitedVolumeRegionsCount() const {
        return mVisitedVolumes.getRegionsCount();
    }

    finline int getVisitedVolumeIds(int* volumeIds) const {
        return mVisitedVolumes.getVolumeIds(volumeIds);
    }

    finline const VolumeAssignmentTable* getVolumeAssignmentTable() const {
        return mVolumeAssignmentTable;
    }

    finline VolumeTransition* getVolumeIntervals() {
        return mVolumeIntervals.data();
    }

    finline const std::vector<VolumeSampleInfo>& getVolumeSampleInfo() const {
        return mVolumeSampleInfo;
    }

    finline VolumeSampleInfo& getVolumeSampleInfo(int volumeId) {
        return mVolumeSampleInfo[volumeId];
    }

    finline int getVolumeId(int assignmentId, int instanceState) const {
        // instancing case
        const VolumeIdFSM &fsm = mVolumeAssignmentTable->getInstanceVolumeIds();
        if (instanceState > 0 && fsm.isLeaf(instanceState)) {
            const int vId = fsm.getVolumeId(instanceState);
            MNRY_ASSERT(vId >= 0);
            return vId;
        }

        // non-instancing
        const std::vector<int> &vIds = mVolumeAssignmentTable->getVolumeIds(assignmentId);
        MNRY_ASSERT(vIds.size() == 1);
        return vIds[0];
    }

    finline int getAssignmentId(int volumeId) const {
        return mVolumeAssignmentTable->getAssignmentId(volumeId);
    }

    finline float getTEnd() const { return mTEnd; }


    inline void setOriginVolumeId(int id, float distance) { mOriginVolumeId = id; mOriginVolumeDistance = distance; }
    inline int getOriginVolumeId() const { return mOriginVolumeId; }
    inline float getOriginVolumeDistance() const { return mOriginVolumeDistance; }

    inline bool getEstimateInScatter() const { return mEstimateInScatter; }

private:
    size_t mIntervalCount;
    const VolumeAssignmentTable* mVolumeAssignmentTable;
    VolumeRegions mVolumeRegions;
    VolumeRegions mVisitedVolumes;
    std::vector<VolumeSampleInfo> mVolumeSampleInfo;
    float mTEnd;
    static constexpr size_t sMaxIntervalCount = 4096;
    std::array<VolumeTransition, sMaxIntervalCount> mVolumeIntervals;

    // volumeId of ray origin position. 3 possible conditions.
    // ORIGIN_VOLUME_INIT : initial condition. origin volume is not computed yet.
    // ORIGIN_VOLUME_EMPTY : no volume at ray origin
    // 0 or positive value : volumeId at ray origin position.
    int mOriginVolumeId;

    // The distance of the origin volume to the origin.  Used for finding the closest
    // origin volume during BVH traversal / intersection filters.
    float mOriginVolumeDistance;

    // true  : This is a case when this class is used for PathIntegrator::estimateInScatteringSourceTerm().
    //         i.e. light transmittance computation for volume.
    // false : all other situations.
    bool mEstimateInScatter;
};

} // namespace internal
} // namespace geom
} // namespace moonray

