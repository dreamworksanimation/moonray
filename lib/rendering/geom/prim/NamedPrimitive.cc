// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "NamedPrimitive.h"

#include <moonray/rendering/geom/prim/VolumeSampleInfo.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

namespace moonray {
namespace geom {
namespace internal {

void
NamedPrimitive::initVolumeSampleInfo(VolumeSampleInfo* info,
        const Vec3f& rayOrg, const Vec3f& rayDir, float time,
        const scene_rdl2::rdl2::VolumeShader* volumeShader,
        int volumeId) const
{
    // if we have an instance feature id for this volume id, use it
    const float featureSize = getInstanceFeatureSize(volumeId);
    info->initialize(volumeShader, rayOrg, rayDir, featureSize,
        (mRdlGeometry->getVisibilityMask() & scene_rdl2::rdl2::SHADOW) != 0,
        /* isVDB = */ false);
}

bool
NamedPrimitive::hasAssignment(int assignmentId) const
{
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        return assignmentId == mLayerAssignmentId.getConstId();
    } else {
        const auto& assignmentIds = mLayerAssignmentId.getVaryingId();
        for (auto id : assignmentIds) {
            if (id == assignmentId) {
                return true;
            }
        }
        return false;
    }
}

bool
NamedPrimitive::hasDisplacementAssignment(const scene_rdl2::rdl2::Layer* layer) const
{
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        int assignmentId = mLayerAssignmentId.getConstId();
        if (assignmentId == -1) {
            return false;
        }
        return (layer->lookupDisplacement(assignmentId) != nullptr);
    } else {
        const auto& assignmentIds = mLayerAssignmentId.getVaryingId();
        for (auto id : assignmentIds) {
            if (id == -1) {
                continue;
            }
            if (layer->lookupDisplacement(id) != nullptr) {
                return true;
            }
        }
        return false;
    }

}

bool
NamedPrimitive::hasSurfaceAssignment(const scene_rdl2::rdl2::Layer* layer) const
{
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        int assignmentId = mLayerAssignmentId.getConstId();
        if (assignmentId == -1) {
            return false;
        }
        const scene_rdl2::rdl2::Material* material = layer->lookupMaterial(assignmentId);
        return (material != nullptr);
    } else {
        const auto& assignmentIds = mLayerAssignmentId.getVaryingId();
        for (auto id : assignmentIds) {
            if (id == -1) {
                continue;
            }
            const scene_rdl2::rdl2::Material* material = layer->lookupMaterial(id);
            if (material != nullptr) {
                return true;
            }
        }
        return false;
    }
}

bool
NamedPrimitive::hasVolumeAssignment(const scene_rdl2::rdl2::Layer* layer) const
{
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        int assignmentId = mLayerAssignmentId.getConstId();
        if (assignmentId == -1) {
            return false;
        }
        return (layer->lookupVolumeShader(assignmentId) != nullptr);
    } else {
        const auto& assignmentIds = mLayerAssignmentId.getVaryingId();
        for (auto id : assignmentIds) {
            if (id == -1) {
                continue;
            }
            if (layer->lookupVolumeShader(id) != nullptr) {
                return true;
            }
        }
        return false;
    }
}

void
NamedPrimitive::resetShadowLinking()
{
    for (auto it = mShadowLinkings.begin(); it != mShadowLinkings.end(); ++it) {
        if (it->second != nullptr) {
            delete it->second;
            it->second = nullptr;
        }
    }
    mShadowLinkings.clear();
    std::unordered_set<int> casterIds;
    getUniqueAssignmentIds(casterIds);
    for (int casterId : casterIds) {
        if (casterId < 0) {
            continue;
        }
        mShadowLinkings[casterId] = nullptr;
    }
}

bool
NamedPrimitive::hasShadowLinking(const scene_rdl2::rdl2::Layer* layer) const
{
    std::unordered_set<int> casterIds;
    getUniqueAssignmentIds(casterIds);
    for (int casterId : casterIds) {
        if (casterId < 0) {
            continue;
        }
        const scene_rdl2::rdl2::ShadowSet* shadowSet = layer->lookupShadowSet(casterId);
        if (shadowSet) {
            return true;
        }
    }
    return false;
}

void
NamedPrimitive::getUniqueAssignmentIds(
        std::unordered_set<int>& uniqueAssignmentIds) const
{
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        uniqueAssignmentIds.insert(mLayerAssignmentId.getConstId());
    } else {
        const auto& assignmentIds = mLayerAssignmentId.getVaryingId();
        for (auto id : assignmentIds) {
            uniqueAssignmentIds.insert(id);
        }
    }
}

} // namespace internal
} // namespace geom
} // namespace moonray


