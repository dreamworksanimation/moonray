// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeAssignmentTable.h
///

#pragma once

#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/ShadowLinking.h>
#include <moonray/rendering/geom/prim/VolumeRegions.h>

#include <scene_rdl2/scene/rdl2/Layer.h>
#include <scene_rdl2/scene/rdl2/Material.h>
#include <scene_rdl2/scene/rdl2/ShadowSet.h>
#include <scene_rdl2/scene/rdl2/VolumeShader.h>

#include <unordered_map>
#include <vector>

namespace moonray {
namespace geom {
namespace internal {

class Instance;
class Primitive;

// This class offers a fast way to track volume IDs through
// instance primitive intersections.
class VolumeIdFSM
{
public:
    VolumeIdFSM();

    // Adds a transition sequence to the FSM.  A transition sequence
    // is a sequence of instance primitives ending in a volumeId
    void add(const std::vector<const geom::internal::Instance *> &sequence, int volumeId);

    // Transition to instance "to"
    // Returns next valid state (>= 0) based on this transition.  If the
    // transition is invalid, return -1.
    int transition(int state, const geom::internal::Instance *to) const;

    // Is this an end state?
    bool isLeaf(int state) const;

    // What is the volumeId of this state?
    // If the state is a leaf state it will be the volumeId otherwise
    // it will be -1.
    int getVolumeId(int state) const;

private:
    struct Node
    {
        using InstanceId = const geom::internal::Instance *;
        using NodeIdx = int;
        using TransitionMap = std::unordered_map<InstanceId, NodeIdx>;

        Node(): mVolumeId(-1) {}

        int addTransition(InstanceId to, NodeIdx nodeIdx);
        NodeIdx findTransition(InstanceId to) const;

        // valid next nodes
        TransitionMap mTransitions;

        // If this node is a leaf node, mTransitions will be
        // empty and this value is the volumeId
        int mVolumeId;
    };

    std::vector<Node> mNodes;
};


/// This class serves the functionality of book keeping the number of
/// unique volume assignments (part/material). It offers fast lookup
/// from layer assignment id to volume id (0 to n-1 where n = number of
/// unique volume assignments) and vise versa.
class VolumeAssignmentTable
{
public:
    void initLookupTable(const scene_rdl2::rdl2::Layer* layer);

    void setFeatureSizes(geom::internal::Primitive* prim,
                         const std::vector<int>& volumeIds,
                         const float fs) const;

    const scene_rdl2::rdl2::VolumeShader* lookupWithVolumeId(int volumeId) const {
        return mVolumeShaders[mVolumeIdToAssignmentId[volumeId]];
    }

    const scene_rdl2::rdl2::VolumeShader* lookupWithAssignmentId(int assignmentId) const {
        return mVolumeShaders[assignmentId];
    }

    const ShadowLinking& lookupShadowLinkingWithVolumeId(int volumeId) const {
        return mShadowLinkings[mVolumeIdToAssignmentId[volumeId]];
    }

    size_t getVolumeCount() const { return mVolumeIdToAssignmentId.size(); }

    const std::vector<int> &getVolumeIds(int assignmentId) const {
        return mAssignmentIdToVolumeIds[assignmentId];
    }

    int getAssignmentId(int volumeId) const {
        return mVolumeIdToAssignmentId[volumeId];
    }

    int getVisibilityMask(int volumeId) const {
        return mVisibilityMasks[volumeId];
    }

    // For a given volumeId return true if volumeId is an instance.
    // If true, then evaluate the instance xform at time, returning the
    // result in primToRender.
    bool evalInstanceXform(int volumeId, float time, Mat43 &primToRender) const;

    // A state machine is used to compute volume ids from a sequence of
    // instance primitive intersections
    const VolumeIdFSM &getInstanceVolumeIds() const { return mInstanceVolumeIds; }

private:
    // In the case of instanced volumes, a layer assignment might
    // correspond to multiple volume ids.  Each instance primitive requires
    // its own volume id.
    std::vector<std::vector<int>> mAssignmentIdToVolumeIds;

    // A given volume id can correspond to only a single assignment id.
    // In the case of instancing, the assignment id is the assignment id
    // of the reference primitive.
    std::vector<int> mVolumeIdToAssignmentId;

    // Given a volume id, what is its visibility mask?
    std::vector<int> mVisibilityMasks;

    // Based on assignment id
    std::vector<const scene_rdl2::rdl2::VolumeShader*> mVolumeShaders;

    // Based on assignment id
    std::vector<ShadowLinking> mShadowLinkings;

    // Given a volume id, there might be a motion xform stack due to
    // instancing
    std::vector<std::vector<MotionTransform>> mInstanceXforms;

    VolumeIdFSM mInstanceVolumeIds;
};

} // namespace internal
} // namespace geom
} // namespace moonray

