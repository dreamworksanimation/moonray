// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file VolumeAssignmentTable.cc

#include "VolumeAssignmentTable.h"

#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>

#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>

#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

// One of the key concepts to understand in this code is that of
// assignment id vs volume id.
//
// An assignment id is basically a row number in a layer.  For example,
//     Layer("/layer") {
//         -- assignment id 0
//         {OpenVdbGeometry("/vdbObj", "", undef(), LightSet("/lightset"), undef(), AmorphousVolume("/vdbVol")},
//
//         -- assignment id 1
//         {BoxGeometry("/myBox", DwaBaseMaterial("/boxMtl"), LightSet("/lightset"), undef(), undef()},
//
//         -- assignment id 2
//         {SphereGeometry("/sphere", undef(), LightSet("/lightset"), undef(), BaseVolume("sphereVol")},
//     }
//
// The volume id is the basic identifier used by the volume renderer to handle
// things like entry and exit events and lookup shaders. We are currently limited
// to 512 unique volume ids.  Any volume primitive that we want thought of as a unique
// volume for purposes of handling overlapping regions will have a unique volume id.  Continuing
// the above example
//
//     Layer("/layer") {
//         -- assignment id 0, volume id 0
//         {OpenVdbGeometry("/vdbObj", "", undef(), LightSet("/lightset"), undef(), AmorphousVolume("/vdbVol")},
//
//         -- assignment id 1
//         {BoxGeometry("/myBox", DwaBaseMaterial("/boxMtl"), LightSet("/lightset"), undef(), undef()},
//
//         -- assignment id 2, volume id 1
//         {SphereGeometry("/sphere", undef(), LightSet("/lightset"), undef(), BaseVolume("sphereVol")},
//     }
//
// Now instancing adds a wrinkle. The volume shader assignment
// should come from the reference, and the reference object will not render directly.
// But since we'll have a unique volume per instance each instance primitive will need
// its own unique volume id. So for example if we instance the vdbObj with 3 instance
// primitives we get
//
//     InstanceGeometry("/instancer") {
//         ["positions"] = { Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(2, 0, 0) },
//         ["references"] = { OpenVdbGeometry("/vdbObj") }
//     }
//
//     Layer("/layer") {
//         -- assignment id 0, since this is now a reference it doesn't get a volume id
//         -- references are not rendered (that is just our convention)
//         {OpenVdbGeometry("/vdbObj", "", undef(), LightSet("/lightset"), undef(), AmorphousVolume("/vdbVol")},
//
//         -- assignment id 1
//         {BoxGeometry("/myBox", DwaBaseMaterial("/boxMtl"), LightSet("/lightset"), undef(), undef()},
//
//         -- assignment id 2, volume id 0
//         {SphereGeometry("/sphere", undef(), LightSet("/lightset"), undef(), BaseVolume("sphereVol")},
//
//         -- assignment id 3, volume id 1, 2, 3 (shaders come from assignment id 0)
//         {InstanceGeometry("/intancer", undef(), LightSet("/lightset"), undef(), undef()}
//     }
//
// The last line in the layer requires some explanation. Since we have 3 unique volume
// primitives, we need three unique volume ids for them. In this case it is volume id 1, 2,
// and 3. Note also that since the reference object "/vdbObj" is not rendered, it doesn't get
// a volume id. Even though the instanceGeometry line has an assignment id of "3". That is not
// the assignment id we are really interested in using when rendering volume id 1, 2, 3. Really
// those volume ids correspond to the assignment id of the reference object (in this case
// assignment id 0).
//

// A word about visibility masks:
// A geometry object can specify a visibility mask.  This visbility mask applies to
// the assigned volume, both for direct rendering and when used as an emission
// for hard surfaces.  We can look up the visibilty mask of a volume based on volume id.
// What is the rule for instancing?  Both the reference object and the instance geometry
// objects can specify visibility masks.  The rule (for hard surfaces and volumes) is that
// the mask is the logical and of all relevant visibility masks.

namespace moonray {
namespace geom {
namespace internal {

// We need a way to go from a shared primitive to a layer
// assignment id and visibility mask
struct VolumeAssignment
{
    int32_t mAId;            // assignment id in layer
    int32_t mVisibilityMask; // visibility mask from shared primitive's geometry
};
using ReferenceAssignments = std::unordered_map<const SharedPrimitive *, VolumeAssignment>;

// We need a way to go from an instance primitive to a visibility mask
using InstanceMasks = std::unordered_map<const geom::Instance *, int32_t>;
using MaskPair = std::pair<const geom::Instance *, int32_t>;

namespace {

// Determine if a layer assignment is the reference of an instance geometry.
bool
isSharedReference(const scene_rdl2::rdl2::Layer *layer, int32_t id)
{
    const scene_rdl2::rdl2::TraceSet::GeometryPartPair p = layer->lookupGeomAndPart(id);
    const scene_rdl2::rdl2::Geometry *g = p.first;
    MNRY_ASSERT(g != nullptr);
    const geom::Procedural *proc = g->getProcedural();
    MNRY_ASSERT(proc != nullptr);
    if (proc->isLeaf()) {
        const geom::ProceduralLeaf *pleaf = static_cast<const geom::ProceduralLeaf *>(proc);
        if (pleaf->isReference()) {
            return true;
        }
    }
    return false;
}

void
evaluateInstanceXform(const std::vector<MotionTransform> &instanceXforms, float time, Mat43 &result)
{
    result = Mat43(scene_rdl2::math::one);
    for (const MotionTransform &xform : instanceXforms) {
        if (xform.isStatic()) {
            result = xform.getStaticXform() * result;
        } else {
            result = xform.eval(time) * result;
        }
    }
}

class AssignInstanceMasks : public geom::PrimitiveVisitor
{
public:
    AssignInstanceMasks(InstanceMasks &instanceMasks, int32_t mask):
        mInstanceMasks(instanceMasks),
        mMask(mask)
    {
    }

    void visitPrimitiveGroup(geom::PrimitiveGroup &pg) override
    {
        pg.forEachPrimitive(*this, /* doParallel = */ false);
    }

    void visitInstance(geom::Instance &i) override
    {
        mInstanceMasks.insert(MaskPair(&i, mMask));
    }

private:
    InstanceMasks &mInstanceMasks;
    int32_t mMask;
};

class AssignVolumeInstances : public geom::PrimitiveVisitor
{
public:
    AssignVolumeInstances(const ReferenceAssignments &refAssignments,
                          const InstanceMasks &instanceVisibilityMasks,
                          const int32_t maxVolumeCount,
                          std::vector<std::vector<int32_t>> &assignmentIdToVolumeIds,
                          std::vector<int32_t> &volumeIdToAssignmentId,
                          std::vector<int32_t> &visibilityMasks,
                          std::vector<std::vector<MotionTransform>> &instanceXforms,
                          geom::internal::VolumeIdFSM &volumeIdFSM,
                          int32_t &volumeCount):
        mRefAssignments(refAssignments),
        mInstanceVisibilityMasks(instanceVisibilityMasks),
        mMaxVolumeCount(maxVolumeCount),
        mAssignmentIdToVolumeIds(assignmentIdToVolumeIds),
        mVolumeIdToAssignmentId(volumeIdToAssignmentId),
        mVisibilityMasks(visibilityMasks),
        mInstanceXforms(instanceXforms),
        mVolumeIdFSM(volumeIdFSM),
        mVolumeCount(volumeCount)
    {
    }

    void visitPrimitiveGroup(geom::PrimitiveGroup &pg) override
    {
        // supports multi-level instancing
        pg.forEachPrimitive(*this, /* isParallel = */ false);
    }

    void visitInstance(geom::Instance &i) override
    {
        // push this instance onto the instance sequence stacks
        geom::internal::Primitive *pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&i);
        geom::internal::Instance *pInst = static_cast<geom::internal::Instance *>(pImpl);
        mInstanceSequence.push_back(pInst);
        mInstanceXform.push_back(pInst->getLocal2Parent());
        InstanceMasks::const_iterator maskItr = mInstanceVisibilityMasks.find(&i);
        MNRY_ASSERT(maskItr != mInstanceVisibilityMasks.end());
        mInstanceVisibilityMask.push_back(maskItr->second);

        const SharedPrimitive *ref = i.getReference().get();
        MNRY_ASSERT(ref != nullptr);
        ReferenceAssignments::const_iterator itr = mRefAssignments.find(ref);
        if (itr != mRefAssignments.end()) {
            if (mVolumeCount < mMaxVolumeCount) {
                // This instance directly references an object with a volume assignment, so we
                // need to assign a volume
                const int32_t assignmentId = itr->second.mAId;
                const int32_t volumeId = mVolumeCount;
                mAssignmentIdToVolumeIds[assignmentId].push_back(volumeId);

                // map this volumeId to its assignmentId
                mVolumeIdToAssignmentId[volumeId] = assignmentId;

                // keep the instance transform stack for this volume id
                mInstanceXforms[volumeId] = mInstanceXform;

                // compute the visibility mask for this volume
                mVisibilityMasks[volumeId] = itr->second.mVisibilityMask;
                for (int32_t m : mInstanceVisibilityMask) {
                    mVisibilityMasks[volumeId] &= m;
                }

                // add this instance sequence to the volumeIdFSM
                mVolumeIdFSM.add(mInstanceSequence, volumeId);
            }
            ++mVolumeCount;

        } else {
            // this might be a primitive group due to multi-level instancing
            ref->getPrimitive()->accept(*this);
        }

        // pop the instance stacks
        mInstanceSequence.pop_back();
        mInstanceXform.pop_back();
        mInstanceVisibilityMask.pop_back();
    }

private:
    const ReferenceAssignments &mRefAssignments;
    const InstanceMasks &mInstanceVisibilityMasks;
    const int32_t mMaxVolumeCount;
    std::vector<std::vector<int32_t>> &mAssignmentIdToVolumeIds;
    std::vector<int32_t> &mVolumeIdToAssignmentId;
    std::vector<int32_t> &mVisibilityMasks;
    std::vector<std::vector<MotionTransform>> &mInstanceXforms;
    geom::internal::VolumeIdFSM &mVolumeIdFSM;
    int32_t &mVolumeCount;

    // Keep track of visibility mask due to instancing
    std::vector<int32_t> mInstanceVisibilityMask;

    // Keep track of transforms due to instancing
    std::vector<MotionTransform> mInstanceXform;

    // Keep track of the instance stack to assign volume ids
    std::vector<const geom::internal::Instance *> mInstanceSequence;
};

}

//-----------------------------------------------------------------------------
//-- VolumeIdFSM
//-----------------------------------------------------------------------------
// We need an efficient way to assign volume ids when using instances.
// As an example of the complication consider this setup:
//
//   sphere     = OpenVdbGeomtry("/sphere") {}
//   i0 = InstanceGeometry("/i0") {
//       ["positions"] = { i0p0, i0p1 i0p2 },
//       ["references"] = { sphere }
//   }
//   i1 = InstanceGeometry("/i1") {
//       ["positions"] = { i1p0, i1p1 },
//       ["references"] = { i0 }
//   }
//   i2 = InstanceGeometry("/i2") {
//       ["positions"] = { i2p0, i2p1, i2p2 }
//       ["references"] = { i1 }
//
//
//   The resulting instance hierarchy and volume id assignments look like
//   i2p0
//       i1p0
//           i0p0 = vId 0
//           i0p1 = vId 1
//           i0p2 = vId 2
//       i1p1
//           i0p0 = vId 3
//           i0p1 = vId 4
//           i0p2 = vId 5
//   i2p1
//       i1p0
//           i0p0 = vId 6
//           i0p1 = vId 7
//           i0p2 = vId 8
//       i1p1
//           i0p0 = vId 9
//           i0p1 = vId 10
//           i0p2 = vId 11
//   i2p2
//       i1p0
//           i0p0 = vId 12
//           i0p1 = vId 13
//           i0p2 = vId 14
//       i1p1
//           i0p0 = vId 15
//           i0p1 = vId 16
//           i0p2 = vId 17
//
//  Note that we have 18 unique volume ids in this hierarchy, but only 3 instance
//  primitives (i0p0, i0p1, and i0p2) that actually reference a volume.  When tracing
//  rays the volume id depends on how the traversed hierarchy reached the instance.
//  For example, if the instance intersection recursion was i2p0->i1p0->i0p0 then we use
//  volumeId 0.  But if the traversal was i2p2->i1p1->i0p0 then we use volumeId 15.
//
//  This implies that we are going to need to maintain an instance history as we recurse
//  in the instance intersection calls.
//
//  To do this, we'll use a finite state machine to record transition events.

VolumeIdFSM::VolumeIdFSM()
{
    // start with a single empty node
    mNodes.push_back(Node());
}

void
VolumeIdFSM::add(const std::vector<const geom::internal::Instance *> &sequence,
                 int volumeId)
{
    int state = 0; // start
    for (const geom::internal::Instance *inst : sequence) {
        int newState = mNodes[state].findTransition(inst);
        if (newState == -1) {
            // need to break off a new node
            mNodes.push_back(Node());
            newState = mNodes[state].addTransition(inst, mNodes.size() - 1);
        }
        state = newState;
    }
    // we should now be at a leaf node with an unset volumeId
    MNRY_ASSERT(!sequence.empty());
    MNRY_ASSERT(state > 0 && state < static_cast<int>(mNodes.size()));
    MNRY_ASSERT(isLeaf(state));
    Node &n = mNodes[state];
    MNRY_ASSERT(n.mVolumeId == -1); // hasn't been set yet
    n.mVolumeId = volumeId;
}

int
VolumeIdFSM::transition(int state, const geom::internal::Instance *to) const
{
    MNRY_ASSERT(state >= 0 && state < static_cast<int>(mNodes.size()));
    const Node &n = mNodes[state];
    return n.findTransition(to);
}

bool
VolumeIdFSM::isLeaf(int state) const
{
    MNRY_ASSERT(state >= 0 && state < static_cast<int>(mNodes.size()));
    const Node &n = mNodes[state];
    return n.mTransitions.size() == 0;
}

int
VolumeIdFSM::getVolumeId(int state) const
{
    MNRY_ASSERT(state >= 0 && state < static_cast<int>(mNodes.size()));
    const Node &n = mNodes[state];
    return n.mVolumeId;
}

int
VolumeIdFSM::Node::addTransition(InstanceId to, NodeIdx nodeIdx)
{
    MNRY_ASSERT(mTransitions.find(to) == mTransitions.end());
    mTransitions.insert(std::pair<InstanceId, NodeIdx>(to, nodeIdx));
    return nodeIdx;
}

VolumeIdFSM::Node::NodeIdx
VolumeIdFSM::Node::findTransition(InstanceId to) const
{
    TransitionMap::const_iterator itr = mTransitions.find(to);
    if (itr == mTransitions.end()) {
        // either a leaf node or an invalid transition
        return -1;
    }
    return itr->second;
}


//-----------------------------------------------------------------------------
//-- VolumeAssignmentTable methods
//-----------------------------------------------------------------------------
void
VolumeAssignmentTable::initLookupTable(const scene_rdl2::rdl2::Layer *layer)
{
    const int32_t assignmentCount = layer->getAssignmentCount();
    mAssignmentIdToVolumeIds.clear();
    mAssignmentIdToVolumeIds.resize(assignmentCount);
    const int maxVolumeCount = VolumeRegions().getMaxRegionsCount();
    mVolumeIdToAssignmentId = std::vector<int32_t>(maxVolumeCount, -1);
    mVisibilityMasks = std::vector<int32_t>(maxVolumeCount, scene_rdl2::rdl2::VisibilityType::ALL_VISIBLE);
    mInstanceXforms = std::vector<std::vector<MotionTransform>>(maxVolumeCount);
    mInstanceVolumeIds = VolumeIdFSM();
    int32_t volumeCount = 0;

    // shared primitives (instance targets) that have volume assignments
    ReferenceAssignments refAssignments;
    for (int32_t aId = 0; aId < layer->getAssignmentCount(); ++aId) {
        const scene_rdl2::rdl2::Geometry *g = layer->lookupGeomAndPart(aId).first;
        const geom::Procedural *proc = g->getProcedural();
        if (proc->isLeaf()) {
            const geom::ProceduralLeaf *pleaf = dynamic_cast<const geom::ProceduralLeaf *>(proc);
            if (pleaf->isReference()) {
                const SharedPrimitive *ref = pleaf->getReference().get();
                if (ref->getHasVolumeAssignment()) {
                    refAssignments[ref] = { aId, g->getVisibilityMask() };
                }
            }
        }
    }

    // If we have shared primitives, setup the bookkeeping required for volume instances
    if (!refAssignments.empty()) {
        // First compute the visibility masks.  This is an input to the next loop.
        InstanceMasks instanceVisibilityMasks;
        for (int32_t aId = 0; aId < layer->getAssignmentCount(); ++aId) {
            const scene_rdl2::rdl2::Geometry *g = layer->lookupGeomAndPart(aId).first;
            geom::Procedural *proc = const_cast<geom::Procedural *>(g->getProcedural());
            if (proc->isLeaf()) {
                const int32_t mask = g->getVisibilityMask();
                AssignInstanceMasks aim(instanceVisibilityMasks, mask);
                const geom::ProceduralLeaf *pleaf = dynamic_cast<const geom::ProceduralLeaf *>(proc);
                if (pleaf->isReference()) {
                    pleaf->getReference()->getPrimitive()->accept(aim);
                } else {
                    proc->forEachPrimitive(aim, /* doParallel = */ false);
                }
            }
        }

        // Handle the rest of the assignments
        for (int32_t aId = 0; aId < layer->getAssignmentCount(); ++aId) {
            const scene_rdl2::rdl2::TraceSet::GeometryPartPair p = layer->lookupGeomAndPart(aId);
            const scene_rdl2::rdl2::Geometry *g = p.first;
            geom::Procedural *proc = const_cast<geom::Procedural *>(g->getProcedural());
            AssignVolumeInstances avi(refAssignments,
                                      instanceVisibilityMasks,
                                      maxVolumeCount,
                                      mAssignmentIdToVolumeIds,
                                      mVolumeIdToAssignmentId,
                                      mVisibilityMasks,
                                      mInstanceXforms,
                                      mInstanceVolumeIds,
                                      volumeCount);
            proc->forEachPrimitive(avi, /* doParallel = */ false);
        }
    }

    // Now handle the non shared, non instances
    for (int32_t aId = 0; aId < assignmentCount; ++aId) {
        const bool isVolume = layer->lookupVolumeShader(aId);
        const bool isShared = isSharedReference(layer, aId);
        if (isVolume && !isShared) {
            if (volumeCount < maxVolumeCount) {
                mAssignmentIdToVolumeIds[aId].push_back(volumeCount);
                mVolumeIdToAssignmentId[volumeCount] = aId;
                mVisibilityMasks[volumeCount] =
                    layer->lookupGeomAndPart(aId).first->getVisibilityMask();
            }
            ++volumeCount;
        }
    }

    // reset volume shaders
    mVolumeShaders = std::vector<const scene_rdl2::rdl2::VolumeShader *>(assignmentCount, nullptr);

    // reset shadow linking lookup table
    mShadowLinkings.clear();
    mShadowLinkings.resize(assignmentCount);

    // Fill in volume shaders and shadow linking
    for (int32_t aId = 0; aId < assignmentCount; ++aId) {
        if (!mAssignmentIdToVolumeIds.empty()) {
            mVolumeShaders[aId] = layer->lookupVolumeShader(aId);
            const scene_rdl2::rdl2::ShadowSet* shadowSet = layer->lookupShadowSet(aId);
            // add lights to shadow set
            if (shadowSet) {
                ShadowLinking& shadowLinking = mShadowLinkings[aId];
                const scene_rdl2::rdl2::SceneObjectVector &lights = shadowSet->getLights();
                for (auto it = lights.begin(); it != lights.end(); it++) {
                    shadowLinking.addLight(static_cast<const scene_rdl2::rdl2::Light *>(*it));
                }
            }
        }
    }

    if (volumeCount > maxVolumeCount) {
        layer->error("The number of volumes (including instanced volumes) "
            "reaches ", volumeCount,
            ", which exceeds the maximum limit of ", maxVolumeCount);
    }

    // Reduce volumeId indexed arrays to the actual number
    // of volumes.  volumeCount could be much greater than maxVolumeCount,
    // so we don't want to blindly resize these arrays to volumeCount.
    if (volumeCount < maxVolumeCount) {
        mVolumeIdToAssignmentId.resize(volumeCount);
        mVisibilityMasks.resize(volumeCount);
        mInstanceXforms.resize(volumeCount);
    }
}

void
VolumeAssignmentTable::setFeatureSizes(geom::internal::Primitive* prim,
                                       const std::vector<int>& volumeIds,
                                       const float fs) const
{
    for (const int volumeId : volumeIds) {
        // store an adjusted feature size in the primitive, if the volume's volume
        // is scaled by the instance transform
        Mat43 primToRender;
        evaluateInstanceXform(mInstanceXforms[volumeId], /* time = */ 0, primToRender);
        const float det = primToRender.l.det();
        MNRY_ASSERT(det > 0.f);
        if (!scene_rdl2::math::isEqual(det, 1.0f)) {
            const float newFs = fs * cbrt(det);
            prim->setInstanceFeatureSize(volumeId, newFs);
        }
    }
}

bool
VolumeAssignmentTable::evalInstanceXform(int volumeId, float time, Mat43 &primToRender) const
{
    if (mInstanceXforms[volumeId].empty()) return false;
    evaluateInstanceXform(mInstanceXforms[volumeId], time, primToRender);
    return true;
}

} // namespace internal
} // namespace geom
} // namespace moonray

