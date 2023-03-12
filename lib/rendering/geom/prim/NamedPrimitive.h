// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file NamedPrimitive.h
/// $Id$
///

#pragma once

#ifndef GEOM_NAMEDPRIMITIVE_HAS_BEEN_INCLUDED
#define GEOM_NAMEDPRIMITIVE_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/geom/prim/ShadowLinking.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/geom/LayerAssignmentId.h>

namespace moonray {
namespace geom {
namespace internal {

/// A NamedPrimitive is a primitive that stores name but no local2parent xform.
/// Most primitive types derive from this class.
class NamedPrimitive : public moonray::geom::internal::Primitive
{
public:
    /// Constructor / Destructor
    NamedPrimitive(LayerAssignmentId&& layerAssignmentId):
        mAttributes(), mLayerAssignmentId(std::move(layerAssignmentId))
    {
        resetShadowLinking();
    }

    virtual ~NamedPrimitive()
    {
        resetShadowLinking();
    }

    NamedPrimitive(const NamedPrimitive&) = delete;
    NamedPrimitive &operator=(const NamedPrimitive&) = delete;
    NamedPrimitive& operator=(NamedPrimitive&&) = delete;

    /// Modify / Access the state associated with this primitive
    void setName(const std::string &name)
    {
        mName = name;
    }

    const std::string &getName() const
    {
        return mName;
    }

    virtual size_t getMemory() const override
    {
        size_t mem = sizeof(NamedPrimitive) - sizeof(Primitive) + Primitive::getMemory();
        if (mAttributes) {
            mem += mAttributes->getMemory();
        }
        return mem;
    }

    // the reason to make this method virtual is some primitive types store
    // primitive attribute outside of mAttributes that need their own
    // customized check
    virtual bool hasAttribute(shading::AttributeKey key) const
    {
        return getAttributes()->hasAttribute(key);
    }

    virtual void initVolumeSampleInfo(VolumeSampleInfo* info,
            const Vec3f& rayOrg, const Vec3f& rayDir, float time,
            const scene_rdl2::rdl2::VolumeShader* volumeShader,
            int volumeId) const override;

    virtual bool hasAssignment(int assignmentId) const override;

    bool hasDisplacementAssignment(const scene_rdl2::rdl2::Layer* layer) const;

    bool hasSurfaceAssignment(const scene_rdl2::rdl2::Layer* layer) const;

    bool hasVolumeAssignment(const scene_rdl2::rdl2::Layer* layer) const;

    void getUniqueAssignmentIds(
            std::unordered_set<int>& uniqueAssignmentIds) const;

    // The set of primitive attributes associated with this primitive.
    // This is generally set by the procedural at primitive construction
    // time and queried at Shading time by Interpolators to assing to
    // Intersection records.
    shading::Attributes* getAttributes() const
    {
        return mAttributes.get();
    }

    void setAttributes(std::unique_ptr<shading::Attributes>&& attributes)
    {
        mAttributes = std::move(attributes);
    }

    void resetShadowLinking();

    void createShadowLinking(int casterId, bool complementReceiverSet)
    {
        mShadowLinkings[casterId] = new ShadowLinking(complementReceiverSet);
    }

    void addShadowLinkedLight(int casterId, const scene_rdl2::rdl2::Light* light)
    {
        MNRY_ASSERT(mShadowLinkings[casterId]);
        mShadowLinkings[casterId]->addLight(light);
    }

    void addShadowLinkedGeom(int casterId, int receiverId)
    {
        MNRY_ASSERT(mShadowLinkings[casterId]);
        mShadowLinkings[casterId]->addReceiver(receiverId);
    }

    const ShadowLinking* getShadowLinking(int casterId) const
    {
        return mShadowLinkings.at(casterId);
    }

    bool hasShadowLinking(int casterId) const
    {
        return mShadowLinkings.find(casterId) != mShadowLinkings.end();
    }

    const std::unordered_map<int, ShadowLinking *>& getShadowLinkings() const
    {
        return mShadowLinkings;
    }

    bool hasShadowLinking(const scene_rdl2::rdl2::Layer* layer) const;

    virtual int getIntersectionAssignmentId(int primID) const = 0;

protected:
    std::unique_ptr<shading::Attributes> mAttributes;
    std::string mName;
    LayerAssignmentId mLayerAssignmentId;
    std::unordered_map<int, ShadowLinking *> mShadowLinkings;
};

} // namespace internal
} // namespace geom
} // namespace moonray

#endif /* GEOM_NAMEDPRIMITIVE_HAS_BEEN_INCLUDED */

