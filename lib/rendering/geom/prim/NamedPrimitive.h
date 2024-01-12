// Copyright 2023-2024 DreamWorks Animation LLC
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

    template<typename InterpolatorType>
    static void setExplicitAttributes(const InterpolatorType& interpolator,
                                      const shading::Attributes& primitiveAttributes,
                                      shading::Intersection& intersection)
    {
        if (primitiveAttributes.isSupported(shading::StandardAttributes::sNormal)) {
            scene_rdl2::math::Vec3f N;
            interpolator.interpolate(shading::StandardAttributes::sNormal,
                reinterpret_cast<char*>(&N));
            N = N.normalize();
            intersection.setAttribute(shading::StandardAttributes::sNormal, N);
            intersection.setN(N);
        }

        if (primitiveAttributes.isSupported(shading::StandardAttributes::sdPds)) {
            scene_rdl2::math::Vec3f dPds;
            interpolator.interpolate(shading::StandardAttributes::sdPds,
                reinterpret_cast<char*>(&dPds));
            intersection.setAttribute(shading::StandardAttributes::sdPds, dPds);
        }

        if (primitiveAttributes.isSupported(shading::StandardAttributes::sdPdt)) {
            Vec3f dPdt;
            interpolator.interpolate(shading::StandardAttributes::sdPdt,
                reinterpret_cast<char*>(&dPdt));
            intersection.setAttribute(shading::StandardAttributes::sdPdt, dPdt);
        }
    }

    static bool getExplicitAttributes(const shading::Attributes& primitiveAttributes,
                                      const shading::Intersection& intersection,
                                      scene_rdl2::math::Vec3f& N,
                                      scene_rdl2::math::Vec3f& dPds,
                                      scene_rdl2::math::Vec3f& dPdt)
    {
        if (!primitiveAttributes.isSupported(shading::StandardAttributes::sExplicitShading) &&
            !intersection.isProvided(shading::StandardAttributes::sExplicitShading)) {
            return false;
        }

        bool hasExplicitNormal = false;
        if (intersection.isProvided(shading::StandardAttributes::sNormal)) {
            N = intersection.getAttribute<Vec3f>(shading::StandardAttributes::sNormal);
            N = normalize(N);
            hasExplicitNormal = true;
        }

        bool hasExplicitDPds = false;
        if (intersection.isProvided(shading::StandardAttributes::sdPds)) {
            dPds = intersection.getAttribute<Vec3f>(shading::StandardAttributes::sdPds);
            hasExplicitDPds = true;
        }

        bool hasExplicitDPdt = false;
        if (intersection.isProvided(shading::StandardAttributes::sdPdt)) {
            dPdt = intersection.getAttribute<Vec3f>(shading::StandardAttributes::sdPdt);
            hasExplicitDPdt = true;
        }

        return hasExplicitNormal && hasExplicitDPds && hasExplicitDPdt;
    }

    std::unique_ptr<shading::Attributes> mAttributes;
    std::string mName;
    LayerAssignmentId mLayerAssignmentId;
    std::unordered_map<int, ShadowLinking *> mShadowLinkings;
};

} // namespace internal
} // namespace geom
} // namespace moonray

#endif /* GEOM_NAMEDPRIMITIVE_HAS_BEEN_INCLUDED */

