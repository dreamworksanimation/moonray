// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Instance.h
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/SharedPrimitive.h>
#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>

#include <moonray/rendering/bvh/shading/InstanceAttributes.h>
#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {
namespace internal {

class Instance : public moonray::geom::internal::Primitive {
public:
    explicit Instance(const shading::XformSamples& xform,
            std::shared_ptr<SharedPrimitive> reference,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        mLocal2Parent(xform, 0, 0), mReference(reference)
    {
        if (!primitiveAttributeTable.empty()) {
            mAttributes.reset(new shading::InstanceAttributes(
                std::move(primitiveAttributeTable)));
        }
    }

    ~Instance()
    {
        mBVHHandle.reset();
    }

    virtual PrimitiveType getType() const override
    {
        return INSTANCE;
    }

    virtual size_t getMemory() const override
    {
        size_t mem = sizeof(Instance) - sizeof(Primitive) + Primitive::getMemory();
        if (mAttributes) {
            mem += mAttributes->getMemory();
        }
        return mem;
    }

    virtual void initVolumeSampleInfo(VolumeSampleInfo* info,
            const Vec3f& rayOrg, const Vec3f& rayDir, float time,
            const scene_rdl2::rdl2::VolumeShader* volumeShader,
            int volumeId) const override
    {
        // This method must be defined because it is a pure virtual method
        // in the base class, but it should never be called.  Volume sample
        // initialization should be called for the referenced volume
        // primitives.
        MNRY_ASSERT(false);
    }

    virtual size_t getMotionSamplesCount() const override
    {
        return mReference->getPrimitive()->getMotionSamplesCount();
    }

    virtual bool canIntersect() const override
    {
        return true;
    }

    virtual RTCBoundsFunction getBoundsFunction() const override;

    virtual RTCIntersectFunctionN getIntersectFunction() const override;

    virtual RTCOccludedFunctionN getOccludedFunction() const override;

    virtual BBox3f computeAABB() const override;
    virtual BBox3f computeAABBAtTimeStep(int timeStep) const override;

    virtual bool hasAssignment(int assignmentId) const override;

    RTCScene getReferenceScene() const
    {
        return static_cast<RTCScene>(
            geom::internal::PrimitivePrivateAccess::getBVHScene(*mReference));
    }

    const std::shared_ptr<SharedPrimitive>& getReference() const
    {
        return mReference;
    }

    void initializeXform()
    {
        mLocal2Parent.initialize();
    }

    void appendXform(const shading::XformSamples& xform,
            float shutterOpenDelta, float shutterCloseDelta)
    {
        mLocal2Parent.appendXform(xform);
        mLocal2Parent.setSampleInterval(shutterOpenDelta, shutterCloseDelta);
    }

    const MotionTransform& getLocal2Parent() const
    {
        return mLocal2Parent;
    }

    shading::InstanceAttributes* getAttributes()
    {
        return mAttributes.get();
    }

    const shading::InstanceAttributes* getAttributes() const
    {
        return mAttributes.get();
    }

    // We keep track of Instance pointers so we can apply the attributes defined
    // on those instances.  There is only room in the ray for a few pointers though,
    // so the maximum depth we can keep track of for attribute handling is limited.
    // Instancing itself has no depth limit, but attributes on instances below
    // sMaxInstanceAttributesDepth will be ignored.
    static const int sMaxInstanceAttributesDepth = 4;

private:
    MotionTransform mLocal2Parent;
    std::shared_ptr<SharedPrimitive> mReference;
    std::unique_ptr<shading::InstanceAttributes> mAttributes;
};

} // namespace internal
} // namespace geom
} // namespace moonray

