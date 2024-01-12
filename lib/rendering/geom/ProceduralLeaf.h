// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ProceduralLeaf.h
/// $Id$
///

#ifndef GEOM_PROCEDURALLEAF_HAS_BEEN_INCLUDED
#define GEOM_PROCEDURALLEAF_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/SharedPrimitive.h>

#include <moonray/rendering/bvh/shading/Xform.h>

#include <tbb/concurrent_vector.h>

namespace moonray {
namespace geom {

//----------------------------------------------------------------------------

///
/// @class ProceduralLeaf ProceduralLeaf.h <rendering/geom/ProceduralLeaf.h>
/// @brief A ProceduralLeaf can be sub-classed by procedurals that generate
/// primitives.
/// 
class ProceduralLeaf : public Procedural
{
public:
    /// Constructor / Destructor
    explicit ProceduralLeaf(const State &state);

    ~ProceduralLeaf();

    ProceduralLeaf(const ProceduralLeaf &other) = delete;

    const ProceduralLeaf &operator=(const ProceduralLeaf &other) = delete;

    virtual bool isLeaf() const override {  return true;  }

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each Primitive in this procedural,
    ///     apply a visitor function.
    virtual void forEachPrimitive(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each static Primitive, apply a visitor function.
    virtual void forEachStatic(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each dynamic Primitive, apply a visitor function.
    virtual void forEachDynamic(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each deformable Primitive, apply a visitor function.
    virtual void forEachDeformable(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// @remark Involves locking! Avoid in performance critical section
    /// @brief The count of primitives in this procedural
    virtual size_type getPrimitivesCount() const override;

    /// @brief Remove all the primitives in this procedural
    virtual void clear() override;

    /// @brief Whether the generated primitives in this procedural can be
    ///     referenced by other geometry procedural through getReference()
    virtual bool isReference() const override;

    /// @brief Query the SharedPrimitive stored in this procedural when the
    ///     owner geometry is referenced by other geometry
    virtual const std::shared_ptr<SharedPrimitive>& getReference() const override;

protected:
    /// Add a Primitive to the scene
    void addPrimitive(std::unique_ptr<Primitive> p,
            const MotionBlurParams& motionBlurParams,
            const shading::XformSamples& parent2render);

    /// request the primitive container be at least enough to
    /// contain n primitives
    void reservePrimitive(Procedural::size_type n);

private:
    /// transform the procedural generated primitives to be shared primitives,
    /// which can be referenced by other geometry procedural
    virtual void transformToReference() override;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};


//----------------------------------------------------------------------------

} // namespace geom
} // namespace moonray

#endif /* GEOM_PROCEDURALLEAF_HAS_BEEN_INCLUDED */

