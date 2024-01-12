// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ProceduralNode.h
/// $Id$
///

#ifndef GEOM_PROCEDURALNODE_HAS_BEEN_INCLUDED
#define GEOM_PROCEDURALNODE_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/Procedural.h>

namespace moonray {
namespace geom {


//----------------------------------------------------------------------------
class PrimitiveVisitor;

/// @todo this class is not actually in use at the moment. It will be used
///     for nested procedural support that a procedural node can generate
///     sub-procedural to create primitives (and sub-procedural potentially)
/// @class ProceduralNode ProceduralNode.h <rendering/geom/ProceduralNode.h>
/// @brief A ProceduralNode can be sub-classed by procedurals that subdivide
/// themselves into sub-procedurals.
/// 
class ProceduralNode : public Procedural
{
public:
    /// Constructor / Destructor
	explicit ProceduralNode(const State &state);

    ~ProceduralNode();

    ProceduralNode(const ProceduralNode &other) = delete;

    const ProceduralNode &operator=(const ProceduralNode &other) = delete;

    virtual bool isLeaf() const override {  return false;  }

    /// Iterate over procedurals
    const ProceduralArray &getChildren() const;

    /// For each Primitive in this procedural, apply a visitor function.
    virtual void forEachPrimitive(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// For each static Primitive, apply a visitor function.
    virtual void forEachStatic(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// For each dynamic Primitive, apply a visitor function.
    virtual void forEachDynamic(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// For each deformative Primitive, apply a visitor function.
    virtual void forEachDeformable(PrimitiveVisitor& visitor,
            bool parallel = true) override;

    /// The count of primitives in this procedural
    virtual Procedural::size_type getPrimitivesCount() const override;

    virtual void clear() override;

    /// @brief Whether the generated primitives in this procedural can be
    ///     referenced by other geometry procedural through getReference()
    virtual bool isReference() const override;

    /// @brief Query the SharedPrimitive stored in this procedural when the
    ///     owner geometry is referenced by other geometry
    virtual const std::shared_ptr<SharedPrimitive>& getReference() const override;

    // TODO: add implementation for:
    // - Iterating over all generated sub-procedurals
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

#endif /* GEOM_PROCEDURALNODE_HAS_BEEN_INCLUDED */

