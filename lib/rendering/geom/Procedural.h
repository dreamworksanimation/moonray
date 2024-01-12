// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Procedural.h
/// $Id$
///

#pragma once

#ifndef GEOM_PROCEDURAL_HAS_BEEN_INCLUDED
#define GEOM_PROCEDURAL_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/SharedPrimitive.h>
#include <moonray/rendering/geom/State.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/ProceduralContext.h>

#include <moonray/rendering/bvh/shading/Xform.h>
#include <scene_rdl2/render/util/Ref.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_set.h>

namespace moonray {
namespace geom {

struct GeometryStatistics {
    GeometryStatistics(): mFaceCount {0}, mMeshVertexCount {0},
        mCurvesCount {0}, mCVCount {0}, mInstanceCount {0} {}

    tbb::atomic<Primitive::size_type> mFaceCount;
    tbb::atomic<Primitive::size_type> mMeshVertexCount;
    tbb::atomic<Primitive::size_type> mCurvesCount;
    tbb::atomic<Primitive::size_type> mCVCount;
    tbb::atomic<Primitive::size_type> mInstanceCount;
};

//----------------------------------------------------------------------------
class PrimitiveVisitor;

// warning #1684: conversion from pointer to
// same-sized integral type (potential portability problem)
#pragma warning push
#pragma warning disable 1684
class SharedPtrHash
{
public:
    size_t operator()(const std::shared_ptr<SharedPrimitive>& k) const {
        return reinterpret_cast<size_t>(k.get());
    }
};
#pragma warning pop
typedef tbb::concurrent_unordered_set<std::shared_ptr<SharedPrimitive>, SharedPtrHash> SharedPrimitiveSet;

///
/// @class Procedural Procedural.h <rendering/geom/Procedural.h>
/// @brief This class defined the basic interface for Procedurals, which
/// responsibility is to manage procedural generation of child Procedural
/// and/or Primitives.
/// 
class Procedural : private scene_rdl2::util::RefCount<Procedural>
{
    // expose private method transformToReference to internal renderer
    friend class internal::PrimitivePrivateAccess;
public:
    typedef size_t size_type;
    /// Constructor / Destructor
    // A Procedural always captures the State when it is created.
    // A procedural can modify its child procedural / primitive' state during
    // update()
    Procedural(const State &state): mDeformed(false), mState(state) {}
    virtual ~Procedural() = default;
    Procedural(const Procedural &other) = delete;
    const Procedural &operator=(const Procedural &other) = delete;

    /// Whether this procedural a leaf procedural
    virtual bool isLeaf() const = 0;

    /// Implementation must compute and return a conservative estimate of
    /// its bounding box in render space
    /// (according to state.getProc2Parent * parent2render transform)
    /// The default implementation will return an invalid bound, in which case
    /// the procedural will be immediately generated and its bound computed by
    /// the renderer based on the generated sub-procedurals / primitives.
    virtual BBox3f bound(const GenerateContext &generateContext,
                         const Mat43 &parent2render) const
    {  return BBox3f(scene_rdl2::util::False);  }
    
    /// Implementation should create child procedurals / geometric Primitives
    virtual void generate(const GenerateContext &generateContext,
            const shading::XformSamples &parent2render) = 0;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief Implementation should update child procedurals /
    ///     geometric Primitives
    virtual void update(const UpdateContext &updateContext,
            const shading::XformSamples &parent2render) {}

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each Primitive in this procedural,
    ///     apply a visitor function.
    virtual void forEachPrimitive(PrimitiveVisitor& visitor,
            bool parallel = true) = 0;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each static Primitive, apply a visitor function.
    virtual void forEachStatic(PrimitiveVisitor& visitor,
            bool parallel = true) = 0;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each dynamic Primitive, apply a visitor function.
    virtual void forEachDynamic(PrimitiveVisitor& visitor,
            bool parallel = true) = 0;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief For each deformable Primitive, apply a visitor function.
    virtual void forEachDeformable(PrimitiveVisitor& visitor,
            bool parallel = true) = 0;

    /// @remark Involves locking! Avoid in performance critical sections
    /// @brief The count of primitives in this procedural
    virtual size_type getPrimitivesCount() const = 0;

    /// @brief Get the primitives memory usage in bytes
    size_type getMemory();

    GeometryStatistics getStatistics() const;

    /// @brief Get transform from procedural space to parent space
    const Mat43& getProc2Parent() const { return mState.getProc2Parent(); }

    /// @brief Remove all the primitives in this procedural
    virtual void clear() = 0;

    /// @brief Whether the generated primitives in this procedural can be
    ///     referenced by other geometry procedural through getReference()
    virtual bool isReference() const = 0;

    /// @brief Query the SharedPrimitive stored in this procedural when the
    ///     owner geometry is referenced by other geometry
    virtual const std::shared_ptr<SharedPrimitive>& getReference() const = 0;

    // TODO: add pure virtual API for:
    // - Iterating over all generated sub-procedurals

    bool deformed() const { return mDeformed; }
    void resetDeformed() { mDeformed = false; }

protected:
    /// Modify / Access the state associated with this procedural
    void setState(const State &state) { mState = state; }
    const State &getState() const { return mState; }
    bool mDeformed;

private:
    /// transform the procedural generated primitives to be shared primitives,
    /// which can be referenced by other geometry procedural
    virtual void transformToReference() = 0;

private:
    // A Procedural always captures the State when it is created.
    // A procedural can modify its child procedural / primitive' state during
    // update()
    State mState;
};

typedef tbb::concurrent_vector<Procedural *> ProceduralArray;

//----------------------------------------------------------------------------

} // namespace geom
} // namespace moonray

#endif /* GEOM_PROCEDURAL_HAS_BEEN_INCLUDED */

