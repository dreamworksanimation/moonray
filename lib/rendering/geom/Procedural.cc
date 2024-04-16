// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Procedural.cc
/// $Id$
///

#include "Procedural.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>

#include <tbb/atomic.h>

#include <numeric>

namespace moonray {
namespace geom {

class PrimitiveMemoryAccumulator : public PrimitiveVisitor
{
public:
    PrimitiveMemoryAccumulator(
            tbb::atomic<Primitive::size_type>& usage,
            SharedPrimitiveSet& sharedPrimitives,
            bool inPrimitiveGroup = false) :
        mUsage(usage), mSharedPrimitives(sharedPrimitives),
        mInPrimitiveGroup(inPrimitiveGroup) {}

    virtual void visitPrimitive(Primitive& p) override {
        // if the primitive is in PrimitiveGroup, the memory usage of this
        // Primitive has been taken into account
        if (!mInPrimitiveGroup) {
            mUsage += p.getMemory();
        }
    }

    virtual void visitPrimitiveGroup(PrimitiveGroup& pg) override {
        if (!mInPrimitiveGroup) {
            mUsage += pg.getMemory();
        }
        // make sure the primitives in this PrimitiveGroup doesn't get added
        // twice. The reason we recursively travel down is that we need to
        // collect all SharedPrimitive that get instanced around
        PrimitiveMemoryAccumulator accumulator(mUsage, mSharedPrimitives, true);
        pg.forEachPrimitive(accumulator);
    }

    virtual void visitInstance(Instance& i) override {
        if (!mInPrimitiveGroup) {
            mUsage += i.getMemory();
        }
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            PrimitiveMemoryAccumulator accumulator(mUsage, mSharedPrimitives);
            ref->getPrimitive()->accept(accumulator);
        }
    }

private:
    tbb::atomic<Primitive::size_type>& mUsage;
    SharedPrimitiveSet& mSharedPrimitives;
    bool mInPrimitiveGroup;
};

class StatisticsAccumulator : public PrimitiveVisitor
{
public:
    StatisticsAccumulator(
            GeometryStatistics& geometryStatistics,
            SharedPrimitiveSet& sharedPrimitives) :
        mGeometryStatistics(geometryStatistics),
        mSharedPrimitives(sharedPrimitives) {}

    virtual void visitPrimitive(Primitive& p) override {
    }

    virtual void visitPolygonMesh(PolygonMesh& p) override {
        mGeometryStatistics.mFaceCount += p.getFaceCount();
        mGeometryStatistics.mMeshVertexCount += p.getVertexCount();
    }

    virtual void visitSubdivisionMesh(SubdivisionMesh& s) override {
        mGeometryStatistics.mFaceCount += s.getSubdivideFaceCount();
        mGeometryStatistics.mMeshVertexCount += s.getSubdivideVertexCount();
    }

    virtual void visitCurves(Curves& c) override {
        mGeometryStatistics.mCurvesCount += c.getCurvesCount();
        const auto& curvesVertexCount = c.getCurvesVertexCount();
        mGeometryStatistics.mCVCount += std::accumulate(curvesVertexCount.begin(),
            curvesVertexCount.end(), 0);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t) override {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitPrimitiveGroup(PrimitiveGroup& pg) override {
        pg.forEachPrimitive(*this);
    }

    virtual void visitInstance(Instance& i) override {
        ++mGeometryStatistics.mInstanceCount;
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            ref->getPrimitive()->accept(*this);
        }
    }

private:
    GeometryStatistics& mGeometryStatistics;
    SharedPrimitiveSet& mSharedPrimitives;
};


Procedural::size_type
Procedural::getMemory()
{
    tbb::atomic<Primitive::size_type> usage {0u};
    SharedPrimitiveSet sharedPrimitives;
    PrimitiveMemoryAccumulator accumulator(usage, sharedPrimitives);
    forEachPrimitive(accumulator);
    return usage;
}

GeometryStatistics
Procedural::getStatistics() const
{
    GeometryStatistics geometryStatistics;

    SharedPrimitiveSet sharedPrimitives;
    StatisticsAccumulator accumulator(geometryStatistics, sharedPrimitives);
    const_cast<Procedural *>(this)->forEachPrimitive(accumulator);

    return geometryStatistics;
}


} // namespace geom
} // namespace moonray


