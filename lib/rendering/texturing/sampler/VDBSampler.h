// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VDBSampler.h
///

#pragma once
#pragma warning(disable:177) // openvdb.h #177: member was declared but never referenced

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <sstream>
#include <string>

namespace moonray {
namespace texture {

class VDBSampler
{
public:
    enum class Interpolation
    {
        Point,
        Box,
        Quadratic
    };

    virtual bool getIsActive(const uint32_t threadIdx,
                             const openvdb::Vec3d& pos) = 0;

    VDBSampler() =default;
    virtual ~VDBSampler() {}
};

template<typename GridT>
class TypedVDBSampler : public VDBSampler
{
private:
    typedef typename GridT::ValueType ValueT;

public:
    TypedVDBSampler(const GridT& grid)
    {
        size_t threadCount = mcrt_common::getMaxNumTLS();
        mSamplers.reserve(threadCount);
        for (size_t i = 0; i < threadCount; ++i) {
            mSamplers.emplace_back(grid);
        }
    }

    ~TypedVDBSampler() override {}

    bool getIsActive(const uint32_t threadIdx,
                     const openvdb::Vec3d& pos) override
    {
        const openvdb::Vec3d p = mSamplers[threadIdx].mGrid.worldToIndex(pos);
        const openvdb::Coord coord(p.x(), p.y(), p.z());
        return mSamplers[threadIdx].mAccessor.isValueOn(coord);
    }

    ValueT sample(const uint32_t threadIdx, const openvdb::Vec3d& pos,
                  Interpolation mode) const
    {
        switch (mode) {
        case Interpolation::Point :
            return mSamplers[threadIdx].evalPoint(pos);
        case Interpolation::Box :
            return mSamplers[threadIdx].evalBox(pos);
        case Interpolation::Quadratic :
            return mSamplers[threadIdx].evalQuadratic(pos);
        default:
            return mSamplers[threadIdx].evalPoint(pos);
        }
    }

private:
    struct GridSampler
    {
        typedef typename GridT::ConstAccessor ConstAccessor;
        typedef openvdb::tools::GridSampler<ConstAccessor, openvdb::tools::PointSampler> PointSampler;
        typedef openvdb::tools::GridSampler<ConstAccessor, openvdb::tools::BoxSampler> BoxSampler;
        typedef openvdb::tools::GridSampler<ConstAccessor, openvdb::tools::QuadraticSampler> QuadraticSampler;

        GridSampler(const GridT& grid):
            mGrid(grid),
            mAccessor(grid.getConstAccessor()),
            mPointSampler(mAccessor, grid.transform()),
            mBoxSampler(mAccessor, grid.transform()),
            mQuadraticSampler(mAccessor, grid.transform())
        {
        }

        ValueT evalPoint(const openvdb::Vec3d& pos) const
        {
            return mPointSampler.wsSample(pos);
        }

        ValueT evalBox(const openvdb::Vec3d& pos) const
        {
            return mBoxSampler.wsSample(pos);
        }

        ValueT evalQuadratic(const openvdb::Vec3d& pos) const
        {
            return mQuadraticSampler.wsSample(pos);
        }

        const GridT &mGrid;
        ConstAccessor mAccessor;
        PointSampler mPointSampler;
        BoxSampler mBoxSampler;
        QuadraticSampler mQuadraticSampler;
    };

    std::vector<GridSampler> mSamplers;
};

} // namespace texture
} // namespace moonray

