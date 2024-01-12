// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file OpenVdbSampler.cc
///

#include "OpenVdbSampler.h"

#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include <openvdb/openvdb.h>

#include <sstream>

using moonray::texture::VDBSampler;
using moonray::texture::TypedVDBSampler;

namespace {

openvdb::GridBase::Ptr
readGrid(const std::string &fileName, const std::string &gridName,
         std::string &foundGrid, std::string &errorMsg)
{
    openvdb::GridBase::Ptr grid;

    if (fileName.empty()) {
        errorMsg = "no .vdb filename specified";
        return grid;
    }

    openvdb::initialize();
    openvdb::io::File file(fileName);
    try {
        file.open();
    } catch (const openvdb::IoError &e) {
        errorMsg = e.what();
        return grid;
    }

    std::string whichGrid = gridName;

    if (gridName.empty()) {
        // use first grid in the file
        openvdb::io::File::NameIterator firstName = file.beginName();
        if (firstName != file.endName()) {
            whichGrid = firstName.gridName();
        }
    } else if (!file.hasGrid(gridName)) {
        std::ostringstream os;
        os << "'" << fileName << "' does not contain a grid named '" << gridName << "'";
        file.close();
        errorMsg = os.str();
        return grid;
    }

    grid = file.readGrid(whichGrid);
    file.close();

    foundGrid = whichGrid;

    return grid;
}

} // anonymous namespace

namespace moonray {
namespace shading {

OpenVdbSampler::OpenVdbSampler() :
    mIsInitialized(false),
    mR2W(nullptr)
{
}

OpenVdbSampler::~OpenVdbSampler() {}

bool
OpenVdbSampler::initialize(const std::string &fileName,
                           const std::string &gridName,
                           const scene_rdl2::math::Mat4d *render2world,
                           const scene_rdl2::math::Mat4d *grid2world,
                           std::string &errorMsg)
{
    mIsInitialized = false;

    // try/catch to catch exceptions that could be thrown from
    // the openvdb library itself, to prevent users of this class
    // from needing to try/catch to prevent moonray from crashing.
    try {
        std::string foundGrid;
        mGrid = readGrid(fileName, gridName, foundGrid, errorMsg);

        if (!mGrid) {
            // errorMsg is already set
            return false;
        }

        mValueType = mGrid->valueType();

        // Append any additional transform provided by the user
        if (grid2world) {
            const scene_rdl2::math::Vec4d &r0 = grid2world->row0();
            const scene_rdl2::math::Vec4d &r1 = grid2world->row1();
            const scene_rdl2::math::Vec4d &r2 = grid2world->row2();
            const scene_rdl2::math::Vec4d &r3 = grid2world->row3();
            const openvdb::Mat4R xform(
                r0[0], r0[1], r0[2], r0[3],
                r1[0], r1[1], r1[2], r1[3],
                r2[0], r2[1], r2[2], r2[3],
                r3[0], r3[1], r3[2], r3[3]
            );
            mGrid->transform().postMult(xform);
        }

        // create the appropriate TypedVDBSampler for this grid type
        if (mGrid->isType<openvdb::FloatGrid>()) {
            openvdb::FloatGrid::ConstPtr g = openvdb::gridConstPtrCast<openvdb::FloatGrid>(mGrid);
            mSampler = fauxstd::make_unique<TypedVDBSampler<openvdb::FloatGrid>>(*g);
        } else if (mGrid->isType<openvdb::VectorGrid>()) {
            openvdb::VectorGrid::ConstPtr g = openvdb::gridConstPtrCast<openvdb::VectorGrid>(mGrid);
            mSampler = fauxstd::make_unique<TypedVDBSampler<openvdb::VectorGrid>>(*g);
        } else {
            // TODO: provide more information to user
            // eg. filename, gridname, which gridtype is it that is unsupported...
            errorMsg = "unsupported grid type";
            return false;
        }

        mFileName = fileName;
        mGridName = foundGrid;
        mR2W = render2world;
        mIsInitialized = true;
    } catch (const std::exception &e) {
        errorMsg = e.what();
    }

    return mIsInitialized;
}

bool
OpenVdbSampler::getIsActive(shading::TLState* tls,
                            const scene_rdl2::math::Vec3f &pos) const
{
    MNRY_ASSERT_REQUIRE(mIsInitialized);

    openvdb::Vec3d p;
    if (mR2W) {
        scene_rdl2::math::Vec3f pw = transformPoint(*mR2W, pos);
        p[0] = pw[0];
        p[1] = pw[1];
        p[2] = pw[2];
    } else {
        p[0] = pos[0];
        p[1] = pos[1];
        p[2] = pos[2];
    }

    return mSampler->getIsActive(tls->mThreadIdx, p);
}

scene_rdl2::math::Color 
OpenVdbSampler::sample(shading::TLState* tls,
                       const scene_rdl2::math::Vec3f &pos,
                       const Interpolation interp) const
{
    MNRY_ASSERT_REQUIRE(mIsInitialized);

    scene_rdl2::math::Color result(0.f, 0.f, 0.f);

    // convert to VDBSampler::Interpolation
    const VDBSampler::Interpolation interpolation =
        static_cast<VDBSampler::Interpolation>(interp);

    openvdb::Vec3d p;
    if (mR2W) {
        scene_rdl2::math::Vec3f pw = transformPoint(*mR2W, pos);
        p[0] = pw[0];
        p[1] = pw[1];
        p[2] = pw[2];
    } else {
        p[0] = pos[0];
        p[1] = pos[1];
        p[2] = pos[2];
    }

    // cast to appropriate type and sample
    if (mGrid->isType<openvdb::FloatGrid>()) {
        TypedVDBSampler<openvdb::FloatGrid>* sampler =
            static_cast<TypedVDBSampler<openvdb::FloatGrid>*>(mSampler.get());
        const float val = sampler->sample(tls->mThreadIdx, p, interpolation);
        result = scene_rdl2::math::Color(val, val, val);
    } else if (mGrid->isType<openvdb::VectorGrid>()) {
        TypedVDBSampler<openvdb::VectorGrid>* sampler =
            static_cast<TypedVDBSampler<openvdb::VectorGrid>*>(mSampler.get());
        const openvdb::Vec3f val = sampler->sample(tls->mThreadIdx, p, interpolation);
        result = scene_rdl2::math::Color(val[0], val[1], val[2]);
    } else {
        // Programming error. This object should have thrown an
        // exception in the ctor if the grid type is unsupported
        MNRY_ASSERT_REQUIRE(false, "only FloatGrid and VectorGrid types currently supported");
    }

    return result;
}

}
}


