// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file OpenVdbSampler.h
///

#pragma once

#include <moonray/rendering/texturing/sampler/VDBSampler.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <memory>
#include <string>

namespace moonray {
namespace shading {

class TLState;

class OpenVdbSampler
{
public:
    // Keep this in sync with VDBSampler::Interpolation.
    // This tracks VDBSampler::Interpolation so that
    // lib/rendering/texturing/sampler/VDBSampler.h
    // does not need to be exposed to the users of this class.
    enum class Interpolation
    {
        Point,
        Box,
        Quadratic,
    };

    OpenVdbSampler();
    ~OpenVdbSampler();

    // Initialize the object and attempt to load grid. This
    // function must be called first. If initialization fails
    // this object is left in an invalid state.
    // ------------------------------------------------------------------------
    // 'filename' - The name of the .vdb file to load.
    // 'gridName' - If blank, the first grid in the .vdb file will be used. If
    //              non-bank, there must be a grid within the .vdb file whose
    //              name matches the name given. If there are multiple grid's in
    //              the .vdb with the same name, they can be individually
    //              accessed by appending an index operator to the grid name,
    //              eg. density[1].
    //  'render2world' - A ptr to a matrix responsible for transforming the
    //              coordinate from render space to world space. If null is
    //              passed, the coordinate is assumed to be given in world space.
    //  'grid2world' - A ptr to a matrix responsible for appending an additional
    //              transform to the grid, in adddition to that stored in the
    //              .vdb file. If null is passed, no additional transformation
    //              is applied.
    // 'errorMsg' - A string that will contain an error message if
    //              initialization fails.
    // ------------------------------------------------------------------------
    //
    // Returns true if the grid is successfully loaded, otherwise false with
    // a (hopefully) informative error message.
    //
    bool initialize(const std::string &filename,
                    const std::string &gridName,
                    const scene_rdl2::math::Mat4d *render2world,
                    const scene_rdl2::math::Mat4d *grid2world,
                    std::string &errorMsg);

    // 'pos' is given in render space or world space, depending on
    // how this class was initialized.  See above.
    bool getIsActive(shading::TLState* tls,
                     const scene_rdl2::math::Vec3f &pos) const;

    // 'pos' is given in render space or world space, depending on
    // how this class was initialized.  See above.
    scene_rdl2::math::Color sample(shading::TLState* tls,
                       const scene_rdl2::math::Vec3f &pos,
                       Interpolation interp) const;

    const std::string& getFileName() const { MNRY_ASSERT_REQUIRE(mIsInitialized); return mFileName; }
    const std::string& getGridName() const { MNRY_ASSERT_REQUIRE(mIsInitialized); return mGridName; }
    const std::string& getValueType() const { MNRY_ASSERT_REQUIRE(mIsInitialized); return mValueType; }

private:
    bool mIsInitialized;
    openvdb::GridBase::Ptr mGrid;
    std::unique_ptr<moonray::texture::VDBSampler> mSampler;
    const scene_rdl2::math::Mat4d* mR2W;

    // for reporting
    std::string mFileName;
    std::string mGridName;
    std::string mValueType;
};

}
}

