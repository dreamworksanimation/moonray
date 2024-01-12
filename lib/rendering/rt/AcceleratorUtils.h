// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>

namespace moonray {
namespace rt {

// This class is used to drill down into a multi-level instancing
// hierarchy looking for volume or surface assignments on a leaf
// node reference.  It assumes all leaf node references have already
// had their assignments set.
class GetAssignments : public geom::PrimitiveVisitor
{
public:
    GetAssignments(bool &hasVolume, bool &hasSurface):
        mHasVolume(hasVolume),
        mHasSurface(hasSurface)
    {
    }

    void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override
    {
        bool isParallel = false;
        pg.forEachPrimitive(*this, isParallel);
    }

    void visitInstance(geom::Instance& i) override
    {
        geom::SharedPrimitive *ref = i.getReference().get();
        mHasVolume |= ref->getHasVolumeAssignment();
        mHasSurface |= ref->getHasSurfaceAssignment();
        // drill down in the case of multi-level instancing
        ref->getPrimitive()->accept(*this);
    }

    bool &mHasVolume;
    bool &mHasSurface;
};

} // namespace rt
} // namespace moonray

