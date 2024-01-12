// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestGeometry.cc
/// $Id$
///

#include "attributes.cc"
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/shading/Shading.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_CLASS_BEGIN(TestGeometry, rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(TestGeometry)
    moonray::geom::Procedural* createProcedural() const;
    void destroyProcedural() const;    
    bool deformed() const;
    void resetDeformed();

RDL2_DSO_CLASS_END(TestGeometry)

//------------------------------------------------------------------------------

namespace moonray {
namespace geom {

class TemplateProcedural : public ProceduralLeaf
{
public:
    // constructor can be freely extended but should always pass in State to
    // construct base Procedural class
    TemplateProcedural(const State& state) :
        ProceduralLeaf(state) {}

    void generate(const GenerateContext& generateContext,
			const shading::XformSamples& parent2render)
    {     
    }

    void update(const UpdateContext& updateContext,
			const shading::XformSamples& parent2render)
    {
    }
};

} // namespace geom
} // namespace moonray

//------------------------------------------------------------------------------

moonray::geom::Procedural* TestGeometry::createProcedural() const
{
    moonray::geom::State state;
    return new moonray::geom::TemplateProcedural(state);
}

void TestGeometry::destroyProcedural() const
{
    delete mProcedural;
}

bool TestGeometry::deformed() const
{
    return mProcedural->deformed();
}

void TestGeometry::resetDeformed()
{
    mProcedural->resetDeformed();
}

