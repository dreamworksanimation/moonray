// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file RdlInstancerGeometry.cc
/// $Id$
///

#include "attributes.cc"

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/InstanceProceduralLeaf.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/shading/Shading.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/math/Mat4.h>

#include <unordered_set>
#include <vector>

using namespace moonray;
using namespace moonray::geom;
using namespace moonray::shading;

RDL2_DSO_CLASS_BEGIN(RdlInstancerGeometry, scene_rdl2::rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(RdlInstancerGeometry)
    moonray::geom::Procedural* createProcedural() const override;
    void destroyProcedural() const override;

RDL2_DSO_CLASS_END(RdlInstancerGeometry)

//------------------------------------------------------------------------------

namespace moonray {
namespace geom {

class RdlInstancerProcedural : public InstanceProceduralLeaf
{
public:
    // constructor can be freely extended but should always pass in State to
    // construct base Procedural class
    explicit RdlInstancerProcedural(const State& state) :
        InstanceProceduralLeaf(state) {}

    void generate(const GenerateContext& generateContext,
                  const XformSamples& parent2render)
    {
        const scene_rdl2::rdl2::Geometry* rdlGeometry = generateContext.getRdlGeometry();
        const scene_rdl2::rdl2::SceneObjectVector& references = rdlGeometry->get(scene_rdl2::rdl2::Geometry::sReferenceGeometries);
        std::vector<std::shared_ptr<SharedPrimitive>> refPrimitiveGroups(references.size(), nullptr);
        std::vector<moonray::shading::XformSamples> refXforms(references.size());
        std::vector<float> refShadowRayEpsilons;
        if (!getReferenceData(*rdlGeometry,
                              references,
                              refPrimitiveGroups,
                              refXforms,  
                              refShadowRayEpsilons)) {
            return;
        }

        const RdlInstancerGeometry* instanceGeometry =
            static_cast<const RdlInstancerGeometry*>(generateContext.getRdlGeometry());

        const scene_rdl2::math::Mat4f nodeXform =
            scene_rdl2::math::toFloat(rdlGeometry->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                                       scene_rdl2::rdl2::TIMESTEP_BEGIN));

        instanceWithXforms(generateContext,
                           parent2render,
                           nodeXform,
                           refPrimitiveGroups,
                           refXforms,
                           refShadowRayEpsilons,
                           static_cast<InstanceMethod>(instanceGeometry->get(attrInstanceMethod)),
                           instanceGeometry->get(attrUseReferenceXforms),
                           instanceGeometry->get(attrUseReferenceAttributes),
                           instanceGeometry->get(attrExplicitShading),
                           instanceGeometry->get(attrInstanceLevel),
                           instanceGeometry->get(attrVelocities),
                           instanceGeometry->get(attrEvaluationFrame),
                           instanceGeometry->get(attrRefIndices),
                           instanceGeometry->get(attrDisableIndices),
                           instanceGeometry->get(attrPrimitiveAttributes),
                           instanceGeometry->get(attrPositions),
                           instanceGeometry->get(attrOrientations),
                           instanceGeometry->get(attrScales),
                           instanceGeometry->get(attrXformList));
    }
};

} // namespace geom
} // namespace moonray

//------------------------------------------------------------------------------

moonray::geom::Procedural* RdlInstancerGeometry::createProcedural() const
{
    moonray::geom::State state;
    return new moonray::geom::RdlInstancerProcedural(state);
}

void RdlInstancerGeometry::destroyProcedural() const
{
    delete mProcedural;
}
