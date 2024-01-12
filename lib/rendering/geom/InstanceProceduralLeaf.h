// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file InstanceProceduralLeaf.h
/// $Id$
///

#ifndef GEOM_INSTANCEPROCEDURALLEAF_HAS_BEEN_INCLUDED
#define GEOM_INSTANCEPROCEDURALLEAF_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/ProceduralLeaf.h>

#include <memory>
#include <vector>

namespace moonray {
namespace geom {

// forward declaration
class SharedPrimitive;

enum class InstanceMethod {
    XFORM_ATTRIBUTES = 0,
    POINT_FILE = 1,
    XFORMS = 2
};

bool
getReferenceData(const scene_rdl2::rdl2::Geometry& rdlGeometry,
                 const scene_rdl2::rdl2::SceneObjectVector& references,
                 std::vector<std::shared_ptr<SharedPrimitive>>& refPrimitiveGroups,
                 std::vector<shading::XformSamples>& refXforms,  
                 std::vector<float>& refShadowRayEpsilons);

//----------------------------------------------------------------------------

///
/// @class InstanceProceduralLeaf InstanceProceduralLeaf.h <rendering/geom/InstanceProceduralLeaf.h>
/// @brief A InstanceProceduralLeaf can be sub-classed by procedurals that generate instance
/// primitives.
/// 
class InstanceProceduralLeaf : public ProceduralLeaf
{
public:
    /// Constructor / Destructor
    explicit InstanceProceduralLeaf(const State &state);

    ~InstanceProceduralLeaf();

    InstanceProceduralLeaf(const InstanceProceduralLeaf &other) = delete;

    const InstanceProceduralLeaf &operator=(const InstanceProceduralLeaf &other) = delete;

protected:
    void instanceWithXforms(const GenerateContext& generateContext,
                            const shading::XformSamples& parent2render,
                            const scene_rdl2::math::Mat4f& nodeXform,
                            const std::vector<std::shared_ptr<SharedPrimitive>>& ref,
                            const std::vector<shading::XformSamples>& refXforms,
                            const std::vector<float>& shadowRayEpsilons,
                            const InstanceMethod instanceMethod,
                            bool useRefXforms,
                            bool useRefAttrs,
                            bool explicitShading,
                            const int instanceLevel,
                            const scene_rdl2::rdl2::Vec3fVector& velocities,
                            const float evaluationFrame,
                            const scene_rdl2::rdl2::IntVector& indices,
                            const scene_rdl2::rdl2::IntVector& disableIndices,
                            const scene_rdl2::rdl2::SceneObjectVector& attributes,
                            const scene_rdl2::rdl2::Vec3fVector& positions,
                            const scene_rdl2::rdl2::Vec4fVector& orientations,
                            const scene_rdl2::rdl2::Vec3fVector& scales,
                            const scene_rdl2::rdl2::Mat4dVector& xforms);

};


//----------------------------------------------------------------------------

} // namespace geom
} // namespace moonray

#endif /* GEOM_INSTANCEPROCEDURALLEAF_HAS_BEEN_INCLUDED */

