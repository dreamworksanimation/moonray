// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Api.h
/// @brief File containing geometry API for Primitive creation and some helpers
//
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/Box.h>
#include <moonray/rendering/geom/Curves.h>
#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/Points.h>
#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/Sphere.h>
#include <moonray/rendering/geom/State.h>
#include <moonray/rendering/geom/SharedPrimitive.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>
#include <moonray/rendering/geom/VdbVolume.h>
#include <moonray/rendering/geom/TransformedPrimitive.h>
#include <moonray/rendering/bvh/shading/Xform.h>

namespace scene_rdl2 {

namespace rdl2 {
class Layer;
class Geometry;
} // namespace rdl2
}

namespace moonray {

namespace geom {

class MotionBlurParams;

enum class PrimitiveType
{
    BOX,
    CURVES,
    INSTANCE,
    POINTS,
    POLYGON_MESH,
    PRIMITIVE_GROUP,
    SPHERE,
    SUBDIVISION_MESH,
    TRANSFORMED_PRIMITIVE,
    VDB_VOLUME,
    UNKNOWN
};

/// @brief helper function that query the PrimitiveType of specified Primitive
/// @return PrimitiveType of specified Primitive
PrimitiveType
getPrimitiveType(const Primitive& p);

/// @brief helper function that concatenate primitive to render space
///     transform samples
/// @param state State object for current Procedural
/// @param parent2render transform from parent space to render space
///     that is passed in to each Procedural::generate/update call
/// @return primitive to render space transform
finline shading::XformSamples
computePrim2Render(const State& state, const shading::XformSamples& parent2render)
{
    shading::XformSamples prim2render;
    prim2render.reserve(parent2render.size());
    for (const auto& p2r : parent2render) {
        prim2render.push_back(state.getProc2Parent() * p2r);
    }
    return prim2render;
}

/// @brief helpter function that concatenate two shading::XformSamples
/// @param xformA shading::XformSamples
/// @param xformB shading::XformSamples
/// @throws AssertionError if both xformA and xformB contains more than one
///     xform sample and the number of sample doesn't match
/// @return shading::XformSamples that is the concatenation of xformA and xformB
finline shading::XformSamples
concatenate(const shading::XformSamples& xformA, const shading::XformSamples& xformB)
{
    MNRY_ASSERT_REQUIRE(!xformA.empty() && !xformB.empty());
    shading::XformSamples result;
    if (xformA.size() == 1) {
        for (size_t i = 0; i < xformB.size(); ++i) {
            result.push_back(xformA[0] * xformB[i]);
        }
    } else if (xformB.size() == 1) {
        for (size_t i = 0; i < xformA.size(); ++i) {
            result.push_back(xformA[i] * xformB[0]);
        }
    } else {
        MNRY_ASSERT_REQUIRE(xformA.size() == xformB.size());
        for (size_t i = 0; i < xformA.size(); ++i) {
            result.push_back(xformA[i] * xformB[i]);
        }
    }
    return result;
}

/// @brief create a Curves primitive
/// @param type the type of Curves to create
/// @param subtype the subtype of Curves to create
/// @param tessellationRate the tessellation rate for the curve segments
/// @param curvesVertexCount vector to specify the curve control vertex count.
///     The size of the vector is the number of curves in created primitive.
///     Each entry of curveVertexCount is the control vertex count of that curve.
/// @param vertices VertexBuffer that contains the curves control points
/// @param layerAssignmentId id that marks a unique combination of
///     geometry/partName/material/lightSet in layer
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (per face id, per vertex uv...etc)
/// @throws AssertionError if the input vertices or primitive attributes
///     are not valid. Util Curves::checkPrimitiveData can be used to verify
///     whether input curves data is valid
/// @return a unique pointer points to the created Curves primitive
std::unique_ptr<Curves>
createCurves(Curves::Type type,
             Curves::SubType subtype,
             int tessellationRate,
             Curves::CurvesVertexCount&& curvesVertexCount,
             Curves::VertexBuffer&& vertices,
             LayerAssignmentId&& layerAssignmentId,
             shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable(),
             bool validateGeometry = false,
             const scene_rdl2::rdl2::Geometry* rdlGeometry = nullptr);

/// @brief create a Points primitive
/// @param position VertexBuffer that contains the 3d coordinate of each point
/// @param radius RadiusBuffer that contains the radius of each point
/// @param layerAssignmentId id that marks a unique combination of
///     geometry/partName/material/lightSet in layer
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (per face id, per vertex uv...etc)
/// @throws AssertionError if the input position or primitive attributes
///     are not valid or the size of position and radius don't match.
/// @return a unique pointer points to the created Points primitive
std::unique_ptr<Points>
createPoints(Points::VertexBuffer&& position, Points::RadiusBuffer&& radius,
             LayerAssignmentId&& layerAssignmentId,
             shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable(),
             bool validateGeometry = false,
             const scene_rdl2::rdl2::Geometry* rdlGeometry = nullptr);

/// @brief create a PolygonMesh primitive
/// @param faceVertexCount vector to specify the face vertex count.
///     The size of the vector is the number of faces in created primitive.
///     Each entry of faceVertexCount is the vertex count of that face.
/// @param indices vector of vertex index describe the topology of polygonmesh
///     The size of the vector is the sum of all entry values in faceVertexCount
/// @param vertices VertexBuffer that contains the vertices of polygon mesh
/// @param layerAssignmentId assignment id that marks a unique combination of
///     geometry/partName/material/lightSet in layer.
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (per face id, per vertex uv...etc)
/// @throws AssertionError if the input vertices or primitive attributes
///     are not valid.
/// @return a unique pointer points to the created PolygonMesh primitive
std::unique_ptr<PolygonMesh>
createPolygonMesh(PolygonMesh::FaceVertexCount&& faceVertexCount,
                  PolygonMesh::IndexBuffer&& indices,
                  PolygonMesh::VertexBuffer&& vertices,
                  LayerAssignmentId&& layerAssignmentId,
                  shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable(),
                  bool validateGeometry = false,
                  const scene_rdl2::rdl2::Geometry* rdlGeometry = nullptr);

std::unique_ptr<VdbVolume>
createVdbVolume(const VdbVolume::VdbInitData& vdbInitData,
                const MotionBlurParams& motionBlurParams,
                LayerAssignmentId&& layerAssignmentId,
                shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable());

/// @brief create a PrimitiveGroup that can hold multiple primitives
/// @return an unique pointer points to the created PrimitiveGroup
std::unique_ptr<PrimitiveGroup>
createPrimitiveGroup();

/// @brief create a Sphere primitive
/// @param radius sphere radius
/// @param layerAssignmentId assignment id that marks a unique combination of
///     geometry/partName/material/lightSet in layer.
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (per face id, per vertex uv...etc)
/// @throws AssertionError if the input primitive attributes are not valid.
/// @return a unique pointer points to the created Sphere primitive
std::unique_ptr<Sphere>
createSphere(float radius, LayerAssignmentId&& layerAssignmentId,
             shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable());

/// @brief create a Box primitive
/// @param length box length
/// @param width box width
/// @param height box height
/// @param layerAssignmentId assignment id that marks a unique combination of
///     geometry/partName/material/lightSet in layer.
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (per face id, per vertex uv...etc)
/// @throws AssertionError if the input primitive attributes are not valid.
/// @return a unique pointer points to the created Box primitive
std::unique_ptr<Box>
createBox(float length, float width, float height, LayerAssignmentId&& layerAssignmentId,
          shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable());

/// @brief create a SubdivisionMesh primitive
/// @param scheme the subdivision scheme (bilinear, catmull clark...)
/// @param faceVertexCount vector to specify the face vertex count.
///     The size of the vector is the number of faces in created primitive.
///     Each entry of faceVertexCount is the vertex count of that face.
/// @param indices vector of vertex index describe the topology of control mesh.
///     The size of the vector is the sum of all entry values in faceVertexCount
/// @param vertices VertexBuffer that contains the vertices of control mesh
/// @param layerAssignmentId assignment id that marks a unique combination of
///     geometry/partName/material/lightSet in layer.
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (per face id, per vertex uv...etc)
/// @throws AssertionError if the input vertices or primitive attributes
///     are not valid.
/// @return a unique pointer points to the created SubdivisionMesh primitive
std::unique_ptr<SubdivisionMesh>
createSubdivisionMesh(SubdivisionMesh::Scheme scheme,
                      SubdivisionMesh::FaceVertexCount&& faceVertexCount,
                      SubdivisionMesh::IndexBuffer&& indices,
                      SubdivisionMesh::VertexBuffer&& vertices,
                      LayerAssignmentId&& layerAssignmentId,
                      shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable(),
                      bool validateGeometry = false,
                      const scene_rdl2::rdl2::Geometry* rdlGeometry = nullptr);

/// @brief create an Instance with a transform and a referenced Primitive
/// @param xform matrix that specify the local transform of instance
/// @param reference the SharedPrimitive this instance referenced to
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (Instance only supports
///     constant rate attributes)
/// @return a unique pointer points to the created Instance primitive
std::unique_ptr<Instance>
createInstance(const Mat43& xform, std::shared_ptr<SharedPrimitive> reference,
               shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable());

/// @brief create an Instance with a transform and a referenced Primitive
/// @param xform shading::XformSamples that specify the local transform of instance
/// @param reference the SharedPrimitive this instance referenced to
/// @param attributeTable a lookup table that can be used to attach
///     arbitrary primitive attribute (Instance only supports
///     constant rate attributes)
/// @return a unique pointer points to the created Instance primitive
std::unique_ptr<Instance>
createInstance(const shading::XformSamples& xform,
               std::shared_ptr<SharedPrimitive> reference,
               shading::PrimitiveAttributeTable&& attributeTable = shading::PrimitiveAttributeTable());

/// @brief create a SharedPrimitive for instancing usage
/// @param primitive the primitive to be shared/referenced by Instance
/// @return a shared pointer points to the SharedPrimitive which can be
///     referenced by Instance
std::shared_ptr<SharedPrimitive>
createSharedPrimitive(std::unique_ptr<Primitive> primitive);

/// @brief create a TransformedPrimitive
/// @param xform matrix that specify the local transform of held primitive
/// @param primitive the primitive to be held
/// @return a unique pointer points to the created TransformedPrimitive
std::unique_ptr<TransformedPrimitive>
createTransformedPrimitive(const Mat43& xform,
                           std::unique_ptr<Primitive> primitive);

/// @brief create a TransformedPrimitive
/// @param xform shading::XformSamples that specify the local transform of held primitive
/// @param primitive the primitive to be held
/// @return a unique pointer points to the created TransformedPrimitive
std::unique_ptr<TransformedPrimitive>
createTransformedPrimitive(const shading::XformSamples& xform,
                           std::unique_ptr<Primitive> primitive);

// utility function that may converts input primitive to Instance primitve
// so that it can handle rotation motion blur
std::unique_ptr<Primitive>
convertForMotionBlur(const ProceduralContext& context,
                     std::unique_ptr<Primitive> primitive, bool useRotationMotionBlur);

/// @brief an util method removes unassigned face/facevarying data
void
removeUnassignedFaces(const scene_rdl2::rdl2::Layer* layer,
                      LayerAssignmentId& layerAssignmentId,
                      std::vector<Primitive::IndexType>& faceToPart,
                      std::vector<Primitive::size_type>& faceVertexCount,
                      std::vector<Primitive::IndexType>& indices,
                      shading::PrimitiveAttributeTable* primitiveAttributeTable = nullptr);

/// @brief an util function to query assignment id from layer
/// @param layer rdl2 Layer to query
/// @param geometry rdl2 Geometry to query
/// @param partName string part name of the geometry to query
/// @param assignmentId the assignment id for specified layer/geometry/partName combination
/// @return whether the assignment id is valid (exist and map to non null shaders)
bool
getAssignmentId(const scene_rdl2::rdl2::Layer* layer, 
                const scene_rdl2::rdl2::Geometry* geometry,
                const std::string& partName, 
                int& assignmentId);

/// @brief an util function to check that all explicit shading attributes (normal, dPds, and dPDt) are present
/// @param pat primitive attribute table to check
/// @param geometry rdl2 Geometry to use for printing errors
/// @return whether all of the explicit shading attribute are present
bool
validateExplicitShading(const shading::PrimitiveAttributeTable& pat,
                        const scene_rdl2::rdl2::Geometry* rdlGeometry);

/// @brief a function that adds the sExplicitShading primitive attribute if the required explicit shading attributes are present
/// @param pat primitive attribute table to check and add sExplicitShading to
/// @param geometry rdl2 Geometry to use for printing errors
/// @return whether the sExplicitShading attribute was added or not
bool
addExplicitShading(const scene_rdl2::rdl2::Geometry* rdlGeometry,
                   shading::PrimitiveAttributeTable& pat);



} // namespace geom
} // namespace moonray

