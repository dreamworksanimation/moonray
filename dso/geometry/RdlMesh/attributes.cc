// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::IntVector>    attrFaceVertexCount;
    rdl2::AttributeKey<rdl2::IntVector>    attrVertexIndex;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrPos0;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrPos1;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrVel0;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrVel1;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrAcc;

    rdl2::AttributeKey<rdl2::StringVector> attrPartList;
    rdl2::AttributeKey<rdl2::IntVector>    attrPartFaceCountList;
    rdl2::AttributeKey<rdl2::IntVector>    attrPartFaceIndices;
    rdl2::AttributeKey<rdl2::Int>          attrOrientation;
    const int ORIENTATION_RIGHT_HANDED = 0;
    const int ORIENTATION_LEFT_HANDED = 1;

    rdl2::AttributeKey<rdl2::Vec2fVector>  attrUVs;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrNormals;

    rdl2::AttributeKey<rdl2::Float>        attrVelocityScale;

    rdl2::AttributeKey<rdl2::Bool>         attrIsSubd;
    rdl2::AttributeKey<rdl2::Int>          attrSubdScheme;
    rdl2::AttributeKey<rdl2::Int>          attrSubdBoundary;
    rdl2::AttributeKey<rdl2::Int>          attrSubdFVarLinear;
    rdl2::AttributeKey<rdl2::IntVector>    attrSubdCreaseIndices;
    rdl2::AttributeKey<rdl2::FloatVector>  attrSubdCreaseSharpnesses;
    rdl2::AttributeKey<rdl2::IntVector>    attrSubdCornerIndices;
    rdl2::AttributeKey<rdl2::FloatVector>  attrSubdCornerSharpnesses;

    // support for arbitrary data. Vector of UserData
    rdl2::AttributeKey<rdl2::SceneObjectVector> attrPrimitiveAttributes;

    DECLARE_COMMON_MESH_ATTRIBUTES
    DECLARE_COMMON_MOTION_BLUR_ATTRIBUTES
    DECLARE_COMMON_EXPLICIT_SHADING_ATTRIBUTES

RDL2_DSO_ATTR_DEFINE(rdl2::Geometry)

    attrFaceVertexCount =
        sceneClass.declareAttribute<rdl2::IntVector>("face_vertex_count", { "face vertex count" });
    sceneClass.setMetadata(attrFaceVertexCount, "label", "face vertex count");
    sceneClass.setMetadata(attrFaceVertexCount, "comment", "List of vertices per "
        "face, used in conjunction with vertices_by_index to construct the mesh");
    sceneClass.setGroup("Mesh", attrFaceVertexCount);

    attrVertexIndex =
        sceneClass.declareAttribute<rdl2::IntVector>("vertices_by_index", { "vertices by index" });
    sceneClass.setMetadata(attrVertexIndex, "label", "vertices by index");
    sceneClass.setMetadata(attrVertexIndex, "comment", "List of vertex indices "
        "used to construct the mesh using the vertex list");
    sceneClass.setGroup("Mesh", attrVertexIndex);

    attrPos0 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("vertex_list_0", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "vertex list 0", "vertex_list", "vertex list" });
    sceneClass.setMetadata(attrPos0, "label", "vertex list 0");
    sceneClass.setMetadata(attrPos0, "comment", "List of vertex positions used by the mesh at motion step 0");
    sceneClass.setGroup("Mesh", attrPos0);

    attrPos1 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("vertex_list_1", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "vertex list 1", "vertex_list_mb", "vertex list mb" });
    sceneClass.setMetadata(attrPos1, "label", "vertex list 1");
    sceneClass.setMetadata(attrPos1, "comment", "If the mesh is in motion, the vertex positions for the second "
        "motion step are stored in this attribute");
    sceneClass.setGroup("Mesh", attrPos1);

    attrVel0 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("velocity_list_0", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "velocity list 0", "velocity_list", "velocity list" });
    sceneClass.setMetadata(attrVel0, "label", "velocity list 0");
    sceneClass.setMetadata(attrVel0, "comment", "Optionally declared explicit vertex velocities to use "
        "instead of vertex positions from a second motion step'");
    sceneClass.setGroup("Motion Blur", attrVel0);

    attrVel1 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("velocity_list_1", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "velocity list 1", "velocity_list_B", "velocity list B" });
    sceneClass.setMetadata(attrVel1, "comment", "Optionally declared second set of "
        "vertex velocities together with vertex positions from the second motion step for cubic motion interpolation");
    sceneClass.setGroup("Motion Blur", attrVel1);

    attrAcc =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("accleration_list", { "acceleration list" });
    sceneClass.setMetadata(attrAcc, "comment", "Optionally declared vertex accelerations "
        "for quadratic motion interpolation");
    sceneClass.setGroup("Motion Blur", attrAcc);

    attrOrientation =
        sceneClass.declareAttribute<rdl2::Int>("orientation", ORIENTATION_RIGHT_HANDED,
                                               rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setEnumValue(attrOrientation, ORIENTATION_RIGHT_HANDED, "right-handed");
    sceneClass.setEnumValue(attrOrientation, ORIENTATION_LEFT_HANDED, "left-handed");
    sceneClass.setMetadata(attrOrientation, "label", "orientation");
    sceneClass.setMetadata(attrOrientation, "comment", "When set to \"left-handed\", normals are generated "
        "using the left-handed rule. This reverses the direction of generated normals, and "
        "which side of surfaces is considered the front, without affecting supplied normals.");
    sceneClass.setGroup("Mesh", attrOrientation);

    attrPartList =
        sceneClass.declareAttribute<rdl2::StringVector>("part_list", {}, { "part list" });
    sceneClass.setMetadata(attrPartList, "label", "part list");
    sceneClass.setMetadata(attrPartList, "comment", "List of part names, used "
        "in conjunction with 'part face count list' and 'part faces indicies' "
        "to assign per-part materials");
    sceneClass.setGroup("Mesh", attrPartList);

    attrPartFaceCountList =
        sceneClass.declareAttribute<rdl2::IntVector>("part_face_count_list", { "part face count list" });
    sceneClass.setMetadata(attrPartFaceCountList, "label", "part face count list");
    sceneClass.setMetadata(attrPartFaceCountList, "comment", "List of the number of faces "
        "belonging to the part with corresponding index in 'part list'.");
    sceneClass.setGroup("Mesh", attrPartFaceCountList);

    attrPartFaceIndices =
        sceneClass.declareAttribute<rdl2::IntVector>("part_face_indices", { "part face indices" });
    sceneClass.setMetadata(attrPartFaceIndices, "label", "part face indices");
    sceneClass.setMetadata(attrPartFaceIndices, "comment", "List of face indices. "
        "No index should have a value greater than the size of 'face_vertex_count'");
    sceneClass.setGroup("Mesh", attrPartFaceIndices);

    attrUVs =
        sceneClass.declareAttribute<rdl2::Vec2fVector>("uv_list", { "uv list" });
    sceneClass.setMetadata(attrUVs, "label", "uv list");
    sceneClass.setMetadata(attrUVs, "comment", "If the mesh is using UVs, store them "
        "per face-vertex in this list");
    sceneClass.setGroup("Mesh", attrUVs);

    attrNormals =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("normal_list", { "normal list" });
    sceneClass.setMetadata(attrNormals, "label", "normal list");
    sceneClass.setMetadata(attrNormals, "comment", " If the mesh is using normals, "
        "store them per face-vertex in this list");
    sceneClass.setGroup("Mesh", attrNormals);

    attrVelocityScale =
        sceneClass.declareAttribute<rdl2::Float>("velocity_scale", 1.0f, { "velocity scale" });
    sceneClass.setMetadata(attrVelocityScale, "label", "velocity scale");
    sceneClass.setMetadata(attrVelocityScale, "comment", "Adjusts magnitude of velocity-"
        "based motion blur");
    sceneClass.setGroup("Motion Blur", attrVelocityScale);

    attrIsSubd =
        sceneClass.declareAttribute<rdl2::Bool>("is_subd", true, { "is subd" });
    sceneClass.setMetadata(attrIsSubd, "label", "is subd");
    sceneClass.setMetadata(attrIsSubd, "comment", "If true, a SubdivisionMesh "
        "primitive will be created - PolygonMesh otherwise");
    sceneClass.setGroup("Mesh", attrIsSubd);

    attrSubdScheme =
        sceneClass.declareAttribute<rdl2::Int>("subd_scheme", 1, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "subd scheme" });
    sceneClass.setMetadata(attrSubdScheme, "label", "subd scheme");
    sceneClass.setEnumValue(attrSubdScheme, 0, "bilinear");
    sceneClass.setEnumValue(attrSubdScheme, 1, "catclark");
    sceneClass.setMetadata(attrSubdScheme, "comment", "CatClark or Bilinear");
    sceneClass.setGroup("Mesh", attrSubdScheme);

    attrSubdBoundary =
        sceneClass.declareAttribute<rdl2::Int>("subd_boundary", 2, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "subd boundary" });
    sceneClass.setMetadata(attrSubdBoundary, "label", "subd boundary");
    sceneClass.setEnumValue(attrSubdBoundary, 0, "none");
    sceneClass.setEnumValue(attrSubdBoundary, 1, "edge only");
    sceneClass.setEnumValue(attrSubdBoundary, 2, "edge and corner");
    sceneClass.setMetadata(attrSubdBoundary, "comment", "Boundary interpolation: "
        "Corners, Edges or None");
    sceneClass.setGroup("Mesh", attrSubdBoundary);

    attrSubdFVarLinear =
        sceneClass.declareAttribute<rdl2::Int>("subd_fvar_linear", 1, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "subd fvar linear" });
    sceneClass.setMetadata(attrSubdFVarLinear, "label", "subd fvar linear");
    sceneClass.setEnumValue(attrSubdFVarLinear, 0, "none");
    sceneClass.setEnumValue(attrSubdFVarLinear, 1, "corners only");
    sceneClass.setEnumValue(attrSubdFVarLinear, 2, "corners plus1");
    sceneClass.setEnumValue(attrSubdFVarLinear, 3, "corners plus2");
    sceneClass.setEnumValue(attrSubdFVarLinear, 4, "boundaries");
    sceneClass.setEnumValue(attrSubdFVarLinear, 5, "all");
    sceneClass.setMetadata(attrSubdFVarLinear, "comment", "Face-varying linear interpolation: "
        "None, Corners Only, Corners Plus 1 or Plus 2 (RenderMan), Boundaries, or All");
    sceneClass.setGroup("Mesh", attrSubdFVarLinear);

    attrSubdCreaseIndices =
        sceneClass.declareAttribute<rdl2::IntVector>("subd_crease_indices", { "subd crease indices" });
    sceneClass.setMetadata(attrSubdCreaseIndices, "label", "subd crease indices");
    sceneClass.setMetadata(attrSubdCreaseIndices, "comment", "List of vertex index pairs for "
        "each crease edge with an assigned sharpness.");
    sceneClass.setGroup("Mesh", attrSubdCreaseIndices);

    attrSubdCreaseSharpnesses =
        sceneClass.declareAttribute<rdl2::FloatVector>("subd_crease_sharpnesses", { "subd crease sharpnesses" });
    sceneClass.setMetadata(attrSubdCreaseSharpnesses, "label", "subd crease sharpnesses");
    sceneClass.setMetadata(attrSubdCreaseSharpnesses, "comment", "Sharpness value for each crease edge.");
    sceneClass.setGroup("Mesh", attrSubdCreaseSharpnesses);

    attrSubdCornerIndices =
        sceneClass.declareAttribute<rdl2::IntVector>("subd_corner_indices", { "subd corner indices" });
    sceneClass.setMetadata(attrSubdCornerIndices, "label", "subd corner indices");
    sceneClass.setMetadata(attrSubdCornerIndices, "comment", "List of indices for "
        "each corner vertex with an assigned sharpness.");
    sceneClass.setGroup("Mesh", attrSubdCornerIndices);

    attrSubdCornerSharpnesses =
        sceneClass.declareAttribute<rdl2::FloatVector>("subd_corner_sharpnesses", { "subd corner sharpnesses" });
    sceneClass.setMetadata(attrSubdCornerSharpnesses, "label", "subd corner sharpnesses");
    sceneClass.setMetadata(attrSubdCornerSharpnesses, "comment", "Sharpness value for each corner vertex.");
    sceneClass.setGroup("Mesh", attrSubdCornerSharpnesses);

    attrPrimitiveAttributes =
        sceneClass.declareAttribute<rdl2::SceneObjectVector>("primitive_attributes", { "primitive attributes" });
    sceneClass.setMetadata(attrPrimitiveAttributes, "label", "primitive attributes");
    sceneClass.setMetadata(attrPrimitiveAttributes, "comment", "Vector of UserData.  "
        "Each key/value pair will be added as a primitive attribute of the mesh.");
    sceneClass.setGroup("User Data", attrPrimitiveAttributes);

    DEFINE_COMMON_MESH_ATTRIBUTES
    DEFINE_COMMON_MOTION_BLUR_ATTRIBUTES
    DEFINE_COMMON_EXPLICIT_SHADING_ATTRIBUTES

RDL2_DSO_ATTR_END


