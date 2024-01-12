// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Vec3fVector>  attrPos0;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrPos1;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrVel0;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrVel1;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrAcc;

    rdl2::AttributeKey<rdl2::FloatVector>  attrRadius;
    rdl2::AttributeKey<rdl2::IntVector>    attrCurvesVertexCount;

    rdl2::AttributeKey<rdl2::StringVector> attrPartList;
    rdl2::AttributeKey<rdl2::IntVector>    attrPartIndices;

    rdl2::AttributeKey<rdl2::Float>        attrVelocityScale;
    rdl2::AttributeKey<rdl2::Vec2fVector>  attrUVs;

    rdl2::AttributeKey<rdl2::Int>          attrCurvesType;

    // support for arbitrary data. Vector of UserData
    rdl2::AttributeKey<rdl2::SceneObjectVector> attrPrimitiveAttributes;

    DECLARE_COMMON_CURVES_ATTRIBUTES
    DECLARE_COMMON_MOTION_BLUR_ATTRIBUTES
    DECLARE_COMMON_EXPLICIT_SHADING_ATTRIBUTES

RDL2_DSO_ATTR_DEFINE(rdl2::Geometry);

    attrCurvesVertexCount =
            sceneClass.declareAttribute<rdl2::IntVector>("curves_vertex_count");
        sceneClass.setMetadata(attrCurvesVertexCount, "label", "curves vertex count");
        sceneClass.setMetadata(attrCurvesVertexCount, "comment", "List of vertices per curve");
    sceneClass.setGroup("Curves", attrCurvesVertexCount);

    attrPos0 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("vertex_list_0");
    sceneClass.setMetadata(attrPos0, "label", "vertex list 0");
    sceneClass.setMetadata(attrPos0, "comment", "List of vertex positions used by the curves at motion step 0");
    sceneClass.setGroup("Curves", attrPos0);

    attrPos1 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("vertex_list_1");
    sceneClass.setMetadata(attrPos1, "label", "vertex list 1");
    sceneClass.setMetadata(attrPos1, "comment", "If the curves are in motion, the vertex positions for the second "
        "motion step are stored in this attribute");
    sceneClass.setGroup("Curves", attrPos1);

    attrVel0 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("velocity_list_0");
    sceneClass.setMetadata(attrVel0, "label", "velocity list 0");
    sceneClass.setMetadata(attrVel0, "comment", "Optionally declared explicit vertex velocities to use "
        "instead of vertex positions from a second motion step'");
    sceneClass.setGroup("Motion Blur", attrVel0);

    attrVel1 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("velocity_list_1");
    sceneClass.setMetadata(attrVel1, "label", "velocity list 1");
    sceneClass.setMetadata(attrVel1, "comment", "Optionally declared second set of "
        "vertex velocities together with vertex positions from the second motion step for cubic motion interpolation");
    sceneClass.setGroup("Motion Blur", attrVel1);

    attrAcc =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("accleration_list");
    sceneClass.setMetadata(attrAcc, "label", "acceleration list");
    sceneClass.setMetadata(attrAcc, "comment", "Optionally declared vertex accelerations "
        "for quadratic motion interpolation");
    sceneClass.setGroup("Motion Blur", attrAcc);

    attrRadius =
        sceneClass.declareAttribute<rdl2::FloatVector>("radius_list");
    sceneClass.setMetadata(attrRadius, "label", "radius list");
    sceneClass.setMetadata(attrRadius, "comment", "List of radius values.  The rate will be determined "
        "based on the number of values.   A single value will be interpreted as constant.   If the "
        " number of values matches the number of curves, the rate will be interpreted as uniform.  "
        " If the number of values matches the number of vertices, the rate will be interpreted as "
        " vertex.");
    sceneClass.setGroup("Curves", attrRadius);

    attrPartList =
        sceneClass.declareAttribute<rdl2::StringVector>("part_list", {}, { "part list" });
    sceneClass.setMetadata(attrPartList, "label", "part list");
    sceneClass.setMetadata(attrPartList, "comment", "List of part names, used "
        "in conjunction with 'part_indices' to assign per-part materials");
    sceneClass.setGroup("Curves", attrPartList);

    attrPartIndices =
        sceneClass.declareAttribute<rdl2::IntVector>("part_indices", { "part indices" });
    sceneClass.setMetadata(attrPartIndices, "label", "part indices");
    sceneClass.setMetadata(attrPartIndices, "comment", "List of part indices.  "
        "No index should have a value greater than the size of 'curves_vertex_count'");
    sceneClass.setGroup("Curves", attrPartIndices);

    attrVelocityScale =
        sceneClass.declareAttribute<rdl2::Float>("velocity_scale", 1.0f);
    sceneClass.setMetadata(attrVelocityScale, "label", "velocity scale");
    sceneClass.setMetadata(attrVelocityScale, "comment", "Adjusts magnitude of velocity-"
        "based motion blur");
    sceneClass.setGroup("Motion Blur", attrVelocityScale);

    attrUVs =
            sceneClass.declareAttribute<rdl2::Vec2fVector>("uv_list");
        sceneClass.setMetadata(attrUVs, "label", "uv list");
        sceneClass.setMetadata(attrUVs, "comment", "If the curves are using UVs, store them "
            "per-curve in this list");
    sceneClass.setGroup("Curves", attrUVs);

    attrCurvesType =
        sceneClass.declareAttribute<rdl2::Int>("curve_type", 1, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setMetadata(attrCurvesType, "label", "curve type");
    sceneClass.setEnumValue(attrCurvesType, 0, "linear");
    sceneClass.setEnumValue(attrCurvesType, 1, "bezier");
    sceneClass.setEnumValue(attrCurvesType, 2, "bspline");
    sceneClass.setMetadata (attrCurvesType, "comment",
        "Curve interpolation type is linear, bezier, or bspline");
    sceneClass.setGroup("Curves", attrCurvesType);

    attrPrimitiveAttributes =
        sceneClass.declareAttribute<rdl2::SceneObjectVector>("primitive_attributes", { "primitive attributes" });
    sceneClass.setMetadata(attrPrimitiveAttributes, "label", "primitive attributes");
    sceneClass.setMetadata(attrPrimitiveAttributes, "comment", "Vector of UserData.  "
        "Each key/value pair will be added as a primitive attribute of the curves.");
    sceneClass.setGroup("User Data", attrPrimitiveAttributes);

    DEFINE_COMMON_CURVES_ATTRIBUTES
    DEFINE_COMMON_MOTION_BLUR_ATTRIBUTES
    DEFINE_COMMON_EXPLICIT_SHADING_ATTRIBUTES

RDL2_DSO_ATTR_END

