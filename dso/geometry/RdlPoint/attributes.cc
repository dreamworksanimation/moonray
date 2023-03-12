// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file attributes.cc

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Vec3fVector>  attrPos0;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrPos1;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrVel0;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrVel1;
    rdl2::AttributeKey<rdl2::Vec3fVector>  attrAcc;

    rdl2::AttributeKey<rdl2::FloatVector>  attrRadius;

    rdl2::AttributeKey<rdl2::StringVector> attrPartList;
    rdl2::AttributeKey<rdl2::IntVector>    attrPartIndices;

    rdl2::AttributeKey<rdl2::Float>        attrVelocityScale;

    // support for arbitrary data. Vector of UserData
    rdl2::AttributeKey<rdl2::SceneObjectVector> attrPrimitiveAttributes;

    DECLARE_COMMON_MOTION_BLUR_ATTRIBUTES

RDL2_DSO_ATTR_DEFINE(rdl2::Geometry);

    attrPos0 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("vertex_list_0", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "vertex_list", "vertex list" });
    sceneClass.setMetadata(attrPos0, "label", "vertex list 0");
    sceneClass.setMetadata(attrPos0, "comment", "Stores all vertices used by the points at motion step 0");

    attrPos1 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("vertex_list_1", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "vertex_list_mb", "vertex list mb" });
    sceneClass.setMetadata(attrPos1, "label", "vertex list 1");
    sceneClass.setMetadata(attrPos1, "comment", "If the points are in motion, the second "
        "motion step is stored in this attribute");

    attrVel0 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("velocity_list_0", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       { "velocity_list", "velocity list" });
    sceneClass.setMetadata(attrVel0, "label", "velocity list 0");
    sceneClass.setMetadata(attrVel0, "comment", "Optionally declare vertex velocities "
        "instead of a second motion step'");

    attrVel1 =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("velocity_list_1");
    sceneClass.setMetadata(attrVel1, "label", "velocity list 1");
    sceneClass.setMetadata(attrVel1, "comment", "Optionally declare second set of"
        "vertex velocities together with second motion step for cubic motion interpolation");

    attrAcc =
        sceneClass.declareAttribute<rdl2::Vec3fVector>("accleration_list");
    sceneClass.setMetadata(attrAcc, "label", "acceleration list");
    sceneClass.setMetadata(attrAcc, "comment", "Optionally declare vertex accelerations "
        "for quadratic motion interpolation");

    attrRadius =
        sceneClass.declareAttribute<rdl2::FloatVector>("radius_list", { "radius list" });
    sceneClass.setMetadata(attrRadius, "label", "radius list");
    sceneClass.setMetadata(attrRadius, "comment", "Stores all radii");

    attrPartList =
        sceneClass.declareAttribute<rdl2::StringVector>("part_list", {}, { "part list" });
    sceneClass.setMetadata(attrPartList, "label", "part list");
    sceneClass.setMetadata(attrPartList, "comment", "Ordered list of part names, used "
        "in conjunction with 'part_indices' to assign per-part materials");

    attrPartIndices =
        sceneClass.declareAttribute<rdl2::IntVector>("part_indices", { "part indices" });
    sceneClass.setMetadata(attrPartIndices, "label", "part indices");
    sceneClass.setMetadata(attrPartIndices, "comment", "Ordered list of part indices. ");

    attrVelocityScale =
        sceneClass.declareAttribute<rdl2::Float>("velocity_scale", 1.0f, { "velocity scale" });
    sceneClass.setMetadata(attrVelocityScale, "label", "velocity scale");
    sceneClass.setMetadata(attrVelocityScale, "comment", "Adjust magnitude of velocity-"
        "based motion blur");

    attrPrimitiveAttributes =
        sceneClass.declareAttribute<rdl2::SceneObjectVector>("primitive_attributes", { "primitive attributes" });
    sceneClass.setMetadata(attrPrimitiveAttributes, "label", "primitive attributes");
    sceneClass.setMetadata(attrPrimitiveAttributes, "comment", "Vector of UserData."
        "Each key/value pair will be added as a primitive attribute of the points.");

    DEFINE_COMMON_MOTION_BLUR_ATTRIBUTES

RDL2_DSO_ATTR_END

