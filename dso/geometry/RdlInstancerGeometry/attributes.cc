// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file attributes.cc
/// $Id$
///

#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_ATTR_DECLARE
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> attrInstanceMethod;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> attrInstanceLevel;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Mat4dVector> attrXformList;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3fVector> attrPositions;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec4fVector> attrOrientations;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3fVector> attrScales;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3fVector> attrVelocities;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector> attrRefIndices;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector> attrDisableIndices;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObjectVector> attrPrimitiveAttributes;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> attrPointFile;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> attrEvaluationFrame;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> attrUseRotationMotionBlur;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> attrUseReferenceXforms;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> attrUseReferenceAttributes;

RDL2_DSO_ATTR_DEFINE(scene_rdl2::rdl2::Geometry)

    attrInstanceMethod =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Int>("method", 0,
        scene_rdl2::rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrInstanceMethod, 0, "xform attributes");
    sceneClass.setEnumValue(attrInstanceMethod, 2, "xform list");
    sceneClass.setMetadata(attrInstanceMethod, "comment",
        "Specify the source of xforms/refIndices for instancing. "
        "If set to \"xform attributes\", taking data from attributes "
        "\"positions\", \"orientations\", \"scales\", \"velocities\" "
        "and \"refIndices\". "
        "If set to \"xform list\", taking data from \"xform list\", \"velocities\" "
        "and \"refIndices\". ");

    attrInstanceLevel =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Int>("instance_level", 0,
        scene_rdl2::rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrInstanceLevel, 0, "instance level 0");
    sceneClass.setEnumValue(attrInstanceLevel, 1, "instance level 1");
    sceneClass.setEnumValue(attrInstanceLevel, 2, "instance level 2");
    sceneClass.setEnumValue(attrInstanceLevel, 3, "instance level 3");
    sceneClass.setEnumValue(attrInstanceLevel, 4, "instance level 4");
    sceneClass.setMetadata(attrInstanceLevel, "comment", "Level of this instance.  This level can be referenced in TransformSpaceMap to allow for transforming data to/from the local space of each instance this instancer produces.");

    attrXformList =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Mat4dVector>("xform_list");
    sceneClass.setMetadata(attrXformList, "label", "xform list");
    sceneClass.setMetadata(attrXformList, "comment",
        "A list of xforms that represent the per-instance xform.");
    sceneClass.setMetadata(attrXformList, "enable if",
            "OrderedDict([(u'method', u'2')])");

    attrPositions =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Vec3fVector>("positions");
    sceneClass.setMetadata(attrPositions, "comment",
        "A list of vec3 that represent the per-instance position.");
    sceneClass.setMetadata(attrPositions, "enable if",
            "OrderedDict([(u'method', u'0')])");

    attrOrientations =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Vec4fVector>("orientations");
    sceneClass.setMetadata(attrOrientations, "comment",
        "A list of quaternions that represent the per-instance orientation. "
        "The length should be either 0 or consistent with \"positions\".");
    sceneClass.setMetadata(attrOrientations, "enable if",
            "OrderedDict([(u'method', u'0')])");

    attrScales =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Vec3fVector>("scales");
    sceneClass.setMetadata(attrScales, "comment",
        "A list of vec3 that represet the per-instance scale. "
        "The length should be either 0 or consistent with \"positions\".");
    sceneClass.setMetadata(attrScales, "enable if",
            "OrderedDict([(u'method', u'0')])");

    attrVelocities =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Vec3fVector>("velocities");
    sceneClass.setMetadata(attrScales, "comment",
        "A list of vec3 that represet the per-instance velocity(motion blur). "
        "The length should be either 0 or consistent with \"positions\".");

    attrRefIndices =
        sceneClass.declareAttribute<scene_rdl2::rdl2::IntVector>("ref_indices", {}, { "refIndices" });
    sceneClass.setMetadata(attrRefIndices, "label", "ref indices");
    sceneClass.setMetadata(attrRefIndices, "comment",
        "A list of index values to specify entry of \"reference\" per instance. "
        "The length should be either 0 or consistent with \"positions\"|\"xform_list\". "
        "The index entry falls back to 0 when this attribute is empty "
        "or the value of entry is out of index range");

    attrDisableIndices =
        sceneClass.declareAttribute<scene_rdl2::rdl2::IntVector>("disable_indices", {}, { "disableIndices" });
    sceneClass.setMetadata(attrDisableIndices, "label", "disable indices");
    sceneClass.setMetadata(attrDisableIndices, "comment",
        "A list of index values to hide / disable. "
        "For example, with 4 instances you can supply a list of 0, 2 to disable those instances. "
        "If an index in this list is out of range, it is ignored.");

    attrPrimitiveAttributes =
        sceneClass.declareAttribute<scene_rdl2::rdl2::SceneObjectVector>(
        "primitive_attributes", scene_rdl2::rdl2::SceneObjectVector(), scene_rdl2::rdl2::FLAGS_NONE,
        scene_rdl2::rdl2::INTERFACE_USERDATA, { "primitive attributes" });
    sceneClass.setMetadata(attrPrimitiveAttributes, "label", "primitive attributes");
    sceneClass.setMetadata(attrPrimitiveAttributes, "comment",
        "A list of UserData to specify arbitrary primitive attributes"
        "(For example, color or roughness multiplier) per -instance");

    attrEvaluationFrame =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Float>("evaluation_frame", 0, { "evaluation frame" });
    sceneClass.setMetadata(attrEvaluationFrame, "label", "evaluation frame");
    sceneClass.setMetadata(attrEvaluationFrame, "comment",
        "Evaluate geometry at specified frame (relative) instead of SceneVariables frame.");

    attrUseReferenceXforms =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Bool>("use_reference_xforms", false);
    sceneClass.setMetadata(attrUseReferenceXforms, "label", "use reference xforms");
    sceneClass.setMetadata(attrUseReferenceXforms, "comment", "Transform the reference (prototype) geometry by it's node_xform parameter before applying the instance transform");

    attrUseReferenceAttributes =
        sceneClass.declareAttribute<scene_rdl2::rdl2::Bool>("use_reference_attributes", true);
    sceneClass.setMetadata(attrUseReferenceAttributes, "label", "use reference attributes");
    sceneClass.setMetadata(attrUseReferenceAttributes, "comment", "Use the geometry attributes of the reference (prototype) instead of the ones on the InstanceGeometry.   Currently only works for shadow_ray_epsilon");

RDL2_DSO_ATTR_END

