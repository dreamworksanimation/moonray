// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>          attrNormalized;
    rdl2::AttributeKey<rdl2::Bool>          attrApplySceneScale;
    rdl2::AttributeKey<rdl2::SceneObject *> attrGeometry;
    rdl2::AttributeKey<rdl2::StringVector>  attrParts;
    rdl2::AttributeKey<rdl2::SceneObject *> attrMapShader;
    DECLARE_ATTR_KEYS_CLEAR_RADIUS

RDL2_DSO_ATTR_DEFINE(rdl2::Light)

    attrNormalized = sceneClass.declareAttribute<rdl2::Bool>("normalized", true);
    sceneClass.setMetadata(attrNormalized, rdl2::SceneClass::sComment,
        "When set to true, the size of the light can be changed without altering the amount of total energy "
        "cast into the scene. This is achieved via scaling the light's radiance by the reciprocal of its "
        "surface area. When set to false, the radiance is used as-is, regardless of surface area.");

    attrApplySceneScale = sceneClass.declareAttribute<rdl2::Bool>("apply_scene_scale", true);
    sceneClass.setMetadata(attrApplySceneScale, rdl2::SceneClass::sComment,
        "Whether to apply scene scale variable when normalized.");

    attrGeometry = sceneClass.declareAttribute<rdl2::SceneObject *>("geometry", nullptr,
        rdl2::FLAGS_NONE, rdl2::INTERFACE_GEOMETRY);
    sceneClass.setMetadata(attrGeometry, rdl2::SceneClass::sComment,
        "The SceneObject holding the mesh data to use as the MeshLight's surface.");


    attrParts = sceneClass.declareAttribute<rdl2::StringVector>("parts", {});
    sceneClass.setMetadata(attrParts, rdl2::SceneClass::sComment,
        "The parts list to use. If empty, all parts are used.");

    attrMapShader = sceneClass.declareAttribute<rdl2::SceneObject *>("map_shader", nullptr,
        rdl2::FLAGS_NONE, rdl2::INTERFACE_MAP);
    sceneClass.setMetadata(attrMapShader, "label", "map shader");
    sceneClass.setMetadata(attrMapShader, rdl2::SceneClass::sComment,
        "A Map shader to sample for radiance values.");

    DECLARE_ATTRS_CLEAR_RADIUS

    sceneClass.setGroup("Properties", attrNormalized);
    sceneClass.setGroup("Properties", attrApplySceneScale);
    sceneClass.setGroup("Properties", attrGeometry);
    sceneClass.setGroup("Properties", attrParts);
    sceneClass.setGroup("Properties", attrMapShader);
    SET_ATTR_GRP_CLEAR_RADIUS

RDL2_DSO_ATTR_END

