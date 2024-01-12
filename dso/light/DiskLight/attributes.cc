// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>      attrNormalized;
    rdl2::AttributeKey<rdl2::Bool>      attrApplySceneScale;
    rdl2::AttributeKey<rdl2::Float>     attrRadius;
    rdl2::AttributeKey<rdl2::Float>     attrSpread;
    rdl2::AttributeKey<rdl2::Int>       attrSidedness;
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

    attrRadius = sceneClass.declareAttribute<rdl2::Float>("radius", 1.0f);
    sceneClass.setMetadata(attrRadius, rdl2::SceneClass::sComment,
        "The radius of the DiskLight.");

    attrSpread = sceneClass.declareAttribute<rdl2::Float>("spread", 1.0f);
    sceneClass.setMetadata(attrSpread, rdl2::SceneClass::sComment,
        "The directionality of light emission. "
        "1 is completely diffuse hemisphere. 0 is parallel to normal of light.");

    attrSidedness = sceneClass.declareAttribute<rdl2::Int>("sidedness", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrSidedness, 0, "regular");
    sceneClass.setEnumValue(attrSidedness, 1, "reverse");
    sceneClass.setEnumValue(attrSidedness, 2, "2-sided");
    sceneClass.setMetadata(attrSidedness, rdl2::SceneClass::sComment,
        "When set to 0 (regular), light is emitted from the front-facing surface of the disk. "
        "When set to 1 (reverse), light is emitted from the back-facing surface of the disk. "
        "When set to 2 (2-sided), light is emitted from both surfaces of the disk. ");

    DECLARE_ATTRS_CLEAR_RADIUS

    sceneClass.setGroup("Properties", attrNormalized);
    sceneClass.setGroup("Properties", attrApplySceneScale);
    sceneClass.setGroup("Properties", attrRadius);
    sceneClass.setGroup("Properties", attrSpread);
    sceneClass.setGroup("Properties", attrSidedness);
    SET_ATTR_GRP_CLEAR_RADIUS

RDL2_DSO_ATTR_END

