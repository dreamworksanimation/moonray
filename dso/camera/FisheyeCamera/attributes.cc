// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Int>   attrMapping;
    rdl2::AttributeKey<rdl2::Int>   attrFormat;
    rdl2::AttributeKey<rdl2::Float> attrZoom;
    rdl2::AttributeKey<rdl2::Float> attrFov;

    RDL2_DSO_ATTR_DEFINE(rdl2::Camera)

    attrMapping = sceneClass.declareAttribute<rdl2::Int>("mapping", 0, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrMapping, "label", "mapping");
    sceneClass.setEnumValue(attrMapping, 0, "stereographic");
    sceneClass.setEnumValue(attrMapping, 1, "equidistant");
    sceneClass.setEnumValue(attrMapping, 2, "equisolid angle");
    sceneClass.setEnumValue(attrMapping, 3, "orthographic");
    sceneClass.setMetadata(attrMapping, rdl2::SceneClass::sComment, "The mapping function "
                                                                    "- see https://en.wikipedia.org/wiki/Fisheye_lens");

    attrFormat = sceneClass.declareAttribute<rdl2::Int>("format", 0, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrFormat, "label", "format");
    sceneClass.setEnumValue(attrFormat, 0, "circular");
    sceneClass.setEnumValue(attrFormat, 1, "cropped");
    sceneClass.setEnumValue(attrFormat, 2, "diagonal");
    sceneClass.setMetadata(attrFormat, rdl2::SceneClass::sComment, "The format for fitting the circle to the image "
                                                                    "- see https://en.wikipedia.org/wiki/Fisheye_lens");

    attrZoom = sceneClass.declareAttribute<rdl2::Float>("zoom", 1.0f);
    sceneClass.setMetadata(attrZoom, "label", "zoom");
    sceneClass.setMetadata(attrZoom, rdl2::SceneClass::sComment, "Scaling factor applied on top of the format scale");

    attrFov = sceneClass.declareAttribute<rdl2::Float>("fov", 180.0f, rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC,
                                                       {"FOV"});
    sceneClass.setMetadata(attrFov, "label", "fov");
    sceneClass.setMetadata(attrFov, rdl2::SceneClass::sComment, "Field of view measured in degrees");

RDL2_DSO_ATTR_END

