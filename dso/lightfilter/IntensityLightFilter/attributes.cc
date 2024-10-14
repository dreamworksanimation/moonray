// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Float>        attrIntensity;
    rdl2::AttributeKey<rdl2::Float>        attrExposure;
    rdl2::AttributeKey<rdl2::Rgb>          attrColor;
    rdl2::AttributeKey<rdl2::Bool>         attrInvert;
    rdl2::AttributeKey<rdl2::Int>          attrLightPathSelection;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrIntensity = sceneClass.declareAttribute<rdl2::Float>("intensity", 1.0);
    sceneClass.setMetadata(attrIntensity, "comment", "Multiply the light radiance by this intensity value");

    attrExposure = sceneClass.declareAttribute<rdl2::Float>("exposure", 0.0);
    sceneClass.setMetadata(attrExposure, "comment", "Multiply the light radiance by exposure = pow(2, exposure)");

    attrColor = sceneClass.declareAttribute<rdl2::Rgb>("color", rdl2::Rgb(1.0, 1.0, 1.0));
    sceneClass.setMetadata(attrColor, "comment", "Multiply the light radiance by this RGB color value");

    attrInvert = sceneClass.declareAttribute<rdl2::Bool>("invert", false);
    sceneClass.setMetadata(attrInvert, "comment", "Invert the light radiance by 1/radiance");

    attrLightPathSelection = sceneClass.declareAttribute<rdl2::Int>("light_path_selection", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setMetadata(attrLightPathSelection, "label", "light path selection");
    sceneClass.setMetadata(attrLightPathSelection, "comment", "Controls which light paths the filter is applied to.");
    sceneClass.setEnumValue(attrLightPathSelection, 0, "all light paths");
    sceneClass.setEnumValue(attrLightPathSelection, 1, "all indirect");
    sceneClass.setEnumValue(attrLightPathSelection, 2, "all indirect first bounce");
    sceneClass.setEnumValue(attrLightPathSelection, 3, "indirect diffuse");
    sceneClass.setEnumValue(attrLightPathSelection, 4, "indirect diffuse first bounce");
    sceneClass.setEnumValue(attrLightPathSelection, 5, "indirect specular");
    sceneClass.setEnumValue(attrLightPathSelection, 6, "indirect specular first bounce");

    sceneClass.setGroup("Properties", attrIntensity);
    sceneClass.setGroup("Properties", attrExposure);
    sceneClass.setGroup("Properties", attrColor);
    sceneClass.setGroup("Properties", attrInvert);
    sceneClass.setGroup("Properties", attrLightPathSelection);

RDL2_DSO_ATTR_END

