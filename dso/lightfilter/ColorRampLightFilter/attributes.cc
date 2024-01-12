// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Mat4d>       attrNodeXform;
    rdl2::AttributeKey<rdl2::Bool>        attrUseXform;
    rdl2::AttributeKey<rdl2::Float>       attrBeginDistance;
    rdl2::AttributeKey<rdl2::Float>       attrEndDistance;
    rdl2::AttributeKey<rdl2::RgbVector>   attrColors;
    rdl2::AttributeKey<rdl2::FloatVector> attrDistances;
    rdl2::AttributeKey<rdl2::IntVector>   attrInterpolationTypes;
    rdl2::AttributeKey<rdl2::Float>       attrIntensity;
    rdl2::AttributeKey<rdl2::Float>       attrDensity;
    rdl2::AttributeKey<rdl2::Int>         attrMode;
    rdl2::AttributeKey<rdl2::Int>         attrWrapMode;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrNodeXform = sceneClass.declareAttribute<rdl2::Mat4d>("node_xform",
        rdl2::FLAGS_BLURRABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrNodeXform, "label", "node xform");
    sceneClass.setMetadata(attrNodeXform, "comment", "Orientation of the light filter");

    attrUseXform = sceneClass.declareAttribute<rdl2::Bool>("use_xform", false, {});
    sceneClass.setMetadata(attrUseXform, "label", "use xform");
    sceneClass.setMetadata(attrUseXform, "comment",
        "The filter can be bound to a light or lights position or when this toggle is set, "
        "can have its own transform");

    attrBeginDistance = sceneClass.declareAttribute<rdl2::Float>("begin_distance", 0.f, {});
    sceneClass.setMetadata(attrBeginDistance, "min", "0.0");  // for UI slider
    sceneClass.setMetadata(attrBeginDistance, "max", "100.0");
    sceneClass.setMetadata(attrBeginDistance, "comment",
        "Where the ramp starts relative to the light or the ramp's independent transform");

    attrEndDistance = sceneClass.declareAttribute<rdl2::Float>("end_distance", 1.f, {});
    sceneClass.setMetadata(attrEndDistance, "min", "0.0");
    sceneClass.setMetadata(attrEndDistance, "max", "100.0");
    sceneClass.setMetadata(attrEndDistance, "comment",
        "Where the ramp ends relative to the light or the ramp's independent transform");

    rdl2::RgbVector colorDefaults = {
        rdl2::Rgb(1.f, 1.f, 1.f),
        rdl2::Rgb(0.f, 0.f, 0.f) };

    attrColors = sceneClass.declareAttribute<rdl2::RgbVector>("colors", colorDefaults, {});
    // the metadata configures the UI widget
    sceneClass.setMetadata(attrColors, "label", "colors");
    sceneClass.setMetadata(attrColors, "structure_name", "ramp");
    sceneClass.setMetadata(attrColors, "structure_path", "values");
    sceneClass.setMetadata(attrColors, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrColors, "comment",
        "Vector of colors specified at different distances");

    rdl2::FloatVector distanceDefaults = {0.f, 1.f};

    attrDistances = sceneClass.declareAttribute<rdl2::FloatVector>("distances", distanceDefaults, {});
    sceneClass.setMetadata(attrDistances, "label", "distances");
    sceneClass.setMetadata(attrDistances, "structure_name", "ramp");
    sceneClass.setMetadata(attrDistances, "structure_path", "positions");
    sceneClass.setMetadata(attrDistances, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrDistances, "comment",
        "Distances between which colors are interpolated");

    rdl2::IntVector interpolationDefaults = {1, 1}; //linear

    attrInterpolationTypes = sceneClass.declareAttribute<rdl2::IntVector>("interpolation_types",
        interpolationDefaults, {});
    sceneClass.setMetadata(attrInterpolationTypes, "label", "interpolation types");
    sceneClass.setMetadata(attrInterpolationTypes, "structure_name", "ramp");
    sceneClass.setMetadata(attrInterpolationTypes, "structure_path", "interpolation_types");
    sceneClass.setMetadata(attrInterpolationTypes, "structure_type", "ramp_color");
    // RDL doesn't support vectors of enums so raw numeric values must be used
    //sceneClass.setEnumValue(attrInterpolationTypes, 0, "none");
    //sceneClass.setEnumValue(attrInterpolationTypes, 1, "linear");
    //sceneClass.setEnumValue(attrInterpolationTypes, 2, "exponential_up");
    //sceneClass.setEnumValue(attrInterpolationTypes, 3, "exponential_down");
    //sceneClass.setEnumValue(attrInterpolationTypes, 4, "smooth");
    //sceneClass.setEnumValue(attrInterpolationTypes, 5, "catmull_rom");
    sceneClass.setMetadata(attrInterpolationTypes, "comment",
        "Interpolation types between the specified distances.  0: None "
        "1: linear 2: exponential_up 3: exponential_down 4: smooth 5: catmull_rom");

    attrIntensity = sceneClass.declareAttribute<rdl2::Float>("intensity", 1.0f);
    sceneClass.setMetadata(attrIntensity, "min", "0.0");
    sceneClass.setMetadata(attrIntensity, "max", "1.0");
    sceneClass.setMetadata(attrIntensity, "comment", "The intensity of the filter");

    attrDensity = sceneClass.declareAttribute<rdl2::Float>("density", 1.0f);
    sceneClass.setMetadata(attrDensity, "min", "0.0");
    sceneClass.setMetadata(attrDensity, "max", "1.0");
    sceneClass.setMetadata(attrDensity, "comment", "The density of the filter");

    attrMode = sceneClass.declareAttribute<rdl2::Int>("mode", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrMode, 0, "radial");
    sceneClass.setEnumValue(attrMode, 1, "directional");
    sceneClass.setMetadata(attrMode, "comment",
        "Ramp: Radiates out from the center of the light or ramp location.  "
        "Directional: Linear starting at the location of the light or ramp location along negative z");

    attrWrapMode = sceneClass.declareAttribute<rdl2::Int>("wrap_mode", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrWrapMode, 0, "extend");
    sceneClass.setEnumValue(attrWrapMode, 1, "mirror");
    sceneClass.setMetadata(attrWrapMode, "comment",
        "For directional filter mode where filter uses distance along -Z axis.  "
        "Extend: f(z) = f(0) for z > 0.  "
        "Mirror: f(z) = f(-z).");

    sceneClass.setGroup("Properties", attrNodeXform);
    sceneClass.setGroup("Properties", attrBeginDistance);
    sceneClass.setGroup("Properties", attrEndDistance);
    sceneClass.setGroup("Properties", attrColors);
    sceneClass.setGroup("Properties", attrDistances);
    sceneClass.setGroup("Properties", attrInterpolationTypes);
    sceneClass.setGroup("Properties", attrIntensity);
    sceneClass.setGroup("Properties", attrDensity);
    sceneClass.setGroup("Properties", attrMode);
    sceneClass.setGroup("Properties", attrWrapMode);

RDL2_DSO_ATTR_END




