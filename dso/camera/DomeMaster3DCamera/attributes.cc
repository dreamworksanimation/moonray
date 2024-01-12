// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE
    rdl2::AttributeKey<rdl2::Float> attrFOVVerticalAngle;
    rdl2::AttributeKey<rdl2::Float> attrFOVHorizontalAngle;
    rdl2::AttributeKey<rdl2::Bool>  attrFlipRayX;
    rdl2::AttributeKey<rdl2::Bool>  attrFlipRayY;

    rdl2::AttributeKey<rdl2::Int>   attrCamera;
    rdl2::AttributeKey<rdl2::Float> attrParallaxDistance;
    rdl2::AttributeKey<rdl2::Float> attrCameraSeparation; // Interocular distance
    rdl2::AttributeKey<rdl2::String> attrCameraSeparationMapFileName;
    rdl2::AttributeKey<rdl2::Float> attrHeadTiltMap;
    rdl2::AttributeKey<rdl2::Bool>  attrZenithMode;

RDL2_DSO_ATTR_DEFINE(rdl2::Camera)
    attrFOVVerticalAngle = sceneClass.declareAttribute<rdl2::Float>("FOV_vertical_angle", 30.0f, { "FOV vertical angle" });
    sceneClass.setMetadata(attrFOVVerticalAngle, "label", "FOV vertical angle");
    attrFOVHorizontalAngle = sceneClass.declareAttribute<rdl2::Float>("FOV_horizontal_angle", 60.0f, { "FOV horizontal angle" });
    sceneClass.setMetadata(attrFOVHorizontalAngle, "label", "FOV horizontal angle");
    attrFlipRayX = sceneClass.declareAttribute<rdl2::Bool>("flip_ray_x", false, { "flip ray x" });
    sceneClass.setMetadata(attrFlipRayX, "label", "flip ray x");
    attrFlipRayY = sceneClass.declareAttribute<rdl2::Bool>("flip_ray_y", false, { "flip ray y" });
    sceneClass.setMetadata(attrFlipRayY, "label", "flip ray y");
    attrCamera = sceneClass.declareAttribute<rdl2::Int>("stereo_view", 0, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "stereo view" });
    sceneClass.setMetadata(attrCamera, "label", "stereo view");
    sceneClass.setEnumValue(attrCamera, 0, "center view");
    sceneClass.setEnumValue(attrCamera, 1, "left view");
    sceneClass.setEnumValue(attrCamera, 2, "right view");

    attrParallaxDistance = sceneClass.declareAttribute<rdl2::Float>("stereo_convergence_distance", 360.0f, { "stereo convergence distance" });
    sceneClass.setMetadata(attrParallaxDistance, "label", "stereo convergence distance");
    attrCameraSeparation = sceneClass.declareAttribute<rdl2::Float>("stereo_interocular_distance", 6.5f, { "stereo interocular distance" });
    sceneClass.setMetadata(attrCameraSeparation, "label", "stereo interocular distance");
    attrCameraSeparationMapFileName = sceneClass.declareAttribute<rdl2::String>("interocular_distance_map_file_name", rdl2::FLAGS_FILENAME, rdl2::INTERFACE_GENERIC, { "interocular distance map file name" });
    sceneClass.setMetadata(attrCameraSeparationMapFileName, "label", "interocular distance map file name");
    attrHeadTiltMap = sceneClass.declareAttribute<rdl2::Float>("head_tilt_map", 1.0f, { "head tilt map" });
    sceneClass.setMetadata(attrHeadTiltMap, "label", "head tilt map");
    attrZenithMode = sceneClass.declareAttribute<rdl2::Bool>("zenith_mode", false, { "zenith mode" });
    sceneClass.setMetadata(attrZenithMode, "label", "zenith mode");

    // Grouping the attributes for Torch - the order of
    // the attributes should be the same as how they are defined.
    sceneClass.setGroup("Stereo", attrCamera);
    sceneClass.setGroup("Stereo", attrCameraSeparation);
    sceneClass.setGroup("Stereo", attrCameraSeparationMapFileName);
    sceneClass.setGroup("Stereo", attrParallaxDistance);
    sceneClass.setGroup("Stereo", attrHeadTiltMap);
    sceneClass.setGroup("Stereo", attrZenithMode);

RDL2_DSO_ATTR_END

