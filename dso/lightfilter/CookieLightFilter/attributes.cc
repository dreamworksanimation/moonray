// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE
    rdl2::AttributeKey<rdl2::SceneObject*> attrProjector;
    rdl2::AttributeKey<rdl2::Mat4d>        attrProjectorXformKey;
    rdl2::AttributeKey<rdl2::Int>          attrProjectorType;
    rdl2::AttributeKey<rdl2::Float>        attrProjectorFocalKey;
    rdl2::AttributeKey<rdl2::Float>        attrProjectorFilmWidthApertureKey;
    rdl2::AttributeKey<rdl2::Float>        attrProjectorPixelAspectRatio;
    rdl2::AttributeKey<rdl2::SceneObject*> attrTextureMap;
    rdl2::AttributeKey<rdl2::Float>        attrBlurNearDistance;
    rdl2::AttributeKey<rdl2::Float>        attrBlurMidpoint;
    rdl2::AttributeKey<rdl2::Float>        attrBlurFarDistance;
    rdl2::AttributeKey<rdl2::Float>        attrBlurNearValue;
    rdl2::AttributeKey<rdl2::Float>        attrBlurMidValue;
    rdl2::AttributeKey<rdl2::Float>        attrBlurFarValue;
    rdl2::AttributeKey<rdl2::Int>          attrBlurType;
    rdl2::AttributeKey<rdl2::Int>          attrOutsideProjection;
    rdl2::AttributeKey<rdl2::Float>        attrDensity;
    rdl2::AttributeKey<rdl2::Bool>         attrInvert;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    // If a projector is specified, it overrides the node_xform and projector_* attributes below.
    attrProjector = sceneClass.declareAttribute<rdl2::SceneObject *>(
        "projector", rdl2::FLAGS_NONE, rdl2::INTERFACE_CAMERA);
    sceneClass.setMetadata(attrProjector, "comment", 
        "If a projector is specified, it overrides the node_xform and projector_* attributes");

    attrProjectorXformKey = sceneClass.declareAttribute<rdl2::Mat4d>("node_xform", rdl2::FLAGS_BLURRABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrProjectorXformKey, "comment", "Filter orientation");

    attrProjectorType = sceneClass.declareAttribute<rdl2::Int>("projector_type", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrProjectorType, 0, "perspective");
    sceneClass.setEnumValue(attrProjectorType, 1, "orthographic");
    sceneClass.setMetadata(attrProjectorType, "comment", 
        "Perspective or orthographic projection");

    attrProjectorFocalKey = sceneClass.declareAttribute<rdl2::Float>("projector_focal", 30.0f);
    sceneClass.setMetadata(attrProjectorFocalKey, "comment", 
        "Focal length of the lens when using perspective projection");

    attrProjectorFilmWidthApertureKey = sceneClass.declareAttribute<rdl2::Float>("projector_film_width_aperture", 24.0f);
    sceneClass.setMetadata(attrProjectorFilmWidthApertureKey, "comment", 
        "Size of the camera image plane");

    attrProjectorPixelAspectRatio = sceneClass.declareAttribute<rdl2::Float>("projector_pixel_aspect_ratio", 1.0);
    sceneClass.setMetadata(attrProjectorPixelAspectRatio, "comment", 
        "Aspect ratio of the projection");

    attrTextureMap = sceneClass.declareAttribute<rdl2::SceneObject *>(
        "texture_map", rdl2::FLAGS_NONE, rdl2::INTERFACE_MAP);
    sceneClass.setMetadata(attrTextureMap, "comment", 
        "Moonray map. Any Moonray map generator, checkerboard, noise, image map.  "
        "You may also add any of the map modifiers, color correct for example.  "
        "The default is an image map.");

    attrBlurNearDistance = sceneClass.declareAttribute<rdl2::Float>("blur_near_distance", 0.0f);
    sceneClass.setMetadata(attrBlurNearDistance, "comment", "Distance from cookie filter");

    attrBlurMidpoint = sceneClass.declareAttribute<rdl2::Float>("blur_midpoint", 0.5f);
    sceneClass.setMetadata(attrBlurMidpoint, "comment", "Distance from cookie filter");

    attrBlurFarDistance = sceneClass.declareAttribute<rdl2::Float>("blur_far_distance", 1.0f);
    sceneClass.setMetadata(attrBlurFarDistance, "comment", "Distance from cookie filter");

    attrBlurNearValue = sceneClass.declareAttribute<rdl2::Float>("blur_near_value", 0.0f);
    sceneClass.setMetadata(attrBlurNearValue, "min", "0.0");
    sceneClass.setMetadata(attrBlurNearValue, "max", "0.1");
    // The max blur value is 0.1 because this is the size of the blur filter in texture
    // uv space, and 0.1 is a very large filter radius.  Anything larger is not
    // useful and would be confusing for the user.
    sceneClass.setMetadata(attrBlurNearValue, "comment", 
        "Blur filter radius (in texture UV space) at the near distance");

    attrBlurMidValue = sceneClass.declareAttribute<rdl2::Float>("blur_mid_value", 0.0f);
    sceneClass.setMetadata(attrBlurMidValue, "min", "0.0");
    sceneClass.setMetadata(attrBlurMidValue, "max", "0.1");
    sceneClass.setMetadata(attrBlurMidValue, "comment", 
        "Blur filter radius (in texture UV space) at the mid distance");

    attrBlurFarValue = sceneClass.declareAttribute<rdl2::Float>("blur_far_value", 0.0f);
    sceneClass.setMetadata(attrBlurFarValue, "min", "0.0");
    sceneClass.setMetadata(attrBlurFarValue, "max", "0.1");
    sceneClass.setMetadata(attrBlurFarValue, "comment", 
        "Blur filter radius (in texture UV space) at the far distance");

    attrBlurType = sceneClass.declareAttribute<rdl2::Int>("blur_type", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrBlurType, 0, "gaussian");
    sceneClass.setEnumValue(attrBlurType, 1, "circular");
    sceneClass.setMetadata(attrBlurType, "comment", 
        "Gaussian or circular blur");

    attrOutsideProjection = sceneClass.declareAttribute<rdl2::Int>("outside_projection", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrOutsideProjection, 0, "black");
    sceneClass.setEnumValue(attrOutsideProjection, 1, "white");
    sceneClass.setEnumValue(attrOutsideProjection, 2, "default");
    sceneClass.setMetadata(attrOutsideProjection, "comment",
        "What happens outside the frustum of the projection camera.  "
        "Black (default), White, or Default (This uses the mode set on the Moonray map shader)");

    attrDensity = sceneClass.declareAttribute<rdl2::Float>("density", 1.0f);
    sceneClass.setMetadata(attrDensity, "min", "0.0");
    sceneClass.setMetadata(attrDensity, "max", "1.0");
    sceneClass.setMetadata(attrDensity, "comment",
        "Controls how much of the cookie is added to the light");

    attrInvert = sceneClass.declareAttribute<rdl2::Bool>("invert", false);
    sceneClass.setMetadata(attrInvert, "comment", "Inverts the map");

    sceneClass.setGroup("Properties", attrProjector);
    sceneClass.setGroup("Properties", attrTextureMap);
    sceneClass.setGroup("Properties", attrBlurNearDistance);
    sceneClass.setGroup("Properties", attrBlurMidpoint);
    sceneClass.setGroup("Properties", attrBlurFarDistance);
    sceneClass.setGroup("Properties", attrBlurNearValue);
    sceneClass.setGroup("Properties", attrBlurMidValue);
    sceneClass.setGroup("Properties", attrBlurFarValue);
    sceneClass.setGroup("Properties", attrBlurType);
    sceneClass.setGroup("Properties", attrOutsideProjection);
    sceneClass.setGroup("Properties", attrDensity);
    sceneClass.setGroup("Properties", attrInvert);

RDL2_DSO_ATTR_END

