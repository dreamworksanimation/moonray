// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE
    rdl2::AttributeKey<rdl2::Mat4d>        attrNodeXform;
    rdl2::AttributeKey<rdl2::Int>          attrProjectorType;
    rdl2::AttributeKey<rdl2::Float>        attrProjectorFocalDist;
    rdl2::AttributeKey<rdl2::Float>        attrProjectorWidth;
    rdl2::AttributeKey<rdl2::Float>        attrProjectorHeight;
    rdl2::AttributeKey<rdl2::Float>        attrEdgeScaleTop;
    rdl2::AttributeKey<rdl2::Float>        attrEdgeScaleBottom;
    rdl2::AttributeKey<rdl2::Float>        attrEdgeScaleLeft;
    rdl2::AttributeKey<rdl2::Float>        attrEdgeScaleRight;
    rdl2::AttributeKey<rdl2::Int>          attrPreBarnMode;
    rdl2::AttributeKey<rdl2::Float>        attrPreBarnDist;
    rdl2::AttributeKey<rdl2::Float>        attrDensity;
    rdl2::AttributeKey<rdl2::Bool>         attrInvert;
    rdl2::AttributeKey<rdl2::Float>        attrRadius;
    rdl2::AttributeKey<rdl2::Float>        attrEdge;
    rdl2::AttributeKey<rdl2::Int>          attrMode;          // analytical/physical
    rdl2::AttributeKey<rdl2::Float>        attrSizeTop;       // additional size in focal plane
    rdl2::AttributeKey<rdl2::Float>        attrSizeBottom;    // additional size in focal plane
    rdl2::AttributeKey<rdl2::Float>        attrSizeLeft;      // additional size in focal plane
    rdl2::AttributeKey<rdl2::Float>        attrSizeRight;     // additional size in focal plane
    rdl2::AttributeKey<rdl2::Bool>         attrUseLightXform; // use light position+orientation+scale
    rdl2::AttributeKey<rdl2::Float>        attrRotation;      // rotation of filter around the focal direction, in degrees
    rdl2::AttributeKey<rdl2::Rgb>          attrColor;


RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrNodeXform = sceneClass.declareAttribute<rdl2::Mat4d>("node_xform", rdl2::FLAGS_BLURRABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrNodeXform, "label", "node xform");
    sceneClass.setMetadata(attrNodeXform, "comment", "transform of the filter");

    attrProjectorType = sceneClass.declareAttribute<rdl2::Int>("projector_type", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setMetadata(attrProjectorType, "label", "projector type");
    sceneClass.setEnumValue(attrProjectorType, 0, "perspective");
    sceneClass.setEnumValue(attrProjectorType, 1, "orthographic");
    sceneClass.setMetadata(attrProjectorType, "comment",
        "projection type used to map points to the flap opening. perspective has a focal point, while orthographic does not.");

    attrProjectorFocalDist = sceneClass.declareAttribute<rdl2::Float>("projector_focal_distance", 30.0f);
    sceneClass.setMetadata(attrProjectorFocalDist, "label", "projector focal distance");
    sceneClass.setMetadata(attrProjectorFocalDist, "comment", "distance of the flap opening from the projector origin. Ignored for orthographic projection");
    sceneClass.setMetadata(attrProjectorFocalDist, "disable when", "{ mode == 'analytical' }");
    sceneClass.setMetadata(attrProjectorFocalDist, "min", "0.0");
    sceneClass.setMetadata(attrProjectorFocalDist, "max", "100.0");

    attrProjectorWidth = sceneClass.declareAttribute<rdl2::Float>("projector_width", 1.0f);
    sceneClass.setMetadata(attrProjectorWidth, "label", "width of the flap opening");
    sceneClass.setMetadata(attrProjectorWidth, "comment", "width of the frustum at distance 1.0");

    attrProjectorHeight = sceneClass.declareAttribute<rdl2::Float>("projector_height", 1.0);
    sceneClass.setMetadata(attrProjectorHeight, "label", "height of the flap opening");
    sceneClass.setMetadata(attrProjectorHeight, "comment", "height of the frustum at distance 1.0");

    attrPreBarnMode = sceneClass.declareAttribute<rdl2::Int>(
        "pre_barn_mode", 2, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setMetadata(attrPreBarnMode, "label", "pre barn mode");
    sceneClass.setEnumValue(attrPreBarnMode, 0, "black");
    sceneClass.setEnumValue(attrPreBarnMode, 1, "white");
    sceneClass.setEnumValue(attrPreBarnMode, 2, "default");
    sceneClass.setMetadata(attrPreBarnMode, "comment",
        "force region before the pre_barn_distance to be fully filtered (black), not filtered at all (white), or treated the same as elsewhere (default)");

    attrPreBarnDist = sceneClass.declareAttribute<rdl2::Float>("pre_barn_distance", 0.5f);
    sceneClass.setMetadata(attrPreBarnDist, "min", "0.0");
    sceneClass.setMetadata(attrPreBarnDist, "comment", "distance from the BarnDoorLightFilter that the pre_barn_mode control takes effect");

    attrEdgeScaleTop = sceneClass.declareAttribute<rdl2::Float>("edge_scale_top", 1.0f);
    sceneClass.setMetadata(attrEdgeScaleTop, "label", "edge scale top");
    sceneClass.setMetadata(attrEdgeScaleTop, "comment", "scale factor for top edge");

    attrEdgeScaleBottom = sceneClass.declareAttribute<rdl2::Float>("edge_scale_bottom", 1.0f);
    sceneClass.setMetadata(attrEdgeScaleBottom, "label", "edge scale bottom");
    sceneClass.setMetadata(attrEdgeScaleBottom, "comment", "scale factor for bottom edge");

    attrEdgeScaleLeft = sceneClass.declareAttribute<rdl2::Float>("edge_scale_left", 1.0f);
    sceneClass.setMetadata(attrEdgeScaleLeft, "label", "edge scale left");
    sceneClass.setMetadata(attrEdgeScaleLeft, "comment", "scale factor for left edge");

    attrEdgeScaleRight = sceneClass.declareAttribute<rdl2::Float>("edge_scale_right", 1.0f);
    sceneClass.setMetadata(attrEdgeScaleRight, "label", "edge scale right");
    sceneClass.setMetadata(attrEdgeScaleRight, "comment", "scale factor for right edge");

    attrDensity = sceneClass.declareAttribute<rdl2::Float>("density", 1.0f);
    sceneClass.setMetadata(attrDensity, "min", "0.0");
    sceneClass.setMetadata(attrDensity, "max", "1.0");
    sceneClass.setMetadata(attrDensity, "comment", "fades the filter effect. "
        "0=no effect (like having no filter), 1=full effect");

    attrInvert = sceneClass.declareAttribute<rdl2::Bool>("invert", false);
    sceneClass.setMetadata(attrInvert, "comment",
        "swap application of filter from inside the Barn Door to outside");

    attrRadius = sceneClass.declareAttribute<rdl2::Float>("radius", 0.f);
    sceneClass.setMetadata(attrRadius, "min", "0.0");
    sceneClass.setMetadata(attrRadius, "max", "1.0");
    sceneClass.setMetadata(attrRadius, "comment",
        "radius by which to convert the base box shape into a rounded box, as a proportion of half the width (or height, whichever is smaller)");

    attrEdge = sceneClass.declareAttribute<rdl2::Float>("edge", 0.f);
    sceneClass.setMetadata(attrEdge, "min", "0.0");
    sceneClass.setMetadata(attrEdge, "comment",
        "size of transition zone from the rounded box to the outside, as a proportion of width (or height, whichever is smaller)");

    attrMode = sceneClass.declareAttribute<rdl2::Int>("mode", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrMode, 0, "analytical");
    sceneClass.setEnumValue(attrMode, 1, "physical");
    sceneClass.setMetadata(attrMode, "comment",
        "analytical mode allows light to shading points that project to the flap opening."
        "physical mode allows light whose direction goes through the flap opening.");

    // Users felt physical mode was confusing when the filter was unattached to the light,
    // so this is a UI hint to disable it.
    sceneClass.setMetadata(attrMode, "disable when", "{ use_light_xform == 0 }");

    attrSizeTop = sceneClass.declareAttribute<rdl2::Float>("size_top", 0.f);
    sceneClass.setMetadata(attrSizeTop, "comment", "additional size on top edge");

    attrSizeBottom = sceneClass.declareAttribute<rdl2::Float>("size_bottom", 0.f);
    sceneClass.setMetadata(attrSizeBottom, "comment", "additional size on bottom edge");

    attrSizeLeft = sceneClass.declareAttribute<rdl2::Float>("size_left", 0.f);
    sceneClass.setMetadata(attrSizeLeft, "comment", "additional size on left edge");

    attrSizeRight = sceneClass.declareAttribute<rdl2::Float>("size_right", 0.f);
    sceneClass.setMetadata(attrSizeRight, "comment", "additional size on right edge");

    attrUseLightXform = sceneClass.declareAttribute<rdl2::Bool>("use_light_xform", true);
    sceneClass.setMetadata(attrUseLightXform, "label", "use light xform");
    sceneClass.setMetadata(attrUseLightXform, "comment",
        "attach to the light (in the -Z direction) and ignore node_xform");

    attrRotation = sceneClass.declareAttribute<rdl2::Float>("rotation", 0.0f);
    sceneClass.setMetadata(attrRotation, "comment",
        "angle to rotate the Barn Door counter-clockwise as seen from the light, in degrees");
    sceneClass.setMetadata(attrRotation, "min", "-180.0");
    sceneClass.setMetadata(attrRotation, "max", "180.0");

    attrColor = sceneClass.declareAttribute<rdl2::Rgb>("color",
        rdl2::Rgb(1.0f, 1.0f, 1.0f));
    sceneClass.setMetadata(attrColor, "label", "color");
    sceneClass.setMetadata(attrColor, "comment",
                           "Color within the Barn Door lit region. "
                           "For each color channel, 0=full shadow, 1=no shadow");

    sceneClass.setGroup("Properties", attrNodeXform);
    sceneClass.setGroup("Properties", attrProjectorType);
    sceneClass.setGroup("Properties", attrProjectorFocalDist);
    sceneClass.setGroup("Properties", attrProjectorWidth);
    sceneClass.setGroup("Properties", attrProjectorHeight);
    sceneClass.setGroup("Properties", attrEdgeScaleTop);
    sceneClass.setGroup("Properties", attrEdgeScaleBottom);
    sceneClass.setGroup("Properties", attrEdgeScaleLeft);
    sceneClass.setGroup("Properties", attrEdgeScaleRight);
    sceneClass.setGroup("Properties", attrPreBarnMode);
    sceneClass.setGroup("Properties", attrPreBarnDist);
    sceneClass.setGroup("Properties", attrDensity);
    sceneClass.setGroup("Properties", attrInvert);
    sceneClass.setGroup("Properties", attrRadius);
    sceneClass.setGroup("Properties", attrEdge);
    sceneClass.setGroup("Properties", attrMode);
    sceneClass.setGroup("Properties", attrSizeTop);
    sceneClass.setGroup("Properties", attrSizeBottom);
    sceneClass.setGroup("Properties", attrSizeLeft);
    sceneClass.setGroup("Properties", attrSizeRight);
    sceneClass.setGroup("Properties", attrUseLightXform);
    sceneClass.setGroup("Properties", attrRotation);
    sceneClass.setGroup("Properties", attrColor);


RDL2_DSO_ATTR_END

