// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Mat4d>         attrNodeXform;
    rdl2::AttributeKey<rdl2::Float>         attrWidth;
    rdl2::AttributeKey<rdl2::Float>         attrDepth;
    rdl2::AttributeKey<rdl2::Float>         attrHeight;
    rdl2::AttributeKey<rdl2::Float>         attrRadius;
    rdl2::AttributeKey<rdl2::Float>         attrEdge;
    rdl2::AttributeKey<rdl2::Rgb>           attrColor;
    rdl2::AttributeKey<rdl2::Float>         attrIntensity;
    rdl2::AttributeKey<rdl2::Float>         attrDensity;
    rdl2::AttributeKey<rdl2::Bool>          attrInvert;
    rdl2::AttributeKey<rdl2::FloatVector>   attrRampIn;
    rdl2::AttributeKey<rdl2::FloatVector>   attrRampOut;
    rdl2::AttributeKey<rdl2::IntVector>     attrRampInterpolationTypes;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrNodeXform   = sceneClass.declareAttribute<rdl2::Mat4d>("node_xform",
        rdl2::FLAGS_BLURRABLE, rdl2::INTERFACE_GENERIC, { "node xform" });
    attrWidth       = sceneClass.declareAttribute<rdl2::Float>("width",     1.0f);
    attrDepth       = sceneClass.declareAttribute<rdl2::Float>("depth",     1.0f);
    attrHeight      = sceneClass.declareAttribute<rdl2::Float>("height",    1.0f);
    attrRadius      = sceneClass.declareAttribute<rdl2::Float>("radius",    0.0f);
    attrEdge        = sceneClass.declareAttribute<rdl2::Float>("edge",      0.0f);
    attrColor       = sceneClass.declareAttribute<rdl2::Rgb>("color", rdl2::Rgb(0.0f, 0.0f, 0.0f));
    attrIntensity   = sceneClass.declareAttribute<rdl2::Float>("intensity", 1.0f);
    attrDensity     = sceneClass.declareAttribute<rdl2::Float>("density",   1.0f);
    attrInvert      = sceneClass.declareAttribute<rdl2::Bool>("invert",     false);
    
    rdl2::FloatVector distanceDefaults = {0.f, 1.f};
    //use MONOTONECUBIC as default so it would default "smoothstep" with 0,1 as default inputs/outputs
    rdl2::IntVector interpolationDefaults = {6, 6};

    attrRampIn      = sceneClass.declareAttribute<rdl2::FloatVector>("ramp_in_distances", distanceDefaults);
    attrRampOut     = sceneClass.declareAttribute<rdl2::FloatVector>("ramp_out_distances", distanceDefaults);
    attrRampInterpolationTypes = sceneClass.declareAttribute<rdl2::IntVector>("ramp_interpolation_types", 
        interpolationDefaults);

    sceneClass.setMetadata(attrNodeXform,               "label", "node xform");
    sceneClass.setMetadata(attrWidth,                   "label", "width");
    sceneClass.setMetadata(attrDepth,                   "label", "depth");
    sceneClass.setMetadata(attrHeight,                  "label", "height");
    sceneClass.setMetadata(attrRadius,                  "label", "radius");
    sceneClass.setMetadata(attrEdge,                    "label", "edge");
    sceneClass.setMetadata(attrColor,                   "label", "color");
    sceneClass.setMetadata(attrIntensity,               "label", "intensity");
    sceneClass.setMetadata(attrDensity,                 "label", "density");
    sceneClass.setMetadata(attrInvert,                  "label", "invert");
    sceneClass.setMetadata(attrRampIn,                  "label", "ramp_in");
    sceneClass.setMetadata(attrRampOut,                 "label", "ramp_out");

    sceneClass.setMetadata(attrRampInterpolationTypes,  "label", "ramp_interpolation_types");
    sceneClass.setMetadata(attrRampInterpolationTypes, "comment",
        "0 means None. 1 means Linear. 2 means Exponential. 3 means Exponential Down. 4 means Smooth. "
        "5 means Catmull Rom. 6 means Monotone Cubic.");

    sceneClass.setMetadata(attrNodeXform, "comment",
                           "Transform of the filter.");
    sceneClass.setMetadata(attrWidth,     "comment",
                           "Width of the base box (before radius and edge).");
    sceneClass.setMetadata(attrDepth,     "comment",
                           "Depth of the base box (before radius and edge).");
    sceneClass.setMetadata(attrHeight,    "comment",
                           "Height of the base box (before radius and edge).");
    sceneClass.setMetadata(attrRadius,    "comment",
                           "Radius by which to expand the base box into a "
                           "rounded box.");
    sceneClass.setMetadata(attrEdge, "comment",
                           "Size of transition zone from the "
                           "rounded box to the outside.");
    sceneClass.setMetadata(attrColor,   "comment",
                           "Filter color. Scales the light within the volume. "
                           "For each color channel: 0 means full shadow, 1 means no shadow.");
    sceneClass.setMetadata(attrIntensity, "comment", "Scalar for multiplying the color. "
                           "0 means black, 1 means no change.");
    sceneClass.setMetadata(attrDensity, "comment", "Fades the filter effect. "
                           "0 means no effect (like having no filter). 1 means full effect.");
    sceneClass.setMetadata(attrInvert,    "comment", "Swap application of "
                           "filter from inside the volume to outside.");
                           
    sceneClass.setMetadata(attrRampIn,    "comment", "Input distance for ramp control.");
    sceneClass.setMetadata(attrRampIn, "label", "input distance for ramp control");
    sceneClass.setMetadata(attrRampIn, "structure_name", "ramp");
    sceneClass.setMetadata(attrRampIn, "structure_path", "positions");
    sceneClass.setMetadata(attrRampIn, "structure_type", "ramp_float");

    sceneClass.setMetadata(attrRampOut,   "comment", "Remapped distances for ramp control.");
    sceneClass.setMetadata(attrRampOut, "label", "remapped distances for ramp control");
    sceneClass.setMetadata(attrRampOut, "structure_name", "ramp");
    sceneClass.setMetadata(attrRampOut, "structure_path", "values");
    sceneClass.setMetadata(attrRampOut, "structure_type", "ramp_float");  

    sceneClass.setMetadata(attrRampInterpolationTypes,   "comment", "Interpolation types"
                            " for ramp control.");
    sceneClass.setMetadata(attrRampInterpolationTypes, "label", "interpolation types");
    sceneClass.setMetadata(attrRampInterpolationTypes, "structure_name", "ramp");
    sceneClass.setMetadata(attrRampInterpolationTypes, "structure_path", "interpolation_types");
    sceneClass.setMetadata(attrRampInterpolationTypes, "structure_type", "ramp_float");

    sceneClass.setGroup("Properties", attrNodeXform);
    sceneClass.setGroup("Properties", attrWidth);
    sceneClass.setGroup("Properties", attrDepth);
    sceneClass.setGroup("Properties", attrHeight);
    sceneClass.setGroup("Properties", attrRadius);
    sceneClass.setGroup("Properties", attrEdge);
    sceneClass.setGroup("Properties", attrColor);
    sceneClass.setGroup("Properties", attrIntensity);
    sceneClass.setGroup("Properties", attrDensity);
    sceneClass.setGroup("Properties", attrInvert);
    sceneClass.setGroup("Properties", attrRampIn);
    sceneClass.setGroup("Properties", attrRampOut);
    sceneClass.setGroup("Properties", attrRampInterpolationTypes);

RDL2_DSO_ATTR_END

