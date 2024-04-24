// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>  attrInsideOut;
    rdl2::AttributeKey<rdl2::Float> attrOffsetRadius;

    RDL2_DSO_ATTR_DEFINE(rdl2::Camera)

    attrInsideOut = sceneClass.declareAttribute<rdl2::Bool>("inside_out", false);
    sceneClass.setMetadata(attrInsideOut, "label", "inside out");

    attrOffsetRadius = sceneClass.declareAttribute<rdl2::Float>("offset_radius", 0.0f);
    sceneClass.setMetadata(attrOffsetRadius, "label", "offset radius");

RDL2_DSO_ATTR_END

