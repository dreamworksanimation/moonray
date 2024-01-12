// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>        attrFalloffNear;
    rdl2::AttributeKey<rdl2::Bool>        attrFalloffFar;
    rdl2::AttributeKey<rdl2::Float>       attrNearStart;
    rdl2::AttributeKey<rdl2::Float>       attrNearEnd;
    rdl2::AttributeKey<rdl2::Float>       attrFarStart;
    rdl2::AttributeKey<rdl2::Float>       attrFarEnd;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrFalloffNear = sceneClass.declareAttribute<rdl2::Bool>("falloff_near", false, { "falloff near" });
    sceneClass.setMetadata(attrFalloffNear, "label", "falloff near");
    sceneClass.setMetadata(attrFalloffNear, "comment", "does the light fade in?");

    attrFalloffFar = sceneClass.declareAttribute<rdl2::Bool>("falloff_far", false, { "falloff far" });
    sceneClass.setMetadata(attrFalloffFar, "label", "falloff far");
    sceneClass.setMetadata(attrFalloffFar, "comment", "does the light fade out?");

    attrNearStart = sceneClass.declareAttribute<rdl2::Float>("near_start", 0.0f, { "near start" });
    sceneClass.setMetadata(attrNearStart, "label", "near start");
    sceneClass.setMetadata(attrNearStart, "comment", "distance from light to start of fade in");
    sceneClass.setMetadata(attrNearStart, "disable when", "{ falloff_near == 0 }");

    attrNearEnd = sceneClass.declareAttribute<rdl2::Float>("near_end", 0.0f, { "near end" });
    sceneClass.setMetadata(attrNearEnd, "label", "near end");
    sceneClass.setMetadata(attrNearEnd, "comment", "distance from light to end of fade in");
    sceneClass.setMetadata(attrNearEnd, "disable when", "{ falloff_near == 0 }");

    attrFarStart = sceneClass.declareAttribute<rdl2::Float>("far_start", 0.0f, { "far start" });
    sceneClass.setMetadata(attrFarStart, "label", "far start");
    sceneClass.setMetadata(attrFarStart, "comment", "distance from light to start of fade out");
    sceneClass.setMetadata(attrFarStart, "disable when", "{ falloff_far == 0 }");

    attrFarEnd = sceneClass.declareAttribute<rdl2::Float>("far_end", 0.0f, { "far end" });
    sceneClass.setMetadata(attrFarEnd, "label", "far end");
    sceneClass.setMetadata(attrFarEnd, "comment", "distance from light to end of fade out");
    sceneClass.setMetadata(attrFarEnd, "disable when", "{ falloff_far == 0 }");

    sceneClass.setGroup("Properties", attrFalloffNear);
    sceneClass.setGroup("Properties", attrFalloffFar);
    sceneClass.setGroup("Properties", attrNearStart);
    sceneClass.setGroup("Properties", attrNearEnd);
    sceneClass.setGroup("Properties", attrFarStart);
    sceneClass.setGroup("Properties", attrFarEnd);

RDL2_DSO_ATTR_END




