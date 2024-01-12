// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::SceneObjectVector> attrLightFilters;
    rdl2::AttributeKey<rdl2::Int>               attrMode;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrLightFilters = sceneClass.declareAttribute<rdl2::SceneObjectVector>("light_filters");
    sceneClass.setMetadata(attrLightFilters, "comment", "List of light filters to combine together");

    attrMode = sceneClass.declareAttribute<rdl2::Int>("mode", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrMode, 0, "multiply");
    sceneClass.setEnumValue(attrMode, 1, "min");
    sceneClass.setEnumValue(attrMode, 2, "max");
    sceneClass.setEnumValue(attrMode, 3, "add");
    sceneClass.setEnumValue(attrMode, 4, "subtract");
    sceneClass.setMetadata(attrMode, "comment", "How the light filters are combined");

RDL2_DSO_ATTR_END

