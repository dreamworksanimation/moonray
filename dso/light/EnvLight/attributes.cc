// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>      attrSampleUpperHemisphereOnly;

RDL2_DSO_ATTR_DEFINE(rdl2::Light)

    // color is assumed to be a radiance value for envmaps, so there is no units

    attrSampleUpperHemisphereOnly = 
        sceneClass.declareAttribute<rdl2::Bool>("sample_upper_hemisphere_only", false, { "sample upper hemisphere only" });
    sceneClass.setMetadata(attrSampleUpperHemisphereOnly, "label", "sample upper hemisphere only");
    sceneClass.setMetadata(attrSampleUpperHemisphereOnly, rdl2::SceneClass::sComment,
        "Set this to true if you want the EnvLight to illuminate from only the \"upper\" hemisphere, defined "
        "as the hemisphere centered around the light's positive local z-axis direction.");
    
    sceneClass.setGroup("Map", attrSampleUpperHemisphereOnly);

RDL2_DSO_ATTR_END

