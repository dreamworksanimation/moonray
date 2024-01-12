// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>      attrNormalized;

    rdl2::AttributeKey<rdl2::Float>     attrAngularExtent;

RDL2_DSO_ATTR_DEFINE(rdl2::Light)

    attrNormalized = sceneClass.declareAttribute<rdl2::Bool>("normalized", true);
    sceneClass.setMetadata(attrNormalized, rdl2::SceneClass::sComment,
        "If set to true, a normalisation factor is applied to the light's radiance. "
        "The normalisation factor used is such that if a distant light of radiance (1,1,1) is directly overhead a "
        "Lambertian surface of colour (1,1,1), the  resulting outgoing radiance at the surface will be (1,1,1) "
        "regardless of the light's angular extent.\n"
        "If set to false, the light's radiance is used as-is.");

    attrAngularExtent = sceneClass.declareAttribute<rdl2::Float>("angular_extent", 0.53f, { "angular extent" });
    sceneClass.setMetadata(attrAngularExtent, "label", "angular extent");
    sceneClass.setMetadata(attrAngularExtent, rdl2::SceneClass::sComment,
        "The angle in degrees subtended by the DistantLight's diameter.");
    
    sceneClass.setGroup("Properties", attrNormalized);
    sceneClass.setGroup("Properties", attrAngularExtent);

RDL2_DSO_ATTR_END

