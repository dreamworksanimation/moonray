// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_ATTR_DECLARE
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*> attrIndirectVolume;

RDL2_DSO_ATTR_DEFINE(scene_rdl2::rdl2::VolumeShader)

    attrIndirectVolume = sceneClass.declareAttribute<scene_rdl2::rdl2::SceneObject*>(
        "indirect_volume",
        scene_rdl2::rdl2::FLAGS_NONE,
        scene_rdl2::rdl2::INTERFACE_VOLUMESHADER,
        {"indirect volume"});

    sceneClass.setMetadata(attrIndirectVolume, "comment", 
        "The volume to cutout / use for indirect illumination and occlusion.  Cutout "
        "behavior is invoked for primary rays but secondary/indirect rays are processed "
        "normally.");

RDL2_DSO_ATTR_END

