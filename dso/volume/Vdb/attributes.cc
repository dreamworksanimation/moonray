// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_ATTR_DECLARE

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> attrExtinctionGainMult;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> attrAlbedoMult;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> attrIncandGainMult;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> attrAnisotropy;

RDL2_DSO_ATTR_DEFINE(scene_rdl2::rdl2::VolumeShader)

    attrExtinctionGainMult = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "opacity_gain_mult", scene_rdl2::rdl2::Rgb(1.0f, 1.0f, 1.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "opacity gain mult" });
    sceneClass.setMetadata(attrExtinctionGainMult, "label", "opacity gain mult");
    sceneClass.setMetadata(attrExtinctionGainMult, "comment",
        "A multiplier applied to the volume density");
    sceneClass.setGroup("Volume", attrExtinctionGainMult);

    attrAlbedoMult = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "color_mult", scene_rdl2::rdl2::Rgb(1.0f, 1.0f, 1.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "color mult" });
    sceneClass.setMetadata(attrAlbedoMult, "label", "color mult");
    sceneClass.setMetadata(attrAlbedoMult, "comment",
        "The albedo of the volume");
    sceneClass.setGroup("Volume", attrAlbedoMult);
    
    attrIncandGainMult = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "incandescence_gain_mult", scene_rdl2::rdl2::Rgb(1.0f, 1.0f, 1.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "incandescence gain mult" });
    sceneClass.setMetadata(attrIncandGainMult, "label", "incandescence gain mult");
    sceneClass.setMetadata(attrIncandGainMult, "comment",
        "A multiplier applied to the volume emission");
    sceneClass.setGroup("Volume", attrIncandGainMult);

    attrAnisotropy = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "anisotropy", 0.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrAnisotropy, "comment",
        "Value in the interval [-1,1] that defines how foward (1) or "
        "backward (-1) scattering the volume is.  A value of 0.0 indicates an isotropic volume.");
    sceneClass.setGroup("Volume", attrAnisotropy);
                                                        
RDL2_DSO_ATTR_END

