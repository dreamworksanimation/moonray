// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_ATTR_DECLARE
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> attrAlbedo;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> attrAnisotropy;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> attrAttenuationIntensity;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> attrAttenuationFactor;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> attrAttenuationColor;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> attrEmissionIntensity;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb> attrEmissionColor;

RDL2_DSO_ATTR_DEFINE(scene_rdl2::rdl2::VolumeShader)
    attrAlbedo = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "diffuse_color", scene_rdl2::rdl2::Rgb(1.0f, 1.0f, 1.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "diffuse color" });
    sceneClass.setMetadata(attrAlbedo, "label", "diffuse color");
    sceneClass.setMetadata(attrAlbedo, "comment",
        "reflectance color of the volume. "
        "Technically this is called scattering albedo, which is "
        "the scattering coefficient divided by the extinction coefficient.");
    sceneClass.setGroup("Scattering Properties", attrAlbedo);

    attrAnisotropy = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "anisotropy", 0.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrAnisotropy, "comment",
        "Value in the interval [-1,1] that defines how foward (1) or "
        "backward (-1) scattering the volume is. 0.0 is isotropic.");
    sceneClass.setGroup("Scattering Properties", attrAnisotropy);


    attrAttenuationIntensity = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "attenuation_intensity", 1.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "attenuation intensity" });
    sceneClass.setMetadata(attrAttenuationIntensity, "label", "attenuation intensity");
    sceneClass.setMetadata(attrAttenuationIntensity, "comment",
        "the rate at which the intensity of a ray traversing a volume is lost. "
        "The attenuation (extinction) coefficient is technically the product of "
        "attenuation_color, attenuation_intensity, and attenuation_factor");
    sceneClass.setGroup("Attenuation Properties", attrAttenuationIntensity);

    attrAttenuationFactor = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "attenuation_factor", 1.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "attenuation factor" });
    sceneClass.setMetadata(attrAttenuationFactor, "label", "attenuation factor");
    sceneClass.setMetadata(attrAttenuationFactor, "comment",
        "An additional factor to scale the attenuation. This attribute behaves "
        "identically to attenuation_intensity - it is provided simply as an "
        "extra way to control attenuation, typically during lighting. Surfacing"
        " should generally avoid setting this.");
    sceneClass.setGroup("Attenuation Properties", attrAttenuationFactor);

    attrAttenuationColor = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "attenuation_color", scene_rdl2::rdl2::Rgb(1.0f, 1.0f, 1.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "attenuation color" });
    sceneClass.setMetadata(attrAttenuationColor, "label", "attenuation color");
    sceneClass.setMetadata(attrAttenuationColor, "comment",
        "a color to tint (multiply to) the attenuation. "
        "Technically the product of attenuation color and intensity is "
        "the attenuation(extinction) coefficient."
        "(Note the inverse behavior of color with this parameter.)");
    sceneClass.setGroup("Attenuation Properties", attrAttenuationColor);


    attrEmissionIntensity = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "emission_intensity", 1.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "emission intensity" });
    sceneClass.setMetadata(attrEmissionIntensity, "label", "emission intensity");
    sceneClass.setMetadata(attrEmissionIntensity, "comment",
        "the rate at which a volume emits light at a given point. "
        "Technically the product of emission color and intensity is "
        "the emission coefficient.");
    sceneClass.setGroup("Emission Properties", attrEmissionIntensity);

    attrEmissionColor = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "emission_color", scene_rdl2::rdl2::Rgb(0.0f, 0.0f, 0.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "emission color" });
    sceneClass.setMetadata(attrEmissionColor, "label", "emission color");
    sceneClass.setMetadata(attrEmissionColor, "comment",
        "a color to tint (multiply to) the emission "
        "Technically the product of emision color and intensity is "
        "the emission coefficient");
    sceneClass.setGroup("Emission Properties", attrEmissionColor);

RDL2_DSO_ATTR_END

