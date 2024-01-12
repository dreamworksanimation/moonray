// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_ATTR_DECLARE
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb>           attrAlbedoColor;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>          attrUseAlbedoRamp;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::RgbVector>     attrAlbedoColors;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector>   attrAlbedoDistances;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector>     attrAlbedoInterpolationTypes;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAlbedoMinDepth;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAlbedoMaxDepth;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAnisotropy;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAttenuationIntensity;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAttenuationFactor;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb>           attrAttenuationColor;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>          attrMatchAlbedo;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>          attrInvertAttenuationColor;
    // Attenuation ramp attributes
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>          attrUseAttenuationRamp;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::RgbVector>     attrAttenuationColors;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector>   attrAttenuationDistances;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector>     attrAttenuationInterpolationTypes;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAttenuationMinDepth;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrAttenuationMaxDepth;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrEmissionIntensity;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb>           attrEmissionColor;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector>   attrDensities;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::FloatVector>   attrDensityDistances;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::IntVector>     attrDensityInterpolationTypes;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrDensityMinDepth;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>         attrDensityMaxDepth;

RDL2_DSO_ATTR_DEFINE(scene_rdl2::rdl2::VolumeShader)

    // -------------------------------------- ALBEDO ATTRIBUTES -------------------------------------------------------

    scene_rdl2::rdl2::RgbVector albedoColorDefaults = {
        scene_rdl2::rdl2::Rgb(1.f, 1.f, 1.f),
        scene_rdl2::rdl2::Rgb(0.f, 0.f, 0.f) };
    scene_rdl2::rdl2::FloatVector albedoDistanceDefaults = {0.f, 1.f};
    scene_rdl2::rdl2::IntVector albedoInterpolationDefaults = {1, 1}; // 1 = linear

    // diffuse_color
    attrAlbedoColor = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "diffuse_color", albedoColorDefaults[0], scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "diffuse color" });
    sceneClass.setMetadata(attrAlbedoColor, "label", "diffuse color");
    sceneClass.setMetadata(attrAlbedoColor, "comment",
        "Reflectance color of the volume. "
        "Technically this is called scattering albedo, which is "
        "the scattering coefficient divided by the extinction coefficient.");
    sceneClass.setMetadata(attrAlbedoColor, "disable when", "{ use_diffuse_ramp == 1 }");

    // use_albedo_ramp
    attrUseAlbedoRamp = sceneClass.declareAttribute<scene_rdl2::rdl2::Bool>("use_diffuse_ramp", false, {});
    sceneClass.setMetadata(attrUseAlbedoRamp, "label", "use ramp");
    sceneClass.setMetadata(attrUseAlbedoRamp, "comment",
        "Use a ramp to define different diffuse colors depending on the depth of the volume.");

    // diffuse_colors
    attrAlbedoColors = sceneClass.declareAttribute<scene_rdl2::rdl2::RgbVector>(
        "diffuse_colors", albedoColorDefaults, {});
    sceneClass.setMetadata(attrAlbedoColors, "label", "diffuse colors");
    sceneClass.setMetadata(attrAlbedoColors, "structure_path", "values");
    sceneClass.setMetadata(attrAlbedoColors, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrAlbedoColors, "disable when", "{ use_diffuse_ramp == 0 }");

    // diffuse_distances
    attrAlbedoDistances = sceneClass.declareAttribute<scene_rdl2::rdl2::FloatVector>(
        "diffuse_distances", albedoDistanceDefaults, {});
    sceneClass.setMetadata(attrAlbedoDistances, "label", "diffuse distances");
    sceneClass.setMetadata(attrAlbedoDistances, "structure_path", "positions");
    sceneClass.setMetadata(attrAlbedoDistances, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrAlbedoDistances, "disable when", "{ use_diffuse_ramp == 0 }");

    // diffuse_interpolations
    attrAlbedoInterpolationTypes = sceneClass.declareAttribute<scene_rdl2::rdl2::IntVector>(
        "diffuse_interpolations", albedoInterpolationDefaults, {});
    sceneClass.setMetadata(attrAlbedoInterpolationTypes, "label", "diffuse interpolations");
    sceneClass.setMetadata(attrAlbedoInterpolationTypes, "structure_path", "interpolation_types");
    sceneClass.setMetadata(attrAlbedoInterpolationTypes, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrAlbedoInterpolationTypes, "disable when", "{ use_diffuse_ramp == 0 }");

    // diffuse_min_depth
    attrAlbedoMinDepth = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "diffuse_min_depth", 1.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "diffuse min depth" });
    sceneClass.setMetadata(attrAlbedoMinDepth, "label", "min depth");
    sceneClass.setMetadata(attrAlbedoMinDepth, "comment",
        "Represents the minimum ray depth, or the shortest visible distance a ray has to travel through the volume. "
        "This sets the lower bound for the ramp. ");
    sceneClass.setMetadata(attrAlbedoMinDepth, "disable when", "{ use_diffuse_ramp == 0 }");

    // diffuse_max_depth
    attrAlbedoMaxDepth = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "diffuse_max_depth", 2.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "max depth" });
    sceneClass.setMetadata(attrAlbedoMaxDepth, "label", "max depth");
    sceneClass.setMetadata(attrAlbedoMaxDepth, "comment",
        "Represents the maximum ray depth, or the longest visible distance a ray has to travel through the volume. "
        "This sets the upper bound for the ramp. ");
    sceneClass.setMetadata(attrAlbedoMaxDepth, "disable when", "{ use_diffuse_ramp == 0 }");

    attrAnisotropy = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "anisotropy", 0.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrAnisotropy, "comment",
        "Value in the interval [-1,1] that defines how foward (1) or "
        "backward (-1) scattering the volume is.  A value of 0.0 indicates an isotropic volume.");

    sceneClass.setGroup("Scattering Properties", attrAnisotropy);
    sceneClass.setGroup("Scattering Properties", attrAlbedoColor);
    sceneClass.setGroup("Scattering Properties", attrUseAlbedoRamp);
    sceneClass.setGroup("Scattering Properties", attrAlbedoColors);
    sceneClass.setGroup("Scattering Properties", attrAlbedoDistances);
    sceneClass.setGroup("Scattering Properties", attrAlbedoInterpolationTypes);
    sceneClass.setGroup("Scattering Properties", attrAlbedoMinDepth);
    sceneClass.setGroup("Scattering Properties", attrAlbedoMaxDepth);

    sceneClass.setMetadata(attrAlbedoColors,             "structure_name", "diffuse_color_ramp");
    sceneClass.setMetadata(attrAlbedoDistances,          "structure_name", "diffuse_color_ramp");
    sceneClass.setMetadata(attrAlbedoInterpolationTypes, "structure_name", "diffuse_color_ramp");

   // ----------------------------------------- EXTINCTION ATTRIBUTES --------------------------------------------------

    scene_rdl2::rdl2::RgbVector attenuationColorDefaults = {
        scene_rdl2::rdl2::Rgb(1.f, 1.f, 1.f),
        scene_rdl2::rdl2::Rgb(0.f, 0.f, 0.f) };
    scene_rdl2::rdl2::FloatVector attenuationDistanceDefaults = {0.f, 1.f};
    scene_rdl2::rdl2::IntVector attenuationInterpolationDefaults = {1, 1}; // 1 = linear
    std::string attenuationDisableRampCondition = "{ use_attenuation_ramp == 0 } { match_diffuse == 1 }";

    // attenuation_color
    attrAttenuationColor = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "attenuation_color", attenuationColorDefaults[0], scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "attenuation color" });
    sceneClass.setMetadata(attrAttenuationColor, "label", "attenuation color");
    sceneClass.setMetadata(attrAttenuationColor, "comment",
        "A color to tint (multiply to) the attenuation. "
        "Technically the product of attenuation color and intensity is "
        "the attenuation (extinction) coefficient."
        "(Note the inverse behavior of color with this parameter.)");
    sceneClass.setMetadata(attrAttenuationColor, "disable when", "{ use_attenuation_ramp == 1 } { match_diffuse == 1 }");

    // attenuation_intensity
    attrAttenuationIntensity = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "attenuation_intensity", 1.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "attenuation intensity" });
    sceneClass.setMetadata(attrAttenuationIntensity, "label", "attenuation intensity");
    sceneClass.setMetadata(attrAttenuationIntensity, "comment",
        "The rate at which the light traversing a volume is attenuated. "
        "The attenuation (extinction) coefficient is the product of "
        "attenuation_color, attenuation_intensity, and attenuation_factor");
    
    // attenuation_factor
    attrAttenuationFactor = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "attenuation_factor", 1.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "attenuation factor" });
    sceneClass.setMetadata(attrAttenuationFactor, "label", "attenuation factor");
    sceneClass.setMetadata(attrAttenuationFactor, "comment",
        "Identical in behavior to attenuation_intensity but provided as a second means "
        " to control attenuation, intended for use during lighting as a per-shot or "
        " per-sequence adjustment.");

    // match_diffuse
    attrMatchAlbedo = sceneClass.declareAttribute<scene_rdl2::rdl2::Bool>("match_diffuse", false, {});
    sceneClass.setMetadata(attrMatchAlbedo, "label", "match diffuse color(s)");
    sceneClass.setMetadata(attrMatchAlbedo, "comment",
        "Use the same color(s) for attenuation that is/are being used for diffuse.");

    // invert_attenuation_color
    attrInvertAttenuationColor = sceneClass.declareAttribute<scene_rdl2::rdl2::Bool>(
        "invert_attenuation_color", false, {});
    sceneClass.setMetadata(attrInvertAttenuationColor, "label", "invert attenuation color(s)");
    sceneClass.setMetadata(attrInvertAttenuationColor, "comment",
        "Invert the input attenuation color(s).");

    // use_attenuation_ramp
    attrUseAttenuationRamp = sceneClass.declareAttribute<scene_rdl2::rdl2::Bool>("use_attenuation_ramp", false, {});
    sceneClass.setMetadata(attrUseAttenuationRamp, "label", "use ramp");
    sceneClass.setMetadata(attrUseAttenuationRamp, "comment",
        "Use a ramp to define different attenuation colors depending on the depth of the volume.");
    sceneClass.setMetadata(attrUseAttenuationRamp, "disable when", "{ match_diffuse == 1 }");

    // RAMP: attenuation_colors
    attrAttenuationColors = sceneClass.declareAttribute<scene_rdl2::rdl2::RgbVector>(
        "attenuation_colors", attenuationColorDefaults, {});
    sceneClass.setMetadata(attrAttenuationColors, "label", "attenuation colors");
    sceneClass.setMetadata(attrAttenuationColors, "structure_name", "attenuation_ramp");
    sceneClass.setMetadata(attrAttenuationColors, "structure_path", "values");
    sceneClass.setMetadata(attrAttenuationColors, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrAttenuationColors, "disable when", attenuationDisableRampCondition);

    // RAMP: attenuation_distances
    attrAttenuationDistances = sceneClass.declareAttribute<scene_rdl2::rdl2::FloatVector>(
        "attenuation_distances", attenuationDistanceDefaults, {});
    sceneClass.setMetadata(attrAttenuationDistances, "label", "attenuation distances");
    sceneClass.setMetadata(attrAttenuationDistances, "structure_name", "attenuation_ramp");
    sceneClass.setMetadata(attrAttenuationDistances, "structure_path", "positions");
    sceneClass.setMetadata(attrAttenuationDistances, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrAttenuationDistances, "disable when", attenuationDisableRampCondition);

    // RAMP: attenuation_interpolations
    attrAttenuationInterpolationTypes = sceneClass.declareAttribute<scene_rdl2::rdl2::IntVector>(
        "attenuation_interpolations", attenuationInterpolationDefaults, {});
    sceneClass.setMetadata(attrAttenuationInterpolationTypes, "label", "attenuation interpolation types");
    sceneClass.setMetadata(attrAttenuationInterpolationTypes, "structure_name", "attenuation_ramp");
    sceneClass.setMetadata(attrAttenuationInterpolationTypes, "structure_path", "interpolation_types");
    sceneClass.setMetadata(attrAttenuationInterpolationTypes, "structure_type", "ramp_color");
    sceneClass.setMetadata(attrAttenuationInterpolationTypes, "disable when", attenuationDisableRampCondition);

    // attenuation_min_depth
    attrAttenuationMinDepth = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "attenuation_min_depth", 1.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "attenuation min depth" });
    sceneClass.setMetadata(attrAttenuationMinDepth, "label", "attenuation min depth");
    sceneClass.setMetadata(attrAttenuationMinDepth, "comment",
        "Represents the minimum ray depth, or the shortest visible distance a ray has to travel through the volume. "
        "This sets the lower bound for the ramp. ");
    sceneClass.setMetadata(attrAttenuationMinDepth, "disable when", attenuationDisableRampCondition);

    // attenuation_max_depth
    attrAttenuationMaxDepth = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "attenuation_max_depth", 2.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "attenuation max depth" });
    sceneClass.setMetadata(attrAttenuationMaxDepth, "label", "attenuation max depth");
    sceneClass.setMetadata(attrAttenuationMaxDepth, "comment",
        "Represents the maximum ray depth, or the longest visible distance a ray has to travel through the volume. "
        "This sets the upper bound for the ramp. ");
    sceneClass.setMetadata(attrAttenuationMaxDepth, "disable when", attenuationDisableRampCondition);

    sceneClass.setGroup("Attenuation Properties", attrAttenuationColor);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationIntensity);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationFactor);
    sceneClass.setGroup("Attenuation Properties", attrMatchAlbedo);
    sceneClass.setGroup("Attenuation Properties", attrInvertAttenuationColor);

    sceneClass.setGroup("Attenuation Properties", attrUseAttenuationRamp);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationColors);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationDistances);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationInterpolationTypes);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationMinDepth);
    sceneClass.setGroup("Attenuation Properties", attrAttenuationMaxDepth);


    // ------------------------------------------ EMISSION ATTRIBUTES --------------------------------------------------

    attrEmissionIntensity = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "emission_intensity", 1.0f,
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "emission intensity" });
    sceneClass.setMetadata(attrEmissionIntensity, "label", "emission intensity");
    sceneClass.setMetadata(attrEmissionIntensity, "comment",
        "The rate at which a volume emits light at a given point.  The product of emission color and intensity is "
        "the emission coefficient.");
    sceneClass.setGroup("Volume", attrEmissionIntensity);

    attrEmissionColor = sceneClass.declareAttribute<scene_rdl2::rdl2::Rgb>(
        "emission_color", scene_rdl2::rdl2::Rgb(0.0f, 0.0f, 0.0f),
        scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC,
        { "emission color" });
    sceneClass.setMetadata(attrEmissionColor, "label", "emission color");
    sceneClass.setMetadata(attrEmissionColor, "comment",
        "A color multiplier for the emission.  The product of emission color and intensity is "
        "the emission coefficient");
    sceneClass.setGroup("Volume", attrEmissionColor);


    // ---------------------------------------- DENSITY ATTRIBUTES -----------------------------------------------------

    scene_rdl2::rdl2::FloatVector densityDefaults = { 1.f, 1.f };
    scene_rdl2::rdl2::FloatVector densityDistanceDefaults = {0.f, 1.f};
    scene_rdl2::rdl2::IntVector densityInterpolationDefaults = {1, 1}; // 1 = linear

    attrDensities = sceneClass.declareAttribute<scene_rdl2::rdl2::FloatVector>("densities", densityDefaults, {});
    // the metadata configures the UI widget
    sceneClass.setMetadata(attrDensities, "label", "densities");
    sceneClass.setMetadata(attrDensities, "structure_name", "density_ramp");
    sceneClass.setMetadata(attrDensities, "structure_path", "values");
    sceneClass.setMetadata(attrDensities, "structure_type", "ramp_float");

    attrDensityDistances = sceneClass.declareAttribute<scene_rdl2::rdl2::FloatVector>(
        "density_distances", densityDistanceDefaults, {});
    sceneClass.setMetadata(attrDensityDistances, "label", "density distances");
    sceneClass.setMetadata(attrDensityDistances, "structure_name", "density_ramp");
    sceneClass.setMetadata(attrDensityDistances, "structure_path", "positions");
    sceneClass.setMetadata(attrDensityDistances, "structure_type", "ramp_float");

    attrDensityInterpolationTypes = sceneClass.declareAttribute<scene_rdl2::rdl2::IntVector>(
        "density_interpolations", densityInterpolationDefaults, {});
    sceneClass.setMetadata(attrDensityInterpolationTypes, "label", "density interpolations");
    sceneClass.setMetadata(attrDensityInterpolationTypes, "structure_name", "density_ramp");
    sceneClass.setMetadata(attrDensityInterpolationTypes, "structure_path", "interpolation_types");
    sceneClass.setMetadata(attrDensityInterpolationTypes, "structure_type", "ramp_float");

    attrDensityMinDepth = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "density_min_depth", 1.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "density min depth" });
    sceneClass.setMetadata(attrDensityMinDepth, "label", "min depth");
    sceneClass.setMetadata(attrDensityMinDepth, "comment",
        "Represents the minimum ray depth, or the shortest visible distance a ray has to travel through the volume. "
        "This sets the lower bound for the ramp. ");

    attrDensityMaxDepth = sceneClass.declareAttribute<scene_rdl2::rdl2::Float>(
        "density_max_depth", 2.0f, scene_rdl2::rdl2::AttributeFlags::FLAGS_BINDABLE, 
        scene_rdl2::rdl2::SceneObjectInterface::INTERFACE_GENERIC, { "density max depth" });
    sceneClass.setMetadata(attrDensityMaxDepth, "label", "max depth");
    sceneClass.setMetadata(attrDensityMaxDepth, "comment",
        "Represents the maximum ray depth, or the longest visible distance a ray has to travel through the volume. "
        "This sets the upper bound for the ramp. ");

    sceneClass.setGroup("Density Properties", attrDensities);
    sceneClass.setGroup("Density Properties", attrDensityDistances);
    sceneClass.setGroup("Density Properties", attrDensityInterpolationTypes);
    sceneClass.setGroup("Density Properties", attrDensityMinDepth);
    sceneClass.setGroup("Density Properties", attrDensityMaxDepth);

    sceneClass.setMetadata(attrDensities,                   "structure_name", "density_ramp");
    sceneClass.setMetadata(attrDensityDistances,            "structure_name", "density_ramp");
    sceneClass.setMetadata(attrDensityInterpolationTypes,   "structure_name", "density_ramp");

                                                        
RDL2_DSO_ATTR_END

