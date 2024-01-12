// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE
    rdl2::AttributeKey<rdl2::Mat4d>        attrVdbXformKey;
    rdl2::AttributeKey<rdl2::String>       attrVdbMap;
    rdl2::AttributeKey<rdl2::Float>        attrDensityRemapInputMin;
    rdl2::AttributeKey<rdl2::Float>        attrDensityRemapInputMax;
    rdl2::AttributeKey<rdl2::Float>        attrDensityRemapOutputMin;
    rdl2::AttributeKey<rdl2::Float>        attrDensityRemapOutputMax;
    rdl2::AttributeKey<rdl2::Bool>         attrDensityRemapRescaleEnable;
    rdl2::AttributeKey<rdl2::FloatVector>  attrDensityRemapOutputs;
    rdl2::AttributeKey<rdl2::FloatVector>  attrDensityRemapInputs;
    rdl2::AttributeKey<rdl2::IntVector>    attrDensityRemapInterpolationTypes;
    rdl2::AttributeKey<rdl2::String>       attrDensityGridName;
    rdl2::AttributeKey<rdl2::Rgb>          attrColorTint;
    rdl2::AttributeKey<rdl2::Int>          attrVdbInterpolation;
    rdl2::AttributeKey<rdl2::Float>        attrBlurValue;
    rdl2::AttributeKey<rdl2::Int>          attrBlurType;
    rdl2::AttributeKey<rdl2::Bool>         attrInvertDensity;
    rdl2::AttributeKey<rdl2::Bool>         attrInvertColor;

RDL2_DSO_ATTR_DEFINE(rdl2::LightFilter)

    attrVdbXformKey = sceneClass.declareAttribute<rdl2::Mat4d>("node_xform", rdl2::FLAGS_BLURRABLE, rdl2::INTERFACE_GENERIC);
    sceneClass.setMetadata(attrVdbXformKey, "comment", "The filter's orientation");

    attrVdbMap = sceneClass.declareAttribute<rdl2::String>(
        "vdb_map", rdl2::FLAGS_FILENAME, rdl2::INTERFACE_GENERIC, { "Vdb map" });
    sceneClass.setMetadata(attrVdbMap, "label", "Vdb map");
    sceneClass.setMetadata(attrVdbMap, "comment", "The path to the vdb");

    attrDensityGridName = sceneClass.declareAttribute<rdl2::String>(
        "density_grid_name", rdl2::FLAGS_NONE, rdl2::INTERFACE_GENERIC, { "Density Grid Name" });
    sceneClass.setMetadata(attrDensityGridName, "label", "density grid name");
    sceneClass.setMetadata(attrDensityGridName, "comment",
    "The name of the grid within the .vdb file from which to sample for density"
    "(hint: use openvdb_print to see contents of .vdb file). "
    "If no grid is specified, it will use 'density' as the default"
    "In cases where there are multiple grids with the same name, the grid name can be indexed (eg. density[1])");

    attrDensityRemapInputMin = sceneClass.declareAttribute<rdl2::Float>("density_remap_input_min", 0.f, {});
    sceneClass.setMetadata(attrDensityRemapInputMin, "label", "density remap input min");
    sceneClass.setMetadata(attrDensityRemapInputMin, "min", "-100.0");  // for UI slider
    sceneClass.setMetadata(attrDensityRemapInputMin, "max", "100.0");
    sceneClass.setMetadata(attrDensityRemapInputMin, "disable when", "{ density_rescale_enable == 0 }");
    sceneClass.setMetadata(attrDensityRemapInputMin, "comment", "Clamp the remapped input to this min value");

    attrDensityRemapInputMax = sceneClass.declareAttribute<rdl2::Float>("density_remap_input_max", 1.f, {});
    sceneClass.setMetadata(attrDensityRemapInputMax, "label", "density remap input max");
    sceneClass.setMetadata(attrDensityRemapInputMax, "min", "-100.0");
    sceneClass.setMetadata(attrDensityRemapInputMax, "max", "100.0");
    sceneClass.setMetadata(attrDensityRemapInputMax, "disable when", "{ density_rescale_enable == 0 }");
    sceneClass.setMetadata(attrDensityRemapInputMax, "comment", "Clamp the remapped input to this max value");

    attrDensityRemapOutputMin = sceneClass.declareAttribute<rdl2::Float>("density_remap_output_min", 0.f, {});
    sceneClass.setMetadata(attrDensityRemapOutputMin, "label", "density remap output min");
    sceneClass.setMetadata(attrDensityRemapOutputMin, "min", "-100.0");  // for UI slider
    sceneClass.setMetadata(attrDensityRemapOutputMin, "max", "100.0");
    sceneClass.setMetadata(attrDensityRemapOutputMin, "disable when", "{ density_rescale_enable == 0 }");
    sceneClass.setMetadata(attrDensityRemapOutputMin, "comment", "Clamp the remapped output to this min value");

    attrDensityRemapOutputMax = sceneClass.declareAttribute<rdl2::Float>("density_remap_output_max", 1.f, {});
    sceneClass.setMetadata(attrDensityRemapOutputMax, "label", "density remap output max");
    sceneClass.setMetadata(attrDensityRemapOutputMax, "min", "-100.0");
    sceneClass.setMetadata(attrDensityRemapOutputMax, "max", "100.0");
    sceneClass.setMetadata(attrDensityRemapOutputMax, "disable when", "{ density_rescale_enable == 0 }");
    sceneClass.setMetadata(attrDensityRemapOutputMax, "comment", "Clamp the remapped output to this max value");

    attrDensityRemapRescaleEnable = sceneClass.declareAttribute<rdl2::Bool>("density_rescale_enable", false, {});
    sceneClass.setMetadata(attrDensityRemapRescaleEnable, "label", "Enable Density Rescale");
    sceneClass.setMetadata(attrDensityRemapRescaleEnable, "comment", "Enable density rescaling");

    rdl2::FloatVector densityDefaults = {0.f, 1.f};

    attrDensityRemapInputs = sceneClass.declareAttribute<rdl2::FloatVector>("density_remap_inputs",
        densityDefaults, {});
    sceneClass.setMetadata(attrDensityRemapInputs, "label", "density remap input");
    sceneClass.setMetadata(attrDensityRemapInputs, "structure_name", "ramp");
    sceneClass.setMetadata(attrDensityRemapInputs, "structure_path", "positions");
    sceneClass.setMetadata(attrDensityRemapInputs, "structure_type", "ramp_float");
    sceneClass.setMetadata(attrDensityRemapInputs, "comment", "List of input remap curve values");

    attrDensityRemapOutputs = sceneClass.declareAttribute<rdl2::FloatVector>("density_remap_outputs",
        densityDefaults, {});
    // the metadata configures the UI widget
    sceneClass.setMetadata(attrDensityRemapOutputs, "label", "density remap outputs");
    sceneClass.setMetadata(attrDensityRemapOutputs, "structure_name", "ramp");
    sceneClass.setMetadata(attrDensityRemapOutputs, "structure_path", "values");
    sceneClass.setMetadata(attrDensityRemapOutputs, "structure_type", "ramp_float");
    sceneClass.setMetadata(attrDensityRemapOutputs, "comment", "List of output remap curve values");

    rdl2::IntVector interpolationDefaults = {1, 1}; //linear

    attrDensityRemapInterpolationTypes = sceneClass.declareAttribute<rdl2::IntVector>("density_remap_interpolation_types",
        interpolationDefaults, {});
    sceneClass.setMetadata(attrDensityRemapInterpolationTypes, "label", "density remap interpolation types");
    sceneClass.setMetadata(attrDensityRemapInterpolationTypes, "structure_name", "ramp");
    sceneClass.setMetadata(attrDensityRemapInterpolationTypes, "structure_path", "interpolation_types");
    sceneClass.setMetadata(attrDensityRemapInterpolationTypes, "structure_type", "ramp_float");
    sceneClass.setMetadata(attrDensityRemapInterpolationTypes, "comment",
        "List of density remap interpolation types");

    attrVdbInterpolation = sceneClass.declareAttribute<rdl2::Int>("vdb_interpolation_type", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrVdbInterpolation, 0, "point");
    sceneClass.setEnumValue(attrVdbInterpolation, 1, "box");
    sceneClass.setEnumValue(attrVdbInterpolation, 2, "quadratic");
    sceneClass.setMetadata(attrVdbInterpolation, "comment", "The type of interpolation to use when sampling the filter");

    attrBlurValue = sceneClass.declareAttribute<rdl2::Float>("blur_value", 0.0f);
    sceneClass.setMetadata(attrBlurValue, "min", "0.0");
    sceneClass.setMetadata(attrBlurValue, "max", "1000.0");
    sceneClass.setMetadata(attrBlurValue, "display", "logarithmic");
    sceneClass.setMetadata(attrBlurValue, "comment", "The blur radius");

    attrBlurType = sceneClass.declareAttribute<rdl2::Int>("blur_type", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrBlurType, 0, "gaussian");
    sceneClass.setEnumValue(attrBlurType, 1, "circular");
    sceneClass.setMetadata(attrBlurType, "comment", "The type of blur to apply");

    attrColorTint = sceneClass.declareAttribute<rdl2::Rgb>("color_tint", rdl2::Rgb(0.0f, 0.0f, 0.0f));
    sceneClass.setMetadata(attrColorTint, "comment",
        "Tints the light filter.  Lower density increases the shift toward the tint color.");

    attrInvertDensity = sceneClass.declareAttribute<rdl2::Bool>("invert_density", false);
    sceneClass.setMetadata(attrInvertDensity, "comment", "Invert the density with density = 1 - density");

    sceneClass.setGroup("Properties", attrVdbMap);
    sceneClass.setGroup("Properties", attrDensityGridName);
    sceneClass.setGroup("Properties", attrVdbInterpolation);
    sceneClass.setGroup("Properties", attrDensityRemapInputMin);
    sceneClass.setGroup("Properties", attrDensityRemapInputMax);
    sceneClass.setGroup("Properties", attrDensityRemapOutputMin);
    sceneClass.setGroup("Properties", attrDensityRemapOutputMax);
    sceneClass.setGroup("Properties", attrDensityRemapOutputs);
    sceneClass.setGroup("Properties", attrDensityRemapInputs);
    sceneClass.setGroup("Properties", attrDensityRemapInterpolationTypes);
    sceneClass.setGroup("Properties", attrDensityRemapRescaleEnable);
    sceneClass.setGroup("Properties", attrColorTint);
    sceneClass.setGroup("Properties", attrBlurValue);
    sceneClass.setGroup("Properties", attrBlurType);
    sceneClass.setGroup("Properties", attrInvertDensity);

RDL2_DSO_ATTR_END
