// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::String> attrModel;
    rdl2::AttributeKey<rdl2::Int>    attrInterpolation;
    rdl2::AttributeKey<rdl2::String> attrDensityGrid;
    rdl2::AttributeKey<rdl2::String> attrEmissionGrid;
    rdl2::AttributeKey<rdl2::String> attrVelocityGrid;
    rdl2::AttributeKey<rdl2::Float>  attrVelocityScale;
    rdl2::AttributeKey<rdl2::Float>  attrVelocitySampleRate;
    rdl2::AttributeKey<rdl2::Float>  attrEmissionSampleRate;

RDL2_DSO_ATTR_DEFINE(rdl2::Geometry)

    attrModel = sceneClass.declareAttribute<rdl2::String>(
        "model", "", rdl2::FLAGS_FILENAME);
    sceneClass.setMetadata(attrModel, "comment",
        "The VDB file to load");
    sceneClass.setGroup("VDB", attrModel);

    // different interpolation mode in vdb
    attrInterpolation = sceneClass.declareAttribute<rdl2::Int>(
        "interpolation", 1, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setEnumValue(attrInterpolation, 0, "nearest neighbor");
    sceneClass.setEnumValue(attrInterpolation, 1, "linear");
    sceneClass.setEnumValue(attrInterpolation, 2, "quadratic");
    sceneClass.setMetadata(attrInterpolation, "comment",
        "Voxel interpolation to use when sampling the volume data");
    sceneClass.setGroup("VDB", attrInterpolation);

    attrDensityGrid = sceneClass.declareAttribute<rdl2::String>(
        "density_grid", "density");
    sceneClass.setMetadata(attrDensityGrid, "label", "density grid");
    sceneClass.setMetadata(attrDensityGrid, "comment",
        "The name of the density grid. If multiple grids have the same name, "
        "only the first grid with that name will be loaded. "
        "If a vdb file has multiple grids with the same name, "
        "you may use a suffix index to pick which grid to load, e.g. "
        "\"density[3]\". The index must be in [] brackets.");
    sceneClass.setGroup("VDB", attrDensityGrid);

    attrEmissionGrid = sceneClass.declareAttribute<rdl2::String>(
        "emission_grid", "");
    sceneClass.setMetadata(attrEmissionGrid, "label", "emission grid");
    sceneClass.setMetadata(attrEmissionGrid, "comment",
        "The name of the emission grid. If multiple grids have the same name, "
        "only the first grid with that name will be loaded. "
        "If a vdb file has multiple grids with the same name, "
        "you may use a suffix index to pick which grid to load, e.g. "
        "\"emission[3]\". The index must be in [] brackets.");
    sceneClass.setGroup("VDB", attrEmissionGrid);

    attrVelocityGrid = sceneClass.declareAttribute<rdl2::String>(
        "velocity_grid", "", {"velocity grid"});
    sceneClass.setMetadata(attrVelocityGrid, "label", "velocity grid");
    sceneClass.setMetadata(attrVelocityGrid, "comment",
        "the name of vector grid representing the velocity field. "
        "Usually named \"v\" or \"vel\" in simulation export. "
        "If multiple velocity grids have the same name, "
        "only the first grid with that name will be loaded. "
        "If a vdb file has multiple grids with the same name, "
        "you may use a suffix index to pick which grid to load, e.g. "
        "\"v[3]\". The index must be in [] brackets. The index "
        "can be different from the index on the \"density_grid\".");
    sceneClass.setGroup("Motion Blur", attrVelocityGrid);

    attrVelocityScale = sceneClass.declareAttribute<rdl2::Float>(
        "velocity_scale", 1.0f, {"velocity scale"});
    sceneClass.setMetadata(attrVelocityScale, "label", "velocity scale");
    sceneClass.setMetadata(attrVelocityScale, "comment",
        "A scale factor for the velocity field. "
        "A value of 0 disables motion blur.");
    sceneClass.setGroup("Motion Blur", attrVelocityScale);

    attrVelocitySampleRate = sceneClass.declareAttribute<rdl2::Float>(
        "velocity_sample_rate", 0.2f, {"velocity sample rate"});
    sceneClass.setMetadata(attrVelocitySampleRate, "label",
        "velocity sample rate");
    sceneClass.setMetadata(attrVelocitySampleRate, "comment",
        "the relative scale of input velocity grid resolution. "
        "Lower value has lower memory overhead and lower fidelity of "
        "motion blur effect, which is sometimes desired for artistic reasons");
    sceneClass.setGroup("Motion Blur", attrVelocitySampleRate);

    attrEmissionSampleRate = sceneClass.declareAttribute<rdl2::Float>(
        "emission_sample_rate", 1.0f, {"emission sample rate"});
    sceneClass.setMetadata(attrEmissionSampleRate, "label",
        "emission sample rate");
    sceneClass.setMetadata(attrEmissionSampleRate, "comment",
        "the relative scale of input emission grid resolution. "
        "Lower value has lower memory overhead and faster render time, "
        "with the cost of lower fidelity of emission shape and illumination");
    sceneClass.setGroup("VDB", attrEmissionSampleRate);


RDL2_DSO_ATTR_END

