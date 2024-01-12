// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file attributes.cc
#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

rdl2::AttributeKey<rdl2::SceneObject *> attrGeometry;
rdl2::AttributeKey<rdl2::Int> attrUdim;
rdl2::AttributeKey<rdl2::String> attrUvAttribute;
rdl2::AttributeKey<rdl2::Int> attrMode;
rdl2::AttributeKey<rdl2::Float> attrBias;
rdl2::AttributeKey<rdl2::Bool> attrUseRelativeBias;
rdl2::AttributeKey<rdl2::Float> attrMapFactor;
rdl2::AttributeKey<rdl2::String> attrNormalMap;
rdl2::AttributeKey<rdl2::Int> attrNormalMapSpace;

RDL2_DSO_ATTR_DEFINE(rdl2::Camera)

attrGeometry = sceneClass.declareAttribute<rdl2::SceneObject *>("geometry", rdl2::FLAGS_NONE, rdl2::INTERFACE_GEOMETRY);
sceneClass.setMetadata(attrGeometry, "comment", "The geometry object to bake");

attrUdim = sceneClass.declareAttribute<rdl2::Int>("udim", 1001);
sceneClass.setMetadata(attrUdim, "comment", "Udim tile to bake");

attrUvAttribute = sceneClass.declareAttribute<rdl2::String>("uv_attribute", "", { "uv attribute" });
sceneClass.setMetadata(attrUvAttribute, "label", "uv attribute");
sceneClass.setMetadata(attrUvAttribute, "comment", "Specifies a Vec2f primitive attribute to "
                       "use as the uv coordinates.  If empty, the default uv "
                       "for the mesh is used.  The uvs must provide a unique parameterization "
                       "of the mesh, i.e. a given (u, v) can appear only once on the mesh "
                       "being baked.");

attrMode = sceneClass.declareAttribute<rdl2::Int>("mode", 3, rdl2::FLAGS_ENUMERABLE);
sceneClass.setEnumValue(attrMode, 0, "from camera to surface");
sceneClass.setEnumValue(attrMode, 1, "from surface along normal");
sceneClass.setEnumValue(attrMode, 2, "from surface along reflection vector");
sceneClass.setEnumValue(attrMode, 3, "above surface reverse normal");
sceneClass.setMetadata(attrMode, "comment", "How to generate primary rays");

attrBias = sceneClass.declareAttribute<rdl2::Float>("bias", 0.003);
sceneClass.setMetadata(attrBias, "comment", "Ray-tracing offset for primary ray origin");

attrUseRelativeBias = sceneClass.declareAttribute<rdl2::Bool>("use_relative_bias", true, { "use relative bias" });
sceneClass.setMetadata(attrUseRelativeBias, "label", "use relative bias");
sceneClass.setMetadata(attrUseRelativeBias, "comment", "If true, bias is scaled based on position magnitude");

attrMapFactor = sceneClass.declareAttribute<rdl2::Float>("map_factor", 1.0, { "map factor" });
sceneClass.setMetadata(attrMapFactor, "label", "map factor");
sceneClass.setMetadata(attrMapFactor, "comment", "Increase or decrease the internal "
                       "position map buffer resolution");

attrNormalMap = sceneClass.declareAttribute<rdl2::String>("normal_map", rdl2::FLAGS_FILENAME, rdl2::INTERFACE_GENERIC, { "normal map" });
sceneClass.setMetadata(attrNormalMap, "label", "normal map");
sceneClass.setMetadata(attrNormalMap, "comment", "Use this option to supply "
                       "your own normals that are used when computing ray directions.  "
                       "Without this option, normals are computed from the geometry and do "
                       "not take into account any material applied normal mapping.");
attrNormalMapSpace = sceneClass.declareAttribute<rdl2::Int>("normal_map_space", 0, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "normal map space" });
sceneClass.setMetadata(attrNormalMapSpace, "label", "normal map space");
sceneClass.setEnumValue(attrNormalMapSpace, 0, "camera space");
sceneClass.setEnumValue(attrNormalMapSpace, 1, "tangent space");
sceneClass.setMetadata(attrNormalMapSpace, "comment", "Use camera space if you generated "
                       "per frame normal maps in a pre-pass using the normal material aov.  "
                       "You probably want to use tangent space if you are using a normal "
                       "map that is also used in the surfacing setup.");

RDL2_DSO_ATTR_END


