// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file attributes.cc
/// $Id$
///

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE
    // declare rdl2 attribute that geometry shader can access
    rdl2::AttributeKey<rdl2::Int> attrTestMode;

RDL2_DSO_ATTR_DEFINE(rdl2::Geometry)
    // define rdl2 attribute type, name and default value
    attrTestMode = sceneClass.declareAttribute<rdl2::Int>("test_mode", 0,
        rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "test mode" });

    sceneClass.setEnumValue(attrTestMode, 0, "polygon");
    sceneClass.setEnumValue(attrTestMode, 1, "instance");
    sceneClass.setEnumValue(attrTestMode, 2, "nested instance");

RDL2_DSO_ATTR_END

