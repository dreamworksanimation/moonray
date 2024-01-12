// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Shadingv.h
#pragma once

// Shadingv.h: vectorized shading

// This file contains c++ aliases of ispc structures defined in
// shading.isph.  Using these structures and helper functions, clients
// can fill out shading inputs and retrieve shading results using the
// same in-memory structures as ispc code.

// Why we include Shading_ispc_stubs.h with ""
// When built within arras, Shading_ispc_stubs.h is not
// in the same directory as Shadingv.h, rather
// it exists in autogenerate/rendering/shading/ispc which is part of the
// include path.  When installed publicly in the arras folio
// Shading_ispc_stubs.h will exist in the same directory as Shadingv.h.
#include "Shading_ispc_stubs.h"

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/common/platform/IspcUtil.h>
#include <scene_rdl2/common/math/ispc/Typesv.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#include <cstddef>

namespace scene_rdl2 {
namespace rdl2 {
    class Displacement;
    class Map;
    class NormalMap;
    class Material;
}
}

namespace moonray {
namespace shading {

class TLState;

// math types
typedef scene_rdl2::math::Vec2fv Vec2fv;
typedef scene_rdl2::math::Vec3fv Vec3fv;
typedef scene_rdl2::math::Colorv Colorv;

// shading state
ISPC_UTIL_TYPEDEF_STRUCT(State, Statev);

/// @brief call a displacement shader
void
displacev(const scene_rdl2::rdl2::Displacement *displacement,
          shading::TLState *tls,
          int numStatev,
          const shading::Statev *statev,
          shading::Vec3fv *result);

/// @brief sample a map shader
void
samplev(const scene_rdl2::rdl2::Map *map,
        shading::TLState *tls,
        const shading::Statev *statev,
        shading::Colorv *result);

/// @brief sample a normal map shader
void
sampleNormalv(const scene_rdl2::rdl2::NormalMap *normalMap,
              shading::TLState *tls,
              const shading::Statev *statev,
              shading::Colorv *result);

/// @brief shade a material
void
shadev(const scene_rdl2::rdl2::Material *material,
       shading::TLState *tls,
       int numStatev,
       const shading::Statev *statev,
       scene_rdl2::rdl2::Bsdfv *bsdfv);

} // namespace shading
} // namespace moonray

