// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <moonray/map/primvar/ispc/Primvar_ispc_stubs.h>

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <moonray/rendering/shading/MapApi.h>
#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonshine  {
namespace primvar  {

void
createLogEvent(const std::string& primAttrType,
               const std::string& primAttrName,
               int& missingAttributeEvent,
               scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry);

bool
getPosition(moonray::shading::TLState* tls,
            const moonray::shading::State& state,
            const ispc::PRIMVAR_Input_Source_Mode inputSourceMode, 
            const scene_rdl2::math::Vec3f& inputPosition,
            const moonray::shading::Xform * const xform,
            const ispc::SHADING_Space returnSpace,
            const int refPKey,
            scene_rdl2::math::Vec3f& outputPosition,
            scene_rdl2::math::Vec3f& outputPositionDdx,
            scene_rdl2::math::Vec3f& outputPositionDdy,
            scene_rdl2::math::Vec3f& outputPositionDdz);

bool
getNormal(moonray::shading::TLState* tls,
          const moonray::shading::State& state,
          const ispc::PRIMVAR_Input_Source_Mode inputSourceMode, 
          const scene_rdl2::math::Vec3f& inputNormal,
          const moonray::shading::Xform * const xform,
          const ispc::SHADING_Space returnSpace,
          const int refPKey,
          const int refNKey,
          scene_rdl2::math::Vec3f& outputNormal);

} // projection
} // moonshine


