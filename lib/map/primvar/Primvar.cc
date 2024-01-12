// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#include "Primvar.h"

#include <sstream>

namespace moonshine {
namespace primvar {

using namespace scene_rdl2;

void
createLogEvent(const std::string& primAttrType,
               const std::string& primAttrName,
               int& missingAttributeEvent,
               scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry)
{
    // setup an appropiate log event message
    std::ostringstream os;
    os << "Missing primitive attribute '"
        << primAttrType << " " << primAttrName
        << "', using default value";

    missingAttributeEvent =
        logEventRegistry.createEvent(scene_rdl2::logging::WARN_LEVEL, os.str());
}

bool
getPosition(moonray::shading::TLState* tls,
            const moonray::shading::State& state,
            const ispc::PRIMVAR_Input_Source_Mode inputSourceMode, 
            const math::Vec3f& inputPosition,
            const moonray::shading::Xform * const xform,
            const ispc::SHADING_Space returnSpace,
            const int refPKey,
            math::Vec3f& outputPosition,
            math::Vec3f& outputPositionDdx,
            math::Vec3f& outputPositionDdy,
            math::Vec3f& outputPositionDdz)
{
    outputPositionDdz = math::Vec3f(0.0f);
    ispc::SHADING_Space inputSpace;
    if (inputSourceMode == ispc::INPUT_SOURCE_MODE_ATTR) {
        outputPosition = inputPosition;
        outputPositionDdx = math::Vec3f(0.0f);
        outputPositionDdy = math::Vec3f(0.0f);
        inputSpace = ispc::SHADING_SPACE_WORLD;
    } else if (inputSourceMode == ispc::INPUT_SOURCE_MODE_REF_P_REF_N) {
        if(!state.getRefP(outputPosition)) {
            return false;
        }
        state.getdVec3fAttrdx(refPKey, outputPositionDdx);
        state.getdVec3fAttrdy(refPKey, outputPositionDdy);
        inputSpace = ispc::SHADING_SPACE_WORLD;
    } else {
        outputPosition = state.getP();
        outputPositionDdx = state.getdPdx();
        outputPositionDdy = state.getdPdy();
        inputSpace = ispc::SHADING_SPACE_RENDER;
    }

    if (xform != nullptr) {
        const math::Vec3f transformedP = xform->transformPoint(inputSpace,
                                                               returnSpace,
                                                               state,
                                                               outputPosition);

        // transform the partials as points to include perspective divide,
        // starting with untransformed position
        math::Vec3f outputPositionPlusDPdx(outputPosition + outputPositionDdx);
        math::Vec3f outputPositionPlusDPdy(outputPosition + outputPositionDdy);

        outputPositionPlusDPdx = xform->transformPoint(inputSpace,
                                                       returnSpace,
                                                       state,
                                                       outputPositionPlusDPdx);

        outputPositionPlusDPdy = xform->transformPoint(inputSpace,
                                                       returnSpace,
                                                       state,
                                                       outputPositionPlusDPdy);

        outputPosition = transformedP;
        outputPositionDdx = outputPositionPlusDPdx - transformedP;
        outputPositionDdy = outputPositionPlusDPdy - transformedP;
    }

    return true;
}

bool
getNormal(moonray::shading::TLState* tls,
          const moonray::shading::State& state,
          const ispc::PRIMVAR_Input_Source_Mode inputSourceMode, 
          const scene_rdl2::math::Vec3f& inputNormal,
          const moonray::shading::Xform * const xform,
          const ispc::SHADING_Space returnSpace,
          const int refPKey,
          const int refNKey,
          math::Vec3f& outputNormal)
{
    ispc::SHADING_Space inputSpace;
    if (inputSourceMode == ispc::INPUT_SOURCE_MODE_ATTR) {
        outputNormal = inputNormal;
        inputSpace = ispc::SHADING_SPACE_WORLD;
    } else if (inputSourceMode == ispc::INPUT_SOURCE_MODE_REF_P_REF_N) {
        if (!state.getRefN(outputNormal)) {
            return false;
        }
        inputSpace = ispc::SHADING_SPACE_WORLD;
    } else {
        outputNormal = state.getN();
        inputSpace = ispc::SHADING_SPACE_RENDER;
    }
    
    if (xform != nullptr) {
        outputNormal = xform->transformNormal(inputSpace,
                                              returnSpace,
                                              state,
                                              outputNormal);
    }

    if (inputSourceMode == ispc::INPUT_SOURCE_MODE_REF_P_REF_N && state.isEntering() == false) {
        // Flip reference space normals on exiting a surface
        outputNormal = -outputNormal;
    }

    outputNormal = scene_rdl2::math::normalize(outputNormal);

    return true;
}

} // primvar
} // moonshine


