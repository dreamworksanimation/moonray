// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Log.h
/// $Id$
///

#include "Log.h"

#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/mcrt_common/Util.h>

void CPP_logEvent(const scene_rdl2::rdl2::Shader *shader, scene_rdl2::logging::LogEvent event)
{
    scene_rdl2::rdl2::Shader::getLogEventRegistry().log(shader, event);
}

