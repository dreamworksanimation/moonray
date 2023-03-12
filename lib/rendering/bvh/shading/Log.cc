// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Log.h
/// $Id$
///

#include "Log.h"

#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/mcrt_common/Util.h>

namespace moonray {
namespace shading {

void
logEvent(const scene_rdl2::rdl2::Shader *shader, const shading::TLState *tls, scene_rdl2::logging::LogEvent event)
{
    int threadIndex = mcrt_common::getThreadIdx(tls);
    shader->getThreadLocalObjectState()[threadIndex].mLogs.log(event);
}

} // namespace shading 
} // namespace moonray

