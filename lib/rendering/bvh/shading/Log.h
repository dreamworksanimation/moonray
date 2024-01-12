// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Log.h
/// $Id$
///

#pragma once

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

/**
 * Convenience function to allow for logging from within a Shader.
 *
 * Sample usage:
 *
 * In constructor/update:
 * LogEvent mBadValueError = sLogEventRegistry.createEvent(logging::ERROR_LEVEL, "Bad value");
 *
 * In sample/shade:
 * shading::localLog(me, tls, mBadValueError)
 *
 * See lib/scene_rdl2/render/logging/logging.h for more details
 */
extern "C" void CPP_logEvent(const scene_rdl2::rdl2::Shader* shader, scene_rdl2::logging::LogEvent event);

namespace moonray {
namespace shading {
inline void logEvent(const scene_rdl2::rdl2::Shader* shader, scene_rdl2::logging::LogEvent event)
{
    CPP_logEvent(shader, event);
}
} // namespace shading
} // namespace moonray
