// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Log.h
/// $Id$
///

#pragma once

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace shading {

class Intersection;

//---------------------------------------------------------------------------

/**
 * Convenience function to allow for fast local (in the thread-local sense)
 * logging from within a Shader.
 *
 * Sample usage:
 * 
 * In constructor/update:
 * LogEvent mBadValueError = mLogEventRegistry.createEvent(logging::ERROR_LEVEL, "Bad value");
 * 
 * In sample/shade:
 * shading::localLog(me, tls, mBadValueError)
 * 
 * See lib/scene_rdl2/render/logging/logging.h for more details
 */
void logEvent(const scene_rdl2::rdl2::Shader *shader, const shading::TLState *tls, scene_rdl2::logging::LogEvent event);


//---------------------------------------------------------------------------

} // namespace shading 
} // namespace moonray


