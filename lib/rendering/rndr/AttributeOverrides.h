// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once


#include "RenderOptions.h"

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <string>
#include <vector>

namespace moonray {
namespace rndr {

class RenderStats;

/**
 * Applies all of the attribute value and attribute binding overrides present
 * in the RenderOptions to the given SceneContext.
 *
 * @param   context     The SceneContext to apply overrides on.
 * @param   stats       The rendering stats for logging purposes.
 * @param   options     The RenderOptions which contain the overrides to apply.
 */
void applyAttributeOverrides(scene_rdl2::rdl2::SceneContext& context,
                             const RenderStats& stats,
                             const RenderOptions& options,
                             std::stringstream& initMessages);

} // namespace rndr
} // namespace moonray

