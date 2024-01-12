// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/shading/Shading.h>

namespace scene_rdl2 {

namespace rdl2 {
class Material;
} // rdl2
}

namespace moonray {

namespace mcrt_common {
class ThreadLocalState;
}

namespace pbr {

class Scene;

// Compute the Z-value of a ray-hit, traced through the center of a pixel
float computeOpenGLDepth(mcrt_common::ThreadLocalState *tls, const Scene* scene,
                         int pixelX, int pixelY);

// Returns by reference the lights affecting the pixel(x, y) and the lights
// contibution value to that pixel.
void computeLightContributions(mcrt_common::ThreadLocalState *tls, const Scene* scene,
                    const int x, const int y,
                    shading::LightContribArray& lightContributions,
                    const int numSamples, const float textureFilterScale);

// Returns by reference the materials affecting the pixel(x, y)
const scene_rdl2::rdl2::Material* computeMaterial(mcrt_common::ThreadLocalState *tls,
                        const Scene* scene,
                        const int x, const int y);

// Returns the primitive at the pixel(x,y)
void computePrimitive(mcrt_common::ThreadLocalState *tls,
                      const Scene* scene,
                      const int x, const int y,
                      int& assignmentId);

} // namespace pbr
} // namespace moonray

