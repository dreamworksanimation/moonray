// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestRender.h
/// @brief A simple test renderer which implicitly raytraces a sphere.

#pragma once

#include <moonray/rendering/texturing/sampler/TextureSampler.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace shading {

class TLState;

class TestRender {
public:
    
    static void update(scene_rdl2::rdl2::Material *mat, int testNum);
    
    // Renders the material and compares the result against the canonical
    static uint64 renderAndCompare(const scene_rdl2::rdl2::Material *mat,
                                   int testNum,
                                   int width,
                                   int height,
                                   int raysPerPixel,
                                   bool isIndirect);
    
    // Setup a basic ISPC scene and call the ISPC render function
    static uint64 render(
            const scene_rdl2::rdl2::Material *mat,
            uint8_t *results,
            int width = 256, int height = 256,
            int raysPerPixel = 1,
            bool primeThePump = false,
            bool isIndirect = false);
private:
    static scene_rdl2::alloc::Arena *createArena();
    static void destroyArena(scene_rdl2::alloc::Arena * arena);
};

} // namespace shading
} // namespace moonray

