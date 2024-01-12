// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfUtil.isph

#pragma once

// Enable/disable printing out detailed BSDF info with each shade.
// Recompile any material shaders after enabling/disabling this.
// Note: You will need to rebuild the dependent shader repository (moonshine),
// and you may need to delete the build directory before rebuilding it.
// Warning: You'll only want to enable this when rendering with
// these moonray/moonray_gui options (for example):
// -threads 1
// -scene_var "bsdf_samples" "1"
// -scene_var "pixel_samples" "1"
// -scene_var "max_depth" "0"
// -debug_pixel 100 100
#if 0
#define SHADING_PRINT_DEBUG_BSDF_INFO_ENABLED
#endif

// Controls whether a single lane or all lanes are printed when
// SHADING_PRINT_DEBUG_BSDF_INFO_ENABLED is enabled.
#if 1
#define BSDF_UTIL_EXTRACT(val) extract(val, 0)
#else
#define BSDF_UTIL_EXTRACT(val) val
#endif




