// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace moonray {

/**
 * rt library provides acceleration data structures and interface for:
 * - Efficient building of acceleration data structures.
 * - Efficient intersection ray queries.
 *
 * The rt library primarily interacts with the geom library where it obtains
 * geometry data to build the acceleration data structures.
 * The user should understand that rt library has a <b>strong</b>
 * dependency on the geom library, not only during the build stage of the
 * acceleration data structure, but also during intersection queries.
 * Therefore it's important for the user of the rt and geom libraries to
 * maintain data consistency between the two.
 */
namespace rt {

enum class ChangeFlag
{
    NONE,        // No changes.
    UPDATE,      // Only update changes.
    ALL          // Everything changed.
};

enum class OptimizationTarget
{
    HIGH_QUALITY_BVH_BUILD, // Build BVH to be optimized for mcrt performance.
                            // Typically used for offline rendering.
    FAST_BVH_BUILD          // Fast BVH build for interactive rendering.
                            // Reduces time to first pixel.
};

struct AcceleratorOptions
{
    int maxThreads = 0;
    bool verbose = false;
};

} // namespace rt
} // namespace moonray

