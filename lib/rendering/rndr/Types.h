// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#ifndef RNDR_TYPES_H
#define RNDR_TYPES_H

#include <scene_rdl2/common/platform/Platform.h>

namespace moonray {
namespace rndr {

enum class RenderMode
{
    // Render each tile to completion before moving onto the next.
    BATCH,

    // Render samples to the GUI as soon as they're available.
    PROGRESSIVE,

    // Fast mode, stop path tracing and render a simplified version
    PROGRESSIVE_FAST,

    // Only show a new frame every n milliseconds without any refinement during
    // that interval.
    REALTIME,

    // Progress-checkpoint rendering mode. every user defined interval, whole image will be output
    PROGRESS_CHECKPOINT,

    NUM_MODES,
};

enum class FastRenderMode
{
    // Set ray-geometry intersection normals as radiance
    NORMALS,

    // Set shading normals as radiance
    NORMALS_SHADING,

    // Set facing ratio as radiance
    FACING_RATIO,

    // Set inverse of facing ratio as radiance
    FACING_RATIO_INVERSE,

    // Set UVs as radiance
    UVS,

    NUM_MODES,
};

enum class ApplicationMode
{
    // default mode in the future should not be used but
    // is currently used for backward compatibility
    UNDEFINED,

    // Specific use for motion capture, used to change key logic where needed
    MOTIONCAPTURE,

    // Beards Application
    BEARDS,

    // VR Application
    VR,

    NUM_MODES,
};

enum class SamplingMode
{
    UNIFORM = 0,        // Previous non-adaptive sampling behavior.
    //ADAPTIVE_V1 = 1,  // Removed, previous adaptive sampler.
    ADAPTIVE = 2,
    NUM_MODES,
};

enum class CheckpointMode
{
    TIME_BASED = 0,    // Time based checkpoint rendering mode
    QUALITY_BASED = 1, // Quality based checkpoint rendering mode
    NUM_MODES,
};

//
// Orthogonal to the concept of tiles are the concept of passes. This allows us to specify
// how many passes we want to make over the scene to produce the final frame. For example,
// for interactive rendering we may want to start with a very coarse pass so we can get
// an approximation of the image very quickly, and then do more passes over the image to
// refine it. For minimum time to the final frame, it would be more optimal to only specify
// a single pass. So it's a tradeoff depending on application requirements.
//
struct Pass
{
    bool isFinePass() const
    {
        // Obviously 1 <= startSampleIdx is a finePass.
        // However we also need to include pixRange=0~64, sample=0~N case into finePass as well.
        // This case we sample start from coarse pass and process all 64 pixels. We should
        // categorize this case as finePass actually. Toshi (Feb/12/2020)
        return (mStartPixelIdx == 0 && mEndPixelIdx == 64) || (1 <= mStartSampleIdx);
    }
    bool isCoarsePass() const   { return !isFinePass(); }

    unsigned getNumSamplesPerPixel() const  { return mEndSampleIdx - mStartSampleIdx; }
    unsigned getNumSamplesPerTile() const   { return (mEndPixelIdx - mStartPixelIdx) * getNumSamplesPerPixel(); }

    bool isValid() const
    {
        MNRY_ASSERT(mEndPixelIdx > mStartPixelIdx && mEndPixelIdx <= 64);

        if (isCoarsePass()) {
            // This is a "coarse" pass, meaning we're still filling in top level pixels on each tile.
            MNRY_ASSERT(mStartSampleIdx == 0);
            MNRY_ASSERT(mEndSampleIdx == 1);
        } else {
            // This is a "fine" pass, meaning we're refining on a subsample basis.
            MNRY_ASSERT(mEndSampleIdx > mStartSampleIdx);
        }

        return true;
    }

    // There are 64 pixels within each tile, indexed 0-63. You can decide which subset of pixels
    // you want filled in by this pass here. For example, by setting mStartPixelIdx to 0, and
    // mEndPixelIdx to 64, you would fill in all pixels of all active tiles in this pass.
    unsigned    mStartPixelIdx;
    unsigned    mEndPixelIdx;

    // All samples for a single pixel within a single pass can be assumed to be
    // executed on the same thread. The number of samples rendered for this pass
    // *per tile* would be: (mEndSampleIdx - mStartSampleIdx) * (mEndPixelIdx - mStartPixelIdx).
    unsigned    mStartSampleIdx;
    unsigned    mEndSampleIdx;
};

} // namespace rndr
} // namespace moonray

#endif // RNDR_TYPES_H
