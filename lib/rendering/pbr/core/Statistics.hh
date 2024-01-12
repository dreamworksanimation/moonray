// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include <scene_rdl2/common/platform/HybridUniformData.hh>

// Define 2 counters, the first records actual lane activity, the second
// records the max potential lane activity.
#define LANE_UTILIZATION_COUNTER(c)  c, c##_MAX

enum StatCounters
{
    STATS_PIXEL_SAMPLES,
    STATS_LIGHT_SAMPLES,
    STATS_BSDF_SAMPLES,
    STATS_SSS_SAMPLES,

    STATS_INTERSECTION_RAYS,
    STATS_BUNDLED_INTERSECTION_RAYS,
    STATS_BUNDLED_GPU_INTERSECTION_RAYS,
    STATS_VOLUME_RAYS,
    STATS_OCCLUSION_RAYS,
    STATS_BUNDLED_OCCLUSION_RAYS,
    STATS_BUNDLED_GPU_OCCLUSION_RAYS,
    STATS_PRESENCE_SHADOW_RAYS,

    STATS_SHADER_EVALS,
    STATS_TEXTURE_SAMPLES,

    // Vectorized only. These count the numbers of samples we're taking assuming
    // all lanes are active. This allows us to compute our actual lane utilization
    // at a later stage.
    STATS_LIGHT_SAMPLE_LANE_MAX,
    STATS_BSDF_SAMPLE_LANE_MAX,
    STATS_SSS_SAMPLE_LANE_MAX,

    // For localized testing.
    STATS_MISC_COUNTER_A,
    STATS_MISC_COUNTER_B,
    STATS_MISC_COUNTER_C,
    STATS_MISC_COUNTER_D,


    //
    // Vector lane utilization counters:
    //

    // Shading related:


    // Texturing related:



    // Integration related:
    LANE_UTILIZATION_COUNTER( STATS_VEC_BSDF_LOBES ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_BSDF_LOBE_SAMPLES_PRE ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_BSDF_LOBE_SAMPLES_POST ),

    LANE_UTILIZATION_COUNTER( STATS_VEC_LIGHT_SAMPLES_PRE ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_LIGHT_SAMPLES_POST ),

    LANE_UTILIZATION_COUNTER( STATS_VEC_COUNTER_A ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_COUNTER_B ),

    LANE_UTILIZATION_COUNTER( STATS_VEC_ADD_DIRECT_VISIBLE_BSDF ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_ADD_DIRECT_VISIBLE_LIGHTING ),

    LANE_UTILIZATION_COUNTER( STATS_VEC_INDIRECT_A ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_INDIRECT_B ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_INDIRECT_C ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_INDIRECT_D ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_INDIRECT_E ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_INDIRECT_F ),

    LANE_UTILIZATION_COUNTER( STATS_VEC_FILL_BUNDLED_RADIANCE ),
    LANE_UTILIZATION_COUNTER( STATS_VEC_FILL_OCCL_RAY ),

    // Total number of counters.
    NUM_STATS_COUNTERS,
};

#define PBR_STATISTICS_MEMBERS                              \
    HUD_ARRAY(uint64_t, mCounters, NUM_STATS_COUNTERS);     \
    HUD_MEMBER(double, mMcrtTime);                          \
    HUD_MEMBER(double, mMcrtUtilization)

#define PBR_STATISTICS_VALIDATION                           \
    HUD_BEGIN_VALIDATION(PbrStatistics);                    \
    HUD_VALIDATE(PbrStatistics, mCounters);                 \
    HUD_VALIDATE(PbrStatistics, mMcrtTime);                 \
    HUD_VALIDATE(PbrStatistics, mMcrtUtilization);          \
    HUD_END_VALIDATION



