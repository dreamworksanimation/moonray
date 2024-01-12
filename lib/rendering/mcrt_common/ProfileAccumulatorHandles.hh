// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// Accumulator handles for profiling the render portion of the frame.
// Not suitable for the update portion of the frame.
//
#pragma once

//
// Accumulators to aid measuring exclusive blocks of time during the MCRT phase.
// Only one of these categories can be active at any one point in time on a
// given thread.
//

enum ExclAccType
{
    // Time spent adding aov samples to the frame buffer.
    EXCL_ACCUM_AOVS,

    // Time spent adding pixel info samples to the frame buffer.
    EXCL_ACCUM_ADD_HEAT_MAP_HANDLER,

    // Time spent adding samples to the frame buffer. Important since a lot of atomic
    // operations are potentially executed in here.
    EXCL_ACCUM_ADD_SAMPLE_HANDLER,

    // Time spend computing ray derivatives, *including* time spent computing shading sort keys.
    EXCL_ACCUM_COMPUTE_RAY_DERIVATIVES,

    // Time spent saving off entries for later processing. See Material::deferEntriesForLaterProcessing.
    EXCL_ACCUM_DEFER_SHADE_ENTRIES,

    // Time spent tracing embree intersection rays.
    EXCL_ACCUM_EMBREE_INTERSECTION,

    // Time spent tracing GPU intersection rays.
    EXCL_ACCUM_GPU_INTERSECTION,

    // Time spent tracing embree occlusion rays.
    EXCL_ACCUM_EMBREE_OCCLUSION,

    // Time spent tracing GPU occlusion rays.
    EXCL_ACCUM_GPU_OCCLUSION,

    // Time spent tracing presence rays (using embree rtcIntersection).
    EXCL_ACCUM_EMBREE_PRESENCE,

    // Time spent tracing volume rays (using embree rtcIntersection).
    EXCL_ACCUM_EMBREE_VOLUME,

    // Total time spent in initializing intersection structures which includes postIntersect calls.
    EXCL_ACCUM_INIT_INTERSECTION,

    // Total time spent integrating rays which hit scene geometry, *excluding* time
    // spent in EXCL_ACCUM_SSS_INTEGRATION and EXCL_ACCUM_VOL_INTEGRATION.
    EXCL_ACCUM_INTEGRATION,

    EXCL_ACCUM_OCCL_QUERY_HANDLER,

    // Total time spent in OIIO *during* the rendering portion of the frame. That is
    // this accumulator ignores displacement and other update related texture lookups.
    EXCL_ACCUM_OIIO,

    // After we're integrated a collection of samples, this is the time spent converting
    // associated SOA structures, include RayStates, back into AOS form and queuing them.
    EXCL_ACCUM_POST_INTEGRATION,

    EXCL_ACCUM_PRESENCE_QUERY_HANDLER,

    // Time spent generating primary rays and associated differentials.
    EXCL_ACCUM_PRIMARY_RAY_GEN,

    // All internal queuing overhead such as sorting and copying entries.
    EXCL_ACCUM_QUEUE_LOGIC,

    // Time spent inside of the ray handler excluding time spent inside of embree.
    EXCL_ACCUM_RAY_HANDLER,

    // Time spent sorting rays.
    EXCL_ACCUM_RAY_SORT_KEY_GEN,

    // Time spent allocating and deallocating RayState objects.
    EXCL_ACCUM_RAYSTATE_ALLOCS,

    // Time spent inside of the parallel portion of the render driver code excluding all
    // activity associated with tracing rays.
    EXCL_ACCUM_RENDER_DRIVER_OVERHEAD,

    EXCL_ACCUM_SHADE_HANDLER,

    // All time spent in shade code, including *OIIO* but excluding sorting and
    // AOS/SOA conversions.
    EXCL_ACCUM_SHADING,

    // Total time spent performing subsurface integration work.
    EXCL_ACCUM_SSS_INTEGRATION,

    // Total time spend allocating or freeing memory from TLS memory pools,
    // excluding RayState objects which are tracked with EXCL_ACCUM_RAYSTATE_ALLOCS.
    EXCL_ACCUM_TLS_ALLOCS,

    // This category exists so the user can avoid assigning time to the wrong category in the
    // cases where no appropriate exclusive category currently exists.
    EXCL_ACCUM_UNRECORDED,

    // Total time spent performing volume integration work.
    EXCL_ACCUM_VOL_INTEGRATION,

    // Time spent rebuilding adaptive tree.
    EXCL_BUILD_ADAPTIVE_TREE,

    // Time spent querying adaptive tree.
    EXCL_QUERY_ADAPTIVE_TREE,

    // Time spent gaining lock on adaptive tree.
    EXCL_EXCL_LOCK_ADAPTIVE_TREE,

    // General misc accumulators for programmer convenience.
    EXCL_ACCUM_MISC_A,
    EXCL_ACCUM_MISC_B,
    EXCL_ACCUM_MISC_C,
    EXCL_ACCUM_MISC_D,
    EXCL_ACCUM_MISC_E,
    EXCL_ACCUM_MISC_F,

    NUM_EXCLUSIVE_ACC,
};


//
// Accumulators to aid measuring the amount of time spent accomplishing a
// particular task. These times overlap with the exclusive blocks above (and each other).
//

enum OverlappedAccType
{
    // Time spend creating SOA embree intersection structures.
    ACCUM_AOS_TO_SOA_EMB_ISECT_RAYS,

    // Time spend creating SOA embree occlusion structures.
    ACCUM_AOS_TO_SOA_EMB_OCCL_RAYS,

    // Time spend converting shading::Intersection structures from AOS to SOA.
    ACCUM_AOS_TO_SOA_INTERSECTIONS,

    // Time spend converting RayState structures from AOS to SOA.
    ACCUM_AOS_TO_SOA_RAYSTATES,

    // Time spent stalled, waiting for pbr::TLState pool memory to be free'd from another thread.
    ACCUM_CL1_ALLOC_STALLS,
    ACCUM_CL2_ALLOC_STALLS,
    ACCUM_CL4_ALLOC_STALLS,

    // Time spend draining queues. Real work is being done during this period so it's
    // more informative.
    ACCUM_DRAIN_QUEUES,

    // Time spent NOT executing render driver specific code.
    ACCUM_NON_RENDER_DRIVER,

    // Time spent stalled, waiting for a RayState to be freed up from another thread.
    // This also includes anytime spend flushing local queues in an effort to free up
    // RayStates, even though actual work is being done during that time.
    ACCUM_RAYSTATE_STALLS,

    // Time spent inside of the parallel portion of the render driver code excluding all
    // activity associated with tracing rays.
    ACCUM_RENDER_DRIVER_PARALLEL,

    // Time spent converting embree intersection ray structures from SOA to AOS.
    ACCUM_SOA_TO_AOS_EMB_ISECT_RAYS,

    // Time spent converting BundledOcclRay structures from SOA to AOS.
    ACCUM_SOA_TO_AOS_OCCL_RAYS,

    // Time spent converting RadianceEntry structures from SOA to AOS.
    ACCUM_SOA_TO_AOS_RADIANCES,

    // Time spent converting RayState structure from SOA to AOS.
    ACCUM_SOA_TO_AOS_RAYSTATES,

    // Time spent sorting rays into directional buckets.
    ACCUM_SORT_RAY_BUCKETS,

    ACCUM_TOTAL_ISPC,

    //
    // General misc accumulators for programmer convenience.
    //
    ACCUM_MISC_A,
    ACCUM_MISC_B,
    ACCUM_MISC_C,
    ACCUM_MISC_D,
    ACCUM_MISC_E,
    ACCUM_MISC_F,

    NUM_OVERLAPPED_ACC,
};

//
// Internal accumulators using for intermediate calculations or to aid displaying results.
//

enum InternalAccType
{
    // This is computed manually internally. It's all the time which we're not profiling
    // and should be zero or very close to it.
    INT_ACCUM_MISSING_TIME,

    // Time spent inside of render driver code excluding all activity associated with tracing rays.
    INT_ACCUM_RENDER_DRIVER,

    // Time spend inside of the serial portion of the render driver.
    INT_ACCUM_RENDER_DRIVER_SERIAL,

    // Total time spent inside of embree.
    INT_ACCUM_TOTAL_EMBREE,

    // Total time spent in doing standard, sss, volume, and post integration.
    INT_ACCUM_TOTAL_INTEGRATION,

    // Total time spent inside of embree, ray and occlusion query handlers, and ray sorting code.
    INT_ACCUM_TOTAL_RAY_INTERSECTION,

    // Total time spent running scalar code outside of embree and ISPC.
    INT_ACCUM_TOTAL_SCALAR_TIME,

    // Total shading plus OIIO time.
    INT_ACCUM_TOTAL_SHADING,

    // The computed total of all the above categories.
    INT_ACCUM_TOTALS,

    NUM_INTERNAL_ACC,
};



