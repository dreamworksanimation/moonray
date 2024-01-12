// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/*
MOONRAY-XPU START HERE

The current Moonray-XPU implementation offloads the occlusion ray processing to
the GPU for supported scenes.  This is an extension of vectorized mode with some
special queuing logic to collect occlusion rays for the GPU.  Future work will
extend the GPU processing to regular non-occlusion rays.

XPU is implemented using NVIDIA's Optix 7.6 API, which is built on NVIDIA's
CUDA SDK.

***** Mode of execution:

1) Individual CPU rendering threads queue up occlusion rays via
   TLState::addOcclusionQueueEntries().  This function then directs rays to the
   regular vector mode occlusion queue or the XPUOcclusionRayQueue, if available.
2) The XPUOcclusionRayQueue is flushed when full, calling the XPU ray handler.
3) The XPU ray handler calls the GPUAccelerator (via GeometryAccelerator::occludedXPU())
   to process the batch of rays on the GPU and read the results back from the
   GPU.  It then creates/queues the BundledRadiances for the render buffer.

***** Vector mode fallback:

Not all vector mode features are supported in XPU mode.  If the user specifies the
XPU exec_mode, Moonray will attempt to create the GPU accelerator for the scene.
If it discovers that the scene requires an unsupported GPU feature, it will
stop creating the GPU accelerator and fall back to regular vector mode.

***** Source code locations:

Most of the source code is located in this directory.  There are additional pieces
here:

lib/rendering/mcrt_common/ExecutionMode.h:
    Defines the XPU execution mode enum.  Moonray generally treats XPU and vectorized
    mode the same in most places.

lib/rendering/mcrt_common/ProfileAccumulatorHandles.hh:
    Defines the XPU execution profiling accumulators, i.e. EXCL_ACCUM_GPU_OCCLUSION

lib/rendering/pbr/core/XPUOcclusionRayQueue.h:
    Implements a queue for XPU bundled occlusion rays.  Future
    expansion of XPU will add specializations for other ray types.

lib/rendering/pbr/core/PbrTLState.h/.cc/.hh:
    Holds a pointer to the XPUOcclusionRayQueue so the individual CPU-side threads
    can submit rays to the GPU.  PbrTLState does not *own* the XPUOcclusionRayQueue
    object - that is owned by the RenderDriver.  There is only one XPUOcclusionRayQueue
    shared by all the threads.

lib/rendering/pbr/handlers/XPURayHandlers.cc/.h:
    Ray handler functions that are called by the XPUAcceleratorQueue when it flushes
    rays out of the queue.  There are both a CPU and GPU ray handler.  The CPU ray
    handler is invoked if the GPU is busy to avoid blocking CPU threads, which
    provides automatic load-balancing between the GPU and CPUs.  Calls the occludedGPU()
    function on GeometryAccelerator.

lib/rendering/rndr/RenderDriver.cc:
    Functions to create/free/flush the XPUAcceleratorQueue that it owns.

lib/rt/GeometryAccelerator.h:
    Implements occludedGPU() which passes rays to the actual GPUAccelerator (in
    GeometryManager.)  In Moonray, all ray tracing queries go through
    GeometryAccelerator whether they are scalar/vector/xpu.

lib/rt/GeometryManager.h/.cc:
    Owns the GPUAccelerator object and creates it in updateGPUAccelerator(), which
    implements the logic to fall back to regular vector mode if there is a problem
    creating the GPUAccelerator.  Note that GeometryManager creates/updates the
    regular embree accelerator in finalizeChanges(), but the GPUAccelerator is
    created at the end of RenderContext::renderPrep() because it needs a fully
    initialized scene.

***** Error handling:

We need to cleanly fall back to regular vector execution mode if there are any
problems encountered while setting up the GPUAccelerator.  The constructor takes
a pointer to a std::string that an error message is placed into.  If the message is
empty after construction, no error has occurred and XPU has been successfully
initialized.  Many of the internal API functions use this same pattern, although
they also return a bool success/fail that is easier to check for than an empty error
string.

***** Optix data structure zero initialization:

All Optix data structures (structs) are expected to be zero-initialized with "= {};"
before the members are filled in.  This is done in many places in this code.
NVIDIA's examples do this and it eliminates a potential source of flaky behavior,
plus the default values tend to be zero so you only need to set the non-defaults.

***** Memory management:

The OptixGPUBuffer class is used everywhere to manage GPU data.  You create an OptixGPUBuffer
object on the host side and it creates/manages a buffer on the GPU.  It has methods
for easily copying to/from the GPU buffer.  It will automatically release the GPU
buffer in its destructor which helps prevent GPU memory leaks.

*/

#pragma once

#include "GPURay.h"
#include <moonray/rendering/rt/rt.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/Layer.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>

namespace moonray {
namespace rt {

// We hide the details of this class behind an impl because we don't want to expose
// all of the CUDA/Optix headers to the rest of Moonray.  All this needs to expose
// to the outside world is an occluded() method and a constructor that takes a scene.

class OptixGPUAccelerator;

class GPUAccelerator
{
public:
    GPUAccelerator(bool allowUnsupportedFeatures,
                   const scene_rdl2::rdl2::Layer *layer,
                   const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                   const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s,
                   std::vector<std::string>& warningMsgs,
                   std::string* errorMsg);
    ~GPUAccelerator();

    // Copy is disabled
    GPUAccelerator(const GPUAccelerator& other) = delete;
    GPUAccelerator &operator=(const GPUAccelerator& other) = delete;

    std::string getGPUDeviceName() const;

    void intersect(const unsigned numRays, const GPURay* rays) const;

    // output intersect results are placed in here
    GPURayIsect* getOutputIsectBuf() const;

    void occluded(const unsigned numRays, const GPURay* rays) const;

    // output occlusion results are placed in here
    unsigned char* getOutputOcclusionBuf() const;

    static unsigned int getRaysBufSize();

private:

#ifdef MOONRAY_USE_CUDA
    std::unique_ptr<OptixGPUAccelerator> mImpl;
#endif

};

} // namespace rt
} // namespace moonray

