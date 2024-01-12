// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ShadingTLState.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <tbb/mutex.h>

namespace ispc {
extern "C" uint32_t ShadingTLState_hudValidation(bool);
}

using namespace scene_rdl2::alloc;
using namespace scene_rdl2::math;
using namespace scene_rdl2::util;

namespace moonray {
namespace shading {

//-----------------------------------------------------------------------------

// Private:
namespace
{

struct Private
{
    Private() :
        mRefCount(0)
    {
    }

    unsigned mRefCount;
};

Private gPrivate;
tbb::mutex gInitMutex;

void
initPrivate(const mcrt_common::TLSInitParams &initParams)
{
    MNRY_ASSERT(gPrivate.mRefCount == 0);

    // Initialize shading specific data here...

}

void
cleanUpPrivate()
{
    MNRY_ASSERT(gPrivate.mRefCount == 0);

    // Clean up shading specific data here...

}

}   // End of anon namespace.

//-----------------------------------------------------------------------------

TLState::TLState(mcrt_common::ThreadLocalState *tls, 
                 const mcrt_common::TLSInitParams &initParams,
                 bool okToAllocBundledResources) :
    texture::TLState(tls, initParams, okToAllocBundledResources),
    mAttributeOffsets(nullptr)
{
}

TLState::~TLState()
{
    MNRY_ASSERT(gPrivate.mRefCount);

    {
        // Protect against races the during gPrivate clean up.
        //tbb::mutex::scoped_lock lock(gInitMutex);

        MOONRAY_THREADSAFE_STATIC_WRITE(--gPrivate.mRefCount);
        if (gPrivate.mRefCount == 0) {
            shading::cleanUpPrivate();
            texture::TLState::cleanUpPrivate();
        }
    }
}

void
TLState::reset()
{
    texture::TLState::reset();

    // Add any reset code specific to this object here...
    mAttributeOffsets = nullptr;
}

std::shared_ptr<TLState>
TLState::allocTls(mcrt_common::ThreadLocalState *tls,
                  const mcrt_common::TLSInitParams &initParams,
                  bool okToAllocBundledResources)
{
    {
        // Protect against races the very first time we initialize gPrivate.
        tbb::mutex::scoped_lock lock(gInitMutex);

        if (gPrivate.mRefCount == 0) {
            texture::TLState::initPrivate(initParams);
            shading::initPrivate(initParams);
        }

        MOONRAY_THREADSAFE_STATIC_WRITE(++gPrivate.mRefCount);
    }

    return std::make_shared<TLState>(tls, initParams, okToAllocBundledResources);
}

void initTexturingSupport(TLState *tls)
{
    MNRY_VERIFY(tls)->initTexturingSupport();
}

HUD_VALIDATOR( ShadingTLState );

//-----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

