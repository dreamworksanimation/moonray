// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TextureSampler.h"
#include "TextureTLState.h"
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <tbb/mutex.h>

namespace ispc {
extern "C" uint32_t TextureTLState_hudValidation(bool);
}

using namespace scene_rdl2::alloc;
using namespace scene_rdl2::math;
using namespace scene_rdl2::util;

namespace moonray {
namespace texture {

//-----------------------------------------------------------------------------

// Private:
namespace
{

struct Private
{
    Private() : mTextureSampler(nullptr)
    {
    }

    texture::TextureSampler *mTextureSampler;
};

Private gPrivate;

}   // End of anon namespace.

//-----------------------------------------------------------------------------

void
TLState::reset()
{
    // Clear OIIO thread local mapping. It can vary depending on whether we
    // are in the update or render portion of the frame so let the next client
    // re-initialize the pointer for their needs.
    mOIIOThreadData = nullptr;
}

void
TLState::initTexturingSupport()
{
    MNRY_ASSERT(mTextureSystem);
    mOIIOThreadData = mTextureSystem->get_perthread_info();
    MNRY_ASSERT(mOIIOThreadData);
}

TLState::TLState(mcrt_common::ThreadLocalState *tls, 
                 const mcrt_common::TLSInitParams &initParams,
                 bool okToAllocBundledResources) :
    BaseTLState(tls->mThreadIdx, tls->mArena, tls->mPixelArena)
{
    mTextureSampler = MNRY_VERIFY(gPrivate.mTextureSampler);
    mTextureSystem = MNRY_VERIFY(mTextureSampler->mTextureSystem);
    mOIIOThreadData = nullptr;
}

void
TLState::initPrivate(const mcrt_common::TLSInitParams &initParams)
{
    MNRY_ASSERT(!gPrivate.mTextureSampler);

    // Initialize Texture system.
    MOONRAY_THREADSAFE_STATIC_WRITE(gPrivate.mTextureSampler 
                                  = new TextureSampler());
}

void
TLState::cleanUpPrivate()
{
    MNRY_ASSERT(gPrivate.mTextureSampler);

    delete gPrivate.mTextureSampler;

    gPrivate.~Private();
    new (&gPrivate) Private;

    MNRY_ASSERT(gPrivate.mTextureSampler == nullptr);
}

TextureSampler *
getTextureSampler()
{
    return gPrivate.mTextureSampler;
}

HUD_VALIDATOR( TextureTLState );

//-----------------------------------------------------------------------------

} // namespace texture
} // namespace moonray

