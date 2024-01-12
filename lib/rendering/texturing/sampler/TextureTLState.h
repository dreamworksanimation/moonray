// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "TextureTLState.hh"
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

//  OiiO includes for access to texture system.
#include <OpenImageIO/version.h>
#include <OpenImageIO/texture.h>

namespace moonray {
namespace texture {

class TextureSampler;
class TextureSystem;

// Expose for HUD validation.
class TLState;
typedef texture::TLState TextureTLState;

//-----------------------------------------------------------------------------

// This class can't be instantiated directly. It is meant to be used as a base
// class for the shading::TLState class to provide texturing related hooks.

class TLState : public mcrt_common::BaseTLState
{
public:
    virtual void reset() override;

    void initTexturingSupport();

    /// HUD validation.
    static uint32_t hudValidation(bool verbose) { TEXTURE_TL_STATE_VALIDATION; }

    typedef OIIO::TextureSystem TextureSystem;
    typedef OIIO::TextureSystem::Perthread Perthread;
    TEXTURE_TL_STATE_MEMBERS;

private:
    friend class shading::TLState;

    TLState(mcrt_common::ThreadLocalState *tls,
            const mcrt_common::TLSInitParams &initParams,
            bool okToAllocBundledResources);

    static void initPrivate(const mcrt_common::TLSInitParams &initParams);
    static void cleanUpPrivate();

    DISALLOW_COPY_OR_ASSIGNMENT(TLState);
};

// Get the global texture sampler.
TextureSampler *getTextureSampler();

//-----------------------------------------------------------------------------

} // namespace texture

inline mcrt_common::ExclusiveAccumulators *
getExclusiveAccumulators(texture::TLState *tls)
{
    MNRY_ASSERT(tls);
    return tls->getInternalExclusiveAccumulatorsPtr();
}

} // namespace moonray

