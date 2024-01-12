// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ShadingTLState.hh"

#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/texturing/sampler/TextureTLState.h>
#include <scene_rdl2/scene/rdl2/RootShader.h>


namespace moonray {
namespace shading {

// Expose for HUD validation.
class TLState;
typedef shading::TLState ShadingTLState;

//-----------------------------------------------------------------------------

class CACHE_ALIGN TLState : public texture::TLState
{
public:
    typedef mcrt_common::ExclusiveAccumulators ExclusiveAccumulators;

    TLState(mcrt_common::ThreadLocalState *tls,
            const mcrt_common::TLSInitParams &initParams,
            bool okToAllocBundledResources);
    virtual ~TLState();

    // Hook up primitive attribute offset table for entire shade tree.
    // Child shaders can access this directly via the TLS bypassing the extra
    // indirections. Returns true if the attributes offsets pointer was updated.
    bool getAttributeOffsetsFromRootShader(const scene_rdl2::rdl2::RootShader& rootShader)
    {
        // If there is already an attribute table, then we are a nested shader
        // (e.g. a cutout). In this case, keep using the parents attribute table.
        if (mAttributeOffsets == nullptr && rootShader.hasExtension()) {
            const shading::AttributeTable *attrTable =
                rootShader.get<shading::RootShader>().getAttributeTable();
            if (attrTable) {
                mAttributeOffsets = attrTable->getKeyOffsets();
                return true;
            }
        }
        return false;
    }

    void clearAttributeOffsets()
    {
        mAttributeOffsets = nullptr;
    }

    /// HUD validation.
    static uint32_t hudValidation(bool verbose) { SHADING_TL_STATE_VALIDATION; }

    virtual void reset() override;

    // Used as a callback which is registered with TLSInitParams.
    static std::shared_ptr<TLState> allocTls(mcrt_common::ThreadLocalState *tls,
                                             const mcrt_common::TLSInitParams &initParams,
                                             bool okToAllocBundledResources);

    SHADING_TL_STATE_MEMBERS;

    DISALLOW_COPY_OR_ASSIGNMENT(TLState);
};

//-----------------------------------------------------------------------------

/// Convenience function for iterating over all existing shading TLS instances.
template <typename Body>
finline void forEachTLS(Body const &body)
{
    unsigned numTLS = mcrt_common::getNumTBBThreads();
    mcrt_common::ThreadLocalState *tlsList = mcrt_common::getTLSList();
    for (unsigned i = 0; i < numTLS; ++i) {
        auto shadingTls = tlsList[i].mShadingTls.get();
        if (shadingTls) {
            body(shadingTls);
        }
    }
}

// Used as a callback which is registered with TLSInitParams. It allow us to
// use OIIO texturing from both the update and rendering phases of a frame.
void initTexturingSupport(TLState *tls);

//-----------------------------------------------------------------------------

} // namespace shading

inline mcrt_common::ExclusiveAccumulators *
getExclusiveAccumulators(shading::TLState *tls)
{
    MNRY_ASSERT(tls);
    return tls->getInternalExclusiveAccumulatorsPtr();
}

} // namespace moonray


