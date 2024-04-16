// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/mcrt_common/Bundle.h>
#include <moonray/rendering/shading/Types.h>
#include <scene_rdl2/scene/rdl2/Map.h>
#include <scene_rdl2/scene/rdl2/Material.h>

#include <memory>

namespace moonray {
namespace shading {

class Material;
typedef std::vector<Material *> MaterialPtrList;

typedef mcrt_common::SharedQueue<SortedRayState, true, offsetof(SortedRayState, mSortKey)> ShadeQueue;
typedef std::vector<ShadeQueue *> ShadeQueueList;

//
// Convenience function for iterating over all shade queues.
// Not thread safe.
//
// Example of syntax:
//
//    forEachShadeQueue(tls, [&](ThreadLocalState *tls, ShadeQueue *queue)
//    {
//        queue->doSomething();
//    });
//

class Material : public RootShader
{
public:
    typedef std::vector<int> LobeLabelIds;
    typedef std::vector<int> LpeLobeLabelIds;

    struct ExtraAov
    {
        int mLabelId;
        const scene_rdl2::rdl2::Map *mMap;
    };

    explicit Material(const scene_rdl2::rdl2::SceneObject & owner);
    virtual ~Material();

    //
    // Primitive Attribute Aovs
    // 
    std::vector<char> &getAovFlags() { return mAovFlags; }
    const std::vector<char> &getAovFlags() const { return mAovFlags; }

    //
    // Material Aov: Lobe LabelIds
    //
    LobeLabelIds &getLobeLabelIds() { return mLobeLabelIds; }
    const LobeLabelIds &getLobeLabelIds() const { return mLobeLabelIds; }

    //
    // Light Aov: Lobe LabelIds
    //
    LpeLobeLabelIds &getLpeLobeLabelIds() { return mLpeLobeLabelIds; }
    const LpeLobeLabelIds &getLpeLobeLabelIds() const { return mLpeLobeLabelIds; }

    //
    // Material Aov: Material LabelId
    //
    void setMaterialLabelId(int32_t labelId) { mMaterialLabelId = labelId; }
    int32_t getMaterialLabelId() const { return mMaterialLabelId; }

    //
    // Light Aov: Material LabelId
    //
    void setLpeMaterialLabelId(int32_t labelId) { mLpeMaterialLabelId = labelId; }
    int32_t getLpeMaterialLabelId() const { return mLpeMaterialLabelId; }

    //
    // Extra Aovs
    //
    void setExtraAovs(const std::vector<ExtraAov> &ea) { mExtraAovs = ea; }
    void setPostScatterExtraAovs(const std::vector<ExtraAov> &ea) { mPostScatterExtraAovs = ea; }
    const std::vector<ExtraAov> &getExtraAovs() const { return mExtraAovs; }
    const std::vector<ExtraAov> &getPostScatterExtraAovs() const { return mPostScatterExtraAovs; }

    //
    // Shade queue APIs.
    //

    // Const hole here, we're returning a non-const ShadeQueue.
    ShadeQueue *getShadeQueue() const { return mShadeQueue; }

    //
    // The following 2 APIs were added to address MOONRAY-2431.
    // The shade queue is a shared resource, so threads can insert both primary
    // and secondary ray hits into any shade queue at anytime. In extreme cases,
    // where there are many bounces like in hair, this can potentially result
    // in a stack overflow.
    //

    // This function stores a set of rays which need to be shaded for later
    // processing.
    void deferEntriesForLaterProcessing(mcrt_common::ThreadLocalState *tls,
                                        unsigned numEntries,
                                        SortedRayState *entries);

    // This function allocates from the passed in arena. It's up to the caller
    // to preserve the allocations until they are no longer needed.
    void retrieveDeferredEntries(mcrt_common::ThreadLocalState *tls,
                                 scene_rdl2::alloc::Arena *arena,
                                 unsigned &numEntries,
                                 SortedRayState *&entries);

    // This function will allocate shade queues for each material which doesn't already
    // own a shade queue. If a material already owns a shade queue, it is ignored by this call.
    // It also sets the material ids.
    // It should only be called when in bundled mode.
    static void allocShadeQueues(unsigned shadeQueueSize, ShadeQueue::Handler handler);
    // This function sets the material ids for scalar mode.
    static void initMaterialIds();

    static ShadeQueueList &getAllShadeQueues()  { return sShadeQueues; }

    // Check that all queues are now empty.
    static bool areAllShadeQueuesEmpty();

    // Returns the number of entries flushed, or 0 if there were no non-empty queues.
    static unsigned flushNonEmptyShadeQueue(mcrt_common::ThreadLocalState *tls);

    // Print out diagnostics for deferred entries.
    static void printDeferredEntryWarnings();

    // Reset all deferred entry state.
    static void resetDeferredEntryState();

protected:

    // This must be called when in bundled mode before starting to render.
    // Calling Material::allocShadeQueues will ensure that it gets called.
    void allocShadeQueue(unsigned shadeQueueSize, ShadeQueue::Handler handler);


    inline void setMaterialId(uint32_t id) { mMaterialId = id; }

    ShadeQueue *            mShadeQueue;
    ShadeQueue::EntryType * mShadeEntries;

    // These are used to mask the aovSchema
    std::vector<char>       mAovFlags;
    // These map bsdf lobe label bits to material aov labelIds
    LobeLabelIds            mLobeLabelIds;
    // These map bsdf lobe label bits to light aov labelIds
    LpeLobeLabelIds         mLpeLobeLabelIds;
    // The labelId used in material aovs
    int32_t                 mMaterialLabelId;
    // the labelId used in light aovs
    int32_t                 mLpeMaterialLabelId;
    // Extra Aovs
    std::vector<ExtraAov>   mExtraAovs;
    std::vector<ExtraAov>   mPostScatterExtraAovs;

    // Deferred entries. In general we shouldn't be locking or allocating heap
    // memory during the mcrt phase. Since this is an exceptional case and
    // should only happen in extreme cases, we're using a mutex and vector now.
    // We will also get warned when executing this codepath at render time via
    // the logger so if it becomes a common case, we need to revisit and remove
    // these locks and heap allocations.
    tbb::mutex              mDeferredEntryMutex;
    std::vector<SortedRayState> mDeferredEntries;

    static tbb::mutex       sMaterialListMutex;
    static MaterialPtrList  sAllMaterials;
    static MaterialPtrList  sQueuelessMaterials;

    static tbb::mutex       sShadeQueueMutex;
    static ShadeQueueList   sShadeQueues;

    // This is used by the flushNonEmptyShadeQueue function to iterate through all queues
    // in a cyclic fashion as opposed to starting the iteration at the beginning of the
    // queue list each time.
    static tbb::atomic<size_t> sFlushCycleIdx;

    // Shared between all Materials.
    static tbb::atomic<uint32_t> sDeferredEntryCalls;
    static tbb::atomic<uint32_t> sTotalDeferredEntries;
};

template <typename Body>
inline void
forEachShadeQueue(mcrt_common::ThreadLocalState *tls, const Body &body)
{
    ShadeQueueList &shadeQueues = Material::getAllShadeQueues();
    for (auto it = shadeQueues.begin(); it != shadeQueues.end(); ++it) {
        body(tls, *it);
    }
}

} // namespace shading
} // namespace moonray


