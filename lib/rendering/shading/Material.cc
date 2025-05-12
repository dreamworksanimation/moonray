// Copyright 2023-2025 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#include "Material.h"

#include <moonray/rendering/mcrt_common/AffinityManager.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/render/util/StrUtil.h>

namespace moonray {
namespace shading {

tbb::mutex Material::sMaterialListMutex;
MaterialPtrList Material::sAllMaterials;
MaterialPtrList Material::sQueuelessMaterials;

std::shared_ptr<ShadeQueueListInfoManager> Material::sShadeQueueListInfoManager;

std::atomic<uint32_t> Material::sDeferredEntryCalls;
std::atomic<uint32_t> Material::sTotalDeferredEntries;

//------------------------------------------------------------------------------------------

#ifdef SHADEQUEUE_ACCESS_LOG
class ShadeQueueAccessLog
{
public:
    ShadeQueueAccessLog()
    {
        mAccCounterTbl.resize(8, 0);
    }
    ~ShadeQueueAccessLog()
    {
        std::cerr << ">> Material.cc " << show() << '\n';
    }

    void increment(const unsigned numaNodeId) {
        if (numaNodeId == ~0) mAccCounter++;
        else mAccCounterTbl[numaNodeId]++;
    }

    std::string show() const
    {
        std::ostringstream ostr;
        ostr << "ShadeQueueAccessLog {\n"
             << "  No-Mem-Affinity {\n"
             << "    mAccCounter counter:" << mAccCounter << '\n'
             << "  }\n";
        size_t total = 0;
        ostr << "  Mem-Affinity {\n";
        for (size_t i = 0 ; i < mAccCounterTbl.size(); ++i) {
            total += mAccCounterTbl[i];
            ostr << "   mAccCounterTbl[" << i << "] counter:" << mAccCounterTbl[i] << '\n';
        }
        ostr << "  } mAccCounterTbl total:" << total << '\n'
             << "}";
        return ostr.str();
    }

protected:
    size_t mAccCounter {0};
    std::vector<size_t> mAccCounterTbl;
};

std::shared_ptr<ShadeQueueAccessLog> gShadeQueueAccessLog;
#endif // end SHADEQUEUE_ACCESS_LOG

//------------------------------------------------------------------------------------------

#ifdef PLATFORM_APPLE
ShadeQueueInfo::ShadeQueueInfo(const unsigned numaNodeId,
                               const unsigned shadeQueueSize,
                               ShadeQueue::Handler handler,
                               const Material* material,
                               ShadeQueueListInfo* shadeQueueListInfo)
//
// for Mac
//
    : mNumaNodeId {numaNodeId}
    , mMaterial {material}
    , mShadeQueueListInfo {shadeQueueListInfo}
{
    //
    // Memory Affinity disabled
    //
    mShadeQueue = scene_rdl2::util::alignedMallocCtor<ShadeQueue>(CACHE_LINE_SIZE);
    mShadeEntries =
        scene_rdl2::util::alignedMallocArray<ShadeQueue::EntryType>(shadeQueueSize, CACHE_LINE_SIZE);
    mShadeQueue->init(shadeQueueSize, mShadeEntries);
    mShadeQueue->setHandler(handler, const_cast<Material*>(mMaterial));
}
#else // else PLATFORM_APPLE

ShadeQueueInfo::ShadeQueueInfo(const unsigned numaNodeId,
                               const unsigned shadeQueueSize,
                               ShadeQueue::Handler handler,
                               const Material* material,
                               ShadeQueueListInfo* shadeQueueListInfo)
//
// for Linux
//
    : mNumaNodeId {numaNodeId}
    , mMaterial {material}
    , mShadeQueueListInfo {shadeQueueListInfo}
{
    if (mNumaNodeId == ~0) {
        //
        // Memory Affinity disabled
        //
        mShadeQueue = scene_rdl2::util::alignedMallocCtor<ShadeQueue>(CACHE_LINE_SIZE);
        mShadeEntries =
            scene_rdl2::util::alignedMallocArray<ShadeQueue::EntryType>(shadeQueueSize, CACHE_LINE_SIZE);
        mShadeQueue->init(shadeQueueSize, mShadeEntries);
        mShadeQueue->setHandler(handler, const_cast<Material*>(mMaterial));

    } else {
        //
        // Memory Affinity enabled
        //
        std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(mNumaNodeId);

        // This is an experimental code and will be kept for future reference.
        // I tested an adjusted version of ShadeQueue but this decreased performance a bit.
        // This makes sense because flush happens more frequently.
        // So, I currently keep the queue size equal to the original non-Multi-Bank version. 
        /*
        const unsigned numThreadCount = mcrt_common::AffinityManager::get()->getCpu()->getNumThreads();
        const unsigned activeThreadCount = currMemNode->getActiveThreadCount();
        const float scale = static_cast<float>(activeThreadCount) / static_cast<float>(numThreadCount);
        */
        const float scale = 1.0f;

        const unsigned currShadeQueueSize = shadeQueueSize * scale;
        const scene_rdl2::NumaNode* currNumaNode = currMemNode->getNumaNode();

        // We should check the alignment size here. We have limited support for alignment size because
        // we use the simplified high speed operation of NUMA-node based memory allocation.
        // See more detail on comment of scene_rdl2/lib/render/util/NumaInfo.h alignmentSizeCheck()
        if (!currNumaNode->alignmentSizeCheck(CACHE_LINE_SIZE)) {
            std::ostringstream ostr;
            ostr << "FATAL-ERROR : Material.cc ShadeQueueInfo constructor. alignementSizeCheck failed."
                 << " requested-alignment-size:" << CACHE_LINE_SIZE << '\n';
            scene_rdl2::logging::Logger::fatal(ostr.str());
            exit(1);
        }

        // callback procedure for memory allocation, alignment size check have been done already. 
        auto numaNodeAlloc = [&](size_t size, size_t alignment) -> void* {
            return currNumaNode->alloc(size);
        };

        mShadeQueue =
            scene_rdl2::util::alignedMallocCtorBasis<ShadeQueue>
            (CACHE_LINE_SIZE, numaNodeAlloc);
        mShadeEntries =
            scene_rdl2::util::alignedMallocArrayBasis<ShadeQueue::EntryType>
            (currShadeQueueSize, CACHE_LINE_SIZE, numaNodeAlloc);
        mShadeQueue->init(currShadeQueueSize, mShadeEntries);
        mShadeQueue->setHandler(handler, const_cast<Material*>(mMaterial));

        mShadeEntriesAllocatedSize = currShadeQueueSize * sizeof(ShadeQueue::EntryType);
    }
}

#endif // end of Non PLATFORM_APPLE

#ifdef PLATFORM_APPLE

ShadeQueueInfo::~ShadeQueueInfo()
//
// for Mac
//
{
    if (!mShadeQueue) return; // early exit

    mShadeQueueListInfo->removeShadeQueue(mShadeQueue);

    //
    // Memory Affinity disabled
    //
    scene_rdl2::util::alignedFreeArray(mShadeEntries);
    scene_rdl2::util::alignedFreeDtor(mShadeQueue);
}

#else // else PLATFORM_APPLE

ShadeQueueInfo::~ShadeQueueInfo()
//
// for Linux
//
{
    if (!mShadeQueue) return; // early exit

    mShadeQueueListInfo->removeShadeQueue(mShadeQueue);

    if (mNumaNodeId == ~0) {
        //
        // Memory Affinity disabled
        //
        scene_rdl2::util::alignedFreeArray(mShadeEntries);
        scene_rdl2::util::alignedFreeDtor(mShadeQueue);

    } else {
        //
        // Memory Affinity enabled
        //
        std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(mNumaNodeId);
        const scene_rdl2::NumaNode* currNumaNode = currMemNode->getNumaNode();

        scene_rdl2::util::alignedFreeArrayBasis<ShadeQueue::EntryType>
            (mShadeEntries, [&](const void* ptr) {
                currNumaNode->free(const_cast<void*>(ptr), mShadeEntriesAllocatedSize);
            });
        scene_rdl2::util::alignedFreeDtorBasis<ShadeQueue>
            (mShadeQueue, [&](const void* ptr) {
                currNumaNode->free(const_cast<void*>(ptr), sizeof(ShadeQueue));
            });
    }
}

#endif // end of Non PLATFORM_APPLE

#ifdef SHADEQUEUE_ACCESS_LOG
void
ShadeQueueInfo::accessLog() const
{
    if (gShadeQueueAccessLog) gShadeQueueAccessLog->increment(mNumaNodeId);
}
#endif // end SHADEQUEUE_ACCESS_LOG

//------------------------------------------------------------------------------------------

ShadeQueueInfoManager::ShadeQueueInfoManager(const unsigned shadeQueueSize,
                                             ShadeQueue::Handler handler,
                                             Material* material,
                                             ShadeQueueListInfoManager* shadeQueueListInfoManager)
{
    auto genShadeQueueInfoShPtrMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity enabled
        //
        // It would be nice if we could allocate the ShadeQueueInfo object itself by NUMA-Node-Based memory
        // allocation precisely but the current implementation is not. More fine-grain NUMA-node-based
        // memory allocation would be the next future task.
        // Non-NUMA-Node-Based memory allocation for the ShadeQueueInfo object itself is not bad and has
        // reasonable performance so far.
        //
        return std::make_shared<ShadeQueueInfo>(numaNodeId,
                                                shadeQueueSize,
                                                handler,
                                                material,
                                                shadeQueueListInfoManager->getShadeQueueListInfoPtr(numaNodeId));
    };
    auto genShadeQueueInfoShPtrNonMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity disabled
        //
        return std::make_shared<ShadeQueueInfo>(numaNodeId,
                                                shadeQueueSize,
                                                handler,
                                                material,
                                                shadeQueueListInfoManager->getShadeQueueListInfoPtr(numaNodeId));
    };

    std::shared_ptr<mcrt_common::MemoryAffinityManager> memAff = mcrt_common::AffinityManager::get()->getMem();
    unsigned materialId = ~0;
    if (memAff->getMemAffinityEnable()) {
        //
        // Memory Affinity enabled
        //
        const std::vector<std::shared_ptr<mcrt_common::MemoryNode>>& memNodeTbl = memAff->getMemNodeTbl();

        mNumaNodeShadeQueueInfoTbl.resize(memNodeTbl.size());
        for (size_t numaNodeId = 0; numaNodeId < mNumaNodeShadeQueueInfoTbl.size(); ++numaNodeId) {
            if (memNodeTbl[numaNodeId]) {
                //
                // Active NUMA-node
                //
                std::shared_ptr<ShadeQueueInfo> currShadeQueueInfo =
                    genShadeQueueInfoShPtrMemAffinity(numaNodeId);

                mNumaNodeShadeQueueInfoTbl[numaNodeId] = currShadeQueueInfo;
                uint32_t currMaterialId =
                    shadeQueueListInfoManager->pushBackShadeQueue(numaNodeId,
                                                                  currShadeQueueInfo->getShadeQueue());
                if (materialId == ~0) {
                    materialId = currMaterialId;
                } else {
                    if (materialId != currMaterialId) {
                        std::ostringstream ostr;
                        ostr << "FATAL-ERROR : Material.cc ShadeQueueInfoManager constructor."
                             << " materialId:" << materialId << " != currMaterialId:" << currMaterialId << '\n';
                        scene_rdl2::logging::Logger::fatal(ostr.str());
                        exit(1);
                    }
                }
            } else {
                //
                // Non Active NUMA-node
                //
                mNumaNodeShadeQueueInfoTbl[numaNodeId] = nullptr;
            }
        } // numaNodeId loop
        mShadeQueueInfo = nullptr;        

        if (materialId == ~0) {
            std::ostringstream ostr;
            ostr << "FATAL-ERROR : Material.cc ShadeQueueInfoManager constructor."
                 << " no active NUMA-node during allocShadeQueueNumaNode()";
            scene_rdl2::logging::Logger::fatal(ostr.str());
            exit(1);
        }

    } else {
        //
        // Memory Affinity disabled
        //
        constexpr unsigned numaNodeId = ~0;

        mShadeQueueInfo = genShadeQueueInfoShPtrNonMemAffinity(numaNodeId);
        materialId = shadeQueueListInfoManager->pushBackShadeQueue(numaNodeId, mShadeQueueInfo->getShadeQueue());
        mNumaNodeShadeQueueInfoTbl.clear();
    }

    mMaterialId = materialId;
}

//------------------------------------------------------------------------------------------

void
ShadeQueueListInfo::removeShadeQueue(ShadeQueue* shadeQueue) // MTsafe
{
    std::lock_guard<std::mutex> lock(mShadeQueueMutex);

    // Check the shade queue size also since it may have already been destroyed
    // during global program destruction time.
    if (mShadeQueues.size()) {
        // Remove ourselves from global list of shade queues.
        for (auto it = mShadeQueues.begin(); it != mShadeQueues.end(); ++it) {
            if (*it == shadeQueue) {
                mShadeQueues.erase(it);
                break;
            }
        }
    }
}

bool
ShadeQueueListInfo::areAllShadeQueueEmpty() const
{
    for (auto it = mShadeQueues.begin(); it != mShadeQueues.end(); ++it) {
        if (!(*it)->isEmpty()) {
            return false;
        }
    }
    return true;
}

unsigned
ShadeQueueListInfo::flushNonEmptyShadeQueue(mcrt_common::ThreadLocalState* tls)
{
    // Always force mFlushCycleIdx to increment by at least 1.
    size_t startIdx = mFlushCycleIdx++;

    scene_rdl2::alloc::Arena *arena = &tls->mArena;
    SCOPED_MEM(arena);

    for (size_t i = 0; i < mShadeQueues.size(); ++i) {

        const size_t cycleIdx = startIdx + i;
        ShadeQueue *queue = mShadeQueues[cycleIdx % mShadeQueues.size()];
#if 1
        const unsigned flushed = queue->drain(tls, VLEN, arena);
#else
        const unsigned flushed = queue->flush(tls, arena);
#endif

        if (flushed) {
            // There is a harmless race here (so we can avoid any heavier synchronization).
            // Overall it's fine and won't cause any problems.
            if (cycleIdx > mFlushCycleIdx) {
                mFlushCycleIdx = cycleIdx;
            }
            return flushed;
        }
    }

    return 0;
}

std::string
ShadeQueueListInfo::show() const
{
    auto showShadeQueues = [&]() {
        std::ostringstream ostr;
        ostr << "mShadeQueues:(size:" << mShadeQueues.size() << ")";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "ShadeQueueListInfo {" << "  mNumaNodeId:" << mNumaNodeId << ' ' << showShadeQueues() << " }";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

ShadeQueueListInfoManager::ShadeQueueListInfoManager()
{
    auto genShadeQueueListInfoShPtrMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity enabled
        //
        // It would be nice if we could allocate the ShadeQueueInfo object itself by NUMA-Node-Based memory
        // allocation precisely but the current implementation is not. More fine-grain NUMA-node-based
        // memory allocation would be the next future task.
        // Non-NUMA-Node-Based memory allocation for the ShadeQueueListInfo object itself is not bad and has
        // reasonable performance so far.
        //
        return std::make_shared<ShadeQueueListInfo>(numaNodeId);
    };
    auto genShadeQueueListInfoShPtrNonMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity disabled
        //
        return std::make_shared<ShadeQueueListInfo>(numaNodeId);
    };

    std::shared_ptr<mcrt_common::MemoryAffinityManager> memAff = mcrt_common::AffinityManager::get()->getMem();
    mMemAffinityEnable = memAff->getMemAffinityEnable();
    if (mMemAffinityEnable) {
        //
        // Memory Affinity enabled
        //
        const std::vector<std::shared_ptr<mcrt_common::MemoryNode>>& memNodeTbl = memAff->getMemNodeTbl();

        mNumaNodeShadeQueueListInfoTbl.resize(memNodeTbl.size());
        for (size_t numaNodeId = 0; numaNodeId < mNumaNodeShadeQueueListInfoTbl.size(); ++numaNodeId) {
            if (memNodeTbl[numaNodeId]) {
                mNumaNodeShadeQueueListInfoTbl[numaNodeId] = genShadeQueueListInfoShPtrMemAffinity(numaNodeId);
            } else {
                mNumaNodeShadeQueueListInfoTbl[numaNodeId] = nullptr;
            }
        }
        mShadeQueueListInfo = nullptr;
    } else {
        //
        // Memory Affinity disabled
        //
        constexpr unsigned numaNodeId = ~0;
        mShadeQueueListInfo = genShadeQueueListInfoShPtrNonMemAffinity(numaNodeId);
        mNumaNodeShadeQueueListInfoTbl.clear();
    }
}

unsigned
ShadeQueueListInfoManager::getAllShadeQueuesCount()
{
    unsigned total = 0;
    if (mMemAffinityEnable) {
        //
        // Memory Affinity enabled
        //
        for (auto itr : mNumaNodeShadeQueueListInfoTbl) {
            if (itr) { 
                total += itr->getShadeQueuesSize();
            }
        }

    } else {
        //
        // Memory Affinity disabled
        //
        total = mShadeQueueListInfo->getShadeQueuesSize();
    }
    return total;
}

std::string
ShadeQueueListInfoManager::show() const
{
    auto showShadeQueueListInfoMain = [](std::shared_ptr<ShadeQueueListInfo> info) -> std::string {
        if (!info) return "empty";
        return info->show();
    };
    auto showShadeQueueListInfo = [&]() {
        std::ostringstream ostr;
        ostr << "mShadeQueueListInfo:" << showShadeQueueListInfoMain(mShadeQueueListInfo);
        return ostr.str();
    };
    auto showShadeQueueListInfoTbl = [&]() {
        std::ostringstream ostr;
        ostr << "mNumaNodeShadeQueueListInfoTbl (size:" << getNumaNodeTblSize() << ") {\n";
        for (size_t i = 0; i < getNumaNodeTblSize(); ++i) {
            ostr << scene_rdl2::str_util::addIndent(showShadeQueueListInfoMain(mNumaNodeShadeQueueListInfoTbl[i])) << '\n';
        }
        ostr << "}";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "ShadeQueueListInfoManager {\n"
         << "  mMemAffinityEnable:" << scene_rdl2::str_util::boolStr(mMemAffinityEnable) << '\n'
         << "  Non-MEM-Affinity {\n"
         << scene_rdl2::str_util::addIndent(showShadeQueueListInfo(), 2) << '\n'
         << "  }\n"
         << "  MEM-Affinity {\n"
         << scene_rdl2::str_util::addIndent(showShadeQueueListInfoTbl(), 2) << '\n'
         << "  }\n"
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

void
DeferredEntriesInfo::retrieveDeferredEntries(mcrt_common::ThreadLocalState* tls,
                                             scene_rdl2::alloc::Arena* arena,
                                             unsigned& numEntries,
                                             SortedRayState*& entries)
{
    if (mDeferredEntries.empty()) {
        return;
    }

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_DEFER_SHADE_ENTRIES);

    std::unique_lock<std::mutex> lock(mDeferredEntryMutex);

    const unsigned numDeferredEntries = (unsigned)mDeferredEntries.size();
    const unsigned totalEntries = numEntries + numDeferredEntries;

    SortedRayState *allEntries = arena->allocArray<SortedRayState>(totalEntries);

    memcpy(allEntries, &mDeferredEntries[0], numDeferredEntries * sizeof(SortedRayState));
    mDeferredEntries.clear();

    // Don't hold the lock any longer than we strictly have to.
    lock.unlock();

    memcpy(allEntries + numDeferredEntries, entries, numEntries * sizeof(SortedRayState));

    numEntries = totalEntries;
    entries = allEntries;

    // The shadeBundleHandler expects sorted entries so redo that sort since
    // we're concatenating various SortRayState arrays from different threads.
    std::sort(entries, entries + numEntries,
              [](const SortedRayState &a, const SortedRayState &b) -> bool
    {
        return EXTRACT_KEY32(a, offsetof(SortedRayState, mSortKey)) <
               EXTRACT_KEY32(b, offsetof(SortedRayState, mSortKey));
    });

    MNRY_ASSERT( (scene_rdl2::util::isSorted32<SortedRayState, offsetof(SortedRayState, mSortKey)>(numEntries, entries)) );
}

void
DeferredEntriesInfo::clear()
{
    std::lock_guard<std::mutex> lock(mDeferredEntryMutex);
    mDeferredEntries.clear();
}

DeferredEntriesManager::DeferredEntriesManager()
{
    auto genDeferredEntriesInfoShPtrMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity enabled
        //
        // It would be nice if we could allocate the ShadeQueueInfo object itself by NUMA-Node-Based memory
        // allocation precisely but the current implementation is not. More fine-grain NUMA-node-based
        // memory allocation would be the next future task.
        // Non-NUMA-Node-Based memory allocation for the DeferredEntriesInfo object itself is not bad and has
        // reasonable performance so far.
        //
        return std::make_shared<DeferredEntriesInfo>();
    };
    auto genDeferredEntriesInfoShPtrNonMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity disabled
        //
        return std::make_shared<DeferredEntriesInfo>();
    };

    std::shared_ptr<mcrt_common::MemoryAffinityManager> memAff = mcrt_common::AffinityManager::get()->getMem();
    if (memAff->getMemAffinityEnable()) {
        //
        // Memory Affinity enabled
        //
        const std::vector<std::shared_ptr<mcrt_common::MemoryNode>>& memNodeTbl = memAff->getMemNodeTbl();

        mNumaNodeDeferredEntriesInfoTbl.resize(memNodeTbl.size());
        for (size_t numaNodeId = 0; numaNodeId < mNumaNodeDeferredEntriesInfoTbl.size(); ++numaNodeId) {
            if (memNodeTbl[numaNodeId]) {
                mNumaNodeDeferredEntriesInfoTbl[numaNodeId] = genDeferredEntriesInfoShPtrMemAffinity(numaNodeId);
            } else {
                mNumaNodeDeferredEntriesInfoTbl[numaNodeId] = nullptr;
            }
        }
        mDeferredEntriesInfo = nullptr;

    } else {
        //
        // Memory Affinity disabled
        //
        constexpr unsigned numaNodeId = ~0;
        mDeferredEntriesInfo = genDeferredEntriesInfoShPtrNonMemAffinity(numaNodeId);
        mNumaNodeDeferredEntriesInfoTbl.clear();
    }
}

void
DeferredEntriesManager::clear()
{
    if (mDeferredEntriesInfo) {
        //
        // Memory Affinity disabled
        //
        mDeferredEntriesInfo->clear();

    } else {
        //
        // Memory Affinity enabled
        //
        for (size_t numaNodeId = 0; numaNodeId < mNumaNodeDeferredEntriesInfoTbl.size(); ++numaNodeId) {
            if (mNumaNodeDeferredEntriesInfoTbl[numaNodeId]) {
                mNumaNodeDeferredEntriesInfoTbl[numaNodeId]->clear();
            }
        }
    }
}

//------------------------------------------------------------------------------------------

Material::Material(const scene_rdl2::rdl2::SceneObject & owner) :
    RootShader(owner),
    mMaterialLabelId(-1),   // no material label
    mLpeMaterialLabelId(-1) // no lpe material label
{
    tbb::mutex::scoped_lock lock(sMaterialListMutex);
    sAllMaterials.push_back(this);
    sQueuelessMaterials.push_back(this);
    mMaterialId = 0;
}

Material::~Material()
{
    // Not thread safe, all objects should be created/destroyed on a single thread.
    mShadeQueueInfoManager = nullptr; // explicitly delete shadeQueueInfoManager;
    {
        tbb::mutex::scoped_lock lock(sMaterialListMutex);

        // Remove ourselves from global list of Materials.
        for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
            if (*it == this) {
                sAllMaterials.erase(it);
                break;
            }
        }
    }
}

void
Material::deferEntriesForLaterProcessing(mcrt_common::ThreadLocalState *tls,
                                         const unsigned numEntries,
                                         SortedRayState *entries)
{
    if (!numEntries) {
        return;
    }

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_DEFER_SHADE_ENTRIES);

    // Keep track of calls, if this gets too high then we'll need to investigate.
    ++sDeferredEntryCalls;
    sTotalDeferredEntries += numEntries;

    const unsigned deferredCalls = sDeferredEntryCalls;

    const unsigned maxCallsToLog = 5;

    if (deferredCalls < maxCallsToLog) {
        mcrt_common::ExclusiveAccumulators *acc = MNRY_VERIFY(getExclusiveAccumulators(tls));
        unsigned accStack = acc->getStackSize();
        scene_rdl2::logging::Logger::warn("Call to Material::deferEntriesForLaterProcessing encountered "
                                          "(accumulator stack = ", accStack,
                                          ", handler stack = ", tls->mHandlerStackDepth, ".");
    } else if (deferredCalls == maxCallsToLog) {
        scene_rdl2::logging::Logger::warn("Multiple calls to Material::deferEntriesForLaterProcessing "
                                          "encountered, no more will be reported this frame.");
    }

    const unsigned numaNodeId = tls->mArena.getNumaNodeId();
    mDeferredEntriesManager->insert(numaNodeId, entries, numEntries);
}

void
Material::retrieveDeferredEntries(mcrt_common::ThreadLocalState *tls,
                                  scene_rdl2::alloc::Arena *arena,
                                  unsigned &numEntries,
                                  SortedRayState *&entries)
{
    const unsigned numaNodeId = arena->getNumaNodeId();
    mDeferredEntriesManager->retrieveDeferredEntries(numaNodeId, tls, arena, numEntries, entries); 
}

// static function
void
Material::allocShadeQueues(const unsigned shadeQueueSize, ShadeQueue::Handler handler)
//
// Thread safe, but should only be called on a single thread after scene update
// time but before rendering starts.
//
{
#ifdef SHADEQUEUE_ACCESS_LOG
    gShadeQueueAccessLog = std::make_shared<ShadeQueueAccessLog>(); // for DEBUG
#endif // end SHADEQUEUE_ACCESS_LOG

    if (!sShadeQueueListInfoManager) {
        // This should be constructed once before the initial action of each material's allocShadeQueue()
        sShadeQueueListInfoManager = std::make_shared<ShadeQueueListInfoManager>();
    }

    {
        tbb::mutex::scoped_lock lockMaterialMutex(sMaterialListMutex);

        for (auto it = sQueuelessMaterials.begin(); it != sQueuelessMaterials.end(); ++it) {
            (*it)->allocShadeQueue(shadeQueueSize, handler);
            (*it)->allocDeferredEntriesManager();
        }

        sQueuelessMaterials.clear();
    }
}

// static function
void
Material::initMaterialIds()
{
    uint32_t id = 1;
    for (auto it = sQueuelessMaterials.begin(); it != sQueuelessMaterials.end(); ++it) {
        (*it)->setMaterialId(id++);
    }

    sQueuelessMaterials.clear();
}

// static function
ShadeQueueList&
Material::getAllShadeQueues(const unsigned numaNodeId)
{
    return sShadeQueueListInfoManager->getShadeQueues(numaNodeId);
}

// static function
unsigned
Material::getAllShadeQueuesCount()
{
    return sShadeQueueListInfoManager->getAllShadeQueuesCount();
}

// static function
bool
Material::areAllShadeQueuesEmptyAllNumaNode()
{
    std::shared_ptr<mcrt_common::MemoryAffinityManager> memAff = mcrt_common::AffinityManager::get()->getMem();
    if (!memAff->getMemAffinityEnable()) {
        constexpr unsigned numaNodeId = ~0;
        return areAllShadeQueuesEmpty(numaNodeId);
    }

    for (unsigned nodeId = 0; nodeId < memAff->getMemNodeTblSize(); ++nodeId) {
        if (memAff->isActiveMemNode(nodeId)) {
            if (!areAllShadeQueuesEmpty(nodeId)) return false;
        }
    }
    return true;
}

// static function
bool
Material::areAllShadeQueuesEmpty(const unsigned numaNodeId)
{
    if (!sShadeQueueListInfoManager->areAllShadeQueueEmpty(numaNodeId)) {
        return false;
    }

    // We also need to check for the case where a Material's queue may be empty
    // but it could still have some entries stored for deferred processing.
    for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
        if (!(*it)->mDeferredEntriesManager->isEmpty(numaNodeId)) {
            return false;
        }
    }

    return true;
}

// static function
unsigned
Material::flushNonEmptyShadeQueue(mcrt_common::ThreadLocalState *tls)
{
    const unsigned numaNodeId = tls->mArena.getNumaNodeId();
    const unsigned flushed = sShadeQueueListInfoManager->flushNonEmptyShadeQueue(numaNodeId, tls);
    if (flushed > 0) {
        return flushed;
    }
    scene_rdl2::alloc::Arena *arena = &tls->mArena;

    // We also need to check for the case where a Material's queue may be empty
    // but it could still have some entries stored for deferred processing.
    // In this case we move the deferred entries into the actual shader
    // queues and return signal that there is still shade queues which contain
    // entries.
    unsigned numDeferredEntries = 0;
    for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
        const bool isDeferredEmpty = (*it)->mDeferredEntriesManager->isEmpty(numaNodeId);
        if (!isDeferredEmpty) {
            Material *material = *it;
            unsigned numEntries = 0;
            SortedRayState *entries = nullptr;
            material->retrieveDeferredEntries(tls, arena, numEntries, entries);

            // This check is here since another thread may have stolen the
            // deferred entries in the meantime.
            if (numEntries && entries) {
                numDeferredEntries += numEntries;
                ShadeQueue* queue = material->getShadeQueue(numaNodeId);
                MNRY_VERIFY(queue)->addEntries(tls, numEntries, entries, arena);
            }
        }
    }

    // This is expected to be zero 99.99% of the time.
    return numDeferredEntries;
}

void
Material::allocShadeQueue(const unsigned shadeQueueSize, ShadeQueue::Handler handler)
{
    mShadeQueueInfoManager = std::make_shared<ShadeQueueInfoManager>(shadeQueueSize,
                                                                     handler,
                                                                     this,
                                                                     sShadeQueueListInfoManager.get()); 
    const uint32_t materialId = mShadeQueueInfoManager->getMaterialId();

    // TODO: We don't support deleting primitives currently. We'll need to add
    // some extra logic to maintain a valid bundled id when that gets implemented.
    //    mMaterialId = uint32_t(materialId);
    setMaterialId(materialId);
}

void
Material::allocDeferredEntriesManager()
{
    mDeferredEntriesManager = std::make_shared<DeferredEntriesManager>();
}

void
Material::printDeferredEntryWarnings()
{
    const unsigned numCalls = sDeferredEntryCalls;
    const unsigned numEntries = sTotalDeferredEntries;

    if (numCalls) {
        scene_rdl2::logging::Logger::warn(numCalls,
                                          " call(s) to Material::deferEntriesForLaterProcessing containing ",
                                          numEntries, " entries.");
    }
}

void
Material::resetDeferredEntryState()
{
    for (auto it = sAllMaterials.begin(); it != sAllMaterials.end(); ++it) {
        Material *material = *it;
        material->mDeferredEntriesManager->clear();
    }

    sDeferredEntryCalls = 0;
    sTotalDeferredEntries = 0;
}

} // namespace shading
} // namespace moonray
