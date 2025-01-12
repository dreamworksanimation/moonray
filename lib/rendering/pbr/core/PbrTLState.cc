// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#include "DebugRay.h"
#include "PbrTLState.h"
#include "RayState.h"
#include "XPUOcclusionRayQueue.h"
#include "XPURayQueue.h"
#include <moonray/rendering/mcrt_common/AffinityManager.h>
#include <moonray/rendering/pbr/handlers/RayHandlers.h>
#include <moonray/rendering/shading/Types.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <tbb/mutex.h>

#include <cstdlib> // abort()
#include <limits>

// These aren't free so only turn it on if you are doing memory profiling.
// This will print out the peak number of pool items used for a particular run.
//#define DEBUG_RECORD_PEAK_RAYSTATE_USAGE
//#define DEBUG_RECORD_PEAK_CL1_USAGE

// The following directive should be commented out for the release version
//#define DEBUG_MSG_POOLINFO
//#define DEBUG_MSG_PRIVATE_INIT

// This directive appies more precise checks but more cost.
// We should be commented out for the release version
//#define STRICT_CHECK_FOR_GETPOOLINFO

extern "C" bool isRenderCanceled();

namespace ispc {
extern "C" uint32_t PbrTLState_hudValidation(bool);
}

namespace scene_rdl2 {
using namespace alloc;
using namespace math;
using namespace util;
}

namespace moonray {
namespace pbr {

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

namespace
{

// constants used for allocList and freeList
const int ALLOC_LIST_MAX_ITEM_SIZE = CACHE_LINE_SIZE;

// 2^4=16. We only have 4 bit space to keep number of items into 32bit int.
// We store (numTotal - 1) value in 4-bit space. We don't need to track numTotal=0 case.
// So, the max numItems for 4-bit space is 16.
const int ALLOC_LIST_MAX_NUM_ITEMS = 16;

// ALLOC_LIST_MAX_NUM_ITEMS is used for stack array allocations, keep small.
// going as high as 256 is probably ok, but this assert is meant to
// remind us to verify this.
MNRY_STATIC_ASSERT(ALLOC_LIST_MAX_NUM_ITEMS == 16);

// allocList returns and freeList accepts a 4 byte handle instead
// of a raw pointer.  this saves precious space in space-constrained
// structures that must store them.  per thread, we maintain a relatively
// small number of items (currently 64k), this means we have a little
// bit of extra space available in the index header to store additional
// information about the list.
const unsigned int ALLOC_LIST_INFO_BIT_SHIFT = 28;
const unsigned int ALLOC_LIST_INFO_BITS      = (0xFu << ALLOC_LIST_INFO_BIT_SHIFT);
MNRY_STATIC_ASSERT(ALLOC_LIST_MAX_NUM_ITEMS <=
                  ((ALLOC_LIST_INFO_BITS >> ALLOC_LIST_INFO_BIT_SHIFT) + 1));

// Per frame counter, gets reset each frame.
CACHE_ALIGN std::atomic<unsigned> gFailedRayStateAllocs;
CACHE_ALIGN std::atomic<unsigned> gFailedCL1Allocs;

// For memory profiling, see DEBUG_RECORD_PEAK_RAYSTATE_USAGE.
unsigned MAYBE_UNUSED gPeakRayStateUsage = 0; // total number of RayState allocated
unsigned MAYBE_UNUSED gPeakCL1Usage      = 0; // total number of CL1 allocated

//------------------------------------------------------------------------------------------

class PoolInfo
//
// This class keeps MemBlockManager for a particular NUMA-node.
// And manages its internal Block/Entry memory itself.
// They are allocated from the NUMA-node by special affinity memory management if proper
// NUMA-nodeId is provided.
// Memory is allocated from standard global memory if NUMA-nodeId is not defined.
//
{
public:
    PoolInfo(const unsigned numaNodeId,
             const size_t poolSize,
             const unsigned numThreads,
             const size_t entrySize,
             const char * const poolName)
        : mPoolName(poolName)
        , mNumaNodeId {numaNodeId}
    {
        init(poolSize, numThreads, entrySize);
    }
    ~PoolInfo() { cleanUp(); }

    void fullReset() { mMemBlockManager->fullReset(); }

    size_t getActualPoolSize() const { return mActualPoolSize; } // byte
    scene_rdl2::alloc::MemBlockManager* getMemBlockManager() const { return mMemBlockManager; }
    size_t getMemBlockManagerMemoryUsage() const { return mMemBlockManager->getMemoryUsage(); } // byte

    std::string show() const;

private:
    void init(const size_t poolSize, // Whole poolSize for MoonRay process, not consider NUMA-node
              const unsigned numThreads, // Total threads count of MoonRay process.
              const size_t entrySize);
    void cleanUp();

    const std::string mPoolName; // for debug
    const unsigned mNumaNodeId {~static_cast<unsigned>(0)};
    size_t mActualPoolSize {0}; // byte

    scene_rdl2::alloc::MemBlockManager* mMemBlockManager {nullptr};
    scene_rdl2::alloc::MemBlock* mBlockMemory {nullptr};
    uint8_t* mEntryMemory {nullptr};

    size_t mMemBlockManagerAllocatedSize {0}; // byte
    size_t mBlockMemoryAllocatedSize {0}; // byte
    size_t mEntryMemoryAllocatedSize {0}; // byte
};

std::string
PoolInfo::show() const
{
    auto showPoolName = [](const std::string& name) -> std::string {
        if (name.empty()) return "Not-Defined";
        else return name;
    };
    auto showNumaNodeId = [](const unsigned numaNodeId) -> std::string {
        if (numaNodeId == ~0) return "Not-Defined";
        else return std::to_string(numaNodeId);
    };
    auto showMemBlockManager = [&]() -> std::string {
        if (!mMemBlockManager) return "mMemBlockManager is empty";
        std::ostringstream ostr;
        ostr << "mMemBlockManager:0x" << std::hex << reinterpret_cast<uintptr_t>(mMemBlockManager) << std::dec << " {\n"
             << scene_rdl2::str_util::addIndent(mMemBlockManager->showStatisticalInfo()) << '\n'
             << "}";
        return ostr.str();
    };
    auto showByteSize = [](const size_t size) {
        std::ostringstream ostr;
        ostr << size << " (" << scene_rdl2::str_util::byteStr(size) << ")";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "PoolInfo {\n"
         << "  mPoolName:" << showPoolName(mPoolName) << '\n'
         << "  mNumaNodeId:" << showNumaNodeId(mNumaNodeId) << '\n'
         << "  mActualPoolSize:" << mActualPoolSize << " byte\n"
         << scene_rdl2::str_util::addIndent(showMemBlockManager()) << '\n'
         << "  mBlockMemory:0x" << std::hex << reinterpret_cast<uintptr_t>(mBlockMemory) << std::dec << '\n'
         << "  mEntryMemory:0x" << std::hex << reinterpret_cast<uintptr_t>(mEntryMemory) << std::dec << '\n'
         << "  mMemBlockManagerAllocatedSize:" << showByteSize(mMemBlockManagerAllocatedSize) << '\n'
         << "  mBlockMemoryAllocatedSize:" << showByteSize(mBlockMemoryAllocatedSize) << '\n'
         << "  mEntryMemoryAllocatedSize:" << showByteSize(mEntryMemoryAllocatedSize) << '\n'
         << "}";
    return ostr.str();
}

void
PoolInfo::init(const size_t poolSize, // Whole poolSize for MoonRay process, not consider NUMA-node
               const unsigned numThreads, // Total threads count of MoonRay process.
               const size_t entrySize)
{
    MNRY_ASSERT(!mMemBlockManager);

    unsigned currNumThreads = numThreads;

    // Using poolSize * numThreads isn't adequate for XPU mode because we
    // run out of space with low numbers of threads for things like BundledOcclRayData.
    // poolSize * 8 seems to be adequate, so for safety poolSize * 16 is used as the
    // minimum number of totalEntries.
    size_t totalEntries = poolSize * static_cast<size_t>(std::max(numThreads, 16u));
#ifndef PLATFORM_APPLE
    if (mNumaNodeId != ~0) {
        //
        // Memory Affinity enabled
        //
        // We have to adjust totalEntries depending on the ratio of active threads counts
        // on this NUMA-node against entire threads.
        const unsigned activeThreadCount = 
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(mNumaNodeId)->getActiveThreadCount();
        const unsigned totalThreadCount =
            mcrt_common::AffinityManager::get()->getCpu()->getNumThreads();
        const float scale = static_cast<float>(activeThreadCount) / static_cast<float>(totalThreadCount);
        totalEntries *= scale;

        currNumThreads = activeThreadCount;
    }
#endif // end of Non PLATFORM_APPLE

    const size_t entryStride = entrySize;
    const size_t entriesPerBlock = scene_rdl2::alloc::MemBlock::getNumEntries();

    size_t numBlocks = totalEntries / entriesPerBlock;
    if (numBlocks * entriesPerBlock < totalEntries) {
        ++numBlocks;
    }

    MNRY_ASSERT(numBlocks * entriesPerBlock >= totalEntries);

    numBlocks = std::max(numBlocks, static_cast<size_t>(currNumThreads));

    // Update the stored pool size so that the assertions in TLState::handleToCL1Ptr() remain valid
    mActualPoolSize = numBlocks * entriesPerBlock; // byte

    const size_t entryMemorySize = scene_rdl2::alloc::MemBlockManager::queryEntryMemoryRequired(numBlocks, entryStride);

    // Uncomment to see how much memory is being allocated for each pool.
    // scene_rdl2::logging::Logger::info("Attempting to allocate ", entryMemorySize, " bytes for ",
    //                                   mPoolName.c_str(), " pool.\n");

#ifndef PLATFORM_APPLE
    if (mNumaNodeId == ~0) { 
#endif // end of Non PLATFORM_APPLE
        //
        // Memory Affinity disabled
        //
        mEntryMemory = scene_rdl2::util::alignedMallocArray<uint8_t>(entryMemorySize, CACHE_LINE_SIZE);
        mBlockMemory = scene_rdl2::util::alignedMallocArrayCtor<scene_rdl2::alloc::MemBlock>(numBlocks, CACHE_LINE_SIZE);
        mMemBlockManager = scene_rdl2::util::alignedMallocCtor<scene_rdl2::alloc::MemBlockManager>(CACHE_LINE_SIZE);

#ifndef PLATFORM_APPLE
    } else {
        //
        // Memory Affinity enabled
        //
        std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(mNumaNodeId);
        const scene_rdl2::NumaNode* currNumaNode = currMemNode->getNumaNode();

        // We should check the alignment size here. We have limited support for alignment size because
        // we use the simplified high speed operation of NUMA-node based memory allocation.
        // See more detail on comment of scene_rdl2/lib/render/util/NumaInfo.h alignmentSizeCheck()
        if (!currNumaNode->alignmentSizeCheck(CACHE_LINE_SIZE)) {
            std::ostringstream ostr;
            ostr << "FATAL-ERROR : PbrTLState.cc PoolInfo::init() alignmentSizeCheck failed."
                 << " requested-alignment-size:" << CACHE_LINE_SIZE << '\n';
            scene_rdl2::logging::Logger::fatal(ostr.str());
            exit(1);
        }

        // callback procedure for memory allocation, alignment size check have been done already.
        auto numaNodeAlloc = [&](size_t size, size_t alignment) -> void* { return currNumaNode->alloc(size); };

        mEntryMemory =
            scene_rdl2::util::alignedMallocArrayBasis<uint8_t>
            (entryMemorySize, CACHE_LINE_SIZE, numaNodeAlloc);
        mBlockMemory =
            scene_rdl2::util::alignedMallocArrayCtorBasis<scene_rdl2::alloc::MemBlock>
            (numBlocks, CACHE_LINE_SIZE, numaNodeAlloc);
        mMemBlockManager =
            scene_rdl2::util::alignedMallocCtorBasis<scene_rdl2::alloc::MemBlockManager>
            (CACHE_LINE_SIZE, numaNodeAlloc);
    }
#endif // end of Non PLATFORM_APPLE
    MNRY_VERIFY(mMemBlockManager);
    if (numBlocks > std::numeric_limits<unsigned>::max()) {
        std::cerr << "FATAL-ERROR : " << __FILE__ << " " << __LINE__ << " " << __FUNCTION__
                  << " oversized internal memory request.\n"
                  << "  numBlocks:" << numBlocks << " > std::numeric_limits<unsigned>::max()\n";
        std::abort();
    }
    if (entryStride > std::numeric_limits<unsigned>::max()) {
        std::cerr << "FATAL-ERROR : " << __FILE__ << " " << __LINE__ << " " << __FUNCTION__
                  << " oversized internal memory request.\n"
                  << "  entryStride:" << entryStride << " > std::numeric_limits<unsigned>::max()\n";
        std::abort();
    }
    mMemBlockManager->init(static_cast<unsigned>(numBlocks),
                           mBlockMemory,
                           mEntryMemory,
                           static_cast<unsigned>(entryStride));

    // We keep size information for memory-free operation
    mEntryMemoryAllocatedSize = entryMemorySize;
    mBlockMemoryAllocatedSize = numBlocks * sizeof(scene_rdl2::alloc::MemBlock);
    mMemBlockManagerAllocatedSize = sizeof(scene_rdl2::alloc::MemBlockManager);

#   ifdef DEBUG_MSG_POOLINFO
    {
        using scene_rdl2::str_util::byteStr;

        const size_t memBlockSize = sizeof(scene_rdl2::alloc::MemBlock);
        const size_t memBlockMemorySize = memBlockSize * numBlocks;

        std::ostringstream ostr;
        ostr << "PoolInfo::init() (PbrTLState.cc) {\n"
             << "  poolName:" << mPoolName << '\n'
             << "  numaNodeId:" << mNumaNodeId << '\n'
             << "  poolSize:" << poolSize << '\n'
             << "  numThreads:" << numThreads << '\n'
             << "  currNumThreads:" << currNumThreads << '\n'
             << "  totalEntries:" << totalEntries << '\n'
             << "  entryStride:" << entryStride << '\n'
             << "  entryMemorySize:" << byteStr(entryMemorySize)
             << " (" << entryMemorySize << " byte) := MemBlockManager::queryEntryMemoryRequired()\n"
             << "  entriesPerBlock:" << entriesPerBlock << " := MemBlock::getNumEntries()\n"
             << "  numBlocks:" << numBlocks << " := totalEntries / entriesPerBlock\n"
             << "  MemBlock-size:" << byteStr(memBlockSize) << " (" << memBlockSize << " byte) := sizeof(MemBlock)\n"
             << "  MemBlockMemorySize:" << byteStr(memBlockMemorySize)
             << " (" << memBlockMemorySize << " byte) := memBlockSize * numBlocks\n"
             << "}";
        std::cerr << ostr.str() << '\n';
    }
#   endif // end DEBUG_MSG_POOLINFO
}

void
PoolInfo::cleanUp()
{
#ifndef PLATFORM_APPLE
    if (mNumaNodeId == ~0) {
#endif // end of Non PLATFORM_APPLE
        //
        // Memory Affinity disabled
        //
        scene_rdl2::util::alignedFreeDtor(mMemBlockManager);
        scene_rdl2::util::alignedFreeArray(mBlockMemory);
        scene_rdl2::util::alignedFreeArray(mEntryMemory);

#ifndef PLATFORM_APPLE
    } else {
        //
        // Memory Affinity enabled
        //
        std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(mNumaNodeId);
        const scene_rdl2::NumaNode* currNumaNode = currMemNode->getNumaNode();

        scene_rdl2::util::alignedFreeDtorBasis<scene_rdl2::alloc::MemBlockManager>
            (mMemBlockManager, [&](const void* ptr) {
                currNumaNode->free(const_cast<void*>(ptr), mMemBlockManagerAllocatedSize);
            });
        scene_rdl2::util::alignedFreeArrayBasis<scene_rdl2::alloc::MemBlock>
            (mBlockMemory, [&](const void* ptr) {
                currNumaNode->free(const_cast<void*>(ptr), mBlockMemoryAllocatedSize);
            });
        scene_rdl2::util::alignedFreeArrayBasis<uint8_t>
            (mEntryMemory, [&](const void* ptr) {
                currNumaNode->free(const_cast<void*>(ptr), mEntryMemoryAllocatedSize);
            });
    }
#endif // end of Non PLATFORM_APPLE
    mMemBlockManager = nullptr;
    mBlockMemory = nullptr;
    mEntryMemory = nullptr;
}

//------------------------------------------------------------------------------------------

class PoolInfoManager
//
// This class keeps a multibank version of PoolInfo when Memory-Affinity is enabled.
// Otherwise, keep global PoolInfo for Non-Memory-Affinity situation
//
{
public:
    PoolInfoManager(const size_t poolSize,
                    const unsigned numThreads,
                    const size_t entrySize,
                    const char* const poolName);

    PoolInfo* getPoolInfo(const unsigned numaNodeId) const {
        if (numaNodeId == ~0) return mPoolInfo.get();
#ifdef STRICT_CHECK_FOR_GETPOOLINFO
        if (numaNodeId >= getNumaNodeTblSize()) return nullptr;
        if (!mNumaNodePoolInfoTbl[numaNodeId]) return nullptr;
#endif // end STRICT_CHECK_FOR_GETPOOLINFO
        return mNumaNodePoolInfoTbl[numaNodeId].get();
    }

    size_t getNumaNodeTblSize() const { return mNumaNodePoolInfoTbl.size(); }

    void fullReset();

    size_t getActualPoolSizeAll() const; // byte
    size_t getMemBlockManagerMemoryUsageAll() const; // byte

    std::string show() const;

private:
    bool mMemAffinityEnable {false};

    // for non Mem-Affinity control
    std::shared_ptr<PoolInfo> mPoolInfo;

    // for Mem-Affinity control
    // size == total NUMA node of this host, some of them might be null
    std::vector<std::shared_ptr<PoolInfo>> mNumaNodePoolInfoTbl;
};

PoolInfoManager::PoolInfoManager(const size_t poolSize,
                                 const unsigned numThreads,
                                 const size_t entrySize,
                                 const char* const poolName)
{
    auto genPoolInfoShPtrMemAffinity = [&](const unsigned numaNodeId) {
        //
        // Memory Affinity enabled
        //
        // It would be nice if we could allocate the PoolInfo object itself by NUMA-Node-Based memory
        // allocation precisely but the current implementation is not. More fine-grain NUMA-node-based
        // memory allocation would be the next future task.
        // Non-NUMA-Node-Based memory allocation for the PoolInfo object itself is not bad and has
        // reasonable performance so far.
        return std::make_shared<PoolInfo>(numaNodeId, poolSize, numThreads, entrySize, poolName);
    };
    auto genPoolInfoShPtrNonMemAffinity = [&]() {
        //
        // Memory Affinity disabled
        //
        constexpr unsigned numaNodeId = ~0;
        return std::make_shared<PoolInfo>(numaNodeId, poolSize, numThreads, entrySize, poolName);
    };

    std::shared_ptr<mcrt_common::MemoryAffinityManager> memAff = mcrt_common::AffinityManager::get()->getMem();
    mMemAffinityEnable = memAff->getMemAffinityEnable();
    if (mMemAffinityEnable) {
        //
        // Memory Affinity enabled
        //
        const std::vector<std::shared_ptr<mcrt_common::MemoryNode>>& memNodeTbl = memAff->getMemNodeTbl();

        mNumaNodePoolInfoTbl.resize(memNodeTbl.size());
        for (size_t numaNodeId = 0; numaNodeId < mNumaNodePoolInfoTbl.size(); ++numaNodeId) {
            if (memNodeTbl[numaNodeId]) {
                //
                // Active NUMA-node
                //
                mNumaNodePoolInfoTbl[numaNodeId] = genPoolInfoShPtrMemAffinity(numaNodeId);
            } else {
                //
                // Not Active NUMA-node (MoonRay does not access this NUMA-node memory with affinity logic)
                //
                mNumaNodePoolInfoTbl[numaNodeId] = nullptr;
            }
        } // numaNodeId loop
        mPoolInfo = nullptr;
    } else {
        //
        // Memory Affinity disabled
        //
        mPoolInfo = genPoolInfoShPtrNonMemAffinity();
        mNumaNodePoolInfoTbl.clear();
    }
}

void
PoolInfoManager::fullReset()
{
    if (mMemAffinityEnable) {
        //
        // Memory Affinity enabled
        //
        for (size_t i = 0; i < mNumaNodePoolInfoTbl.size(); ++i) {
            if (mNumaNodePoolInfoTbl[i]) {
                mNumaNodePoolInfoTbl[i]->fullReset();
            }
        }
    } else {
        //
        // Memory Affinity disabled
        //
        if (mPoolInfo) mPoolInfo->fullReset();
    }
}

size_t
PoolInfoManager::getActualPoolSizeAll() const
{
    size_t total = 0;
    if (mMemAffinityEnable) {
        //
        // Memory Affinity enabled
        //
        for (size_t i = 0; i < mNumaNodePoolInfoTbl.size(); ++i) {
            if (mNumaNodePoolInfoTbl[i]) {
                total += mNumaNodePoolInfoTbl[i]->getActualPoolSize();
            }
        }
    } else {
        //
        // Memory Affinity disabled
        //
        if (mPoolInfo) total += mPoolInfo->getActualPoolSize();
    }
    return total;
}

size_t
PoolInfoManager::getMemBlockManagerMemoryUsageAll() const
{
    size_t total = 0;
    if (mMemAffinityEnable) {
        //
        // Memory Affinity enabled
        //
        for (size_t i = 0; i < mNumaNodePoolInfoTbl.size(); ++i) {
            if (mNumaNodePoolInfoTbl[i]) {
                total += mNumaNodePoolInfoTbl[i]->getMemBlockManagerMemoryUsage();
            }
        }
    } else {
        //
        // Memory Affinity disabled
        //
        if (mPoolInfo) total += mPoolInfo->getMemBlockManagerMemoryUsage();
    }
    return total;
}

std::string
PoolInfoManager::show() const
{
    auto showPoolInfoMain = [](std::shared_ptr<PoolInfo> info) -> std::string {
        if (!info) return "empty";
        return info->show();
    };
    auto showPoolInfo = [&]() {
        std::ostringstream ostr;
        ostr << "mPoolInfo:" << showPoolInfoMain(mPoolInfo);
        return ostr.str();
    };
    auto showPoolInfoTbl = [&]() -> std::string {
        if (!getNumaNodeTblSize()) return "mNumaNodePoolInfoTbl is empty";
        std::ostringstream ostr;
        ostr << "mNumaNodePoolInfoTbl (size:" << getNumaNodeTblSize() << ") {\n";
        for (size_t i = 0; i < getNumaNodeTblSize(); ++i) {
            std::ostringstream ostr2;
            ostr2 << "numaNodeId:" << i << ' ' << showPoolInfoMain(mNumaNodePoolInfoTbl[i]);
            ostr << scene_rdl2::str_util::addIndent(ostr2.str()) << '\n';
        }
        ostr << "}";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "PoolInfoManager {\n"
         << "  mMemAffinityEnable:" << scene_rdl2::str_util::boolStr(mMemAffinityEnable) << '\n'
         << "  Non-MEM-Affinity {\n"
         << scene_rdl2::str_util::addIndent(showPoolInfo(), 2) << '\n'
         << "  }\n"
         << "  MEM-Affinity {\n"
         << scene_rdl2::str_util::addIndent(showPoolInfoTbl(), 2) << '\n'
         << "  }\n"
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

class PoolData
//
// This class keeps the Multi-Bank version of the RayState pool and CL1 pool if memory affinity is ON.
// Otherwise, keeps the single global memory version of the RayState pool and CL1 pool for all TLS.
// Also maintains pool memory related info.
//
{
public:
    PoolData() {}

    void init(const mcrt_common::TLSInitParams& initParams);
    void cleanUp();
    bool fullReset();

    bool isValidRayStatePoolInfoManager() const { return mRayStatePoolInfoManager.get() != nullptr; }
    bool isValidCL1PoolInfoManager() const { return mCL1PoolInfoManager.get() != nullptr; }
    
    PoolInfo* getRayState(const unsigned numaNodeId)
    {
        return mRayStatePoolInfoManager->getPoolInfo(numaNodeId);
    }

    PoolInfo* getCL1(const unsigned numaNodeId)
    {
        return mCL1PoolInfoManager->getPoolInfo(numaNodeId);
    }

    size_t getRayStateActualPoolSizeAll() const; // byte
    size_t getCL1ActualPoolSizeAll() const; // byte

    size_t getRayStateMemBlockManagerMemoryUsageAll() const; // byte
    size_t getCL1MemBlockManagerMemoryUsageAll() const; // byte

    // for memory-free operations when memory affinity enabled
    unsigned getRayQueueSize() const { return mRayQueueSize; } // number of entries
    unsigned getOcclusionQueueSize() const { return mOcclusionQueueSize; } // number of entries
    unsigned getPresenceShadowsQueueSize() const { return mPresenceShadowsQueueSize; } // number of entries
    unsigned getRadianceQueueSize() const { return mRadianceQueueSize; } // number of entries
    unsigned getAovQueueSize() const { return mAovQueueSize; } // number of entries
    unsigned getHeatMapQueueSize() const { return mHeatMapQueueSize; } // number of entries

    std::string show() const;

private:
    std::mutex mInitMutex;

    unsigned mRefCount {0};
    const mcrt_common::TLSInitParams* mInitParams {nullptr};

    std::shared_ptr<PoolInfoManager> mRayStatePoolInfoManager;
    std::shared_ptr<PoolInfoManager> mCL1PoolInfoManager;

    // for memory-free operations when memory affinity enabled
    unsigned mRayQueueSize {0}; // number of entries
    unsigned mOcclusionQueueSize {0}; // number of entries
    unsigned mPresenceShadowsQueueSize {0}; // number of entries
    unsigned mRadianceQueueSize {0}; // number of entries
    unsigned mAovQueueSize {0}; // number of entries
    unsigned mHeatMapQueueSize {0}; // number of entries
};

void
PoolData::init(const mcrt_common::TLSInitParams& initParams)
{
    // Protect against races the very first time we initialize gPoolData object
    std::lock_guard<std::mutex> lock(mInitMutex);

    if (mRefCount == 0) {
        //
        // Allocate pooled memory:
        //
        if (initParams.mPerThreadRayStatePoolSize) {
            mRayStatePoolInfoManager =
                std::make_shared<PoolInfoManager>(static_cast<size_t>(initParams.mPerThreadRayStatePoolSize),
                                                  initParams.mDesiredNumTBBThreads,
                                                  sizeof(RayState),
                                                  "RayState");
        }
        if (initParams.mPerThreadCL1PoolSize) {
            mCL1PoolInfoManager =
                std::make_shared<PoolInfoManager>(static_cast<size_t>(initParams.mPerThreadCL1PoolSize),
                                                  initParams.mDesiredNumTBBThreads,
                                                  sizeof(TLState::CacheLine1),
                                                  "CL1");
        }

        // for memory-free operations when memory affinity enabled
        mRayQueueSize = initParams.mRayQueueSize;
        mOcclusionQueueSize = initParams.mOcclusionQueueSize;
        mPresenceShadowsQueueSize = initParams.mPresenceShadowsQueueSize;
        mRadianceQueueSize = initParams.mRadianceQueueSize;
        mAovQueueSize = initParams.mAovQueueSize;
        mHeatMapQueueSize = initParams.mHeatMapQueueSize;

        // initParams is owned by the top level ThreadLocalState object so we know
        // it persists whilst this TLState persists.
        mInitParams = &initParams;

#ifdef DEBUG_MSG_PRIVATE_INIT
        std::cerr << ">> PbrTLState.cc PoolData::init() " << show() << '\n';
#endif // end DEBUG_MSG_PRIVATE_INIT
    }
    ++mRefCount;
}

void
PoolData::cleanUp()
{
    MNRY_ASSERT(mRefCount);

    // Protect against races the during gPoolData clean up.
    std::lock_guard<std::mutex> lock(mInitMutex);

    --mRefCount;
    if (mRefCount == 0) {
        MNRY_ASSERT(mRefCount == 0);
        MNRY_ASSERT(mInitParams);

        mInitParams = nullptr;
        mRayStatePoolInfoManager.reset();
        mCL1PoolInfoManager.reset();
    }
}

bool
PoolData::fullReset()
{
    if (!mRefCount) return false;

    mRayStatePoolInfoManager->fullReset();
    mCL1PoolInfoManager->fullReset();
    return true;
}

size_t
PoolData::getRayStateActualPoolSizeAll() const
{
    if (!mRayStatePoolInfoManager) return 0;
    return mRayStatePoolInfoManager->getActualPoolSizeAll(); // byte
}

size_t
PoolData::getCL1ActualPoolSizeAll() const
{
    if (!mCL1PoolInfoManager) return 0;
    return mCL1PoolInfoManager->getActualPoolSizeAll(); // byte
}

size_t
PoolData::getRayStateMemBlockManagerMemoryUsageAll() const
{
    if (!mRayStatePoolInfoManager) return 0;
    return mRayStatePoolInfoManager->getMemBlockManagerMemoryUsageAll(); // byte
}

size_t
PoolData::getCL1MemBlockManagerMemoryUsageAll() const
{
    if (!mCL1PoolInfoManager) return 0;
    return mCL1PoolInfoManager->getMemBlockManagerMemoryUsageAll(); // byte
}

std::string
PoolData::show() const
{
    auto showRayState = [&]() -> std::string {
        if (!mRayStatePoolInfoManager) return "mRayStatePoolInfoManager:empty";
        std::ostringstream ostr;
        ostr << "mRayStatePoolInfoManager {\n"
             << scene_rdl2::str_util::addIndent(mRayStatePoolInfoManager->show()) << '\n'
             << "}";
        return ostr.str();
    };
    auto showCL1 = [&]() -> std::string {
        if (!mCL1PoolInfoManager) return "mCL1PoolInfoManager:empty";
        std::ostringstream ostr;
        ostr << "mCL1PoolInfoManager {\n"
             << scene_rdl2::str_util::addIndent(mCL1PoolInfoManager->show()) << '\n'
             << "}";
        return ostr.str();
    };
    auto showQueueSizeInfo = [&]() {
        std::stringstream ostr;
        ostr << "queueSizeInfo {\n"
             << "  mRayQueueSize:" << mRayQueueSize << '\n'
             << "  mOcclusionQueueSize:" << mOcclusionQueueSize << '\n'
             << "  mPresenceShadowsQueueSize:" << mPresenceShadowsQueueSize << '\n'
             << "  mRadianceQueueSize:" << mRadianceQueueSize << '\n'
             << "  mAovQueueSize:" << mAovQueueSize << '\n'
             << "  mHeatMapQueueSize:" << mHeatMapQueueSize << '\n'
             << "}";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "moonray::pbr::PoolData {\n"
         << "  mRefCount:" << mRefCount << '\n'
         << "  mInitParams:0x" << std::hex << reinterpret_cast<uintptr_t>(mInitParams) << std::dec << '\n'
         << scene_rdl2::str_util::addIndent(showRayState()) << '\n'
         << scene_rdl2::str_util::addIndent(showCL1()) << '\n'
         << scene_rdl2::str_util::addIndent(showQueueSizeInfo()) << '\n'
         << "}";
    return ostr.str();
}

PoolData gPoolData;

//------------------------------------------------------------------------------------------

template <typename T>
void
inline setQueueSize(T *queue, const float t)
{
    MNRY_ASSERT(queue);

    const unsigned maxEntries = queue->getMaxEntries();

    // Only mess with queue sizes if we have any queue elements allocated in the
    // first place.
    if (maxEntries > 0) {

        unsigned size = std::max(unsigned(float(maxEntries) * t), 1u);

        // Round up to VLEN.
        size = (size + VLEN) & ~VLEN;

        queue->setQueueSize(std::min(size, maxEntries));
    }
}

} // End of anonymous namespace.

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

void
resetPools()
{
    if (!gPoolData.fullReset()) {
        return;
    }

    forEachTLS([&](TLState *tls) {
        tls->mRayStatePool.fastReset();   // The call to fastReset is deliberate.
        tls->mCL1Pool.fastReset();
    });

#ifdef DEBUG_RECORD_PEAK_RAYSTATE_USAGE
    static unsigned prevPeakRayStateUsage = 0;
    if (gPeakRayStateUsage > prevPeakRayStateUsage) {
        scene_rdl2::logging::Logger::info("\nPeak RayState usage = ", gPeakRayStateUsage,
                                          ", (", gPoolData.getRayStateActualPoolSizeAll(), " allocated)\n");
        MOONRAY_THREADSAFE_STATIC_WRITE(prevPeakRayStateUsage = gPeakRayStateUsage);
    }
#endif

#ifdef DEBUG_RECORD_PEAK_CL1_USAGE
    static unsigned prevPeakCL1Usage = 0;
    if (gPeakCL1Usage > prevPeakCL1Usage) {
        scene_rdl2::logging::Logger::info("\nPeak CL1 pool usage = ", gPeakCL1Usage,
                                          ", (", gPoolData.getCL1ActualPoolSizeAll(), " allocated)\n");
        MOONRAY_THREADSAFE_STATIC_WRITE(prevPeakCL1Usage = gPeakCL1Usage);
    }
#endif
}

TLState::TLState(mcrt_common::ThreadLocalState *tls,
                 const mcrt_common::TLSInitParams &initParams,
                 bool okToAllocBundledResources) :
    BaseTLState(tls->mThreadIdx, tls->mArena, tls->mPixelArena),
    mTopLevelTls(tls),
    mRadianceQueue(nullptr),
    mAovQueue(nullptr),
    mHeatMapQueue(nullptr),
    mXPUOcclusionRayQueue(nullptr),
    mXPURayQueue(nullptr),
    mFs(nullptr),
    mCancellationState(DISABLED),
    mCurrentPassIdx(0),
    mRayRecorder(nullptr),
    mRayEntries(nullptr),
    mOcclusionEntries(nullptr),
    mPresenceShadowsEntries(nullptr),
    mRadianceEntries(nullptr),
    mAovEntries(nullptr),
    mHeatMapEntries(nullptr)
{
    const unsigned numaNodeId = tls->mArena.getNumaNodeId();

#ifndef PLATFORM_APPLE
    if (numaNodeId == ~0) {
#endif // end of Non PLATFORM_APPLE
        //
        // Memory Affinity disabled
        //
        mExclusiveAccumulators = scene_rdl2::util::alignedMallocCtor<ExclusiveAccumulators>(CACHE_LINE_SIZE);

#ifndef PLATFORM_APPLE
    } else {
        //
        // Memory Affinity enabled
        //
        std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(numaNodeId);
        const scene_rdl2::NumaNode* currNumaNode = currMemNode->getNumaNode();
        if (!currNumaNode->alignmentSizeCheck(CACHE_LINE_SIZE)) {
            std::ostringstream ostr;
            ostr << "FATAL-ERROR : PbrTLState.cc TLState::TLState() alignmentSizeCheck failed."
                 << " requested-alignment-size:" << CACHE_LINE_SIZE << '\n';
            scene_rdl2::logging::Logger::fatal(ostr.str());
            exit(1);
        }

        // callback procedure for memory allocation, alignment size check have been done already.
        auto numaNodeAlloc = [&](size_t size, size_t alignment) -> void* { return currNumaNode->alloc(size); };

        mExclusiveAccumulators = 
            scene_rdl2::util::alignedMallocCtorBasis<ExclusiveAccumulators>(CACHE_LINE_SIZE, numaNodeAlloc);
    }
#endif // end of Non PLATFORM_APPLE

    if (okToAllocBundledResources) {
        const unsigned numaNodeId = tls->mArena.getNumaNodeId();

        if (gPoolData.isValidRayStatePoolInfoManager() &&
            gPoolData.getRayState(numaNodeId)->getMemBlockManager()) {
            mRayStatePool.init(gPoolData.getRayState(numaNodeId)->getMemBlockManager());            
        }
        if (gPoolData.isValidCL1PoolInfoManager() &&
            gPoolData.getCL1(numaNodeId)->getMemBlockManager()) {
            mCL1Pool.init(gPoolData.getCL1(numaNodeId)->getMemBlockManager());
        }

#ifndef PLATFORM_APPLE
        const scene_rdl2::NumaNode* currNumaNode = nullptr;
        if (numaNodeId != ~0) {
            //
            // Memory Affinity enabled
            //
            std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
                mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(numaNodeId);
            currNumaNode = currMemNode->getNumaNode();

            // We should check the alignment size here. We have limited support for alignment size because
            // we use the simplified high speed operation of NUMA-node based memory allocation.
            // See more detail on comment of scene_rdl2/lib/render/util/NumaInfo.h alignmentSizeCheck()
            if (!currNumaNode->alignmentSizeCheck(CACHE_LINE_SIZE)) {
                std::ostringstream ostr;
                ostr << "FATAL-ERROR : PbrTLState.cc TLState::TLState() alignmentSizeCheck failed."
                     << " requested-alignment-size:" << CACHE_LINE_SIZE << '\n';
                scene_rdl2::logging::Logger::fatal(ostr.str());
                exit(1);
            }
        }
#endif // end of Non PLATFORM_APPLE

        // callback procedure for memory allocation, alignment size check have been done already
        // if Mem-affinity is on.
#ifdef PLATFORM_APPLE
        auto memAllocCallBack = [&](size_t size, size_t alignment) -> void* {
            return scene_rdl2::util::alignedMalloc(size, alignment); // Memory Affinity disabled
        };
#else // else PLATFORM_APPLE
        auto memAllocCallBack = [&](size_t size, size_t alignment) -> void* {
            if (!currNumaNode) return scene_rdl2::util::alignedMalloc(size, alignment); // Memory Affinity disabled
            else return currNumaNode->alloc(size); // Memory Affinity enabled
        };
#endif // end of Non PLATFORM_APPLE
        
        // Allocate ray queue (contains RayState* pointers).
        if (initParams.mRayQueueSize) {
            const unsigned queueSize = initParams.mRayQueueSize;
            mRayEntries =
                scene_rdl2::util::alignedMallocArrayBasis<RayQueue::EntryType>
                (queueSize, CACHE_LINE_SIZE, memAllocCallBack);
            mRayQueue.init(queueSize, mRayEntries);
            uint32_t rayHandlerFlags = 0;
            mRayQueue.setHandler(rayBundleHandler, (void *)((uint64_t)rayHandlerFlags));
        }

        // Allocate occlusion queue.
        if (initParams.mOcclusionQueueSize) {
            const unsigned queueSize = initParams.mOcclusionQueueSize;
            mOcclusionEntries =
                scene_rdl2::util::alignedMallocArrayBasis<OcclusionQueue::EntryType>
                (queueSize, CACHE_LINE_SIZE, memAllocCallBack);
            mOcclusionQueue.init(queueSize, mOcclusionEntries);
            uint32_t rayHandlerFlags = 0;
            mOcclusionQueue.setHandler(occlusionQueryBundleHandler, (void *)((uint64_t)rayHandlerFlags));
        }

        // Allocate presence shadows queue.
        if (initParams.mPresenceShadowsQueueSize) {
            const unsigned queueSize = initParams.mPresenceShadowsQueueSize;
            mPresenceShadowsEntries =
                scene_rdl2::util::alignedMallocArrayBasis<PresenceShadowsQueue::EntryType>
                (queueSize, CACHE_LINE_SIZE, memAllocCallBack);
            mPresenceShadowsQueue.init(queueSize, mPresenceShadowsEntries);
            // don't need to set handler flags because presence shadows code only does scalar
            //  ray tracing
            mPresenceShadowsQueue.setHandler(presenceShadowsQueryBundleHandler, nullptr);
        }

        // Allocate radiance queue.
        if (initParams.mRadianceQueueSize) {
            mRadianceQueue =
                scene_rdl2::util::alignedMallocCtorBasis<RadianceQueue>
                (CACHE_LINE_SIZE, memAllocCallBack);
            const unsigned queueSize = initParams.mRadianceQueueSize;
            mRadianceEntries =
                scene_rdl2::util::alignedMallocArrayBasis<RadianceQueue::EntryType>
                (queueSize, CACHE_LINE_SIZE, memAllocCallBack);
            mRadianceQueue->init(queueSize, mRadianceEntries);
            // Radiance queue handler is setup by the RenderDriver.
        }

        // Allocate aov queue
        if (initParams.mAovQueueSize) {
            mAovQueue =
                scene_rdl2::util::alignedMallocCtorBasis<AovQueue>
                (CACHE_LINE_SIZE, memAllocCallBack);
            const unsigned queueSize = initParams.mAovQueueSize;
            mAovEntries =
                scene_rdl2::util::alignedMallocArrayBasis<AovQueue::EntryType>
                (queueSize, CACHE_LINE_SIZE, memAllocCallBack);
            mAovQueue->init(queueSize, mAovEntries);
            // Aov queue handler is setup by the RenderDriver.
        }

        // Allocate heat map queue
        if (initParams.mHeatMapQueueSize) {
            mHeatMapQueue =
                scene_rdl2::util::alignedMallocCtorBasis<HeatMapQueue>
                (CACHE_LINE_SIZE, memAllocCallBack);
            const unsigned queueSize = initParams.mHeatMapQueueSize;
            mHeatMapEntries =
                scene_rdl2::util::alignedMallocArrayBasis<HeatMapQueue::EntryType>
                (queueSize, CACHE_LINE_SIZE, memAllocCallBack);
            mHeatMapQueue->init(queueSize, mHeatMapEntries);
            // HeatMap queue handler is setup by the RenderDriver.
        }
    }

    std::fill(mPrimaryRaysSubmitted, mPrimaryRaysSubmitted + MAX_RENDER_PASSES, 0);

    mRayRecorder = new DebugRayRecorder(tls->mThreadIdx);
    mRayVertexStack.reserve(32);
}

TLState::~TLState()
{
    delete mRayRecorder;

    const unsigned numaNodeId = mTopLevelTls->mArena.getNumaNodeId();
 
#ifndef PLATFORM_APPLE
    if (numaNodeId == ~0) {
#endif // end of Non PLATFORM_APPLE
        //
        // Memory Affinity disabled
        //
        scene_rdl2::util::alignedFreeDtor(mRadianceQueue);
        scene_rdl2::util::alignedFreeDtor(mAovQueue);
        scene_rdl2::util::alignedFreeDtor(mHeatMapQueue);

        scene_rdl2::util::alignedFreeArray(mRadianceEntries);
        scene_rdl2::util::alignedFreeArray(mAovEntries);
        scene_rdl2::util::alignedFreeArray(mOcclusionEntries);
        scene_rdl2::util::alignedFreeArray(mPresenceShadowsEntries);
        scene_rdl2::util::alignedFreeArray(mRayEntries);
        scene_rdl2::util::alignedFreeArray(mHeatMapEntries);

        scene_rdl2::util::alignedFreeDtor(mExclusiveAccumulators);

#ifndef PLATFORM_APPLE
    } else {
        //
        // Memory Affinity enabled
        //
        std::shared_ptr<mcrt_common::MemoryNode> currMemNode =
            mcrt_common::AffinityManager::get()->getMem()->getMemoryNodeByNumaNodeId(numaNodeId);
        const scene_rdl2::NumaNode* currNumaNode = currMemNode->getNumaNode();

        scene_rdl2::util::alignedFreeDtorBasis<RadianceQueue>
            (mRadianceQueue,
             [&](const void* ptr) {
                 currNumaNode->free(const_cast<void*>(ptr), sizeof(RadianceQueue));
             });
        scene_rdl2::util::alignedFreeDtorBasis<AovQueue>
            (mAovQueue,
             [&](const void* ptr) {
                 currNumaNode->free(const_cast<void*>(ptr), sizeof(AovQueue));
             });
        scene_rdl2::util::alignedFreeDtorBasis<HeatMapQueue>
            (mHeatMapQueue,
             [&](const void* ptr) {
                 currNumaNode->free(const_cast<void*>(ptr), sizeof(HeatMapQueue));
             });

        scene_rdl2::util::alignedFreeArrayBasis<RadianceQueue::EntryType>
            (mRadianceEntries,
             [&](const void* ptr) {
                 currNumaNode->free(const_cast<void*>(ptr),
                                    gPoolData.getRadianceQueueSize() * sizeof(RadianceQueue::EntryType));
             });
        scene_rdl2::util::alignedFreeArrayBasis<AovQueue::EntryType>
            (mAovEntries,
             [&](const void*ptr) {
                 currNumaNode->free(const_cast<void*>(ptr),
                                    gPoolData.getAovQueueSize() * sizeof(AovQueue::EntryType));
             });
        scene_rdl2::util::alignedFreeArrayBasis<OcclusionQueue::EntryType>
            (mOcclusionEntries,
             [&](const void*ptr) {
                 currNumaNode->free(const_cast<void*>(ptr),
                                    gPoolData.getOcclusionQueueSize() * sizeof(OcclusionQueue::EntryType));
             });
        scene_rdl2::util::alignedFreeArrayBasis<PresenceShadowsQueue::EntryType>
            (mPresenceShadowsEntries,
             [&](const void*ptr) {
                 currNumaNode->free(const_cast<void*>(ptr),
                                    gPoolData.getPresenceShadowsQueueSize() * sizeof(PresenceShadowsQueue::EntryType));
             });
        scene_rdl2::util::alignedFreeArrayBasis<RayQueue::EntryType>
            (mRayEntries,
             [&](const void*ptr) {
                 currNumaNode->free(const_cast<void*>(ptr),
                                    gPoolData.getRayQueueSize() * sizeof(RayQueue::EntryType));
             }); 
       scene_rdl2::util::alignedFreeArrayBasis<HeatMapQueue::EntryType>
            (mHeatMapEntries,
             [&](const void*ptr) {
                 currNumaNode->free(const_cast<void*>(ptr),
                                    gPoolData.getHeatMapQueueSize() * sizeof(HeatMapQueue::EntryType));
             });
       
       scene_rdl2::util::alignedFreeDtorBasis<ExclusiveAccumulators>
           (mExclusiveAccumulators,
            [&](const void* ptr) {
                currNumaNode->free(const_cast<void*>(ptr), sizeof(ExclusiveAccumulators));
            });
    }
#endif // end of Non PLATFORM_APPLE

    mRayStatePool.cleanUp();
    mCL1Pool.cleanUp();

    gPoolData.cleanUp();
}

void
TLState::reset()
{
    mRayQueue.reset();
    mOcclusionQueue.reset();
    mPresenceShadowsQueue.reset();

    if (mRadianceQueue) {
        mRadianceQueue->reset();
    }
    if (mAovQueue) {
        mAovQueue->reset();
    }
    if (mHeatMapQueue) {
        mHeatMapQueue->reset();
    }

    setAllQueueSizes(1.f);

    std::fill(mPrimaryRaysSubmitted, mPrimaryRaysSubmitted + MAX_RENDER_PASSES, 0);

    mFs = nullptr;

    mTilesRenderedTo.clearAll();

    mCancellationState = DISABLED;
    mCurrentPassIdx = 0;

    // From class BaseTLState.
    mIspcAccumulator = nullptr;

    MOONRAY_THREADSAFE_STATIC_WRITE(gFailedRayStateAllocs = 0);
    MOONRAY_THREADSAFE_STATIC_WRITE(gFailedCL1Allocs = 0);
}

// static function
size_t
TLState::getCL1PoolSize()
//
// return all memBlockManager memory usage for CL1
//
{
    return gPoolData.getCL1MemBlockManagerMemoryUsageAll(); // byte
}

RayState **
TLState::allocRayStates(unsigned numRayStates)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_RAYSTATE_ALLOCS);

    MNRY_ASSERT(numRayStates);

    RayState **rayStates = mArena->allocArray<RayState *>(numRayStates);

    poolAlloc<RayState, RayStatePool>("RayState", mRayStatePool, numRayStates,
                                      rayStates, ACCUM_RAYSTATE_STALLS,
                                      gFailedRayStateAllocs);

#ifdef DEBUG_RECORD_PEAK_RAYSTATE_USAGE
    // Slowdown here - turn off unless doing memory profiles.
    unsigned numRayStatesAllocated = 0;
    forEachTLS([&](pbr::TLState *tls) {
        numRayStatesAllocated += tls->mRayStatePool.getNumEntriesAllocated();
    });

    if (numRayStatesAllocated > gPeakRayStateUsage) {
        MOONRAY_THREADSAFE_STATIC_WRITE(gPeakRayStateUsage = numRayStatesAllocated);
    }
#endif

    return rayStates;
}

void
TLState::freeRayStates(unsigned numRayStates, RayState **rayStates)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_RAYSTATE_ALLOCS);

    for (size_t i = 0; i < numRayStates; i++) {
        RayState *rs = rayStates[i];
        if (rs->mDeepDataHandle != nullHandle) {
            releaseDeepData(rs->mDeepDataHandle);
        }
        if (rs->mCryptomatteDataHandle != nullHandle) {
            releaseCryptomatteData(rs->mCryptomatteDataHandle);
        }
    }

    mRayStatePool.freeList(numRayStates, rayStates);
}

// static function
size_t
TLState::getRayStatePoolSize()
//
// return all memBlockManager memory usage for RayState
//
{
    return gPoolData.getRayStateMemBlockManagerMemoryUsageAll(); // byte
}

uint32_t
TLState::acquireDeepData(uint32_t deepDataHandle)
{
    if (deepDataHandle != pbr::nullHandle) {
        pbr::DeepData *deepData = static_cast<pbr::DeepData*>(getListItem(deepDataHandle, 0));
        ++(deepData->mRefCount);
    }
    return deepDataHandle;
}

void
TLState::releaseDeepData(uint32_t deepDataHandle)
{
    if (deepDataHandle != pbr::nullHandle) {
        pbr::DeepData *deepData = static_cast<pbr::DeepData*>(getListItem(deepDataHandle, 0));
        MNRY_ASSERT(deepData->mRefCount > 0);
        if (--(deepData->mRefCount) == 0) {
            freeList(deepDataHandle);
        }
    }
}

uint32_t
TLState::acquireCryptomatteData(uint32_t cryptomatteDataHandle)
{
    if (cryptomatteDataHandle != pbr::nullHandle) {
        pbr::CryptomatteData *cryptomatteData =
            static_cast<pbr::CryptomatteData*>(getListItem(cryptomatteDataHandle, 0));
        ++(cryptomatteData->mRefCount);
    }
    return cryptomatteDataHandle;
}

void
TLState::releaseCryptomatteData(uint32_t cryptomatteDataHandle)
{
    if (cryptomatteDataHandle != pbr::nullHandle) {
        pbr::CryptomatteData *cryptomatteData =
            static_cast<pbr::CryptomatteData*>(getListItem(cryptomatteDataHandle, 0));
        MNRY_ASSERT(cryptomatteData->mRefCount > 0);
        if (--(cryptomatteData->mRefCount) == 0) {
            freeList(cryptomatteDataHandle);
        }
    }
}

uint32_t
TLState::allocList(unsigned itemSize, unsigned numItems)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_TLS_ALLOCS);

    // we need a 4 byte index for every item in the list
    MNRY_ASSERT(numItems <= ALLOC_LIST_MAX_NUM_ITEMS);
    if (numItems > ALLOC_LIST_MAX_NUM_ITEMS) {
        return nullHandle; // fail, should never happen
    }

    // itemSize must be within our max allocation
    MNRY_ASSERT(itemSize <= ALLOC_LIST_MAX_ITEM_SIZE);
    if (itemSize > ALLOC_LIST_MAX_ITEM_SIZE) {
        return nullHandle; // fail, should never happen
    }

    uint32_t *header = nullptr;

    if (numItems == 1) {
        // special case a list of 1 item.  there is no need to allocate
        // a header and an item.
        CacheLine1 *results[1];
        poolAlloc<CacheLine1, CL1Pool>("CacheLine1 (1 item)", mCL1Pool, numItems,
                                       results, ACCUM_CL1_ALLOC_STALLS, gFailedCL1Allocs);
        header = reinterpret_cast<uint32_t *>(results[0]);
    } else {
        // allocate numItems + 1 (for the header)
        CacheLine1 *results[ALLOC_LIST_MAX_NUM_ITEMS + 1];
        poolAlloc<CacheLine1, CL1Pool>("CacheLine1", mCL1Pool, numItems + 1,
                                       results, ACCUM_CL1_ALLOC_STALLS, gFailedCL1Allocs);

        // initialize the header
        header = reinterpret_cast<uint32_t *>(results[0]);
        for (unsigned i = 0; i < numItems; ++i) {
            header[i] = cl1PtrToHandle(results[i + 1], /* numItems = */ 1);
        }

        // set a nullIndex to indicate end of list, if we used less
        // than a full cache line
        const int HEADER_ENTRIES_PER_LINE = CACHE_LINE_SIZE / sizeof(uint32_t);
        if (numItems % HEADER_ENTRIES_PER_LINE) {
            header[numItems] = nullHandle;
        }
    }

#ifdef DEBUG_RECORD_PEAK_CL1_USAGE
    // Slowdown here - turn off unless doing memory profiles.
    unsigned numAllocated = 0;
    forEachTLS([&](pbr::TLState *tls) {
            numAllocated += tls->mCL1Pool.getNumEntriesAllocated();
        });

    if (numAllocated > gPeakCL1Usage) {
        MOONRAY_THREADSAFE_STATIC_WRITE(gPeakCL1Usage = numAllocated);
    }
#endif

    // return pointer to list header, as an index
    return cl1PtrToHandle(header, numItems);
}

void
TLState::freeList(uint32_t listHandle)
{
    EXCL_ACCUMULATOR_PROFILE(this, EXCL_ACCUM_TLS_ALLOCS);

    MNRY_ASSERT(listHandle != nullHandle);

    CacheLine1 *results[ALLOC_LIST_MAX_NUM_ITEMS + 1];
    results[0] = static_cast<CacheLine1 *>(handleToCL1Ptr(listHandle));

    MNRY_ASSERT(isValidCL1Addr(results[0], reinterpret_cast<CacheLine1*>(mCL1Pool.getEntryMemory())));

    const unsigned numItems = getNumListItems(listHandle);
    unsigned numResultsToFree = numItems;

    if (numItems > 1) {
        // we are an actual list, fill out list pointers
        uint32_t *header = reinterpret_cast<uint32_t *>(results[0]);

        MNRY_ASSERT(numItems <= ALLOC_LIST_MAX_NUM_ITEMS);

        for (unsigned i = 0; i < numItems; ++i) {
            results[i + 1] = static_cast<CacheLine1 *>(handleToCL1Ptr(header[i]));
            MNRY_ASSERT(isValidCL1Addr(results[i + 1],
                                       reinterpret_cast<CacheLine1*>(mCL1Pool.getEntryMemory())));
        }

        numResultsToFree = numItems + 1;
    }

    mCL1Pool.freeList(numResultsToFree, results);
}

unsigned
TLState::getNumListItems(uint32_t listHandle)
{
    // list length is stored in the info bits
    return ((listHandle & ALLOC_LIST_INFO_BITS) >> ALLOC_LIST_INFO_BIT_SHIFT) + 1;
}

void *
TLState::getListItem(uint32_t listHandle, unsigned item)
{
    const unsigned numListItems = getNumListItems(listHandle);
    MNRY_ASSERT(item < numListItems);

    // special case of a single item list has no header
    if (numListItems == 1) {
        return handleToCL1Ptr(listHandle);
    }

    uint32_t *header = static_cast<uint32_t *>(handleToCL1Ptr(listHandle));
    return handleToCL1Ptr(header[item]);
}

bool
TLState::verifyNoOutstandingAllocs()
{
    MNRY_ASSERT(mRayStatePool.verifyNoOutstandingAllocs());
    MNRY_ASSERT(mCL1Pool.verifyNoOutstandingAllocs());

    return true;
}

void
TLState::addRayQueueEntries(unsigned numEntries, RayState **entries)
{
    if (!numEntries) {
        return;
    }
    if (MNRY_VERIFY(mFs)->mExecutionMode == mcrt_common::ExecutionMode::XPU) {
        mXPURayQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
    } else {
        mRayQueue.addEntries(mTopLevelTls, numEntries, entries, mArena);
    }
}

void
TLState::addRadianceQueueEntries(unsigned numEntries, BundledRadiance *entries)
{
    if (!numEntries) {
        return;
    }
    mRadianceQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::addAovQueueEntries(unsigned numEntries, BundledAov *entries)
{
    if (!numEntries) {
        return;
    }
    mAovQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::addOcclusionQueueEntries(unsigned numEntries, BundledOcclRay *entries)
{
    if (!numEntries) {
        return;
    }
    if (MNRY_VERIFY(mFs)->mExecutionMode == mcrt_common::ExecutionMode::XPU) {
        mXPUOcclusionRayQueue->addEntries(mTopLevelTls, numEntries, entries);
    } else {
        mOcclusionQueue.addEntries(mTopLevelTls, numEntries, entries, mArena);
    }
}

void
TLState::addPresenceShadowsQueueEntries(unsigned numEntries, BundledOcclRay *entries)
{
    if (!numEntries) {
        return;
    }
    mPresenceShadowsQueue.addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::addHeatMapQueueEntries(unsigned numEntries, BundledHeatMapSample *entries)
{
    if (!numEntries) {
        return;
    }
    mHeatMapQueue->addEntries(mTopLevelTls, numEntries, entries, mArena);
}

void
TLState::setXPUOcclusionRayQueue(XPUOcclusionRayQueue* queue)
{
    mXPUOcclusionRayQueue = queue;
}

void
TLState::setXPURayQueue(XPURayQueue* queue)
{
    mXPURayQueue = queue;
}

void
TLState::flushRadianceQueue()
{
     mRadianceQueue->flush(mTopLevelTls, mArena);
}

unsigned
TLState::flushLocalQueues()
{
    if (MNRY_VERIFY(mFs)->mExecutionMode == mcrt_common::ExecutionMode::SCALAR) {
        return 0;
    }

    unsigned processed = 0;

    processed += mRayQueue.flush(mTopLevelTls, mArena);
    if (isCanceled()) {
        return 0;
    }

    processed += mOcclusionQueue.flush(mTopLevelTls, mArena);
    if (isCanceled()) {
        return 0;
    }

    processed += mPresenceShadowsQueue.flush(mTopLevelTls, mArena);
    if (isCanceled()) {
        return 0;
    }

    processed += mRadianceQueue->flush(mTopLevelTls, mArena);
    processed += mAovQueue->flush(mTopLevelTls, mArena);
    processed += mHeatMapQueue->flush(mTopLevelTls, mArena);

    if (isCanceled()) {
        return 0;
    }

    return processed;
}

bool
TLState::areAllLocalQueuesEmpty()
{
    if (!mRayQueue.isEmpty()) {
        return false;
    }

    if (!mOcclusionQueue.isEmpty()) {
        return false;
    }

    if (!mPresenceShadowsQueue.isEmpty()) {
        return false;
    }

    if (!mRadianceQueue->isEmpty()) {
        return false;
    }
    if (!mAovQueue->isEmpty()) {
        return false;
    }
    if (!mHeatMapQueue->isEmpty()) {
        return false;
    }

    return true;
}

void
TLState::setAllQueueSizes(float t)
{
    MNRY_ASSERT(t >= 0.f && t <= 1.f);

    setQueueSize(&mRayQueue, t);
    setQueueSize(&mOcclusionQueue, t);
    setQueueSize(&mPresenceShadowsQueue, t);

    if (mRadianceQueue) {
        setQueueSize(mRadianceQueue, t);
    }
    if (mAovQueue) {
        setQueueSize(mAovQueue, t);
    }
    if (mHeatMapQueue) {
        setQueueSize(mHeatMapQueue, t);
    }
}

void
TLState::enableCancellation(bool waitUntilReadyForDisplay)
{
    MNRY_ASSERT(mFs);

    // To ignore the sample per pixel constraint, mCancellationState can be
    // initialized to WAITING_FOR_CANCEL. We do this for batch mode.
    mCancellationState = waitUntilReadyForDisplay ? WAITING_FOR_SAMPLE_PER_PIXEL
                                                  : WAITING_FOR_CANCEL;
    mCurrentPassIdx = 0;
}

void
TLState::disableCancellation()
{
    MNRY_ASSERT(mFs);
    mCancellationState = DISABLED;
    mCurrentPassIdx = 0;
}

bool
TLState::isCanceled()
{
    if (mCancellationState <= CANCELED) {
        return mCancellationState != DISABLED;
    }

    MNRY_ASSERT(mFs);

    switch (mCancellationState) {

    case WAITING_FOR_SAMPLE_PER_PIXEL:
        if (mCurrentPassIdx > 0) {
            mCancellationState = WAITING_FOR_CANCEL;
            if (isRenderCanceled()) {
                mCancellationState = CANCELED;
                return true;
            }
        }
        break;

    case WAITING_FOR_CANCEL:
        if (isRenderCanceled()) {
            mCancellationState = CANCELED;
            return true;
        }
        break;

    default:
        MNRY_ASSERT(0);
    }

    return false;
}

void
TLState::cacheThreadLocalAccumulators()
{
    MNRY_VERIFY(mExclusiveAccumulators)->cacheThreadLocalAccumulators(mThreadIdx);
    mIspcAccumulator = &getAccumulator(ACCUM_TOTAL_ISPC)->mThreadLocal[mThreadIdx];
}

bool
TLState::isIntegratorAccumulatorRunning() const
{
    return mExclusiveAccumulators->isRunning(EXCL_ACCUM_INTEGRATION) ||
           mExclusiveAccumulators->isRunning(EXCL_ACCUM_SSS_INTEGRATION) ||
           mExclusiveAccumulators->isRunning(EXCL_ACCUM_VOL_INTEGRATION);
}

void
TLState::setRadianceQueueHandler(RadianceQueue::Handler handler, void *handlerData)
{
    mRadianceQueue->setHandler(handler, handlerData);
}

void
TLState::setAovQueueHandler(AovQueue::Handler handler, void *handlerData)
{
    mAovQueue->setHandler(handler, handlerData);
}

void
TLState::setHeatMapQueueHandler(HeatMapQueue::Handler handler, void *handlerData)
{
    mHeatMapQueue->setHandler(handler, handlerData);
}

// static function
std::shared_ptr<TLState>
TLState::allocTls(mcrt_common::ThreadLocalState *tls,
                  const mcrt_common::TLSInitParams &initParams,
                  bool okToAllocBundledResources)
{
    gPoolData.init(initParams);

    return std::make_shared<pbr::TLState>(tls, initParams, okToAllocBundledResources);
}

//------------------------------------------------------------------------------------------

// private function
template<typename ResType, typename PoolType>
inline void
TLState::poolAlloc(const char * const typeName,
                   PoolType &pool,
                   unsigned numEntries,
                   ResType **entries,
                   OverlappedAccType accumStall,
                   std::atomic<unsigned> &numFailedAllocs)
{
    bool success = pool.allocList(numEntries, entries);

    if (__builtin_expect(success, true)) {
        // 99.9999% case, allocation should succeed.

    } else {
        // 0.0001% case:

        // Watchdog loop, report if we can't allocate a new entry
        // within a set amount of time.
        ACCUMULATOR_PROFILE(mThreadIdx, accumStall);

        const double allocTimeout = 1.f;
        double time = scene_rdl2::util::getSeconds();

        do {
            ++numFailedAllocs;
            scene_rdl2::logging::Logger::warn("Couldn't allocate a ", typeName, " on thread ", mThreadIdx,
                " (numEntries: ", numEntries, " sizeof(Entry): ", sizeof(ResType),
                "), falling back to early draining of queues (", numFailedAllocs, ").\n");

            // Try and free up some entries with an explicit flush.
            flushLocalQueues();

            if (pool.allocList(numEntries, entries)) {
                success = true;
                break;
            }

            // Sleep for a while as a last resort.
            mcrt_common::threadSleep();

        } while (scene_rdl2::util::getSeconds() - time < allocTimeout);

        if (!success) {
            scene_rdl2::logging::Logger::fatal("Rendering thread timed out whilst trying to allocate a ",
                          typeName, "! Try increasing pool size.");
        }
    }
}

inline void*
TLState::handleToCL1Ptr(uint32_t handle) const
{
    CacheLine1* baseCL1 = reinterpret_cast<CacheLine1*>(mCL1Pool.getEntryMemory());

    handle = handle & ~ALLOC_LIST_INFO_BITS; // mask off the info bits
    MNRY_ASSERT(handle < mCL1Pool.getActualPoolSize());
    void* result = baseCL1 + handle;
    return result;
}            

inline uint32_t
TLState::cl1PtrToHandle(const void* bytes, const unsigned numItems) const
{
    auto isValidAddr = [&](const void* bytes, CacheLine1* const baseCL1) {
        if (!bytes) return false;
        return isValidCL1Addr(static_cast<const CacheLine1*>(bytes), baseCL1);
    };

    CacheLine1* const baseCL1 = reinterpret_cast<CacheLine1*>(mCL1Pool.getEntryMemory());
    MNRY_ASSERT(isValidAddr(bytes, baseCL1));

    unsigned result = (const CacheLine1*)bytes - baseCL1;

    // encode number of items in the info bits
    MNRY_ASSERT(!(result & ALLOC_LIST_INFO_BITS));
    result = result | ((numItems - 1) << ALLOC_LIST_INFO_BIT_SHIFT);

    MNRY_ASSERT(handleToCL1Ptr(result) == bytes);

    return result;
}

inline bool
TLState::isValidCL1Addr(const CacheLine1* ptr, const CacheLine1* baseCL1) const
{
    const unsigned actualPoolSize = mCL1Pool.getActualPoolSize();
    return (baseCL1 <= ptr && ptr < (baseCL1 + actualPoolSize));
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

// hooks for ispc - must match declarations in TLState.ispc
extern "C" uint32_t
CPP_PbrTLState_allocList(TLState *pbrTls, unsigned itemSize, unsigned numItems)
{
    return pbrTls->allocList(itemSize, numItems);
}

extern "C" void
CPP_PbrTLState_freeList(TLState *pbrTls, uint32_t listHandle)
{
    return pbrTls->freeList(listHandle);
}

extern "C" unsigned
CPP_PbrTLState_getNumListItems(TLState *pbrTls, uint32_t listHandle)
{
    return pbrTls->getNumListItems(listHandle);
}

extern "C" void *
CPP_PbrTLState_getListItem(TLState *pbrTls, uint32_t listHandle, unsigned item)
{
    return pbrTls->getListItem(listHandle, item);
}

extern "C" void
CPP_PbrTLState_acquireDeepData(TLState *pbrTls, uint32_t deepPtr)
{
    pbrTls->acquireDeepData(deepPtr);
}

extern "C" void
CPP_PbrTLState_releaseDeepData(TLState *pbrTls, uint32_t deepPtr)
{
    pbrTls->releaseDeepData(deepPtr);
}

extern "C" void
CPP_PbrTLState_acquireCryptomatteData(TLState *pbrTls, uint32_t cryptomattePtr)
{
    pbrTls->acquireCryptomatteData(cryptomattePtr);
}

extern "C" void
CPP_PbrTLState_releaseCryptomatteData(TLState *pbrTls, uint32_t cryptomattePtr)
{
    pbrTls->releaseCryptomatteData(cryptomattePtr);
}

//------------------------------------------------------------------------------------------

void heatMapBundledUpdate(TLState *pbrTls,
                          int64_t ticks,
                          const pbr::RayState * const *rayStates,
                          unsigned numEntries)
{
    MNRY_ASSERT(numEntries > 0);
    // This function is an estimate!  It assumes that the amount
    // of time spent on any one of the ray-states is equal to any of the
    // others.  So it works best when numEntries is minimal.
    const int64_t ticksPerEntry = ticks / numEntries;

    if (!ticksPerEntry) return;

    SCOPED_MEM(pbrTls->mArena);

    BundledHeatMapSample *b = pbrTls->mArena->allocArray<BundledHeatMapSample>(numEntries, CACHE_LINE_SIZE);
    unsigned bIndx;
    BundledHeatMapSample::initArray(b, bIndx, rayStates[0]->mSubpixel.mPixel);

    // This works best when there is some pixel coherence
    // in rayStates.  If random w.r.t pixels, then we'll
    // submit an entry for every ray state.
    for (unsigned i = 0; i < numEntries; ++i) {
        BundledHeatMapSample::addTicksToArray(b, bIndx, ticksPerEntry,
                                              rayStates[i]->mSubpixel.mPixel);
    }

    pbrTls->addHeatMapQueueEntries(bIndx + 1, b);
}

HUD_VALIDATOR( PbrTLState );

//-----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray
