// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <scene_rdl2/render/util/Arena.h>
#ifndef PLATFORM_APPLE
#include <scene_rdl2/render/util/NumaUtil.h>
#endif // end of Not PLATFORM_APPLE

#include <memory> // shared_ptr
#include <vector>

namespace moonray {
namespace mcrt_common {

class CpuAffinityManager;

class MemoryNode
//
// This class keeps ArenaBlockPool and related memory info for the single NUMA-node
//
{
public:
    // Constructor regarding Memory Affinity Off
    MemoryNode(const unsigned activeThreadCount);

#ifndef PLATFORM_APPLE
    // Constructor regarding Both Memory Affinity On/Off depending on the numaNodeId argument
    // numaNodeId : ~0        : Memory Affinity Off
    //              otherwise : Memory Affinity On
    MemoryNode(const unsigned numaNodeId,
               const scene_rdl2::NumaNode* numaNode,
               const unsigned activeThreadCount);
#endif // end of Non PLATFORM_APPLE

#ifndef PLATFORM_APPLE
    unsigned getNumaNodeId() const { return mNumaNodeId; }
    const scene_rdl2::NumaNode* getNumaNode() const { return mNumaNode; }
#endif // end of Non PLATFORM_APPLE

    // Return the active thread count of this NUMA-node.
    // This NUMA-node memory is shared with this number of threads
    unsigned getActiveThreadCount() const { return mActiveThreadCount; }

    scene_rdl2::alloc::ArenaBlockPool* getArenaBlockPool() { return mArenaBlockPool.get(); }

    std::string show() const;

    static std::string showNumaNodeId(const unsigned numaNodeId);

private:
    unsigned mNumaNodeId {~static_cast<unsigned>(0)};
#ifndef PLATFORM_APPLE
    const scene_rdl2::NumaNode* mNumaNode {nullptr};
#endif // end PLATFORM_APPLE
    unsigned mActiveThreadCount {0}; // Memory is shared with this number of threads

    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> mArenaBlockPool;
};

class MemoryAffinityManager
//
// This class manages Memory-Affinity control info for the MoonRay Process
//
{
public:
    // memAffinityDef : "true"  : set true
    //                  "false" : set false   
    MemoryAffinityManager(const std::string& memAffinityDef);

    bool init(const std::shared_ptr<const CpuAffinityManager>& cpuAff);

    bool getMemAffinityEnable() const { return mMemAffinityEnable; }

    unsigned getActiveNumaNodeCount() const;

    std::shared_ptr<MemoryNode> getMemoryNodeByThreadId(const unsigned mcrtThreadId) const { 
        if (mMemAffinityEnable) {
            if (mcrtThreadId <= mMaxThreadId) {
                return mMemNodeTbl[mMcrtThreadIdToMemNodeIdTbl[mcrtThreadId]];
            }
        }
        return mMemGlobal;
    }
    std::shared_ptr<MemoryNode> getMemoryNodeByNumaNodeId(const unsigned numaNodeId) const {
        return mMemNodeTbl[numaNodeId];
    }

    unsigned getMemNodeTblSize() const { return mMemNodeTbl.size(); }
    bool isActiveMemNode(const unsigned memNodeId) const {
        if (memNodeId >= getMemNodeTblSize()) return false;
        return (mMemNodeTbl[memNodeId] != nullptr);
    }
    const std::vector<std::shared_ptr<MemoryNode>>& getMemNodeTbl() { return mMemNodeTbl; }

    const std::string& getMcrtMessage() const { return mMcrtMessage; }

    void setupLogInfo(std::vector<std::string>& titleTbl,
                      std::vector<std::string>& msgTbl,
                      const std::shared_ptr<const CpuAffinityManager>& cpuAff) const;

    std::string show() const;

private:

#ifndef PLATFORM_APPLE
    std::vector<unsigned> calcActiveNumaNodeIdTbl(const std::shared_ptr<const CpuAffinityManager>& cpuAff) const;
    void setupMcrtThreadIdToMemNodeIdTbl(const std::shared_ptr<const CpuAffinityManager>& cpuAff);
    unsigned calcThreadCountOnActiveNumaNode(const unsigned numaNodeId) const;
#endif // end of Non PLATFORM_APPLE

    std::string showNumaUtil() const;
    std::string showMemGlobal() const;
    std::string showMemNodeTbl() const;
    std::string showMcrtThreadIdToMemNodeIdTbl() const;

    //------------------------------

    const std::string mMemAffinityDef;
    bool mMemAffinityEnable {false};
    unsigned mMaxThreadId {0};

#ifndef PLATFORM_APPLE
    std::shared_ptr<scene_rdl2::NumaUtil> mNumaUtil;
#endif // end PLATFORM_APPLE

    std::shared_ptr<MemoryNode> mMemGlobal; // for non memory affinity configuration and GUI TLS

    // size == total NUMA node of this host, some of them might be null
    std::vector<std::shared_ptr<MemoryNode>> mMemNodeTbl;

    std::vector<unsigned> mMcrtThreadIdToMemNodeIdTbl;

    std::string mMcrtMessage;
};

//------------------------------------------------------------------------------------------

class CpuAffinityManager
//
// This class manages CPU-Affinity control info for the MoonRay process
//
{
public:
    using CpuIdTbl = std::vector<unsigned>;

    CpuAffinityManager(const unsigned desiredNumThreads,
                       const std::string& cpuAffinityDef,
                       const std::string& socketAffinityDef);

    const std::string& getCpuAffinityDef() const { return mCpuAffinityDef; }
    const std::string& getSocketAffinityDef() const { return mSocketAffinityDef; }

    const CpuIdTbl& getAffinityCpuIdTbl() const { return mCpuIdTbl; }
    unsigned getNumThreads() const { return mNumThreads; }

    //------------------------------
    // RenderPrep
    const std::string& getRenderPrepMessage() const { return mRenderPrepMessage; }
    bool getEnableRenderPrepCpuAffinity() const { return mEnableRenderPrepCpuAffinity; }
    bool doRenderPrepCpuAffinity(std::string& msg); // update internal mNumThreads

    //------------------------------
    // MCRT
    const std::string& getMcrtMessage() const { return mMcrtMessage; }
    bool getEnableMcrtCpuAffinity() const { return mEnableMcrtCpuAffinity; }
    bool getEnableMcrtCpuAffinityAll() const { return mEnableMcrtCpuAffinityAll; }

    unsigned mcrtThreadIdToCpuId(const unsigned mcrtThreadId) const 
    {
        if (!mEnableMcrtCpuAffinity || mEnableMcrtCpuAffinityAll) return mcrtThreadId;
        return mCpuIdTbl[mcrtThreadId];
    }

    //------------------------------

    void setupLogInfo(std::vector<std::string>& titleTbl, std::vector<std::string>& msgTbl) const;

    std::string show() const;

private:

    void configureCpuAffinity();

    //------------------------------

    unsigned mNumThreads {0};
    const std::string mCpuAffinityDef;
    const std::string mSocketAffinityDef;

    CpuIdTbl mCpuIdTbl;
    std::string mRenderPrepMessage;
    std::string mMcrtMessage;

    bool mEnableRenderPrepCpuAffinity {false};
    bool mEnableMcrtCpuAffinity {false};
    bool mEnableMcrtCpuAffinityAll {false}; // MCRT stage uses all cores or not
};

//------------------------------------------------------------------------------------------

class AffinityManager
//
// This class manages CPU + Memory affinity control for the MoonRay process
//
{
public:
    AffinityManager(const unsigned desiredNumThreads,
                    const std::string& cpuAffinityDef,
                    const std::string& socketAffinityDef,
                    const std::string& memAffinityDef);

    static bool init(const unsigned desiredNumThreads,
                     const std::string& autoAffinityDef,
                     const std::string& cpuAffinityDef,
                     const std::string& socketAffinityDef,
                     const std::string& memAffinityDef);

    static std::shared_ptr<AffinityManager> get();

    std::shared_ptr<MemoryAffinityManager> getMem() const { return mMemManager; }
    std::shared_ptr<CpuAffinityManager> getCpu() const { return mCpuManager; }

    void setupLogInfo(std::vector<std::string>& titleTbl, std::vector<std::string>& msgTbl) const;

    std::string show() const;

    static std::string showTbl(const std::string& msg, const std::vector<unsigned>& tbl);

private:
    static void calcAutoAffinityOptions(const unsigned desiredNumThreads,
                                        std::string& cpuAffinityDef,
                                        std::string& memAffinityDef);

    std::shared_ptr<MemoryAffinityManager> mMemManager;
    std::shared_ptr<CpuAffinityManager> mCpuManager;
};

} // namespace mcrt_common
} // namespace moonray
