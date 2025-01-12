// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0
#include "AffinityManager.h"

#include <scene_rdl2/common/except/exceptions.h>
#ifndef PLATFORM_APPLE
#include <scene_rdl2/render/util/CpuSocketUtil.h>
#include <scene_rdl2/render/util/ProcCpuAffinity.h>
#endif
#include <scene_rdl2/render/util/StrUtil.h>

#include <numeric> // std::iota
#include <sstream>
#include <thread>

//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------

namespace moonray {
namespace mcrt_common {
    
MemoryNode::MemoryNode(const unsigned activeThreadCount)
    : mNumaNodeId {~static_cast<unsigned>(0)}
#ifndef PLATFORM_APPLE
    , mNumaNode {nullptr}
#endif // end non PLATFORM_APPLE
    , mActiveThreadCount {activeThreadCount}
//
// MemoryNode constructor regarding Memory Affinity Off
//
{
    mArenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);

    //
    // Memory Affinity disabled
    //
    mArenaBlockPool->setupNumaInfo(mNumaNodeId, nullptr, nullptr); // we don't need special callBacks

    /* useful debug message
    std::cerr << "AffinityManager.cc MemoryNode::MemoryNode {\n"
              << scene_rdl2::str_util::addIndent(mArenaBlockPool->show()) << '\n'
              << "}\n";
    */
}

#ifndef PLATFORM_APPLE
    
MemoryNode::MemoryNode(const unsigned numaNodeId,
                       const scene_rdl2::NumaNode* numaNode,
                       const unsigned activeThreadCount)
    : mNumaNodeId {numaNodeId}
    , mNumaNode {numaNode}
    , mActiveThreadCount {activeThreadCount}
//
// MemoryNode constructor regarding both of Memory Affinity On/Off depending on the numaNodeId argument
//   numaNodeId : ~0        : Memory Affinity Off
//                otherwise : Memory Affinity On
//
{
    mArenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);
    if (mNumaNodeId != ~0) {
        //
        // Memory Affinity enabled
        //
        mArenaBlockPool->setupNumaInfo(mNumaNodeId,
                                       [&](size_t size, size_t alignment) -> void* { // allocCallBack
                                           MNRY_ASSERT(mNumaNode->alignmentSizeCheck(alignment)); 
                                           return mNumaNode->alloc(size);
                                       },
                                       [&](void* addr, size_t size) { // freeCallBack
                                           mNumaNode->free(addr, size);
                                       });
    } else {
        //
        // Memory Affinity disabled
        //
        mArenaBlockPool->setupNumaInfo(mNumaNodeId, nullptr, nullptr); // we don't need special callBacks
    }

    /* useful debug message
    std::cerr << "AffinityManager.cc MemoryNode::MemoryNode {\n"
              << scene_rdl2::str_util::addIndent(mArenaBlockPool->show()) << '\n'
              << "}\n";
    */
}

#endif // end of Non PLATFORM_APPLE

std::string
MemoryNode::show() const
{
#ifdef PLATFORM_APPLE
    auto showNumaNode = [&]() -> std::string {
        return "mNumaNode is nullptr";
    };
#else // else PLATFORM_APPLE
    auto showNumaNode = [&]() -> std::string {
        if (!mNumaNode) return "mNumaNode is nullptr";
        return mNumaNode->show();
    };
#endif // end of Non PLATFORM_APPLE

    std::ostringstream ostr;
    ostr << "MemoryNode {\n"
         << "  mNumaNodeId:" << showNumaNodeId(mNumaNodeId) << '\n'
         << scene_rdl2::str_util::addIndent(showNumaNode()) << '\n'
         << "  mActiveThreadCount:" << mActiveThreadCount << '\n'
         << "  mArenaBlockPool:0x" << std::hex << reinterpret_cast<uintptr_t>(mArenaBlockPool.get()) << '\n'
         << scene_rdl2::str_util::addIndent(mArenaBlockPool->show()) << '\n'
         << "}";
    return ostr.str();
}

// static function
std::string
MemoryNode::showNumaNodeId(const unsigned numaNodeId)
{
    if (numaNodeId == ~0) return "not-defined";
    return std::to_string(numaNodeId);
}

//-----------------------------------------------------------------------------------------

MemoryAffinityManager::MemoryAffinityManager(const std::string& memAffinityDef)
    : mMemAffinityDef {memAffinityDef}
{
}

#ifdef PLATFORM_APPLE

bool
MemoryAffinityManager::init(const std::shared_ptr<const CpuAffinityManager>& cpuAff)
//
// for Mac
//
{
    mMaxThreadId = cpuAff->getNumThreads() - 1;

    mMemAffinityEnable = false; // memory affinity is always turn off on Mac

    mMemNodeTbl.clear();
    mMcrtMessage = "MEM-affinity control disabled"; 

    // setup memGlobal for non MemAffinity condition and for GUI TLS
    const unsigned activeThreadCount = mMaxThreadId + 1;
    mMemGlobal = std::make_shared<MemoryNode>(activeThreadCount);

    return true;
}

#else // else PLATFORM_APPLE    

bool
MemoryAffinityManager::init(const std::shared_ptr<const CpuAffinityManager>& cpuAff)
//
// for Linux
//
{
    mMaxThreadId = cpuAff->getNumThreads() - 1;

    mMemAffinityEnable = false;
    if (cpuAff->getEnableMcrtCpuAffinity()) {
        if (!mMemAffinityDef.empty()) {
            mMemAffinityEnable = scene_rdl2::str_util::isTrueStr(mMemAffinityDef);
        }
    }

    mMemNodeTbl.clear();
    if (mMemAffinityEnable) {
        //
        // enable memory affinity : initialize only active NumaNode. Non active NumaNode should be nullptr
        //

        // we have to construct NumaUtil before execute calcActiveNumaNodeIdTbl()
        mNumaUtil = std::make_shared<scene_rdl2::NumaUtil>();

        setupMcrtThreadIdToMemNodeIdTbl(cpuAff);

        std::vector<unsigned> activeNumaNodeIdTbl = calcActiveNumaNodeIdTbl(cpuAff);
        {
            std::ostringstream ostr;
            ostr << "MEM-affinity control enabled : "
                 << AffinityManager::showTbl("active-NUMA-node", activeNumaNodeIdTbl);
            mMcrtMessage = ostr.str();
        }

        mMemNodeTbl.resize(mNumaUtil->getTotalNumaNode());
        for (auto numaNodeId : activeNumaNodeIdTbl) {
            if (numaNodeId < mMemNodeTbl.size()) {
                const scene_rdl2::NumaNode* numaNode = mNumaUtil->getNumaNode(numaNodeId);
                const unsigned activeThreadCount = calcThreadCountOnActiveNumaNode(numaNodeId);
                mMemNodeTbl[numaNodeId] =
                    std::make_shared<MemoryNode>(numaNodeId, numaNode, activeThreadCount);
            }
        }
    } else {
        mMcrtMessage = "MEM-affinity control disabled"; 
    }

    // setup memGlobal for non MemAffinity condition and for GUI TLS
    const unsigned activeThreadCount = mMaxThreadId + 1;
    mMemGlobal = std::make_shared<MemoryNode>(activeThreadCount);

    return true;
}

#endif // end of Non PLATFORM_APPLE 

unsigned
MemoryAffinityManager::getActiveNumaNodeCount() const
{
    unsigned total = 0;
    for (auto itr : mMemNodeTbl) {
        if (itr) total++;
    }
    return total;
}

#ifdef PLATFORM_APPLE

void
MemoryAffinityManager::setupLogInfo(std::vector<std::string>& titleTbl,
                                    std::vector<std::string>& msgTbl,
                                    const std::shared_ptr<const CpuAffinityManager>& cpuAff) const
{
    titleTbl.push_back("MCRT MEM-affinity");
    msgTbl.push_back("disable");
}

#else // else PLATFORM_APPLE

void
MemoryAffinityManager::setupLogInfo(std::vector<std::string>& titleTbl,
                                    std::vector<std::string>& msgTbl,
                                    const std::shared_ptr<const CpuAffinityManager>& cpuAff) const
{
    titleTbl.push_back("MCRT MEM-affinity");
    if (mMemAffinityEnable) {
        std::vector<unsigned> activeNumaNodeTbl = calcActiveNumaNodeIdTbl(cpuAff);
        msgTbl.push_back(AffinityManager::showTbl("active-NUMA-node", activeNumaNodeTbl));
    } else {
        msgTbl.push_back("disable");
    }
}

#endif // end of Non PLATFORM_APPLE

std::string
MemoryAffinityManager::show() const
{
    std::ostringstream ostr;
    ostr << "MemoryAffinityManager {\n"
         << "  mMemAffinityDef:" << mMemAffinityDef << '\n'
         << "  mMemAffinityEnable:" << scene_rdl2::str_util::boolStr(mMemAffinityEnable) << '\n'
         << "  mMaxThreadId:" << mMaxThreadId << '\n'
         << scene_rdl2::str_util::addIndent(showNumaUtil()) << '\n'
         << scene_rdl2::str_util::addIndent(showMemGlobal()) << '\n'
         << scene_rdl2::str_util::addIndent(showMemNodeTbl()) << '\n'
         << scene_rdl2::str_util::addIndent(showMcrtThreadIdToMemNodeIdTbl()) << '\n'
         << "}";
    return ostr.str();
}

#ifndef PLATFORM_APPLE
    
std::vector<unsigned>
MemoryAffinityManager::calcActiveNumaNodeIdTbl(const std::shared_ptr<const CpuAffinityManager>& cpuAff) const
//
// for Linux
//
{
    std::vector<unsigned> activeNumaNodeIdTbl;
    if (!cpuAff->getEnableMcrtCpuAffinity()) return activeNumaNodeIdTbl; // Just in case, return empty tbl

    if (cpuAff->getEnableMcrtCpuAffinityAll()) {
        // all
        activeNumaNodeIdTbl.resize(mNumaUtil->getTotalNumaNode());
        std::iota(activeNumaNodeIdTbl.begin(), activeNumaNodeIdTbl.end(), 0);
    } else {
        activeNumaNodeIdTbl = mNumaUtil->genActiveNumaNodeIdTblByCpuIdTbl(cpuAff->getAffinityCpuIdTbl());
    }
    return activeNumaNodeIdTbl;
}

void
MemoryAffinityManager::setupMcrtThreadIdToMemNodeIdTbl(const std::shared_ptr<const CpuAffinityManager>& cpuAff)
//
// for Linux
//
{
    const unsigned numThreads = cpuAff->getNumThreads();
    mMcrtThreadIdToMemNodeIdTbl.resize(numThreads);
    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        unsigned cpuId = cpuAff->mcrtThreadIdToCpuId(threadId);
        mMcrtThreadIdToMemNodeIdTbl[threadId] = mNumaUtil->findNumaNodeByCpuId(cpuId)->getNodeId();
    }
}

unsigned
MemoryAffinityManager::calcThreadCountOnActiveNumaNode(const unsigned numaNodeId) const
//
// for Linux
//
{
    int totalThreads = 0;
    for (auto itr : mMcrtThreadIdToMemNodeIdTbl) {
        if (itr == numaNodeId) totalThreads++;
    }
    return totalThreads;
}

#endif // end of Non PLATFORM_APPLE

std::string
MemoryAffinityManager::showNumaUtil() const
{
    std::ostringstream ostr;
#ifdef PLATFORM_APPLE
    ostr << "mNumaUtil is empty";
#else // else PLATFORM_APPLE
    if (!mNumaUtil) {
        ostr << "mNumaUtil is empty";
    } else {
        ostr << "mNumaUtil {\n"
             << scene_rdl2::str_util::addIndent(mNumaUtil->show()) << '\n'
             << "}";
    }
#endif // end of Non PLATFORM_APPLE
    return ostr.str();
}

std::string
MemoryAffinityManager::showMemGlobal() const
{
    std::ostringstream ostr;
    if (!mMemGlobal) {
        ostr << "mMemGlobal is empty";
    } else {
        ostr << "mMemGlobal {\n"
             << scene_rdl2::str_util::addIndent(mMemGlobal->show()) << '\n'
             << "}";
    }
    return ostr.str();
}

std::string
MemoryAffinityManager::showMemNodeTbl() const
{
    std::ostringstream ostr;
    if (mMemNodeTbl.empty()) {
        ostr << "mMemNodeTbl is empty";
    } else {
        ostr << "mMemNodeTbl (size:" << mMemNodeTbl.size() << ") {\n";
        for (size_t i = 0; i < mMemNodeTbl.size(); ++i) {
            if (mMemNodeTbl[i]) {
                ostr << scene_rdl2::str_util::addIndent(std::to_string(i) + ' ' + mMemNodeTbl[i]->show())
                     << '\n';
            } else {
                ostr << "  " << i << " is not active then empty\n";
            }
        }
        ostr << "}";
    }
    return ostr.str();
}

std::string
MemoryAffinityManager::showMcrtThreadIdToMemNodeIdTbl() const
{
    std::ostringstream ostr;
    if (mMcrtThreadIdToMemNodeIdTbl.empty()) {
        ostr << "mMcrtThreadIdToMemNodeIdTbl is empty";
    } else {
        unsigned maxNodeId =
            *(std::max_element(mMcrtThreadIdToMemNodeIdTbl.begin(), mMcrtThreadIdToMemNodeIdTbl.end()));
        ostr << "mMcrtThreadIdToMemNodeIdTbl (size:" << mMcrtThreadIdToMemNodeIdTbl.size() << ") {\n";
        int w0 = scene_rdl2::str_util::getNumberOfDigits(mMcrtThreadIdToMemNodeIdTbl.size());
        int w1 = scene_rdl2::str_util::getNumberOfDigits(maxNodeId);
        for (size_t id = 0; id < mMcrtThreadIdToMemNodeIdTbl.size(); ++id) {
            ostr << "  threadId:" << std::setw(w0) << id
                 << " -> NUMA-nodeId:" << std::setw(w1) << mMcrtThreadIdToMemNodeIdTbl[id] << '\n';
        }
        ostr << "}";
    }
    return ostr.str();
}

//------------------------------------------------------------------------------------------

CpuAffinityManager::CpuAffinityManager(const unsigned desiredNumThreads,
                                       const std::string& cpuAffinityDef,
                                       const std::string& socketAffinityDef)
    : mNumThreads {desiredNumThreads}
#ifdef PLATFORM_APPLE
    , mCpuAffinityDef {""}
    , mSocketAffinityDef {""}
#else // else PLATFORM_APPLE
    , mCpuAffinityDef {cpuAffinityDef}
    , mSocketAffinityDef {socketAffinityDef}
#endif // end of Non PLATFORM_APPLE
{
    configureCpuAffinity();
}

#ifdef PLATFORM_APPLE
bool
CpuAffinityManager::doRenderPrepCpuAffinity(std::string& msg)
{
    // empty function
    return true;
}

#else // Not PLATFORM_APPLE

bool
CpuAffinityManager::doRenderPrepCpuAffinity(std::string& msg)
{
    if (!mEnableRenderPrepCpuAffinity) return true; // skip cpu affinity

    //
    // Set process based CPU affinity
    //
    bool result = true;
    std::ostringstream ostr;
    try {
        scene_rdl2::ProcCpuAffinity procCpuAffinity;
        for (auto cpuId : mCpuIdTbl) { procCpuAffinity.set(cpuId); }
        std::string tmpMsg;
        if (!procCpuAffinity.bindAffinity(tmpMsg)) {
            ostr << "RenderPrep Bind CPU-affinity failed. " << tmpMsg
                 << " RenderPrep CPU-affinity control skipped";
            result = false;
        } else {
            ostr << "RenderPrep " << tmpMsg;
        }
    }
    catch (scene_rdl2::except::RuntimeError& e) {
        ostr << "RenderPrep doRenderPrepCpuAffinity() failed. " << e.what()
             << " RenderPrep CPU-affinity control skipped";
        result = false;
    }

    msg = ostr.str();
    return result;
}
#endif // end of Not PLATFORM_APPLE

#ifdef PLATFORM_APPLE

void
CpuAffinityManager::setupLogInfo(std::vector<std::string>& titleTbl,
                                 std::vector<std::string>& msgTbl) const
//
// for Mac
//
{
    titleTbl.push_back("RenderPrep CPU-affinity");
    msgTbl.push_back("disabled");

    titleTbl.push_back("MCRT CPU-affinity");
    msgTbl.push_back("disabled");
}

#else // else PLATFORM_APPLE

void
CpuAffinityManager::setupLogInfo(std::vector<std::string>& titleTbl,
                                 std::vector<std::string>& msgTbl) const
//
// for Linux
//
{
    titleTbl.push_back("RenderPrep CPU-affinity");
    if (mEnableRenderPrepCpuAffinity) {
        if (mSocketAffinityDef.empty()) {
            // -cpu_affinity
            msgTbl.push_back(AffinityManager::showTbl("cpuId", mCpuIdTbl));
        } else {
            // -socket_affinity
            std::ostringstream ostr;
            ostr << "socketId " << mSocketAffinityDef << " cpuId";
            msgTbl.push_back(AffinityManager::showTbl(ostr.str(), mCpuIdTbl));
        }
    } else {
        msgTbl.push_back("disabled");
    }

    titleTbl.push_back("MCRT CPU-affinity");
    if (mEnableMcrtCpuAffinity) {
        if (mEnableMcrtCpuAffinityAll) {
            std::ostringstream ostr;
            ostr << "all : total " << mNumThreads << " threads";
            msgTbl.push_back(ostr.str());
        } else {
            if (mSocketAffinityDef.empty()) {
                // -cpu_affinity
                msgTbl.push_back(AffinityManager::showTbl("cpuId", mCpuIdTbl));
            } else {
                // -socket_affinity
                std::ostringstream ostr;
                ostr << "socketId " << mSocketAffinityDef << " cpuId";
                msgTbl.push_back(AffinityManager::showTbl(ostr.str(), mCpuIdTbl));
            }
        }
    } else {
        msgTbl.push_back("disabled");
    }
}

#endif // end of Non PLATFORM_APPLE

std::string
CpuAffinityManager::show() const
{
    using scene_rdl2::str_util::boolStr;
    using scene_rdl2::str_util::addIndent;

    std::ostringstream ostr;
    ostr << "CpuAffinityManager {\n"
         << "  mNumThreads:" << mNumThreads << '\n'
         << "  mCpuAffinityDef:" << mCpuAffinityDef << '\n'
         << "  mSocketAffinityDef:" << mSocketAffinityDef << '\n'
         << addIndent(AffinityManager::showTbl("mCpuIdTbl", mCpuIdTbl)) << '\n'
         << "  mRenderPrepMessage:" << mRenderPrepMessage << '\n'
         << "  mMcrtMessage:" << mMcrtMessage << '\n'
         << "  mEnableRenderPrepCpuAffinity:" << boolStr(mEnableRenderPrepCpuAffinity) << '\n'
         << "  mEnableMcrtCpuAffinity:" << boolStr(mEnableMcrtCpuAffinity) << '\n'
         << "  mEnableMcrtCpuAffinityAll:" << boolStr(mEnableMcrtCpuAffinityAll) << '\n'
         << "}";
    return ostr.str();
}

#ifdef PLATFORM_APPLE
void
CpuAffinityManager::configureCpuAffinity()
{
    // Skip all configuration regarding CPU affinity for Mac

    mCpuIdTbl.clear();
    mRenderPrepMessage = "RenderPrep CPU-affinity control disabled";
    mMcrtMessage = "CPU-affinity control disabled";

    mEnableRenderPrepCpuAffinity = false;
    mEnableMcrtCpuAffinity = false;
    mEnableMcrtCpuAffinityAll = false;
}

#else // Not PLATFORM_APPLE

void
CpuAffinityManager::configureCpuAffinity()
{
    auto setAllCpus = [&]() {
        mCpuIdTbl.resize(std::thread::hardware_concurrency());
        std::iota(mCpuIdTbl.begin(), mCpuIdTbl.end(), 0);            
    };

    bool forceToDisable = false;

    std::ostringstream ostr;
    std::string errMsg;

    //------------------------------
    //
    // RenderPrep CPU affinity
    //
    if (!mCpuAffinityDef.empty()) {
        if (mCpuAffinityDef == "-1") {
            // We will try to apply CPU-Affinity control for MCRT threads even if no CPU-Affinity control
            // for renderPrep. However, if the user sets "-1" (= explicitly disables CPU-Affinity),
            // we disable CPU-Affinity control regarding both renderPrep and MCRT threads.
            ostr << "RenderPrep CPU-affinity control disabled";
            mCpuIdTbl.clear();
            forceToDisable = true;
        } else if (mCpuAffinityDef == "all") {
            //
            // use entire CPUs
            //
            setAllCpus();
        } else {
            //
            // pick CpuAffinity info
            //
            if (!scene_rdl2::CpuSocketUtil::cpuIdDefToCpuIdTbl(mCpuAffinityDef,
                                                               mCpuIdTbl,
                                                               errMsg)) {
                ostr << "CPU-affinity definition failed. " << errMsg
                     << " RenderPrep CPU-affinity control skipped";
                mCpuIdTbl.clear();
            } else {
                if (!mCpuIdTbl.size()) {
                    ostr << "CPU-affinity definition is empty. RenderPrep CPU-affinity control skipped";
                } else {
                    ostr << AffinityManager::showTbl("RenderPrep CPU-affinity cpuIdTbl", mCpuIdTbl);
                }
            }
        }
    } else if (!mSocketAffinityDef.empty()) {
        try {
            scene_rdl2::CpuSocketUtil cpuSocketUtil;
            if (mSocketAffinityDef == "all") {
                //
                // use entire CPUs
                //
                setAllCpus();
            } else {
                //
                // pick SocketAffinity info
                //
                if (!cpuSocketUtil.socketIdDefToCpuIdTbl(mSocketAffinityDef,
                                                         mCpuIdTbl,
                                                         errMsg)) {
                    ostr << "Socket-affinity definition failed. " << errMsg
                         << " RenderPrep CPU-affinity control skipped.";
                    mCpuIdTbl.clear();
                } else {
                    if (!mCpuIdTbl.size()) {
                        ostr << "Socket-affinity definition is empty. RenderPrep CPU-affinity control skipped";
                    } else {
                        ostr << "RenderPrep Socket-affinity " << mSocketAffinityDef
                             << AffinityManager::showTbl(" cpuIdTbl", mCpuIdTbl);
                    }
                }
            }
        }
        catch (scene_rdl2::except::RuntimeError& e) {
            ostr << "Socket-affinity processing failed. RenderPrep CPU-affinity control skipped. " << e.what();
            mCpuIdTbl.clear();
        }
    } else {
        ostr << "RenderPrep CPU-affinity control skipped";
    }
    mRenderPrepMessage = ostr.str();

    mEnableRenderPrepCpuAffinity = false;
    if (!mCpuIdTbl.empty()) {
        mEnableRenderPrepCpuAffinity = true;
        mNumThreads = std::min(static_cast<unsigned>(mCpuIdTbl.size()),
                               std::thread::hardware_concurrency());
    }

    //------------------------------
    //
    // MCRT CPU affinity
    //
    ostr.str("");
    mEnableMcrtCpuAffinity = false;
    mEnableMcrtCpuAffinityAll = false;
    if (!forceToDisable) {
        mEnableMcrtCpuAffinity = true;
        if (mNumThreads == std::thread::hardware_concurrency()) {
            // We want to use all cores. We activate CPU-affinity control and
            // all MCRT threads are individually attached to the core.
            mEnableMcrtCpuAffinityAll = true;
            ostr << "CPU-affinity control enabled : all : ";
            {
                std::vector<unsigned> cpuIdTbl(std::thread::hardware_concurrency());
                std::iota(cpuIdTbl.begin(), cpuIdTbl.end(), 0);                
                ostr << AffinityManager::showTbl("CPU-Tbl", cpuIdTbl);
            }
        } else if (!mCpuIdTbl.empty()) {
            // We have {CPU,Socket}-Affinity setup.
            ostr << "CPU-affinity control enabled"
                 << " : " << AffinityManager::showTbl("CPU-Tbl", mCpuIdTbl);
        }
    } else {
        ostr << "MCRT-CPU-affinity control disabled : numRenderThreads:" << mNumThreads;
    }
    mMcrtMessage = ostr.str();
}

#endif // end of Not PLATFORM_APPLE

//------------------------------------------------------------------------------------------

static std::shared_ptr<AffinityManager> sAffinityManager;

AffinityManager::AffinityManager(const unsigned desiredNumThreads,
                                 const std::string& cpuAffinityDef,
                                 const std::string& socketAffinityDef,
                                 const std::string& memAffinityDef)
    : mMemManager {std::make_shared<MemoryAffinityManager>(memAffinityDef)}
    , mCpuManager {std::make_shared<CpuAffinityManager>(desiredNumThreads, cpuAffinityDef, socketAffinityDef)}
{
    mMemManager->init(getCpu());
}

// static function    
bool
AffinityManager::init(const unsigned desiredNumThreads,
                      const std::string& autoAffinityDef,
                      const std::string& cpuAffinityDef,
                      const std::string& socketAffinityDef,
                      const std::string& memAffinityDef)
{
    if (sAffinityManager) return true; // already initialized

    if (scene_rdl2::str_util::isTrueStr(autoAffinityDef)) {
        std::string autoCpuAffinityDef;
        std::string autoMemAffinityDef;
        std::string dummy;
        calcAutoAffinityOptions(desiredNumThreads,
                                autoCpuAffinityDef,
                                autoMemAffinityDef);
        sAffinityManager = std::make_shared<AffinityManager>(desiredNumThreads,
                                                             autoCpuAffinityDef,
                                                             dummy,
                                                             autoMemAffinityDef);
    } else {
        sAffinityManager = std::make_shared<AffinityManager>(desiredNumThreads,
                                                             cpuAffinityDef,
                                                             socketAffinityDef,
                                                             memAffinityDef);
    }

    return sAffinityManager.get() != nullptr;
}

// static function
std::shared_ptr<AffinityManager>
AffinityManager::get()
{
    return sAffinityManager;
}

void
AffinityManager::setupLogInfo(std::vector<std::string>& titleTbl,
                              std::vector<std::string>& msgTbl) const
{
    mCpuManager->setupLogInfo(titleTbl, msgTbl);
    mMemManager->setupLogInfo(titleTbl, msgTbl, mCpuManager);
}

std::string
AffinityManager::show() const
{
    auto showMemManager = [&]() -> std::string {
        if (!mMemManager) return "mMemManager is empty";
        return mMemManager->show();
    };
    auto showCpuManager = [&]() -> std::string {
        if (!mCpuManager) return "mCpuManager is empty";
        return mCpuManager->show();
    };

    std::ostringstream ostr;
    ostr << "AffinityManager {\n"
         << scene_rdl2::str_util::addIndent(showMemManager()) << '\n'
         << scene_rdl2::str_util::addIndent(showCpuManager()) << '\n'
         << "}";
    return ostr.str();
}

// static function
std::string
AffinityManager::showTbl(const std::string& msg, const std::vector<unsigned>& tbl)
{
    std::vector<unsigned> workTbl = tbl;
    std::sort(workTbl.begin(), workTbl.end());

    auto showIds = [&] {
        std::string idString;
        int startId {-1}; // initial condition
        int endId {-1}; // initial condition
        auto initRange = [&](const unsigned id) { startId = endId = static_cast<int>(id); };
        auto extendRange = [&](const unsigned id) { endId = static_cast<int>(id); };
        auto flushRangeId = [&] {
            if (!idString.empty()) idString += ',';
            idString += std::to_string(startId);
            if (startId != endId) idString += ('-' + std::to_string(endId));
        };
        for (size_t i = 0; i < workTbl.size(); ++i) {
            if (startId < 0) initRange(workTbl[i]); 
            else if (workTbl[i] == endId + 1) extendRange(workTbl[i]);
            else {
                flushRangeId();
                initRange(workTbl[i]);
            }
        }
        if (startId >= 0) flushRangeId();
        return idString;
    };

    std::ostringstream ostr;
    if (!msg.empty()) ostr << msg << ' ';
    ostr << "(total:" << tbl.size() << ") {" << showIds() << '}';
    return ostr.str();
}

//-----------------------------------------------------------------------------------------

#ifdef PLATFORM_APPLE

// static function
void
AffinityManager::calcAutoAffinityOptions(const unsigned desiredNumThreads,
                                         std::string& cpuAffinityDef,
                                         std::string& memAffinityDef)
{
    cpuAffinityDef = "";
    memAffinityDef = "";
}

#else // Not PLATFORM_APPLE

// static function
void
AffinityManager::calcAutoAffinityOptions(const unsigned desiredNumThreads,
                                         std::string& cpuAffinityDef,
                                         std::string& memAffinityDef)
//
// This function computes CPU and memory affinity option definition strings based on the given
// number of total threads.
//
{
    if (desiredNumThreads == std::thread::hardware_concurrency()) {
        // Try to use full cores with CPU/Mem affinity both on
        cpuAffinityDef = "all";
        memAffinityDef = "on";
        return;
    }

    // Try to access shared memory and get some information regarding other MoonRay process's affinity condition.
    // MOONRAY-5367 (Record MoonRay affinity status into shared memory to share this info with other MoonRay)
}
#endif // end of Not PLATFORM_APPLE

} // namespace mcrt_common
} // namespace moonray
