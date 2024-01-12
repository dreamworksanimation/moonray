// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "TestAosSoa.h"
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/util/Random.h>

using namespace scene_rdl2;

namespace moonray {
namespace mcrt_common {

namespace {

// Generate random sample data. The actual data is unimportant, just that it's unordered.
void
generateAOSData( util::Random& rng,
                 unsigned numElems,
                 AOSData *dstAosData,
                 SortOrder *dstSortOrder )
{
    uint32_t counter = 0;

    for (unsigned i = 0; i < numElems; ++i) {

        counter = rng.getNextUInt();
        dstSortOrder[i].mSortKey = counter;
        dstSortOrder[i].mElemIdx = i;

        for (unsigned j = 0; j < sizeof(AOSData) / sizeof(uint32_t); ++j) {
            dstAosData[i].mData[j] = counter++;   // Ok to wrap around.
        }
    }
}

template <typename SOABlockType>
bool
validateAOSToSOAResults( unsigned numElems,
                         const AOSData *aosData,
                         const SOABlockType *soaBlocks,
                         const SortOrder *refSortOrder,
                         scene_rdl2::alloc::Arena *arena )
{
    const unsigned vlen = SOABlockType::LANE_WIDTH;
    const unsigned vlenMask = vlen - 1;
    const unsigned vlenShift = scene_rdl2::math::log2i(vlen);

    SCOPED_MEM(arena);

    SortOrder *sortOrder = arena->allocArray<SortOrder>(numElems, CACHE_LINE_SIZE);
    memcpy(sortOrder, refSortOrder, sizeof(SortOrder) * numElems);

    std::sort(sortOrder, sortOrder + numElems, [](const SortOrder &a, const SortOrder &b) -> bool {
        return a.mSortKey < b.mSortKey;
    });

    for (unsigned i = 0; i < numElems; ++i) {

        unsigned blockIdx = i >> vlenShift;
        unsigned laneIdx = i & vlenMask;

        // Validate data in object.
        uint32_t baseVal = aosData[sortOrder[i].mElemIdx].mData[0];
        for (unsigned j = 0; j < sizeof(AOSData) / sizeof(uint32_t); ++j) {
            if (soaBlocks[blockIdx].mData[j][laneIdx] != baseVal + j) {
                return false;
            }
        }
    }

    // Check we've correctly smeared results over trailing SOA entries.
    if ((numElems & vlenMask) != 0) {

        const SOABlockType &finalSoa = soaBlocks[numElems >> vlenShift];
        unsigned finalLaneIdx = (numElems - 1) & vlenMask;

        for (unsigned i = 0; i < sizeof(AOSData) / sizeof(uint32_t); ++i) {
            uint32_t ref = finalSoa.mData[i][finalLaneIdx];
            for (unsigned j = finalLaneIdx + 1; j < vlen; ++j) {
                if (finalSoa.mData[i][j] != ref) {
                    return false;
                }
            }
        }
    }

    return true;
}

template <typename SOABlockType>
bool
validateSOAToAOSResults( unsigned numElems,
                         const uint32_t *indices,
                         const SOABlockType *soaBlocks,
                         const AOSData *aosData )
{
    const unsigned vlen = SOABlockType::LANE_WIDTH;
    const unsigned vlenMask = vlen - 1;
    const unsigned vlenShift = scene_rdl2::math::log2i(vlen);

    for (unsigned i = 0; i < numElems; ++i) {

        unsigned elemIdx = indices[i];

        unsigned blockIdx = elemIdx >> vlenShift;
        unsigned laneIdx = elemIdx & vlenMask;

        const SOABlockType &soa_block = soaBlocks[blockIdx];
        const AOSData &aos = aosData[i];

        // Validate data in object.
        for (unsigned j = 0; j < sizeof(AOSData) / sizeof(uint32_t); ++j) {
            if (soa_block.mData[j][laneIdx] != aos.mData[j]) {
                return false;
            }
        }
    }

    return true;
}

void
displayStats(const char *heading, const Ticks &ticks)
{
    double milSortTicks = double(ticks.mPostSort - ticks.mPreSort) / 1000000.0;
    double milTransposeTicks = double(ticks.mPostTranspose - ticks.mPostSort) / 1000000.0;

    fprintf(stderr,
            "\"%s\" results are valid:\n"
            "          sorting (millions of ticks) = %13.6f\n"
            "    transposition (millions of ticks) = %13.6f\n"
            "                                         ------------\n"
            "                                        %13.6f\n",
            heading,
            milSortTicks,
            milTransposeTicks,
            milSortTicks + milTransposeTicks);
}


// 1. Compute reference results for AOS->SOA.
// 2. Compute optimized results for AOS->SOA and compare to ref results.
// 3. Do optimized SOA->AOS conversion and compare to input results.
template <typename SOABlockType, typename REF_A2S, typename OPT_A2S, typename OPT_S2A>
bool
runTests( REF_A2S doRefAOSToSOA,
          OPT_A2S doOptAOSToSOA,
          OPT_S2A doOptSOAToAOS,
          unsigned numElems,
          const AOSData *__restrict refAosData,
          const SortOrder *refSortOrder,
          scene_rdl2::util::Random& rng,
          scene_rdl2::alloc::Arena *arena )
{
    SCOPED_MEM(arena);

    const unsigned vlen = SOABlockType::LANE_WIDTH;
    const unsigned numSoaBlocks = scene_rdl2::util::alignUp(numElems, vlen) / vlen;

    SortOrder *sortOrder = arena->allocArray<SortOrder>(numElems, CACHE_LINE_SIZE);

    //
    // Step 1. Compute reference results for AOS->SOA.
    //

    SOABlockType *refSoaBlocks = arena->allocArray<SOABlockType>(numSoaBlocks, CACHE_LINE_SIZE);

    memcpy(sortOrder, refSortOrder, sizeof(SortOrder) * numElems);

    Ticks refTicks;
    doRefAOSToSOA(numElems, refAosData, refSoaBlocks, sortOrder, arena, &refTicks);

    if (!validateAOSToSOAResults(numElems, refAosData, refSoaBlocks, refSortOrder, arena)) {
        fprintf(stderr, "Error! Reference AOS->SOA results are invalid!\n");
        CPPUNIT_ASSERT(0);
        return false;
    } else {
        displayStats("Reference AOS->SOA", refTicks);
    }

    //
    // Step 2. Compute optimized results for AOS->SOA.
    //

    SOABlockType *optSoaBlocks = arena->allocArray<SOABlockType>(numSoaBlocks, CACHE_LINE_SIZE);

    memcpy(sortOrder, refSortOrder, sizeof(SortOrder) * numElems);

    Ticks optTicks;
    doOptAOSToSOA(numElems, refAosData, optSoaBlocks, sortOrder, arena, &optTicks);

    if (!validateAOSToSOAResults(numElems, refAosData, optSoaBlocks, refSortOrder, arena)) {
        fprintf(stderr, "Error! Optimized AOS->SOA results are invalid!\n");
        CPPUNIT_ASSERT(0);
        return false;
    } else {
        displayStats("Optimized AOS->SOA", optTicks);
    }

    //
    // Step 3. Do optimized SOA->AOS conversion and compare to inputs.
    //

    // Randomly cull out half of the elements before converting SOA to AOS.
    unsigned numIndices = 0;
    uint32_t *indices = arena->allocArray<uint32_t>(numElems, CACHE_LINE_SIZE);
    std::bernoulli_distribution dist(0.5);
    for (unsigned i = 0; i < numElems; ++i) {
        if (dist(rng)) {
            indices[numIndices++] = i; 
        }
    }

    AOSData *optAosDataMem = arena->allocArray<AOSData>(numIndices, CACHE_LINE_SIZE);
    AOSData **optAosData = arena->allocArray<AOSData *>(numIndices, CACHE_LINE_SIZE);
    for (unsigned i = 0; i < numIndices; ++i) {
        optAosData[i] = &optAosDataMem[i];
    }

    doOptSOAToAOS(numIndices, optSoaBlocks, optAosData, indices, arena, &optTicks);

    if (!validateSOAToAOSResults(numIndices, indices, optSoaBlocks, optAosDataMem)) {
        fprintf(stderr, "Error! Optimized SOA->AOS results are invalid!\n");
        CPPUNIT_ASSERT(0);
        return false;
    } else { 
        displayStats("Optimized SOA->AOS", optTicks);
    }

    return true;
}

}

//----------------------------------------------------------------------------

TestAosSoa::TestAosSoa() :
    mArenaBlockPool(scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE)),
    mRNG(1234),
    mNumElems(NUM_AOS_ELEMS),
    mRefAosData(nullptr),
    mRefSortOrder(nullptr)
{
    mArena.init(mArenaBlockPool.get());

    // Generate AOS data.
    mRefAosData = mArena.allocArray<AOSData>(mNumElems, CACHE_LINE_SIZE);
    mRefSortOrder = mArena.allocArray<SortOrder>(mNumElems, CACHE_LINE_SIZE);
    generateAOSData(mRNG, mNumElems, mRefAosData, mRefSortOrder);
}

TestAosSoa::~TestAosSoa()
{
    mArena.cleanUp();
    mArenaBlockPool = nullptr;
}

void TestAosSoa::testAVX()
{
    // Run AVX tests.
    fprintf(stderr, "\n");
    fprintf(stderr, "Running AVX tests\n");
    fprintf(stderr, "-----------------\n");
    runTests<SOABlock8>( doAVXRefAOSToSOA,
                         doAVXOptAOSToSOA,
                         doAVXOptSOAToAOS,
                         mNumElems,
                         mRefAosData,
                         mRefSortOrder,
                         mRNG,
                         &mArena );
}

//----------------------------------------------------------------------------

} // namespace mcrt_common
} // namespace moonray

CPPUNIT_TEST_SUITE_REGISTRATION(moonray::mcrt_common::TestAosSoa);

