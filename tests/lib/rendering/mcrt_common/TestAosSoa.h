// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// 
//
#pragma once
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>

// The number of objects we want to transpose.
#define NUM_AOS_ELEMS           17011

// In bytes. Size must be a multiple of SIMD_MEMORY_ALIGNMENT for the given ISA.
#define SIZE_OF_AOS_OBJECT      160

MNRY_STATIC_ASSERT((SIZE_OF_AOS_OBJECT % SIMD_MEMORY_ALIGNMENT) == 0);

namespace moonray {
namespace mcrt_common {

// A single element of AOS data.
struct AOSData
{
    uint32_t mData[SIZE_OF_AOS_OBJECT / sizeof(uint32_t)];
};
MNRY_STATIC_ASSERT(sizeof(AOSData) == SIZE_OF_AOS_OBJECT);

// 4 wide SOA block.
struct ALIGN(SSE_SIMD_MEMORY_ALIGNMENT) SOABlock4
{
    enum { LANE_WIDTH = 4 };
    uint32_t mData[SIZE_OF_AOS_OBJECT / sizeof(uint32_t)][4];
};
MNRY_STATIC_ASSERT(sizeof(SOABlock4) == sizeof(AOSData) * 4);

// 8 wide SOA block.
struct ALIGN(AVX_SIMD_MEMORY_ALIGNMENT) SOABlock8
{
    enum { LANE_WIDTH = 8 };
    uint32_t mData[SIZE_OF_AOS_OBJECT / sizeof(uint32_t)][8];
};
MNRY_STATIC_ASSERT(sizeof(SOABlock8) == sizeof(AOSData) * 8);

// 16 wide SOA block.
struct ALIGN(AVX512_SIMD_MEMORY_ALIGNMENT) SOABlock16
{
    enum { LANE_WIDTH = 16 };
    uint32_t mData[SIZE_OF_AOS_OBJECT / sizeof(uint32_t)][16];
};
MNRY_STATIC_ASSERT(sizeof(SOABlock16) == sizeof(AOSData) * 16);

// Contains information about how to sort the AOSData.
struct SortOrder
{
    uint32_t    mSortKey;
    uint32_t    mElemIdx;
};

// For profiling.
struct Ticks
{
    uint64_t    mPreSort;
    uint64_t    mPostSort;
    uint64_t    mPostTranspose;
};

//
// Prototypes.
//

void doAVXRefAOSToSOA( unsigned numElems,
                       const AOSData *__restrict aosData,
                       SOABlock8 *__restrict soaBlocks,
                       SortOrder *sortOrder,
                       scene_rdl2::alloc::Arena *arena,
                       Ticks *ticks );

void doAVXOptAOSToSOA( unsigned numElems,
                       const AOSData *__restrict aosData,
                       SOABlock8 *__restrict soaBlocks,
                       SortOrder *sortOrder,
                       scene_rdl2::alloc::Arena *arena,
                       Ticks *ticks );

void doAVXOptSOAToAOS( unsigned numElems,
                       const SOABlock8 *__restrict soaBlocks,
                       AOSData **__restrict aosData,
                       uint32_t *indices,
                       scene_rdl2::alloc::Arena *arena,
                       Ticks *ticks );

//----------------------------------------------------------------------------

class TestAosSoa : public CppUnit::TestFixture
{
public:
    TestAosSoa();
    ~TestAosSoa();

    CPPUNIT_TEST_SUITE(TestAosSoa);

#ifdef __AVX__
    CPPUNIT_TEST(testAVX);
#endif

    CPPUNIT_TEST_SUITE_END();

private:
    void testAVX();

    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> mArenaBlockPool;
    scene_rdl2::alloc::Arena mArena;
    scene_rdl2::util::Random mRNG;

    unsigned mNumElems;
    AOSData *mRefAosData;
    SortOrder *mRefSortOrder;
};

//----------------------------------------------------------------------------

} // namespace mcrt_common
} // namespace moonray

