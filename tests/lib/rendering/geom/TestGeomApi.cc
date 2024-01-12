// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "TestGeomApi.h"
#include "SizeVerifyingAllocator.h"
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>
#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/VertexBuffer.h>

namespace moonray {
namespace geom {
namespace unittest {

void TestGeomApi::setUp() {
}

void TestGeomApi::tearDown() {
}

void TestGeomApi::testLayerAssignmentId()
{
    LayerAssignmentId l00(5);
    LayerAssignmentId l01(7);
    // Test initializer_list constructor.
    LayerAssignmentId l02({2, 4, 6, 8});
    // Test r-value reference vector constructor.
    LayerAssignmentId l03(std::vector<int>({3, 6, 9, 12}));

    CPPUNIT_ASSERT(l00.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l01.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l02.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l03.getType() == LayerAssignmentId::Type::VARYING);

    CPPUNIT_ASSERT(l00.getConstId() == 5);
    CPPUNIT_ASSERT(l01.getConstId() == 7);
    CPPUNIT_ASSERT(l02.getVaryingId() == std::vector<int>({2, 4, 6, 8}));
    CPPUNIT_ASSERT(l03.getVaryingId() == std::vector<int>({3, 6, 9, 12}));

    // Test constant copy construction.
    LayerAssignmentId l04(l00);
    CPPUNIT_ASSERT(l04.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l04.getConstId() == 5);

    // Test varying move construction.
    LayerAssignmentId l05(std::move(l02));
    CPPUNIT_ASSERT(l05.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l05.getVaryingId() == std::vector<int>({2, 4, 6, 8}));

    // Test varying copy construction.
    LayerAssignmentId l06(l03);
    CPPUNIT_ASSERT(l06.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l06.getVaryingId() == std::vector<int>({3, 6, 9, 12}));

    // Swap constant-constant
    std::swap(l00, l01);
    CPPUNIT_ASSERT(l00.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l01.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l00.getConstId() == 7);
    CPPUNIT_ASSERT(l01.getConstId() == 5);

    // Swap varying-varying
    std::swap(l05, l03);
    CPPUNIT_ASSERT(l02.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l05.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l03.getVaryingId() == std::vector<int>({2, 4, 6, 8}));
    CPPUNIT_ASSERT(l05.getVaryingId() == std::vector<int>({3, 6, 9, 12}));

    // Swap constant-varying
    std::swap(l00, l03);
    CPPUNIT_ASSERT(l00.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l03.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l00.getVaryingId() == std::vector<int>({2, 4, 6, 8}));
    CPPUNIT_ASSERT(l03.getConstId() == 7);

    // Swap varying-constant
    std::swap(l05, l01);
    CPPUNIT_ASSERT(l01.getType() == LayerAssignmentId::Type::VARYING);
    CPPUNIT_ASSERT(l05.getType() == LayerAssignmentId::Type::CONSTANT);
    CPPUNIT_ASSERT(l01.getVaryingId() == std::vector<int>({3, 6, 9, 12}));
    CPPUNIT_ASSERT(l05.getConstId() == 5);
}

void TestGeomApi::testPrimitiveAttribute()
{
    using list_type = typename shading::PrimitiveAttribute<int>::list_type;
    shading::PrimitiveAttribute<int> pa0(shading::RATE_VARYING, list_type({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));

    CPPUNIT_ASSERT(pa0.size() == 10);
    CPPUNIT_ASSERT(pa0[0] == 0);
    CPPUNIT_ASSERT(pa0[1] == 1);
    CPPUNIT_ASSERT(pa0[2] == 2);
    CPPUNIT_ASSERT(pa0[3] == 3);
    CPPUNIT_ASSERT(pa0[4] == 4);
    CPPUNIT_ASSERT(pa0[5] == 5);
    CPPUNIT_ASSERT(pa0[6] == 6);
    CPPUNIT_ASSERT(pa0[7] == 7);
    CPPUNIT_ASSERT(pa0[8] == 8);
    CPPUNIT_ASSERT(pa0[9] == 9);

    erase(pa0, 3);
    CPPUNIT_ASSERT(pa0.size() == 9);
    CPPUNIT_ASSERT(pa0[0] == 0);
    CPPUNIT_ASSERT(pa0[1] == 1);
    CPPUNIT_ASSERT(pa0[2] == 2);
    CPPUNIT_ASSERT(pa0[3] == 4);
    CPPUNIT_ASSERT(pa0[4] == 5);
    CPPUNIT_ASSERT(pa0[5] == 6);
    CPPUNIT_ASSERT(pa0[6] == 7);
    CPPUNIT_ASSERT(pa0[7] == 8);
    CPPUNIT_ASSERT(pa0[8] == 9);

    insert(pa0, 0, 3, 42);
    CPPUNIT_ASSERT(pa0.size() == 12);
    CPPUNIT_ASSERT(pa0[ 0] == 42);
    CPPUNIT_ASSERT(pa0[ 1] == 42);
    CPPUNIT_ASSERT(pa0[ 2] == 42);
    CPPUNIT_ASSERT(pa0[ 3] ==  0);
    CPPUNIT_ASSERT(pa0[ 4] ==  1);
    CPPUNIT_ASSERT(pa0[ 5] ==  2);
    CPPUNIT_ASSERT(pa0[ 6] ==  4);
    CPPUNIT_ASSERT(pa0[ 7] ==  5);
    CPPUNIT_ASSERT(pa0[ 8] ==  6);
    CPPUNIT_ASSERT(pa0[ 9] ==  7);
    CPPUNIT_ASSERT(pa0[10] ==  8);
    CPPUNIT_ASSERT(pa0[11] ==  9);

    pa0.erase(pa0.begin(), pa0.begin() + 3);
    CPPUNIT_ASSERT(pa0.size() == 9);
    CPPUNIT_ASSERT(pa0[0] == 0);
    CPPUNIT_ASSERT(pa0[1] == 1);
    CPPUNIT_ASSERT(pa0[2] == 2);
    CPPUNIT_ASSERT(pa0[3] == 4);
    CPPUNIT_ASSERT(pa0[4] == 5);
    CPPUNIT_ASSERT(pa0[5] == 6);
    CPPUNIT_ASSERT(pa0[6] == 7);
    CPPUNIT_ASSERT(pa0[7] == 8);
    CPPUNIT_ASSERT(pa0[8] == 9);

    insert(pa0, 3, 3);
    CPPUNIT_ASSERT(pa0.size() == 10);
    CPPUNIT_ASSERT(pa0[0] == 0);
    CPPUNIT_ASSERT(pa0[1] == 1);
    CPPUNIT_ASSERT(pa0[2] == 2);
    CPPUNIT_ASSERT(pa0[3] == 3);
    CPPUNIT_ASSERT(pa0[4] == 4);
    CPPUNIT_ASSERT(pa0[5] == 5);
    CPPUNIT_ASSERT(pa0[6] == 6);
    CPPUNIT_ASSERT(pa0[7] == 7);
    CPPUNIT_ASSERT(pa0[8] == 8);
    CPPUNIT_ASSERT(pa0[9] == 9);

    insert(pa0, 10, 4, 5);
    CPPUNIT_ASSERT(pa0.size() == 14);
    CPPUNIT_ASSERT(pa0[ 0] ==  0);
    CPPUNIT_ASSERT(pa0[ 1] ==  1);
    CPPUNIT_ASSERT(pa0[ 2] ==  2);
    CPPUNIT_ASSERT(pa0[ 3] ==  3);
    CPPUNIT_ASSERT(pa0[ 4] ==  4);
    CPPUNIT_ASSERT(pa0[ 5] ==  5);
    CPPUNIT_ASSERT(pa0[ 6] ==  6);
    CPPUNIT_ASSERT(pa0[ 7] ==  7);
    CPPUNIT_ASSERT(pa0[ 8] ==  8);
    CPPUNIT_ASSERT(pa0[ 9] ==  9);
    CPPUNIT_ASSERT(pa0[10] ==  5);
    CPPUNIT_ASSERT(pa0[11] ==  5);
    CPPUNIT_ASSERT(pa0[12] ==  5);
    CPPUNIT_ASSERT(pa0[13] ==  5);
}

void TestGeomApi::testVertexBufferAlignment()
{
    VertexBuffer<Vec3fa, InterleavedTraits, scene_rdl2::alloc::AlignedAllocator<Vec3fa, 16>> it0(4, 2);

    it0(0, 0) = Vec3fa( 3.4f,  4.4f,  5.4f, 0.f);
    it0(1, 0) = Vec3fa( 6.4f,  7.4f,  8.4f, 0.f);
    it0(2, 0) = Vec3fa( 9.4f, 10.4f, 11.4f, 0.f);
    it0(3, 0) = Vec3fa(12.4f, 13.4f, 14.4f, 0.f);

    it0(0, 1) = Vec3fa(23.4f, 24.4f, 25.4f, 0.f);
    it0(1, 1) = Vec3fa(26.4f, 27.4f, 28.4f, 0.f);
    it0(2, 1) = Vec3fa(29.4f, 30.4f, 31.4f, 0.f);
    it0(3, 1) = Vec3fa(32.4f, 33.4f, 34.4f, 0.f);

    CPPUNIT_ASSERT(scene_rdl2::alloc::isAligned(it0.data(), 16));

    CPPUNIT_ASSERT(it0.size() == 4);

    for (size_t i = 0; i < it0.size(); ++i) {
        // This only works because our type should be of the correct size to be
        // aligned.
        CPPUNIT_ASSERT(scene_rdl2::alloc::isAligned(&it0(i, 0), 16));
        CPPUNIT_ASSERT(scene_rdl2::alloc::isAligned(&it0(i, 1), 16));
    }
}

void testInterleaved4D2T(const VertexBuffer<Vec3fa, InterleavedTraits>& vb)
{
    const auto s = vb.data_size();
    const auto d = vb.data();

    CPPUNIT_ASSERT(vb.get_time_steps() == 2);

    // How many floats do we have?
    // 2 time samples
    // 4 Vec3fa
    // 4 floats each
    CPPUNIT_ASSERT(s == 2 * 4 * 4);

    CPPUNIT_ASSERT(d[ 0] ==  3.4f);
    CPPUNIT_ASSERT(d[ 1] ==  4.4f);
    CPPUNIT_ASSERT(d[ 2] ==  5.4f);

    CPPUNIT_ASSERT(d[ 4] == 23.4f);
    CPPUNIT_ASSERT(d[ 5] == 24.4f);
    CPPUNIT_ASSERT(d[ 6] == 25.4f);

    CPPUNIT_ASSERT(d[ 8] ==  6.4f);
    CPPUNIT_ASSERT(d[ 9] ==  7.4f);
    CPPUNIT_ASSERT(d[10] ==  8.4f);

    CPPUNIT_ASSERT(d[12] == 26.4f);
    CPPUNIT_ASSERT(d[13] == 27.4f);
    CPPUNIT_ASSERT(d[14] == 28.4f);

    CPPUNIT_ASSERT(d[16] ==  9.4f);
    CPPUNIT_ASSERT(d[17] == 10.4f);
    CPPUNIT_ASSERT(d[18] == 11.4f);

    CPPUNIT_ASSERT(d[20] == 29.4f);
    CPPUNIT_ASSERT(d[21] == 30.4f);
    CPPUNIT_ASSERT(d[22] == 31.4f);

    CPPUNIT_ASSERT(d[24] == 12.4f);
    CPPUNIT_ASSERT(d[25] == 13.4f);
    CPPUNIT_ASSERT(d[26] == 14.4f);

    CPPUNIT_ASSERT(d[28] == 32.4f);
    CPPUNIT_ASSERT(d[29] == 33.4f);
    CPPUNIT_ASSERT(d[30] == 34.4f);

}

void testInterleaved4D1T(const VertexBuffer<Vec3fa, InterleavedTraits>& vb)
{
    const auto s = vb.data_size();
    const auto d = vb.data();

    CPPUNIT_ASSERT(vb.get_time_steps() == 1);

    // How many floats do we have?
    // 1 time sample
    // 4 Vec3fa
    // 4 floats each
    CPPUNIT_ASSERT(s == 1 * 4 * 4);

    CPPUNIT_ASSERT(d[ 0] ==  3.4f);
    CPPUNIT_ASSERT(d[ 1] ==  4.4f);
    CPPUNIT_ASSERT(d[ 2] ==  5.4f);

    CPPUNIT_ASSERT(d[ 4] ==  6.4f);
    CPPUNIT_ASSERT(d[ 5] ==  7.4f);
    CPPUNIT_ASSERT(d[ 6] ==  8.4f);

    CPPUNIT_ASSERT(d[ 8] ==  9.4f);
    CPPUNIT_ASSERT(d[ 9] == 10.4f);
    CPPUNIT_ASSERT(d[10] == 11.4f);

    CPPUNIT_ASSERT(d[12] == 12.4f);
    CPPUNIT_ASSERT(d[13] == 13.4f);
    CPPUNIT_ASSERT(d[14] == 14.4f);
}

template <typename T>
void testInterleaved3D2T(const VertexBuffer<T, InterleavedTraits>& vb)
{
    const auto s = vb.data_size();
    const auto d = vb.data();

    CPPUNIT_ASSERT(vb.get_time_steps() == 2);

    // How many floats do we have?
    // 2 time samples
    // 4 Vec3f
    // 3 floats each
    CPPUNIT_ASSERT(s == 2 * 4 * 3);

    CPPUNIT_ASSERT(d[ 0] ==  3.4f);
    CPPUNIT_ASSERT(d[ 1] ==  4.4f);
    CPPUNIT_ASSERT(d[ 2] ==  5.4f);

    CPPUNIT_ASSERT(d[ 3] == 23.4f);
    CPPUNIT_ASSERT(d[ 4] == 24.4f);
    CPPUNIT_ASSERT(d[ 5] == 25.4f);

    CPPUNIT_ASSERT(d[ 6] ==  6.4f);
    CPPUNIT_ASSERT(d[ 7] ==  7.4f);
    CPPUNIT_ASSERT(d[ 8] ==  8.4f);

    CPPUNIT_ASSERT(d[ 9] == 26.4f);
    CPPUNIT_ASSERT(d[10] == 27.4f);
    CPPUNIT_ASSERT(d[11] == 28.4f);

    CPPUNIT_ASSERT(d[12] ==  9.4f);
    CPPUNIT_ASSERT(d[13] == 10.4f);
    CPPUNIT_ASSERT(d[14] == 11.4f);

    CPPUNIT_ASSERT(d[15] == 29.4f);
    CPPUNIT_ASSERT(d[16] == 30.4f);
    CPPUNIT_ASSERT(d[17] == 31.4f);

    CPPUNIT_ASSERT(d[18] == 12.4f);
    CPPUNIT_ASSERT(d[19] == 13.4f);
    CPPUNIT_ASSERT(d[20] == 14.4f);

    CPPUNIT_ASSERT(d[21] == 32.4f);
    CPPUNIT_ASSERT(d[22] == 33.4f);
    CPPUNIT_ASSERT(d[23] == 34.4f);
}

template <typename T>
void testInterleaved3D1T(const VertexBuffer<T, InterleavedTraits>& vb)
{
    const auto s = vb.data_size();
    const auto d = vb.data();

    CPPUNIT_ASSERT(vb.get_time_steps() == 1);

    // How many floats do we have?
    // 1 time sample
    // 4 Vec3f
    // 3 floats each
    CPPUNIT_ASSERT(s == 1 * 4 * 3);

    CPPUNIT_ASSERT(d[ 0] ==  3.4f);
    CPPUNIT_ASSERT(d[ 1] ==  4.4f);
    CPPUNIT_ASSERT(d[ 2] ==  5.4f);

    CPPUNIT_ASSERT(d[ 3] ==  6.4f);
    CPPUNIT_ASSERT(d[ 4] ==  7.4f);
    CPPUNIT_ASSERT(d[ 5] ==  8.4f);

    CPPUNIT_ASSERT(d[ 6] ==  9.4f);
    CPPUNIT_ASSERT(d[ 7] == 10.4f);
    CPPUNIT_ASSERT(d[ 8] == 11.4f);

    CPPUNIT_ASSERT(d[ 9] == 12.4f);
    CPPUNIT_ASSERT(d[10] == 13.4f);
    CPPUNIT_ASSERT(d[11] == 14.4f);
}

void TestGeomApi::testVertexBufferVec3fa0()
{
    using std::swap; // Allow ADL

    VertexBuffer<Vec3fa, InterleavedTraits> it0(4);
    VertexBuffer<Vec3fa, InterleavedTraits> it1(4, 2);

    it0(0)    = Vec3fa(3.4f, 4.4f, 5.4f, 0.f);
    it0(1)    = Vec3fa(6.4f, 7.4f, 8.4f, 0.f);
    it0(2)    = Vec3fa(9.4f, 10.4f, 11.4f, 0.f);
    it0(3)    = Vec3fa(12.4f, 13.4f, 14.4f, 0.f);

    it1(0, 0) = Vec3fa(3.4f, 4.4f, 5.4f, 0.f);
    it1(1, 0) = Vec3fa(6.4f, 7.4f, 8.4f, 0.f);
    it1(2, 0) = Vec3fa(9.4f, 10.4f, 11.4f, 0.f);
    it1(3, 0) = Vec3fa(12.4f, 13.4f, 14.4f, 0.f);

    it1(0, 1) = Vec3fa(23.4f, 24.4f, 25.4f, 0.f);
    it1(1, 1) = Vec3fa(26.4f, 27.4f, 28.4f, 0.f);
    it1(2, 1) = Vec3fa(29.4f, 30.4f, 31.4f, 0.f);
    it1(3, 1) = Vec3fa(32.4f, 33.4f, 34.4f, 0.f);

    testInterleaved4D1T(it0);
    testInterleaved4D2T(it1);

    swap(it0, it1);

    testInterleaved4D1T(it1);
    testInterleaved4D2T(it0);

    auto it2 = it0.copy();
    auto it3 = it1.copy();

    testInterleaved4D1T(it3);
    testInterleaved4D2T(it2);
}

void TestGeomApi::testVertexBufferVec3fa1()
{
    using std::swap; // Allow ADL

    VertexBuffer<Vec3fa, InterleavedTraits> it0;
    VertexBuffer<Vec3fa, InterleavedTraits> it1(0, 2);

    it0.push_back(Vec3fa( 3.4f,  4.4f,  5.4f, 0.f));
    it0.push_back(Vec3fa( 6.4f,  7.4f,  8.4f, 0.f));
    it0.push_back(Vec3fa( 9.4f, 10.4f, 11.4f, 0.f));
    it0.push_back(Vec3fa(12.4f, 13.4f, 14.4f, 0.f));

    Vec3fa a[2];
    a[0] = Vec3fa( 3.4f,  4.4f,  5.4f, 0.f);
    a[1] = Vec3fa(23.4f, 24.4f, 25.4f, 0.f);
    it1.push_back(a);
    a[0] = Vec3fa( 6.4f,  7.4f,  8.4f, 0.f);
    a[1] = Vec3fa(26.4f, 27.4f, 28.4f, 0.f);
    it1.push_back(a);
    a[0] = Vec3fa( 9.4f, 10.4f, 11.4f, 0.f);
    a[1] = Vec3fa(29.4f, 30.4f, 31.4f, 0.f);
    it1.push_back(a);
    a[0] = Vec3fa(12.4f, 13.4f, 14.4f, 0.f);
    a[1] = Vec3fa(32.4f, 33.4f, 34.4f, 0.f);
    it1.push_back(a);

    testInterleaved4D1T(it0);
    testInterleaved4D2T(it1);

    swap(it0, it1);

    testInterleaved4D1T(it1);
    testInterleaved4D2T(it0);

    auto it2 = it0.copy();
    auto it3 = it1.copy();

    testInterleaved4D1T(it3);
    testInterleaved4D2T(it2);

    it0.shrink_to_fit();
    it1.shrink_to_fit();

    CPPUNIT_ASSERT(it0.capacity() == it0.size());
    CPPUNIT_ASSERT(it1.capacity() == it1.size());

    testInterleaved4D1T(it1);
    testInterleaved4D2T(it0);

}

void TestGeomApi::testVertexBufferVec3f0()
{
    using std::swap; // Allow ADL

    VertexBuffer<Vec3f, InterleavedTraits> it0(4);
    VertexBuffer<Vec3f, InterleavedTraits> it1(4, 2);

    it0(0) = Vec3f(3.4f, 4.4f, 5.4f);
    it0(1) = Vec3f(6.4f, 7.4f, 8.4f);
    it0(2) = Vec3f(9.4f, 10.4f, 11.4f);
    it0(3) = Vec3f(12.4f, 13.4f, 14.4f);

    it1(0, 0) = Vec3f(3.4f, 4.4f, 5.4f);
    it1(1, 0) = Vec3f(6.4f, 7.4f, 8.4f);
    it1(2, 0) = Vec3f(9.4f, 10.4f, 11.4f);
    it1(3, 0) = Vec3f(12.4f, 13.4f, 14.4f);

    it1(0, 1) = Vec3f(23.4f, 24.4f, 25.4f);
    it1(1, 1) = Vec3f(26.4f, 27.4f, 28.4f);
    it1(2, 1) = Vec3f(29.4f, 30.4f, 31.4f);
    it1(3, 1) = Vec3f(32.4f, 33.4f, 34.4f);

    testInterleaved3D1T(it0);
    testInterleaved3D2T(it1);

    swap(it0, it1);

    testInterleaved3D1T(it1);
    testInterleaved3D2T(it0);

    auto it2 = it0.copy();
    auto it3 = it1.copy();

    testInterleaved3D1T(it3);
    testInterleaved3D2T(it2);

    it0.shrink_to_fit();
    it1.shrink_to_fit();

    CPPUNIT_ASSERT(it0.capacity() == it0.size());
    CPPUNIT_ASSERT(it1.capacity() == it1.size());

    testInterleaved3D1T(it1);
    testInterleaved3D2T(it0);
}

void TestGeomApi::testVertexBufferVec3f1()
{
    using std::swap; // Allow ADL

    VertexBuffer<Vec3f, InterleavedTraits> it0;
    VertexBuffer<Vec3f, InterleavedTraits> it1(0, 2);

    Vec3f a[2];
    it0.push_back(Vec3f(3.4f, 4.4f, 5.4f));
    it0.push_back(Vec3f(6.4f, 7.4f, 8.4f));
    it0.push_back(Vec3f(9.4f, 10.4f, 11.4f));
    it0.push_back(Vec3f(12.4f, 13.4f, 14.4f));

    a[0] = Vec3f(3.4f, 4.4f, 5.4f);
    a[1] = Vec3f(23.4f, 24.4f, 25.4f);
    it1.push_back(a);
    a[0] = Vec3f(6.4f, 7.4f, 8.4f);
    a[1] = Vec3f(26.4f, 27.4f, 28.4f);
    it1.push_back(a);
    a[0] = Vec3f(9.4f, 10.4f, 11.4f);
    a[1] = Vec3f(29.4f, 30.4f, 31.4f);
    it1.push_back(a);
    a[0] = Vec3f(12.4f, 13.4f, 14.4f);
    a[1] = Vec3f(32.4f, 33.4f, 34.4f);
    it1.push_back(a);

    testInterleaved3D1T(it0);
    testInterleaved3D2T(it1);

    swap(it0, it1);

    testInterleaved3D1T(it1);
    testInterleaved3D2T(it0);

    auto it2 = it0.copy();
    auto it3 = it1.copy();

    testInterleaved3D1T(it3);
    testInterleaved3D2T(it2);

    it0.shrink_to_fit();
    it1.shrink_to_fit();

    CPPUNIT_ASSERT(it0.capacity() == it0.size());
    CPPUNIT_ASSERT(it1.capacity() == it1.size());

    testInterleaved3D1T(it1);
    testInterleaved3D2T(it0);
}

void TestGeomApi::testVertexBufferVec3f2()
{
    VertexBuffer<Vec3f, InterleavedTraits> it0;
    VertexBuffer<Vec3f, InterleavedTraits> it1(0, 2);

    CPPUNIT_ASSERT(it0.empty());
    CPPUNIT_ASSERT(it1.empty());

    for (int i = 0; i < 1000; ++i) {
        CPPUNIT_ASSERT(it0.size() == i);
        CPPUNIT_ASSERT(it1.size() == i);

        it0.push_back(Vec3f(1.1f, 2.2f, 3.3f));
        Vec3f a[2] = { Vec3f(4.4f, 5.5f, 6.6f), Vec3f(7.7f, 8.8f, 9.9f) };
        it1.push_back(a);
    }

    it0.shrink_to_fit();
    it1.shrink_to_fit();

    CPPUNIT_ASSERT(it0.capacity() == it0.size());
    CPPUNIT_ASSERT(it1.capacity() == it1.size());

    it0.reserve(it0.size() * 2);
    it1.reserve(it0.size() * 2);

    CPPUNIT_ASSERT(it0.capacity() == it0.size() * 2);
    CPPUNIT_ASSERT(it1.capacity() == it1.size() * 2);

    it0.shrink_to_fit();
    it1.shrink_to_fit();

    CPPUNIT_ASSERT(it0.capacity() == it0.size());
    CPPUNIT_ASSERT(it1.capacity() == it1.size());

}

template <template <typename> class Allocator>
void resizeTest()
{
    VertexBuffer<Vec3f, InterleavedTraits, Allocator<Vec3f>> it0(0, 2);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    it0.resize(10);

    CPPUNIT_ASSERT(it0.size() == 10);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    it0.resize(20, Vec3f(1.0f, 2.0f, 3.0f));

    CPPUNIT_ASSERT(it0.size() == 20);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 20);

    for (size_t i = 10; i < it0.size(); ++i) {
        CPPUNIT_ASSERT(it0(i, 0) == Vec3f(1.0f, 2.0f, 3.0f));
        CPPUNIT_ASSERT(it0(i, 1) == Vec3f(1.0f, 2.0f, 3.0f));
    }

    it0.resize(10);

    CPPUNIT_ASSERT(it0.size() == 10);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    Vec3f vals[2];
    vals[0] = Vec3f(8.0f, 7.0f, 6.0f);
    vals[1] = Vec3f(5.0f, 4.0f, 3.0f);
    it0.push_back(vals);

    CPPUNIT_ASSERT(it0(10, 0) == Vec3f(8.0f, 7.0f, 6.0f));
    CPPUNIT_ASSERT(it0(10, 1) == Vec3f(5.0f, 4.0f, 3.0f));
}

template <template <typename> class Allocator>
void resizeTest3fa()
{
    VertexBuffer<Vec3fa, InterleavedTraits, Allocator<Vec3fa>> it0(0, 2);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    it0.resize(10);

    CPPUNIT_ASSERT(it0.size() == 10);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    it0.resize(20, Vec3fa(1.0f, 2.0f, 3.0f, 0.f));

    CPPUNIT_ASSERT(it0.size() == 20);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 20);

    for (size_t i = 10; i < it0.size(); ++i) {
        CPPUNIT_ASSERT(it0(i, 0) == Vec3fa(1.0f, 2.0f, 3.0f, 0.f));
        CPPUNIT_ASSERT(it0(i, 1) == Vec3fa(1.0f, 2.0f, 3.0f, 0.f));
    }

    it0.resize(10);

    CPPUNIT_ASSERT(it0.size() == 10);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    Vec3fa vals[2];
    vals[0] = Vec3fa(8.0f, 7.0f, 6.0f, 0.f);
    vals[1] = Vec3fa(5.0f, 4.0f, 3.0f, 0.f);
    it0.push_back(vals);

    CPPUNIT_ASSERT(it0(10, 0) == Vec3fa(8.0f, 7.0f, 6.0f, 0.f));
    CPPUNIT_ASSERT(it0(10, 1) == Vec3fa(5.0f, 4.0f, 3.0f, 0.f));
}

template <template <typename> class Allocator>
void clearTest()
{
    VertexBuffer<Vec3f, InterleavedTraits, Allocator<Vec3f>> it0(0, 2);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    it0.resize(10);

    CPPUNIT_ASSERT(it0.size() == 10);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    it0.clear();

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    it0.shrink_to_fit();

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());
    CPPUNIT_ASSERT(it0.capacity() == 0);

    Vec3f vals[2];
    vals[0] = Vec3f(8.0f, 7.0f, 6.0f);
    vals[1] = Vec3f(5.0f, 4.0f, 3.0f);
    it0.push_back(vals);

    CPPUNIT_ASSERT(it0.size() == 1);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 1);

    CPPUNIT_ASSERT(it0(0, 0) == Vec3f(8.0f, 7.0f, 6.0f));
    CPPUNIT_ASSERT(it0(0, 1) == Vec3f(5.0f, 4.0f, 3.0f));
}

template <template <typename> class Allocator>
void clearTest3fa()
{
    VertexBuffer<Vec3fa, InterleavedTraits, Allocator<Vec3fa>> it0(0, 2);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    it0.resize(10);

    CPPUNIT_ASSERT(it0.size() == 10);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    it0.clear();

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 10);

    it0.shrink_to_fit();

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());
    CPPUNIT_ASSERT(it0.capacity() == 0);

    Vec3fa vals[2];
    vals[0] = Vec3fa(8.0f, 7.0f, 6.0f, 0.f);
    vals[1] = Vec3fa(5.0f, 4.0f, 3.0f, 0.f);
    it0.push_back(vals);

    CPPUNIT_ASSERT(it0.size() == 1);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 1);

    CPPUNIT_ASSERT(it0(0, 0) == Vec3fa(8.0f, 7.0f, 6.0f, 0.f));
    CPPUNIT_ASSERT(it0(0, 1) == Vec3fa(5.0f, 4.0f, 3.0f, 0.f));
}

template <template <typename> class Allocator>
void appendTest()
{
    VertexBuffer<Vec3f, InterleavedTraits, Allocator<Vec3f>> it0(0, 2);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    VertexBuffer<Vec3f, InterleavedTraits, Allocator<Vec3f>> it1(0, 2);

    CPPUNIT_ASSERT(it1.size() == 0);
    CPPUNIT_ASSERT(it1.empty());

    it0.append(it1);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    CPPUNIT_ASSERT(it1.size() == 0);
    CPPUNIT_ASSERT(it1.empty());

    it1.resize(20, Vec3f(1.0f, 2.0f, 3.0f));

    CPPUNIT_ASSERT(it1.size() == 20);
    CPPUNIT_ASSERT(!it1.empty());
    CPPUNIT_ASSERT(it1.capacity() >= 20);

    it0.append(it1);

    CPPUNIT_ASSERT(it0.size() == 20);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 20);

    for (size_t i = 0; i < it0.size(); ++i) {
        CPPUNIT_ASSERT(it0(i, 0) == Vec3f(1.0f, 2.0f, 3.0f));
        CPPUNIT_ASSERT(it0(i, 1) == Vec3f(1.0f, 2.0f, 3.0f));
    }

    it0.append(it1);

    CPPUNIT_ASSERT(it0.size() == 40);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 40);

    for (size_t i = 0; i < it0.size(); ++i) {
        CPPUNIT_ASSERT(it0(i, 0) == Vec3f(1.0f, 2.0f, 3.0f));
        CPPUNIT_ASSERT(it0(i, 1) == Vec3f(1.0f, 2.0f, 3.0f));
    }
}

template <template <typename> class Allocator>
void appendTest3fa()
{
    VertexBuffer<Vec3fa, InterleavedTraits, Allocator<Vec3fa>> it0(0, 2);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    VertexBuffer<Vec3fa, InterleavedTraits, Allocator<Vec3fa>> it1(0, 2);

    CPPUNIT_ASSERT(it1.size() == 0);
    CPPUNIT_ASSERT(it1.empty());

    it0.append(it1);

    CPPUNIT_ASSERT(it0.size() == 0);
    CPPUNIT_ASSERT(it0.empty());

    CPPUNIT_ASSERT(it1.size() == 0);
    CPPUNIT_ASSERT(it1.empty());

    it1.resize(20, Vec3fa(1.0f, 2.0f, 3.0f, 0.f));

    CPPUNIT_ASSERT(it1.size() == 20);
    CPPUNIT_ASSERT(!it1.empty());
    CPPUNIT_ASSERT(it1.capacity() >= 20);

    it0.append(it1);

    CPPUNIT_ASSERT(it0.size() == 20);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 20);

    for (size_t i = 0; i < it0.size(); ++i) {
        CPPUNIT_ASSERT(it0(i, 0) == Vec3fa(1.0f, 2.0f, 3.0f, 0.f));
        CPPUNIT_ASSERT(it0(i, 1) == Vec3fa(1.0f, 2.0f, 3.0f, 0.f));
    }

    it0.append(it1);

    CPPUNIT_ASSERT(it0.size() == 40);
    CPPUNIT_ASSERT(!it0.empty());
    CPPUNIT_ASSERT(it0.capacity() >= 40);

    for (size_t i = 0; i < it0.size(); ++i) {
        CPPUNIT_ASSERT(it0(i, 0) == Vec3fa(1.0f, 2.0f, 3.0f, 0.f));
        CPPUNIT_ASSERT(it0(i, 1) == Vec3fa(1.0f, 2.0f, 3.0f, 0.f));
    }
}

void TestGeomApi::testVertexBufferResize()
{
    resizeTest3fa<std::allocator>();
    resizeTest<std::allocator>();
    resizeTest3fa<SizeVerifyingAllocator>();
    resizeTest<SizeVerifyingAllocator>();
}

void TestGeomApi::testVertexBufferClear()
{
    clearTest3fa<std::allocator>();
    clearTest<std::allocator>();
    clearTest3fa<SizeVerifyingAllocator>();
    clearTest<SizeVerifyingAllocator>();
}

void TestGeomApi::testVertexBufferAppend()
{
    appendTest3fa<std::allocator>();
    appendTest<std::allocator>();
    appendTest3fa<SizeVerifyingAllocator>();
    appendTest<SizeVerifyingAllocator>();
}

} // namespace unittest
} // namespace geom
} // namespace moonray

