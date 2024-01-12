// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestPrimitiveAttribute.cc

#include "TestPrimitiveAttribute.h"
#include "TestPrimitiveAttribute_ispc_stubs.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

using namespace moonray;
using namespace scene_rdl2::math;
using namespace moonray::geom;
using moonray::shading::unittest::TestPrimitiveAttribute;

namespace {

size_t keysPerStride = 7;
size_t nStrides = 4;

inline char
getBool(size_t stride, size_t lane)
{
    return (stride + lane) % 2;
}

inline int
getInt(size_t stride, size_t lane)
{
    return stride + lane;
}

inline float
getFloat(size_t stride, size_t lane)
{
    return stride - lane;
}

inline Vec2f
getVec2f(size_t stride, size_t lane)
{
    return Vec2f(stride + lane, stride + lane + 1);
}

inline Vec3f
getVec3f(size_t stride, size_t lane)
{
    return Vec3f(stride + lane, stride + lane + 1, stride + lane + 2);
}

inline Mat4f
getMat4f(size_t stride, size_t lane)
{
    return Mat4f(stride + lane + 0, stride + lane + 1, stride + lane + 2, stride + lane + 3,
                 stride + lane + 4, stride + lane + 5, stride + lane + 6, stride + lane + 7,
                 stride + lane + 8, stride + lane + 9, stride + lane + 10, stride + lane + 11,
                 stride + lane + 12, stride + lane + 13, stride + lane + 14, stride + lane + 15);
}

inline Color
getColor(size_t stride, size_t lane)
{
    return Color(stride - lane, stride - lane + 1, stride - lane + 2);
}

}

void
TestPrimitiveAttribute::setUp()
{
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    mTls = MNRY_VERIFY(tls->mShadingTls.get());
    scene_rdl2::alloc::Arena *arena = mTls->mArena;

    const int nkeys = keysPerStride * nStrides;

    // stride layout:
    //   bool:  stored as uint8_t, padded to 32 bits
    //   int:   stored as int32_t, no padding
    //   float: stored as float, no padding
    //   Vec2f: stored as Vec2f, no padding
    //   Vec3f: stored as Vec3f, no padding
    //   Mat4f: stored as Mat4f, no padding
    //   Color: stored as Color, no padding
    int *offsets = arena->allocArray<int>(nkeys);
    mTls->mAttributeOffsets = offsets;
    const size_t strideSize = sizeof(int32_t) + sizeof(int32_t) + sizeof(float) +
                              sizeof(Vec2f) + sizeof(Vec3f) + sizeof(Mat4f) + sizeof(Color);

    for (size_t stride = 0; stride < nStrides; ++stride) {
        const int sk = stride * keysPerStride;

        offsets[sk + 0] = stride * strideSize;
        offsets[sk + 1] = offsets[sk + 0] + sizeof(int32_t);
        offsets[sk + 2] = offsets[sk + 1] + sizeof(int32_t);
        offsets[sk + 3] = offsets[sk + 2] + sizeof(float);
        offsets[sk + 4] = offsets[sk + 3] + sizeof(Vec2f);
        offsets[sk + 5] = offsets[sk + 4] + sizeof(Vec3f);
        offsets[sk + 6] = offsets[sk + 5] + sizeof(Mat4f);
    }
    const size_t dataSize = strideSize * nStrides;

    uint8_t *data[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        data[lane] = arena->allocArray<uint8_t>(dataSize);

        // mimics Address64_set from Util.isph
        intptr_t addr = (intptr_t) data[lane];
        mStatev.mData.mHigh[lane] = (addr >> 32);
        mStatev.mData.mLow[lane] = (addr & 0xffffffff);
    }

    for (size_t stride = 0; stride < nStrides; ++stride) {
        for (size_t lane = 0; lane < VLEN; ++lane) {

            uint8_t *d = data[lane];
            const int sk = stride * keysPerStride;

            // fill data out with test data
            *reinterpret_cast<char  *>(d + offsets[sk + 0]) = getBool(stride, lane);
            *reinterpret_cast<int   *>(d + offsets[sk + 1]) = getInt(stride, lane);
            *reinterpret_cast<float *>(d + offsets[sk + 2]) = getFloat(stride, lane);
            *reinterpret_cast<Vec2f *>(d + offsets[sk + 3]) = getVec2f(stride, lane);
            *reinterpret_cast<Vec3f *>(d + offsets[sk + 4]) = getVec3f(stride, lane);
            *reinterpret_cast<Mat4f *>(d + offsets[sk + 5]) = getMat4f(stride, lane);
            *reinterpret_cast<Color *>(d + offsets[sk + 6]) = getColor(stride, lane);
        }
    }
}

void
TestPrimitiveAttribute::tearDown()
{
}

void
TestPrimitiveAttribute::testGetBoolAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 0;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride;
    }
    int8_t uniResult[VLEN];
    int8_t varResult[VLEN];
    ispc::testGetBoolAttribute(tls, &mStatev, uniKey, varKey, uniResult, varResult);
    for (size_t lane = 0; lane < VLEN; ++lane) {
        CPPUNIT_ASSERT(uniResult[lane] == getBool(0, lane));
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        CPPUNIT_ASSERT(varResult[lane] == getBool(lane % nStrides, lane));
    }
}

void
TestPrimitiveAttribute::testGetIntAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 1;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride + 1;
    }
    int32_t uniResult[VLEN];
    int32_t varResult[VLEN];
    ispc::testGetIntAttribute(tls, &mStatev, uniKey, varKey, uniResult, varResult);
    for (size_t lane = 0; lane < VLEN; ++lane) {
        CPPUNIT_ASSERT(uniResult[lane] == getInt(0, lane));
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        CPPUNIT_ASSERT(varResult[lane] == getInt(lane % nStrides, lane));
    }
}

void
TestPrimitiveAttribute::testGetFloatAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 2;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride + 2;
    }
    float uniResult[VLEN];
    float varResult[VLEN];
    ispc::testGetFloatAttribute(tls, &mStatev, uniKey, varKey, uniResult, varResult);
    for (size_t lane = 0; lane < VLEN; ++lane) {
        CPPUNIT_ASSERT(uniResult[lane] == getFloat(0, lane));
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        CPPUNIT_ASSERT(varResult[lane] == getFloat(lane % nStrides, lane));
    }
}

void
TestPrimitiveAttribute::testGetVec2fAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 3;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride + 3;
    }
    float uniResultX[VLEN];
    float uniResultY[VLEN];
    float varResultX[VLEN];
    float varResultY[VLEN];
    ispc::testGetVec2fAttribute(tls, &mStatev, uniKey, varKey,
                                uniResultX, uniResultY,
                                varResultX, varResultY);
    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Vec2f expected = getVec2f(0, lane);
        CPPUNIT_ASSERT(uniResultX[lane] == expected.x);
        CPPUNIT_ASSERT(uniResultY[lane] == expected.y);
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Vec2f expected = getVec2f(lane % nStrides, lane);
        CPPUNIT_ASSERT(varResultX[lane] == expected.x);
        CPPUNIT_ASSERT(varResultY[lane] == expected.y);
    }
}

void
TestPrimitiveAttribute::testGetVec3fAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 4;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride + 4;
    }
    float uniResultX[VLEN];
    float uniResultY[VLEN];
    float uniResultZ[VLEN];
    float varResultX[VLEN];
    float varResultY[VLEN];
    float varResultZ[VLEN];
    ispc::testGetVec3fAttribute(tls, &mStatev, uniKey, varKey,
                                uniResultX, uniResultY, uniResultZ,
                                varResultX, varResultY, varResultZ);
    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Vec3f expected = getVec3f(0, lane);
        CPPUNIT_ASSERT(uniResultX[lane] == expected.x);
        CPPUNIT_ASSERT(uniResultY[lane] == expected.y);
        CPPUNIT_ASSERT(uniResultZ[lane] == expected.z);
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Vec3f expected = getVec3f(lane % nStrides, lane);
        CPPUNIT_ASSERT(varResultX[lane] == expected.x);
        CPPUNIT_ASSERT(varResultY[lane] == expected.y);
        CPPUNIT_ASSERT(varResultZ[lane] == expected.z);
    }
}

void
TestPrimitiveAttribute::testGetMat4fAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 5;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride + 5;
    }
    float uniResultVXX[VLEN];
    float uniResultVXY[VLEN];
    float uniResultVXZ[VLEN];
    float uniResultVXW[VLEN];
    float uniResultVYX[VLEN];
    float uniResultVYY[VLEN];
    float uniResultVYZ[VLEN];
    float uniResultVYW[VLEN];
    float uniResultVZX[VLEN];
    float uniResultVZY[VLEN];
    float uniResultVZZ[VLEN];
    float uniResultVZW[VLEN];
    float uniResultVWX[VLEN];
    float uniResultVWY[VLEN];
    float uniResultVWZ[VLEN];
    float uniResultVWW[VLEN];

    float varResultVXX[VLEN];
    float varResultVXY[VLEN];
    float varResultVXZ[VLEN];
    float varResultVXW[VLEN];
    float varResultVYX[VLEN];
    float varResultVYY[VLEN];
    float varResultVYZ[VLEN];
    float varResultVYW[VLEN];
    float varResultVZX[VLEN];
    float varResultVZY[VLEN];
    float varResultVZZ[VLEN];
    float varResultVZW[VLEN];
    float varResultVWX[VLEN];
    float varResultVWY[VLEN];
    float varResultVWZ[VLEN];
    float varResultVWW[VLEN];

    ispc::testGetMat4fAttribute(tls, &mStatev, uniKey, varKey,
                                uniResultVXX, uniResultVXY, uniResultVXZ, uniResultVXW,
                                uniResultVYX, uniResultVYY, uniResultVYZ, uniResultVYW,
                                uniResultVZX, uniResultVZY, uniResultVZZ, uniResultVZW,
                                uniResultVWX, uniResultVWY, uniResultVWZ, uniResultVWW,
                                varResultVXX, varResultVXY, varResultVXZ, varResultVXW,
                                varResultVYX, varResultVYY, varResultVYZ, varResultVYW,
                                varResultVZX, varResultVZY, varResultVZZ, varResultVZW,
                                varResultVWX, varResultVWY, varResultVWZ, varResultVWW);

    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Mat4f expected = getMat4f(0, lane);
        CPPUNIT_ASSERT(uniResultVXX[lane] == expected.vx.x);
        CPPUNIT_ASSERT(uniResultVXY[lane] == expected.vx.y);
        CPPUNIT_ASSERT(uniResultVXZ[lane] == expected.vx.z);
        CPPUNIT_ASSERT(uniResultVXW[lane] == expected.vx.w);
        CPPUNIT_ASSERT(uniResultVYX[lane] == expected.vy.x);
        CPPUNIT_ASSERT(uniResultVYY[lane] == expected.vy.y);
        CPPUNIT_ASSERT(uniResultVYZ[lane] == expected.vy.z);
        CPPUNIT_ASSERT(uniResultVYW[lane] == expected.vy.w);
        CPPUNIT_ASSERT(uniResultVZX[lane] == expected.vz.x);
        CPPUNIT_ASSERT(uniResultVZY[lane] == expected.vz.y);
        CPPUNIT_ASSERT(uniResultVZZ[lane] == expected.vz.z);
        CPPUNIT_ASSERT(uniResultVZW[lane] == expected.vz.w);
        CPPUNIT_ASSERT(uniResultVWX[lane] == expected.vw.x);
        CPPUNIT_ASSERT(uniResultVWY[lane] == expected.vw.y);
        CPPUNIT_ASSERT(uniResultVWZ[lane] == expected.vw.z);
        CPPUNIT_ASSERT(uniResultVWW[lane] == expected.vw.w);
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Mat4f expected = getMat4f(lane % nStrides, lane);
        CPPUNIT_ASSERT(varResultVXX[lane] == expected.vx.x);
        CPPUNIT_ASSERT(varResultVXY[lane] == expected.vx.y);
        CPPUNIT_ASSERT(varResultVXZ[lane] == expected.vx.z);
        CPPUNIT_ASSERT(varResultVXW[lane] == expected.vx.w);
        CPPUNIT_ASSERT(varResultVYX[lane] == expected.vy.x);
        CPPUNIT_ASSERT(varResultVYY[lane] == expected.vy.y);
        CPPUNIT_ASSERT(varResultVYZ[lane] == expected.vy.z);
        CPPUNIT_ASSERT(varResultVYW[lane] == expected.vy.w);
        CPPUNIT_ASSERT(varResultVZX[lane] == expected.vz.x);
        CPPUNIT_ASSERT(varResultVZY[lane] == expected.vz.y);
        CPPUNIT_ASSERT(varResultVZZ[lane] == expected.vz.z);
        CPPUNIT_ASSERT(varResultVZW[lane] == expected.vz.w);
        CPPUNIT_ASSERT(varResultVWX[lane] == expected.vw.x);
        CPPUNIT_ASSERT(varResultVWY[lane] == expected.vw.y);
        CPPUNIT_ASSERT(varResultVWZ[lane] == expected.vw.z);
        CPPUNIT_ASSERT(varResultVWW[lane] == expected.vw.w);
    }
}

void
TestPrimitiveAttribute::testGetColorAttribute()
{
    ispc::ShadingTLState *tls = reinterpret_cast<ispc::ShadingTLState *>(mTls);
    const int uniKey = 6;
    int varKey[VLEN];
    for (size_t lane = 0; lane < VLEN; ++lane) {
        varKey[lane] = (lane % nStrides) * keysPerStride + 6;
    }
    float uniResultR[VLEN];
    float uniResultG[VLEN];
    float uniResultB[VLEN];
    float varResultR[VLEN];
    float varResultG[VLEN];
    float varResultB[VLEN];
    ispc::testGetColorAttribute(tls, &mStatev, uniKey, varKey,
                                uniResultR, uniResultG, uniResultB,
                                varResultR, varResultG, varResultB);
    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Color expected = getColor(0, lane);
        CPPUNIT_ASSERT(uniResultR[lane] == expected.r);
        CPPUNIT_ASSERT(uniResultG[lane] == expected.g);
        CPPUNIT_ASSERT(uniResultB[lane] == expected.b);
    }

    for (size_t lane = 0; lane < VLEN; ++lane) {
        const Color expected = getColor(lane % nStrides, lane);
        CPPUNIT_ASSERT(varResultR[lane] == expected.r);
        CPPUNIT_ASSERT(varResultG[lane] == expected.g);
        CPPUNIT_ASSERT(varResultB[lane] == expected.b);
    }
}

