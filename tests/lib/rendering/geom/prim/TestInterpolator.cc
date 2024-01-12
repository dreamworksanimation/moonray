// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestInterpolator
/// $Id$
///

#include "TestInterpolator.h"

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/geom/prim/BezierSpanChains.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <scene_rdl2/common/math/Math.h>

#include <memory>
#include <vector>

namespace moonray {
namespace geom {
namespace unittest {

using namespace moonray::shading;

// taken from BezierSpanChains since CubicSplineInterpolator constructor
// needs w0, w1, w2, w3 as input parameters
template<typename ParamType>
static __forceinline void evalWeights(const ParamType& t,
        ParamType& w0, ParamType& w1, ParamType& w2, ParamType& w3)
{
    const ParamType s = ParamType(scene_rdl2::math::one) - t;
    w0 = s * s * s;
    w1 = ParamType(3.0f) * s * s * t;
    w2 = ParamType(3.0f) * s * t * t;
    w3 = t * t * t;
}

void TestInterpolator::setUp() {
    TestAttributes::init();
}

void TestInterpolator::tearDown() {
}

template <typename T>
static bool verifyWithEqual(const TypedAttributeKey<T>& key, 
    const shading::Interpolator& interpolator, const T& canonical) {
    T result;
    interpolator.interpolate(key, (char*)(&result));
    return result == canonical;
}

template <typename T>
static bool verifyResult(const TypedAttributeKey<T>& key, 
    const shading::Interpolator& interpolator, const T& canonical) {
    T result;
    interpolator.interpolate(key, (char*)(&result));
    return scene_rdl2::math::isEqual(result, canonical);
}

template <>
bool verifyResult(const TypedAttributeKey<bool>& key, 
    const shading::Interpolator& interpolator, const bool& canonical) {
    return verifyWithEqual(key, interpolator, canonical);
}

template <>
bool verifyResult(const TypedAttributeKey<int>& key, 
    const shading::Interpolator& interpolator, const int& canonical) {
    return verifyWithEqual(key, interpolator, canonical);
}

template <>
bool verifyResult(const TypedAttributeKey<long>& key, 
    const shading::Interpolator& interpolator, const long& canonical) {
    return verifyWithEqual(key, interpolator, canonical);
}

template <>
bool verifyResult(const TypedAttributeKey<std::string>& key, 
    const shading::Interpolator& interpolator, const std::string& canonical) {
    std::string* result;
    interpolator.interpolate(key, (char*)(&result));
    return *result == canonical;
}

template <>
bool verifyResult(const TypedAttributeKey<scene_rdl2::math::Color4>& key, 
    const shading::Interpolator& interpolator, const scene_rdl2::math::Color4& canonical) {
    scene_rdl2::math::Color4 result;
    interpolator.interpolate(key, (char*)(&result));
    return moonray::geom::unittest::isEqual(result, canonical);
}

template <typename T>
static  __forceinline T weightSum(
    const T* values, const int* indexes, const float* weights, int n) {
    T result(scene_rdl2::math::zero);
    for (int i = 0; i < n; ++i) {
        result = result + (T)(weights[i] * values[indexes[i]]);
    }
    return result;
}

template <typename T>
static  __forceinline T weightSumMB(
    const T* values0, const T* values1, const int* indexes,
    const float* weights, int n, float t) {
    T result(scene_rdl2::math::zero);
    for (int i = 0; i < n; ++i) {
        result = result +
            (T)(weights[i] * ((1.0f - t) * values0[indexes[i]] +
            t * values1[indexes[i]]));
    }
    return result;
}

void TestInterpolator::testCurvesConstant()
{
    PrimitiveAttributeTable table;
    bool constantBool = mRNG.randomBool();
    table.addAttribute(TestAttributes::sTestBool0, RATE_CONSTANT,
        std::vector<bool>{constantBool});
    int constantInt = mRNG.randomInt();
    table.addAttribute(TestAttributes::sTestInt0, RATE_CONSTANT,
        std::vector<int>{constantInt});
    long constantLong = mRNG.randomLong();
    table.addAttribute(TestAttributes::sTestLong0, RATE_CONSTANT,
        std::vector<long>{constantLong});
    float constantFloat = mRNG.randomFloat();
    table.addAttribute(TestAttributes::sTestFloat0, RATE_CONSTANT,
        std::vector<float>{constantFloat});
    std::string constantString("constant testing string");
    table.addAttribute(TestAttributes::sTestString0, RATE_CONSTANT,
        {constantString});
    scene_rdl2::math::Color constantColor = mRNG.randomColor();
    table.addAttribute(TestAttributes::sTestColor0, RATE_CONSTANT,
        {constantColor});
    scene_rdl2::math::Color4 constantRGBA = mRNG.randomColor4();
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_CONSTANT,
        {constantRGBA});
    scene_rdl2::math::Vec2f constantVec2f = mRNG.randomVec2f();
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_CONSTANT,
        {constantVec2f});
    scene_rdl2::math::Vec3f constantVec3f = mRNG.randomVec3f();
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_CONSTANT,
        {constantVec3f});
    scene_rdl2::math::Mat4f constantMat4f = mRNG.randomMat4f();
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_CONSTANT,
        {constantMat4f});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), 0));

    int chainIndex = 0;
    float u = 0.0f;
    float w0, w1, w2, w3;
    evalWeights(u, w0, w1, w2, w3);
    internal::CubicSplineInterpolator interpolator(attr.get(), 0,
        chainIndex, 0, 0, u, 0, w0, w1, w2, w3);
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestBool0,
        interpolator, constantBool));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestInt0,
        interpolator, constantInt));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestLong0,
        interpolator, constantLong));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
        interpolator, constantFloat));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestString0,
        interpolator, constantString));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
        interpolator, constantColor));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
        interpolator, constantRGBA));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
        interpolator, constantVec2f));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
        interpolator, constantVec3f));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
        interpolator, constantMat4f));
}

void TestInterpolator::testCurvesConstantMB()
{
    PrimitiveAttributeTable table;

    float constantFloat0 = mRNG.randomFloat();
    float constantFloat1 = mRNG.randomFloat();
    std::vector<std::vector<float>> constantFloat(2);
    constantFloat[0] = {constantFloat0};
    constantFloat[1] = {constantFloat1};
    table.addAttribute(TestAttributes::sTestFloat0, RATE_CONSTANT,
        std::move(constantFloat));

    scene_rdl2::math::Color constantColor0 = mRNG.randomColor();
    scene_rdl2::math::Color constantColor1 = mRNG.randomColor();
    std::vector<std::vector<scene_rdl2::math::Color>> constantColor(2);
    constantColor[0] = {constantColor0};
    constantColor[1] = {constantColor1};
    table.addAttribute(TestAttributes::sTestColor0, RATE_CONSTANT,
        std::move(constantColor));

    scene_rdl2::math::Color4 constantRGBA0 = mRNG.randomColor4();
    scene_rdl2::math::Color4 constantRGBA1 = mRNG.randomColor4();
    std::vector<std::vector<scene_rdl2::math::Color4>> constantRGBA(2);
    constantRGBA[0] = {constantRGBA0};
    constantRGBA[1] = {constantRGBA1};
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_CONSTANT,
        std::move(constantRGBA));

    scene_rdl2::math::Vec2f constantVec2f0 = mRNG.randomVec2f();
    scene_rdl2::math::Vec2f constantVec2f1 = mRNG.randomVec2f();
    std::vector<std::vector<scene_rdl2::math::Vec2f>> constantVec2f(2);
    constantVec2f[0] = {constantVec2f0};
    constantVec2f[1] = {constantVec2f1};
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_CONSTANT,
        std::move(constantVec2f));

    scene_rdl2::math::Vec3f constantVec3f0 = mRNG.randomVec3f();
    scene_rdl2::math::Vec3f constantVec3f1 = mRNG.randomVec3f();
    std::vector<std::vector<scene_rdl2::math::Vec3f>> constantVec3f(2);
    constantVec3f[0] = {constantVec3f0};
    constantVec3f[1] = {constantVec3f1};
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_CONSTANT,
        std::move(constantVec3f));

    scene_rdl2::math::Mat4f constantMat4f0 = mRNG.randomMat4f();
    scene_rdl2::math::Mat4f constantMat4f1 = mRNG.randomMat4f();
    std::vector<std::vector<scene_rdl2::math::Mat4f>> constantMat4f(2);
    constantMat4f[0] = {constantMat4f0};
    constantMat4f[1] = {constantMat4f1};
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_CONSTANT,
        std::move(constantMat4f));

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), 0));

    int chainIndex = 0;
    float u = 0.0f;
    float w0, w1, w2, w3;
    evalWeights(u, w0, w1, w2, w3);
    float time = mRNG.randomFloat();
    internal::CubicSplineInterpolator interpolator(attr.get(), time,
        chainIndex, 0, 0, u, 0, w0, w1, w2, w3);
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0, interpolator,
        (1.0f - time) * constantFloat0 + time * constantFloat1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0, interpolator,
        (1.0f - time) * constantColor0 + time * constantColor1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0, interpolator,
        (1.0f - time) * constantRGBA0 + time * constantRGBA1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0, interpolator,
        (1.0f - time) * constantVec2f0 + time * constantVec2f1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0, interpolator,
        (1.0f - time) * constantVec3f0 + time * constantVec3f1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0, interpolator,
        (1.0f - time) * constantMat4f0 + time * constantMat4f1));
}

void TestInterpolator::testCurvesUniform()
{
    const size_t nCurves = 3;
    std::vector<bool> uniformBool;
    std::vector<int> uniformInt;
    std::vector<long> uniformLong;
    std::vector<float> uniformFloat;
    std::vector<std::string> uniformString;
    std::vector<scene_rdl2::math::Color> uniformColor;
    std::vector<scene_rdl2::math::Color4> uniformRGBA;
    std::vector<scene_rdl2::math::Vec2f> uniformVec2f;
    std::vector<scene_rdl2::math::Vec3f> uniformVec3f;
    std::vector<scene_rdl2::math::Mat4f> uniformMat4f;
    for (size_t i = 0; i < nCurves; ++i) {
        uniformBool.push_back(mRNG.randomBool());
        uniformInt.push_back(mRNG.randomInt());
        uniformLong.push_back(mRNG.randomLong());
        uniformFloat.push_back(mRNG.randomFloat());
        uniformString.push_back(std::string("uniform test string ") +
            std::to_string(i));
        uniformColor.push_back(mRNG.randomColor());
        uniformRGBA.push_back(mRNG.randomColor4());
        uniformVec2f.push_back(mRNG.randomVec2f());
        uniformVec3f.push_back(mRNG.randomVec3f());
        uniformMat4f.push_back(mRNG.randomMat4f());
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestBool0, RATE_UNIFORM,
        std::vector<bool>(uniformBool));
    table.addAttribute(TestAttributes::sTestInt0, RATE_UNIFORM,
        std::vector<int>(uniformInt));
    table.addAttribute(TestAttributes::sTestLong0, RATE_UNIFORM,
        std::vector<long>(uniformLong));
    table.addAttribute(TestAttributes::sTestFloat0, RATE_UNIFORM,
        std::vector<float>(uniformFloat));
    table.addAttribute(TestAttributes::sTestString0, RATE_UNIFORM,
        std::vector<std::string>(uniformString));
    table.addAttribute(TestAttributes::sTestColor0, RATE_UNIFORM,
        std::vector<scene_rdl2::math::Color>(uniformColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_UNIFORM,
        std::vector<scene_rdl2::math::Color4>(uniformRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_UNIFORM,
        std::vector<scene_rdl2::math::Vec2f>(uniformVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_UNIFORM,
        std::vector<scene_rdl2::math::Vec3f>(uniformVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_UNIFORM,
        std::vector<scene_rdl2::math::Mat4f>(uniformMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nCurves, 0,
        std::vector<size_t>(), 0));

    for (size_t i = 0; i < nCurves; ++i) {
        float u = 0.0f;
        float w0, w1, w2, w3;
        evalWeights(u, w0, w1, w2, w3);
        internal::CubicSplineInterpolator interpolator(attr.get(), 0,
            i, 0, 0, u, 0, w0, w1, w2, w3);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestBool0,
            interpolator,(bool)uniformBool[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestInt0,
            interpolator, uniformInt[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestLong0,
            interpolator, uniformLong[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator, uniformFloat[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestString0,
            interpolator, uniformString[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator, uniformColor[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator, uniformRGBA[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator, uniformVec2f[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator, uniformVec3f[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator, uniformMat4f[i]));
    }
}

void TestInterpolator::testCurvesUniformMB()
{
    const size_t nCurves = 3;
    const size_t nTimes = 2;
    std::vector<std::vector<float>> uniformFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> uniformColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> uniformRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> uniformVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> uniformVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> uniformMat4f(nTimes);
    for(size_t t = 0; t < nTimes; ++t) {
        for (size_t i = 0; i < nCurves; ++i) {
            uniformFloat[t].push_back(mRNG.randomFloat());
            uniformColor[t].push_back(mRNG.randomColor());
            uniformRGBA[t].push_back(mRNG.randomColor4());
            uniformVec2f[t].push_back(mRNG.randomVec2f());
            uniformVec3f[t].push_back(mRNG.randomVec3f());
            uniformMat4f[t].push_back(mRNG.randomMat4f());
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_UNIFORM,
        std::vector<std::vector<float>>(uniformFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Color>>(uniformColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Color4>>(uniformRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(uniformVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(uniformVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(uniformMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nCurves, 0,
        std::vector<size_t>(), 0));

    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nCurves; ++i) {
        float u = 0.0f;
        float w0, w1, w2, w3;
        evalWeights(u, w0, w1, w2, w3);
        internal::CubicSplineInterpolator interpolator(attr.get(), t,
            i, 0, 0, u, 0, w0, w1, w2, w3);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator,
            (1.0f - t) * uniformFloat[0][i] + t * uniformFloat[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator,
            (1.0f - t) * uniformColor[0][i] + t * uniformColor[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator,
            (1.0f - t) * uniformRGBA[0][i] + t * uniformRGBA[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator,
            (1.0f - t) * uniformVec2f[0][i] + t * uniformVec2f[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator,
            (1.0f - t) * uniformVec3f[0][i] + t * uniformVec3f[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator,
            (1.0f - t) * uniformMat4f[0][i] + t * uniformMat4f[1][i]));
    }
}

void TestInterpolator::testCurvesVarying()
{
    // varying     : should supply sum(nSpansi + 1) values 
    const size_t nCurves = 3;
    const size_t nSpans[] = {7, 5, 6};

    std::vector<float> varyingFloat;
    std::vector<scene_rdl2::math::Color> varyingColor;
    std::vector<scene_rdl2::math::Color4> varyingRGBA;
    std::vector<scene_rdl2::math::Vec2f> varyingVec2f;
    std::vector<scene_rdl2::math::Vec3f> varyingVec3f;
    std::vector<scene_rdl2::math::Mat4f> varyingMat4f;

    size_t nVaryings = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i] + 1; ++j) {
            varyingFloat.push_back(mRNG.randomFloat());
            varyingColor.push_back(mRNG.randomColor());
            varyingRGBA.push_back(mRNG.randomColor4());
            varyingVec2f.push_back(mRNG.randomVec2f());
            varyingVec3f.push_back(mRNG.randomVec3f());
            varyingMat4f.push_back(mRNG.randomMat4f());
            nVaryings += 1;
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VARYING,
        std::vector<float>(varyingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VARYING,
        std::vector<scene_rdl2::math::Color>(varyingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VARYING,
        std::vector<scene_rdl2::math::Color4>(varyingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VARYING,
        std::vector<scene_rdl2::math::Vec2f>(varyingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VARYING,
        std::vector<scene_rdl2::math::Vec3f>(varyingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VARYING,
        std::vector<scene_rdl2::math::Mat4f>(varyingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, nVaryings,
        std::vector<size_t>(), 0));
    int varyingOffset = 0;
    int vertexOffset = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i]; ++j) {
            float u = mRNG.randomFloat();
            float w0, w1, w2, w3;
            evalWeights(u, w0, w1, w2, w3);
            int varyingIndex = varyingOffset + j;
            int faceVaryingIndex = j;
            int vertexIndex = vertexOffset + 3 * j;
            internal::CubicSplineInterpolator interpolator(attr.get(),
                    0, i, varyingIndex, faceVaryingIndex, u,
                    vertexIndex, w0, w1, w2, w3);
            float w[2] = {1.0f - u, u};
            int v[2] = {varyingIndex, varyingIndex + 1};
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
                interpolator,
                weightSum(&varyingFloat[0], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
                interpolator,
                weightSum(&varyingColor[0], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
                interpolator,
                weightSum(&varyingRGBA[0], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
                interpolator,
                weightSum(&varyingVec2f[0], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
                interpolator,
                weightSum(&varyingVec3f[0], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
                interpolator,
                weightSum(&varyingMat4f[0], v, w, 2)));
        }
        varyingOffset += nSpans[i] + 1;
        vertexOffset += nSpans[i] * 3 + 1;
    }
}

void TestInterpolator::testCurvesVaryingMB()
{
    // varying     : should supply sum(nSpansi + 1) values
    const size_t nCurves = 3;
    const size_t nSpans[] = {7, 5, 6};
    const size_t nTimes = 2;
    std::vector<std::vector<float>> varyingFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> varyingColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> varyingRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> varyingVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> varyingVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> varyingMat4f(nTimes);

    size_t nVaryings = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i] + 1; ++j) {
            for (size_t t = 0; t < nTimes; ++t) {
                varyingFloat[t].push_back(mRNG.randomFloat());
                varyingColor[t].push_back(mRNG.randomColor());
                varyingRGBA[t].push_back(mRNG.randomColor4());
                varyingVec2f[t].push_back(mRNG.randomVec2f());
                varyingVec3f[t].push_back(mRNG.randomVec3f());
                varyingMat4f[t].push_back(mRNG.randomMat4f());
            }
            nVaryings += 1;
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VARYING,
        std::vector<std::vector<float>>(varyingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color>>(varyingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color4>>(varyingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(varyingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(varyingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(varyingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, nVaryings,
        std::vector<size_t>(), 0));

    int varyingOffset = 0;
    int vertexOffset = 0;
    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i]; ++j) {
            float u = mRNG.randomFloat();
            float w0, w1, w2, w3;
            evalWeights(u, w0, w1, w2, w3);
            int varyingIndex = varyingOffset + j;
            int faceVaryingIndex = j;
            int vertexIndex = vertexOffset + 3 * j;
            internal::CubicSplineInterpolator interpolator(attr.get(),
                    t, i, varyingIndex, faceVaryingIndex, u,
                    vertexIndex, w0, w1, w2, w3);
            float w[2] = {1.0f - u, u};
            int v[2] = {varyingIndex, varyingIndex + 1};
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
                interpolator, weightSumMB(
                &varyingFloat[0][0], &varyingFloat[1][0], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
                interpolator, weightSumMB(
                &varyingColor[0][0], &varyingColor[1][0], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
                interpolator, weightSumMB(
                &varyingRGBA[0][0], &varyingRGBA[1][0], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
                interpolator, weightSumMB(
                &varyingVec2f[0][0], &varyingVec2f[1][0], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
                interpolator, weightSumMB(
                &varyingVec3f[0][0], &varyingVec3f[1][0], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
                interpolator, weightSumMB(
                &varyingMat4f[0][0], &varyingMat4f[1][0], v, w, 2, t)));
        }
        varyingOffset += nSpans[i] + 1;
        vertexOffset += nSpans[i] * 3 + 1;
    }
}

void TestInterpolator::testCurvesFaceVarying()
{

    // facevarying : should supply sum(nSpansi + 1) values
    const size_t nCurves = 3;
    const std::vector<size_t> nSpans{7, 5, 6};
    std::vector<size_t> nFaceVaryings;

    std::vector<float> faceVaryingFloat;
    std::vector<scene_rdl2::math::Color> faceVaryingColor;
    std::vector<scene_rdl2::math::Color4> faceVaryingRGBA;
    std::vector<scene_rdl2::math::Vec2f> faceVaryingVec2f;
    std::vector<scene_rdl2::math::Vec3f> faceVaryingVec3f;
    std::vector<scene_rdl2::math::Mat4f> faceVaryingMat4f;
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i] + 1; ++j) {
            faceVaryingFloat.push_back(mRNG.randomFloat());
            faceVaryingColor.push_back(mRNG.randomColor());
            faceVaryingRGBA.push_back(mRNG.randomColor4());
            faceVaryingVec2f.push_back(mRNG.randomVec2f());
            faceVaryingVec3f.push_back(mRNG.randomVec3f());
            faceVaryingMat4f.push_back(mRNG.randomMat4f());
        }
        nFaceVaryings.push_back(nSpans[i] + 1);
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        std::vector<float>(faceVaryingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Color>(faceVaryingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Color4>(faceVaryingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Vec2f>(faceVaryingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Vec3f>(faceVaryingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Mat4f>(faceVaryingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nCurves, 0, nFaceVaryings, 0));

    int varyingOffset = 0;
    int vertexOffset = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i]; ++j) {
            float u = mRNG.randomFloat();
            float w0, w1, w2, w3;
            evalWeights(u, w0, w1, w2, w3);
            int varyingIndex = varyingOffset + j;
            int faceVaryingIndex = j;
            int vertexIndex = vertexOffset + 3 * j;
            internal::CubicSplineInterpolator interpolator(attr.get(),
                    0, i, varyingIndex, faceVaryingIndex, u,
                    vertexIndex, w0, w1, w2, w3);
            float w[2] = {1.0f - u, u};
            int v[2] = {(int)j, (int)j + 1};
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
                interpolator,
                weightSum(&faceVaryingFloat[varyingOffset], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
                interpolator,
                weightSum(&faceVaryingColor[varyingOffset], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
                interpolator,
                weightSum(&faceVaryingRGBA[varyingOffset], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
                interpolator,
                weightSum(&faceVaryingVec2f[varyingOffset], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
                interpolator,
                weightSum(&faceVaryingVec3f[varyingOffset], v, w, 2)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
                interpolator,
                weightSum(&faceVaryingMat4f[varyingOffset], v, w, 2)));
        }
        varyingOffset += nSpans[i] + 1;
        vertexOffset += nSpans[i] * 3 + 1;
    }
}

void TestInterpolator::testCurvesFaceVaryingMB()
{

    // facevarying : should supply sum(nSpansi + 1) values
    const size_t nCurves = 3;
    const std::vector<size_t> nSpans{7, 5, 6};
    std::vector<size_t> nFaceVaryings;
    const size_t nTimes = 2;
    std::vector<std::vector<float>> faceVaryingFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> faceVaryingColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> faceVaryingRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> faceVaryingVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> faceVaryingVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> faceVaryingMat4f(nTimes);
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i] + 1; ++j) {
            for (size_t t = 0; t < nTimes; ++t) {
                faceVaryingFloat[t].push_back(mRNG.randomFloat());
                faceVaryingColor[t].push_back(mRNG.randomColor());
                faceVaryingRGBA[t].push_back(mRNG.randomColor4());
                faceVaryingVec2f[t].push_back(mRNG.randomVec2f());
                faceVaryingVec3f[t].push_back(mRNG.randomVec3f());
                faceVaryingMat4f[t].push_back(mRNG.randomMat4f());
            }
        }
        nFaceVaryings.push_back(nSpans[i] + 1);
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        std::vector<std::vector<float>>(faceVaryingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color>>(faceVaryingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color4>>(faceVaryingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(faceVaryingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(faceVaryingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(faceVaryingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nCurves, 0, nFaceVaryings, 0));

    int varyingOffset = 0;
    int vertexOffset = 0;
    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i]; ++j) {
            float u = mRNG.randomFloat();
            float w0, w1, w2, w3;
            evalWeights(u, w0, w1, w2, w3);
            int varyingIndex = varyingOffset + j;
            int faceVaryingIndex = j;
            int vertexIndex = vertexOffset + 3 * j;
            internal::CubicSplineInterpolator interpolator(attr.get(),
                    t, i, varyingIndex, faceVaryingIndex, u,
                    vertexIndex, w0, w1, w2, w3);
            float w[2] = {1.0f - u, u};
            int v[2] = {(int)j, (int)j + 1};
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
                interpolator,
                weightSumMB(&faceVaryingFloat[0][varyingOffset],
                &faceVaryingFloat[1][varyingOffset], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
                interpolator,
                weightSumMB(&faceVaryingColor[0][varyingOffset],
                &faceVaryingColor[1][varyingOffset], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
                interpolator,
                weightSumMB(&faceVaryingRGBA[0][varyingOffset],
                &faceVaryingRGBA[1][varyingOffset], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
                interpolator,
                weightSumMB(&faceVaryingVec2f[0][varyingOffset],
                &faceVaryingVec2f[1][varyingOffset], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
                interpolator,
                weightSumMB(&faceVaryingVec3f[0][varyingOffset],
                &faceVaryingVec3f[1][varyingOffset], v, w, 2, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
                interpolator,
                weightSumMB(&faceVaryingMat4f[0][varyingOffset],
                &faceVaryingMat4f[1][varyingOffset], v, w, 2, t)));
        }
        varyingOffset += nSpans[i] + 1;
        vertexOffset += nSpans[i] * 3 + 1;
    }
}

void TestInterpolator::testCurvesVertex()
{
    // vertex      : should supply sum(nVerticesi) values
    const size_t nCurves = 3;
    const size_t nSpans[] = {7, 5, 6};

    std::vector<float> vertexFloat;
    std::vector<scene_rdl2::math::Color> vertexColor;
    std::vector<scene_rdl2::math::Color4> vertexRGBA;
    std::vector<scene_rdl2::math::Vec2f> vertexVec2f;
    std::vector<scene_rdl2::math::Vec3f> vertexVec3f;
    std::vector<scene_rdl2::math::Mat4f> vertexMat4f;

    size_t nVertices = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        size_t nVerticesi = 3 * nSpans[i] + 1;
        for (size_t j = 0; j < nVerticesi; ++j) {
            vertexFloat.push_back(mRNG.randomFloat());
            vertexColor.push_back(mRNG.randomColor());
            vertexRGBA.push_back(mRNG.randomColor4());
            vertexVec2f.push_back(mRNG.randomVec2f());
            vertexVec3f.push_back(mRNG.randomVec3f());
            vertexMat4f.push_back(mRNG.randomMat4f());
        }
        nVertices += nVerticesi;
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VERTEX,
        std::vector<float>(vertexFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Color>(vertexColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Color4>(vertexRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Vec2f>(vertexVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Vec3f>(vertexVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Mat4f>(vertexMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), nVertices));

    int varyingOffset = 0;
    int vertexOffset = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i]; ++j) {
            float u = mRNG.randomFloat();
            float w0, w1, w2, w3;
            evalWeights(u, w0, w1, w2, w3);
            int varyingIndex = varyingOffset + j;
            int faceVaryingIndex = j;
            int vertexIndex = vertexOffset + 3 * j;
            internal::CubicSplineInterpolator interpolator(attr.get(),
                    0, i, varyingIndex, faceVaryingIndex, u,
                    vertexIndex, w0, w1, w2, w3);
            float w[4] = {w0, w1, w2, w3};
            int v[4] = {vertexIndex, vertexIndex + 1,
                vertexIndex + 2, vertexIndex + 3};
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
                interpolator,
                weightSum(&vertexFloat[0], v, w, 4)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
                interpolator,
                weightSum(&vertexColor[0], v, w, 4)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
                interpolator,
                weightSum(&vertexRGBA[0], v, w, 4)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
                interpolator,
                weightSum(&vertexVec2f[0], v, w, 4)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
                interpolator,
                weightSum(&vertexVec3f[0], v, w, 4)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
                interpolator,
                weightSum(&vertexMat4f[0], v, w, 4)));
        }
        varyingOffset += nSpans[i] + 1;
        vertexOffset += nSpans[i] * 3 + 1;
    }
}

void TestInterpolator::testCurvesVertexMB()
{
    // vertex      : should supply sum(nVerticesi) values
    const size_t nCurves = 3;
    const size_t nSpans[] = {7, 5, 6};
    size_t nTimes = 2;
    std::vector<std::vector<float>> vertexFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> vertexColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> vertexRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> vertexVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> vertexVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> vertexMat4f(nTimes);

    size_t nVertices = 0;
    for (size_t i = 0; i < nCurves; ++i) {
        size_t nVerticesi = 3 * nSpans[i] + 1;
        for (size_t j = 0; j < nVerticesi; ++j) {
            for (size_t t = 0; t < nTimes; ++t) {
                vertexFloat[t].push_back(mRNG.randomFloat());
                vertexColor[t].push_back(mRNG.randomColor());
                vertexRGBA[t].push_back(mRNG.randomColor4());
                vertexVec2f[t].push_back(mRNG.randomVec2f());
                vertexVec3f[t].push_back(mRNG.randomVec3f());
                vertexMat4f[t].push_back(mRNG.randomMat4f());
            }
        }
        nVertices += nVerticesi;
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VERTEX,
        std::vector<std::vector<float>>(vertexFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Color>>(vertexColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Color4>>(vertexRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(vertexVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(vertexVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(vertexMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), nVertices));

    int varyingOffset = 0;
    int vertexOffset = 0;
    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nCurves; ++i) {
        for (size_t j = 0; j < nSpans[i]; ++j) {
            float u = mRNG.randomFloat();
            float w0, w1, w2, w3;
            evalWeights(u, w0, w1, w2, w3);
            int varyingIndex = varyingOffset + j;
            int faceVaryingIndex = j;
            int vertexIndex = vertexOffset + 3 * j;
            internal::CubicSplineInterpolator interpolator(attr.get(),
                    t, i, varyingIndex, faceVaryingIndex, u,
                    vertexIndex, w0, w1, w2, w3);
            float w[4] = {w0, w1, w2, w3};
            int v[4] = {vertexIndex, vertexIndex + 1,
                vertexIndex + 2, vertexIndex + 3};
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
                interpolator,
                weightSumMB(&vertexFloat[0][0], &vertexFloat[1][0],
                v, w, 4, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
                interpolator,
                weightSumMB(&vertexColor[0][0], &vertexColor[1][0],
                v, w, 4, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
                interpolator,
                weightSumMB(&vertexRGBA[0][0], &vertexRGBA[1][0],
                v, w, 4, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
                interpolator,
                weightSumMB(&vertexVec2f[0][0], &vertexVec2f[1][0],
                v, w, 4, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
                interpolator,
                weightSumMB(&vertexVec3f[0][0], &vertexVec3f[1][0],
                v, w, 4, t)));
            CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
                interpolator,
                weightSumMB(&vertexMat4f[0][0], &vertexMat4f[1][0],
                v, w, 4, t)));
        }
        varyingOffset += nSpans[i] + 1;
        vertexOffset += nSpans[i] * 3 + 1;
    }
}

void TestInterpolator::testMeshConstant()
{
    PrimitiveAttributeTable table;
    bool constantBool = mRNG.randomBool();
    table.addAttribute(TestAttributes::sTestBool0, RATE_CONSTANT,
        std::vector<bool>{constantBool});
    int constantInt = mRNG.randomInt();
    table.addAttribute(TestAttributes::sTestInt0, RATE_CONSTANT,
        std::vector<int>{constantInt});
    long constantLong = mRNG.randomLong();
    table.addAttribute(TestAttributes::sTestLong0, RATE_CONSTANT,
        std::vector<long>{constantLong});
    float constantFloat = mRNG.randomFloat();
    table.addAttribute(TestAttributes::sTestFloat0, RATE_CONSTANT,
        std::vector<float>{constantFloat});
    std::string constantString("constant testing string");
    table.addAttribute(TestAttributes::sTestString0, RATE_CONSTANT,
        {constantString});
    scene_rdl2::math::Color constantColor = mRNG.randomColor();
    table.addAttribute(TestAttributes::sTestColor0, RATE_CONSTANT,
        {constantColor});
    scene_rdl2::math::Color4 constantRGBA = mRNG.randomColor4();
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_CONSTANT,
        {constantRGBA});
    scene_rdl2::math::Vec2f constantVec2f = mRNG.randomVec2f();
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_CONSTANT,
        {constantVec2f});
    scene_rdl2::math::Vec3f constantVec3f = mRNG.randomVec3f();
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_CONSTANT,
        {constantVec3f});
    scene_rdl2::math::Mat4f constantMat4f = mRNG.randomMat4f();
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_CONSTANT,
        {constantMat4f});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), 0));

    int face = 0;
    int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
    float w0 = 0.25f, w1 = 0.25f, w2 = 0.25f, w3 = 0.25f;
    internal::MeshInterpolator interpolator(attr.get(), 0, 0,
        face, v0, v1, v2, v3, w0, w1, w2, w3,
        face, v0, v1, v2, w0, w1, w2);

    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestBool0,
        interpolator, constantBool));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestInt0,
        interpolator, constantInt));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestLong0,
        interpolator, constantLong));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
        interpolator, constantFloat));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
        interpolator, constantColor));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
        interpolator, constantRGBA));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
        interpolator, constantVec2f));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
        interpolator, constantVec3f));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
        interpolator, constantMat4f));
}

void TestInterpolator::testMeshConstantMB()
{
    PrimitiveAttributeTable table;
    float constantFloat0 = mRNG.randomFloat();
    float constantFloat1 = mRNG.randomFloat();
    std::vector<std::vector<float>> constantFloat(2);
    constantFloat[0] = {constantFloat0};
    constantFloat[1] = {constantFloat1};
    table.addAttribute(TestAttributes::sTestFloat0, RATE_CONSTANT,
        std::move(constantFloat));

    scene_rdl2::math::Color constantColor0 = mRNG.randomColor();
    scene_rdl2::math::Color constantColor1 = mRNG.randomColor();
    std::vector<std::vector<scene_rdl2::math::Color>> constantColor(2);
    constantColor[0] = {constantColor0};
    constantColor[1] = {constantColor1};
    table.addAttribute(TestAttributes::sTestColor0, RATE_CONSTANT,
        std::move(constantColor));

    scene_rdl2::math::Color4 constantRGBA0 = mRNG.randomColor4();
    scene_rdl2::math::Color4 constantRGBA1 = mRNG.randomColor4();
    std::vector<std::vector<scene_rdl2::math::Color4>> constantRGBA(2);
    constantRGBA[0] = {constantRGBA0};
    constantRGBA[1] = {constantRGBA1};
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_CONSTANT,
        std::move(constantRGBA));

    scene_rdl2::math::Vec2f constantVec2f0 = mRNG.randomVec2f();
    scene_rdl2::math::Vec2f constantVec2f1 = mRNG.randomVec2f();
    std::vector<std::vector<scene_rdl2::math::Vec2f>> constantVec2f(2);
    constantVec2f[0] = {constantVec2f0};
    constantVec2f[1] = {constantVec2f1};
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_CONSTANT,
        std::move(constantVec2f));

    scene_rdl2::math::Vec3f constantVec3f0 = mRNG.randomVec3f();
    scene_rdl2::math::Vec3f constantVec3f1 = mRNG.randomVec3f();
    std::vector<std::vector<scene_rdl2::math::Vec3f>> constantVec3f(2);
    constantVec3f[0] = {constantVec3f0};
    constantVec3f[1] = {constantVec3f1};
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_CONSTANT,
        std::move(constantVec3f));

    scene_rdl2::math::Mat4f constantMat4f0 = mRNG.randomMat4f();
    scene_rdl2::math::Mat4f constantMat4f1 = mRNG.randomMat4f();
    std::vector<std::vector<scene_rdl2::math::Mat4f>> constantMat4f(2);
    constantMat4f[0] = {constantMat4f0};
    constantMat4f[1] = {constantMat4f1};
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_CONSTANT,
        std::move(constantMat4f));

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), 0));

    int face = 0;
    int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
    float w0 = 0.25f, w1 = 0.25f, w2 = 0.25f, w3 = 0.25f;
    float time = mRNG.randomFloat();
    internal::MeshInterpolator interpolator(attr.get(), time,
        0, face, v0, v1, v2, v3, w0, w1, w2, w3,
        face, v0, v1, v2, w0, w1, w2);
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0, interpolator,
        (1.0f - time) * constantFloat0 + time * constantFloat1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0, interpolator,
        (1.0f - time) * constantColor0 + time * constantColor1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0, interpolator,
        (1.0f - time) * constantRGBA0 + time * constantRGBA1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0, interpolator,
        (1.0f - time) * constantVec2f0 + time * constantVec2f1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0, interpolator,
        (1.0f - time) * constantVec3f0 + time * constantVec3f1));
    CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0, interpolator,
        (1.0f - time) * constantMat4f0 + time * constantMat4f1));
}

void TestInterpolator::testMeshPart()
{
    const size_t nParts = 3;
    std::vector<bool> uniformBool;
    std::vector<int> uniformInt;
    std::vector<long> uniformLong;
    std::vector<float> uniformFloat;
    std::vector<std::string> uniformString;
    std::vector<scene_rdl2::math::Color> uniformColor;
    std::vector<scene_rdl2::math::Color4> uniformRGBA;
    std::vector<scene_rdl2::math::Vec2f> uniformVec2f;
    std::vector<scene_rdl2::math::Vec3f> uniformVec3f;
    std::vector<scene_rdl2::math::Mat4f> uniformMat4f;
    for (size_t i = 0; i < nParts; ++i) {
        uniformBool.push_back(mRNG.randomBool());
        uniformInt.push_back(mRNG.randomInt());
        uniformLong.push_back(mRNG.randomLong());
        uniformFloat.push_back(mRNG.randomFloat());
        uniformString.push_back(std::string("uniform test string ") +
            std::to_string(i));
        uniformColor.push_back(mRNG.randomColor());
        uniformRGBA.push_back(mRNG.randomColor4());
        uniformVec2f.push_back(mRNG.randomVec2f());
        uniformVec3f.push_back(mRNG.randomVec3f());
        uniformMat4f.push_back(mRNG.randomMat4f());
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestBool0,
        RATE_PART, std::vector<bool>(uniformBool));
    table.addAttribute(TestAttributes::sTestInt0,
        RATE_PART, std::vector<int>(uniformInt));
    table.addAttribute(TestAttributes::sTestLong0,
        RATE_PART, std::vector<long>(uniformLong));
    table.addAttribute(TestAttributes::sTestFloat0,
        RATE_PART, std::vector<float>(uniformFloat));
    table.addAttribute(TestAttributes::sTestString0,
        RATE_PART, std::vector<std::string>(uniformString));
    table.addAttribute(TestAttributes::sTestColor0,
        RATE_PART, std::vector<scene_rdl2::math::Color>(uniformColor));
    table.addAttribute(TestAttributes::sTestRGBA0,
        RATE_PART, std::vector<scene_rdl2::math::Color4>(uniformRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0,
        RATE_PART, std::vector<scene_rdl2::math::Vec2f>(uniformVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0,
        RATE_PART, std::vector<scene_rdl2::math::Vec3f>(uniformVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0,
        RATE_PART, std::vector<scene_rdl2::math::Mat4f>(uniformMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, nParts, 0, 0,
        std::vector<size_t>(), 0));

    for (size_t i = 0; i < nParts; ++i) {
        int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
        float w0 = 0.25f, w1 = 0.25f, w2 = 0.25f, w3 = 0.25f;
        internal::MeshInterpolator interpolator(attr.get(), 0, i,
            0, v0, v1, v2, v3, w0, w1, w2, w3,
            0, v0, v1, v2, w0, w1, w2);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestBool0,
            interpolator, (bool)uniformBool[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestInt0,
            interpolator, uniformInt[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestLong0,
            interpolator, uniformLong[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator, uniformFloat[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestString0,
            interpolator, uniformString[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator, uniformColor[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator, uniformRGBA[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator, uniformVec2f[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator, uniformVec3f[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator, uniformMat4f[i]));
    }
}

void TestInterpolator::testMeshPartMB()
{
    const size_t nParts = 3;
    const size_t nTimes = 2;
    std::vector<std::vector<float>> uniformFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> uniformColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> uniformRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> uniformVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> uniformVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> uniformMat4f(nTimes);
    for(size_t t = 0; t < nTimes; ++t) {
        for (size_t i = 0; i < nParts; ++i) {
            uniformFloat[t].push_back(mRNG.randomFloat());
            uniformColor[t].push_back(mRNG.randomColor());
            uniformRGBA[t].push_back(mRNG.randomColor4());
            uniformVec2f[t].push_back(mRNG.randomVec2f());
            uniformVec3f[t].push_back(mRNG.randomVec3f());
            uniformMat4f[t].push_back(mRNG.randomMat4f());
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_PART,
        std::vector<std::vector<float>>(uniformFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_PART,
        std::vector<std::vector<scene_rdl2::math::Color>>(uniformColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_PART,
        std::vector<std::vector<scene_rdl2::math::Color4>>(uniformRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_PART,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(uniformVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_PART,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(uniformVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_PART,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(uniformMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, nParts, 0, 0,
        std::vector<size_t>(), 0));

    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nParts; ++i) {
        int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
        float w0 = 0.25f, w1 = 0.25f, w2 = 0.25f, w3 = 0.25f;
        internal::MeshInterpolator interpolator(attr.get(), t,
            i, 0, v0, v1, v2, v3, w0, w1, w2, w3,
            0, v0, v1, v2, w0, w1, w2);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator,
            (1.0f - t) * uniformFloat[0][i] + t * uniformFloat[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator,
            (1.0f - t) * uniformColor[0][i] + t * uniformColor[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator,
            (1.0f - t) * uniformRGBA[0][i] + t * uniformRGBA[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator,
            (1.0f - t) * uniformVec2f[0][i] + t * uniformVec2f[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator,
            (1.0f - t) * uniformVec3f[0][i] + t * uniformVec3f[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator,
            (1.0f - t) * uniformMat4f[0][i] + t * uniformMat4f[1][i]));
    }
}

void TestInterpolator::testMeshUniform()
{
    const size_t nFaces = 3;
    std::vector<bool> uniformBool;
    std::vector<int> uniformInt;
    std::vector<long> uniformLong;
    std::vector<float> uniformFloat;
    std::vector<std::string> uniformString;
    std::vector<scene_rdl2::math::Color> uniformColor;
    std::vector<scene_rdl2::math::Color4> uniformRGBA;
    std::vector<scene_rdl2::math::Vec2f> uniformVec2f;
    std::vector<scene_rdl2::math::Vec3f> uniformVec3f;
    std::vector<scene_rdl2::math::Mat4f> uniformMat4f;
    for (size_t i = 0; i < nFaces; ++i) {
        uniformBool.push_back(mRNG.randomBool());
        uniformInt.push_back(mRNG.randomInt());
        uniformLong.push_back(mRNG.randomLong());
        uniformFloat.push_back(mRNG.randomFloat());
        uniformString.push_back(std::string("uniform test string ") +
            std::to_string(i));
        uniformColor.push_back(mRNG.randomColor());
        uniformRGBA.push_back(mRNG.randomColor4());
        uniformVec2f.push_back(mRNG.randomVec2f());
        uniformVec3f.push_back(mRNG.randomVec3f());
        uniformMat4f.push_back(mRNG.randomMat4f());
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestBool0,
        RATE_UNIFORM, std::vector<bool>(uniformBool));
    table.addAttribute(TestAttributes::sTestInt0,
        RATE_UNIFORM, std::vector<int>(uniformInt));
    table.addAttribute(TestAttributes::sTestLong0,
        RATE_UNIFORM, std::vector<long>(uniformLong));
    table.addAttribute(TestAttributes::sTestFloat0,
        RATE_UNIFORM, std::vector<float>(uniformFloat));
    table.addAttribute(TestAttributes::sTestString0,
        RATE_UNIFORM, std::vector<std::string>(uniformString));
    table.addAttribute(TestAttributes::sTestColor0,
        RATE_UNIFORM, std::vector<scene_rdl2::math::Color>(uniformColor));
    table.addAttribute(TestAttributes::sTestRGBA0,
        RATE_UNIFORM, std::vector<scene_rdl2::math::Color4>(uniformRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0,
        RATE_UNIFORM, std::vector<scene_rdl2::math::Vec2f>(uniformVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0,
        RATE_UNIFORM, std::vector<scene_rdl2::math::Vec3f>(uniformVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0,
        RATE_UNIFORM, std::vector<scene_rdl2::math::Mat4f>(uniformMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nFaces, 0,
        std::vector<size_t>(), 0));

    for (size_t i = 0; i < nFaces; ++i) {
        int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
        float w0 = 0.25f, w1 = 0.25f, w2 = 0.25f, w3 = 0.25f;
        internal::MeshInterpolator interpolator(attr.get(), 0, 0,
            i, v0, v1, v2, v3, w0, w1, w2, w3,
            i, v0, v1, v2, w0, w1, w2);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestBool0,
            interpolator, (bool)uniformBool[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestInt0,
            interpolator, uniformInt[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestLong0,
            interpolator, uniformLong[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator, uniformFloat[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestString0,
            interpolator, uniformString[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator, uniformColor[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator, uniformRGBA[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator, uniformVec2f[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator, uniformVec3f[i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator, uniformMat4f[i]));
    }
}

void TestInterpolator::testMeshUniformMB()
{
    const size_t nFaces = 3;
    const size_t nTimes = 2;
    std::vector<std::vector<float>> uniformFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> uniformColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> uniformRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> uniformVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> uniformVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> uniformMat4f(nTimes);
    for(size_t t = 0; t < nTimes; ++t) {
        for (size_t i = 0; i < nFaces; ++i) {
            uniformFloat[t].push_back(mRNG.randomFloat());
            uniformColor[t].push_back(mRNG.randomColor());
            uniformRGBA[t].push_back(mRNG.randomColor4());
            uniformVec2f[t].push_back(mRNG.randomVec2f());
            uniformVec3f[t].push_back(mRNG.randomVec3f());
            uniformMat4f[t].push_back(mRNG.randomMat4f());
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_UNIFORM,
        std::vector<std::vector<float>>(uniformFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Color>>(uniformColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Color4>>(uniformRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(uniformVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(uniformVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_UNIFORM,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(uniformMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nFaces, 0,
        std::vector<size_t>(), 0));

    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nFaces; ++i) {
        int v0 = 0, v1 = 1, v2 = 2, v3 = 3;
        float w0 = 0.25f, w1 = 0.25f, w2 = 0.25f, w3 = 0.25f;
        internal::MeshInterpolator interpolator(attr.get(), t,
            0, i, v0, v1, v2, v3, w0, w1, w2, w3,
            i, v0, v1, v2, w0, w1, w2);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator,
            (1.0f - t) * uniformFloat[0][i] + t * uniformFloat[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator,
            (1.0f - t) * uniformColor[0][i] + t * uniformColor[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator,
            (1.0f - t) * uniformRGBA[0][i] + t * uniformRGBA[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator,
            (1.0f - t) * uniformVec2f[0][i] + t * uniformVec2f[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator,
            (1.0f - t) * uniformVec3f[0][i] + t * uniformVec3f[1][i]));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator,
            (1.0f - t) * uniformMat4f[0][i] + t * uniformMat4f[1][i]));
    }
}

void TestInterpolator::testMeshVarying()
{
    const size_t nFaces = 2;
    std::vector<std::vector<int> > faceIndexes(nFaces);
    // a 3 vertices face
    faceIndexes[0].push_back(0);
    faceIndexes[0].push_back(1);
    faceIndexes[0].push_back(2);
    // a 4 vertices face
    faceIndexes[1].push_back(2);
    faceIndexes[1].push_back(1);
    faceIndexes[1].push_back(3);
    faceIndexes[1].push_back(4);
    // two faces share 2 vertices
    const size_t nVertices = 5;

    std::vector<float> varyingFloat;
    std::vector<scene_rdl2::math::Color> varyingColor;
    std::vector<scene_rdl2::math::Color4> varyingRGBA;
    std::vector<scene_rdl2::math::Vec2f> varyingVec2f;
    std::vector<scene_rdl2::math::Vec3f> varyingVec3f;
    std::vector<scene_rdl2::math::Mat4f> varyingMat4f;
    for (size_t i = 0; i < nVertices; ++i) {
        varyingFloat.push_back(mRNG.randomFloat());
        varyingColor.push_back(mRNG.randomColor());
        varyingRGBA.push_back(mRNG.randomColor4());
        varyingVec2f.push_back(mRNG.randomVec2f());
        varyingVec3f.push_back(mRNG.randomVec3f());
        varyingMat4f.push_back(mRNG.randomMat4f());
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VARYING,
        std::vector<float>(varyingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VARYING,
        std::vector<scene_rdl2::math::Color>(varyingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VARYING,
        std::vector<scene_rdl2::math::Color4>(varyingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VARYING,
        std::vector<scene_rdl2::math::Vec2f>(varyingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VARYING,
        std::vector<scene_rdl2::math::Vec3f>(varyingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VARYING,
        std::vector<scene_rdl2::math::Mat4f>(varyingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, nVertices,
        std::vector<size_t>(), 0));

    std::unique_ptr<internal::MeshInterpolator> interpolator;
    for (size_t i = 0; i < nFaces; ++i) {
        int v[4];
        float w[4];
        if (faceIndexes[i].size() == 3) {
            v[0] = faceIndexes[i][0];
            v[1] = faceIndexes[i][1];
            v[2] = faceIndexes[i][2];
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = 1.0f - w[0] - w[1];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), 0,
                0, i, v[0], v[1], v[2], w[0], w[1], w[2],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        } else if (faceIndexes[i].size() == 4) {
            v[0] = faceIndexes[i][0];
            v[1] = faceIndexes[i][1];
            v[2] = faceIndexes[i][2];
            v[3] = faceIndexes[i][3];
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = (1.0f - w[0] - w[1]) * mRNG.randomFloat();
            w[3] = 1.0f - w[0] - w[1] - w[2];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), 0,
                0, i, v[0], v[1], v[2], v[3], w[0], w[1], w[2], w[3],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        }
        size_t nSamples = faceIndexes[i].size();
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0, *interpolator,
            weightSum(&varyingFloat[0], &faceIndexes[i][0], w, nSamples)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0, *interpolator,
            weightSum(&varyingColor[0], &faceIndexes[i][0], w, nSamples)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0, *interpolator,
            weightSum(&varyingRGBA[0], &faceIndexes[i][0], w, nSamples)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0, *interpolator,
            weightSum(&varyingVec2f[0], &faceIndexes[i][0], w, nSamples)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0, *interpolator,
            weightSum(&varyingVec3f[0], &faceIndexes[i][0], w, nSamples)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0, *interpolator,
            weightSum(&varyingMat4f[0], &faceIndexes[i][0], w, nSamples)));

    }
}

void TestInterpolator::testMeshVaryingMB()
{
    const size_t nFaces = 2;
    std::vector<std::vector<int> > faceIndexes(nFaces);
    // a 3 vertices face
    faceIndexes[0].push_back(0);
    faceIndexes[0].push_back(1);
    faceIndexes[0].push_back(2);
    // a 4 vertices face
    faceIndexes[1].push_back(2);
    faceIndexes[1].push_back(1);
    faceIndexes[1].push_back(3);
    faceIndexes[1].push_back(4);
    // two faces share 2 vertices
    const size_t nVertices = 5;

    const size_t nTimes = 2;
    std::vector<std::vector<float>> varyingFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> varyingColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> varyingRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> varyingVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> varyingVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> varyingMat4f(nTimes);

    for (size_t i = 0; i < nVertices; ++i) {
        for (size_t t = 0; t < nTimes; ++t) {
            varyingFloat[t].push_back(mRNG.randomFloat());
            varyingColor[t].push_back(mRNG.randomColor());
            varyingRGBA[t].push_back(mRNG.randomColor4());
            varyingVec2f[t].push_back(mRNG.randomVec2f());
            varyingVec3f[t].push_back(mRNG.randomVec3f());
            varyingMat4f[t].push_back(mRNG.randomMat4f());
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VARYING,
        std::vector<std::vector<float>>(varyingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color>>(varyingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color4>>(varyingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(varyingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(varyingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(varyingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, nVertices,
        std::vector<size_t>(), 0));


    std::unique_ptr<internal::MeshInterpolator> interpolator;
    float t = mRNG.randomFloat();
    for (size_t i = 0; i < nFaces; ++i) {
        int v[4];
        float w[4];
        if (faceIndexes[i].size() == 3) {
            v[0] = faceIndexes[i][0];
            v[1] = faceIndexes[i][1];
            v[2] = faceIndexes[i][2];
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = 1.0f - w[0] - w[1];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), t,
                0, i, v[0], v[1], v[2], w[0], w[1], w[2],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        } else if (faceIndexes[i].size() == 4) {
            v[0] = faceIndexes[i][0];
            v[1] = faceIndexes[i][1];
            v[2] = faceIndexes[i][2];
            v[3] = faceIndexes[i][3];
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = (1.0f - w[0] - w[1]) * mRNG.randomFloat();
            w[3] = 1.0f - w[0] - w[1] - w[2];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), t,
                0, i, v[0], v[1], v[2], v[3], w[0], w[1], w[2], w[3],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        }
        size_t nSamples = faceIndexes[i].size();
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0, *interpolator,
            weightSumMB(&varyingFloat[0][0], &varyingFloat[1][0],
            &faceIndexes[i][0], w, nSamples, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0, *interpolator,
            weightSumMB(&varyingColor[0][0], &varyingColor[1][0],
            &faceIndexes[i][0], w, nSamples, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0, *interpolator,
            weightSumMB(&varyingRGBA[0][0], &varyingRGBA[1][0],
            &faceIndexes[i][0], w, nSamples, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0, *interpolator,
            weightSumMB(&varyingVec2f[0][0], &varyingVec2f[1][0],
            &faceIndexes[i][0], w, nSamples, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0, *interpolator,
            weightSumMB(&varyingVec3f[0][0], &varyingVec3f[1][0],
            &faceIndexes[i][0], w, nSamples, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0, *interpolator,
            weightSumMB(&varyingMat4f[0][0], &varyingMat4f[1][0],
            &faceIndexes[i][0], w, nSamples, t)));

    }
}



void TestInterpolator::testMeshFaceVarying()
{
    const size_t nFaces = 2;
    const std::vector<size_t> nFaceVNum{3, 4};

    std::vector<float> faceVaryingFloat;
    std::vector<scene_rdl2::math::Color> faceVaryingColor;
    std::vector<scene_rdl2::math::Color4> faceVaryingRGBA;
    std::vector<scene_rdl2::math::Vec2f> faceVaryingVec2f;
    std::vector<scene_rdl2::math::Vec3f> faceVaryingVec3f;
    std::vector<scene_rdl2::math::Mat4f> faceVaryingMat4f;

    for (size_t i = 0; i < nFaces; ++i) {
        for (size_t j = 0; j < nFaceVNum[i]; ++j) {
            faceVaryingFloat.push_back(mRNG.randomFloat());
            faceVaryingColor.push_back(mRNG.randomColor());
            faceVaryingRGBA.push_back(mRNG.randomColor4());
            faceVaryingVec2f.push_back(mRNG.randomVec2f());
            faceVaryingVec3f.push_back(mRNG.randomVec3f());
            faceVaryingMat4f.push_back(mRNG.randomMat4f());
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        std::vector<float>(faceVaryingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Color>(faceVaryingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Color4>(faceVaryingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Vec2f>(faceVaryingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Vec3f>(faceVaryingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Mat4f>(faceVaryingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nFaces, 0, nFaceVNum, 0));

    std::unique_ptr<internal::MeshInterpolator> interpolator;
    size_t faceVaryingOffset = 0;
    for(size_t i = 0; i < nFaces; ++i) {
        int v[4] = {0, 1, 2, 3};
        float w[4];
        if (nFaceVNum[i] == 3) {
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = 1.0f - w[0] - w[1];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), 0,
                0, i, v[0], v[1], v[2], w[0], w[1], w[2],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        } else if (nFaceVNum[i] == 4) {
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = (1.0f - w[0] - w[1]) * mRNG.randomFloat();
            w[3] = 1.0f - w[0] - w[1] - w[2];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), 0,
                0, i, v[0], v[1], v[2], v[3], w[0], w[1], w[2], w[3],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        }
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            *interpolator,
            weightSum(&faceVaryingFloat[faceVaryingOffset],
            v, w, nFaceVNum[i])));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            *interpolator,
            weightSum(&faceVaryingColor[faceVaryingOffset],
            v, w, nFaceVNum[i])));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            *interpolator,
            weightSum(&faceVaryingRGBA[faceVaryingOffset],
            v, w, nFaceVNum[i])));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            *interpolator,
            weightSum(&faceVaryingVec2f[faceVaryingOffset],
            v, w, nFaceVNum[i])));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            *interpolator,
            weightSum(&faceVaryingVec3f[faceVaryingOffset],
            v, w, nFaceVNum[i])));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            *interpolator,
            weightSum(&faceVaryingMat4f[faceVaryingOffset],
            v, w, nFaceVNum[i])));
        faceVaryingOffset += nFaceVNum[i];
    }
}

void TestInterpolator::testMeshFaceVaryingMB()
{
    const size_t nFaces = 2;
    const std::vector<size_t> nFaceVNum{3, 4};
    const size_t nTimes = 2;
    std::vector<std::vector<float>> faceVaryingFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> faceVaryingColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> faceVaryingRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> faceVaryingVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> faceVaryingVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> faceVaryingMat4f(nTimes);

    for (size_t i = 0; i < nFaces; ++i) {
        for (size_t j = 0; j < nFaceVNum[i]; ++j) {
            for (size_t t = 0; t < nTimes; ++t) {
                faceVaryingFloat[t].push_back(mRNG.randomFloat());
                faceVaryingColor[t].push_back(mRNG.randomColor());
                faceVaryingRGBA[t].push_back(mRNG.randomColor4());
                faceVaryingVec2f[t].push_back(mRNG.randomVec2f());
                faceVaryingVec3f[t].push_back(mRNG.randomVec3f());
                faceVaryingMat4f[t].push_back(mRNG.randomMat4f());
            }
        }
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        std::vector<std::vector<float>>(faceVaryingFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color>>(faceVaryingColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Color4>>(faceVaryingRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(faceVaryingVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(faceVaryingVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_FACE_VARYING,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(faceVaryingMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nFaces, 0, nFaceVNum, 0));

    std::unique_ptr<internal::MeshInterpolator> interpolator;
    float t = mRNG.randomFloat();
    size_t faceVaryingOffset = 0;
    for(size_t i = 0; i < nFaces; ++i) {
        int v[4] = {0, 1, 2, 3};
        float w[4];
        if (nFaceVNum[i] == 3) {
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = 1.0f - w[0] - w[1];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), t,
                0, i, v[0], v[1], v[2], w[0], w[1], w[2],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        } else if (nFaceVNum[i] == 4) {
            w[0] = mRNG.randomFloat();
            w[1] = (1.0f - w[0]) * mRNG.randomFloat();
            w[2] = (1.0f - w[0] - w[1]) * mRNG.randomFloat();
            w[3] = 1.0f - w[0] - w[1] - w[2];
            interpolator.reset(new internal::MeshInterpolator(attr.get(), t,
                0, i, v[0], v[1], v[2], v[3], w[0], w[1], w[2], w[3],
                i, v[0], v[1], v[2], w[0], w[1], w[2]));
        }
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            *interpolator,
            weightSumMB(&faceVaryingFloat[0][faceVaryingOffset],
            &faceVaryingFloat[1][faceVaryingOffset], v, w, nFaceVNum[i], t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            *interpolator,
            weightSumMB(&faceVaryingColor[0][faceVaryingOffset],
            &faceVaryingColor[1][faceVaryingOffset], v, w, nFaceVNum[i], t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            *interpolator,
            weightSumMB(&faceVaryingRGBA[0][faceVaryingOffset],
            &faceVaryingRGBA[1][faceVaryingOffset], v, w, nFaceVNum[i], t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            *interpolator,
            weightSumMB(&faceVaryingVec2f[0][faceVaryingOffset],
            &faceVaryingVec2f[1][faceVaryingOffset], v, w, nFaceVNum[i], t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            *interpolator,
            weightSumMB(&faceVaryingVec3f[0][faceVaryingOffset],
            &faceVaryingVec3f[1][faceVaryingOffset], v, w, nFaceVNum[i], t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            *interpolator,
            weightSumMB(&faceVaryingMat4f[0][faceVaryingOffset],
            &faceVaryingMat4f[1][faceVaryingOffset], v, w, nFaceVNum[i], t)));
        faceVaryingOffset += nFaceVNum[i];
    }
}



void TestInterpolator::testMeshVertex()
{
    const size_t nFaces = 2;
    std::vector<std::vector<int> > faceIndexes(nFaces);
    // a 3 vertices face
    faceIndexes[0].push_back(0);
    faceIndexes[0].push_back(1);
    faceIndexes[0].push_back(2);
    // a 3 vertices face
    faceIndexes[1].push_back(2);
    faceIndexes[1].push_back(1);
    faceIndexes[1].push_back(3);
    // two faces share 2 vertices
    const size_t nVertices = 4;

    std::vector<float> vertexFloat;
    std::vector<scene_rdl2::math::Color> vertexColor;
    std::vector<scene_rdl2::math::Color4> vertexRGBA;
    std::vector<scene_rdl2::math::Vec2f> vertexVec2f;
    std::vector<scene_rdl2::math::Vec3f> vertexVec3f;
    std::vector<scene_rdl2::math::Mat4f> vertexMat4f;

    for (size_t i = 0; i < nVertices; ++i) {
        vertexFloat.push_back(mRNG.randomFloat());
        vertexColor.push_back(mRNG.randomColor());
        vertexRGBA.push_back(mRNG.randomColor4());
        vertexVec2f.push_back(mRNG.randomVec2f());
        vertexVec3f.push_back(mRNG.randomVec3f());
        vertexMat4f.push_back(mRNG.randomMat4f());
    }

    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VERTEX,
        std::vector<float>(vertexFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Color>(vertexColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Color4>(vertexRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Vec2f>(vertexVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Vec3f>(vertexVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VERTEX,
        std::vector<scene_rdl2::math::Mat4f>(vertexMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), nVertices));

    float w[3];
    for (size_t i = 0; i < nFaces; ++i) {
        int v[3] = {faceIndexes[i][0], faceIndexes[i][1], faceIndexes[i][2]};
        w[0] = mRNG.randomFloat();
        w[1] = (1.0f - w[0]) * mRNG.randomFloat();
        w[2] = (1.0f - w[0] - w[1]) * mRNG.randomFloat();
        internal::MeshInterpolator interpolator(attr.get(), 0,
            0, i, v[0], v[1], v[2], w[0], w[1], w[2],
            i, v[0], v[1], v[2], w[0], w[1], w[2]);
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0, interpolator,
            weightSum(&vertexFloat[0], v, w, 3)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0, interpolator,
            weightSum(&vertexColor[0], v, w, 3)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0, interpolator,
            weightSum(&vertexRGBA[0], v, w, 3)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0, interpolator,
            weightSum(&vertexVec2f[0], v, w, 3)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0, interpolator,
            weightSum(&vertexVec3f[0], v, w, 3)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0, interpolator,
            weightSum(&vertexMat4f[0], v, w, 3)));
    }
}

void TestInterpolator::testMeshVertexMB()
{
    const size_t nFaces = 2;
    std::vector<std::vector<int> > faceIndexes(nFaces);
    // a 3 vertices face
    faceIndexes[0].push_back(0);
    faceIndexes[0].push_back(1);
    faceIndexes[0].push_back(2);
    // a 3 vertices face
    faceIndexes[1].push_back(2);
    faceIndexes[1].push_back(1);
    faceIndexes[1].push_back(3);
    // two faces share 2 vertices
    const size_t nVertices = 4;
    size_t nTimes = 2;
    std::vector<std::vector<float>> vertexFloat(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color>> vertexColor(nTimes);
    std::vector<std::vector<scene_rdl2::math::Color4>> vertexRGBA(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec2f>> vertexVec2f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Vec3f>> vertexVec3f(nTimes);
    std::vector<std::vector<scene_rdl2::math::Mat4f>> vertexMat4f(nTimes);

    for (size_t t = 0; t < nTimes; ++t) {
        for (size_t i = 0; i < nVertices; ++i) {
            vertexFloat[t].push_back(mRNG.randomFloat());
            vertexColor[t].push_back(mRNG.randomColor());
            vertexRGBA[t].push_back(mRNG.randomColor4());
            vertexVec2f[t].push_back(mRNG.randomVec2f());
            vertexVec3f[t].push_back(mRNG.randomVec3f());
            vertexMat4f[t].push_back(mRNG.randomMat4f());
        }
    }


    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VERTEX,
        std::vector<std::vector<float>>(vertexFloat));
    table.addAttribute(TestAttributes::sTestColor0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Color>>(vertexColor));
    table.addAttribute(TestAttributes::sTestRGBA0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Color4>>(vertexRGBA));
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Vec2f>>(vertexVec2f));
    table.addAttribute(TestAttributes::sTestVec3f0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Vec3f>>(vertexVec3f));
    table.addAttribute(TestAttributes::sTestMat4f0, RATE_VERTEX,
        std::vector<std::vector<scene_rdl2::math::Mat4f>>(vertexMat4f));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), nVertices));

    float t = mRNG.randomFloat();
    float w[3];
    for (size_t i = 0; i < nFaces; ++i) {
        int v[3] = {faceIndexes[i][0], faceIndexes[i][1], faceIndexes[i][2]};
        w[0] = mRNG.randomFloat();
        w[1] = (1.0f - w[0]) * mRNG.randomFloat();
        w[2] = (1.0f - w[0] - w[1]) * mRNG.randomFloat();
        internal::MeshInterpolator interpolator(attr.get(), t,
            0, i, v[0], v[1], v[2], w[0], w[1], w[2],
            i, v[0], v[1], v[2], w[0], w[1], w[2]);

        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestFloat0,
            interpolator,
            weightSumMB(&vertexFloat[0][0], &vertexFloat[1][0],
            v, w, 3, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestColor0,
            interpolator,
            weightSumMB(&vertexColor[0][0], &vertexColor[1][0],
            v, w, 3, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestRGBA0,
            interpolator,
            weightSumMB(&vertexRGBA[0][0], &vertexRGBA[1][0],
            v, w, 3, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec2f0,
            interpolator,
            weightSumMB(&vertexVec2f[0][0], &vertexVec2f[1][0],
            v, w, 3, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestVec3f0,
            interpolator,
            weightSumMB(&vertexVec3f[0][0], &vertexVec3f[1][0],
            v, w, 3, t)));
        CPPUNIT_ASSERT(verifyResult(TestAttributes::sTestMat4f0,
            interpolator,
            weightSumMB(&vertexMat4f[0][0], &vertexMat4f[1][0],
            v, w, 3, t)));
    }
}


} // namespace unittest
} // namespace geom
} // namespace moonray

