// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBsdfCommon.h
/// $Id$
///

#pragma once

#include "BsdfFactory.h"

#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

#include <tbb/tbb.h>

#include <cstdint>

namespace moonray {
namespace pbr {


//----------------------------------------------------------------------------

struct TestBsdfSettings {
    enum Type {
        BSDF_SAMPLER,
        BSDF_ONE_SAMPLER,
        BSDFV,
        BSDF_ONE_SAMPLERV
    };

    TestBsdfSettings(const BsdfFactory &bf, const scene_rdl2::math::ReferenceFrame &f,
            bool tr, float tc, float ti, bool ap, bool ae, Type t)
    : bsdfFactory(bf)
    , frame(f)
    , testReciprocity(tr)
    , toleranceConsistency(tc)
    , toleranceIntegral(ti)
    , assertPdfIntegral(ap)
    , assertEvalIntegral(ae)
    , testType(t)
    {
    }

public:
    const BsdfFactory &bsdfFactory;
    const scene_rdl2::math::ReferenceFrame &frame;
    bool testReciprocity;
    float toleranceConsistency;
    float toleranceIntegral;
    bool assertPdfIntegral;
    bool assertEvalIntegral;
    Type testType;

    static const int sMaxSamplesPerLobe = 16;
};


void runBsdfTest(const TestBsdfSettings &test, int viewAnglesTheta, int viewAnglesPhy, int sampleCount);


//----------------------------------------------------------------------------

struct TestBsdfConsistencyTask
{
    struct Result
    {
        constexpr Result() noexcept
        : mZeroSamplePdfCount(0)
        , mZeroEvalPdfCount(0)
        , mZeroEvalRecipPdfCount(0)
        , mSampleCount(0)
        {
        }

        int mZeroSamplePdfCount;
        int mZeroEvalPdfCount;
        int mZeroEvalRecipPdfCount;
        int mSampleCount;
    };


    TestBsdfConsistencyTask(unsigned sampleBegin,
                            unsigned sampleEnd,
                            std::uint32_t rSeed,
                            std::uint32_t rStream,
                            const TestBsdfSettings& test)
    : inTest(std::addressof(test))
    , randomSeed(rSeed)
    , randomStream(rStream)
    , sampleFirst(sampleBegin)
    , sampleLast(sampleEnd)
    , mResult()
    {
    }

    Result operator()()
    {
        switch(inTest->testType) {
        case TestBsdfSettings::BSDF_SAMPLER:
            testBsdfSampler();
            break;
        case TestBsdfSettings::BSDF_ONE_SAMPLER:
            testBsdfOneSampler();
            break;
        case TestBsdfSettings::BSDFV:
            testBsdfv();
            break;
        case TestBsdfSettings::BSDF_ONE_SAMPLERV:
            testBsdfOneSamplerv();
        }

        return mResult;
    }

    void testBsdfOneSampler();
    void testBsdfSampler();
    void testBsdfv();
    void testBsdfOneSamplerv();

    const TestBsdfSettings* inTest;
    std::uint32_t randomSeed;
    std::uint32_t randomStream;
    unsigned sampleFirst;
    unsigned sampleLast;
    Result mResult;
};


//----------------------------------------------------------------------------

struct TestBsdfPdfIntegralTask
{
    struct Result
    {
        Result() noexcept
        : mIntegral(0)
        , mSampleCount(0)
        , mSpherical(true)
        , mDoAssert(false)
        {
        }

        Result(bool spherical, bool asrt) noexcept
        : mIntegral(0)
        , mSampleCount(0)
        , mSpherical(spherical)
        , mDoAssert(asrt)
        {
        }

        float mIntegral;
        int mSampleCount;
        bool mSpherical;
        bool mDoAssert;
    };

    TestBsdfPdfIntegralTask(unsigned sampleBegin,
                            unsigned sampleEnd,
                            std::uint32_t rSeed,
                            std::uint32_t rStream,
                            const TestBsdfSettings& test,
                            const shading::BsdfSlice& slice,
                            bool sph,
                            bool asrt)
    : inTest(std::addressof(test))
    , inSlice(std::addressof(slice))
    , randomSeed(rSeed)
    , randomStream(rStream)
    , sampleFirst(sampleBegin)
    , sampleLast(sampleEnd)
    , mResult(sph, asrt)
    {
    }

    Result operator()()
    {
        switch(inTest->testType) {
        case TestBsdfSettings::BSDF_SAMPLER:
            testBsdfSampler();
            break;
        case TestBsdfSettings::BSDF_ONE_SAMPLER:
            testBsdfOneSampler();
            break;
        case TestBsdfSettings::BSDFV:
            testBsdfv();
            break;
        case TestBsdfSettings::BSDF_ONE_SAMPLERV:
            testBsdfOneSamplerv();
        }

        return mResult;
    }

    void testBsdfOneSampler();
    void testBsdfSampler();
    void testBsdfv();
    void testBsdfOneSamplerv();

    const TestBsdfSettings* inTest;
    const shading::BsdfSlice* inSlice;
    std::uint32_t randomSeed;
    std::uint32_t randomStream;
    unsigned sampleFirst;
    unsigned sampleLast;
    Result mResult;
};


//----------------------------------------------------------------------------

struct TestBsdfEvalIntegralTask
{
    struct Result
    {
        explicit Result()
        : mIntegralUniform(scene_rdl2::math::sBlack)
        , mIntegralImportance(scene_rdl2::math::sBlack)
        , mSampleCount(0)
        , mDoAssert(false)
        {
        }

        explicit Result(bool asrt)
        : mIntegralUniform(scene_rdl2::math::sBlack)
        , mIntegralImportance(scene_rdl2::math::sBlack)
        , mSampleCount(0)
        , mDoAssert(asrt)
        {
        }

        scene_rdl2::math::Color mIntegralUniform;
        scene_rdl2::math::Color mIntegralImportance;
        int mSampleCount;
        bool mDoAssert;
    };

    TestBsdfEvalIntegralTask(unsigned sampleBegin,
                             unsigned sampleEnd,
                             std::uint32_t rSeed,
                             std::uint32_t rStream,
                             const TestBsdfSettings& test,
                             const shading::BsdfSlice& slice,
                             bool asrt)
    : inTest(std::addressof(test))
    , inSlice(std::addressof(slice))
    , randomSeed(rSeed)
    , randomStream(rStream)
    , sampleFirst(sampleBegin)
    , sampleLast(sampleEnd)
    , mResult(asrt)
    {
    }

    Result operator()()
    {
        switch(inTest->testType) {
        case TestBsdfSettings::BSDF_SAMPLER:
            testBsdfSampler();
            break;
        case TestBsdfSettings::BSDF_ONE_SAMPLER:
            testBsdfOneSampler();
            break;
        case TestBsdfSettings::BSDFV:
            testBsdfv();
            break;
        case TestBsdfSettings::BSDF_ONE_SAMPLERV:
            testBsdfOneSamplerv();
            break;
        }

        return mResult;
    }

    void testBsdfOneSampler();
    void testBsdfSampler();
    void testBsdfv();
    void testBsdfOneSamplerv();

    const TestBsdfSettings* inTest;
    const shading::BsdfSlice* inSlice;
    std::uint32_t randomSeed;
    std::uint32_t randomStream;
    unsigned sampleFirst;
    unsigned sampleLast;
    Result mResult;
};


//----------------------------------------------------------------------------

// Adjusts 10K sample count based on roughness
finline int
getSampleCount(float roughness)
{
    return 10000 / (roughness * roughness);
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

