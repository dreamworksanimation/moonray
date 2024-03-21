// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestSampler.cc
/// $Id$
///


#include "TestSampler.h"
#include "TestSampler_ispc_stubs.h"

#include <moonray/rendering/pbr/sampler/Sampler.h>
#include <moonray/rendering/pbr/sampler/SequenceID.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>
#include <moonray/rendering/pbr/camera/LensDistribution.h>
#include <scene_rdl2/common/fb_util/RunningStats.h>
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/common/fb_util/StatisticalTestSuite.h>

#include <random>
#include <fstream>

using namespace scene_rdl2::StatisticalTestSuite;

namespace moonray {
namespace pbr {

// The variance of a uniform distribution is 1/12 * (b - a)^2
// Our range is [0, 1), making 1/12 * (1 - 0)^2 = 1/12
const float kUniformUnitSquareVariance1D = 1.0f/12.0f;

// Wigner semicircle distribution, radius = 1.
const float kUniformDiskVariance1D = 1.0/4.0;

template <template <typename> class LowerCheck, template <typename> class UpperCheck, std::size_t N>
void testRange(float lower, float upper, float (&values)[N])
{
    const float* const first = values;
    const float* const last = first + N;
    scene_rdl2::StatisticalTestSuite::testRange<LowerCheck, UpperCheck>(lower, upper, first, last);
}

template <template <typename> class LowerCheck, template <typename> class UpperCheck>
void testRange(float lower, float upper, const std::vector<float>& values)
{
    scene_rdl2::StatisticalTestSuite::testRange<LowerCheck, UpperCheck>(lower, upper, values.begin(), values.end());
}

void TestSampler::setUp()
{
}

void TestSampler::tearDown()
{
}

template <typename CDF>
void testKolmogorovSmirnov(const std::vector<float>& samples, CDF cdf)
{
    scene_rdl2::StatisticalTestSuite::testKolmogorovSmirnov(samples.begin(), samples.end(), cdf);
}

template <typename CDF, std::size_t N>
void testKolmogorovSmirnov(float (&samples)[N], CDF cdf)
{
    scene_rdl2::StatisticalTestSuite::testKolmogorovSmirnov(samples, samples + N, cdf);
}

void TestSampler::testPrimaryDeterminism()
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    const utype kNumTests = 5;
    for (utype test = 0; test < kNumTests; ++test) {
        for (int filternum = 0; filternum < 3; ++filternum) {
            std::unique_ptr<PixelFilter> filter;
            switch (filternum) {
                case 0:
                    filter.reset(new BoxPixelFilter(2));
                    break;
                case 1:
                    filter.reset(new CubicBSplinePixelFilter(2));
                    break;
                case 2:
                    filter.reset(new QuadraticBSplinePixelFilter(2));
                    break;
                default:
                    CPPUNIT_FAIL("Oops!");
            }

            std::vector<Sample> samples[2];
            samples[0].reserve(kNumSamples);
            samples[1].reserve(kNumSamples);

            for (int i = 0; i < 2; ++i) {
                Sampler sampler(PixelScramble(test), filter.get(), false, 64);
                for (utype s = 0; s < kNumSamples; ++s) {
                    samples[i].push_back(sampler.getPrimarySample(701 * test, 1051 * test, 7*test, s, true, 0.0));
                }
            }

            CPPUNIT_ASSERT(samples[0] == samples[1]);
        }
    }
}

template <utype D>
void doTestIntegratorDeterminism0()
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    const utype kNumTests = 5;
    for (utype test = 0; test < kNumTests; ++test) {
        std::vector<float> samples[2];
        samples[0].reserve(kNumSamples * D);
        samples[1].reserve(kNumSamples * D);

        SequenceID sid(7*test, 43*test, 89*test);

        for (int i = 0; i < 2; ++i) {
            IntegratorSample<D> sampler(sid);
            for (utype s = 0; s < kNumSamples; ++s) {
                float data[D];
                sampler.getSample(data, test);

                for (utype k = 0; k < D; ++k) {
                    samples[i].push_back(data[k]);
                }
            }
        }

        CPPUNIT_ASSERT(samples[0] == samples[1]);

        for (const auto& s : samples[0]) {
            CPPUNIT_ASSERT(s >= 0.0f);
            CPPUNIT_ASSERT(s <  1.0f);
        }
    }
}

template <utype D>
void doTestIntegratorDeterminism1()
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    const utype kNumTests = 5;
    for (utype test = 0; test < kNumTests; ++test) {
        std::vector<float> samples[2];
        samples[0].reserve(kNumSamples * D);
        samples[1].reserve(kNumSamples * D);

        SequenceID sid(23*test, 97*test, 89*test);

        for (int i = 0; i < 2; ++i) {
            IntegratorSample<D> sampler(sid);
            for (utype s = 0; s < kNumSamples; ++s) {
                if (i == 0) {
                    // Compare one sample set where we expicitly set progressive
                    // against one that's implicit.
                    sampler.resume(sid, s);
                }
                float data[D];
                sampler.getSample(data, test);

                for (utype k = 0; k < D; ++k) {
                    samples[i].push_back(data[k]);
                }
            }
        }

        CPPUNIT_ASSERT(samples[0] == samples[1]);

        for (const auto& s : samples[0]) {
            CPPUNIT_ASSERT(s >= 0.0f);
            CPPUNIT_ASSERT(s <  1.0f);
        }
    }
}

void TestSampler::testIntegratorDeterminism()
{
    doTestIntegratorDeterminism0<1>();
    doTestIntegratorDeterminism0<2>();
    doTestIntegratorDeterminism1<1>();
    doTestIntegratorDeterminism1<2>();
}

void TestSampler::testPrimaryDistribution()
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    std::vector<float> pixelx;
    std::vector<float> pixely;
    std::vector<float> lensu;
    std::vector<float> lensv;
    std::vector<float> time;

    pixelx.reserve(kNumSamples);
    pixely.reserve(kNumSamples);
    lensu.reserve(kNumSamples);
    lensv.reserve(kNumSamples);
    time.reserve(kNumSamples);

    const LensDistribution lens = LensDistribution::createUnitTestDistribution();

    BoxPixelFilter filter(1);
    Sampler sampler(PixelScramble{0xdeadbeef}, &filter, false, 64);
    for (utype s = 0; s < kNumSamples; ++s) {
        const auto sample = sampler.getPrimarySample(701, 1051, 7, s, true, 0.0);
        pixelx.push_back(sample.pixelX);
        pixely.push_back(sample.pixelY);
        lensu.push_back(sample.lensU);
        lensv.push_back(sample.lensV);
        time.push_back(sample.time);

        lens.sampleLens(lensu.back(), lensv.back());
    }

    // Test range of pixelX for box filter of extent 1 [0, 1)
    testRange<std::greater_equal, std::less>(0.0f, 1.0f, pixelx);

    // Test range of pixelY for box filter of extent 1 [0, 1)
    testRange<std::greater_equal, std::less>(0.0f, 1.0f, pixely);

    // Test range of lensU (-1, 1)
    testRange<std::greater, std::less>(-1.0f, 1.0f, lensu);

    // Test range of lensV (-1, 1)
    testRange<std::greater, std::less>(-1.0f, 1.0f, lensv);

    // Test range of time [0, 1)
    testRange<std::greater_equal, std::less>(0.0f, 1.0f, time);

    scene_rdl2::fb_util::RunningStats<> statsPixelx;
    for (auto f : pixelx) {
        statsPixelx.push(f);
    }

    scene_rdl2::fb_util::RunningStats<> statsPixely;
    for (auto f : pixely) {
        statsPixely.push(f);
    }

    scene_rdl2::fb_util::RunningStats<> statsLensu;
    for (auto f : lensu) {
        statsLensu.push(f);
    }

    scene_rdl2::fb_util::RunningStats<> statsLensv;
    for (auto f : lensv) {
        statsLensv.push(f);
    }

    scene_rdl2::fb_util::RunningStats<> statsTime;
    for (auto f : time) {
        statsTime.push(f);
    }

    testMean(statsPixelx, 0.5f, kUniformUnitSquareVariance1D);
    testMean(statsPixely, 0.5f, kUniformUnitSquareVariance1D);
    testMean(statsTime,   0.5f, kUniformUnitSquareVariance1D);
    testMean(statsLensu,  0.0f, kUniformDiskVariance1D);
    testMean(statsLensv,  0.0f, kUniformDiskVariance1D);

    testVariance(statsPixelx, kUniformUnitSquareVariance1D);
    testVariance(statsPixely, kUniformUnitSquareVariance1D);
    testVariance(statsTime,   kUniformUnitSquareVariance1D);
    testVariance(statsLensu,  kUniformDiskVariance1D);
    testVariance(statsLensv,  kUniformDiskVariance1D);

    testKolmogorovSmirnov(pixelx, UniformCDFContinuous<float>(0.0f, 1.0f));
    testKolmogorovSmirnov(pixely, UniformCDFContinuous<float>(0.0f, 1.0f));
    testKolmogorovSmirnov(time,   UniformCDFContinuous<float>(0.0f, 1.0f));

#if 0 // Used to test the tests! We expect each to fail about 5% of the time.
    std::random_device rd;
    math::Random gen(rd());
    for (int j = 0; j < 100; ++j) {
        scene_rdl2::fb_util::RunningStats<> rs;
        for (int i = 0; i < 10000; ++i) {
            rs.push(gen.getNextFloat());
        }
        PRINT(j);
        testMean(rs, 0.5f, 1.0 / 12.0);
        testVariance(rs, 1.0 / 12.0);
    }
#endif
}

template <utype D>
void doTestIntegratorDistribution()
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    SequenceID sid(23, 97, 89);
    IntegratorSample<D> sampler(sid);

    std::vector<float> samples[D];
    for (utype k = 0; k < D; ++k) {
        samples[k].reserve(kNumSamples);
    }

    for (utype s = 0; s < kNumSamples; ++s) {
        float data[D];
        sampler.getSample(data, 0);
        for (utype k = 0; k < D; ++k) {
            samples[k].push_back(data[k]);
        }
    }

    for (utype k = 0; k < D; ++k) {
        // Test range [0, 1)
        testRange<std::greater_equal, std::less>(0.0f, 1.0f, samples[k]);

        scene_rdl2::fb_util::RunningStats<> stats;
        for (auto f : samples[k]) {
            stats.push(f);
        }
        testMean(stats, 0.5f, kUniformUnitSquareVariance1D);
        testVariance(stats,   kUniformUnitSquareVariance1D);
        testKolmogorovSmirnov(samples[k], UniformCDFContinuous<float>(0.0f, 1.0f));
    }
}

using Sample1DFunction = void (*)(float*, uint32_t, uint32_t);
using Sample2DFunction = void (*)(float*, float*, uint32_t, uint32_t);

void doTestISPCIntegratorDistribution1D(Sample1DFunction getSamples, const char* fileToken)
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    float samples0[kNumSamples];

    for (utype lane = 0; lane < VLEN; ++lane) {
        getSamples(samples0, lane, kNumSamples);

#define TID1D_WRITE_FILE 0
#if TID1D_WRITE_FILE
        const std::string filename = std::string("integrator1D_") +
                                     fileToken + '_' +
                                     std::to_string(lane) + ".dat";

        std::ofstream outs(filename.c_str());
        for (utype i = 0; i < kNumSamples; ++i) {
            outs << samples0[i] << '\n';
        }
#endif
        // Test range [0, 1)
        testRange<std::greater_equal, std::less>(0.0f, 1.0f, samples0);

        scene_rdl2::fb_util::RunningStats<> stats;
        for (auto f : samples0) {
            stats.push(f);
        }

        // If these were purely random values, we would have a .95^V chance of
        // success over all of our lanes (or, more formally):
        //
        // choose(V, V) * .95^V * .5^0
        //
        // where V = VLEN
        // (by the binomial distribution)
        // With eight lanes, we only have a 66% chance of success!  Thankfully,
        // we should be better distributed than purely random values.
        testMean(stats, 0.5f, kUniformUnitSquareVariance1D);
        testVariance(stats,   kUniformUnitSquareVariance1D);
        testKolmogorovSmirnov(samples0, UniformCDFContinuous<float>(0.0f, 1.0f));
    }
}

void doTestISPCIntegratorDistribution2D(Sample2DFunction getSamples, const char* fileToken)
{
    // Try to take enough samples that we get out of the pre-computed range for
    // things like Poisson disk or best-candidate.
    const utype kNumSamples = 4096;
    float samples0[kNumSamples];
    float samples1[kNumSamples];

    for (utype lane = 0; lane < VLEN; ++lane) {
        getSamples(samples0, samples1, lane, kNumSamples);

#define TID2D_WRITE_FILE 0
#if TID2D_WRITE_FILE
        const std::string filename = std::string("integrator2D_") +
                                     fileToken + '_' +
                                     std::to_string(lane) + ".dat";

        std::ofstream outs(filename.c_str());
        for (utype i = 0; i < kNumSamples; ++i) {
            outs << samples0[i] << ' ' << samples1[i] << '\n';
        }
#endif
        // Test range [0, 1)
        testRange<std::greater_equal, std::less>(0.0f, 1.0f, samples0);
        testRange<std::greater_equal, std::less>(0.0f, 1.0f, samples1);

        scene_rdl2::fb_util::RunningStats<> stats[2];
        for (auto f : samples0) {
            stats[0].push(f);
        }
        for (auto f : samples1) {
            stats[1].push(f);
        }

        // If these were purely random values, we would have a .95^V chance of
        // success over all of our lanes (or, more formally):
        //
        // choose(V, V) * .95^V * .5^0
        //
        // where V = VLEN
        // (by the binomial distribution)
        // With eight lanes, we only have a 66% chance of success!  Thankfully,
        // we should be better distributed than purely random values.
        testMean(stats[0], 0.5f, kUniformUnitSquareVariance1D);
        testMean(stats[1], 0.5f, kUniformUnitSquareVariance1D);
        testVariance(stats[0],   kUniformUnitSquareVariance1D);
        testVariance(stats[1],   kUniformUnitSquareVariance1D);
        testKolmogorovSmirnov(samples0, UniformCDFContinuous<float>(0.0f, 1.0f));
        testKolmogorovSmirnov(samples1, UniformCDFContinuous<float>(0.0f, 1.0f));
    }
}

void TestSampler::testIntegratorDistribution()
{
    doTestIntegratorDistribution<1>();
    doTestIntegratorDistribution<2>();
#if 0
    doTestISPCIntegratorDistribution1D(&ispc::PBRTest_IntegratorSample1DIndefiniteSize, "indefinite");
    doTestISPCIntegratorDistribution1D(&ispc::PBRTest_IntegratorSample1DDefiniteSize, "definite");
    doTestISPCIntegratorDistribution2D(&ispc::PBRTest_IntegratorSample2DIndefiniteSize, "indefinite");
    doTestISPCIntegratorDistribution2D(&ispc::PBRTest_IntegratorSample2DDefiniteSize, "definite");
#endif
}

namespace {

uint32_t countBits(uint32_t v)
{
    uint32_t c;
    for (c = 0; v; ++c) {
        v &= v - 1; // clear the least significant bit set
    }
    return c;
}

uint32_t hammingDistance(uint32_t a, uint32_t b)
{
    return countBits(a ^ b);
}
} // anonymous namespace

void TestSampler::testISPCSequenceID()
{
    alignas(kSIMDAlignment) int32_t seed[kSIMDSize];
    alignas(kSIMDAlignment) int32_t output[kSIMDSize];
    std::iota(seed, seed + kSIMDSize, 0);

    typedef typename std::remove_cv<decltype(kSIMDSize)>::type IntType;
    IntType hammingSum;
    const IntType arbitraryCheck = 8;

    hammingSum = 0;
    ispc::PBRTest_SequenceID1(seed, output, 1);
    for (IntType i = 1; i < kSIMDSize; ++i) {
        hammingSum += hammingDistance(output[i-1], output[i]);
    }
    // We want neighboring values to generally be far apart. Check to see that
    // the average is greater than our arbitrary value.
    CPPUNIT_ASSERT(hammingSum > (kSIMDSize - 1) * arbitraryCheck);

    hammingSum = 0;
    ispc::PBRTest_SequenceID2(seed, output, 1, 2);
    for (IntType i = 1; i < kSIMDSize; ++i) {
        hammingSum += hammingDistance(output[i-1], output[i]);
    }
    // We want neighboring values to generally be far apart. Check to see that
    // the average is greater than our arbitrary value.
    CPPUNIT_ASSERT(hammingSum > (kSIMDSize - 1) * arbitraryCheck);

    hammingSum = 0;
    ispc::PBRTest_SequenceID3(seed, output, 1, 2, 3);
    for (IntType i = 1; i < kSIMDSize; ++i) {
        hammingSum += hammingDistance(output[i-1], output[i]);
    }
    // We want neighboring values to generally be far apart. Check to see that
    // the average is greater than our arbitrary value.
    CPPUNIT_ASSERT(hammingSum > (kSIMDSize - 1) * arbitraryCheck);
}

void TestSampler::testSequenceID()
{
    alignas(kSIMDAlignment) int32_t seed[kSIMDSize];
    alignas(kSIMDAlignment) int32_t output[kSIMDSize];
    std::iota(seed, seed + kSIMDSize, 0);

    typedef typename std::remove_cv<decltype(kSIMDSize)>::type IntType;

    {
        ispc::PBRTest_SequenceID0(seed, output);
        SequenceID scalarSID;
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID1(seed, output, 1);
        SequenceID scalarSID(1);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID2(seed, output, 1, 2);
        SequenceID scalarSID(1, 2);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID2(seed, output, 2, 1);
        SequenceID scalarSID(2, 1);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID3(seed, output, 1, 2, 1);
        SequenceID scalarSID(1, 2, 1);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID4(seed, output, 1, 1, 1, 1);
        SequenceID scalarSID(1, 1, 1, 1);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID4(seed, output, 1, 1, 1, 1);
        SequenceID scalarSID(1, 1, 1, 1);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID5(seed, output, 1, 1, 1, 1, 2);
        SequenceID scalarSID(1, 1, 1, 1, 2);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        ispc::PBRTest_SequenceID5(seed, output, 2, 1, 1, 1, 2);
        SequenceID scalarSID(2, 1, 1, 1, 2);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar = scalarSID.getHash(seed[i]);
            CPPUNIT_ASSERT(scalar == output[i]);
        }
    }

    {
        SequenceID sid0(1, 2);
        SequenceID sid1(2, 1);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar0 = sid0.getHash(seed[i]);
            const auto scalar1 = sid1.getHash(seed[i]);

            // Technically, these could be the same, but the odds should be
            // astronomically low.
            CPPUNIT_ASSERT(scalar0 != scalar1);
        }
    }

    {
        SequenceID sid0(1, 2, 1);
        SequenceID sid1(1, 2);
        for (IntType i = 0; i < kSIMDSize; ++i) {
            const auto scalar0 = sid0.getHash(seed[i]);
            const auto scalar1 = sid1.getHash(seed[i]);

            // Technically, these could be the same, but the odds should be
            // astronomically low.
            CPPUNIT_ASSERT(scalar0 != scalar1);
        }
    }
}

template <typename Iter>
static bool isPermutation(Iter first, Iter last)
{
    std::cout << "Checking : ";

    using value_type = typename std::iterator_traits<Iter>::value_type;
    std::vector<value_type> results;
    for ( ; first != last; ++first) {
        std::cout << *first << " ";
        results.push_back(*first);
    }
    std::cout << std::endl;

    std::sort(results.begin(), results.end());
    if (std::unique(results.begin(), results.end()) != results.end()) {
        return false;
    }

    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i] != i) {
            return false;
        }
    }
    return true;
}

void TestSampler::testISCPPermutations()
{
    constexpr int32_t kMaxStreamSize = 19;
    using ArrayT = std::array<uint32_t, kMaxStreamSize>;
    alignas(kSIMDAlignment) ArrayT output;

    constexpr uint32_t ntests = 32u;
    static_assert(kMaxStreamSize >= 8u, "We test up to 8 elements. We need enough memory.");
    for (uint32_t permSize = 1u; permSize <= 8u; ++permSize) {
        for (uint32_t t = 0; t < ntests; ++t) {
            const uint32_t seed = t;
            ispc::PBRTest_testPermutation(kMaxStreamSize, seed, permSize, output.data());
            CPPUNIT_ASSERT(isPermutation(output.data(), output.data() + permSize));
        }
    }

    constexpr uint32_t kMaxPermSize = 8u;
    static_assert(kMaxStreamSize >= kMaxPermSize*2u, "We test two independent permutations. We need enough memory.");
    for (uint32_t t = 0; t < ntests; ++t) {
        const uint32_t seed = t;
        ispc::PBRTest_testPermutation(16u, seed, kMaxPermSize, output.data());

        // This should give us two independent 8-element permutations of [0, 8)
        CPPUNIT_ASSERT(isPermutation(output.data(), output.data() + kMaxPermSize));
        CPPUNIT_ASSERT(isPermutation(output.data() + kMaxPermSize, output.data() + kMaxPermSize*2u));
    }

    for (uint32_t t = 0; t < ntests; ++t) {
        const uint32_t seed = t;
        ispc::PBRTest_testPermutationSequence(kMaxStreamSize, seed, kMaxStreamSize, output.data());
        CPPUNIT_ASSERT(isPermutation(output.data(), output.data() + kMaxStreamSize));
    }
}

struct TestPoint
{
    float x;
    float y;
    int row;
    int col;
    char id;
};

float  getPrimaryValue0(const TestPoint& s)   { return s.x; }
float& getPrimaryValue0(TestPoint& s)         { return s.x; }
float  getPrimaryValue1(const TestPoint& s)   { return s.y; }
float& getPrimaryValue1(TestPoint& s)         { return s.y; }

template <>
struct GenerateRandomPointImpl<TestPoint>
{
    template <typename RNG>
    static TestPoint apply(RNG& rng)
    {
        TestPoint s;
        s.x    = rng();
        s.y    = rng();
        s.row  = 0;
        s.col  = 0;
        s.id   = 'x';
        return s;
    }
};

const int kTestDimensions = 4;

void verifySSP(const SpatialSamplePartition<TestPoint, kTestDimensions>& ssp)
{
    // In rotating on the torus, we expect rows and columns to maintain their
    // integrity. A value always has the same row mates, just in a different
    // rotation, and a value always has the same column mates, just in a
    // different rotation.

    for (int y = 0; y < kTestDimensions; ++y) {
        const int row = ssp(0, y, 0).row;
        for (int x = 1; x < kTestDimensions; ++x) {
            CPPUNIT_ASSERT(row == ssp(x, y, 0).row);
        }
    }

    for (int x = 0; x < kTestDimensions; ++x) {
        const int col = ssp(x, 0, 0).col;
        for (int y = 1; y < kTestDimensions; ++y) {
            CPPUNIT_ASSERT(col == ssp(x, y, 0).col);
        }
    }
}

std::string buildID(const SpatialSamplePartition<TestPoint, kTestDimensions>& ssp)
{
    std::string ret;
    ret.reserve(kTestDimensions*kTestDimensions);
    for (int y = 0; y < kTestDimensions; ++y) {
        for (int x = 1; x < kTestDimensions; ++x) {
            ret.push_back(ssp(x, y, 0).id);
        }
    }
    return ret;
}

void TestSampler::testSamplePartition()
{
    const float spacing = 1.0f/kTestDimensions;
    const float offset = spacing/2.0f;

    std::vector<TestPoint> input = {
        { offset + spacing * 0, offset + spacing * 0, 0, 0, 'a' },
        { offset + spacing * 1, offset + spacing * 0, 0, 1, 'b' },
        { offset + spacing * 2, offset + spacing * 0, 0, 2, 'c' },
        { offset + spacing * 3, offset + spacing * 0, 0, 3, 'd' },

        { offset + spacing * 0, offset + spacing * 1, 1, 0, 'e' },
        { offset + spacing * 1, offset + spacing * 1, 1, 1, 'f' },
        { offset + spacing * 2, offset + spacing * 1, 1, 2, 'g' },
        { offset + spacing * 3, offset + spacing * 1, 1, 3, 'h' },

        { offset + spacing * 0, offset + spacing * 2, 2, 0, 'i' },
        { offset + spacing * 1, offset + spacing * 2, 2, 1, 'j' },
        { offset + spacing * 2, offset + spacing * 2, 2, 2, 'k' },
        { offset + spacing * 3, offset + spacing * 2, 2, 3, 'l' },

        { offset + spacing * 0, offset + spacing * 3, 3, 0, 'm' },
        { offset + spacing * 1, offset + spacing * 3, 3, 1, 'n' },
        { offset + spacing * 2, offset + spacing * 3, 3, 2, 'o' },
        { offset + spacing * 3, offset + spacing * 3, 3, 3, 'p' }
    };

    SpatialSamplePartition<TestPoint, kTestDimensions> ssp(input.cbegin(), input.cend());
    CPPUNIT_ASSERT(ssp.numPixelSamples() == 1);

    // Since we're rotating on the torus, we should only have
    // kTestDimensions*kTestDimensions permutations (let's run through more
    // than that...). We do NOT have all of the permutations of a set (i.e.
    // kTestDimensions!).
    std::set<std::string> ids;
    for (int i = 0; i < kTestDimensions*kTestDimensions*3; ++i) {
        ssp.rotate(i);
        verifySSP(ssp);
        ids.insert(buildID(ssp));
    }
    CPPUNIT_ASSERT_EQUAL(kTestDimensions*kTestDimensions, static_cast<int>(ids.size()));
}

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestSampler);


