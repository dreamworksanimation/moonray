// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestDistribution.cc
/// $Id$
///


#include "TestDistribution.h"
#include "TestDistribution_ispc_stubs.h"

#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/sampler/IntegratorSample.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/util/Random.h>

#include <iostream>
#include <numeric>


namespace moonray {
namespace pbr {


using namespace scene_rdl2::math;


//----------------------------------------------------------------------------

static void
generate2DSequence(uint32_t size, FloatArray &r1, FloatArray &r2)
{
    IntegratorSample2D sampler(SequenceID{}, size, 0);
    r1.resize(size);
    r2.resize(size);

    float s[2];
    for (uint32_t i = 0; i < size; ++i) {
        constexpr unsigned depth = 0;
        sampler.getSample(s, depth);
        r1.at(i) = s[0];
        r2.at(i) = s[1];
    }
}

static void
sampleDistribution2D(const Distribution2D &dist, const FloatArray &r1, const FloatArray &r2,
                     FloatArray &u, FloatArray &v)
{
    uint32_t size = uint32_t(r1.size());

    u.resize(size);
    v.resize(size);

    for (uint32_t i=0; i < size; ++i) {
        Vec2f uv;
        float pdf;
        dist.sample(r1[i], r2[i], &uv, &pdf, TEXTURE_FILTER_NEAREST);
        u[i] = uv[0];
        v[i] = uv[1];
    }
}


static void
sampleImageDistribution(const ImageDistribution &dist, const FloatArray &r1, const FloatArray &r2,
                        FloatArray &u, FloatArray &v)
{
    uint32_t size = uint32_t(r1.size());

    u.resize(size);
    v.resize(size);

    for (uint32_t i=0; i < size; ++i) {
        Vec2f uv;
        float pdf;
        dist.sample(r1[i], r2[i], 0.0f, &uv, &pdf, TEXTURE_FILTER_NEAREST);
        u[i] = uv[0];
        v[i] = uv[1];
    }
}


//----------------------------------------------------------------------------

void
TestDistribution::testUniform()
{
    static const int distSize = 100;
    Distribution2D dist(distSize, distSize);
    for (int v=0; v < distSize; v++) {
        for (int u=0; u < distSize; u++) {
            dist.setWeight(u, v, 3.5f);
        }
    }
    dist.tabulateCdf();

    static const int maxPower = 13;
    uint32_t size = 1 << maxPower;
    FloatArray r1, r2;
    generate2DSequence(size, r1, r2);

    FloatArray u, v;
    sampleDistribution2D(dist, r1, r2, u, v);
    save2dTxtFile("testDistributionUniform.txt", u, v);

    bool equal = asCppBool(ispc::sampleDistribution2D(dist.asIspc(), size,
            &(r1[0]), &(r2[0]), &(u[0]), &(v[0])));
    save2dTxtFile("testDistributionUniformIspc.txt", u, v);

    CPPUNIT_ASSERT(equal);
}


void
TestDistribution::testGradient()
{
    static const int distSize = 100;
    Distribution2D dist(distSize, distSize);
    for (int v=0; v < distSize; v++) {
        for (int u=0; u < distSize; u++) {
            dist.setWeight(u, v, scene_rdl2::math::pow(scene_rdl2::math::cos(float(u + v) / (2.0f * distSize) *
                    scene_rdl2::math::sHalfPi), 4.0f));
        }
    }
    dist.tabulateCdf();

    static const int maxPower = 13;
    uint32_t size = 1 << maxPower;
    FloatArray r1, r2;
    generate2DSequence(size, r1, r2);

    FloatArray u, v;
    sampleDistribution2D(dist, r1, r2, u, v);
    save2dTxtFile("testDistributionGradient.txt", u, v);

    bool equal = asCppBool(ispc::sampleDistribution2D(dist.asIspc(), size,
            &(r1[0]), &(r2[0]), &(u[0]), &(v[0])));
    save2dTxtFile("testDistributionGradientIspc.txt", u, v);

    CPPUNIT_ASSERT(equal);
}


void
TestDistribution::testImage(const std::string &path, const std::string &filename)
{
    std::cout << "testing image: " << path + filename << std::endl;

    try {
        ImageDistribution dist(path + filename, Distribution2D::PLANAR);

        static const int maxPower = 13;
        uint32_t size = 1 << maxPower;
        FloatArray r1, r2;
        generate2DSequence(size, r1, r2);

        FloatArray u, v;
        sampleImageDistribution(dist, r1, r2, u, v);
        std::string outFilename = "testDistributionImage." + filename + ".txt";
        save2dTxtFile(outFilename, u, v);

        bool equal = asCppBool(ispc::sampleImageDistribution(dist.asIspc(), size,
                &(r1[0]), &(r2[0]), &(u[0]), &(v[0])));
        std::string outFilenameIspc = "testDistributionImage." + filename + ".ispc.txt";
        save2dTxtFile(outFilenameIspc, u, v);

        CPPUNIT_ASSERT(equal);

    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        CPPUNIT_ASSERT(0);
        return;
    }
}


void
TestDistribution::testImages()
{
    std::string path("/work/rd/raas/maps/env_maps/");

    testImage(path, "parking_lot-med.exr");
    testImage(path, "parking_lot-vsmall.exr");
    testImage(path, "parking_lot-vvsmall.exr");

//    testImage(path, "burning_man-med.exr");
//    testImage(path, "green_house-med.exr");
//    testImage(path, "monument_valley-med.exr");
//    testImage(path, "papermill_ruin-med.exr");
}

namespace
{
    template <typename Iter>
    Distribution1D createDistribution(Iter first, Iter last)
    {
        Distribution1D dist(std::distance(first, last));
        for (std::size_t i = 0; first != last; ++first, ++i) {
            dist.setWeight(i, *first);
        }
        dist.tabulateCdf();
        return dist;
    }
}

// Stochastically sample the distribution to estimate its mean/expected value.
float
stochasticDiscreteMean(const std::vector<float>& vals, unsigned samples)
{
    const Distribution1D dist = createDistribution(vals.begin(), vals.end());
    scene_rdl2::util::Random rnd(0xdeadbeef);

    float mean = 0.0f;
    for (unsigned i = 0; i < samples; ++i) {
        const std::size_t idx = dist.sampleDiscrete(rnd.getNextFloat());
        testAssert(idx < vals.size(), "Invalid index");
        mean += vals[idx];
    }

    return mean / static_cast<float>(samples);
}

// This function should return close to the sum of the values.
//   P(x1) * x1 / P(x1)
// + P(x2) * x2 / P(x2)
// + ...
// + P(xn) * xn / P(xn)
// = x1 + x2 + ... + xn
float
stochasticDiscreteSum(const std::vector<float>& vals, unsigned samples)
{
    const Distribution1D dist = createDistribution(vals.begin(), vals.end());
    scene_rdl2::util::Random rnd(0xdeadbeef);

    float mean = 0.0f;
    for (unsigned i = 0; i < samples; ++i) {
        float pdf;
        const std::size_t idx = dist.sampleDiscrete(rnd.getNextFloat(), &pdf);
        testAssert(idx < vals.size(), "Invalid index");
        mean += vals[idx] / pdf;
    }

    return mean / static_cast<float>(samples);
}


// Calculate expected value of sampling from vals.
float
discreteExpectedValue(const std::vector<float>& vals)
{
    const float sum = std::accumulate(vals.begin(), vals.end(), 0.0f);
    return std::accumulate(vals.begin(), vals.end(), 0.0f, [sum](float init, float x) { return init + x/sum * x; });
}

// Calculate variance of sampling from vals.
float
discreteVariance(const std::vector<float>& vals)
{
    const float sum = std::accumulate(vals.begin(), vals.end(), 0.0f);
    const float mean = discreteExpectedValue(vals);

    float s2 = 0.0f;
    for (std::vector<float>::const_iterator it = vals.begin(); it != vals.end(); ++it) {
        const float x = *it;
        const float px = x/sum;
        s2 += px * (x - mean)*(x - mean);
    }

    return s2;
}

// Using the central limit theorem, if we average enough results, we get a normal distribution.
// We're going to use this fact for testing the distribution.
void
testDiscreteMean(const std::vector<float>& vals)
{
    const unsigned kNumSamples = 100000;

    // Determine the actual mean and variance for the distribution.
    const float mean = discreteExpectedValue(vals);
    const float s2   = discreteVariance(vals);
    const float s    = scene_rdl2::math::sqrt(s2);

    // The standard deviation of the mean is 1/sqrt(N) smaller than the
    // standard deviation of the individual samples.
    const float sOfMean = s / scene_rdl2::math::sqrt(static_cast<float>(kNumSamples));
    const float testedMean = stochasticDiscreteMean(vals, kNumSamples);

    // The mean should be within 2 standard deviations about 95% of the time.
    // This means that this test is _expected_ to fail about 5% of the time!
    const float rangeMin = mean - 2.0f * sOfMean;
    const float rangeMax = mean + 2.0f * sOfMean;

    testAssert(testedMean >= rangeMin && testedMean <= rangeMax, "Mean not in expected range (this test will fail 5 percent of the time)");
}

// Place samples into buckets (this is easy for sampling discrete. They have to
// be arbitrarily sized for sampling continuous). Use the distribution of in
// the buckets in a chi-square test.
void
testDiscreteChiSquare(const std::vector<float>& vals, unsigned samples)
{
    if (vals.size() < 1) {
        return;
    }

    const Distribution1D dist = createDistribution(vals.begin(), vals.end());
    scene_rdl2::util::Random rnd(0xdeadbeef);

    std::vector<unsigned> buckets(vals.size());

    for (unsigned i = 0; i < samples; ++i) {
        const std::size_t idx = dist.sampleDiscrete(rnd.getNextFloat());
        testAssert(idx < vals.size(), "Invalid index");
        ++buckets[idx];
    }

    float chiSquare = 0.0f;
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        const float expectedNumSamples = dist.pdfDiscrete(i) * static_cast<float>(samples);
        const float numSamplesObserved = static_cast<float>(buckets[i]);
        const float d = numSamplesObserved - expectedNumSamples;
        if (expectedNumSamples > 0) {
            chiSquare += (d*d)/expectedNumSamples;
        }
    }

    // For a large amount of samples, a chi-square distribution with b buckets
    // approximates a normal distribution with a mean of b-1 and a variance of
    // 2b-2.
    const float chiSquareMean = static_cast<float>(buckets.size() - 1u);
    const float chiSquareS2   = static_cast<float>(2u*buckets.size() - 2u);
    const float chiSquareS    = scene_rdl2::math::sqrt(chiSquareS2);

    // The mean should be within 2 standard deviations about 95% of the time.
    // This means that this test is _expected_ to fail about 5% of the time!
    const float rangeMin = chiSquareMean - 2.0f * chiSquareS;
    const float rangeMax = chiSquareMean + 2.0f * chiSquareS;

    testAssert(chiSquare >= rangeMin && chiSquare <= rangeMax, "Chi-square not in expected range (this test will fail 5 percent of the time)");
}

void
TestDistribution::testDiscrete()
{
    // A version that's really easy to verify by intuition. We're going to
    // sample the second value 10 times as many times as the first.
    std::vector<float> v1;
    v1.push_back(1.0f);
    v1.push_back(10.0f);

    // Make sure that a uniform collection works.
    std::vector<float> v2;
    v2.push_back(1.0f);
    v2.push_back(1.0f);

    // Degenerate.
    std::vector<float> v3;
    v3.push_back(0.0f);
    v3.push_back(0.0f);

    std::vector<float> v4;
    v4.push_back(1.0f);
    v4.push_back(5.0f);
    v4.push_back(3.0f);
    v4.push_back(8.0f);
    v4.push_back(2.0f);

    // Make sure a zero value doesn't ruin things.
    std::vector<float> v5;
    v5.push_back(1.0f);
    v5.push_back(5.0f);
    v5.push_back(0.0f);
    v5.push_back(8.0f);
    v5.push_back(2.0f);

    // Make sure consecutive equal values work as expected.
    std::vector<float> v6;
    v6.push_back(1.0f);
    v6.push_back(5.0f);
    v6.push_back(5.0f);
    v6.push_back(0.0f);
    v6.push_back(2.0f);

    // A version in which the mean should be vastly different from the variance.
    std::vector<float> v7;
    v7.push_back(100.0f);
    v7.push_back(105.0f);
    v7.push_back(105.0f);
    v7.push_back(100.0f);
    v7.push_back(102.0f);

    // Large collection.
    std::vector<float> v8;
    v8.push_back(100.0f);
    v8.push_back(105.0f);
    v8.push_back(105.0f);
    v8.push_back(100.0f);
    v8.push_back(106.0f);
    v8.push_back(108.0f);
    v8.push_back(102.0f);
    v8.push_back(101.0f);
    v8.push_back(109.0f);
    v8.push_back(100.0f);
    v8.push_back(102.0f);
    v8.push_back(103.0f);
    v8.push_back(103.0f);
    v8.push_back(107.0f);
    v8.push_back(108.0f);
    v8.push_back(107.0f);
    v8.push_back(103.0f);
    v8.push_back(105.0f);
    v8.push_back(101.0f);

    std::vector<float> v9;
    v9.push_back(102.0f);
    v9.push_back(108.0f);
    v9.push_back(108.0f);
    v9.push_back(109.0f);
    v9.push_back(101.0f);
    v9.push_back(103.0f);
    v9.push_back(108.0f);
    v9.push_back(102.0f);
    v9.push_back(103.0f);
    v9.push_back(105.0f);
    v9.push_back(108.0f);
    v9.push_back(109.0f);
    v9.push_back(108.0f);
    v9.push_back(108.0f);
    v9.push_back(106.0f);
    v9.push_back(102.0f);
    v9.push_back(106.0f);
    v9.push_back(102.0f);
    v9.push_back(102.0f);

    testDiscreteMean(v1);
    testDiscreteMean(v2);
    //testDiscreteMean(v3); // TODO: The test doesn't know how to handle this case.
    testDiscreteMean(v4);
    testDiscreteMean(v5);
    testDiscreteMean(v6);
    testDiscreteMean(v7);
    testDiscreteMean(v8);
    testDiscreteMean(v9);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v1, 1000), std::accumulate(v1.begin(), v1.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v2, 1000), std::accumulate(v2.begin(), v2.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v3, 1000), std::accumulate(v3.begin(), v3.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v4, 1000), std::accumulate(v4.begin(), v4.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v5, 1000), std::accumulate(v5.begin(), v5.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v6, 1000), std::accumulate(v6.begin(), v6.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v7, 1000), std::accumulate(v7.begin(), v7.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v8, 1000), std::accumulate(v8.begin(), v8.end(), 0.0f), 0.001f);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(stochasticDiscreteSum(v9, 1000), std::accumulate(v9.begin(), v9.end(), 0.0f), 0.001f);

    // This test works best with large distributions.
    testDiscreteChiSquare(v9, 100000000);
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


CPPUNIT_TEST_SUITE_REGISTRATION(moonray::pbr::TestDistribution);


