// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestBsdfCommon.cc
/// $Id$
///

#include "TestBsdfCommon.h"
#include "TestUtil.h"


#include <moonray/common/mcrt_util/RingBuffer.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/shading/Util.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>

#include <tbb/tbb.h>

#include <thread>
#include <vector>

namespace moonray {
namespace pbr {


using namespace scene_rdl2::math;


// Un-comment the line below to run only the integrals assertions
//#define ONLY_ASSERT_INTEGRALS


// Un-comment the line below to run single-threaded (for debugging)
//#define RUN_SINGLE_THREADED

const int TestBsdfSettings::sMaxSamplesPerLobe;

// To add a BSDF test, you need the following (as passed into TestBsdfCommon.cc runTest()):
//
// * A function (or function object) that runs the test for a single thread.
// * A type that holds the results from a single thread (Result).
// * A type used to accumulate the results from various threads (Aggregate).
//   (The Result type and the Aggregate type are often the same type).
// * An overloaded function accumulate() which takes aggregate type by reference and the result type by const reference.
//   This function will accumulate the results from a completed thread.
//
// void accumulate(Aggregate&, const Result&);
//
// * A check() function that takes TestBsdfSettings by const reference and the result type by const reference. This
//   function is run once all threads have completed and is where the unit test assertions should be placed.
//
// void check(const TestBsdfSettings&, const Result&);
//
// TestBsdfCommon.cc's runTest function will set up producer/consumer queues to dispatch the work to threads and
// accumulate the results from the threads.
//
// Each task function will be passed a random seed and a random stream number (for use with our scene_rdl2::util::Random
// class).

namespace {

constexpr unsigned numTasks = 512u;
constexpr unsigned kLogTaskQueueSize   = 9u;
constexpr unsigned kLogResultQueueSize = 9u;

template <typename T>
using TaskQueueType   = RingBufferSingleProducer<T, kLogTaskQueueSize>;
template <typename T>
using ResultQueueType = RingBufferSingleConsumer<T, kLogResultQueueSize>;

void accumulate(TestBsdfConsistencyTask::Result& aggregate, const TestBsdfConsistencyTask::Result& task)
{
    aggregate.mSampleCount           += task.mSampleCount;
    aggregate.mZeroEvalPdfCount      += task.mZeroEvalPdfCount;
    aggregate.mZeroEvalRecipPdfCount += task.mZeroEvalRecipPdfCount;
    aggregate.mZeroSamplePdfCount    += task.mZeroSamplePdfCount;
}

void check(const TestBsdfSettings&, const TestBsdfConsistencyTask::Result& results)
{
    const auto sampleCount           = results.mSampleCount;
    const auto zeroSamplePdfCount    = results.mZeroSamplePdfCount;
    const auto zeroEvalPdfCount      = results.mZeroEvalPdfCount;
    const auto zeroEvalRecipPdfCount = results.mZeroEvalRecipPdfCount;

    MNRY_ASSERT_REQUIRE(sampleCount > 0);

    if (zeroSamplePdfCount > 0) {
        printInfo("sample() returned a zero probability (%f%%)",
                  static_cast<float>(zeroSamplePdfCount) / sampleCount * 100.0);
    }
    if (zeroEvalPdfCount > 0) {
        printInfo("eval() returned a zero probability (%f%%)",
                  static_cast<float>(zeroEvalPdfCount) / sampleCount * 100.0);
    }
    if (zeroEvalRecipPdfCount > 0) {
        printInfo("eval_recip() returned a zero probability (%f%%)",
                  static_cast<float>(zeroEvalRecipPdfCount) / sampleCount * 100.0);
    }
}

void accumulate(TestBsdfPdfIntegralTask::Result& aggregate, const TestBsdfPdfIntegralTask::Result& task)
{
    aggregate.mIntegral              += task.mIntegral;
    aggregate.mSampleCount           += task.mSampleCount;
    aggregate.mSpherical              = task.mSpherical;
    aggregate.mDoAssert               = task.mDoAssert;
}

void check(const TestBsdfSettings& test, const TestBsdfPdfIntegralTask::Result& results)
{
    MNRY_ASSERT_REQUIRE(results.mSampleCount > 0);

    float integral = results.mIntegral;
    if (results.mSpherical) {
        // Divide by pdf and number of samples to get the MC estimate
        integral *= sFourPi / (results.mSampleCount);

        // If the integral is zero, the bsdf is likely made only of delta functions
        printDebug("pdf integral sphere      = %f", integral);
        if (results.mDoAssert && integral != 0.0) {
            testAssert(isValidPdfIntegral(integral, test.toleranceIntegral),
                       "pdf() does not integrate to 1.0 over the sphere");
        }
    } else {
        // Divide by pdf and number of samples to get the MC estimate
        integral *= sTwoPi / results.mSampleCount;

        // If the integral is zero, the bsdf is likely made only of delta functions
        printDebug("pdf integral hemisphere  = %f", integral);
    }
}

void accumulate(TestBsdfEvalIntegralTask::Result& aggregate, const TestBsdfEvalIntegralTask::Result& task)
{
    //PRINT(task.mSampleCount);
    aggregate.mIntegralUniform      += task.mIntegralUniform;
    aggregate.mIntegralImportance   += task.mIntegralImportance;
    aggregate.mSampleCount          += task.mSampleCount;
    aggregate.mDoAssert              = task.mDoAssert;
}

void check(const TestBsdfSettings& test, const TestBsdfEvalIntegralTask::Result& results)
{
    MNRY_ASSERT_REQUIRE(results.mSampleCount > 0);

    // Divide by pdf and/or number of samples to get the MC estimate
    const auto integralUniform    = results.mIntegralUniform / results.mSampleCount;
    const auto integralImportance = results.mIntegralImportance / results.mSampleCount;

    // Test energy conservation
    bool gray;
    if (isEqual(integralImportance[0], integralImportance[1], 1e-4f)  &&
        isEqual(integralImportance[0], integralImportance[2], 1e-4f)) {
        gray = true;
        printDebug("eval integral importance = %f",
                   integralImportance[0]);
    } else {
        gray = false;
        printDebug("eval integral importance = (%f, %f, %f)",
                   integralImportance[0], integralImportance[1], integralImportance[2]);
    }

    // If the integral is zero, the filter is likely a delta function
    if (isEqual(integralUniform, sBlack)) {
        printDebug("bsdf is a delta-function");
        return;
    }

    // Test energy conservation again
    if (gray) {
        printDebug("eval integral uniform    = %f", integralUniform[0]);
    } else {
        printDebug("eval integral uniform    = (%f, %f, %f)",
                   integralUniform[0], integralUniform[1], integralUniform[2]);
    }

    // Warn about energy conservation
    if (!isValidEvalIntegral(integralUniform, test.toleranceIntegral * 4.0f)) {
        printWarning("eval() uniform is not energy preserving");
    }
    if (!isValidEvalIntegral(integralImportance, test.toleranceIntegral)) {
        printWarning("eval() importance is not energy preserving");
    }

#if 0
    // This test has been failing for a long time...

    // Test that sample() is unbiased, making sure that the sampling behavior
    // is consistent with its returned pdf. We make sure that the uniform
    // sampling and importance sampling produce the same integral.
    bool sampleIsBiased = false;
    for (int i = 0; i < (gray ? 1 : 3); ++i) {
        const float error = computeError(integralUniform[i], integralImportance[i]);
        printDebug("importance vs. uniform sampling error = %f", error);
        sampleIsBiased |= error > test.toleranceIntegral;
    }

    if (results.mDoAssert) {
        testAssert(!sampleIsBiased,
                   "sample() is a biased importance sampling scheme");
    }
#endif

    // TODO: To really make sure that the probability of choosing a sample
    // when calling sample() actually equals the returned pdf or pdf(), we would
    // need to compare the integrals above over many small domains that overlap
    // the bsdf lobe (tricky!).
}

// This function populates the task queue.
template <typename Task, typename... Args>
void runProducer(TaskQueueType<Task>& queueTasks, unsigned totalSamples, Args&&... args)
{
    using namespace std::chrono_literals;
    scene_rdl2::util::Random seedRNG(2285938u, 185283u);

    unsigned tasksToDistribute = numTasks;
    unsigned samplesLeft = totalSamples;
    unsigned previousStart = 0;
    while (tasksToDistribute > 0) {
        const unsigned samples = samplesLeft/tasksToDistribute;
        const std::uint32_t seed = seedRNG.getNextUInt();
        const std::uint32_t stream = tasksToDistribute * 64u; // Allow space for ISPC code to modify lane values.
        Task task(previousStart, previousStart + samples, seed, stream, std::forward<Args>(args)...);
        queueTasks.push(std::move(task));
        --tasksToDistribute;
        samplesLeft -= samples;
        previousStart += samples;
    }
    MNRY_ASSERT(samplesLeft == 0);
    MNRY_ASSERT(tasksToDistribute == 0);
}

// This function consumes results until toConsume is zero.
template <typename Task, typename Result>
void runTasks(std::atomic<unsigned>& toConsume, TaskQueueType<Task>& queueTasks, ResultQueueType<Result>& queueResults)
{
    while (true) {
        unsigned count = toConsume.load();
        if (count == 0) {
            return;
        }
        while (!toConsume.compare_exchange_weak(count, count - 1)) {
            if (count == 0) {
                return;
            }
        }

        Task task = queueTasks.pop();
        Result result = task();
        queueResults.push(std::move(result));
    }
}

template <typename Aggregate, typename Result>
Aggregate runConsumer(unsigned toConsume, ResultQueueType<Result>& queueResults)
{
    Aggregate aggregate;

    // This is threaded: we can't just rely on the queue being empty.
    for ( ; toConsume > 0; --toConsume) {
        accumulate(aggregate, queueResults.pop());
    }
    MNRY_ASSERT(queueResults.empty());

    return aggregate;
}
} // anonymous namespace

//----------------------------------------------------------------------------

#if 0
template <typename Aggregate, typename Task, typename Result, typename... ProducerArgs>
static void runTest(const TestBsdfSettings& test, int sampleCount, ProducerArgs&&... args)
{
#if defined(RUN_SINGLE_THREADED)
    const unsigned sTaskCount = 1u;
#else
    const unsigned sTaskCount = std::thread::hardware_concurrency() - 1u; // Leave one for the producer
#endif

    TaskQueueType<Task> taskQueue;
    ResultQueueType<Result> resultQueue;

    std::atomic<unsigned> toConsume(numTasks);

#if defined(RUN_SINGLE_THREADED)
    runProducer(taskQueue, sampleCount, test, args...);
#else
    std::thread producerThread([&, sampleCount]() { runProducer(taskQueue, sampleCount, test, args...); });
#endif

    std::vector<std::thread> taskThreads;
    taskThreads.reserve(nthreads);
    for (unsigned i = 0; i < sTaskCount; ++i) {
        taskThreads.emplace_back(&runTasks<Task, Result>, std::ref(toConsume), std::ref(taskQueue), std::ref(resultQueue));
    }

    const auto results = runConsumer<Aggregate>(numTasks, resultQueue);

#if !defined(RUN_SINGLE_THREADED)
    producerThread.join();
#endif
    for (auto& t : taskThreads) {
        t.join();
    }
    MNRY_ASSERT(taskQueue.empty());
    MNRY_ASSERT(resultQueue.empty());

    check(test, results);
}
#else
template <typename Aggregate, typename Task, typename Result, typename... ProducerArgs>
static void runTest(const TestBsdfSettings& test, int sampleCount, ProducerArgs&&... args)
{
#if defined(RUN_SINGLE_THREADED)
    const unsigned sTaskCount = 1u;
#else
    const unsigned sTaskCount = tbb::task_scheduler_init::default_num_threads() - 1u; // Leave one for producer
#endif

    TaskQueueType<Task> taskQueue;
    ResultQueueType<Result> resultQueue;

    std::atomic<unsigned> toConsume(numTasks);

    // While I would love to do this with std::thread and remove a dependency on TBB, because we allocate
    // ThreadLocalStates based on TBB threads, we error on going beyond our TLS pool size.
    tbb::task_group taskGroup;
#if defined(RUN_SINGLE_THREADED)
    runProducer(taskQueue, sampleCount, test, std::forward<ProducerArgs>(args)...);
#else
    taskGroup.run([&taskQueue, sampleCount, &test, &args...]() { runProducer(taskQueue, sampleCount, test, std::forward<ProducerArgs>(args)...); });
#endif

    for (unsigned i = 0; i < sTaskCount; ++i) {
        taskGroup.run([&toConsume, &taskQueue, &resultQueue]() { runTasks<Task, Result>(toConsume, taskQueue, resultQueue); });
    }

    const auto results = runConsumer<Aggregate>(numTasks, resultQueue);

    taskGroup.wait();
    MNRY_ASSERT(taskQueue.empty());
    MNRY_ASSERT(resultQueue.empty());

    check(test, results);
}
#endif

//----------------------------------------------------------------------------

static void
testConsistency(const TestBsdfSettings &test, int sampleCount)
{
    using Task      = TestBsdfConsistencyTask;
    using Result    = TestBsdfConsistencyTask::Result;
    using Aggregate = TestBsdfConsistencyTask::Result;

    runTest<Aggregate, Task, Result>(test, sampleCount);
}

static void
testPdfIntegral(const TestBsdfSettings &test, const Vec3f &wo, int sampleCount, bool doAssert = false)
{
    // Don't include cosine term since we only want to get to the pdf.
    shading::BsdfSlice slice(test.frame.getN(), wo, false, true, ispc::SHADOW_TERMINATOR_FIX_OFF);

    using Task      = TestBsdfPdfIntegralTask;
    using Result    = TestBsdfPdfIntegralTask::Result;
    using Aggregate = TestBsdfPdfIntegralTask::Result;

    runTest<Aggregate, Task, Result>(test, 2 * sampleCount, slice, true, doAssert);
    runTest<Aggregate, Task, Result>(test, 1 * sampleCount, slice, false, doAssert);
}


static void
testEvalIntegral(const TestBsdfSettings &test, const Vec3f &wo, int sampleCount, bool doAssert = false)
{
    // Get ready to sample, including the cosine term since we want to
    // integrate the bsdf * cosine term
    shading::BsdfSlice slice(test.frame.getN(), wo, true, true, ispc::SHADOW_TERMINATOR_FIX_OFF);

    using Task      = TestBsdfEvalIntegralTask;
    using Result    = TestBsdfEvalIntegralTask::Result;
    using Aggregate = TestBsdfEvalIntegralTask::Result;

    runTest<Aggregate, Task, Result>(test, sampleCount, slice, doAssert);

    // TODO: To really make sure that the probability of choosing a sample
    // when calling sample() actually equals the returned pdf or pdf(), we would
    // need to compare the integrals above over many small domains that overlap
    // the bsdf lobe (tricky!).
}


//----------------------------------------------------------------------------

void
runBsdfTest(const TestBsdfSettings &test, int viewAnglesTheta,
        int viewAnglesPhy, int sampleCount)
{
#ifndef ONLY_ASSERT_INTEGRALS
    printInfo("----- testConsistency() -----");
    testConsistency(test, sampleCount);
#endif

    if (test.assertPdfIntegral  ||  test.assertEvalIntegral) {
        printInfo("----- Assert integrals -----");
        float theta = 30.0f / 180.0f * sPi;
        float phy = 0.0f;
        Vec3f wo = shading::computeLocalSphericalDirection(scene_rdl2::math::cos(theta), scene_rdl2::math::sin(theta), phy);
        wo = test.frame.localToGlobal(wo);

        if (test.assertPdfIntegral) {
            testPdfIntegral(test, wo, sampleCount * 10, true);
        }
        if (test.assertEvalIntegral) {
            testEvalIntegral(test, wo, sampleCount * 10, true);
        }
    }

#ifndef ONLY_ASSERT_INTEGRALS
    float thetaInc = sHalfPi / viewAnglesTheta;
    float phyInc = sTwoPi / viewAnglesPhy;

    for (int t=0; t < viewAnglesTheta; t++) {
        for (int p=0; p < viewAnglesPhy; p++) {

            float theta = t * thetaInc + thetaInc / 2.0f;
            float phy = p * phyInc;
            Vec3f wo = shading::computeLocalSphericalDirection(scene_rdl2::math::cos(theta), scene_rdl2::math::sin(theta), phy);
            wo = test.frame.localToGlobal(wo);

            printInfo("----- thetaWo - %f , phyWo - %f -----",
                    theta / sPi * 180.0f,
                    phy / sPi * 180.0f);

            testPdfIntegral(test, wo, sampleCount);
            testEvalIntegral(test, wo, sampleCount);
        }
    }
#endif
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

