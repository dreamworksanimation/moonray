// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/Random.h>
#include <moonray/rendering/pbr/integrator/BsdfOneSampler.h>
#include <moonray/rendering/shading/bsdf/cook_torrance/BsdfCookTorrance.h>
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/Util.h>

#include <tbb/tbb.h>

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>


using namespace moonray;
using namespace scene_rdl2::math;
using namespace moonray::pbr;
using namespace moonray::shading;

// Are we running single-threaded (for debugging) ?
#define RUN_SINGLE_THREADED 0

// Are we doing .csv output (vs. human-readable formatting)
#define PRINT_CSV 1


//---------------------------------------------------------------------------

class BsdfFactory {
public:
    virtual ~BsdfFactory()  {}
    virtual Bsdf *operator()(scene_rdl2::alloc::Arena& arena,
            const scene_rdl2::math::ReferenceFrame &frame) const = 0;
};


class CookTorranceBsdfFactory : public BsdfFactory {
public:
    CookTorranceBsdfFactory(float roughness, const Color &ks) :
        mRoughness(roughness), mKs(ks)  {}
    Bsdf *operator()(scene_rdl2::alloc::Arena& arena, const scene_rdl2::math::ReferenceFrame &frame) const
    {
        Bsdf *bsdf = arena.allocWithCtor<Bsdf>();
        BsdfLobe *lobe = arena.allocWithArgs<CookTorranceBsdfLobe>(
                frame.getN(), mRoughness);
        lobe->setFresnel(arena.allocWithArgs<SchlickFresnel>(mKs, 1.f));
        bsdf->addLobe(lobe);
        return bsdf;
    }

    float getRoughness() const  { return mRoughness;  }
    const Color &getKs() const  { return mKs;  }

private:
    float mRoughness;
    Color mKs;
};


//---------------------------------------------------------------------------

#if RUN_SINGLE_THREADED
static const int sTaskCount = 1;
#else
// This gives the scheduler enough granularity to load balance well, but not
// too much that it causes contention on the atomics / mutexes at the end of
// each task
static const int sTaskCount = tbb::task_scheduler_init::default_num_threads() * 4;
#endif


struct TaskSettings {
    TaskSettings(scene_rdl2::util::Random *r, const BsdfFactory &bf, const scene_rdl2::math::ReferenceFrame &f) :
        seeder(r), bsdfFactory(bf), frame(f)   {}

    std::uint32_t getSeed() const
    {
        sMutex.lock();
        const std::uint32_t seed = seeder->getNextUInt();
        sMutex.unlock();
        return seed;
    }

private:
    scene_rdl2::util::Random *seeder;
public:
    const BsdfFactory &bsdfFactory;
    const scene_rdl2::math::ReferenceFrame &frame;

    static tbb::mutex sMutex;
};

tbb::mutex TaskSettings::sMutex;


struct EvalBsdfIntegralTask {
    EvalBsdfIntegralTask(const TaskSettings &test,
            const BsdfSlice &slice, Color &integralUniform, Color &integralImportance) :
        inTest(test), inSlice(slice), outIntegralUniform(integralUniform),
                outIntegralImportance(integralImportance)  {}

    void operator()(const tbb::blocked_range<int>& range) const;

    const TaskSettings &inTest;
    const BsdfSlice &inSlice;
    Color &outIntegralUniform;
    Color &outIntegralImportance;
};


void
EvalBsdfIntegralTask::operator()(const tbb::blocked_range<int> &range) const
{
    scene_rdl2::util::Random rand1(inTest.getSeed(), 145 /* Select random stream */);
    scene_rdl2::util::Random rand2(inTest.getSeed(), 827 /* Select random stream */);
    scene_rdl2::util::Random rand3(inTest.getSeed(), 929 /* Select random stream */);

    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);
    scene_rdl2::alloc::Arena arena;
    arena.init(arenaBlockPool.get());

    Bsdf *bsdf = inTest.bsdfFactory(arena, inTest.frame);
    BsdfOneSampler sampler(*bsdf, inSlice);

    const float pdfUniform = 1.0f /
            (bsdf->getIsSpherical()  ?  sFourPi  :  sTwoPi);

    Color integralUniform(sBlack);
    Color integralImportance(sBlack);
    for (int s = range.begin(); s != range.end(); ++s) {

        float r1 = rand1.getNextFloat();
        float r2 = rand2.getNextFloat();

        // Compute the integrated bsdf using uniform sampling
        Vec3f wi;
        if (bsdf->getIsSpherical()) {
            wi = shading::sampleSphereUniform(r1, r2);
        } else {
            wi = shading::sampleLocalHemisphereUniform(r1, r2);
            wi = inTest.frame.localToGlobal(wi);
        }
        float pdf;
        Color color = sampler.eval(wi, pdf);
        integralUniform += color / pdfUniform;

        // Compute the integrated bsdf using importance sampling
        color = sampler.sample(rand3.getNextFloat(), r1, r2, wi, pdf);
        if ((pdf == 0.0f)  ||  scene_rdl2::math::isExactlyZero(color)) {
            continue;
        }
        integralImportance += color / pdf;
    }

    inTest.sMutex.lock();
    outIntegralUniform += integralUniform;
    outIntegralImportance += integralImportance;
    inTest.sMutex.unlock();
}

//---------------------------------------------------------------------------

inline void
printInfo(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}


static void
evalIntegral(const TaskSettings &test, const Vec3f &wo, int sampleCount,
        Color &integralUniform, Color &integralImportance)
{
    // Get ready to sample, including the cosine term since we want to
    // integrate the bsdf * cosine term
    BsdfSlice slice(test.frame.getN(), wo, true, true, ispc::SHADOW_TERMINATOR_FIX_OFF);

    integralUniform = sBlack;
    integralImportance = sBlack;
    EvalBsdfIntegralTask task(test, slice, integralUniform, integralImportance);
    tbb::parallel_for(tbb::blocked_range<int>(0, sampleCount,
            sampleCount / sTaskCount), task);

    // Divide by pdf and/or number of samples to get the MC estimate
    integralUniform /= sampleCount;
    integralImportance /= sampleCount;
}


void
runTest(const TaskSettings &test, const Fresnel *omFPrime,
        int viewAnglesTheta, int viewAnglesPhy, int sampleCount)
{
    float thetaInc = sHalfPi / viewAnglesTheta;
    float phyInc = sTwoPi / viewAnglesPhy;

#if PRINT_CSV
    printInfo("ThetaWo,Rs,Fprime,WeightedSum,Furnace");
#endif

    for (int t=0; t < viewAnglesTheta; t++) {
        for (int p=0; p < viewAnglesPhy; p++) {

            float theta = t * thetaInc;
            float cosTheta = scene_rdl2::math::cos(theta);
            float phy = p * phyInc;
            Vec3f wo = shading::computeLocalSphericalDirection(
                cosTheta, scene_rdl2::math::sin(theta), phy);
            wo = test.frame.localToGlobal(wo);

#if !PRINT_CSV
            printInfo("----- thetaWo - %f , phiWo - %f -----",
                    theta / sPi * 180.0f,
                    phy / sPi * 180.0f);
#endif

            Color RsUniform;
            Color RsImportance;
            evalIntegral(test, wo, sampleCount, RsUniform, RsImportance);

            Color Fprime = Color(1.0f) - omFPrime->eval(cosTheta);
            Color RsPlusOmFprime = RsImportance + (Color(1.0f) - Fprime);
            Color RsPlusOmFprimeFurnace = Color(0.217f) * RsPlusOmFprime;

#if PRINT_CSV
            printInfo("%f,%f,%f,%f,%f", theta / sPi * 180.0f,
                    RsImportance.r, Fprime.r, RsPlusOmFprime.r, RsPlusOmFprimeFurnace.r);
#else
            printInfo("Rs importance = %f", RsImportance.r);
            printInfo("Fprime        = %f", Fprime.r);
            printfInfo("Rs + (1-Fprime) = %f", RsPlusOmFprime.r);
            printfInfo("0.217 * (Rs + (1-Fprime)) = %f", RsPlusOmFprimeFurnace.r);
#endif
        }
    }
}


int
main(int argc, char* argv[])
{
    static const int sSampleCount = 10000;
    static const int sViewAnglesTheta = 9;
    static const int sKsCount = 11;
    static const float sKs[sKsCount] =
        { 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.6, 0.9, 1.0 };
    static const int sRoughnessCount = 11;
    static const float sRoughness[sRoughnessCount] =
        { 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };

    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);
    scene_rdl2::alloc::Arena arena;
    arena.init(arenaBlockPool.get());

    // Fixed seed for repeatable tests
    constexpr std::uint32_t seed = 0xdeadbeef;
    scene_rdl2::util::Random rand(seed);
    scene_rdl2::math::ReferenceFrame frame;

    for (int j=0; j < sKsCount; j++) {
        printInfo("########################################");
        printInfo("Ks = ,%f", sKs[j]);
        for (int i=0; i < sRoughnessCount; i++) {
#if !PRINT_CSV
            printInfo("===== Ks = %f  -  Roughness: %f ========", sKs[j], sRoughness[i]);
#endif

            Color ks(sKs[j]);
            float specRoughness = sRoughness[i] * sRoughness[i];

            CookTorranceBsdfFactory factory(specRoughness, ks);

            auto schlick = arena.allocWithArgs<SchlickFresnel>(ks, 1.0f);
            auto omFPrime = arena.allocWithArgs<OneMinusRoughSchlickFresnel>(
                    schlick, specRoughness);

            TaskSettings test(&rand, factory, frame);
            runTest(test, omFPrime, sViewAnglesTheta, 1, sSampleCount);
        }
    }

    return EXIT_SUCCESS;
}

