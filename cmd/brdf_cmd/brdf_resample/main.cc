// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/shading/bsdf/BsdfTable.h>
#include <moonray/rendering/shading/bsdf/BsdfTableAniso.h>
#include <moonray/rendering/shading/bsdf/LightStage.h>
#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/render/util/Args.h>
#include <scene_rdl2/render/logging/logging.h>

#include <string>
#include <sstream>
#include <exception>


using namespace moonray;
using namespace scene_rdl2::math;

using scene_rdl2::util::Args;
using scene_rdl2::util::stringToFloat;
using scene_rdl2::util::stringToIntArray;
using scene_rdl2::logging::Logger;
using moonray::shading::AnisotropicBsdfTable;
using moonray::shading::IsotropicBsdfTable;
using moonray::shading::LightStage;
using moonray::shading::LightStageCylinderBsdfSlice;


typedef std::vector<LightStageCylinderBsdfSlice *> SlicePtrArray;


//---------------------------------------------------------------------------

void
reSampleBsdf(const std::vector<int> &phiValues,
        const SlicePtrArray &slicesBottom, const SlicePtrArray &slicesTop,
        const scene_rdl2::math::Vec2f &smootThetaWoRange, AnisotropicBsdfTable &bsdf)
{
    MNRY_ASSERT(bsdf.getSizePhiH() == phiValues.size() * 2);

    int size = phiValues.size();
    for (int indexThetaH = 0; indexThetaH < bsdf.getSizeThetaH(); indexThetaH++) {
        for (int indexPhiH = 0; indexPhiH < size; indexPhiH++) {
            for (int indexThetaD = 0; indexThetaD < bsdf.getSizeThetaD(); indexThetaD++) {
                for (int indexPhiD = 0; indexPhiD < bsdf.getSizePhiD(); indexPhiD++) {

                    scene_rdl2::math::Vec3f localWo, localWi;
                    scene_rdl2::math::Color color;
                    int iph;

                    iph = indexPhiH;
                    bsdf.indexHD2localWoWi(indexThetaH, iph,
                                           indexThetaD, indexPhiD,
                                           localWo, localWi);
                    color = slicesBottom[iph]->getBsdf(localWo, localWi, smootThetaWoRange);
                    bsdf.setBsdf(indexThetaH, iph, indexThetaD, indexPhiD, color);

                    iph = indexPhiH + size;
                    bsdf.indexHD2localWoWi(indexThetaH, iph,
                                           indexThetaD, indexPhiD,
                                           localWo, localWi);
                    color = slicesTop[indexPhiH]->getBsdf(localWo, localWi, smootThetaWoRange);
                    bsdf.setBsdf(indexThetaH, iph, indexThetaD, indexPhiD, color);
                }
            }
        }
    }
}


//---------------------------------------------------------------------------

void
sampleLightStageTriangles(const LightStageCylinderBsdfSlice &slice)
{
    // Use this to sample triangles out of the lights
    scene_rdl2::math::Vec3f wo(0.0f, 0.0f, 1.0f);
    for (int indexTheta = 0; indexTheta <= 180; indexTheta++) {
        for (int indexPhi = 0; indexPhi <= 360; indexPhi++) {
            float theta = float(indexTheta) * sHalfPi / 90.0f;
            float phi = float(indexPhi) * sHalfPi / 90.0f;
            float s, c;
            sincos(theta, &s, &c);
            scene_rdl2::math::Vec3f wi = moonray::shading::computeLocalSphericalDirection(c, s, phi);
            slice.getBsdf(wo, wi, scene_rdl2::math::Vec2f(0.0f));
        }
    }
}


void
debugLightStage(const LightStageCylinderBsdfSlice &slice, const IsotropicBsdfTable &bsdf)
{
    static const int count = 6;
    const float localWo[count][3] = {
            {0.345851, 0.197468, 0.917275},
            {0.3796, 0.192993, 0.904797},
            {0.413411, 0.189192, 0.890672},
            {0.445757, 0.184927, 0.875844},
            {0.477533, 0.180847, 0.8598},
            {0.510601, 0.178477, 0.84109}
    };
    const float localWi[count][3] = {
            {-0.518066, -0.202658, 0.830985},
            {-0.484293, -0.204995, 0.850551},
            {-0.453608, -0.189106, 0.870907},
            {-0.42439, -0.213791, 0.879879},
            {-0.387111, -0.215342, 0.896534},
            {-0.355627, -0.218802, 0.908656}
    };

    for (int i = 0; i < 6; i++) {
        scene_rdl2::math::Vec3f lwo(localWo[i][0], localWo[i][1], localWo[i][2]);
        scene_rdl2::math::Vec3f lwi(localWi[i][0], localWi[i][1], localWi[i][2]);
        std::cout << "localWo = " << lwo << ", localWi = " << lwi;
        scene_rdl2::math::Color f = slice.getBsdf(lwo, lwi, scene_rdl2::math::Vec2f(0.0f));
        std::cout << ", f = " << f << std::endl;
    }

    std::cout << "----------" << std::endl;

    for (int i = 0; i < 6; i++) {
        scene_rdl2::math::Vec3f lwo(localWo[i][0], localWo[i][1], localWo[i][2]);
        scene_rdl2::math::Vec3f lwi(localWi[i][0], localWi[i][1], localWi[i][2]);
        std::cout << "localWo = " << lwo << ", localWi = " << lwi;
        scene_rdl2::math::Color f = bsdf.getBsdf(lwo, lwi, true);
        std::cout << ", f = " << f << std::endl;
    }
}


void
sampleDebugBsdfPatternHD(IsotropicBsdfTable &bsdf)
{
    // Sample a debug pattern that varies smoothly only along thetaO
    for (int indexThetaH = 0; indexThetaH < bsdf.getSizeThetaH(); indexThetaH++) {
        for (int indexThetaD = 0; indexThetaD < bsdf.getSizeThetaD(); indexThetaD++) {
            for (int indexPhiD = 0; indexPhiD < bsdf.getSizePhiD(); indexPhiD++) {
                scene_rdl2::math::Vec3f localWo, localWi;
                bsdf.indexHD2localWoWi(indexThetaH, indexThetaD, indexPhiD, localWo, localWi);

                // ThetaO / PhiO
                float cosThetaO = clamp(localWo.z, -1.0f, 1.0f);
                float thetaO = acos(cosThetaO);
                float phiO = atan2(localWo.y, localWo.x);

                // ThetaI / PhiI
                float cosThetaI = clamp(localWi.z, -1.0f, 1.0f);
                float thetaI = acos(cosThetaI);
                float phiI = atan2(localWi.y, localWi.x);

                float phi = moonray::shading::rangeAngle(phiI - phiO);

                scene_rdl2::math::Color color(thetaO / sHalfPi, thetaI / sHalfPi,
                                  scene_rdl2::math::abs(phi) / sPi);

                bsdf.setBsdf(indexThetaH, indexThetaD, indexPhiD, color);

//                std::cout << "ith, itd, ipd = " << indexThetaH << ", " << indexThetaD << ", " << indexPhiD <<
//                        ", wo = " << localWo << ", wi = " << localWi << ", color = " << color << std::endl;

                int ith, itd, ipd;
                bsdf.localWoWi2IndexHD(localWo, localWi, ith, itd, ipd);
                if (scene_rdl2::math::abs(ith - indexThetaH) > 1  ||
                    scene_rdl2::math::abs(itd - indexThetaD) > 1  ||
                    scene_rdl2::math::abs(ipd - indexPhiD) > 1) {

                    MNRY_ASSERT_REQUIRE(0);
                    bsdf.indexHD2localWoWi(indexThetaH, indexThetaD, indexPhiD, localWo, localWi);
                    bsdf.localWoWi2IndexHD(localWo, localWi, ith, itd, ipd);
                }
            }
        }
    }
}


//---------------------------------------------------------------------------

void usage(char *argv0, const std::string &message)
{
    if (!message.empty()) {
        Logger::error(message);
    }
    std::cerr << "Usage: " << argv0 << "[options]" << std::endl;
    std::cerr << "Which: re-samples various brdf slices into an anisotropic brdf" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -in bsdf_slice_prefix 0,45,90  input brdf slice file prefix and given phi values" << std::endl;
    std::cerr << "  -out file.bsdf                 output bsdf file" << std::endl;
    std::cerr << "  -smoothThetaWoRange 0.05 0.1   smooth bsdf around pole in the wo range of angles (*pi/2)" << std::endl;
}


//---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
    try {
        //------------------------------------

        // Initialize our logging
        Logger::init();

        // Check for no flags or help flag.
        if (argc == 1 || std::string(argv[1]) == "-h") {
            usage(argv[0], "");
            std::exit(EXIT_FAILURE);
        }

        //------------------------------------

        // Args parsing
        Args args(argc, argv);
        Args::StringArray values;

        if (args.getFlagValues("-in", 2, values) < 0) {
            usage(argv[0], "Missing input prefix");
            std::exit(EXIT_FAILURE);
        }
        std::string inPrefix = values[0];
        std::vector<int> phiValues;
        stringToIntArray(values[1], phiValues);

        if (args.getFlagValues("-out", 1, values) < 0) {
            usage(argv[0], "Missing output filename");
            std::exit(EXIT_FAILURE);
        }
        std::string outFilename = values[0];

        scene_rdl2::math::Vec2f smoothThetaWoRange(0.05f * sHalfPi, 0.1f * sHalfPi);
        if (args.getFlagValues("-smoothThetaWoRange", 2, values) >= 0) {
            smoothThetaWoRange[0] = stringToFloat(values[0]);
            smoothThetaWoRange[1] = stringToFloat(values[1]);
            smoothThetaWoRange *= sHalfPi;
        }

        //------------------------------------

        Logger::info("Processing input sequence: ");
        SlicePtrArray slicesBottom;
        SlicePtrArray slicesTop;
        int size = phiValues.size();
        for (int i = 0; i < size; i++) {
            LightStageCylinderBsdfSlice *slice;

            std::ostringstream number;
            number << phiValues[i];

            std::string filenameBottom = inPrefix + "_" + number.str() + "_bottom.slice";
            Logger::info("Reading bsdf slice file: \"" , filenameBottom , "\"");
            slice = new LightStageCylinderBsdfSlice(filenameBottom);
            slicesBottom.push_back(slice);

            std::string filenameTop = inPrefix + "_" + number.str() + "_top.slice";
            Logger::info("Reading bsdf slice file: \"" , filenameTop , "\"");
            slice = new LightStageCylinderBsdfSlice(filenameTop);
            slicesTop.push_back(slice);
        }

        Logger::info("Re-Sampling bsdf...");
        AnisotropicBsdfTable bsdf(90, 2 * size, 90, 360, false);
        reSampleBsdf(phiValues, slicesBottom, slicesTop, smoothThetaWoRange, bsdf);
        Logger::info("Writing bsdf file: \"" , outFilename , "\"");
        bsdf.saveAs(outFilename);

    } catch (const std::exception& e) {
        Logger::error(e.what());
        std::exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------

