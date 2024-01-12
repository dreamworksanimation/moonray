// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "EdgeDetect.h"
#include "Image.h"

#include <scene_rdl2/render/util/Args.h>
#include <scene_rdl2/render/logging/logging.h>
#include <moonray/rendering/shading/bsdf/LightStage.h>

#include <string>
#include <sstream>
#include <exception>


using scene_rdl2::util::Args;
using scene_rdl2::util::stringToFloat;
using scene_rdl2::util::stringToIntArray;
using scene_rdl2::logging::Logger;
using namespace scene_rdl2::math;
using moonray::shading::LightStageCylinderBsdfSlice;
using moonray::shading::LightStage;


//---------------------------------------------------------------------------

// We work in the following coordinate systems:
// --------------------------------------------
//
// Note: the camera is rotated so the cylinder shows horizontally in the images
// the image would need to be rotated 90 degrees counter-clockwise to be vertical
//
// - Image space: (o, x, y):
//   o: the lower left corner
//   x: increases horizontally rightwards, length = 1 pixel
//   y: increases vertically upwards, length = 1 pixel
//
// - Cylinder space (image-projected): (o, s, t):
//   o: leftmost point (according to imageRatio) on the cylinder center axis
//   u: increases rightwards along cylinder axis, length = 1 pixel
//   v: increases downwards perpendicular to cylinder axis, length = cylinder radius
//
// - ICT World-space (o, x, y, z) also called ICT "light-stage" space (right-handed):
//   o: physical center of the light-stage
//   x: points left,  y: points up,  z: points away from camera
//   This is the space in which we get the light coordinates
//
// - DWA World-space (o, x, y, z) also called DWA "light-stage" space (right-handed):
//   o: physical center of the light-stage
//   x: points right,  y: points up,  z: points towards the camera
//
//
// We expect the following data in the input sequence:
// ---------------------------------------------------
// - Total 177 frames per video clip
//
// - 1st, 2nd frame: black
// - 3rd frame: all-on lighting condition for tracking
// - 4th ~  173th frame: diffuse lighting, followed by specular
// lighting from 85 lights.
// - 174th, 175th frame: black
// - 176th frame: all-on lighting condition for tracking
// - 177th frame: black


//---------------------------------------------------------------------------

std::string
paddedFrameNumber(int n, int padding)
{
    std::ostringstream stream;
    stream << n;
    std::string number = stream.str();
    std::string paddingZeroes(padding - number.size(), '0');
    std::string paddedNumber = paddingZeroes + number;
    return paddedNumber;
}


//---------------------------------------------------------------------------

Color
samplePoint(const Vec2f &Pimg, float thetaO,
        const LightStageCylinderBsdfSlice &bsdfSlice,
        int lightOrderIndex, Image &image, bool draw)
{
    static const LightStage &ls = LightStage::singleton();

    // Light position
    Vec3f Plight(0.0f, 0.0f, 100000.0f);
    if (lightOrderIndex >= 0) {
        int lightIndex = ls.getSeparatedLightIndex(lightOrderIndex);
        Plight = ls.getLightPosition(lightIndex);
    }

    // Compute position P and normal N on the cylinder in light-stage space
    Vec3f P, N, T;
    bsdfSlice.computeCylLightStagePNT(thetaO, P, N, T);

    // Light direction wi
    Vec3f wi = Plight - P;
    wi = normalize(wi);

    // Are we receiving any light from light direction wi ?
    Color color(0.0f);
    float NdotWi = dot(N, wi);
    static const float sCosineTermThreshold = 3e-2f;
    if (NdotWi > sCosineTermThreshold) {
        if (draw) {
            // Draw our point sample
            image.drawPixel(int(floor(Pimg.x)), Pimg.y, Color(1.0f));
        } else {
            // Point sample the image, divide by cosine term and set the
            // resulting bsdf value
            color = image.sampleColor(int(floor(Pimg.x)), Pimg.y);
            color = max(color, Color(0.0f));
            color /= NdotWi;
        }
    }

    return color;
}


void
sampleCylinder(const Vec2f &topEdge, const Vec2f &bottomEdge, Vec2f xRange,
        int lightOrderIndex, Image &image,
        LightStageCylinderBsdfSlice &bsdfSliceTop,
        LightStageCylinderBsdfSlice &bsdfSliceBottom)
{
    MNRY_ASSERT(bsdfSliceTop.getCylAlpha() == bsdfSliceBottom.getCylAlpha());
    MNRY_ASSERT(bsdfSliceTop.getSizeThetaO() == bsdfSliceBottom.getSizeThetaO());

    // Cylinder center axis edge in image space
    Vec2f axisEdge = 0.5f * (topEdge + bottomEdge);

    // Compute cylinder samples coordinate system (O, U, V), where O is the
    // origin, U is a vector down the cylinder center axis (with length to
    // match one scanline) and V is a vector perpendicular to U (with length
    // equal to the screen-projected cylinder radius)
    Vec2f O;
    O.x = xRange[0];
    O.y = axisEdge.x * O.x + axisEdge.y;

    Vec2f U(1.0f, axisEdge.x);
    Vec2f V(axisEdge.x, -1.0f);
    float yBottom = bottomEdge.x * O.x + bottomEdge.y;
    V = normalize(V) * (O.y - yBottom) * cos(bsdfSliceTop.getCylAlpha());

    //std::cout << "Cylinder samples coordinate system (O, U, V) = " <<
    //        O << ", " << U << ", " << V << std::endl;


    // TODO: For now we'll assume a camera orthographic projection down -z axis.

    // Sample the cylinder with a uniformly distributed viewing angle thetaO
    const float uMax = float(xRange[1] - xRange[0] + 1);
    const float uInc = 1.0f;
    const int vCount = bsdfSliceTop.getSizeThetaO();

    // Sample the cylinder
    if (lightOrderIndex >= 0) {

        //std::cout << "lightOrderIndex = " << lightOrderIndex << std::endl;

        for (int vi = 0; vi < vCount; vi++) {
            Color colorBottom(0.0f);
            Color colorTop(0.0f);
            int uCount = 0;
            // Sample viewing angle interval [0,pi/2[
            float thetaO = vi * sHalfPi / vCount;
            float v = sin(thetaO);
            for (float u = 0.0f; u < uMax; u += uInc, uCount++) {
                Vec2f Pimg;

                // Compute image space position on the cylinder
                Pimg = O + u * U + v * V;
                colorBottom += samplePoint(Pimg, thetaO, bsdfSliceBottom,
                        lightOrderIndex, image, false);

                Pimg = O + u * U - v * V;
                colorTop += samplePoint(Pimg, -thetaO, bsdfSliceTop,
                        lightOrderIndex, image, false);
            }

            // Average point samples along each line sample
            colorBottom /= float(uCount);
            colorTop /= float(uCount);

            // Store in the 3D Bsdf table
            // NOTE: indexPhiO = 0 maps to phiWo = pi in the cylinder reference frame
            bsdfSliceBottom.setBsdf(lightOrderIndex, vi, colorBottom);
            bsdfSliceTop.setBsdf(lightOrderIndex, vi, colorTop);

            //std::cout << "indexThetaO = " << vi << ", colorBottom = " << colorBottom << std::endl;
        }
    }

    // Draw the cylinder samples for debugging
    for (int vi = 0; vi < vCount; vi++) {
        // Sample viewing angle interval [0,pi/2[
        float thetaO = vi * sHalfPi / vCount;
        float v = sin(thetaO);
        for (float u = 0.0f; u < uMax; u += uInc) {
            Vec2f Pimg;

            // Compute image space position on the cylinder
            Pimg = O + u * U + v * V;
            samplePoint(Pimg, thetaO, bsdfSliceBottom, lightOrderIndex, image, true);

            Pimg = O + u * U - v * V;
            samplePoint(Pimg, -thetaO, bsdfSliceTop, lightOrderIndex, image, true);
        }
    }
}


void
sampleSeqImage(const Image &inImage, const Vec2f &topEdge, const Vec2f &bottomEdge,
        const Vec2f &xRange, int lightOrderIndex, Image &inSeqImage,
        LightStageCylinderBsdfSlice &bsdfSliceTop,
        LightStageCylinderBsdfSlice &bsdfSliceBottom)
{
    // Scale edges and xRange from inImage to inSeqImage coordinates
    Vec2f scale(float(inSeqImage.getWidth()) / inImage.getWidth(),
                float(inSeqImage.getHeight()) / inImage.getHeight());
    Vec2f topEdgeSeq(topEdge.x, topEdge.y * scale.y);
    Vec2f bottomEdgeSeq(bottomEdge.x, bottomEdge.y * scale.y);
    Vec2f xRangeSeq = xRange * scale.x;

    sampleCylinder(topEdgeSeq, bottomEdgeSeq, xRangeSeq, lightOrderIndex,
            inSeqImage, bsdfSliceTop, bsdfSliceBottom);
}


void
sampleLight(Image &inImage, const Vec2f &topEdge, const Vec2f &bottomEdge,
        const Vec2f &xRange, const std::string &inSeqPrefix, int lightOrderIndex,
        const std::vector<int> &ignoreFrames,
        LightStageCylinderBsdfSlice &bsdfSliceDifTop,
        LightStageCylinderBsdfSlice &bsdfSliceDifBottom,
        LightStageCylinderBsdfSlice &bsdfSliceSpecTop,
        LightStageCylinderBsdfSlice &bsdfSliceSpecBottom)
{
    const int frameNumberDiffuse = 2 * lightOrderIndex + 3;
    const int frameNumberSpecular = frameNumberDiffuse + 1;

    std::string filenameDiffuse = inSeqPrefix + "." +
            paddedFrameNumber(frameNumberDiffuse, 4) + ".exr";
    std::string filenameSpecular = inSeqPrefix + "." +
            paddedFrameNumber(frameNumberSpecular, 4) + ".exr";

    if (find(ignoreFrames.begin(), ignoreFrames.end(), frameNumberDiffuse) != ignoreFrames.end()) {
        Logger::info("Skipping   image (diffuse) : \"" , filenameDiffuse , "\"");
    } else {
        Logger::info("Processing image (diffuse) : \"" , filenameDiffuse , "\"");
        Image inSeqImage(filenameDiffuse);

        sampleSeqImage(inImage, topEdge, bottomEdge, xRange, lightOrderIndex,
                inSeqImage, bsdfSliceDifTop, bsdfSliceDifBottom);

        std::string filenameSamplesOut = inSeqPrefix + "_samples." +
                paddedFrameNumber(frameNumberDiffuse, 4) + ".exr";
        inSeqImage.saveAs(filenameSamplesOut);
    }

    if (find(ignoreFrames.begin(), ignoreFrames.end(), frameNumberSpecular) != ignoreFrames.end()) {
        Logger::info("Skipping   image (specular): \"" , filenameSpecular , "\"");
    } else {
        Logger::info("Processing image (specular): \"" , filenameSpecular , "\"");
        Image inSeqImageDiffuse(filenameDiffuse);
        Image inSeqImage(filenameSpecular);
        inSeqImage.subtract(inSeqImageDiffuse);

        sampleSeqImage(inImage, topEdge, bottomEdge, xRange, lightOrderIndex,
                inSeqImage, bsdfSliceSpecTop, bsdfSliceSpecBottom);

        std::string filenameSamplesOut = inSeqPrefix + "_samples." +
                paddedFrameNumber(frameNumberSpecular, 4) + ".exr";
        inSeqImage.saveAs(filenameSamplesOut);
    }
}


//---------------------------------------------------------------------------

void usage(char *argv0, const std::string &message)
{
    if (!message.empty()) {
        Logger::error(message);
    }
    std::cerr << "Usage: " << argv0 << "[options]" << std::endl;
    std::cerr << "Which: extracts a brdf slice from cylinder OLAT images" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -in input.exr             input image file used for edge detection" << std::endl;
    std::cerr << "  -in_seq prefix            input image sequence prefix (prefix_####.exr)" << std::endl;
    std::cerr << "  -out_sobel output.exr     output sobel image file" << std::endl;
    std::cerr << "  -out_edges output.exr     output edges image file" << std::endl;
    std::cerr << "  -out_samples output.exr   output samples image file" << std::endl;
    std::cerr << "  -out_bsdf_slice prefix 0  output brdf slice file prefix for given phi value (in degrees)" << std::endl;
    std::cerr << "  -ignore_frames 10,22,34   ignore frames from input image sequence" << std::endl;
    std::cerr << "  -image_ratio 0.5          ratio of the image to use for edge detection & sampling" << std::endl;
    std::cerr << "  -edge_threshold 0.25      threshold for edge detection in sobel image" << std::endl;
    std::cerr << "  -edge_top_bottom a b a b  edge equation parameters" << std::endl;
    std::cerr << "  -sample_scale 1.0 1.0     scale sample location along cylinder axis and radius" << std::endl;
    std::cerr << "  -sample_offset 0.0 0.0    offset sample location along cylinder axis and radius" << std::endl;
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

        if (args.getFlagValues("-in", 1, values) < 0) {
            usage(argv[0], "Missing input filename");
            std::exit(EXIT_FAILURE);
        }
        std::string inFilename = values[0];

        std::string inSeqPrefix;
        if (args.getFlagValues("-in_seq", 1, values) >= 0) {
            inSeqPrefix = values[0];
        }

        std::string outSobelFilename;
        if (args.getFlagValues("-out_sobel", 1, values) >= 0) {
            outSobelFilename = values[0];
        }

        std::string outEdgesFilename;
        if (args.getFlagValues("-out_edges", 1, values) >= 0) {
            outEdgesFilename = values[0];
        }

        std::string outSamplesFilename;
        if (args.getFlagValues("-out_samples", 1, values) >= 0) {
            outSamplesFilename = values[0];
        }

        std::string outBsdfSlicePrefix;
        std::string outBrdfPhi;
        if (args.getFlagValues("-out_bsdf_slice", 2, values) >= 0) {
            outBsdfSlicePrefix = values[0];
            outBrdfPhi = values[1];
        }

        std::vector<int> ignoreFrames;
        if (args.getFlagValues("-ignore_frames", 1, values) >= 0) {
            stringToIntArray(values[0], ignoreFrames);
        }

        float imageRatio = 0.5f;
        if (args.getFlagValues("-image_ratio", 1, values) >= 0) {
            imageRatio = stringToFloat(values[0]);
        }

        float edgeThreshold = 0.25f;
        if (args.getFlagValues("-edge_threshold", 1, values) >= 0) {
            edgeThreshold = stringToFloat(values[0]);
        }

        bool doFit = true;
        Vec2f topEdge(0.0f, 0.0f);
        Vec2f bottomEdge(0.0f, 0.0f);
        if (args.getFlagValues("-edge_top_bottom", 4, values) >= 0) {
            doFit = false;
            topEdge.x = stringToFloat(values[0]);
            topEdge.y = stringToFloat(values[1]);
            bottomEdge.x = stringToFloat(values[2]);
            bottomEdge.y = stringToFloat(values[3]);
        }

        Vec2f sampleScale(1.0f, 1.0f);
        Vec2f sampleOffset(0.0f, 0.0f);
        if (args.getFlagValues("-sample_scale", 2, values) >= 0) {
            sampleScale.x = stringToFloat(values[0]);
            sampleScale.y = stringToFloat(values[1]);
        }
        if (args.getFlagValues("-sample_offset", 2, values) >= 0) {
            sampleOffset.x = stringToFloat(values[0]);
            sampleOffset.y = stringToFloat(values[1]);
        }


        //------------------------------------

        // Load input image
        Logger::info("Reading image: \"" , inFilename , "\"");
        Image inImage(inFilename);
        Logger::info("Image width: " , inImage.getWidth());
        Logger::info("Image height: " , inImage.getHeight());


        // Sobel image for edge detection
        Logger::info("Applying sobel filter");
        Image sobelImage;
        inImage.sobel(sobelImage);

        if (!outSobelFilename.empty()) {
            Logger::info("Writing sobel image: \"" , outSobelFilename , "\"");
            sobelImage.saveAs(outSobelFilename);
        }


        // Select xMin / xMax of the image to perform detection
        const int xSize = int(sobelImage.getWidth() * imageRatio) + 1;
        const int xMin = (sobelImage.getWidth() - xSize) / 2;
        const int xMax = (sobelImage.getWidth() + xSize) / 2;

        // Detect top and bottom edge samples
        Logger::info("Detecting edge samples");
        std::vector<Vec2f> topEdgeSamples;
        std::vector<Vec2f> bottomEdgeSamples;
        detectEdgeSamples(sobelImage, xMin, xMax, edgeThreshold,
                topEdgeSamples, bottomEdgeSamples);

        // Fit top and bottom edge, compute cylinder axis
        if (doFit) {
            fitEdge(topEdgeSamples, topEdge.x, topEdge.y);
            fitEdge(bottomEdgeSamples, bottomEdge.x, bottomEdge.y);
        }
        Logger::info("Top edge: y = " , topEdge.x , " * x + " , topEdge.y);
        Logger::info("Bottom edge: y = " , bottomEdge.x , " * x + " , bottomEdge.y);

        // Scale / offset edges and xRange to affect samples
        Vec2f axisEdge = 0.5f * (topEdge + bottomEdge);
        topEdge.y = axisEdge.y + (topEdge.y - axisEdge.y) * sampleScale.y + sampleOffset.y;
        bottomEdge.y = axisEdge.y + (bottomEdge.y - axisEdge.y) * sampleScale.y + sampleOffset.y;
        Vec2f xRange(xMin + 0.5f, xMax + 0.5f);
        float xCenter = 0.5f * (xRange[0] + xRange[1]);
        xRange[0] = xCenter + (xRange[0] - xCenter) * sampleScale.x + sampleOffset.x;
        xRange[1] = xCenter + (xRange[1] - xCenter) * sampleScale.x + sampleOffset.x;

        // Draw edge samples and save edges image
        if (!outEdgesFilename.empty()) {
            Image edgesImage(inImage);
            drawEdge(topEdge.x, topEdge.y, edgesImage);
            drawEdge(bottomEdge.x, bottomEdge.y, edgesImage);
            drawEdgeSamples(topEdgeSamples, edgesImage);
            drawEdgeSamples(bottomEdgeSamples, edgesImage);
            Logger::info("Writing edges image: \"" , outEdgesFilename , "\"");
            edgesImage.saveAs(outEdgesFilename);
        }


        //------------------------------------

        // Cylinder properties
        // The cylinder diameter is 7.9cm. The front-most surface point of the
        // cylinder is at z=-7cm from the center of the light-stage, in other
        // words the center axis of the cylinder is at z= -7 - 7.9/2 = -10.95
        // The cylinder axis in world space is off from the Y axis by an angle alpha
        static const float cylDiameter = 7.9f;
        static const float cylRadius = cylDiameter / 2.0f;
        static const float cylZCenter = -7.0f - cylRadius;
        const float cylCosAlpha = 1.0f / sqrt(1 + axisEdge.x * axisEdge.x);
        const float cylAlpha = acos(cylCosAlpha);

        // Sample cylinder on input image for tuning sample placement
        //Image samplesImage(inImage.getWidth(), inImage.getHeight(), 1);
        Image samplesImage(inImage);
        LightStageCylinderBsdfSlice dummyTop(true, 16,
                LightStage::sSeparatedLightCount, cylRadius, cylZCenter, cylAlpha);
        LightStageCylinderBsdfSlice dummyBottom(false, 16,
                LightStage::sSeparatedLightCount, cylRadius, cylZCenter, cylAlpha);
        sampleCylinder(topEdge, bottomEdge, xRange, -1, samplesImage,
                dummyTop, dummyBottom);
        if (!outSamplesFilename.empty()) {
            Logger::info("Writing samples image: \"" , outSamplesFilename , "\"");
            samplesImage.saveAs(outSamplesFilename);
        }

        // Process input sequence
        if (!inSeqPrefix.empty()) {
            LightStageCylinderBsdfSlice bsdfSliceDifTop(true, 90,
                    LightStage::sSeparatedLightCount, cylRadius, cylZCenter, cylAlpha);
            LightStageCylinderBsdfSlice bsdfSliceDifBottom(false, 90,
                    LightStage::sSeparatedLightCount, cylRadius, cylZCenter, cylAlpha);
            LightStageCylinderBsdfSlice bsdfSliceSpecTop(true, 90,
                    LightStage::sSeparatedLightCount, cylRadius, cylZCenter, cylAlpha);
            LightStageCylinderBsdfSlice bsdfSliceSpecBottom(false, 90,
                    LightStage::sSeparatedLightCount, cylRadius, cylZCenter, cylAlpha);
            Logger::info("Processing input sequence: ");
            // We ignore last light which causes big flare
#if 1
            for (int i = 0; i < LightStage::sSeparatedLightCount - 1; i++) {
#else
            for (int i = 0; i < 3; i++) {
#endif
                sampleLight(inImage, topEdge, bottomEdge, xRange,
                        inSeqPrefix, i, ignoreFrames,
                        bsdfSliceDifTop, bsdfSliceDifBottom,
                        bsdfSliceSpecTop, bsdfSliceSpecBottom);
            }
            if (!outBsdfSlicePrefix.empty()) {
                std::string filenameDifTop = outBsdfSlicePrefix + "_dif_" + outBrdfPhi + "_top.slice";
                Logger::info("Writing bsdf slice file: \"" , filenameDifTop , "\"");
                bsdfSliceDifTop.saveAs(filenameDifTop);
                std::string filenameDifBottom = outBsdfSlicePrefix + "_dif_" + outBrdfPhi + "_bottom.slice";
                Logger::info("Writing bsdf slice file: \"" , filenameDifBottom , "\"");
                bsdfSliceDifBottom.saveAs(filenameDifBottom);
                std::string filenameSpecTop = outBsdfSlicePrefix + "_spc_" + outBrdfPhi + "_top.slice";
                Logger::info("Writing bsdf slice file: \"" , filenameSpecTop , "\"");
                bsdfSliceSpecTop.saveAs(filenameSpecTop);
                std::string filenameSpecBottom = outBsdfSlicePrefix + "_spc_" + outBrdfPhi + "_bottom.slice";
                Logger::info("Writing bsdf slice file: \"" , filenameSpecBottom , "\"");
                bsdfSliceSpecBottom.saveAs(filenameSpecBottom);
            }
        }

    } catch (const std::exception& e) {
        Logger::error(e.what());
        std::exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------

