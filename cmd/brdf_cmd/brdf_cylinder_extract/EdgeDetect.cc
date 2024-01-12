// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "EdgeDetect.h"


using namespace scene_rdl2::math;


//---------------------------------------------------------------------------

// Compute edge sample location using a weighted average over 5 pixels
// horizontally, centered around yEdge
static Vec2f
computeEdgeSample(const Image &image, int yEdge, int x, float threshold)
{
    const int yMin = max(yEdge - 2, 0);
    const int yMax = min(yEdge + 2, image.getHeight() - 1);

    // Compute weighted average: y = sum(y_i * g_i) / sum(g_i) + 0.5
    float sumYG = 0.0f;
    float sumG = 0.0f;
    for (int y = yMin; y <= yMax; y++) {
        const int index = image.getPixelIndex(x, y);
        const Color c = image.getPixel(index);
        const float g = c.r - threshold;
        if (g > 0.0f) {
            sumYG += y * g;
            sumG += g;
        }
    }

    Vec2f sample(x + 0.5f, sumYG / sumG + 0.5f);
    return sample;
}


//void detectEdgeSamples(const Image &image, float imageRatio, float threshold,
//        std::vector<scene_rdl2::math::Vec2f> &topEdgeSamples,
//        std::vector<scene_rdl2::math::Vec2f> &bottomEdgeSamples);

void
detectEdgeSamples(const Image &image, int xMin, int xMax, float threshold,
        std::vector<Vec2f> &topEdgeSamples, std::vector<Vec2f> &bottomEdgeSamples)
{
    const int yCenter = image.getHeight() / 2;

    // Loop over pixel columns in the center portion
    for (int x = xMin; x < xMax; x++) {
        Vec2i yEdge(-1, -1);
        Vec2f gMax(0.0f, 0.0f);

        // Detect approximate top-most edge
         for (int y = yCenter; y < image.getHeight(); y++) {
            int index = image.getPixelIndex(x, y);
            const Color c = image.getPixel(index);
            if (c.r > threshold  &&  c.r > gMax[0]) {
                gMax[0] = c.r;
                yEdge[0] = y;
            }
        }

        // Detect approximate bottom-most edge
         for (int y = yCenter; y >= 0; y--) {
            int index = image.getPixelIndex(x, y);
            const Color c = image.getPixel(index);
            if (c.r > threshold  &&  c.r > gMax[1]) {
                gMax[1] = c.r;
                yEdge[1] = y;
            }
        }

        // Compute floating point (x,y) coordinates for each edge point sample
        if (yEdge[0] > 0) {
            Vec2f edgeSample = computeEdgeSample(image, yEdge[0], x, threshold);
            topEdgeSamples.push_back(edgeSample);
            //std::cout << "Top = (" << x << ", " << yEdge[0] << ") --> " << edgeSample << std::endl;
        }
        if (yEdge[1] > 0) {
            Vec2f edgeSample = computeEdgeSample(image, yEdge[1], x, threshold);
            bottomEdgeSamples.push_back(edgeSample);
            //std::cout << "Bottom = (" << x << ", " << yEdge[1] << ") --> " << edgeSample << std::endl;
        }
    }
}


void
drawEdgeSamples(const std::vector<Vec2f> &edgeSamples, Image &image)
{
    for (auto &iter : edgeSamples) {
        int x = int(floor(iter.x));
        image.drawPixel(x, iter.y, Color(1.0f));
    }
}


//---------------------------------------------------------------------------

void
fitEdge(const std::vector<Vec2f> &edgeSamples, float &resultA, float &resultB)
{
    resultA = resultB = 0.0f;

    // Average various line samples
    const int count = edgeSamples.size() / 4;
    MNRY_ASSERT(count > 0);

    for (int i=0; i < count; i++) {
        int j = edgeSamples.size() - count + i;

        float a = (edgeSamples[j].y - edgeSamples[i].y) /
                  (edgeSamples[j].x - edgeSamples[i].x);
        float b = edgeSamples[i].y - a * edgeSamples[i].x;

        resultA += a;
        resultB += b;
    }
    resultA /= count;
    resultB /= count;
}


void
drawEdge(float a, float b, Image &image)
{
    int width = image.getWidth();
    for (int x=0; x < width; x++) {
        float y = a * (x + 0.5f) + b;
        image.drawPixel(x, y, Color(0.0f, 1.0f, 0.0f));
    }
}


//---------------------------------------------------------------------------

