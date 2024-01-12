// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Image.h"

#include <scene_rdl2/common/math/Vec2.h>


//---------------------------------------------------------------------------

// We detect two main edges of a horizontal cylinder in the image that may be
// slightly slanted. The given image is assumed to be the result of a sobel
// filter.
//
// We detect edges in the imageRatio center portion of the image (horizontally)
// to avoid dealing with the ends of the cylinder that may show up at the top
// and/or bottom of the image.
//
// We detect top and bottom edges above the given threshold, scanning outwards
// vertically from the center of the image. Edges are in image space
void detectEdgeSamples(const Image &image, int xMin, int xMax, float threshold,
        std::vector<scene_rdl2::math::Vec2f> &topEdgeSamples,
        std::vector<scene_rdl2::math::Vec2f> &bottomEdgeSamples);

void drawEdgeSamples(const std::vector<scene_rdl2::math::Vec2f> &edgeSamples,
        Image &image);

// Returns the parameters of the edge equation y = a * x + b in image coordinates
void fitEdge(const std::vector<scene_rdl2::math::Vec2f> &edgeSamples, float &a, float &b);

void drawEdge(float a, float b, Image &image);


//---------------------------------------------------------------------------

