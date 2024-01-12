// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "SamplingConstants.h"
#include <moonray/rendering/pbr/sampler/Sampling_ispc_stubs.h>

#include <algorithm>

namespace moonray {
namespace pbr {

class PixelFilter
{
public:
    explicit PixelFilter(float footprint) : mFootprint(footprint) {}
    virtual ~PixelFilter() {}

    // This assumes a separatable filter. Call twice, once for each dimension.
    void apply(utype dataSize, float* data) const
    {
        applyFilter(dataSize, data);
        applyOffset(dataSize, data);
    }

protected:
    float getDesiredFootprint() const { return mFootprint; }

private:
    virtual void applyFilter(utype dataSize, float* data) const = 0;
    virtual void applyOffset(utype dataSize, float* data) const = 0;

    float mFootprint;
};

class BoxPixelFilter final : public PixelFilter
{
public:
    explicit BoxPixelFilter(float footprint) :
        PixelFilter(footprint)
    {
    }

private:
    void applyFilter(utype /*dataSize*/, float* /*data*/) const override
    {
        // It's box...do nothing.
    }

    void applyOffset(utype dataSize, float* data) const override
    {
        const float widthMultiplier = getDesiredFootprint();

        std::transform(data, data + dataSize, data,
                [widthMultiplier](float x) {
                    // The mean is 0.5. To scale property, we need to transform to
                    // 0, scale, and then transform back.
                    return (x - 0.5f) * widthMultiplier + 0.5f;
                });
    }
};

class CubicBSplinePixelFilter final : public PixelFilter
{
public:
    explicit CubicBSplinePixelFilter(float footprint) :
        PixelFilter(footprint)
    {
    }

private:
    void applyFilter(utype dataSize, float* data) const override
    {
        ispc::PBR_cubicBSplineWarp(dataSize, data);
    }

    void applyOffset(utype dataSize, float* data) const override
    {
        const float filterFootprint = 4.0f;
        const float widthMultiplier = getDesiredFootprint()/filterFootprint;

        // The pixel filters will move the sample so that the mean is around
        // zero. Without the pixel filters, the mean is 0.5.
        ispc::PBR_filterOffset(dataSize, data, widthMultiplier);
    }
};

class QuadraticBSplinePixelFilter final : public PixelFilter
{
public:
    explicit QuadraticBSplinePixelFilter(float footprint) :
        PixelFilter(footprint)
    {
    }

private:
    void applyFilter(utype dataSize, float* data) const override
    {
        ispc::PBR_quadraticBSplineWarp(dataSize, data);
    }

    void applyOffset(utype dataSize, float* data) const override
    {
        const float filterFootprint = 3.0f;
        const float widthMultiplier = getDesiredFootprint()/filterFootprint;

        // The pixel filters will move the sample so that the mean is around
        // zero. Without the pixel filters, the mean is 0.5.
        ispc::PBR_filterOffset(dataSize, data, widthMultiplier);
    }
};

} // namespace pbr
} // namespace moonray

