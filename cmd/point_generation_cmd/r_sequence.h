// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "NPoint.h"

#include <array>
#include <cmath>

inline float mod1(float x)
{
    float throwaway;
    return std::modf(x, &throwaway);
}

// x^(d+1) = x + 1
// for d == 1, this should be the Golden Ratio
inline float phi(int d)
{
    float x = 2.0f;
    for (int i = 0; i < 10; ++i) {
        x = std::pow(1.0f+x, 1.0f/(d+1));
    }
    return x;
}

template <unsigned D>
class RSequenceGenerator
{
public:
    RSequenceGenerator()
    : mAlpha(generateAlpha())
    {
    }

    NPoint<D> operator()(int n, float seed) const
    {
        NPoint<D> ret;
        for (unsigned i = 0; i < D; ++i) {
            ret[i] = mod1(seed + mAlpha[i]*n);
        }
        return ret;
    }

private:
    static std::array<float, D> generateAlpha()
    {
        const float g = phi(D);
        std::array<float, D> alpha = { 0.0f };
        for (unsigned i = 0; i < D; ++i) {
            alpha[i] = mod1(std::pow(1.0f/g, i+1));
        }
        return alpha;
    }

    std::array<float, D> mAlpha;

};

