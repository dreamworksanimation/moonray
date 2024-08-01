// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// This is pulled from Embree 4 almost verbatim so we match Embree's round
// bspline curves behavior.  There are things in here that can be simplified
// or cleaned up, but we don't want to do that because we want to remain 1:1 with
// the original Embree code for 100% matching behavior.

#include <limits>

#include "EmbreeSupport.h"

#pragma once

class BSplineBasis
{
public:
    static inline __device__
    float4 eval(const float u)
    {
        const float t  = u;
        const float s  = 1.0f - u;
        const float n0 = s*s*s;
        const float n1 = (4.0f*(s*s*s)+(t*t*t)) + (12.0f*((s*t)*s) + 6.0f*((t*s)*t));
        const float n2 = (4.0f*(t*t*t)+(s*s*s)) + (12.0f*((t*s)*t) + 6.0f*((s*t)*s));
        const float n3 = t*t*t;
        const float4 n = {n0,n1,n2,n3};
        return (1.0f/6.0f)*n;
    }

    static inline __device__
    float4 derivative(const float u)
    {
        const float t  =  u;
        const float s  =  1.0f - u;
        const float n0 = -s*s;
        const float n1 = -t*t - 4.0f*(t*s);
        const float n2 =  s*s + 4.0f*(s*t);
        const float n3 =  t*t;
        const float4 n = {n0,n1,n2,n3};
        return (0.5f)*n;
    }

    static inline __device__
    float4 derivative2(const float u)
    {
        const float t  =  u;
        const float s  =  1.0f - u;
        const float n0 = s;
        const float n1 = t - 2.0f*s;
        const float n2 = s - 2.0f*t;
        const float n3 = t;
        const float4 n = {n0,n1,n2,n3};
        return n;
    }
};

struct BSplineCurveT
{
    float4 v0, v1, v2, v3;

    inline __device__
    BSplineCurveT(const float4& v0, const float4& v1, const float4& v2, const float4& v3)
        : v0(v0), v1(v1), v2(v2), v3(v3) {}

    inline __device__
    BBox bounds() const
    {
        return merge(BBox(v0), BBox(v1), BBox(v2), BBox(v3));
    }

    inline __device__ float4 center() const
    {
        return 0.25f*(v0+v1+v2+v3);
    }

    inline __device__
    float4 eval(const float t) const
    {
        const float4 b = BSplineBasis::eval(t);
        return b.x*v0+b.y*v1+b.z*v2+b.w*v3;
    }

    inline __device__
    float4 eval_du(const float t) const
    {
        const float4 b = BSplineBasis::derivative(t);
        return b.x*v0+b.y*v1+b.z*v2+b.w*v3;
    }

    inline __device__
    float4 eval_dudu(const float t) const
    {
        const float4 b = BSplineBasis::derivative2(t);
        return b.x*v0+b.y*v1+b.z*v2+b.w*v3;
    }

    inline __device__
    void eval(const float t, float4& p, float4& dp) const
    {
        p = eval(t);
        dp = eval_du(t);
    }

    inline __device__
    void eval(const float t, float4& p, float4& dp, float4& ddp) const
    {
        p = eval(t);
        dp = eval_du(t);
        ddp = eval_dudu(t);
    }

    inline __device__
    BSplineCurveT friend operator -(const BSplineCurveT& a, const float4& b)
    {
        return BSplineCurveT(a.v0-b,a.v1-b,a.v2-b,a.v3-b);
    }
};
