// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// Embree code common to the different Embree intersectors.

#include <limits>

#pragma once

inline __device__
float rcp(const float x)
{
    return 1.f / x;
}

inline __device__
float select(const bool s, const float t, const float f)
{
    return s ? t : f;
}

inline __device__
float3 select(const bool s, const float3& t, const float3& f)
{
    return s ? t : f;
}

inline __device__
float sqr(const float a)
{
    return a * a;
}

inline __device__
float clamp(const float x, const float lower = 0.f, const float upper = 1.f)
{
    return fmaxf(fminf(x, upper), lower);
}

// Calling this normalize() causes compiler confusion in some code as it
// can't figure out which one to use.
inline __device__
float4 normalize4(const float4& a)
{
    return a*rsqrtf(dot3(a,a));
}

// differentiated normalization
inline __device__
float4 dnormalize(const float4& p, const float4& dp)
{
    const float pp  = dot3(p,p);
    const float pdp = dot3(p,dp);
    return (pp*dp-pdp*p)*rcp(pp)*rsqrtf(pp);
}

inline __device__
unsigned int bsf(unsigned int v)
{
    // Count trailing zero bits.  Embree calls _tzcnt_u32() here but that's not available in CUDA.
    if (v == 0) {
        return 32;
    }
    unsigned int count = 0;
    while ((v & 1) == 0) {
        count += 1;
        v >>= 1;
    }
    return count;
}

constexpr __device__ float pos_inf = std::numeric_limits<float>::infinity();
constexpr __device__ float inf = pos_inf;
constexpr __device__ float neg_inf = -pos_inf;
constexpr __device__ float ulp = std::numeric_limits<float>::epsilon();
constexpr __device__ float min_rcp_input = 1E-18f;

struct EmbreeRayHit
{
    float3 org;
    float3 dir;
    float tnear;
    float tfar;
    float3 Ng;
    float u;
    float v;
};

struct BBox
{
    float4 lower, upper;

    inline __device__
    BBox() {}

    inline __device__
    BBox(const float4& v) : lower(v), upper(v) {}

    inline __device__
    BBox(const float4& lower, const float4& upper) : lower(lower), upper(upper) {}
};

inline __device__
BBox merge(const BBox& a, const BBox& b)
{
    return BBox(min(a.lower, b.lower), max(a.upper, b.upper));
}

inline __device__
BBox merge(const BBox& a, const BBox& b, const BBox& c, const BBox& d)
{
    return merge(merge(a,b), merge(c,d));
}

struct BBox1
{
    float lower, upper;

    inline __device__
    BBox1() {}

    inline __device__
    BBox1(const float& v) : lower(v), upper(v) {}

    inline __device__
    BBox1(const float& lower, const float& upper) : lower(lower), upper(upper) {}
};

inline __device__
BBox1 intersect(const BBox1& a, const BBox1& b)
{
    return BBox1(fmaxf(a.lower, b.lower), fminf(a.upper, b.upper));
}

inline __device__
void subtract(const BBox1& a, const BBox1& b, BBox1& c, BBox1& d)
{
    c.lower = a.lower;
    c.upper = fminf(a.upper,b.lower);
    d.lower = fmaxf(a.lower,b.upper);
    d.upper = a.upper;
}

struct HalfPlane
{
    const float4 P;  //!< plane origin
    const float4 N;  //!< plane normal

    inline __device__ HalfPlane(const float4& P, const float4& N)
      : P(P), N(N) {}

    inline __device__
    BBox1 intersect(const float4& ray_org, const float4& ray_dir) const
    {
        float4 O = ray_org - P;
        float4 D = ray_dir;
        float ON = dot3(O,N);
        float DN = dot3(D,N);
        bool eps = abs(DN) < min_rcp_input;
        float t = -ON*rcp(DN);
        float lower = select(eps || DN < 0.0f, float(neg_inf), t);
        float upper = select(eps || DN > 0.0f, float(pos_inf), t);
        return BBox1(lower,upper);
    }
};

////////////////////////////////////////////////////////////////////////////////
/// 2D Linear Transform (2x2 Matrix)
////////////////////////////////////////////////////////////////////////////////
struct LinearSpace2
{
    /*! default matrix constructor */
    inline __device__
    LinearSpace2() {}

    /*! matrix construction from column vectors */
    inline __device__
    LinearSpace2(const float2& vx, const float2& vy)
      : vx(vx), vy(vy) {}

    /*! matrix construction from row major data */
    inline __device__
    LinearSpace2(const float m00, const float m01,
                 const float m10, const float m11)
    {
        vx.x = m00;
        vx.y = m10;
        vy.x = m01;
        vy.y = m11;
    }

    /*! compute the determinant of the matrix */
    inline __device__
    float det() const { return vx.x*vy.y - vx.y*vy.x; }

    /*! compute adjoint matrix */
    inline __device__
    LinearSpace2 adjoint() const { return LinearSpace2(vy.y,-vy.x,-vx.y,vx.x); }

    /*! compute inverse matrix */
    inline __device__
    LinearSpace2 inverse() const;

    /*! the column vectors of the matrix */
    float2 vx,vy;
};

inline __device__
LinearSpace2 rcp(const LinearSpace2& a)
{
    return a.inverse();
}

inline __device__
float2 operator*(const LinearSpace2& a, const float2& b)
{
    return b.x*a.vx + b.y*a.vy;
}

inline __device__
LinearSpace2 operator/(const LinearSpace2& a, const float b)
{
    return LinearSpace2(a.vx/b, a.vy/b);
}

inline __device__
LinearSpace2 LinearSpace2::inverse() const
{
    return adjoint()/det();
}

struct Cylinder
{
    const float4 p0;  //!< start location
    const float4 p1;  //!< end position
    const float rr;   //!< squared radius of cylinder

    inline __device__ Cylinder(const float4& p0, const float4& p1, const float r)
        : p0(p0), p1(p1), rr(sqr(r)) {}

    inline __device__
    bool intersect(const float4& org,
                   const float4& dir,
                   BBox1& t_o,
                   float& u0_o, float4& Ng0_o,
                   float& u1_o, float4& Ng1_o) const
    {
        /* calculate quadratic equation to solve */
        const float rl = 1.f/length3(p1-p0);
        const float4 P0 = p0, dP = (p1-p0)*rl;
        const float4 O = org-P0, dO = dir;

        const float dOdO = dot3(dO,dO);
        const float OdO = dot3(dO,O);
        const float OO = dot3(O,O);
        const float dOz = dot3(dP,dO);
        const float Oz = dot3(dP,O);

        const float A = dOdO - sqr(dOz);
        const float B = 2.0f * (OdO - dOz*Oz);
        const float C = OO - sqr(Oz) - rr;

        /* we miss the cylinder if determinant is smaller than zero */
        const float D = B*B - 4.0f*A*C;
        if (D < 0.0f) {
            t_o = BBox1(pos_inf,neg_inf);
            return false;
        }

        /* special case for rays that are parallel to the cylinder */
        const float eps = 16.0f*float(ulp)*max(abs(dOdO),abs(sqr(dOz)));
        if (fabsf(A) < eps)
        {
            if (C <= 0.0f) {
                t_o = BBox1(neg_inf,pos_inf);
                return true;
            } else {
                t_o = BBox1(pos_inf,neg_inf);
                return false;
            }
        }

        /* standard case for rays that are not parallel to the cylinder */
        const float Q = sqrt(D);
        const float rcp_2A = rcp(2.0f*A);
        const float t0 = (-B-Q)*rcp_2A;
        const float t1 = (-B+Q)*rcp_2A;

        /* calculates u and Ng for near hit */
        {
            u0_o = (t0*dOz+Oz)*rl;
            const float4 Pr = t0*dir;
            const float4 Pl = u0_o*(p1-p0)+p0;
            Ng0_o = Pr-Pl;
        }

        /* calculates u and Ng for far hit */
        {
            u1_o = (t1*dOz+Oz)*rl;
            const float4 Pr = t1*dir;
            const float4 Pl = u1_o*(p1-p0)+p0;
            Ng1_o = Pr-Pl;
        }

        t_o.lower = t0;
        t_o.upper = t1;
        return true;
    }
};
