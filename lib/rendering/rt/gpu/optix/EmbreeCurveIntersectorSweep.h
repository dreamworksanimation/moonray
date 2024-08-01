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
#include "EmbreeBsplineCurve.h"

#pragma once

static const size_t numBezierSubdivisions = 3;
static const size_t numJacobianIterations = 5;

struct BezierCurveHit
{
    inline __device__
    BezierCurveHit() {}

    inline __device__
    BezierCurveHit(const float t, const float u, const float4& Ng)
        : t(t), u(u), v(0.0f), Ng(Ng) {}

    inline __device__
    BezierCurveHit(const float t, const float u, const float v, const float4& Ng)
        : t(t), u(u), v(v), Ng(Ng) {}

public:
    float t;
    float u;
    float v;
    float4 Ng;
};

struct RoundBezierCurveEpilog
{
    inline __device__
    RoundBezierCurveEpilog(EmbreeRayHit* ray) : ray(ray) {}

    inline __device__
    bool operator() (BezierCurveHit& hit) const
    {
        ray->tfar = hit.t;
        ray->Ng.x = hit.Ng.x;
        ray->Ng.y = hit.Ng.y;
        ray->Ng.z = hit.Ng.z;
        ray->u = hit.u;
        ray->v = hit.v;
        return true;
    }

    EmbreeRayHit* ray;
};

__device__
bool intersect_bezier_iterative_jacobian(const EmbreeRayHit& ray,
                                         const float dt,
                                         const BSplineCurveT& curve,
                                         float u,
                                         float t,
                                         const RoundBezierCurveEpilog& epilog)
{
    const float4 org = {0.f, 0.f, 0.f, 0.f};
    const float4 dir = {ray.dir.x, ray.dir.y, ray.dir.z, 0.f}; // does ray.dir need to be float4?
    const float length_ray_dir = length3(dir);

    /* error of curve evaluations is proportional to largest coordinate */
    const BBox box = curve.bounds();
    const float P_err = 16.0f*float(ulp)*reduce_max3(max(abs(box.lower),abs(box.upper)));

    for (size_t i=0; i<numJacobianIterations; i++)
    {
        const float4 Q = dir * t + org;
        //const float3 dQdu = zero;
        const float4 dQdt = dir;
        const float Q_err = 16.0f*float(ulp)*length_ray_dir*t; // works as org=zero here

        float4 P,dPdu,ddPdu; curve.eval(u,P,dPdu,ddPdu);
        //const Vec3fa dPdt = zero;

        const float4 R = Q-P;
        const float len_R = length3(R); //reduce_max(abs(R));
        const float R_err = max(Q_err,P_err);
        const float4 dRdu = /*dQdu*/make_float4(-dPdu.x, -dPdu.y, -dPdu.z, -dPdu.w);
        const float4 dRdt = dQdt;//-dPdt;

        const float4 T = normalize4(dPdu);
        const float4 dTdu = dnormalize(dPdu,ddPdu);
        //const Vec3fa dTdt = zero;
        const float cos_err = P_err/length3(dPdu);

        /* Error estimate for dot(R,T):

           dot(R,T) = cos(R,T) |R| |T|
                    = (cos(R,T) +- cos_error) * (|R| +- |R|_err) * (|T| +- |T|_err)
                    = cos(R,T)*|R|*|T| 
                      +- cos(R,T)*(|R|*|T|_err + |T|*|R|_err)
                      +- cos_error*(|R| + |T|)
                      +- lower order terms
           with cos(R,T) being in [0,1] and |T| = 1 we get:
             dot(R,T)_err = |R|*|T|_err + |R|_err = cos_error*(|R|+1)
        */

        const float f = dot3(R,T);
        const float f_err = len_R*P_err + R_err + cos_err*(1.0f+len_R);
        const float dfdu = dot3(dRdu,T) + dot3(R,dTdu);
        const float dfdt = dot3(dRdt,T);// + dot(R,dTdt);

        const float K = dot3(R,R)-sqr(f);
        const float dKdu = /*2.0f*/(dot3(R,dRdu)-f*dfdu);
        const float dKdt = /*2.0f*/(dot3(R,dRdt)-f*dfdt);
        const float rsqrt_K = rsqrtf(K);

        const float g = sqrt(K)-P.w;
        const float g_err = R_err + f_err + 16.0f*float(ulp)*box.upper.w;
        const float dgdu = /*0.5f*/dKdu*rsqrt_K-dPdu.w;
        const float dgdt = /*0.5f*/dKdt*rsqrt_K;//-dPdt.w;

        const LinearSpace2 J = LinearSpace2(dfdu,dfdt,dgdu,dgdt);
        const float2 dut = rcp(J)*make_float2(f, g);
        const float2 ut = make_float2(u,t) - dut;
        u = ut.x; t = ut.y;

        if (fabsf(f) < f_err && fabsf(g) < g_err)
        {
            t+=dt;
            if (!(ray.tnear <= t && t <= ray.tfar)) return false; // rejects NaNs
            if (!(u >= 0.0f && u <= 1.0f)) return false; // rejects NaNs
            const float4 R = normalize4(Q-P);
            const float4 U = dPdu.w*R+dPdu;
            const float4 V = cross(dPdu,R);
            BezierCurveHit hit(t,u,cross(V,U));
            return epilog(hit);
        }
    }
    return false;
}

__device__
bool intersect_bezier_recursive_jacobian(const EmbreeRayHit& ray,
                                         const float dt,
                                         const BSplineCurveT& curve,
                                         const RoundBezierCurveEpilog& epilog)
{
    // This is the "scalar" SYCL version of this function

    const float4 org = {0.f, 0.f, 0.f, 0.f};
    const float4 dir = {ray.dir.x, ray.dir.y, ray.dir.z, 0.f}; // does ray.dir need to be float4?
    const unsigned int max_depth = 7;

    bool found = false;

    struct ShortStack
    {
        /* pushes both children */
        inline __device__ void push() {
          depth++;
        }

        /* pops next node */
        inline __device__ void pop() {
          short_stack += (1<<(31-depth));
          depth = 31-bsf(short_stack);
        }

        unsigned int depth = 0;
        unsigned int short_stack = 0;
    };

    ShortStack stack;

    do
    {
        const float u0 = (stack.short_stack+0*(1<<(31-stack.depth)))/float(0x80000000);
        const float u1 = (stack.short_stack+1*(1<<(31-stack.depth)))/float(0x80000000);

        /* subdivide bezier curve */
        float4 P0, dP0du; curve.eval(u0,P0,dP0du); dP0du = dP0du * (u1-u0);
        float4 P3, dP3du; curve.eval(u1,P3,dP3du); dP3du = dP3du * (u1-u0);
        const float4 P1 = P0 + dP0du*(1.0f/3.0f);
        const float4 P2 = P3 - dP3du*(1.0f/3.0f);

        /* check if curve is well behaved, by checking deviation of tangents from straight line */
        const float4 W = make_float4(P3-P0,0.0f);
        const float4 dQ0 = abs(3.0f*(P1-P0) - W);
        const float4 dQ1 = abs(3.0f*(P2-P1) - W);
        const float4 dQ2 = abs(3.0f*(P3-P2) - W);
        const float4 max_dQ = max(dQ0,dQ1,dQ2);
        const float m = fmaxf(max_dQ.x,max_dQ.y,max_dQ.z); //,max_dQ.w);
        const float l = length(make_float3(W));
        const bool well_behaved = m < 0.2f*l;

        if (!well_behaved && stack.depth < max_depth) {
            stack.push();
            continue;
        }

        /* calculate bounding cylinders */
        const float rr1 = sqr_point_to_line_distance(make_float3(dP0du),make_float3(P3-P0));
        const float rr2 = sqr_point_to_line_distance(make_float3(dP3du),make_float3(P3-P0));
        const float maxr12 = sqrt(max(rr1,rr2));
        const float one_plus_ulp  = 1.0f+2.0f*float(ulp);
        const float one_minus_ulp = 1.0f-2.0f*float(ulp);
        float r_outer = fmaxf(P0.w,P1.w,P2.w,P3.w)+maxr12;
        float r_inner = fminf(P0.w,P1.w,P2.w,P3.w)-maxr12;
        r_outer = one_plus_ulp*r_outer;
        r_inner = max(0.0f,one_minus_ulp*r_inner);
        const Cylinder cylinder_outer(P0,P3,r_outer);
        const Cylinder cylinder_inner(P0,P3,r_inner);

        /* intersect with outer cylinder */
        BBox1 tc_outer; float u_outer0; float4 Ng_outer0; float u_outer1; float4 Ng_outer1;
        if (!cylinder_outer.intersect(org,dir,tc_outer,u_outer0,Ng_outer0,u_outer1,Ng_outer1))
        {
            stack.pop();
            continue;
        }

        /* intersect with cap-planes */
        BBox1 tp(ray.tnear-dt,ray.tfar-dt);
        tp = intersect(tp,tc_outer);
        BBox1 h0 = HalfPlane(remove_w(P0),remove_w(dP0du)).intersect(org,dir);
        tp = intersect(tp,h0);
        BBox1 h1 = HalfPlane(remove_w(P3),-remove_w(dP3du)).intersect(org,dir);
        tp = intersect(tp,h1);
        if (tp.lower > tp.upper)
        {
            stack.pop();
            continue;
        }

        bool valid = true;

        /* clamp and correct u parameter */
        u_outer0 = clamp(u_outer0,0.0f,1.0f);
        u_outer1 = clamp(u_outer1,0.0f,1.0f);
        u_outer0 = lerpf(u0,u1,u_outer0);
        u_outer1 = lerpf(u0,u1,u_outer1);

        /* intersect with inner cylinder */
        BBox1 tc_inner;
        float u_inner0 = 0.f; float4 Ng_inner0 = make_float4_zero();
        float u_inner1 = 0.f; float4 Ng_inner1 = make_float4_zero();
        const bool valid_inner = cylinder_inner.intersect(org,dir,tc_inner,u_inner0,Ng_inner0,u_inner1,Ng_inner1);

        /* subtract the inner interval from the current hit interval */
        BBox1 tp0, tp1;
        subtract(tp,tc_inner,tp0,tp1);
        bool valid0 = valid & (tp0.lower <= tp0.upper);
        bool valid1 = valid & (tp1.lower <= tp1.upper);
        if (!(valid0 | valid1))
        {
            stack.pop();
            continue;
        }

        /* at the unstable area we subdivide deeper */
        const bool unstable0 = valid0 && ((!valid_inner) | (abs(dot3(make_float4(ray.dir),Ng_inner0)) < 0.3f));
        const bool unstable1 = valid1 && ((!valid_inner) | (abs(dot3(make_float4(ray.dir),Ng_inner1)) < 0.3f));
    
        if ((unstable0 | unstable1) && (stack.depth < max_depth)) {
            stack.push();
            continue;
        }

        if (valid0)
            found |= intersect_bezier_iterative_jacobian(ray,dt,curve,u_outer0,tp0.lower,epilog);
          
        /* the far hit cannot be closer, thus skip if we hit entry already */
        valid1 &= tp1.lower+dt <= ray.tfar;
        
        /* iterate over second hit */
        if (valid1)
            found |= intersect_bezier_iterative_jacobian(ray,dt,curve,u_outer1,tp1.upper,epilog);

        stack.pop();
        
    } while (stack.short_stack != 0x80000000);

    return found;
}

struct SweepCurveIntersector
{
    inline __device__
    bool intersect(const EmbreeRayHit& ray,
                   const float4& v0, const float4& v1,
                   const float4& v2, const float4& v3,
                   const RoundBezierCurveEpilog& epilog)
    {
        /* move ray closer to make intersection stable */
        BSplineCurveT curve0(v0,v1,v2,v3);
        // curve0 = enlargeRadiusToMinWidth(context,geom,ray.org,curve0); // need this?
        const float dt = dot(make_float3(curve0.center())-ray.org,ray.dir)*rcp(dot(ray.dir,ray.dir));
        const float4 ref = make_float4(ray.dir * dt + ray.org, 0.0f);
        const BSplineCurveT curve1 = curve0-ref;
        return intersect_bezier_recursive_jacobian(ray,dt,curve1,epilog);
    }
};
