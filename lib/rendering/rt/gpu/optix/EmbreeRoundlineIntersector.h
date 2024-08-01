// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// This is pulled from Embree 4 almost verbatim so we match Embree's round
// linear curves behavior.  There are things in here that can be simplified
// or cleaned up, but we don't want to do that because we want to remain 1:1 with
// the original Embree code for 100% matching behavior.

#include <limits>

#include "EmbreeSupport.h"

#pragma once

/*

  This file implements the intersection of a ray with a round linear
  curve segment. We define the geometry of such a round linear curve
  segment from point p0 with radius r0 to point p1 with radius r1
  using the cone that touches spheres p0/r0 and p1/r1 tangentially
  plus the sphere p1/r1. We denote the tangentially touching cone from
  p0/r0 to p1/r1 with cone(p0,r0,p1,r1) and the cone plus the ending
  sphere with cone_sphere(p0,r0,p1,r1).

  For multiple connected round linear curve segments this construction
  yield a proper shape when viewed from the outside. Using the
  following CSG we can also handle the interior in most common cases:

     round_linear_curve(pl,rl,p0,r0,p1,r1,pr,rr) =
       cone_sphere(p0,r0,p1,r1) - cone(pl,rl,p0,r0) - cone(p1,r1,pr,rr)

  Thus by subtracting the neighboring cone geometries, we cut away
  parts of the center cone_sphere surface which lie inside the
  combined curve. This approach works as long as geometry of the
  current cone_sphere penetrates into direct neighbor segments only,
  and not into segments further away.

  To construct a cone that touches two spheres at p0 and p1 with r0
  and r1, one has to increase the cone radius at r0 and r1 to obtain
  larger radii w0 and w1, such that the infinite cone properly touches
  the spheres.  From the paper "Ray Tracing Generalized Tube
  Primitives: Method and Applications"
  (https://www.researchgate.net/publication/334378683_Ray_Tracing_Generalized_Tube_Primitives_Method_and_Applications)
  one can derive the following equations for these increased
  radii:

     sr = 1.0f / sqrt(1-sqr(dr)/sqr(p1-p0))
     w0 = sr*r0
     w1 = sr*r1

  Further, we want the cone to start where it touches the sphere at p0
  and to end where it touches sphere at p1.  Therefore, we need to
  construct clipping locations y0 and y1 for the start and end of the
  cone. These start and end clipping location of the cone can get
  calculated as:

     Y0 =               - r0 * (r1-r0) / length(p1-p0)
     Y1 = length(p1-p0) - r1 * (r1-r0) / length(p1-p0)

  Where the cone starts a distance Y0 and ends a distance Y1 away of
  point p0 along the cone center. The distance between Y1-Y0 can get
  calculated as:

    dY = length(p1-p0) - (r1-r0)^2 / length(p1-p0)

  In the code below, Y will always be scaled by length(p1-p0) to
  obtain y and you will find the terms r0*(r1-r0) and
  (p1-p0)^2-(r1-r0)^2.

 */

struct RoundLineIntersectorHit
{
    inline __device__
    RoundLineIntersectorHit() {}

    inline __device__
    RoundLineIntersectorHit(const float u, const float v, const float t, const float3& Ng)
      : u(u), v(v), t(t), Ng(Ng) {}

public:
    float u;
    float v;
    float t;
    float3 Ng;
};

struct ConeGeometry
{
    inline __device__
    ConeGeometry (const float4& a, const float4& b) :
        p0{make_float3(a)},
        p1{make_float3(b)},
        dP(p1-p0),
        dPdP(dot(dP,dP)),
        r0(a.w),
        sqr_r0(sqr(r0)),
        r1(b.w),
        dr(r1-r0),
        drdr(dr*dr),
        r0dr(r0*dr),
        g(dPdP - drdr) {}

/*
    This function tests if a point is accepted by first cone
    clipping plane.

    First, we need to project the point onto the line p0->p1:

        Y = (p-p0)*(p1-p0)/length(p1-p0)

    This value y is the distance to the projection point from
    p0. The clip distances are calculated as:

        Y0 =               - r0 * (r1-r0) / length(p1-p0)
        Y1 = length(p1-p0) - r1 * (r1-r0) / length(p1-p0)

    Thus to test if the point p is accepted by the first
    clipping plane we need to test Y > Y0 and to test if it
    is accepted by the second clipping plane we need to test
    Y < Y1.

    By multiplying the calculations with length(p1-p0) these
    calculation can get simplied to:

        y = (p-p0)*(p1-p0)
        y0 =           - r0 * (r1-r0)
        y1 = (p1-p0)^2 - r1 * (r1-r0)

    and the test y > y0 and y < y1.
*/
    __device__
    bool isClippedByPlane (const bool valid_i, const float3& p) const
    {
        const float3 p0p = p - p0;
        const float y = dot(p0p, dP);
        const float cap0 = -r0dr;
        const bool inside_cone = y > cap0;
        return valid_i & (p0.x != inf) & (p1.x != inf) & inside_cone;
    }

/*
    This function tests whether a point lies inside the capped cone
    tangential to its ending spheres.

    Therefore one has to check if the point is inside the
    region defined by the cone clipping planes, which is
    performed similar as in the previous function.

    To perform the inside cone test we need to project the
    point onto the line p0->p1:

        dP = p1-p0
        Y = (p-p0)*dP/length(dP)

    This value Y is the distance to the projection point from
    p0. To obtain a parameter value u going from 0 to 1 along
    the line p0->p1 we calculate:

        U = Y/length(dP)

    The radii to use at points p0 and p1 are:

        w0 = sr * r0
        w1 = sr * r1
        dw = w1-w0

    Using these radii and u one can directly test if the point
    lies inside the cone using the formula dP*dP < wy*wy with:

        wy = w0 + u*dw
        py = p0 + u*dP - p

    By multiplying the calculations with length(p1-p0) and
    inserting the definition of w can obtain simpler equations:

        y = (p-p0)*dP
        ry = r0 + y/dP^2 * dr
        wy = sr*ry
        py = p0 + y/dP^2*dP - p
        y0 =      - r0 * dr
        y1 = dP^2 - r1 * dr

    Thus for the in-cone test we get:

        py^2 < wy^2
        <=>  py^2 < sr^2 * ry^2
        <=>  py^2 * ( dP^2 - dr^2 ) < dP^2 * ry^2

    This can further get simplified to:

        (p0-p)^2 * (dP^2 - dr^2) - y^2 < dP^2 * r0^2 + 2.0f*r0*dr*y;
*/
    __device__
    bool isInsideCappedCone (const bool valid_i, const float3& p) const
    {
        const float3 p0p = p - p0;
        const float y = dot(p0p, dP);
        const float cap0 = -r0dr + ulp;
        const float cap1 = -r1 * dr + dPdP;

        bool inside_cone = valid_i & (p0.x != inf) & (p1.x != inf);
        inside_cone &= y > cap0;  // start clipping plane
        inside_cone &= y < cap1;  // end clipping plane
        inside_cone &= sqr(p0p) * g - sqr(y) < dPdP * sqr_r0 + 2.0f * r0dr * y; // in cone test
        return inside_cone;
    }

protected:
    float3 p0;
    float3 p1;
    float3 dP;
    float dPdP;
    float r0;
    float sqr_r0;
    float r1;
    float dr;
    float drdr;
    float r0dr;
    float g;
};

struct ConeGeometryIntersector : public ConeGeometry
{
    using ConeGeometry::p0;
    using ConeGeometry::p1;
    using ConeGeometry::dP;
    using ConeGeometry::dPdP;
    using ConeGeometry::r0;
    using ConeGeometry::sqr_r0;
    using ConeGeometry::r1;
    using ConeGeometry::dr;
    using ConeGeometry::r0dr;
    using ConeGeometry::g;

    inline __device__
    ConeGeometryIntersector(const float3& ray_org, const float3& ray_dir, const float& dOdO, const float& rcp_dOdO, const float4& a, const float4& b)
        : ConeGeometry(a,b),
        org(ray_org),
        O(ray_org-p0),
        dO(ray_dir),
        dOdO(dOdO),
        rcp_dOdO(rcp_dOdO),
        OdP(dot(dP,O)),
        dOdP(dot(dP,dO)),
        yp(OdP + r0dr) {}

/*
    This function intersects a ray with a cone that touches a
    start sphere p0/r0 and end sphere p1/r1.

    To find this ray/cone intersections one could just
    calculate radii w0 and w1 as described above and use a
    standard ray/cone intersection routine with these
    radii. However, it turns out that calculations can get
    simplified when deriving a specialized ray/cone
    intersection for this special case. We perform
    calculations relative to the cone origin p0 and define:

        O  = ray_org - p0
        dO = ray_dir
        dP = p1-p0
        dr = r1-r0
        dw = w1-w0

    For some t we can compute the potential hit point h = O + t*dO and
    project it onto the cone vector dP to obtain u = (h*dP)/(dP*dP). In
    case of an intersection, the squared distance from the hit point
    projected onto the cone center line to the hit point should be equal
    to the squared cone radius at u:

        (u*dP - h)^2 = (w0 + u*dw)^2

    Inserting the definition of h, u, w0, and dw into this formula, then
    factoring out all terms, and sorting by t^2, t^1, and t^0 terms
    yields a quadratic equation to solve.

    Inserting u:
        ( (h*dP)*dP/dP^2 - h )^2 = ( w0 + (h*dP)*dw/dP^2 )^2

    Multiplying by dP^4:
        ( (h*dP)*dP - h*dP^2 )^2 = ( w0*dP^2 + (h*dP)*dw )^2

    Inserting w0 and dw:
        ( (h*dP)*dP - h*dP^2 )^2 = ( r0*dP^2 + (h*dP)*dr )^2 / (1-dr^2/dP^2)
        ( (h*dP)*dP - h*dP^2 )^2 *(dP^2 - dr^2) = dP^2 * ( r0*dP^2 + (h*dP)*dr )^2

    Now one can insert the definition of h, factor out, and presort by t:
        ( ((O + t*dO)*dP)*dP - (O + t*dO)*dP^2 )^2 *(dP^2 - dr^2) = dP^2 * ( r0*dP^2 + ((O + t*dO)*dP)*dr )^2
        ( (O*dP)*dP-O*dP^2 + t*( (dO*dP)*dP - dO*dP^2 ) )^2 *(dP^2 - dr^2) = dP^2 * ( r0*dP^2 + (O*dP)*dr + t*(dO*dP)*dr )^2

    Factoring out further and sorting by t^2, t^1 and t^0 yields:

        0 =   t^2 * [ ((dO*dP)*dP - dO-dP^2)^2 * (dP^2 - dr^2) - dP^2*(dO*dP)^2*dr^2 ]
          + 2*t^1 * [ ((O*dP)*dP - O*dP^2) * ((dO*dP)*dP - dO*dP^2) * (dP^2 - dr^2) - dP^2*(r0*dP^2 + (O*dP)*dr)*(dO*dP)*dr ]
          +   t^0 * [ ( (O*dP)*dP - O*dP^2)^2 * (dP^2-dr^2) - dP^2*(r0*dP^2 + (O*dP)*dr)^2 ]

    This can be simplified to:

        0 =   t^2 * [ (dP^2 - dr^2)*dO^2 - (dO*dP)^2 ]
          + 2*t^1 * [ (dP^2 - dr^2)*(O*dO) - (dO*dP)*(O*dP + r0*dr) ]
          +   t^0 * [ (dP^2 - dr^2)*O^2 - (O*dP)^2 - r0^2*dP^2 - 2.0f*r0*dr*(O*dP) ]

    Solving this quadratic equation yields the values for t at which the
    ray intersects the cone.
*/
    __device__
    bool intersectCone(bool& valid, float& lower, float& upper)
    {
        /* return no hit by default */
        lower = pos_inf;
        upper = neg_inf;

        /* compute quadratic equation A*t^2 + B*t + C = 0 */
        const float OO = dot(O,O);
        const float OdO = dot(dO,O);
        const float A = g * dOdO - sqr(dOdP);
        const float B = 2.0f * (g*OdO - dOdP*yp);
        const float C = g*OO - sqr(OdP) - sqr_r0*dPdP - 2.0f*r0dr*OdP;

        /* we miss the cone if determinant is smaller than zero */
        const float D = B*B - 4.0f*A*C;
        valid &= (D >= 0.0f & g > 0.0f);  // if g <= 0 then the cone is inside a sphere end

        /* When rays are parallel to the cone surface, then the
         * ray may be inside or outside the cone. We just assume a
         * miss in that case, which is fine as rays inside the
         * cone would anyway hit the ending spheres in that
         * case. */
        valid &= abs(A) > min_rcp_input;
        if (!valid) {
            return false;
        }

        /* compute distance to front and back hit */
        const float Q = sqrt(D);
        const float rcp_2A = rcp(2.0f*A);
        t_cone_front = (-B-Q)*rcp_2A;
        y_cone_front = yp + t_cone_front*dOdP;
        lower = select( (y_cone_front > -ulp) & (y_cone_front <= g) & (g > 0.0f), t_cone_front, pos_inf);
#if !defined (EMBREE_BACKFACE_CULLING_CURVES)
        t_cone_back = (-B+Q)*rcp_2A;
        y_cone_back  = yp + t_cone_back *dOdP;
        upper = select( (y_cone_back  > -ulp) & (y_cone_back  <= g) & (g > 0.0f), t_cone_back , neg_inf);
#endif
        return true;
    }

/*
    This function intersects the ray with the end sphere at
    p1. We already clip away hits that are inside the
    neighboring cone segment.
*/
    __device__
    void intersectEndSphere(bool& valid,
                            const ConeGeometry& coneR,
                            float& lower, float& upper)
    {
        /* calculate front and back hit with end sphere */
        const float3 O1 = org - p1;
        const float O1dO = dot(O1,dO);
        const float h2 = sqr(O1dO) - dOdO*(sqr(O1) - sqr(r1));
        const float rhs1 = select( h2 >= 0.0f, sqrt(h2), neg_inf );

        /* clip away front hit if it is inside next cone segment */
        t_sph1_front = (-O1dO - rhs1)*rcp_dOdO;
        const float3 hit_front = org + t_sph1_front*dO;
        bool valid_sph1_front = h2 >= 0.0f & yp + t_sph1_front*dOdP > g & !coneR.isClippedByPlane(valid, hit_front);
        lower = select(valid_sph1_front, t_sph1_front, pos_inf);

#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
        /* clip away back hit if it is inside next cone segment */
        t_sph1_back = (-O1dO + rhs1)*rcp_dOdO;
        const float3 hit_back = org + t_sph1_back*dO;
        bool valid_sph1_back = h2 >= 0.0f & yp + t_sph1_back*dOdP > g & !coneR.isClippedByPlane(valid, hit_back);
        upper = select(valid_sph1_back, t_sph1_back, neg_inf);
#else
        upper = neg_inf;
#endif
    }

    __device__
    void intersectBeginSphere(const bool valid,
                              float& lower, float& upper)
    {
        /* calculate front and back hit with end sphere */
        const float3 O1 = org - p0;
        const float O1dO = dot(O1,dO);
        const float h2 = sqr(O1dO) - dOdO*(sqr(O1) - sqr(r0));
        const float rhs1 = select(h2 >= 0.0f, sqrt(h2), neg_inf);

        /* clip away front hit if it is inside next cone segment */
        t_sph0_front = (-O1dO - rhs1)*rcp_dOdO;
        bool valid_sph1_front = valid & h2 >= 0.0f & yp + t_sph0_front*dOdP < 0;
        lower = select(valid_sph1_front, t_sph0_front, pos_inf);

#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
        /* clip away back hit if it is inside next cone segment */
        t_sph0_back = (-O1dO + rhs1)*rcp_dOdO;
        bool valid_sph1_back  = valid & h2 >= 0.0f & yp + t_sph0_back*dOdP < 0;
        upper = select(valid_sph1_back, t_sph0_back, neg_inf);
#else
        upper = neg_inf;
#endif
    }

/*
    This function calculates the geometry normal of some cone hit.

    For a given hit point h (relative to p0) with a cone
    starting at p0 with radius w0 and ending at p1 with
    radius w1 one normally calculates the geometry normal by
    first calculating the parmetric u hit location along the
    cone:

        u = dot(h,dP)/dP^2

    Using this value one can now directly calculate the
    geometry normal by bending the connection vector (h-u*dP)
    from hit to projected hit with some cone dependent value
    dw/sqrt(dP^2) * normalize(dP):

        Ng = normalize(h-u*dP) - dw/length(dP) * normalize(dP)

    The length of the vector (h-u*dP) can also get calculated
    by interpolating the radii as w0+u*dw which yields:

        Ng = (h-u*dP)/(w0+u*dw) - dw/dP^2 * dP

    Multiplying with (w0+u*dw) yield a scaled Ng':

        Ng' = (h-u*dP) - (w0+u*dw)*dw/dP^2*dP

    Inserting the definition of w0 and dw and refactoring
    yield a further scaled Ng'':

        Ng'' = (dP^2 - dr^2) (h-q) - (r0+u*dr)*dr*dP

    Now inserting the definition of u gives and multiplying
    with the denominator yields:

        Ng''' = (dP^2-dr^2)*(dP^2*h-dot(h,dP)*dP) - (dP^2*r0+dot(h,dP)*dr)*dr*dP

    Factoring out, cancelling terms, dividing by dP^2, and
    factoring again yields finally:

        Ng'''' = (dP^2-dr^2)*h - dP*(dot(h,dP) + r0*dr)
*/
    __device__
    float3 Ng_cone(const bool front_hit) const
    {
#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
        const float y = select(front_hit, y_cone_front, y_cone_back);
        const float t = select(front_hit, t_cone_front, t_cone_back);
        const float3 h = O + t*dO;
        return g*h-dP*y;
#else
        const float3 h = O + t_cone_front*dO;
        return g*h-dP*y_cone_front;
#endif
    }

    /* compute geometry normal of sphere hit as the difference
     * vector from hit point to sphere center */
    __device__
    float3 Ng_sphere1(const bool front_hit) const
    {
#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
        const float t_sph1 = select(front_hit, t_sph1_front, t_sph1_back);
        return org+t_sph1*dO-p1;
#else
        return org+t_sph1_front*dO-p1;
#endif
    }

    __device__
    float3 Ng_sphere0(const bool front_hit) const
    {
#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
        const float t_sph0 = select(front_hit, t_sph0_front, t_sph0_back);
        return org+t_sph0*dO-p0;
#else
        return org+t_sph0_front*dO-p0;
#endif
    }

/*
    This function calculates the u coordinate of a
    hit. Therefore we use the hit distance y (which is zero
    at the first cone clipping plane) and divide by distance
    g between the clipping planes.
*/
    __device__
    float u_cone(const bool front_hit) const
    {
#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
        const float y = select(front_hit, y_cone_front, y_cone_back);
        return clamp(y*rcp(g));
#else
        return clamp(y_cone_front*rcp(g));
#endif
    }

private:
    float3 org;
    float3 O;
    float3 dO;
    float dOdO;
    float rcp_dOdO;
    float OdP;
    float dOdP;

    /* for ray/cone intersection */
private:
    float yp;
    float y_cone_front;
    float t_cone_front;
#if !defined (EMBREE_BACKFACE_CULLING_CURVES)
    float y_cone_back;
    float t_cone_back;
#endif

    /* for ray/sphere intersection */
private:
    float t_sph1_front;
    float t_sph0_front;
#if !defined (EMBREE_BACKFACE_CULLING_CURVES)
    float t_sph1_back;
    float t_sph0_back;
#endif
};


struct RoundLinearCurveEpilog
{
    inline __device__
    RoundLinearCurveEpilog(EmbreeRayHit* ray) : ray(ray) {}

    __device__
    bool operator() (const bool valid_i, RoundLineIntersectorHit& hit) const
    {
        if (!valid_i) {
            return false;
        }

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
bool intersectConeSphere(const bool valid_i,
                         const float3& ray_org_in, const float3& ray_dir,
                         const float ray_tnear, const float& ray_tfar,
                         const float4& v0, const float4& v1,
                         const float4& vL, const float4& vR,
                         const RoundLinearCurveEpilog& epilog)
{
    bool valid = valid_i;

    /* move ray origin closer to make calculations numerically stable */
    const float dOdO = sqr(ray_dir);
    const float rcp_dOdO = rcp(dOdO);
    const float3 center = float(0.5f)*(make_float3(v0)+make_float3(v1));
    const float dt = dot(center-ray_org_in,ray_dir)*rcp_dOdO;
    const float3 ray_org = ray_org_in + dt*ray_dir;

    /* intersect with cone from v0 to v1 */
    float t_cone_lower, t_cone_upper;
    ConeGeometryIntersector cone(ray_org, ray_dir, dOdO, rcp_dOdO, v0, v1);
    bool validCone = valid;
    cone.intersectCone(validCone, t_cone_lower, t_cone_upper);

    valid &= (validCone | (cone.g <= 0.0f));  // if cone is entirely in sphere end - check sphere
    if (!valid) {
        return false;
    }

    /* cone hits inside the neighboring capped cones are inside the geometry and thus ignored */
    const ConeGeometry coneL(v0, vL);
    const ConeGeometry coneR(v1, vR);
#if !defined(EMBREE_BACKFACE_CULLING_CURVES)
    const float3 hit_lower = ray_org + t_cone_lower*ray_dir;
    const float3 hit_upper = ray_org + t_cone_upper*ray_dir;
    t_cone_lower = select(!coneL.isInsideCappedCone(validCone, hit_lower) & !coneR.isInsideCappedCone(validCone, hit_lower), t_cone_lower, float(pos_inf));
    t_cone_upper = select(!coneL.isInsideCappedCone(validCone, hit_upper) & !coneR.isInsideCappedCone(validCone, hit_upper), t_cone_upper, float(neg_inf));
#endif

    /* intersect ending sphere */
    float t_sph1_lower, t_sph1_upper;
    float t_sph0_lower = float(pos_inf);
    float t_sph0_upper = float(neg_inf);
    cone.intersectEndSphere(valid, coneR, t_sph1_lower, t_sph1_upper);

    const bool isBeginPoint = valid & (vL.x == float(pos_inf));
    if (isBeginPoint) {
        cone.intersectBeginSphere (isBeginPoint, t_sph0_lower, t_sph0_upper);
    }

    /* CSG union of cone and end sphere */
    float t_sph_lower = fminf(t_sph0_lower, t_sph1_lower);
    float t_cone_sphere_lower = fminf(t_cone_lower, t_sph_lower);
#if !defined (EMBREE_BACKFACE_CULLING_CURVES)
    float t_sph_upper = fmaxf(t_sph0_upper, t_sph1_upper);
    float t_cone_sphere_upper = fmaxf(t_cone_upper, t_sph_upper);

    /* filter out hits that are not in tnear/tfar range */
    const bool valid_lower = valid & ray_tnear <= dt+t_cone_sphere_lower & dt+t_cone_sphere_lower <= ray_tfar & t_cone_sphere_lower != float(pos_inf);
    const bool valid_upper = valid & ray_tnear <= dt+t_cone_sphere_upper & dt+t_cone_sphere_upper <= ray_tfar & t_cone_sphere_upper != float(neg_inf);

    /* check if there is a first hit */
    const bool valid_first = valid_lower | valid_upper;
    if (!valid_first) {
        return false;
    }

    /* construct first hit */
    const float t_first = select(valid_lower, t_cone_sphere_lower, t_cone_sphere_upper);
    const bool cone_hit_first = t_first == t_cone_lower | t_first == t_cone_upper;
    const bool sph0_hit_first = t_first == t_sph0_lower | t_first == t_sph0_upper;
    const float3 Ng_first = select(cone_hit_first, cone.Ng_cone(valid_lower), select(sph0_hit_first, cone.Ng_sphere0(valid_lower), cone.Ng_sphere1(valid_lower)));
    const float u_first  = select(cone_hit_first, cone.u_cone(valid_lower), select(sph0_hit_first, 0.f, 1.f));

    /* invoke intersection filter for first hit */
    RoundLineIntersectorHit hit(u_first,0.f,dt+t_first,Ng_first);
    const bool is_hit_first = epilog(valid_first, hit);

    /* check for possible second hits before potentially accepted hit */
    const float t_second = t_cone_sphere_upper;
    const bool valid_second = valid_lower & valid_upper & (dt+t_cone_sphere_upper <= ray_tfar);
    if (!valid_second) {
        return is_hit_first;
    }

    /* invoke intersection filter for second hit */
    const bool cone_hit_second = t_second == t_cone_lower | t_second == t_cone_upper;
    const bool sph0_hit_second = t_second == t_sph0_lower | t_second == t_sph0_upper;
    const float3 Ng_second = select(cone_hit_second, cone.Ng_cone(false), select(sph0_hit_second, cone.Ng_sphere0(false), cone.Ng_sphere1(false)));
    const float u_second  = select(cone_hit_second, cone.u_cone(false), select(sph0_hit_second, 0.f, 1.f));

    hit = RoundLineIntersectorHit(u_second,0.f,dt+t_second,Ng_second);
    const bool is_hit_second = epilog(valid_second, hit);

    return is_hit_first | is_hit_second;
#else
    /* filter out hits that are not in tnear/tfar range */
    const bool valid_lower = valid & ray_tnear <= dt+t_cone_sphere_lower & dt+t_cone_sphere_lower <= ray_tfar & t_cone_sphere_lower != float(pos_inf);

    /* check if there is a valid hit */
    if (!valid_lower) {
        return false;
    }

    /* construct first hit */
    const bool cone_hit_first = t_cone_sphere_lower == t_cone_lower | t_cone_sphere_lower == t_cone_upper;
    const bool sph0_hit_first = t_cone_sphere_lower == t_sph0_lower | t_cone_sphere_lower == t_sph0_upper;
    const float3 Ng_first = select(cone_hit_first, cone.Ng_cone(valid_lower), select (sph0_hit_first, cone.Ng_sphere0(valid_lower), cone.Ng_sphere1(valid_lower)));
    const float u_first  = select(cone_hit_first, cone.u_cone(valid_lower), select (sph0_hit_first, float(zero), float(one)));

    /* invoke intersection filter for first hit */
    RoundLineIntersectorHit hit(u_first,zero,dt+t_cone_sphere_lower,Ng_first);
    const bool is_hit_first = epilog(valid_lower, hit);

    return is_hit_first;
#endif
}

struct RoundLinearCurveIntersector
{
    inline __device__
    bool intersect(const bool valid_i,
                   const EmbreeRayHit& ray,
                   const float4& v0i, const float4& v1i,
                   const float4& vLi, const float4& vRi,
                   const RoundLinearCurveEpilog& epilog)
    {
        return intersectConeSphere(valid_i,ray.org,ray.dir,ray.tnear,ray.tfar,v0i,v1i,vLi,vRi,epilog);
    }
};
