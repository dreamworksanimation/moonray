// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestRender.ispc

#include <moonray/rendering/shading/ispc/EvalShader.isph>
#include <moonray/rendering/shading/ispc/Shading.isph>
#include <moonray/rendering/shading/ispc/bsdf/Bsdf.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>

#include <scene_rdl2/common/math/ispc/autodiff.isph>
#include <scene_rdl2/render/util/Random.isph>

// simple user inputs for this test.
struct Scene
{
    const uniform Material * uniform mMat;
    uniform int mWidth;
    uniform int mHeight;
    uniform int mRaysPerPixel;
    uniform ShadingTLState * uniform mShadingTLState;
};

     
#define MAX_RAYS_PP 128

static void
computeIsect(const varying float sx, const varying float sy, // screen x, y hitpoint
             const varying Vec3f &origin, // sphere origin
             const varying float r, // sphere radius
             varying Intersection &isect,
             const uniform int raysPerPixel,
             const uniform bool indirect
             )
{
    // to compute differential geometry using dual numbers, build
    // a chain of functions P(s, t) < St(x, y) <- (x, y)

    // seed x and y such that  we read dx from b0 and dy from b1
    const Dual x = Dual_ctor(sx, 1.0, 0.0, 0.0);
    const Dual y = Dual_ctor(sy, 0.0, 1.0, 0.0);

    const Dual xd = x - origin.x;
    const Dual yd = y - origin.y;
    const Dual zd = sqrt(r * r - xd * xd - yd * yd);

    // for surface parameterization,
    // use spherical coords chosen such that the seam is hidden down -z
    const Dual phi = atan2(xd, zd);     // (-PI->PI longitude)
    const Dual theta = acos(yd / r);    // (0->PI latitude)
    const Dual s = (phi + PI) / (2. * PI);
    const Dual t = (PI - theta) / PI;

    const Dual px = origin.x + r * sin(theta) * sin(phi);
    const Dual py = origin.y + r * cos(theta);
    const Dual pz = origin.z + r * sin(theta) * cos(phi);

    // now fill out the isect
    isect.mP = Vec3f_ctor(eval(px), eval(py), eval(pz));
    isect.mSt = Vec2f_ctor(eval(s), eval(t));

    const float footprint = (raysPerPixel >= 4  ?  0.5f  :  1.0f);
    isect.mdSdx = dDx(s) * footprint;
    isect.mdTdx = dDx(t) * footprint;
    isect.mdSdy = dDy(s) * footprint;
    isect.mdTdy = dDy(t) * footprint;

    // we have dpdx, dpdy, dsdx, dsdy, dtdx, and dtdy.
    // we would like to compute dpds and dpdt.
    // we can use the relationships:
    //   dpdx = dpds * dsdx + dpdt * dtdx
    //   dpdy = dpds * dsdy + dpdt * dtdy
    // and solve for dpds and dpdt as a system of 2 equations
    // with 2 unknowns
    const float dsdx = isect.mdSdx;
    const float dsdy = isect.mdSdy;
    const float dtdx = isect.mdTdx;
    const float dtdy = isect.mdTdy;
    const Vec3f dpdx = Vec3f_ctor(dDx(px), dDx(py), dDx(pz));
    const Vec3f dpdy = Vec3f_ctor(dDy(px), dDy(py), dDy(pz));

    // Determinant gets naturally very small, so we need extra small epsilon
    const float det = dsdx * dtdy - dsdy * dtdx;
    assert(!isEqual(det, 0.0f, 1e-8f));
    const Vec3f dpds = (dtdy * dpdx - dtdx * dpdy) / det;
    const Vec3f dpdt = (-dsdy * dpdx + dsdx * dpdy) / det;
    isect.mdPds = dpds;
    isect.mdPdt = dpdt;

    // computing the normal is now just a cross product of our
    // differentials
    isect.mNg = normalize(cross(isect.mdPds, isect.mdPdt));
    isect.mN = isect.mNg;

    // and the rest...
    Flags_clearAll(&isect.mFlags);
    if (indirect) {
        setPathType(&isect, PATH_TYPE_INDIRECT_MIRROR);
    }
    Flags_set(&isect.mFlags, INTERSECTION_FLAGS_CAUSTIC_PATH, false);

    isect.mMinRoughness = Vec2f_ctor(0.0f);

    // Add this line to initialize the wo direction
    isect.mWo = Vec3f_ctor(0.0f, 0.0f, 1.0f);

    assert(!isnan(isect.mP.x));
    assert(!isnan(isect.mP.y));
    assert(!isnan(isect.mP.z));
    assert(!isnan(isect.mSt.x));
    assert(!isnan(isect.mSt.y));
    assert(!isnan(isect.mdSdx));
    assert(!isnan(isect.mdTdx));
    assert(!isnan(isect.mdSdy));
    assert(!isnan(isect.mdTdy));
    assert(!isnan(isect.mdPds.x));
    assert(!isnan(isect.mdPds.y));
    assert(!isnan(isect.mdPds.z));
    assert(!isnan(isect.mdPdt.x));
    assert(!isnan(isect.mdPdt.y));
    assert(!isnan(isect.mdPdt.z));
    assert(!isnan(isect.mN.x));
    assert(!isnan(isect.mN.y));
    assert(!isnan(isect.mN.z));
    assert(!isnan(isect.mNg.x));
    assert(!isnan(isect.mNg.y));
    assert(!isnan(isect.mNg.z));
}

static varying Color
renderPixel(const uniform Scene &scene,
            const varying int    x,
            const varying int    y,
            uniform Arena *uniform arena,
            uniform unsigned int64 &shaderTicks,
            const uniform bool indirect)
{
    // scene setup
    //   camera:
    //     projection: orthographic
    //     location: (width/2, height/2)
    //   sphere
    //     radius: width / 2
    //     location: (width/2, height/2)
    //   image is width x height
    // coordinate system is right handed with -z pointing away

    assert(scene.mShadingTLState);

    Color result = Color_ctor(0.f);
    varying float xsched[MAX_RAYS_PP];
    varying float ysched[MAX_RAYS_PP];
    struct DWARNGState rng;
    seed_rng(&rng, 1);
    for (uniform int rayNo = 0; rayNo < scene.mRaysPerPixel; ++rayNo) {
        xsched[rayNo] = (varying float) x + frandom(&rng);
        ysched[rayNo] = (varying float) y + frandom(&rng);
    }

    Intersection isect;
    const State &state = asAState(isect);

    for (uniform int rayNo = 0; rayNo < scene.mRaysPerPixel; ++rayNo) {
        // "cast a ray"
        const float xd = xsched[rayNo] - scene.mWidth / 2.f;
        const float yd = ysched[rayNo] - scene.mHeight / 2.f;
        const float xd2 = xd * xd;
        const float yd2 = yd * yd;
        const float d2 = xd2 + yd2;
        const float r = scene.mWidth / 2.f;
        const float r2 = r * r;
        if (d2 < r2) { // hit
            // fill out a shading state
            const varying Vec3f origin = Vec3f_ctor(scene.mWidth / 2.f,
                                                    scene.mHeight / 2.f,
                                                   -scene.mWidth / 2.f);
            computeIsect(xsched[rayNo], ysched[rayNo], origin, r, isect,
                    scene.mRaysPerPixel, indirect);

            // shade
            uniform unsigned int64 sticks = clock();
            varying Bsdf bsdf;
            varying BsdfBuilder builder;
            BsdfBuilder_init(builder, bsdf, *scene.mShadingTLState, state);
            shadeMaterial(scene.mMat, scene.mShadingTLState, 1, &state, &builder);
            shaderTicks += (clock() - sticks);

            // check for early out
            if (!((1 << programIndex) & bsdf.mEarlyTerminationMask)) {
                // integrate
                const varying Vec3f wi = normalize(Vec3f_ctor(1.f, 1.f, 1.f)); // surface to light dir
                const varying Vec3f wo = Vec3f_ctor(0.f, 0.f, 1.f); // surface to camera dir

                varying BsdfSlice slice;
                BsdfSlice_init(&slice, state.mNg, wo, BSDF_LOBE_TYPE_ALL, true, true,
                               SHADOW_TERMINATOR_FIX_OFF);
                float pdf;
                result = result + Bsdf_eval(&bsdf, slice, wi, pdf);
                result = result + Bsdf_getSelfEmission(&bsdf);
            }
        }
    }
    return result * rcp((uniform float) scene.mRaysPerPixel);
}

/// @brief a very simple ispc test renderer
/// @return results 8-bit gamma top down rgb (ppm format)
export uniform unsigned int64
render(const uniform Scene * uniform scene,
       uniform uint8_t        * uniform results,
       const uniform bool indirect)
{
    uniform Arena * uniform arena = scene->mShadingTLState->mArena;

    uniform uint8_t *uniform bookMark = Arena_getPtr(arena);

    uniform unsigned int64 shaderTicks = 0;
    foreach_tiled(y = 0 ... scene->mHeight, x = 0 ... scene->mWidth) {

        // reset memory arena
        Arena_setPtr(arena, bookMark);

        // render the pixel
        Color c = renderPixel(*scene, x, y, arena, shaderTicks, indirect);

        // 8-bit gamma correction
        uint8_t gcR = (int) floor(pow(max(0.f, min(c.r, 1.f)), rcp(2.2f)) * 255);
        uint8_t gcG = (int) floor(pow(max(0.f, min(c.g, 1.f)), rcp(2.2f)) * 255);
        uint8_t gcB = (int) floor(pow(max(0.f, min(c.b, 1.f)), rcp(2.2f)) * 255);

        // write results
        foreach_active(i) {
            const uniform uint8_t uR = extract(gcR, i);
            const uniform uint8_t uG = extract(gcG, i);
            const uniform uint8_t uB = extract(gcB, i);
            const uniform int uX = extract(x, i);
            const uniform int uY = extract(y, i);
            uniform uint8_t * uniform rPtr = results +
                ((scene->mHeight - 1 - uY) * scene->mWidth + uX) * 3;
            rPtr[0] = uR;
            rPtr[1] = uG;
            rPtr[2] = uB;
        }
    }

    return shaderTicks;
}
