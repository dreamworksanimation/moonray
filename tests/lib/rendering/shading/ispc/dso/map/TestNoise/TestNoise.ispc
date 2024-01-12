// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestNoise.ispc

#include <moonray/rendering/shading/ispc/Shading.isph>
#include "attributes.isph"

#include <scene_rdl2/render/util/Random.isph>

#define TABSIZE 256

struct NoiseTables
{
    uniform int mPerm[TABSIZE];
    uniform Vec3f mGrad[TABSIZE];
};

struct TestNoise
{
    NoiseTables mNoiseTables;
};

export const uniform TestNoise * uniform
TestNoise_get(const uniform Map * uniform map)
{
    return MAP_GET_ISPC_CPTR(TestNoise, map);
}

export void
TestNoise_update(uniform Map * uniform map)
{
    uniform TestNoise * uniform testNoise = MAP_GET_ISPC_PTR(TestNoise, map);
    uniform struct DWARNGState rng;
    seed_rng(&rng, getAttrSeed(map));
    for (uniform int i = 0; i < TABSIZE; ++i) {
        uniform Vec3f v;
        do {
            v.x = frandom(&rng) * 2 - 1;
            v.y = frandom(&rng) * 2 - 1;
            v.z = frandom(&rng) * 2 - 1;
        } while (dot(v, v) > 1);

        v = normalize(v);
        testNoise->mNoiseTables.mGrad[i] = v;
        testNoise->mNoiseTables.mPerm[i] = i;
    }
    for (uniform int i = 0; i < TABSIZE - 1; ++i) {
        uniform int j = i + (int)(frandom(&rng) * (TABSIZE - i));
        uniform int k = testNoise->mNoiseTables.mPerm[i];
        testNoise->mNoiseTables.mPerm[i] = testNoise->mNoiseTables.mPerm[j];
        testNoise->mNoiseTables.mPerm[j] = k;
    }
}

inline varying float
scurve(varying float t)
{
    return t * t * (3 - 2 * t);
}

inline varying float
lcurve(varying float t, varying float a, varying float b)
{
    // different signature than Math lib's lerp
    return a + t * (b - a);
}

inline varying int
perm(const uniform TestNoise * uniform n,
     varying int i)
{
    const uniform int TABMASK = TABSIZE - 1;
    return n->mNoiseTables.mPerm[(i) & TABMASK];
}

inline varying int
index3D(const uniform TestNoise * uniform n,
        varying int ix, varying int iy, varying int iz)
{
    return perm(n, ix + perm(n, iy + perm(n, iz)));
}

inline varying float
glattice3D(const uniform TestNoise * uniform n,
           varying int ix, varying int iy, varying int iz,
           varying float fx, varying float fy, varying float fz)
{
    varying int i = index3D(n, ix, iy, iz);
    return n->mNoiseTables.mGrad[i].x * fx +
        n->mNoiseTables.mGrad[i].y * fy +
        n->mNoiseTables.mGrad[i].z * fz;
}

static varying float
perlin3D(const uniform TestNoise * uniform n,
         const varying Vec3f &pos)
{
    const varying int ix = (varying int) floor(pos.x);
    const varying int iy = (varying int) floor(pos.y);
    const varying int iz = (varying int) floor(pos.z);

    const varying float fx0 = pos.x - (varying float) ix;
    const varying float fy0 = pos.y - (varying float) iy;
    const varying float fz0 = pos.z - (varying float) iz;

    const varying float fx1 = fx0 - 1.f;
    const varying float fy1 = fy0 - 1.f;
    const varying float fz1 = fz0 - 1.f;

    const varying float wx = scurve(fx0);
    const varying float wy = scurve(fy0);
    const varying float wz = scurve(fz0);

    varying float vx0 = glattice3D(n, ix, iy, iz, fx0, fy0, fz0);
    varying float vx1 = glattice3D(n, ix + 1, iy, iz, fx1, fy0, fz0);
    varying float vy0 = lcurve(wx, vx0, vx1);

    vx0 = glattice3D(n, ix, iy + 1, iz, fx0, fy1, fz0);
    vx1 = glattice3D(n, ix + 1, iy + 1, iz, fx1, fy1, fz0);
    varying float vy1 = lcurve(wx, vx0, vx1);

    varying float vz0 = lcurve(wy, vy0, vy1);

    vx0 = glattice3D(n, ix, iy, iz + 1, fx0, fy0, fz1);
    vx1 = glattice3D(n, ix + 1, iy, iz + 1, fx1, fy0, fz1);
    vy0 = lcurve(wx, vx0, vx1);

    vx0 = glattice3D(n, ix, iy + 1, iz + 1, fx0, fy1, fz1);
    vx1 = glattice3D(n, ix + 1, iy + 1, iz + 1, fx1, fy1, fz1);
    vy1 = lcurve(wx, vx0, vx1);

    varying float vz1 = lcurve(wy, vy0, vy1);

    return lcurve(wz, vz0, vz1);
}

#define NOISE_SAMPLES 5

static varying Color
sample(const uniform Map *            uniform map,
             uniform ShadingTLState * uniform tls,
       const varying State &                  state)
{
    const uniform TestNoise * uniform testNoise =
        MAP_GET_ISPC_CPTR(TestNoise, map);

    varying Vec3f pos = evalAttrFrequencyMultiplier(map, tls, state) * state.mP;
    varying float levels = evalAttrMaxLevel(map, tls, state);
    varying int numLevels = (varying int) ceil(levels);
    varying float scale = 1.f;
    varying float noise = 0.f;
    for (varying int i = 0; i < numLevels; ++i) {
        if (i + 1 == numLevels) {
            scale = scale * (1.f - ((varying float) numLevels - levels));
        }
        noise = noise + perlin3D(testNoise, pos) * scale;
        scale = scale / 2.f;
        pos = 2.f * pos;
    }

    return Color_ctor(noise);
}

DEFINE_MAP_SHADER(TestNoise, sample)
               
