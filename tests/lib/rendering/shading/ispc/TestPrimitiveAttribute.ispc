// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestPrimitiveAttribute.ispc

#include <moonray/rendering/shading/ispc/Shading.isph>

export void
testGetBoolAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                     const varying State * uniform state,
                     const uniform int uniKey,
                     const uniform int * uniform varKey,
                     uniform int8_t * uniform uniResult,
                     uniform int8_t * uniform varResult)
{
    // uniform key case
    {
        const bool b = getBoolAttribute(tls, *state, uniKey);
        const int8_t i8 = b;
        foreach_active(lane) {
            uniResult[lane] = extract(i8, lane)? 1 : 0;
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const bool b = getBoolAttribute(tls, *state, k);
        const int8_t i8 = b;
        foreach_active(lane) {
            varResult[lane] = extract(i8, lane)? 1 : 0;
        }
    }
}

export void
testGetIntAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                    const varying State * uniform state,
                    const uniform int uniKey,
                    const uniform int * uniform varKey,
                    uniform int32_t * uniform uniResult,
                    uniform int32_t * uniform varResult)
{
    // uniform key case
    {
        const int i = getIntAttribute(tls, *state, uniKey);
        const int32_t i32 = i;
        foreach_active(lane) {
            uniResult[lane] = extract(i32, lane);
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const int i = getIntAttribute(tls, *state, k);
        const int32_t i32 = i;
        foreach_active(lane) {
            varResult[lane] = extract(i32, lane);
        }
    }
}

export void
testGetFloatAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                      const varying State * uniform state,
                      const uniform int uniKey,
                      const uniform int * uniform varKey,
                      uniform float * uniform uniResult,
                      uniform float * uniform varResult)
{
    // uniform key case
    {
        const float f = getFloatAttribute(tls, *state, uniKey);
        foreach_active(lane) {
            uniResult[lane] = extract(f, lane);
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const float f = getFloatAttribute(tls, *state, k);
        foreach_active(lane) {
            varResult[lane] = extract(f, lane);
        }
    }
}

export void
testGetVec2fAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                      const varying State * uniform state,
                      const uniform int uniKey,
                      const uniform int * uniform varKey,
                      uniform float * uniform uniResultX,
                      uniform float * uniform uniResultY,
                      uniform float * uniform varResultX,
                      uniform float * uniform varResultY)
{
    // uniform key case
    {
        const Vec2f v2f = getVec2fAttribute(tls, *state, uniKey);
        foreach_active(lane) {
            uniResultX[lane] = extract(v2f.x, lane);
            uniResultY[lane] = extract(v2f.y, lane);
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const Vec2f v2f = getVec2fAttribute(tls, *state, k);
        foreach_active(lane) {
            varResultX[lane] = extract(v2f.x, lane);
            varResultY[lane] = extract(v2f.y, lane);
        }
    }
}

export void
testGetVec3fAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                      const varying State * uniform state,
                      const uniform int uniKey,
                      const uniform int * uniform varKey,
                      uniform float * uniform uniResultX,
                      uniform float * uniform uniResultY,
                      uniform float * uniform uniResultZ,
                      uniform float * uniform varResultX,
                      uniform float * uniform varResultY,
                      uniform float * uniform varResultZ)
{
    // uniform key case
    {
        const Vec3f v3f = getVec3fAttribute(tls, *state, uniKey);
        foreach_active(lane) {
            uniResultX[lane] = extract(v3f.x, lane);
            uniResultY[lane] = extract(v3f.y, lane);
            uniResultZ[lane] = extract(v3f.z, lane);
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const Vec3f v3f = getVec3fAttribute(tls, *state, k);
        foreach_active(lane) {
            varResultX[lane] = extract(v3f.x, lane);
            varResultY[lane] = extract(v3f.y, lane);
            varResultZ[lane] = extract(v3f.z, lane);
        }
    }
}

export void
testGetMat4fAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                      const varying State * uniform state,
                      const uniform int uniKey,
                      const uniform int * uniform varKey,
                      uniform float * uniform uniResultVXX,
                      uniform float * uniform uniResultVXY,
                      uniform float * uniform uniResultVXZ,
                      uniform float * uniform uniResultVXW,
                      uniform float * uniform uniResultVYX,
                      uniform float * uniform uniResultVYY,
                      uniform float * uniform uniResultVYZ,
                      uniform float * uniform uniResultVYW,
                      uniform float * uniform uniResultVZX,
                      uniform float * uniform uniResultVZY,
                      uniform float * uniform uniResultVZZ,
                      uniform float * uniform uniResultVZW,
                      uniform float * uniform uniResultVWX,
                      uniform float * uniform uniResultVWY,
                      uniform float * uniform uniResultVWZ,
                      uniform float * uniform uniResultVWW,
                      uniform float * uniform varResultVXX,
                      uniform float * uniform varResultVXY,
                      uniform float * uniform varResultVXZ,
                      uniform float * uniform varResultVXW,
                      uniform float * uniform varResultVYX,
                      uniform float * uniform varResultVYY,
                      uniform float * uniform varResultVYZ,
                      uniform float * uniform varResultVYW,
                      uniform float * uniform varResultVZX,
                      uniform float * uniform varResultVZY,
                      uniform float * uniform varResultVZZ,
                      uniform float * uniform varResultVZW,
                      uniform float * uniform varResultVWX,
                      uniform float * uniform varResultVWY,
                      uniform float * uniform varResultVWZ,
                      uniform float * uniform varResultVWW)
{
    // uniform key case
    {
        const Mat4f m4f = getMat4fAttribute(tls, *state, uniKey);
        foreach_active(lane) {
            uniResultVXX[lane] = extract(m4f.vx.x, lane);
            uniResultVXY[lane] = extract(m4f.vx.y, lane);
            uniResultVXZ[lane] = extract(m4f.vx.z, lane);
            uniResultVXW[lane] = extract(m4f.vx.w, lane);
            uniResultVYX[lane] = extract(m4f.vy.x, lane);
            uniResultVYY[lane] = extract(m4f.vy.y, lane);
            uniResultVYZ[lane] = extract(m4f.vy.z, lane);
            uniResultVYW[lane] = extract(m4f.vy.w, lane);
            uniResultVZX[lane] = extract(m4f.vz.x, lane);
            uniResultVZY[lane] = extract(m4f.vz.y, lane);
            uniResultVZZ[lane] = extract(m4f.vz.z, lane);
            uniResultVZW[lane] = extract(m4f.vz.w, lane);
            uniResultVWX[lane] = extract(m4f.vw.x, lane);
            uniResultVWY[lane] = extract(m4f.vw.y, lane);
            uniResultVWZ[lane] = extract(m4f.vw.z, lane);
            uniResultVWW[lane] = extract(m4f.vw.w, lane);
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const Mat4f m4f = getMat4fAttribute(tls, *state, k);
        foreach_active(lane) {
            varResultVXX[lane] = extract(m4f.vx.x, lane);
            varResultVXY[lane] = extract(m4f.vx.y, lane);
            varResultVXZ[lane] = extract(m4f.vx.z, lane);
            varResultVXW[lane] = extract(m4f.vx.w, lane);
            varResultVYX[lane] = extract(m4f.vy.x, lane);
            varResultVYY[lane] = extract(m4f.vy.y, lane);
            varResultVYZ[lane] = extract(m4f.vy.z, lane);
            varResultVYW[lane] = extract(m4f.vy.w, lane);
            varResultVZX[lane] = extract(m4f.vz.x, lane);
            varResultVZY[lane] = extract(m4f.vz.y, lane);
            varResultVZZ[lane] = extract(m4f.vz.z, lane);
            varResultVZW[lane] = extract(m4f.vz.w, lane);
            varResultVWX[lane] = extract(m4f.vw.x, lane);
            varResultVWY[lane] = extract(m4f.vw.y, lane);
            varResultVWZ[lane] = extract(m4f.vw.z, lane);
            varResultVWW[lane] = extract(m4f.vw.w, lane);
        }
    }
}

export void
testGetColorAttribute(uniform ShadingTLState * uniform tls,
#pragma ignore warning(all)
                      const varying State * uniform state,
                      const uniform int uniKey,
                      const uniform int * uniform varKey,
                      uniform float * uniform uniResultR,
                      uniform float * uniform uniResultG,
                      uniform float * uniform uniResultB,
                      uniform float * uniform varResultR,
                      uniform float * uniform varResultG,
                      uniform float * uniform varResultB)
{
    // uniform key case
    {
        const Color c = getColorAttribute(tls, *state, uniKey);
        foreach_active(lane) {
            uniResultR[lane] = extract(c.r, lane);
            uniResultG[lane] = extract(c.g, lane);
            uniResultB[lane] = extract(c.b, lane);
        }
    }

    // varying key case
    {
        const varying int k = *((const varying int * uniform)(varKey));
        const Color c = getColorAttribute(tls, *state, k);
        foreach_active(lane) {
            varResultR[lane] = extract(c.r, lane);
            varResultG[lane] = extract(c.g, lane);
            varResultB[lane] = extract(c.b, lane);
        }
    }
}


