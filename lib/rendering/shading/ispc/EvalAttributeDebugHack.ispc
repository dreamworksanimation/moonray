// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "EvalAttribute.isph"

// There is a bug in the optimization pass with ISPC 1.9.2 which causes certain
// generated code to crash in opt builds. The workaround is to move the offending
// code out into a separate file and build it in debug.

varying bool derefAndCastToVaryingBool(const uniform uint8_t *varying ptr)
{
    return *((const uniform bool *varying)(ptr));
}

varying int derefAndCastToVaryingInt(const uniform uint8_t *varying ptr)
{
    return *((const uniform int *varying)(ptr));
}

varying float derefAndCastToVaryingFloat(const uniform uint8_t *varying ptr)
{
    return *((const uniform float *varying)(ptr));
}

varying Color derefAndCastToVaryingColor(const uniform uint8_t *varying ptr)
{
    return *((const uniform Color *varying)(ptr));
}

varying Vec2f derefAndCastToVaryingVec2f(const uniform uint8_t *varying ptr)
{
    return *((const uniform Vec2f *varying)(ptr));
}

varying Vec3f derefAndCastToVaryingVec3f(const uniform uint8_t *varying ptr)
{
    return *((const uniform Vec3f *varying)(ptr));
}

varying Mat4f derefAndCastToVaryingMat4f(const uniform uint8_t *varying ptr)
{
    return *((const uniform Mat4f *varying)(ptr));
}

varying bool
isProvided(const varying State &me, const varying int key)
{
    return isProvided(asAnIntersection(me), key);
}

varying bool
isdsProvided(const varying State &me, const varying int key)
{
    return isdsProvided(asAnIntersection(me), key);
}

varying bool
isdtProvided(const varying State &me, const varying int key)
{
    return isdtProvided(asAnIntersection(me), key);
}

