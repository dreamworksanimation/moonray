// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Xform.isph"
#include <moonray/rendering/bvh/shading/ispc/State.isph>

// Export enums and structs which will force the C++ definition 
// of SHADER_XFORM_Constants into Xform_ispc_stubs.h - which we 
// then include in the C++ code.
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(SHADING_Space);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(Xform);

varying Xform3f
computeR2O(const uniform Xform * const uniform xform,
           const varying State &state)
{
    Xform3f result;
    // need to build a varying Xform3f
    // TODO:
    //   optimize for the case where geom is equal on all lanes
    //   optimize for the case where geom is equal on consecutive lanes
    //   optimize by calling getR2O only on unique geom
    intptr_t geometryObject = getGeometryObject(state);
    foreach_active(lane) {
        const uniform intptr_t geom = extract(geometryObject, lane);
        uniform Xform3f r2o;
        typedef void (* uniform GetR2OFnPtr)(const uniform intptr_t geom,
                                             uniform Xform3f * uniform out);
        GetR2OFnPtr getR2O = (GetR2OFnPtr) xform->mGetR2OFn;
        getR2O(geom, &r2o);
        result.l.vx.x = insert(result.l.vx.x, lane, r2o.l.vx.x);
        result.l.vx.y = insert(result.l.vx.y, lane, r2o.l.vx.y);
        result.l.vx.z = insert(result.l.vx.z, lane, r2o.l.vx.z);
        result.l.vy.x = insert(result.l.vy.x, lane, r2o.l.vy.x);
        result.l.vy.y = insert(result.l.vy.y, lane, r2o.l.vy.y);
        result.l.vy.z = insert(result.l.vy.z, lane, r2o.l.vy.z);
        result.l.vz.x = insert(result.l.vz.x, lane, r2o.l.vz.x);
        result.l.vz.y = insert(result.l.vz.y, lane, r2o.l.vz.y);
        result.l.vz.z = insert(result.l.vz.z, lane, r2o.l.vz.z);
        result.p.x = insert(result.p.x, lane, r2o.p.x);
        result.p.y = insert(result.p.y, lane, r2o.p.y);
        result.p.z = insert(result.p.z, lane, r2o.p.z);
    }

    return result;
}

varying Vec3f
transformPoint(const uniform Xform * const uniform xform,
               const uniform int srcSpace,
               const uniform int dstSpace,
               const varying State &state,
               const varying Vec3f inPoint)
{
    varying Vec3f result;
    varying Vec3f rPoint;

    // Transform to render space if necessary
    switch (srcSpace) {
    case SHADING_SPACE_RENDER:
        rPoint = inPoint;
        break;
    case SHADING_SPACE_CAMERA:
        rPoint = transformPoint(xform->mR2CInv, inPoint);
        break;
    case SHADING_SPACE_WORLD:
        rPoint = transformPoint(xform->mR2WInv, inPoint);
        break;
    case SHADING_SPACE_SCREEN:
        {
            const varying Vec4f PH = Vec4f_ctor(inPoint.x, inPoint.y, inPoint.z, 1.f);
            rPoint = transformH(xform->mR2SInv, PH);
        }
        break;
    case SHADING_SPACE_OBJECT:
        {
            if (xform->mUseExternalObj) {
                rPoint = transformPoint(xform->mR2OInv, inPoint);
            } else {
                const varying Xform3f r2o = computeR2O(xform, state);
                rPoint = transformPoint(inverse(r2o), inPoint);
            }
        }
        break;
    default:
        MNRY_ASSERT(0);
    };

    // Transform from render space
    switch (dstSpace) {
    case SHADING_SPACE_RENDER:
        result = rPoint;
        break;
    case SHADING_SPACE_CAMERA:
        result = transformPoint(xform->mR2C, rPoint);
        break;
    case SHADING_SPACE_WORLD:
        result = transformPoint(xform->mR2W, rPoint);
        break;
    case SHADING_SPACE_SCREEN:
        {
            const Vec4f PH = Vec4f_ctor(rPoint.x, rPoint.y, rPoint.z, 1.f);
            result = transformH(xform->mR2S, PH);
        }
        break;
    case SHADING_SPACE_OBJECT:
        {
            if (xform->mUseExternalObj) {
                result = transformPoint(xform->mR2O, rPoint);
            } else {
                const varying Xform3f r2o = computeR2O(xform, state);
                result = transformPoint(r2o, rPoint);
            }
        }
        break;
    default:
        MNRY_ASSERT(0);
    };

    return result;
}

varying Vec3f
transformNormal(const uniform Xform * const uniform xform,
                const uniform int srcSpace,
                const uniform int dstSpace,
                const varying State &state,
                const varying Vec3f inNormal)
{
    varying Vec3f result;
    varying Vec3f rNormal;

    // Transform to render space if necessary
    // Normal transform requires inverse transpose of a point xform matrix
    // transformNormal takes care of the transpose internally
    switch (srcSpace) {
    case SHADING_SPACE_RENDER:
        rNormal = inNormal;
        break;
    case SHADING_SPACE_CAMERA:
        rNormal = transformNormal(xform->mR2C, inNormal);
        break;
    case SHADING_SPACE_WORLD:
        rNormal = transformNormal(xform->mR2W, inNormal);
        break;
    case SHADING_SPACE_SCREEN:
        rNormal = transformNormal(xform->mR2S, inNormal);
        break;
    case SHADING_SPACE_OBJECT:
        {
            if (xform->mUseExternalObj) {
                rNormal = transformNormal(xform->mR2O, inNormal);
            } else {
                const varying Xform3f r2o = computeR2O(xform, state);
                rNormal = transformNormal(r2o, inNormal);
            }
        }
        break;
    default:
        MNRY_ASSERT(0);
    };

    // Transform from render space
    switch (dstSpace) {
    case SHADING_SPACE_RENDER:
        result = rNormal;
        break;
    case SHADING_SPACE_CAMERA:
        result = transformNormal(xform->mR2CInv, rNormal);
        break;
    case SHADING_SPACE_WORLD:
        result = transformNormal(xform->mR2WInv, rNormal);
        break;
    case SHADING_SPACE_SCREEN:
        result = transformNormal(xform->mR2SInv, rNormal);
        break;
    case SHADING_SPACE_OBJECT:
        {
            if (xform->mUseExternalObj) {
                result = transformNormal(xform->mR2OInv, rNormal);
            } else {
                const varying Xform3f r2o = computeR2O(xform, state);
                result = transformNormal(inverse(r2o), rNormal);
            }
        }
        break;
    default:
        MNRY_ASSERT(0);
    };

    return result;
}

varying Vec3f
transformVector(const uniform Xform * const uniform xform,
                const uniform int srcSpace,
                const uniform int dstSpace,
                const varying State &state,
                const varying Vec3f inVector)
{
    varying Vec3f result;
    varying Vec3f rVector;

    // Transform to render space if necessary
    switch (srcSpace) {
    case SHADING_SPACE_RENDER:
        rVector = inVector;
        break;
    case SHADING_SPACE_CAMERA:
        rVector = transformVector(xform->mR2CInv, inVector);
        break;
    case SHADING_SPACE_WORLD:
        rVector = transformVector(xform->mR2WInv, inVector);
        break;
    case SHADING_SPACE_SCREEN:
        rVector = transformVector(xform->mR2SInv, inVector);
        break;
    case SHADING_SPACE_OBJECT:
        {
            if (xform->mUseExternalObj) {
                rVector = transformVector(xform->mR2OInv, inVector);
            } else {
                const varying Xform3f r2o = computeR2O(xform, state);
                rVector = transformVector(inverse(r2o), inVector);
            }
        }
        break;
    default:
        MNRY_ASSERT(0);
    };

    // Transform from render space
    switch (dstSpace) {
    case SHADING_SPACE_RENDER:
        result = rVector;
        break;
    case SHADING_SPACE_CAMERA:
        result = transformVector(xform->mR2C, rVector);
        break;
    case SHADING_SPACE_WORLD:
        result = transformVector(xform->mR2W, rVector);
        break;
    case SHADING_SPACE_SCREEN:
        result = transformVector(xform->mR2S, rVector);
        break;
    case SHADING_SPACE_OBJECT:
        {
            if (xform->mUseExternalObj) {
                result = transformVector(xform->mR2O, rVector);
            } else {
                const varying Xform3f r2o = computeR2O(xform, state);
                result = transformVector(r2o, rVector);
            }
        }
        break;
    default:
        MNRY_ASSERT(0);
    };

    return result;
}

