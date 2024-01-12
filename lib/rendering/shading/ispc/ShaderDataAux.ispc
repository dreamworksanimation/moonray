// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "ShaderDataAux.isph"

bool
SHADER_DATA_getRefP(const uniform int refPKey,
                    uniform ShadingTLState* uniform tls,
                    const varying State& state,
                    varying Vec3f * uniform refP)
{
    if (isProvided(state, refPKey)) {
        *refP = getVec3fAttribute(tls, state, refPKey);
        return true;
    } else {
        *refP = Vec3f_ctor(0.f, 0.f, 0.f);
        return false;
    }
}

bool
SHADER_DATA_getRefN(const uniform int refPKey,
                    const uniform int refNKey,
                    uniform ShadingTLState* uniform tls,
                    const varying State& state,
                    varying Vec3f * uniform refN)
{
    *refN = Vec3f_ctor(0.f);

    if (isProvided(state, refNKey)) {
        *refN = getVec3fAttribute(tls, state, refNKey);
    } else if (isProvided(state, refPKey)) {
        // Compute refN from refP partials
        varying Vec3f dPds = getdVec3fAttributeds(tls, state, refPKey);
        varying Vec3f dPdt = getdVec3fAttributedt(tls, state, refPKey);
        if (isZero(lengthSqr(dPds))) {
            // Invalid zero vector. Avoid divide by zero and NaNs
            // that would result from math::normalize().
            return false;
        }
        dPds = normalize(dPds);
        // For curves, since dPdt is zero, we use the camera direction
        // Wo and make it orthogonal to dPds
        if (isZero(lengthSqr(dPdt))) {
            dPdt = getWo(state);
            dPdt = cross(dPds, cross(dPds, normalize(dPdt)));

        }
        varying Vec3f normal = cross(dPds, normalize(dPdt));
        if (isZero(lengthSqr(normal))) {
            // dPds and dPdt are identical so their cross product is a zero vector.
            return false;
        }

        // TODO: for now no partial derivs when we compute ref_N from ref_P,
        *refN = normal;
    } else {
        return false;
    }

    // TODO: verify that this change is safe
    if (isEntering(state) == false) {
        // Flip reference space normals on exiting a surface
        *refN = neg(*refN);
    }

    return true;
}

