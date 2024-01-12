// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file OpenVdbMap_v2.ispc

#include "attributes.isph"

#include <moonray/rendering/shading/ispc/MapApi.isph>
#include <moonray/rendering/shading/ispc/OpenVdbUtil.isph>

struct OpenVdbMap_v2
{
    uniform const Xform * uniform mXform;
    uniform int mRefPKey;
    uniform intptr_t mSampleFn;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(OpenVdbMap_v2);

static Color
sample(const uniform Map *uniform map,
       uniform ShadingTLState * uniform tls,
       const varying State &state)
{
    const uniform OpenVdbMap_v2* uniform me = MAP_GET_ISPC_PTR(OpenVdbMap_v2, map);

    varying Vec3f P = getP(state);
    varying intptr_t geom = getGeometryObject(state);

    if (getAttrTextureCoordSource(map) == OPENVDBMAP_COORD_REFP) {
        varying Vec3f refP;
        SHADER_DATA_getRefP(me->mRefPKey, tls, state, &refP);
        P = refP;
    } else if (getAttrTextureCoordSource(map) == OPENVDBMAP_COORD_INPUT) {
        P = evalAttrInputTextureCoordinate(map, tls, state);
    }

    varying Color result;

    foreach_active(lane) {
        const uniform float Px = extract(P.x, lane);
        const uniform float Py = extract(P.y, lane);
        const uniform float Pz = extract(P.z, lane);

        typedef void (* uniform SampleVdbFn_t)(const uniform Map * uniform map,
                                               uniform ShadingTLState * uniform tls,
                                               const uniform float Px,
                                               const uniform float Py,
                                               const uniform float Pz,
                                               uniform intptr_t geom,
                                               uniform Color * uniform out);
        SampleVdbFn_t sampleFn = (SampleVdbFn_t) me->mSampleFn;

        uniform intptr_t geomLane = extract(geom, lane);
        uniform Color c;
        sampleFn(map, tls, Px, Py, Pz, geomLane, &c);
        result.r = insert(result.r, lane, c.r);
        result.g = insert(result.g, lane, c.g);
        result.b = insert(result.b, lane, c.b);
    }

    return result;
}
DEFINE_MAP_SHADER(OpenVdbMap_v2, sample)


