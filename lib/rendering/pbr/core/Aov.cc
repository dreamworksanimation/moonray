// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Aov.cc

#include <moonray/rendering/pbr/core/Scene.h>

#include "Aov.h"
#include "PbrTLState.h"

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/camera/ProjectiveCamera.h>
#include <moonray/rendering/pbr/core/Aov_ispc_stubs.h>
#include <moonray/rendering/pbr/integrator/BsdfSampler.h>
#include <moonray/rendering/shading/AovLabels.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/bsdf/Bsdfv.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>
#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>

#include <scene_rdl2/scene/rdl2/Map.h>
#include <scene_rdl2/scene/rdl2/UserData.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Strings.h>

#include <algorithm>
#include <cctype>
#include <sstream>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;
using namespace shading;

std::string
aovSchemaIdToString(const AovSchemaId &aovSchemaId)
{
    switch (aovSchemaId) {
    case AovSchemaId::AOV_SCHEMA_ID_UNKNOWN : return "AOV_SCHEMA_ID_UNKNOWN";
    case AovSchemaId::AOV_SCHEMA_ID_BEAUTY : return "AOV_SCHEMA_ID_BEAUTY";
    case AovSchemaId::AOV_SCHEMA_ID_ALPHA : return "AOV_SCHEMA_ID_ALPHA";

    // state position (vec3f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_P : return "AOV_SCHEMA_ID_STATE_P";

    // state geometric normal (vec3f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_NG : return "AOV_SCHEMA_ID_STATE_NG";

    // state shading normal (vec3f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_N : return "AOV_SCHEMA_ID_STATE_N";

    // state St coord (vec2f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_ST : return "AOV_SCHEMA_ID_STATE_ST";

    // state dpds (vec3f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DPDS : return "AOV_SCHEMA_ID_STATE_DPDS";

    // state dpdt (vec3f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DPDT : return "AOV_SCHEMA_ID_STATE_DPDT";

    // state ds, dt (floats)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DSDX : return "AOV_SCHEMA_ID_STATE_DSDX";
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DSDY : return "AOV_SCHEMA_ID_STATE_DSDY";
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DTDX : return "AOV_SCHEMA_ID_STATE_DTDX";
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DTDY : return "AOV_SCHEMA_ID_STATE_DTDY";

    // wireframe output is handled like a state attribute aov
    case AovSchemaId::AOV_SCHEMA_ID_WIREFRAME : return "AOV_SCHEMA_ID_WIREFRAME";

    // state world position (vec3f)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_WP : return "AOV_SCHEMA_ID_STATE_WP";

    // state depth (-1.f * P.z)
    case AovSchemaId::AOV_SCHEMA_ID_STATE_DEPTH : return "AOV_SCHEMA_ID_STATE_DEPTH";

    // motion vectors are handled like a state attribute aov
    case AovSchemaId::AOV_SCHEMA_ID_STATE_MOTION : return "AOV_SCHEMA_ID_STATE_MOTION";

    default :
        // Beyond here, everything should be a range type.
        // We have to compare aovSchemaId with enum definitions by reverse order
        // in order to find the id range to which given aovSchemeId belong.
        if (aovSchemaId >= AOV_SCHEMA_ID_LIGHT_AOV) {
            return "AOV_SCHEMA_ID_LIGHT_AOV (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_VISIBILITY_AOV) {
            return "AOV_SCHEMA_ID_VISIBILITY_AOV (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_MATERIAL_AOV_RGB) {
            return "AOV_SCHEMA_ID_MATERIAL_AOV_RGB (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F) {
            return "AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F) {
            return "AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT) {
            return "AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_PRIM_ATTR_RGB) {
            return "AOV_SCHEMA_ID_PRIM_ATTR_RGB (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_PRIM_ATTR_VEC3F) {
            return "AOV_SCHEMA_ID_PRIM_ATTR_VEC3F (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_PRIM_ATTR_VEC2F) {
            return "AOV_SCHEMA_ID_PRIM_ATTR_VEC2F (range)";
        } else if (aovSchemaId >= AOV_SCHEMA_ID_PRIM_ATTR_FLOAT) {
            return "AOV_SCHEMA_ID_PRIM_ATTR_FLOAT (range)";
        }
        break;
    }
    return "?";
}

std::string
aovFilterToString(const AovFilter &aovFilter)
{
    switch (aovFilter) {
    case AovFilter::AOV_FILTER_AVG : return "AOV_FILTER_AVG";
    case AovFilter::AOV_FILTER_SUM : return "AOV_FILTER_SUM";
    case AovFilter::AOV_FILTER_MIN : return "AOV_FILTER_MIN";
    case AovFilter::AOV_FILTER_MAX : return "AOV_FILTER_MAX";
    case AovFilter::AOV_FILTER_FORCE_CONSISTENT_SAMPLING : return "AOV_FILTER_FORCE_CONSISTENT_SAMPLING";
    case AovFilter::AOV_FILTER_CLOSEST : return "AOV_FILTER_CLOSEST";
    default : break;
    }
    return "?";
}

std::string
aovTypeToString(const AovType &aovType)
{
    switch (aovType) {
    case AovType::AOV_TYPE_UNKNOWN : return "AOV_TYPE_UNKNOWN";
    case AovType::AOV_TYPE_BEAUTY : return "AOV_TYPE_BEAUTY";
    case AovType::AOV_TYPE_ALPHA : return "AOV_TYPE_ALPHA";
    case AovType::AOV_TYPE_STATE_VAR : return "AOV_TYPE_STATE_VAR";
    case AovType::AOV_TYPE_PRIM_ATTR : return "AOV_TYPE_PRIM_ATTR";
    case AovType::AOV_TYPE_MATERIAL_AOV : return "AOV_TYPE_MATERIAL_AOV";
    case AovType::AOV_TYPE_VISIBILITY_AOV : return "AOV_TYPE_VISIBILITY_AOV";
    case AovType::AOV_TYPE_LIGHT_AOV : return "AOV_TYPE_LIGHT_AOV";
    default : break;
    }
    return "?";
}

std::string
aovStorageTypeToString(const AovStorageType &aovStorageType)
{
    switch (aovStorageType) {
    case AovStorageType::UNSPECIFIED : return "UNSPECIFIED";
    case AovStorageType::FLOAT : return "FLOAT";
    case AovStorageType::VEC2 : return "VEC2";
    case AovStorageType::VEC3 : return "VEC3";
    case AovStorageType::VEC4 : return "VEC4";
    case AovStorageType::RGB : return "RGB";
    case AovStorageType::RGB4 : return "RGB4";
    case AovStorageType::VISIBILITY : return "VISIBILITY";
    }
    return "?";
}

AovType
aovType(int aovSchemaId)
{
    AovType result = AOV_TYPE_UNKNOWN;

    if      (aovSchemaId == AOV_SCHEMA_ID_BEAUTY)     result = AOV_TYPE_BEAUTY;
    else if (aovSchemaId == AOV_SCHEMA_ID_ALPHA)      result = AOV_TYPE_ALPHA;
    else if (aovSchemaId >= AOV_START_LIGHT_AOV)      result = AOV_TYPE_LIGHT_AOV;
    else if (aovSchemaId >= AOV_START_VISIBILITY_AOV) result = AOV_TYPE_VISIBILITY_AOV;
    else if (aovSchemaId >= AOV_START_MATERIAL_AOV)   result = AOV_TYPE_MATERIAL_AOV;
    else if (aovSchemaId >= AOV_START_PRIM_ATTR)      result = AOV_TYPE_PRIM_ATTR;
    else if (aovSchemaId >= AOV_START_STATE_VAR)      result = AOV_TYPE_STATE_VAR;

    return result;
}

unsigned
aovNumChannels(int aovSchemaId)
{
    // the vast majority of aovs are RGB triples.  we'll just
    // explicitly check for those that are not.
    unsigned result = 3;

    switch (aovType(aovSchemaId)) {
    case AOV_TYPE_STATE_VAR:
        switch (aovSchemaId) {
        case AOV_SCHEMA_ID_STATE_ST:
        case AOV_SCHEMA_ID_STATE_MOTION:
            result = 2;
            break;
        case AOV_SCHEMA_ID_STATE_DSDX:
        case AOV_SCHEMA_ID_STATE_DSDY:
        case AOV_SCHEMA_ID_STATE_DTDX:
        case AOV_SCHEMA_ID_STATE_DTDY:
        case AOV_SCHEMA_ID_WIREFRAME:
        case AOV_SCHEMA_ID_STATE_DEPTH:
            result = 1;
            break;
        }
        break;
    case AOV_TYPE_BEAUTY:
        result = 3;
        break;
    case AOV_TYPE_ALPHA:
        result = 1;
        break;
    case AOV_TYPE_PRIM_ATTR:
    case AOV_TYPE_MATERIAL_AOV:
        switch (aovToRangeTypeSchemaId(aovSchemaId)) {
        case AOV_SCHEMA_ID_PRIM_ATTR_FLOAT:
        case AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT:
            result = 1;
            break;
        case AOV_SCHEMA_ID_PRIM_ATTR_VEC2F:
        case AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F:
            result = 2;
            break;
        }
        break;
    case AOV_TYPE_VISIBILITY_AOV:
        // channel 1 is the number of light samples that hit a light
        // channel 2 is the total number of light samples
        result = 2;
        break;
    case AOV_TYPE_LIGHT_AOV:
        // all are 3 channel RGB
        break;
    default:
        std::cerr << "unknown aov type " << aovSchemaId << '\n';
        MNRY_ASSERT(0 && "unknown AOV type");
    }

    return result;
}

AovSchemaId
aovToRangeTypeSchemaId(int aovSchemaId)
{
    MNRY_ASSERT(aovSchemaId > AOV_MAX_RANGE_TYPE);
    return static_cast<AovSchemaId>(aovSchemaId - aovToRangeTypeOffset(aovSchemaId));
}

int
aovToRangeTypeOffset(int aovSchemaId)
{
    MNRY_ASSERT(aovSchemaId > AOV_MAX_RANGE_TYPE);
    return aovSchemaId % AOV_MAX_RANGE_TYPE;
}

HUD_VALIDATOR(AovSchema);

std::string
AovSchema::EntryData::toString() const
{
    std::ostringstream ostr;
    ostr << "EntryData {\n"
         << "  schemaId:" << schemaID << ' ' << aovSchemaIdToString((AovSchemaId)schemaID) << '\n'
         << "  filter:" << aovFilterToString(filter) << '\n'
         << "  storageType:" << aovStorageTypeToString(storageType) << '\n'
         << "}";
    return ostr.str();
}

std::string    
AovSchema::Entry::toString() const
{
    std::ostringstream ostr;
    ostr << "Entry {\n"
         << "  mId:" << mId << ' ' << aovSchemaIdToString((AovSchemaId)mId) << '\n'
         << "  mType:" << aovTypeToString(mType) << '\n'
         << "  mNumChannels:" << mNumChannels << '\n'
         << "  mFilter:" << aovFilterToString(mFilter) << '\n'
         << "  mStorageType:" << aovStorageTypeToString(mStorageType) << '\n'
         << "}";
    return ostr.str();
}

AovSchema::AovSchema(): mAllLpePrefixFlags(sLpePrefixNone), mNumChannels(0) {}

void
AovSchema::init(const std::vector<AovSchema::EntryData> &data)
{
    mNumChannels = 0;
    mHasAovFilter = false;
    mHasClosestFilter = false;
    mEntries.clear();

    for (size_t aov = 0; aov < data.size(); ++aov) {
        mEntries.emplace_back(data[aov]);
        if (data[aov].filter > AOV_FILTER_SUM) mHasAovFilter = true;
        if (data[aov].filter == AOV_FILTER_CLOSEST) mHasClosestFilter = true;
        mNumChannels += mEntries.back().numChannels();
        mAllLpePrefixFlags |= mEntries.back().lpePrefixFlags();
    }
}

void
AovSchema::initFloatArray(float *aovs) const
{
    if (!mHasAovFilter) {
        // no special aov filter, we can just use memset
        memset(aovs, 0, mNumChannels * sizeof(float));
    } else {
        // need to properly set the initial value
        float *fptr = aovs;
        for (unsigned int aov = 0; aov < mEntries.size(); ++aov) {
            const Entry &entry = mEntries[aov];

            // Special case for closest filtering
            // Force all floats to be zero as defaultValue returns inf for closest filtering
            // For a sample, when we compute radiance we add the value to the existing AOV variable on the stack.
            // That wouldn't work if the initial value of the AOV variable is inf itself.
            // Rather then checking for inf every time we accumulate AOV's, 
            // it's simpler to initialize the AOV variable on the stack to 0.
            // We perform a depth test after evaluating a sample to see if it should
            // replace another sample if it's closer.
            // Use default value for all other filter types
            float initVal = entry.filter() == AOV_FILTER_CLOSEST ? 0.0f : entry.defaultValue(); 
            for (unsigned int j = 0; j < entry.numChannels(); ++j) {
                *fptr++ = initVal;
            }
        }
    }
}


//----------------------------------------------------------------------------
// Bundling
//----------------------------------------------------------------------------

static void
addToBundledQueue(pbr::TLState *pbrTls,
                  const AovSchema &aovSchema,
                  float depth,
                  const uint32_t aovTypeMask,
                  const float *aovValues,
                  bool sparseAovValues,
                  const uint32_t lane,
                  const uint32_t vlen,
                  uint32_t pixel,
                  uint32_t deepDataHandle)
{
    // sparseAovValues means that the aovValues array is laid out
    // according to the aov schema and will have space for unused data
    // not matching the aovTypeMask.  If the values are not sparse, then
    // the values are compacted based on the aovTypeMask.  Material Aovs
    // are not sparse, while state and primitive attribute aovs are sparse
    unsigned int aovIdx = 0;
    unsigned int aov = 0;
    BundledAov bundledAov(pixel, pbr::nullHandle);

    // function called when the bundledAov fills up and is added to the queue
    auto queueBundledAov = [&bundledAov, pbrTls, deepDataHandle]() {
        bundledAov.mDeepDataHandle = pbrTls->acquireDeepData(deepDataHandle);
        pbrTls->addAovQueueEntries(1, &bundledAov);
    };
    // function called to initialize bundledAov for more data
    auto initBundledAov = [&aov, &bundledAov, pixel]() {
        bundledAov.init(pixel, pbr::nullHandle);
        aov = 0;
    };

    const float *result = aovValues;
    for (const auto &entry: aovSchema) {
        const unsigned int nchans = entry.numChannels();
        if (entry.type() & aovTypeMask) {
            if (entry.filter() == AOV_FILTER_CLOSEST &&
                aov + nchans + 1 >= BundledAov::MAX_AOV) {
                // A closest filtered value cannot be split
                // across bundles.  So flush if needed
                MNRY_ASSERT(aov != 0);
                queueBundledAov();
                initBundledAov();
            }

            for (unsigned int c = 0; c < nchans; ++c) {
                const float value = result[lane + c * vlen];
                // only queue non-zero results when averaging or summing
                // don't queue +/-inf when using min or max filters, the
                // frame buffer is already cleared with the appropriate inf
                if ((value != 0.f && entry.filter() <= AOV_FILTER_SUM) ||
                    (scene_rdl2::math::isfinite(value) && entry.filter() > AOV_FILTER_SUM)) {
                    bundledAov.setAov(aov++, value, aovIdx + c);
                }

                // flush if needed
                if (aov == BundledAov::MAX_AOV) {
                    MNRY_ASSERT(entry.filter() != AOV_FILTER_CLOSEST);
                    queueBundledAov();
                    initBundledAov();
                }
            }

            // Add the depth if indicated
            if (entry.stateAovId() == AOV_SCHEMA_ID_STATE_DEPTH ||
                entry.filter() == AOV_FILTER_CLOSEST) {
                bundledAov.setAov(aov++, depth);
                // flush if needed
                if (aov == BundledAov::MAX_AOV) {
                    queueBundledAov();
                    initBundledAov();
                }
            }

            if (!sparseAovValues) {
                // advance result now
                result += nchans * vlen;
            }
        }
        // advance aovIdx
        aovIdx += nchans;
        if (sparseAovValues) {
            // advance result now
            result += nchans * vlen;
        }
    }

    // queue any remainders
    if (aov > 0) {
        queueBundledAov();
    }
}

// Note that we pass an array of per-entry lane masks for the material aovs.
// This is done because processing is split across 2 functions, one which may invalidate some
// of the vector lanes, and this one, which needs a knowledge of which lanes were invalidated.
static void
addToBundledQueue(pbr::TLState *pbrTls,
                  const AovSchema &aovSchema,
                  const float depth[],
                  const float *aovValues,
                  const uint32_t lane,
                  const uint32_t materialAovLanemasks[],
                  const uint32_t pixel[],
                  const uint32_t deepDataHandle[])
{
    uint32_t handle = deepDataHandle[lane];
    uint32_t pix    = pixel[lane];

    unsigned int aovIdx = 0;
    unsigned int aov = 0;
    BundledAov bundledAov(pix, pbr::nullHandle);

    // function called when the bundledAov fills up and is added to the queue
    auto queueBundledAov = [&bundledAov, pbrTls, handle]() {
        bundledAov.mDeepDataHandle = pbrTls->acquireDeepData(handle);
        pbrTls->addAovQueueEntries(1, &bundledAov);
    };
    // function called to initialize bundledAov for more data
    auto initBundledAov = [&aov, &bundledAov, pix]() {
        bundledAov.init(pix, pbr::nullHandle);
        aov = 0;
    };

    const float *result = aovValues;
    int i = 0;
    for (const auto &entry: aovSchema) {
        const unsigned int nchans = entry.numChannels();

        if (entry.type() & AOV_TYPE_MATERIAL_AOV) {
            // Retrieve the lane mask which MaterialAovs::computeVector() modified locally.
            // This is a terrible hack, but there doesn't seem to be any other way given the way the processing
            // is split across multiple functions each of which needs access to the lane mask.
            const uint32_t localLanemask = materialAovLanemasks[i++];
            if (!(localLanemask & (1 << lane))) {
                // The continue statement would skip over the updates to result and aovIdx, so we increment them here
                result += nchans * VLEN;
                aovIdx += nchans;
                continue;
            }

            if (entry.filter() == AOV_FILTER_CLOSEST &&
                aov + nchans + 1 >= BundledAov::MAX_AOV) {
                // A closest filtered value cannot be split
                // across bundles.  So flush if needed
                MNRY_ASSERT(aov != 0);
                queueBundledAov();
                initBundledAov();
            }

            for (unsigned int c = 0; c < nchans; ++c) {
                const float value = result[lane + c * VLEN];
                // When averaging or summing, only queue non-zero values (otherwise it's wasted effort).
                // When using min, max or closest filters, only queue finite values (because the frame
                // buffer is already cleared with the appropriate +/-inf).
                if ((value != 0.f && entry.filter() <= AOV_FILTER_SUM) ||
                    (scene_rdl2::math::isfinite(value) && entry.filter() > AOV_FILTER_SUM)) {
                    bundledAov.setAov(aov++, value, aovIdx + c);
                }

                // flush if needed
                if (aov == BundledAov::MAX_AOV) {
                    MNRY_ASSERT(entry.filter() != AOV_FILTER_CLOSEST);
                    queueBundledAov();
                    initBundledAov();
                }
            }

            // Add the depth if indicated
            if (entry.stateAovId() == AOV_SCHEMA_ID_STATE_DEPTH) {
                bundledAov.setAov(aov++, depth[lane]);
                // flush if needed
                if (aov == BundledAov::MAX_AOV) {
                    queueBundledAov();
                    initBundledAov();
                }
            }

            // Not sparse, so advance result now
            result += nchans * VLEN;
        }

        // advance aovIdx
        aovIdx += nchans;
    }

    // queue any remainders
    if (aov > 0) {
        queueBundledAov();
    }
}

void
aovAddToBundledQueue(pbr::TLState *pbrTls,
                     const AovSchema &aovSchema,
                     const shading::Intersection &isect,
                     const mcrt_common::RayDifferential &ray,
                     const uint32_t aovTypeMask,
                     const float *aovValues,
                     uint32_t pixel,
                     uint32_t deepDataHandle)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    // Do we need to compute a depth value?
    float depth = scene_rdl2::math::inf;
    const FrameState &fs = *pbrTls->mFs;
    const Scene &scene = *fs.mScene;
    for (const auto &entry: aovSchema) {
        if (entry.type() & aovTypeMask) {
            if (entry.stateAovId() == AOV_SCHEMA_ID_STATE_DEPTH ||
                entry.filter() == AOV_FILTER_CLOSEST) {
                // compute it
                depth = scene.getCamera()->computeZDistance(isect.getP(),
                                                            ray.getOrigin(),
                                                            ray.getTime());
                break;
            }
        }
    }

    addToBundledQueue(pbrTls, aovSchema, depth, aovTypeMask, aovValues, true, 0, 1,
                      pixel, deepDataHandle);
}

void
aovAddToBundledQueueVolumeOnly(pbr::TLState *pbrTls,
                               const AovSchema &aovSchema,
                               const mcrt_common::RayDifferential &ray,
                               const uint32_t aovTypeMask,
                               const float *aovValues,
                               uint32_t pixel,
                               uint32_t deepDataHandle)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    float depth = scene_rdl2::math::inf; // hard surface depth, not applicable here

    addToBundledQueue(pbrTls, aovSchema, depth, aovTypeMask, aovValues, true, 0, 1,
                      pixel, deepDataHandle);
}

// ---------------------------------------------------------------------------
// State aovs
// ---------------------------------------------------------------------------

// wireframe computation
// although not really an aov, wireframe is handled and computed with the state
// aovs
static float
sampleWireframe(const shading::Intersection &isect)
{
    // return 1 if isect point is on a polygon edge, 0 if not
    float result = 0.f;

    // what type of data is in our poly vertex array?
    const int ptype = isect.getAttribute(StandardAttributes::sPolyVertexType);

    if (ptype == StandardAttributes::POLYVERTEX_TYPE_UNKNOWN) {
        // well if you don't know, then neither do we.
        return result;
    }

    if (ptype == StandardAttributes::POLYVERTEX_TYPE_CUBIC_SPLINE ||
        ptype == StandardAttributes::POLYVERTEX_TYPE_LINE) {
        // curves are basically all edge.  the main issue is curve thickness
        // that will not in general match the desired raster pixel width.
        // for now, a hit on a curve is a hit on the edge.
        result = 1.f;
        return result;
    }

    MNRY_ASSERT(ptype == StandardAttributes::POLYVERTEX_TYPE_POLYGON);

    const int numPolyVertices = isect.getAttribute(StandardAttributes::sNumPolyVertices);
    if (numPolyVertices) {
        // estimate pixel width in render space at point P
        //
        //        pwX
        //    +--------+ P + dpdx
        //    |       /
        //    |      /
        //    |     /
        //    |    /
        //    |   /
        //    |  /
        //    | /
        //    +
        //    P
        const Vec3f dpdx = isect.getdPds() * isect.getdSdx() + isect.getdPdt() * isect.getdTdx();
        const Vec3f dpdy = isect.getdPds() * isect.getdSdy() + isect.getdPdt() * isect.getdTdy();
        const Vec3f I = scene_rdl2::math::normalize(isect.getP());
        const float pwXSq = scene_rdl2::math::lengthSqr(dpdx - scene_rdl2::math::dot(I, dpdx) * I);
        const float pwYSq = scene_rdl2::math::lengthSqr(dpdy - scene_rdl2::math::dot(I, dpdy) * I);

        // we want to consider just half the estimated pixel width for this
        // polygon, neighboring polygons will account for the other half
        //    (pwX / 2) * (pwX / 2) = pwXSq * .25
        //    (pwY / 2) * (PwY / 2) = pwYSq * .25
        // and then we want to average these values
        // to get our final pwSq estimate that we'll use for comparisons
        //    (pwXSq * .25 + pwYSq * .25) * .5
        const float pwSq = (pwXSq + pwYSq) * .125f;

        // loop over verts looking for a nearby edge
        for (int i1 = 0; i1 < numPolyVertices; ++i1) {
            int i0 = i1? i1 - 1 : numPolyVertices - 1;

            const Vec3f p0 = isect.getAttribute(StandardAttributes::sPolyVertices[i0]);
            const Vec3f p1 = isect.getAttribute(StandardAttributes::sPolyVertices[i1]);

            const Vec3f edge = p1 - p0;
            const Vec3f dir = isect.getP() - p0;

            // "A" is the quad defined by edge and dir
            const float areaASq = scene_rdl2::math::lengthSqr(scene_rdl2::math::cross(edge, dir));

            // "B" is the triangle (p1, p, p0), which is half of "A"
            //      areaA = areaB * 2
            //      areaAsq = areaBSq * 4
            //
            // "d" is the height of B, and more importantly, the
            // closest distance from the intersection point p, to the edge.
            //    areaB   = d * length(edge) / 2
            //    areaBSq = dSq * lengthSqr(edge) / 4
            //    areaASq = dSq * lengthSqr(edge)
            //
            // in the case where p is on the edge, d must be within
            // pixel width ditance of the edge (i.e. d < pw).
            //    dSq < pwSq
            //    areaASq / lengthSqr(edge) < pwSq
            //    areaASq < pwSq * lengthSqr(edge)
            if (areaASq < pwSq * lengthSqr(edge)) {
                result = 1.f;
                break; // all done
            }
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
// Motion Vectors

// ispc hook
void
CPP_computeMotion(intptr_t scenePtr, float time, const Vec3f *p, const Vec3f *dp,
                  Vec2f *result)
{
    const Scene *scene = reinterpret_cast<const Scene *>(scenePtr);
    const Camera *pbrCamera = scene->getCamera();
    MNRY_ASSERT(pbrCamera);

    const float halfDt = 0.01f;
    Vec3f p0, p1; // p in render space at t - halfDt, t + halfDt
    p0 = p1 = *p;

    // motion from primitive itself
    if (dp) {
        // dp contains the instantaneous velocity of the intersection point in
        // render space units per shutter interval
        p0 = p0 - halfDt * *dp;
        p1 = p1 + halfDt * *dp;
    }

    // motion from the camera and projection to screen space
    Mat4f renderToScreen0;
    Mat4f renderToScreen1;
    // camera motion
    const Mat4f renderToCamera0 = pbrCamera->computeRender2Camera(time - halfDt);
    const Mat4f renderToCamera1 = pbrCamera->computeRender2Camera(time + halfDt);
    // camera projection
    const scene_rdl2::rdl2::Camera *rdlCamera = pbrCamera->getRdlCamera();
    if (rdlCamera->doesSupportProjectionMatrix()) {
        const ProjectiveCamera *projCamera =
            static_cast<const ProjectiveCamera *>(pbrCamera);
        const Mat4f cameraToScreen0 = projCamera->computeCamera2Screen(time - halfDt);
        const Mat4f cameraToScreen1 = projCamera->computeCamera2Screen(time + halfDt);
        renderToScreen0 = renderToCamera0 * cameraToScreen0;
        renderToScreen1 = renderToCamera1 * cameraToScreen1;
    } else {
        // it's not clear what to do in the case where the camera is not
        // projective, we'll just output in camera space for now.
        renderToScreen0 = renderToCamera0;
        renderToScreen1 = renderToCamera1;
    }
    const Vec3f pNdc0 = transformH(renderToScreen0, p0);
    const Vec3f pNdc1 = transformH(renderToScreen1, p1);
    // p in screen space at time - halfDt, time + halfDt.  Dropping z
    const Vec2f pss0 = Vec2f(pNdc0.x, pNdc0.y);
    const Vec2f pss1 = Vec2f(pNdc1.x, pNdc1.y);
    
    // approximate the derivative as (pss1 - pss0) / (2 * halfDt)
    // motion is screen space units / shutter_internval
    Vec2f motion = (pss1 - pss0) / (2.0f * halfDt);

    // make the motion in terms of frames, not shutter intervals
    const float shutterDuration = rdlCamera->get(scene_rdl2::rdl2::Camera::sMbShutterCloseKey) -
        rdlCamera->get(scene_rdl2::rdl2::Camera::sMbShutterOpenKey);
    motion /= shutterDuration;

    *result = motion;
}

// compute a 2D screen space motion vector from an isect
static Vec2f
computeMotion(const shading::Intersection &isect,
              const mcrt_common::RayDifferential &ray,
              const Scene &scene)
{
    // use CPP_computeMotion, which is shared with ISPC
    intptr_t scenePtr = (intptr_t) &scene;
    const float time = ray.getTime();
    Vec3f p = isect.getP();
    Vec3f dp;
    Vec3f *dpPtr = nullptr;
    MNRY_ASSERT(isect.isProvided(StandardAttributes::sMotion));
    if (isect.isProvided(StandardAttributes::sMotion)) {
        dp = isect.getAttribute(StandardAttributes::sMotion);
        dpPtr = &dp;
    }
    Vec2f motion;
    CPP_computeMotion(scenePtr, time, &p, dpPtr, &motion);

    return motion;
}

//-----------------------------------------------------------------------------

// Loop over aovSchema. For each entry that corresponds to a beauty aov,
// get the value and write it to the appropriate slot. Skip any locations not
// corresponding to beauty or alpha.
// The corresponding vectorized code is Film::addBeautyAndAlphaSamplesToBuffer().
void
aovSetBeautyAndAlpha(pbr::TLState *pbrTls,
                     const AovSchema &aovSchema,
                     const Color &c,
                     float alpha,
                     float pixelWeight,
                     float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    for (const auto &entry: aovSchema) {
        if (entry.type() == AOV_TYPE_BEAUTY) {
            const float weight = entry.filter() == AOV_FILTER_AVG ? pixelWeight : 1.0f;
            dest[0] = c[0] * weight;
            dest[1] = c[1] * weight;
            dest[2] = c[2] * weight;
        } else if (entry.type() == AOV_TYPE_ALPHA) {
            const float weight = entry.filter() == AOV_FILTER_AVG ? pixelWeight : 1.0f;
            dest[0] = alpha * weight;
        }

        // onto the next
        dest += entry.numChannels();
    }
}

// given an ID from an aov schema, extract the corresponding value
// from the intersection structure
static inline void
getStateVarVolumeOnly(int aovSchemaId, float volumeT,
                      const mcrt_common::RayDifferential &ray, const Scene &scene,
                      float pixelWeight, float *dest)
{
    MNRY_ASSERT(aovType(aovSchemaId) == AOV_TYPE_STATE_VAR);

    switch (aovSchemaId) {
        case AOV_SCHEMA_ID_STATE_P:
            {
                Vec3f renderPos = ray.getOrigin() + ray.getDirection() * volumeT;
                *dest       = renderPos.x * pixelWeight;
                *(dest + 1) = renderPos.y * pixelWeight;
                *(dest + 2) = renderPos.z * pixelWeight;
            }
            break;
        case AOV_SCHEMA_ID_STATE_NG:
            break;
        case AOV_SCHEMA_ID_STATE_N:
            break;
        case AOV_SCHEMA_ID_STATE_ST:
            break;
        case AOV_SCHEMA_ID_STATE_DPDS:
            break;
        case AOV_SCHEMA_ID_STATE_DPDT:
            break;
        case AOV_SCHEMA_ID_STATE_DSDX:
            break;
        case AOV_SCHEMA_ID_STATE_DSDY:
            break;
        case AOV_SCHEMA_ID_STATE_DTDX:
            break;
        case AOV_SCHEMA_ID_STATE_DTDY:
            break;
        case AOV_SCHEMA_ID_WIREFRAME:
            break;
        case AOV_SCHEMA_ID_STATE_WP:
            {
                Vec3f renderPos = ray.getOrigin() + ray.getDirection() * volumeT;
                Vec3d renderPos64(renderPos.x, renderPos.y, renderPos.z);
                Vec3d worldPos = transformPoint(scene.getRender2World(), renderPos64);
                *dest       = static_cast<float>(worldPos.x) * pixelWeight;
                *(dest + 1) = static_cast<float>(worldPos.y) * pixelWeight;
                *(dest + 2) = static_cast<float>(worldPos.z) * pixelWeight;
            }
            break;
        case AOV_SCHEMA_ID_STATE_DEPTH:
            {
                // Same conversion we use in the DeepBuffer
                float volumeZ = -volumeT * ray.getDirection().z;
                *dest = volumeZ * pixelWeight;
            break;
            }
        case AOV_SCHEMA_ID_STATE_MOTION:
            {
            }
            break;
        default:
            MNRY_ASSERT(0 && "unknown aov type in schema");
    }
}

// given an ID from an aov schema, extract the corresponding value
// from the intersection structure
static inline void
getStateVar(int aovSchemaId, const shading::Intersection &isect, float volumeT,
            const mcrt_common::RayDifferential &ray, const Scene &scene,
            float pixelWeight, float *dest)
{
    MNRY_ASSERT(aovType(aovSchemaId) == AOV_TYPE_STATE_VAR);

    switch (aovSchemaId) {
        case AOV_SCHEMA_ID_STATE_P:
            {
                if (ray.getEnd() < volumeT) {
                    *dest       = isect.getP().x * pixelWeight;
                    *(dest + 1) = isect.getP().y * pixelWeight;
                    *(dest + 2) = isect.getP().z * pixelWeight;
                } else {
                    // ray origin is now intersection point so we need to go backwards
                    Vec3f renderPos = ray.getOrigin() - ray.getDirection() * ray.getEnd() + ray.getDirection() * volumeT;
                    *dest       = renderPos.x * pixelWeight;
                    *(dest + 1) = renderPos.y * pixelWeight;
                    *(dest + 2) = renderPos.z * pixelWeight;
                }
                break;
            }
        case AOV_SCHEMA_ID_STATE_NG:
            *dest       = isect.getNg().x * pixelWeight;
            *(dest + 1) = isect.getNg().y * pixelWeight;
            *(dest + 2) = isect.getNg().z * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_N:
            *dest       = isect.getN().x * pixelWeight;
            *(dest + 1) = isect.getN().y * pixelWeight;
            *(dest + 2) = isect.getN().z * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_ST:
            *dest       = isect.getSt().x * pixelWeight;
            *(dest + 1) = isect.getSt().y * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_DPDS:
            *dest       = isect.getdPds().x * pixelWeight;
            *(dest + 1) = isect.getdPds().y * pixelWeight;
            *(dest + 2) = isect.getdPds().z * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_DPDT:
            *dest       = isect.getdPdt().x * pixelWeight;
            *(dest + 1) = isect.getdPdt().y * pixelWeight;
            *(dest + 2) = isect.getdPdt().z * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_DSDX:
            *dest = isect.getdSdx() * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_DSDY:
            *dest = isect.getdSdy() * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_DTDX:
            *dest = isect.getdTdx() * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_DTDY:
            *dest = isect.getdTdy() * pixelWeight;
            break;
        case AOV_SCHEMA_ID_WIREFRAME:
            *dest = sampleWireframe(isect) * pixelWeight;
            break;
        case AOV_SCHEMA_ID_STATE_WP:
            {
                // precision loss is unavoidable as our output is necessarily 32 bits
                if (ray.getEnd() < volumeT) {
                    Vec3f renderPos = isect.getP();
                    Vec3d renderPos64(renderPos.x, renderPos.y, renderPos.z);
                    Vec3d worldPos = transformPoint(scene.getRender2World(), renderPos64);
                    *dest       = static_cast<float>(worldPos.x) * pixelWeight;
                    *(dest + 1) = static_cast<float>(worldPos.y) * pixelWeight;
                    *(dest + 2) = static_cast<float>(worldPos.z) * pixelWeight;
                } else {
                    // ray origin is now intersection point so we need to go backwards
                    Vec3f renderPos = ray.getOrigin() - ray.getDirection() * ray.getEnd() + ray.getDirection() * volumeT;
                    Vec3d renderPos64(renderPos.x, renderPos.y, renderPos.z);
                    Vec3d worldPos = transformPoint(scene.getRender2World(), renderPos64);
                    *dest       = static_cast<float>(worldPos.x) * pixelWeight;
                    *(dest + 1) = static_cast<float>(worldPos.y) * pixelWeight;
                    *(dest + 2) = static_cast<float>(worldPos.z) * pixelWeight;
                }
            }
            break;
        case AOV_SCHEMA_ID_STATE_DEPTH:
            {
                float cameraZ = scene.getCamera()->computeZDistance(
                                    isect.getP(), ray.getOrigin(), ray.getTime());
                // Same conversion we use in the DeepBuffer
                float volumeZ = -volumeT * ray.getDirection().z;
                float z = min(cameraZ, volumeZ);
                *dest = z * pixelWeight;
            break;
            }
        case AOV_SCHEMA_ID_STATE_MOTION:
            {
                const Vec2f v = computeMotion(isect, ray, scene) * pixelWeight;
                *dest       = v.x;
                *(dest + 1) = v.y;
            }
            break;
        default:
            MNRY_ASSERT(0 && "unknown aov type in schema");
    }
}

// Loop over aovSchema, for each entry that corresponds to a state var,
// get the value from the isect and place it in the appropriate slot.  skip
// over any locations not corresponding to state vars.
void
aovSetStateVars(pbr::TLState *pbrTls,
                const AovSchema &aovSchema,
                const shading::Intersection &isect,
                float volumeT,
                const mcrt_common::RayDifferential &ray,
                const Scene &scene,
                float pixelWeight,
                float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    for (const auto &entry: aovSchema) {
        if (entry.type() == AOV_TYPE_STATE_VAR) {
            float weight = entry.filter() == AOV_FILTER_AVG ? pixelWeight : 1.0f;
            getStateVar(entry.id(), isect, volumeT, ray, scene, weight, dest);
        }

        // onto the next
        dest += entry.numChannels();
    }
}

void
aovSetStateVarsVolumeOnly(pbr::TLState *pbrTls,
                          const AovSchema &aovSchema,
                          float volumeT,
                          const mcrt_common::RayDifferential &ray,
                          const Scene &scene,
                          float pixelWeight,
                          float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    for (const auto &entry: aovSchema) {
        if (entry.type() == AOV_TYPE_STATE_VAR) {
            float weight = (entry.filter() == AOV_FILTER_AVG) ? pixelWeight : 1.0f;
            getStateVarVolumeOnly(entry.id(), volumeT, ray, scene, weight, dest);
        }

        // onto the next
        dest += entry.numChannels();
    }
}

// ---------------------------------------------------------------------------
// Primitive attribute aovs
// ---------------------------------------------------------------------------

// Given a primitive attribute index from the geom library
// compute an integer code that can be used in an aovSchema array.
int
aovFromGeomIndex(int geomIndex)
{
    MNRY_ASSERT(geomIndex >= 0);
    MNRY_ASSERT(geomIndex < AOV_MAX_RANGE_TYPE);
    AttributeKey key(geomIndex);
    MNRY_ASSERT(key.isValid());

    AovSchemaId schemaCode = AOV_SCHEMA_ID_UNKNOWN;
    switch(key.getType())
    {
    case scene_rdl2::rdl2::TYPE_FLOAT:
        schemaCode = AOV_SCHEMA_ID_PRIM_ATTR_FLOAT;
        break;
    case scene_rdl2::rdl2::TYPE_VEC2F:
        schemaCode = AOV_SCHEMA_ID_PRIM_ATTR_VEC2F;
        break;
    case scene_rdl2::rdl2::TYPE_VEC3F:
        schemaCode = AOV_SCHEMA_ID_PRIM_ATTR_VEC3F;
        break;
    case scene_rdl2::rdl2::TYPE_RGB:
        schemaCode = AOV_SCHEMA_ID_PRIM_ATTR_RGB;
        break;
    default:
        MNRY_ASSERT(0 && "unsupported aov primitive attribute type");
    }

    return geomIndex + schemaCode;
}

// Given an integer from an aovSchema array, compute the corresponding
// primitive attribute index (use by the geom library).
int
aovToGeomIndex(int aovSchemaId)
{
    MNRY_ASSERT(aovType(aovSchemaId) == AOV_TYPE_PRIM_ATTR);

    const int result = aovToRangeTypeOffset(aovSchemaId);

    return result;
}

// given an integer code from an aov schema, extract the corresponding value
// from the intersection structure.
static inline void
getPrimAttr(int geomIndex, const shading::Intersection &isect,
            float pixelWeight, float missValue, float *dest)
{
    AttributeKey key(geomIndex);
    bool miss = !isect.isProvided(key);

    switch (key.getType()) {
    case scene_rdl2::rdl2::TYPE_RGB:
        {
            scene_rdl2::rdl2::Rgb res = miss ?
                scene_rdl2::rdl2::Rgb(missValue) :
                isect.getAttribute(TypedAttributeKey<scene_rdl2::rdl2::Rgb>(key));
            res *= pixelWeight;
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    case scene_rdl2::rdl2::TYPE_VEC3F:
        {
            scene_rdl2::rdl2::Vec3f res = miss ?
                scene_rdl2::rdl2::Vec3f(missValue) :
                isect.getAttribute(TypedAttributeKey<scene_rdl2::rdl2::Vec3f>(key));
            res *= pixelWeight;
            *dest       = res.x;
            *(dest + 1) = res.y;
            *(dest + 2) = res.z;
        }
        break;
    case scene_rdl2::rdl2::TYPE_VEC2F:
        {
            scene_rdl2::rdl2::Vec2f res = miss ?
                scene_rdl2::rdl2::Vec2f(missValue) :
                isect.getAttribute(TypedAttributeKey<scene_rdl2::rdl2::Vec2f>(key));
            res *= pixelWeight;
            *dest       = res.x;
            *(dest + 1) = res.y;
        }
        break;
    case scene_rdl2::rdl2::TYPE_FLOAT:
        *dest = (miss ? missValue : isect.getAttribute(TypedAttributeKey<scene_rdl2::rdl2::Float>(key))) * pixelWeight;
        break;
    default:
        MNRY_ASSERT(0 && "unsupported primitive attribute type for aov");
    }
}

static float
getMissValue(AovFilter filter)
{
    float missValue = 0.0f; // for avg and sum
    if (filter == AOV_FILTER_MIN) {
        missValue = scene_rdl2::math::pos_inf;
    } else if (filter == AOV_FILTER_MAX) {
        missValue = scene_rdl2::math::neg_inf;
    }

    return missValue;
}

void
aovSetPrimAttrs(pbr::TLState *pbrTls,
                const AovSchema &aovSchema,
                const std::vector<char> &activeFlags,
                const shading::Intersection &isect,
                float pixelWeight,
                float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    auto af = activeFlags.begin();
    for (const auto &entry: aovSchema) {
        if (entry.type() == AOV_TYPE_PRIM_ATTR && *af) {
            float weight = entry.filter() == AOV_FILTER_AVG ? pixelWeight : 1.0f;
            float missValue = getMissValue(entry.filter());
            getPrimAttr(aovToGeomIndex(entry.id()), isect, weight, missValue, dest);
        }
        // onto the next one
        ++af;
        dest += entry.numChannels();
    }
}

// ---------------------------------------------------------------------------
// Material aovs
// ---------------------------------------------------------------------------

// Material AOVs are a mixed bag.  Some can be computed directly
// from the Bsdf(v) (e.g. emission).  Some require sampling results from
// the integration code (e.g. albedo, subsurface).

// Every material aov entry has a single vectorized and non-vectorized compute
// function associated with it.

// ---------------------------------------------------------------------------
// utility functions

static bool
labelMatch(int labelId, const std::vector<int> &entryLabelIds)
{
    // if no labels exist on the aov entry
    // then the labelId is irrelvant
    const auto begin = entryLabelIds.begin();
    const auto end = entryLabelIds.end();

    // entry doesn't have labels
    if (begin == end) return true;

    // entry has labels, one must match labelId
    if (std::find(begin, end, labelId) != end) {
        return true;
    }

    // entry has labels, but labelId does not match
    return false;
}

static std::string
trim(const std::string &in)
{
    // trim leading and trailing white space
    const auto leftTrim = [](const std::string& str) {
        std::string result = str;
        auto it = std::find_if(result.begin(), result.end(), [](char ch) { return !std::isspace(
                    static_cast<unsigned char>(ch)); });
        result.erase(result.begin(), it);
        return result;
    };

    const auto rightTrim = [](const std::string& str) {
        std::string result = str;
        auto it = std::find_if(result.rbegin(), result.rend(), [](char ch) { return !std::isspace(
                    static_cast<unsigned char>(ch)); });
        result.erase(it.base(), result.end());
        return result;
    };

    return leftTrim(rightTrim(in));
}

// ---------------------------------------------------------------------------
// albedo material aov computations

// "albedo" in this context is meant to be a cheap material diagnostic that
// shows the result of integrating the material with an unoccluded pure white
// environment.

// returns albedo sum for all matching selections

static void
computeAlbedo(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    // result
    Color result = scene_rdl2::math::sBlack;
    unsigned int numContributions = 0;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // lobe albedo
        if (flags != shading::BsdfLobe::NONE) {
            if (p.mBSampler && p.mBsmps) {
                // can use bsdf sampler for higher quality albedo result

                // make sure that the sampler's flags contains our flags
                MNRY_ASSERT((p.mBSampler->getBsdfSlice().getFlags() & flags) == flags);

                int s = 0;
                for (int lobeIdx = 0; lobeIdx < p.mBSampler->getLobeCount(); ++lobeIdx) {
                    const shading::BsdfLobe *lobe = p.mBSampler->getLobe(lobeIdx);
                    const int ni = p.mBSampler->getLobeSampleCount(lobeIdx);
                    const float invNi = p.mBSampler->getInvLobeSampleCount(lobeIdx);
                    if (lobe->matchesFlags(flags) &&
                        labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {
                        ++numContributions;
                        for (int i = 0; i < ni; ++i) {
                            const BsdfSample &bsmp = p.mBsmps[s + i];
                            if (bsmp.isValid()) {
                                result += invNi * bsmp.f / bsmp.pdf;
                            }
                        }
                    }
                    s += ni;
                }
            } else if (p.mBsdfSlice) {
                // Not using the bsdf sampler, which means we don't have bsdf sample objects.
                // We'll fallback to the cheaper lobe->albedo() method.  This might actually
                // be a more useful result since the lobe albedo method is used directly by the
                // bsdf one sampler to drive sampling decisions.
                for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                    const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                    if (lobe->matchesFlags(flags) &&
                        labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {
                        ++numContributions;
                        result += lobe->albedo(*p.mBsdfSlice);
                    }
                }
            }

            // pixel weight
            result *= p.mPixelWeight;
        }

        // subsurface albedo
        if (subsurface) {
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                ++numContributions;
                // ssAov already takes pixel weight into account
                result += p.mSsAov;
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                ++numContributions;
                // ssAov already takes pixel weight into account
                result += p.mSsAov;
            }
        }
    }

    // don't double count the albedo
    if (numContributions > 1) {
        // take the average
        result /= numContributions;
    }

    // pack results
    *dest       += result.r;
    *(dest + 1) += result.g;
    *(dest + 2) += result.b;
}

static void
computeAlbedov(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computeAlbedo(flags, subsurface,
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                        &p.mBsdf, p.mBSampler, p.mBsmps, p.mBsdfSlice, p.mPixelWeight,
                        &p.mSsAov, dest, p.mLanemask);
}

// ---------------------------------------------------------------------------
// color material aov computations

// returns color sum of all matching selections

static void
computeColor(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    Color result = scene_rdl2::math::sBlack;

    bool diffuseLobeContributed = false;
    bool subsurfaceContributed = false;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                if (lobe->matchesFlags(flags) &&
                    lobe->hasProperty(shading::BsdfLobe::PROPERTY_COLOR) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {

                    diffuseLobeContributed = true;

                    Color c;
                    lobe->getProperty(shading::BsdfLobe::PROPERTY_COLOR, reinterpret_cast<float *>(&c));
                    result += c;
                }
            }
        }

        // subsurface
        if (subsurface) {
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices) &&
                bssrdf->hasProperty(shading::Bssrdf::PROPERTY_COLOR)) {
                subsurfaceContributed = true;
                scene_rdl2::math::Color c;
                bssrdf->getProperty(shading::Bssrdf::PROPERTY_COLOR,
                    reinterpret_cast<float *>(&c));
                result += c;
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices) &&
                volumeSubsurface->hasProperty(shading::VolumeSubsurface::PROPERTY_COLOR)) {
                scene_rdl2::math::Color c;
                volumeSubsurface->getProperty(shading::VolumeSubsurface::PROPERTY_COLOR,
                    reinterpret_cast<float *>(&c));
                result += c;
            }
        }

        // pixel weight
        result *= p.mPixelWeight;

    }

    // don't double count the albedo
    if (diffuseLobeContributed && subsurfaceContributed) {
        result *= 0.5f;
    }

    // pack results
    *dest       += result.r;
    *(dest + 1) += result.g;
    *(dest + 2) += result.b;
}

static void
computeColorv(const MaterialAovs::ComputeParamsv &p, float *dest)

{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computeColor(&p.mBsdf, flags, subsurface,
                       reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                       reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                       reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                       p.mPixelWeight, dest, p.mLanemask);
}

// ---------------------------------------------------------------------------
// emission material aov computations

static void
computeEmission(const MaterialAovs::ComputeParams &p, float *dest)
{
    if (!p.mBsdf.getEarlyTermination() &&
        labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // get the emission
        Color emission = p.mBsdf.getSelfEmission();

        // apply pixel weight
        emission *= p.mPixelWeight;

        // pack results
        *dest       += emission.r;
        *(dest + 1) += emission.g;
        *(dest + 2) += emission.b;
    }
}

static void
computeEmissionv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    ispc::computeEmission(&p.mBsdf,
                          reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                          reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                          p.mPixelWeight, dest, p.mLanemask);
}

// ---------------------------------------------------------------------------
// Matte material aov computations

static void
computeMatte(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    float result = 0.f;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                if (lobe->matchesFlags(flags) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {
                    // result is the pixelWeight
                    result = p.mPixelWeight;
                    break;
                }
            }
        }

        // subsurface
        if (result == 0.f && p.mEntry.mSubsurface) {
            const Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                result = p.mPixelWeight;
            }
            const VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                result = p.mPixelWeight;
            }
        }

        // if our bsdf has no lobes or sub-surface, and our entry flags are empty
        // we want to output the matte aov
        if (!p.mBsdf.getLobeCount() && !p.mBsdf.getBssrdf() &&
            !p.mBsdf.getVolumeSubsurface() && flags == shading::BsdfLobe::ALL) {
            result = p.mPixelWeight;
        }
    }

    // pack results
    *dest += result;
}

static void
computeMattev(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;

    ispc::computeMatte(flags,
                       p.mEntry.mSubsurface,
                       reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                       reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                       reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                       &p.mBsdf, p.mPixelWeight, dest, p.mLanemask);
}

// --------------------------------------------------------------------------
// Depth maps
// --------------------------------------------------------------------------

// get a render space to camera space matrix at
// shutter time t.  this function is used by ispc,
// where there is no Scene struct, so the type for
// Scene* is just inptr_t
static void
getRender2Camera(intptr_t scenePtr, float t, Mat4f *r2c)
{
    const Scene *scene = reinterpret_cast<const Scene *>(scenePtr);
    const Camera *camera = scene->getCamera();
    if (scene_rdl2::math::isZero(t)) {
        // the pbr::Camera caches this xform
        *r2c = camera->getRender2Camera();
    } else {
        *r2c = camera->computeRender2Camera(t);
    }
}

// ---------------------------------------------------------------------------
// specular roughness

// average specular roughness of all matching selections

static inline void
computeRoughness(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;

    Vec2f result(0.f, 0.f);
    int matchingLobes = 0;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int i = 0; i < p.mBsdf.getLobeCount(); ++i) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(i);
                if (lobe->matchesFlags(flags) &&
                    lobe->hasProperty(shading::BsdfLobe::PROPERTY_ROUGHNESS) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {
                    Vec2f sr;
                    lobe->getProperty(shading::BsdfLobe::PROPERTY_ROUGHNESS,
                                      reinterpret_cast<float *>(&sr));
                    result += sr;
                    ++matchingLobes;
                }
            }
        }

        if (matchingLobes) result /= static_cast<float>(matchingLobes);

        // apply pixel weight
        result *= p.mPixelWeight;

    }

    // pack results
    *dest       += result.x;
    *(dest + 1) += result.y;
}

static inline void
computeRoughnessv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;

    ispc::computeRoughness(flags,
                           reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                           reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                           reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                           &p.mBsdf, p.mPixelWeight, dest, p.mLanemask);
}

// ---------------------------------------------------------------------------
// normal computations

// this computes an average normal for all matching selections

static inline void
computeNormal(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    Vec3f result(0.f, 0.f, 0.f);

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        int matchingLobes = 0;

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int i = 0; i < p.mBsdf.getLobeCount(); ++i) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(i);
                if (lobe->matchesFlags(flags) &&
                    lobe->hasProperty(shading::BsdfLobe::PROPERTY_NORMAL) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()) , p.mEntry.mLabelIndices)) {
                    Vec3f N;
                    lobe->getProperty(shading::BsdfLobe::PROPERTY_NORMAL,
                                      reinterpret_cast<float *>(&N));
                    result += N;

                    ++matchingLobes;
                }
            }
        }

        // subsurface
        if (subsurface) {
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                const scene_rdl2::math::ReferenceFrame &frame = bssrdf->getFrame();
                result += frame.getN();
                ++matchingLobes;
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                result += volumeSubsurface->getN();
                ++matchingLobes;
            }
        }

        // average
        if (matchingLobes) result /= static_cast<float>(matchingLobes);

        // apply pixel weight
        result *= p.mPixelWeight;
    }

    // pack results
    *dest       += result.x;
    *(dest + 1) += result.y;
    *(dest + 2) += result.z;
}

static inline void
computeNormalv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computeNormal(flags, subsurface,
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                        &p.mBsdf, p.mPixelWeight, dest, p.mLanemask);
}

// ---------------------------------------------------------------------------
// subsurface radius computations

static inline void
computeRadius(const MaterialAovs::ComputeParams &p, float *dest)
{
    Color result = scene_rdl2::math::sBlack;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
        if (bssrdf && labelMatch(shading::aovDecodeLabel(
            bssrdf->getLabel()), p.mEntry.mLabelIndices) &&
            bssrdf->hasProperty(shading::Bssrdf::PROPERTY_RADIUS)) {
            bssrdf->getProperty(shading::Bssrdf::PROPERTY_RADIUS,
                reinterpret_cast<float *>(&result));
        }
        const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
        if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
            volumeSubsurface->getLabel()), p.mEntry.mLabelIndices) &&
            volumeSubsurface->hasProperty(shading::VolumeSubsurface::PROPERTY_RADIUS)) {
            volumeSubsurface->getProperty(shading::VolumeSubsurface::PROPERTY_RADIUS,
                reinterpret_cast<float *>(&result));
        }
        // apply pixel weight
        result *= p.mPixelWeight;
    }

    // pack results
    *dest       += result.r;
    *(dest + 1) += result.g;
    *(dest + 2) += result.b;
}

static inline void
computeRadiusv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    ispc::computeRadius(&p.mBsdf,
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                        reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                        p.mPixelWeight, dest, p.mLanemask);
}


// ---------------------------------------------------------------------------
// fresnel color computations

// computes fresnel color sum of matching selections

static void
computeFresnelColor(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    Color result = scene_rdl2::math::sBlack;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {
        // lobe fresnel color
        if (flags != shading::BsdfLobe::NONE) {
            for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                if (lobe->matchesFlags(flags) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {

                    const shading::Fresnel *fresnel = lobe->getFresnel();
                    if (fresnel && fresnel->hasProperty(shading::Fresnel::PROPERTY_COLOR)) {
                        Color c;
                        fresnel->getProperty(shading::Fresnel::PROPERTY_COLOR, reinterpret_cast<float *>(&c));
                        result += c;
                    }
                }
            }
        }

        // subsurface fresnel color
        if (subsurface) {
            const shading::Fresnel *fresnel = nullptr;
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                fresnel = bssrdf->getTransmissionFresnel();
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                fresnel = volumeSubsurface->getTransmissionFresnel();
            }
            if (fresnel && fresnel->hasProperty(shading::Fresnel::PROPERTY_COLOR)) {
                scene_rdl2::math::Color ssResult;
                fresnel->getProperty(shading::Fresnel::PROPERTY_COLOR,
                    reinterpret_cast<float *>(&ssResult));
                result += ssResult;
            }
        }

        // pixel weight
        result *= p.mPixelWeight;
    }

    // pack results
    *dest       += result.r;
    *(dest + 1) += result.g;
    *(dest + 2) += result.b;
}

static void
computeFresnelColorv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computeFresnelColor(&p.mBsdf, flags, subsurface,
                              reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                              reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                              reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                              p.mPixelWeight, dest, p.mLanemask);

}

// ---------------------------------------------------------------------------
// fresnel factor

// This computation produces an average of all fresnel factors
// for matching selections
static void
computeFresnelFactor(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    float result = 0.f;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        int s = 0;

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                if (lobe->matchesFlags(flags) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {

                    const shading::Fresnel *fresnel = lobe->getFresnel();
                    if (fresnel && fresnel->hasProperty(shading::Fresnel::PROPERTY_FACTOR)) {
                        float f;
                        fresnel->getProperty(shading::Fresnel::PROPERTY_FACTOR, &f);
                        result += f;
                        ++s;
                    }
                }
            }
        }

        // subsurface
        if (subsurface) {
            const shading::Fresnel *fresnel = nullptr;
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                fresnel = bssrdf->getTransmissionFresnel();
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                fresnel = volumeSubsurface->getTransmissionFresnel();
            }
            if (fresnel && fresnel->hasProperty(shading::Fresnel::PROPERTY_FACTOR)) {
                float f;
                fresnel->getProperty(shading::Fresnel::PROPERTY_FACTOR, &f);
                result += f;
                ++s;
            }
        }

        // average
        if (s > 0) result /= s;

        // pixel weight
        result *= p.mPixelWeight;

    }

    // pack results
    *dest = +result;
}

static void
computeFresnelFactorv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computeFresnelFactor(&p.mBsdf, flags, subsurface,
                               reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                               reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                               reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                               p.mPixelWeight, dest, p.mLanemask);
}

//----------------------------------------------------------------------------
// fresnel roughness computations

// compute average roughness of all selected items

static void
computeFresnelRoughness(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    Vec2f result(0.f, 0.f);

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {
        int matchingLobes = 0;

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int i = 0; i < p.mBsdf.getLobeCount(); ++i) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(i);
                if (lobe->matchesFlags(flags) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {

                    const shading::Fresnel *fresnel = lobe->getFresnel();
                    if (fresnel && fresnel->hasProperty(shading::Fresnel::PROPERTY_ROUGHNESS)) {
                        Vec2f sr;
                        fresnel->getProperty(shading::Fresnel::PROPERTY_ROUGHNESS,
                                             reinterpret_cast<float *>(&sr));
                        result += sr;
                        ++matchingLobes;
                    }
                }
            }
        }

        // subsurface
        if (subsurface) {
            const shading::Fresnel *fresnel = nullptr;
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                fresnel = bssrdf->getTransmissionFresnel();
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                fresnel = volumeSubsurface->getTransmissionFresnel();
            }
            if (fresnel && fresnel->hasProperty(shading::Fresnel::PROPERTY_ROUGHNESS)) {
                scene_rdl2::math::Vec2f sr;
                fresnel->getProperty(shading::Fresnel::PROPERTY_ROUGHNESS,
                    reinterpret_cast<float *>(&sr));
                result += sr;
                ++matchingLobes;
            }
        }

        if (matchingLobes) result /= static_cast<float>(matchingLobes);

        // apply pixel weight
        result *= p.mPixelWeight;
    }

    // pack results
    *dest       += result.x;
    *(dest + 1) += result.y;
}

static void
computeFresnelRoughnessv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computeFresnelRoughness(&p.mBsdf, flags, subsurface,
                                  reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                                  reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                                  reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                                  p.mPixelWeight, dest, p.mLanemask);
}

// ---------------------------------------------------------------------------
// pbr validity material aov computations

// returns pbr validity sum of all matching selections

static void
computePbrValidity(const MaterialAovs::ComputeParams &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    Color result = scene_rdl2::math::sBlack;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // lobes
        if (flags != shading::BsdfLobe::NONE) {
            for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                if (lobe->matchesFlags(flags) &&
                    lobe->hasProperty(shading::BsdfLobe::PROPERTY_PBR_VALIDITY) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {

                    Color c;
                    lobe->getProperty(shading::BsdfLobe::PROPERTY_PBR_VALIDITY, reinterpret_cast<float *>(&c));
                    result += c;
                }
            }
        }

        // subsurface
        if (subsurface) {
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices) &&
                bssrdf->hasProperty(shading::Bssrdf::PROPERTY_PBR_VALIDITY)) {
                scene_rdl2::math::Color c;
                bssrdf->getProperty(shading::Bssrdf::PROPERTY_PBR_VALIDITY,
                    reinterpret_cast<float *>(&c));
                result += c;
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices) &&
                volumeSubsurface->hasProperty(shading::VolumeSubsurface::PROPERTY_PBR_VALIDITY)) {
                scene_rdl2::math::Color c;
                volumeSubsurface->getProperty(shading::VolumeSubsurface::PROPERTY_PBR_VALIDITY,
                    reinterpret_cast<float *>(&c));
                result += c;
            }
        }

        // pixel weight
        result *= p.mPixelWeight;

    }

    // pack results
    *dest       += result.r;
    *(dest + 1) += result.g;
    *(dest + 2) += result.b;
}

static void
computePbrValidityv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;
    const int subsurface = p.mEntry.mSubsurface;

    ispc::computePbrValidity(&p.mBsdf, flags, subsurface,
                             reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mLabelIndices),
                             reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mMaterialLabelIndices),
                             reinterpret_cast<const ispc::std_vector *>(&p.mEntry.mGeomLabelIndices),
                             p.mPixelWeight, dest, p.mLanemask);
}

//----------------------------------------------------------------------------
// state aovs as material aovs
// the advantage of these over state aovs is the ability to apply lobe and
// label filtering.
//

// label matching is identical for state and primitive attributes
static bool
labelMatch(const MaterialAovs::ComputeParams &p)
{
    const int flags = p.mEntry.mLobeFlags;
    bool matched = false;

    if (labelMatch(p.mBsdf.getGeomLabelId(), p.mEntry.mGeomLabelIndices) &&
        labelMatch(p.mBsdf.getMaterialLabelId(), p.mEntry.mMaterialLabelIndices)) {

        // This initial case looks complicated, but all it really says is if
        // the user specified a default (empty) lobe selector then
        // match an empty material - unless that empty material is a cutout.
        const bool selectorIsDefault = (p.mEntry.mLobeFlags == shading::BsdfLobe::ALL && p.mEntry.mSubsurface);
        if (selectorIsDefault) {
            if (p.mBsdf.getLobeCount() == 0 && p.mBsdf.getBssrdf() == nullptr) {
                matched = !p.mBsdf.getEarlyTermination();
            }
        }

        // lobes
        if (!matched && flags != shading::BsdfLobe::NONE) {
            for (int lobeIdx = 0; lobeIdx < p.mBsdf.getLobeCount(); ++lobeIdx) {
                const shading::BsdfLobe *lobe = p.mBsdf.getLobe(lobeIdx);
                if (lobe->matchesFlags(flags) &&
                    labelMatch(shading::aovDecodeLabel(lobe->getLabel()), p.mEntry.mLabelIndices)) {
                    matched = true;
                    break;
                }
            }
        }

        // subsurface
        if (!matched && p.mEntry.mSubsurface) {
            // no match yet, check if sub-surface matches
            const shading::Bssrdf *bssrdf = p.mBsdf.getBssrdf();
            if (bssrdf && labelMatch(shading::aovDecodeLabel(
                bssrdf->getLabel()), p.mEntry.mLabelIndices)) {
                matched = true;
            }
            const shading::VolumeSubsurface *volumeSubsurface = p.mBsdf.getVolumeSubsurface();
            if (volumeSubsurface && labelMatch(shading::aovDecodeLpeLabel(
                volumeSubsurface->getLabel()), p.mEntry.mLabelIndices)) {
                matched = true;
            }
        }
    }

    return matched;
}

// set the missed value for material state and prim attr aovs
static void
setMissValue(const MaterialAovs::ComputeParams &p, float *dest)
{
    const float missValue = getMissValue(p.mEntry.mFilter);
    // assumes miss value is 0, +inf, or -inf, so no need to apply pixel weight
    MNRY_ASSERT(missValue == 0.0f || isinf(missValue));
    const int numChannels = aovNumChannels(p.mEntry.mAovSchemaId);
    for (int i = 0; i < numChannels; ++i) {
        *dest++ = missValue;
    }
}

// get the missed value for material state and prim attr aovs
// which is passed into the vector compute functions
static float
getMissValue(const MaterialAovs::ComputeParamsv &p)
{
    return getMissValue(p.mEntry.mFilter);
}

static void
computeStateAov(const MaterialAovs::ComputeParams &p, float *dest)
{
    const bool matched = labelMatch(p);

    if (matched) {
        // if we matched, we compute the state aov
        getStateVar(p.mEntry.mStateAovId, p.mIsect, scene_rdl2::math::sMaxValue, p.mRay, p.mScene,
                    p.mPixelWeight, dest);
    } else {
        // if not, need to fill in dest with an appropriate miss value
        setMissValue(p, dest);
    }
}

static void
computeStateAovv(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;

    // what value should rejected intersections store?
    const float missValue = getMissValue(p);

    // how many channels in this aov?
    const int numChannels = aovNumChannels(p.mEntry.mAovSchemaId);

    // The world position aov needs the render2world xform
    const Mat4d r2w = p.mScene.getRender2World();

    // Motion vector functions needed for motion vector aov
    static ispc::MotionVectorFns mvFns {
        /* computeMotionFunction = */ (intptr_t) CPP_computeMotion,
        /* motionKey = */ StandardAttributes::sMotion.getIndex(),
    };

    // The state aov id does not match the values on the ispc side
    // translate it now
    int stateVar = 0;
    switch (p.mEntry.mStateAovId) {
    case AOV_SCHEMA_ID_STATE_P:
        stateVar = AOV_STATE_VAR_P;
        break;
    case AOV_SCHEMA_ID_STATE_NG:
        stateVar = AOV_STATE_VAR_NG;
        break;
    case AOV_SCHEMA_ID_STATE_N:
        stateVar = AOV_STATE_VAR_N;
        break;
    case AOV_SCHEMA_ID_STATE_ST:
        stateVar = AOV_STATE_VAR_ST;
        break;
    case AOV_SCHEMA_ID_STATE_DPDS:
        stateVar = AOV_STATE_VAR_DPDS;
        break;
    case AOV_SCHEMA_ID_STATE_DPDT:
        stateVar = AOV_STATE_VAR_DPDT;
        break;
    case AOV_SCHEMA_ID_STATE_DSDX:
        stateVar = AOV_STATE_VAR_DSDX;
        break;
    case AOV_SCHEMA_ID_STATE_DSDY:
        stateVar = AOV_STATE_VAR_DSDY;
        break;
    case AOV_SCHEMA_ID_STATE_DTDX:
        stateVar = AOV_STATE_VAR_DTDX;
        break;
    case AOV_SCHEMA_ID_STATE_DTDY:
        stateVar = AOV_STATE_VAR_DTDY;
        break;
    case AOV_SCHEMA_ID_STATE_WP:
        stateVar = AOV_STATE_VAR_WP;
        break;
    case AOV_SCHEMA_ID_STATE_DEPTH:
        stateVar = AOV_STATE_VAR_DEPTH;
        break;
    case AOV_SCHEMA_ID_STATE_MOTION:
        stateVar = AOV_STATE_VAR_MOTION;
        break;
    default:
        MNRY_ASSERT(0 && "unknown aov type in schema");
    }

    ispc::computeStateAov(flags,
                          stateVar,
                          missValue,
                          numChannels,
                          p.mEntry.mSubsurface,
                          (const ispc::std_vector *) &p.mEntry.mLabelIndices,
                          (const ispc::std_vector *) &p.mEntry.mMaterialLabelIndices,
                          (const ispc::std_vector *) &p.mEntry.mGeomLabelIndices,
                          (intptr_t) &p.mIsect,
                          (intptr_t) &p.mRay,
                          (intptr_t) &p.mScene,
                          &p.mBsdf,
                          (intptr_t) getRender2Camera, // needed by depth aov
                          (const double *) &r2w, // needed by Wp aov
                          &mvFns, // needed by motion vector aov
                          p.mPixelWeight,
                          dest,
                          p.mLanemask);
                          
}


//----------------------------------------------------------------------------
// primitive attribute aovs as material aovs
// the advantage of these over primitive attribute aovs is the ability to
// apply lobe and label filtering.

static void
computePrimitiveAttribute(const MaterialAovs::ComputeParams &p, float *dest)
{
    const bool matched = labelMatch(p);

    if (matched) {
        // if we matched, we compute the prim attr aov
        getPrimAttr(p.mEntry.mPrimAttrKey, p.mIsect, p.mPixelWeight, getMissValue(p.mEntry.mFilter), dest);
    } else {
        // if not, need to fill in dest with an appropriate miss value
        setMissValue(p, dest);
    }
}

static void
computePrimitiveAttributev(const MaterialAovs::ComputeParamsv &p, float *dest)
{
    const int flags = p.mEntry.mLobeFlags;

    // what value should rejected intersections store?
    const float missValue = getMissValue(p);

    // how many channels in this aov?
    const int numChannels = aovNumChannels(p.mEntry.mAovSchemaId);

    // get geom key
    const int geomKey = p.mEntry.mPrimAttrKey;

    ispc::computePrimitiveAttribute(flags,
                                    geomKey,
                                    missValue,
                                    numChannels,
                                    p.mEntry.mSubsurface,
                                    (const ispc::std_vector *) &p.mEntry.mLabelIndices,
                                    (const ispc::std_vector *) &p.mEntry.mMaterialLabelIndices,
                                    (const ispc::std_vector *) &p.mEntry.mGeomLabelIndices,
                                    (intptr_t) &p.mIsect,
                                    &p.mBsdf,
                                    p.mPixelWeight,
                                    dest,
                                    p.mLanemask);
}

//----------------------------------------------------------------------------

HUD_VALIDATOR(MaterialAovs);

// prepare the parsing output structure for parsing
void
ParsedMaterialExpression::init(const std::string &expression)
{
    mProperty        = PROPERTY_UNKNOWN;
    mFresnelProperty = false;
    mSelector        = EMPTY;

    // parser expects the space for lobe labels
    // to be pre-allocated
    while (!mLabels.empty()) mLabels.pop();
    mLabels.push(std::vector<std::string>());

    mError           = "";
    mExpression      = expression;
    mNextLex         = 0;
}

// We support a very limited "light path" style syntax for material aov expressions.
AovSchemaId
MaterialAovs::parseExpression(const std::string &expression,
                              ComputeFn &computeFn,
                              ComputeFnv &computeFnv,
                              std::vector<std::string> &geomLabels,
                              std::vector<std::string> &materialLabels,
                              std::vector<std::string> &labels,
                              int &lobeFlags,
                              int &primAttrKey,
                              AovSchemaId &stateAovId,
                              bool &subsurface)
{
    // parse the expression
    ParsedMaterialExpression m;
    m.init(trim(expression));
    aovParseMaterialExpression(&m);

    // check for syntax errors
    if (!m.mError.empty()) {
        // some sort of parsing failure
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] Unable to parse material expression \"", expression, "\": ",
                      m.mError, " at \"", trim(expression).substr(0, m.mNextLex), "\".");
        return AOV_SCHEMA_ID_UNKNOWN;
    }

    // parsing looks good
    // there may still be semantic errors in the expression
    // we'll determine that by examining the labels, selectors, and properties.
    // A null compute computFn is how we communicate such errors.

    // Labels
    labels = m.mLabels.top();
    m.mLabels.pop();
    if (!m.mLabels.empty()) {
        materialLabels = m.mLabels.top();
        m.mLabels.pop();
        if (!m.mLabels.empty()) {
            geomLabels = m.mLabels.top();
        }
    }

    // sematic check: lobe labels cannot be applied to bsdf
    // properties
    if (!labels.empty()) {
        // disallow labels on bsdf properties
        if (m.mProperty == ParsedMaterialExpression::PROPERTY_EMISSION) {
            // this only is a bsdf property
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "lobe labels cannot be applied to the emission property: \"",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }
    }

    // Lobe flags
    // Convert from ParsedMaterialExpression flags to BsdfLobe::Type
    if (m.mSelector == ParsedMaterialExpression::EMPTY) {
        // user specified nothing in the string, this is equivalent to 'RTDGMSS.'
        lobeFlags = shading::BsdfLobe::ALL;
    } else if (m.mSelector == ParsedMaterialExpression::SUBSURFACE) {
        // user specified just 'SS.' this means to ignore all lobes and just
        // consider subsurface
        lobeFlags = shading::BsdfLobe::NONE;
    } else {
        // some mixed combo of symbols
        lobeFlags = shading::BsdfLobe::NONE;

        // first determine side categories
        // specifying neither R nor T in this case means select both sides
        if (m.mSelector & (ParsedMaterialExpression::REFLECTION |
                           ParsedMaterialExpression::TRANSMISSION)) {
            // user has selected sides
            if (m.mSelector & ParsedMaterialExpression::REFLECTION)   lobeFlags |= shading::BsdfLobe::REFLECTION;
            if (m.mSelector & ParsedMaterialExpression::TRANSMISSION) lobeFlags |= shading::BsdfLobe::TRANSMISSION;
        } else {
            // user did not select sides, so use both sides by default
            lobeFlags |= shading::BsdfLobe::ALL_SURFACE_SIDES;
        }

        // now determine lobe categories
        // selecting none of D G or M means to select all
        if (m.mSelector & (ParsedMaterialExpression::DIFFUSE |
                           ParsedMaterialExpression::GLOSSY  |
                           ParsedMaterialExpression::MIRROR)) {
            // user has selected lobe categories
            if (m.mSelector & ParsedMaterialExpression::DIFFUSE)  lobeFlags |= shading::BsdfLobe::DIFFUSE;
            if (m.mSelector & ParsedMaterialExpression::GLOSSY)   lobeFlags |= shading::BsdfLobe::GLOSSY;
            if (m.mSelector & ParsedMaterialExpression::MIRROR)   lobeFlags |= shading::BsdfLobe::MIRROR;
        } else {
            // user did not select categories, so use all categories by default
            lobeFlags |= shading::BsdfLobe::ALL_LOBES;
        }
    }

    // Subsurface flag
    subsurface = (m.mSelector & ParsedMaterialExpression::SUBSURFACE);

    // Consider subsurface active if the mselector string is empty
    subsurface |= (m.mSelector == ParsedMaterialExpression::EMPTY);

    AovSchemaId idBase = AOV_SCHEMA_ID_UNKNOWN;

    switch (m.mProperty) {
    case ParsedMaterialExpression::PROPERTY_ALBEDO:
        // semantic check: no albedo for fresnel
        if (m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "albedo is not a supported fresnel property: \"",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        } else {
            computeFn  = computeAlbedo;
            computeFnv = computeAlbedov;
        }

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_RGB;
        break;

    case ParsedMaterialExpression::PROPERTY_COLOR:
        if (m.mFresnelProperty) {
            computeFn  = computeFresnelColor;
            computeFnv = computeFresnelColorv;
        } else {
            computeFn  = computeColor;
            computeFnv = computeColorv;
        }

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_RGB;
        break;

    case ParsedMaterialExpression::PROPERTY_EMISSION:
        // semantic check: emission is a bsdf property
        if (m.mSelector != ParsedMaterialExpression::EMPTY || m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error(
                "[MCRT-RENDER] material expression semantic error: "
                "emission cannot have lobe, subsurface, or fresnel qualifiers: \"",
                expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }
        computeFn  = computeEmission;
        computeFnv = computeEmissionv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_RGB;
        break;

    case ParsedMaterialExpression::PROPERTY_FACTOR:
        // semantic check: "factor" is a fresnel property
        if (!m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "factor is a fresnel property: \"",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }

        computeFn  = computeFresnelFactor;
        computeFnv = computeFresnelFactorv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        break;

    case ParsedMaterialExpression::PROPERTY_NORMAL:
        // semantic check: normal can only be output from lobes or subsurface
        if (m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "'normal' property is not available for fresnel: ",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }

        computeFn  = computeNormal;
        computeFnv = computeNormalv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        break;

    case ParsedMaterialExpression::PROPERTY_RADIUS:
        // semantic check: the selector must be "SS." or empty
        // radius is not a fresnel property
        if ((m.mSelector != ParsedMaterialExpression::EMPTY && m.mSelector != ParsedMaterialExpression::SUBSURFACE) ||
            (m.mFresnelProperty)) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "'radius' property applies to subsurface only: ",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }

        computeFn = computeRadius;
        computeFnv = computeRadiusv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_RGB;
        break;

    case ParsedMaterialExpression::PROPERTY_ROUGHNESS:
        // semantic check: roughness is a bsdf lobe and fresnel property
        // in other words, if "SS" was specified, then fresnel. is also needed
        if (m.mSelector & ParsedMaterialExpression::SUBSURFACE && !m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "roughness can only be output from bsdf lobes or fresnels: \"",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }

        if (m.mFresnelProperty) {
            computeFn  = computeFresnelRoughness;
            computeFnv = computeFresnelRoughnessv;
        } else {
            computeFn  = computeRoughness;
            computeFnv = computeRoughnessv;
        }

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F;
        break;

    case ParsedMaterialExpression::PROPERTY_MATTE:
        if (m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "matte is not a supported fresnel property: \"",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }

        computeFn = computeMatte;
        computeFnv = computeMattev;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        break;

    case ParsedMaterialExpression::PROPERTY_PBR_VALIDITY:
        if (m.mFresnelProperty) {
            scene_rdl2::logging::Logger::error("[MCRT-RENDER] material expression semantic error: "
                          "pbr validity is not a supported fresnel property: \"",
                          expression, "\"");
            return AOV_SCHEMA_ID_UNKNOWN;
        }

        computeFn = computePbrValidity;
        computeFnv = computePbrValidityv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_RGB;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_P:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        stateAovId = AOV_SCHEMA_ID_STATE_P;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_N:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        stateAovId = AOV_SCHEMA_ID_STATE_N;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_NG:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        stateAovId = AOV_SCHEMA_ID_STATE_NG;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_ST:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F;
        stateAovId = AOV_SCHEMA_ID_STATE_ST;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DPDS:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        stateAovId = AOV_SCHEMA_ID_STATE_DPDS;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DPDT:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        stateAovId = AOV_SCHEMA_ID_STATE_DPDT;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DSDX:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        stateAovId = AOV_SCHEMA_ID_STATE_DSDX;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DSDY:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        stateAovId = AOV_SCHEMA_ID_STATE_DSDY;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DTDX:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        stateAovId = AOV_SCHEMA_ID_STATE_DTDX;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DTDY:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        stateAovId = AOV_SCHEMA_ID_STATE_DTDY;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_WP:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        stateAovId = AOV_SCHEMA_ID_STATE_WP;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_DEPTH:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        stateAovId = AOV_SCHEMA_ID_STATE_DEPTH;
        break;

    case ParsedMaterialExpression::PROPERTY_STATE_VARIABLE_MOTION:
        computeFn  = computeStateAov;
        computeFnv = computeStateAovv;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F;
        stateAovId = AOV_SCHEMA_ID_STATE_MOTION;
        break;

    case ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_FLOAT:
        computeFn  = computePrimitiveAttribute;
        computeFnv = computePrimitiveAttributev;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
        primAttrKey = TypedAttributeKey<float>(m.mPrimitiveAttribute).getIndex();
        break;

    case ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_VEC2:
        computeFn  = computePrimitiveAttribute;
        computeFnv = computePrimitiveAttributev;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F;
        primAttrKey = TypedAttributeKey<Vec2f>(m.mPrimitiveAttribute).getIndex();
        break;

    case ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_VEC3:
        computeFn  = computePrimitiveAttribute;
        computeFnv = computePrimitiveAttributev;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F;
        primAttrKey = TypedAttributeKey<Vec3f>(m.mPrimitiveAttribute).getIndex();
        break;

    case ParsedMaterialExpression::PROPERTY_PRIMITIVE_ATTRIBUTE_RGB:
        computeFn  = computePrimitiveAttribute;
        computeFnv = computePrimitiveAttributev;

        idBase = AOV_SCHEMA_ID_MATERIAL_AOV_RGB;
        primAttrKey = TypedAttributeKey<Color>(m.mPrimitiveAttribute).getIndex();
        break;

    default:
        MNRY_ASSERT(0 && "unhandled property type");

    }

    return idBase;
}

MaterialAovs::Entry::Entry(const std::string &name, int aovSchemaId,
                           ComputeFn computeFn, ComputeFnv computeFnv,
                           const std::vector<int> &geomLabelIndices,
                           const std::vector<int> &materialLabelIndices,
                           const std::vector<int> &labelIndices,
                           int lobeFlags,
                           AovFilter filter,
                           int primAttrKey,
                           AovSchemaId stateAovId,
                           AovSchemaId lpeSchemaId,
                           int lpeLabelId,
                           bool subsurface):
    mName(name), mAovSchemaId(aovSchemaId),
    mComputeFn(computeFn), mComputeFnv(computeFnv),
    mGeomLabelIndices(geomLabelIndices),
    mMaterialLabelIndices(materialLabelIndices),
    mLabelIndices(labelIndices),
    mLobeFlags(lobeFlags),
    mFilter(filter),
    mPrimAttrKey(primAttrKey),
    mStateAovId(stateAovId),
    mLpeSchemaId(lpeSchemaId),
    mLpeLabelId(lpeLabelId),
    mSubsurface(subsurface)
{
}

MaterialAovs::MaterialAovs()
{
}

int
MaterialAovs::getLabelIndex(const std::string &label) const
{
    auto result = std::find(mLabels.begin(), mLabels.end(), label);
    if (result == mLabels.end()) return -1; // not found

    return result - mLabels.begin();
}

int
MaterialAovs::getMaterialLabelIndex(const std::string &label) const
{
    auto result = std::find(mMaterialLabels.begin(), mMaterialLabels.end(), label);
    if (result == mMaterialLabels.end()) return -1; // not found

    return result - mMaterialLabels.begin();
}

int
MaterialAovs::getGeomLabelIndex(const std::string &label) const
{
    auto result = std::find(mGeomLabels.begin(), mGeomLabels.end(), label);
    if (result == mGeomLabels.end()) return -1; // not found

    return result - mGeomLabels.begin();
}

int
MaterialAovs::findEntry(const std::string &name, AovSchemaId lpeSchemaId) const
{
    // Assume mEntries is small enough that linear searches
    // at startup are not too onerous.
    for (auto &entry: mEntries) {
        if (entry.mName == name && entry.mLpeSchemaId == lpeSchemaId) {
            // found it
            return entry.mAovSchemaId;
        }
    }

    return AOV_SCHEMA_ID_UNKNOWN;
}

int
MaterialAovs::createEntry(const std::string &name,
                          AovFilter filter,
                          AovSchemaId lpeSchemaId,
                          int lpeLabelId,
                          AovSchemaId &stateAovId,
                          int &primAttrKey)
{
    // If we already have the entry, return it.
    int result = findEntry(name, lpeSchemaId);
    if (result != AOV_SCHEMA_ID_UNKNOWN) return result;

    // Didn't find it, need to create
    MNRY_ASSERT(mEntries.size() < AOV_MAX_RANGE_TYPE);

    ComputeFn computeFn = nullptr;
    ComputeFnv computeFnv = nullptr;
    std::vector<std::string> geomLabels;
    std::vector<std::string> materialLabels;
    std::vector<std::string> labels;
    int lobeFlags;
    bool subsurface;
    const AovSchemaId idBase = parseExpression(name, computeFn, computeFnv,
                                               geomLabels, materialLabels, labels,
                                               lobeFlags, primAttrKey, stateAovId,
                                               subsurface);

    // error - unknown expression type
    if (computeFn == nullptr || computeFnv == nullptr) {
        MNRY_ASSERT(idBase == AOV_SCHEMA_ID_UNKNOWN);
        return AOV_SCHEMA_ID_UNKNOWN;
    }

    // labels should be unique, convert to a list of indices
    std::vector<int> labelIndices;
    for (const auto &label: labels) {
        int li = getLabelIndex(label);
        if (li < 0) { // need to add it
            mLabels.push_back(label);
            li = mLabels.size() - 1;
        }
        labelIndices.push_back(li);
    }

    std::vector<int> materialLabelIndices;
    for (const auto &label: materialLabels) {
        int li = getMaterialLabelIndex(label);
        if (li < 0) { // need to add it
            mMaterialLabels.push_back(label);
            li = mMaterialLabels.size() - 1;
        }
        materialLabelIndices.push_back(li);
    }

    std::vector<int> geomLabelIndices;
    for (const auto &label: geomLabels) {
        int li = getGeomLabelIndex(label);
        if (li < 0) { // need to add it
            mGeomLabels.push_back(label);
            li = mGeomLabels.size() - 1;
        }
        geomLabelIndices.push_back(li);
    }

    mEntries.emplace_back(name, idBase + mEntries.size(), computeFn, computeFnv,
                          geomLabelIndices, materialLabelIndices, labelIndices,
                          lobeFlags, filter, primAttrKey, stateAovId, lpeSchemaId,
                          lpeLabelId, subsurface);

    return mEntries.back().mAovSchemaId;
}

void
MaterialAovs::computeScalar(pbr::TLState *pbrTls,
                            int aovSchemaId,
                            const LightAovs& lightAovs,
                            const shading::Intersection &isect,
                            const mcrt_common::RayDifferential &ray,
                            const Scene &scene,
                            const shading::Bsdf &bsdf,
                            const Color &ssAov,
                            const BsdfSampler *bSampler,
                            const BsdfSample *bsmps,
                            const BsdfSlice *bsdfSlice,
                            float pixelWeight,
                            int lpeStateId,
                            float *dest) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    MNRY_ASSERT(aovType(aovSchemaId) == AOV_TYPE_MATERIAL_AOV);
    unsigned indx = aovSchemaId % AOV_MAX_RANGE_TYPE;
    MNRY_ASSERT(indx < mEntries.size());
    const Entry &entry = mEntries[indx];
    const bool isPrimaryRay = ray.getDepth() == 0;
    if (entry.mLpeSchemaId != AOV_SCHEMA_ID_UNKNOWN) {
        // An LPE has been specified for this material AOV.  This is the actual code where
        // we apply the LPE.  This works the same way the "extra aovs" work with their LPEs.
        MNRY_ASSERT(entry.mLpeLabelId != -1);
        int stateId = lightAovs.materialAovEventTransition(pbrTls, lpeStateId, entry.mLpeLabelId);
        // No AOVs match, don't compute the material AOV
        if (stateId == -1) return;
    } else if (!isPrimaryRay) {
        // No LPE, this is a "regular" material AOV.  Skip if this isn't a primary ray.
        return;
    }
    MNRY_ASSERT(entry.mAovSchemaId == aovSchemaId);
    ComputeParams params { entry, isect, ray, scene, bsdf, ssAov, bSampler, bsmps, bsdfSlice, pixelWeight };
    entry.mComputeFn(params, dest);
}

uint32_t
MaterialAovs::computeVector(pbr::TLState *pbrTls,
                            int aovSchemaId,
                            const LightAovs& lightAovs,
                            const shading::Intersectionv &isect,
                            const mcrt_common::RayDifferentialv &ray,
                            const Scene &scene,
                            const shading::Bsdfv &bsdf,
                            const Colorv &ssAov,
                            const BsdfSamplerv *bSampler,
                            const BsdfSamplev *bsmps,
                            const BsdfSlicev *bsdfSlice,
                            const float *pixelWeight,
                            const int *lpeStateId,
                            const uint32_t *isPrimaryRay,
                            float *dest,
                            const uint32_t inputLanemask) const
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    MNRY_ASSERT(aovType(aovSchemaId) == AOV_TYPE_MATERIAL_AOV);

    unsigned indx = aovSchemaId % AOV_MAX_RANGE_TYPE;
    MNRY_ASSERT(indx < mEntries.size());

    const Entry &entry = mEntries[indx];
    MNRY_ASSERT(entry.mAovSchemaId == aovSchemaId);

    // Track the locally-modified lane mask because we need to refer to it later
    uint32_t localLanemask = inputLanemask;

    if (entry.mLpeSchemaId != AOV_SCHEMA_ID_UNKNOWN) {
        // An LPE has been specified for this material AOV.  We examine each lane individually and
        // then disable the lane mask for lanes that we don't want to compute the material AOV.
        MNRY_ASSERT(entry.mLpeLabelId != -1);
        for (unsigned int lane = 0; lane < VLEN; ++lane) {
            if (!(localLanemask & (1 << lane))) continue;
            int stateId = lightAovs.materialAovEventTransition(pbrTls, lpeStateId[lane], entry.mLpeLabelId);
            if (stateId == -1) {
                // No AOVs match, don't compute the material AOV.
                localLanemask &= ~(1 << lane);
            }
        }
    } else {
        // No LPE.  Skip if this isn't a primary ray.
        for (unsigned int lane = 0; lane < VLEN; ++lane) {
            if (!(localLanemask & (1 << lane))) continue;
            if (isPrimaryRay[lane] == 0) {
                // Not a primary ray; disable the lane mask for this lane.
                localLanemask &= ~(1 << lane);
            }
        }
    }

    if (localLanemask != 0) {
        ComputeParamsv params { entry, isect, ray, scene, bsdf, ssAov, bSampler, bsmps, bsdfSlice, pixelWeight, localLanemask };
        entry.mComputeFnv(params, dest);
    }

    // Return the locally-modified lane mask because we need to refer to it later
    return localLanemask;
}

void
aovSetMaterialAovs(pbr::TLState *pbrTls,
                   const AovSchema &aovSchema,
                   const LightAovs &lightAovs,
                   const MaterialAovs &materialAovs,
                   const shading::Intersection &isect,
                   const mcrt_common::RayDifferential &ray,
                   const Scene &scene,
                   const shading::Bsdf &bsdf,
                   const Color &ssAov,
                   const BsdfSampler *bSampler,
                   const BsdfSample *bsmps,
                   float pixelWeight,
                   int lpeStateId,
                   float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    for (const auto &entry: aovSchema) {

        if (entry.type() == AOV_TYPE_MATERIAL_AOV) {
            float weight = entry.filter() == AOV_FILTER_AVG ? pixelWeight : 1.0f;
            materialAovs.computeScalar(pbrTls, entry.id(), lightAovs, isect,
                                       ray, scene, bsdf,
                                       ssAov, bSampler, bsmps,
                                       nullptr, weight, lpeStateId, dest);
        }

        // onto the next one
        dest += entry.numChannels();
    }
}

void
aovSetMaterialAovs(pbr::TLState *pbrTls,
                   const AovSchema &aovSchema,
                   const LightAovs &lightAovs,
                   const MaterialAovs &materialAovs,
                   const shading::Intersection &isect,
                   const mcrt_common::RayDifferential &ray,
                   const Scene &scene,
                   const shading::Bsdf &bsdf,
                   const Color &ssAov,
                   const BsdfSlice *bsdfSlice,
                   float pixelWeight,
                   int lpeStateId,
                   float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    for (const auto &entry: aovSchema) {

        if (entry.type() == AOV_TYPE_MATERIAL_AOV) {
            float weight = entry.filter() == AOV_FILTER_AVG ? pixelWeight : 1.0f;
            materialAovs.computeScalar(pbrTls, entry.id(), lightAovs, isect,
                                       ray, scene, bsdf,
                                       ssAov, nullptr, nullptr,
                                       bsdfSlice, weight, lpeStateId, dest);
        }

        // onto the next one
        dest += entry.numChannels();
    }
}

void
CPP_aovSetMaterialAovs(pbr::TLState *pbrTls,
                       const AovSchema &aovSchema,
                       const LightAovs &lightAovs,
                       const MaterialAovs &materialAovs,
                       const shading::Intersectionv &isect,
                       const mcrt_common::RayDifferentialv &ray,
                       intptr_t scenePtr,
                       const shading::Bsdfv &bsdf,
                       const Colorv &ssAov,
                       const BsdfSamplerv *bSampler,
                       const BsdfSamplev *bsmps,
                       const BsdfSlicev *bsdfSlice,
                       const float pixelWeight[],
                       const uint32_t pixel[],
                       const uint32_t deepDataHandle[],
                       const int lpeStateId[],
                       const uint32_t isPrimaryRay[],
                       const uint32_t lanemask)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    const Scene &scene(*reinterpret_cast<const Scene *>(scenePtr));

    scene_rdl2::alloc::Arena *arena = pbrTls->mArena;
    SCOPED_MEM(arena);
    unsigned int numMaterialAovChannels = 0;
    unsigned int numMaterialAovEntries = 0;
    bool needsDepth = false;
    for (const auto &entry: aovSchema) {
        if (entry.type() == AOV_TYPE_MATERIAL_AOV) {
            numMaterialAovChannels += entry.numChannels();
            numMaterialAovEntries++;
            if (entry.stateAovId() == AOV_SCHEMA_ID_STATE_DEPTH) {
                needsDepth = true;
            }
        }
    }

    if (!numMaterialAovChannels) return;

    // If we need depth values, compute them now
    alignas(SIMD_MEMORY_ALIGNMENT) float depth[VLEN];
    if (needsDepth) {
        ispc::computeAndStoreDepthFromIsect((intptr_t) &isect,
                                            (intptr_t) &ray,
                                            (intptr_t) &scene,
                                            (intptr_t) getRender2Camera,
                                            &depth);
    }

    // Allocate a destination buffer large enough to hold results for all our material aovs across all lanes
    float *buffer = arena->allocArray<float>(numMaterialAovChannels * VLEN);
    memset(buffer, 0, numMaterialAovChannels * VLEN * sizeof(float));

    // Keep track of locally modified per-entry lane masks for the material aovs
    uint32_t *materialAovLanemasks = arena->allocArray<uint32_t>(numMaterialAovEntries);
    memset(materialAovLanemasks, 0, numMaterialAovEntries * sizeof(uint32_t));

#if (VLEN == 16u)
            const float onev[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
#elif (VLEN == 8u)
            const float onev[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
#else
            #error Unknown vector width
#endif

    // Compute aov results
    float *dest = buffer;
    int i = 0;
    for (const auto &entry: aovSchema) {
        if (entry.type() == AOV_TYPE_MATERIAL_AOV) {
            materialAovLanemasks[i++] = materialAovs.computeVector(pbrTls, entry.id(), lightAovs, isect, ray, scene,
                                                        bsdf, ssAov, bSampler, bsmps, bsdfSlice,
                                                        entry.filter() == AOV_FILTER_AVG ? pixelWeight : onev,
                                                        lpeStateId, isPrimaryRay, dest, lanemask);
            dest += entry.numChannels() * VLEN;
        }
    }

    // Bundling
    for (unsigned int lane = 0; lane < VLEN; ++lane) {
        if (!(lanemask & (1 << lane))) continue;
        // We need to call a version of addToBundledQueue() that passes the array of per-entry lane masks from above
        addToBundledQueue(pbrTls, aovSchema, depth,
                          buffer, lane, materialAovLanemasks, pixel,
                          deepDataHandle);
    }
}

//----------------------------------------------------------------------------
// Light Aovs
//----------------------------------------------------------------------------

HUD_VALIDATOR(LightAovs);

// template allows us to duck-type BsdfLobe/Bssrdf/VolumeSubsurface
template<typename LobeType>
int
LightAovs::computeScatterEventLabelId(const shading::Bsdf &bsdf, const LobeType *lobe)
{
    int result = lpe::sNoLabel; // no labels

    if (lobe && shading::aovLabelIsTransformed(lobe->getLabel())) {
        result = shading::aovDecodeLpeLabel(lobe->getLabel());
    } else {
        // no lobe, or a lobe with label = 0 (no label assigned)
        // will use the material label, if assigned
        MNRY_ASSERT(!lobe || lobe->getLabel() == 0);
        result = bsdf.getLpeMaterialLabelId();
    }

    return result;
}

// Bsdfv only supports bsdf lobes
int
LightAovs::computeScatterEventLabelIdVector(const shading::Bsdfv &bsdf, const shading::BsdfLobev *lobe)
{
    int result = lpe::sNoLabel; // no labels

    if (lobe && shading::aovLabelIsTransformed(shading::getLabel(*lobe))) {
        result = shading::aovDecodeLpeLabel(shading::getLabel(*lobe));
    } else {
        // no lobe, or a lobe with label = 0 (no label assigned)
        // will use the material label, if assigned
        MNRY_ASSERT(!lobe || shading::getLabel(*lobe) == 0);
        result = shading::getLpeMaterialLabelId(bsdf);
    }

    return result;
}

std::string
LightAovs::replaceLpeAliases(const std::string &lpe)
{
    std::string val = trim(lpe);
    if (val == "caustic") {
        // add in G?
        return "CD[S]+[<L.>O]";
    } else if (val == "diffuse") {
        return "CD[<L.>O]";
    } else if (val == "emission") {
        return "CO";
    } else if (val == "glossy") {
        return "CG[<L.>O]";
    } else if (val == "mirror") {
        return "CS[<L.>O]";
    } else if (val == "reflection") {
        return "C<RS>[DSG]+[<L.>O]";
    } else if (val == "translucent") {
        return "C<TD>[DSG]+[<L.>O]";
    } else if (val == "transmission") {
        return "C<TS>[DSG]+[<L.>O]";
    }

    return lpe;
}

static std::string
parseFlagsFromLpePrefix(const std::string& lpe, int& flag)
{
    flag = AovSchema::sLpePrefixNone;

    // We simply check for an unoccluded flag value
    const size_t semicolonPos = lpe.find_first_of(";");
    if (semicolonPos == std::string::npos) {
        // No semicolon found, so we can just return the LPE unmodified
        return lpe;
    }

    // Otherwise, we have a flag
    const std::string flagString = trim(lpe.substr(0, semicolonPos));

    if (flagString == "unoccluded") {
        flag |= AovSchema::sLpePrefixUnoccluded;
    } else {
        scene_rdl2::logging::Logger::warn("Unrecognized prefix found: \"", flagString, "\". Ignoring...");
    }

    // Return everything except for the semicolon value
    return lpe.substr(semicolonPos + 1);
}

LightAovs::LightAovs(const scene_rdl2::rdl2::Layer *layer):
    mNextLightAovSchemaId(AOV_SCHEMA_ID_LIGHT_AOV),
    mNextVisibilityAovSchemaId(AOV_SCHEMA_ID_VISIBILITY_AOV),
    mFinalized(false)
{
    // layer can be null if there are no render
    // output objects in the scene
    if (layer) {
        buildLabelSubstitutions(*layer);
    }
}

int
LightAovs::createEntry(const std::string &lpe, bool visibility, int &prefixFlags)
{
    MNRY_ASSERT(!mFinalized);
    if (mFinalized) return AOV_SCHEMA_ID_UNKNOWN;

    int result = AOV_SCHEMA_ID_UNKNOWN;

    std::string newLpe = parseFlagsFromLpePrefix(lpe, prefixFlags);
    newLpe = replaceLpeAliases(newLpe);
    newLpe = expandLpeLabels(newLpe);

    // Both light and visibility aovs use the same state machine
    // but they have different start points to their ids
    // which distinguishes a light aov from a visibility aov.
    if (visibility) {
        // If there are so many visibility aovs that the id moves into the
        // light aov range, then we do not create the entry.
        if (mNextVisibilityAovSchemaId >= AOV_SCHEMA_ID_LIGHT_AOV) {
            scene_rdl2::logging::Logger::error
                (scene_rdl2::util::buildString("There are too many visibility aovs "
                                   "already in use. Maximum allowed: ",
                                   std::to_string(AOV_MAX_RANGE_TYPE - 1)));
            return result;
        }
        int error = mLpeStateMachine.addExpression(newLpe, mNextVisibilityAovSchemaId);
        if (!error) {
            result = mNextVisibilityAovSchemaId;
            ++mNextVisibilityAovSchemaId;
        }

    } else {
        int error = mLpeStateMachine.addExpression(newLpe, mNextLightAovSchemaId);
        if (!error) {
            result = mNextLightAovSchemaId;
            ++mNextLightAovSchemaId;
        }
    }

    return result;
}

int
LightAovs::getLabelId(const std::string &label) const
{
    return mLpeStateMachine.getLabelId(label);
}

int
LightAovs::getMaterialLabelId(const std::string &label) const
{
    return mLpeStateMachine.getLabelId(label);
}

void
LightAovs::finalize()
{
    MNRY_ASSERT(!mFinalized);
    if (mFinalized) return;

    // only build the machine if we have entries
    if (hasEntries()) {
        mLpeStateMachine.build();
    }

    // get label ids for the background extra aovs
    for (int i = 0; i < BackgroundExtraAovs::sNum; ++i) {
        mBackgroundExtraAovs[i].mLabelId = getLabelId(mBackgroundExtraAovs[i].mLabel);
    }

    mFinalized = true;
}

int
LightAovs::cameraEventTransition(pbr::TLState *pbrTls) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    return mLpeStateMachine.transition(lpe::StateMachine::sInitialStateId,
                                       lpe::EVENT_TYPE_CAMERA,
                                       lpe::EVENT_SCATTERING_TYPE_NONE,
                                       lpe::sNoLabel);
}

int
LightAovs::scatterEventTransition(pbr::TLState *pbrTls, int lpeStateId, const shading::Bsdf &bsdf,
                                  const shading::BsdfLobe &lobe) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    const int lobeType = lobe.getType();

    // Construct an event and event scattering type based on the lobe type
    lpe::EventType ev = lpe::EVENT_TYPE_NONE;
    if (lobeType & shading::BsdfLobe::REFLECTION) ev = lpe::EVENT_TYPE_REFLECTION;
    else if (lobeType & shading::BsdfLobe::TRANSMISSION) ev = lpe::EVENT_TYPE_TRANSMISSION;

    lpe::EventScatteringType evs = lpe::EVENT_SCATTERING_TYPE_NONE;
    if (lobeType & shading::BsdfLobe::DIFFUSE) evs = lpe::EVENT_SCATTERING_TYPE_DIFFUSE;
    else if (lobeType & shading::BsdfLobe::GLOSSY) evs = lpe::EVENT_SCATTERING_TYPE_GLOSSY;
    else if (lobeType & shading::BsdfLobe::MIRROR) evs = lpe::EVENT_SCATTERING_TYPE_MIRROR;

    MNRY_ASSERT(ev != lpe::EVENT_TYPE_NONE);
    MNRY_ASSERT(evs != lpe::EVENT_SCATTERING_TYPE_NONE);

    // labels
    const int labelId = computeScatterEventLabelId(bsdf, &lobe);

    return mLpeStateMachine.transition(lpeStateId, ev, evs, labelId);
}

int
LightAovs::scatterEventTransitionVector(pbr::TLState *pbrTls, int lpeStateId, const shading::Bsdfv &bsdf,
                                        const shading::BsdfLobev &lobe) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    const int lobeType = shading::getType(lobe);

    // Construct an event and event scattering type based on the lobe type
    lpe::EventType ev = lpe::EVENT_TYPE_NONE;
    if (lobeType & shading::BsdfLobe::REFLECTION) ev = lpe::EVENT_TYPE_REFLECTION;
    else if (lobeType & shading::BsdfLobe::TRANSMISSION) ev = lpe::EVENT_TYPE_TRANSMISSION;

    lpe::EventScatteringType evs = lpe::EVENT_SCATTERING_TYPE_NONE;
    if (lobeType & shading::BsdfLobe::DIFFUSE) evs = lpe::EVENT_SCATTERING_TYPE_DIFFUSE;
    else if (lobeType & shading::BsdfLobe::GLOSSY) evs = lpe::EVENT_SCATTERING_TYPE_GLOSSY;
    else if (lobeType & shading::BsdfLobe::MIRROR) evs = lpe::EVENT_SCATTERING_TYPE_MIRROR;

    MNRY_ASSERT(ev != lpe::EVENT_TYPE_NONE);
    MNRY_ASSERT(evs != lpe::EVENT_SCATTERING_TYPE_NONE);

    // labels
    const int labelId = computeScatterEventLabelIdVector(bsdf, &lobe);

    return mLpeStateMachine.transition(lpeStateId, ev, evs, labelId);
}

int
LightAovs::straightEventTransition(pbr::TLState *pbrTls, int lpeStateId) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    // Construct an event and event scattering type based on the lobe type
    lpe::EventType ev = lpe::EVENT_TYPE_TRANSMISSION;
    lpe::EventScatteringType evs = lpe::EVENT_SCATTERING_TYPE_STRAIGHT;

    return mLpeStateMachine.transition(lpeStateId, ev, evs, lpe::sNoLabel);
}

int
LightAovs::lightEventTransition(pbr::TLState *pbrTls, int lpeStateId, const Light *light) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;
    if (!light) return -1;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    // labels
    const int labelId = light->getLabelId();
    return mLpeStateMachine.transition(lpeStateId, lpe::EVENT_TYPE_LIGHT,
                                       lpe::EVENT_SCATTERING_TYPE_NONE, labelId);
}

int
LightAovs::extraAovEventTransition(pbr::TLState *pbrTls, int lpeStateId, int labelId) const
{
    MNRY_ASSERT(mFinalized);
    MNRY_ASSERT(labelId >= 0);  // extra aov events require labels
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    return mLpeStateMachine.transition(lpeStateId,
                                       lpe::EVENT_TYPE_EXTRA,
                                       lpe::EVENT_SCATTERING_TYPE_NONE,
                                       labelId);
}

int
LightAovs::materialAovEventTransition(pbr::TLState *pbrTls, int lpeStateId, int labelId) const
{
    MNRY_ASSERT(mFinalized);
    MNRY_ASSERT(labelId >= 0);  // material aov events require labels
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    return mLpeStateMachine.transition(lpeStateId,
                                       lpe::EVENT_TYPE_MATERIAL,
                                       lpe::EVENT_SCATTERING_TYPE_NONE,
                                       labelId);
}

int
LightAovs::subsurfaceEventTransition(pbr::TLState *pbrTls, int lpeStateId, const shading::Bsdf &bsdf) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    const shading::Bssrdf* bssrdf = bsdf.getBssrdf();
    const shading::VolumeSubsurface* volumeSubsurface = bsdf.getVolumeSubsurface();

    MNRY_ASSERT(bssrdf != nullptr || volumeSubsurface != nullptr,
        "subsurface event requires Bssrdf or VolumeSubsurface");

    // labels
    const int labelId = (bssrdf != nullptr) ?
        computeScatterEventLabelId(bsdf, bssrdf) :
        computeScatterEventLabelId(bsdf, volumeSubsurface);

    // TODO: we don't really have a subsurface event in our machine
    // RIS uses <TD> to indicate the start of a subsurface event, we'll
    // do the same for now.
    return mLpeStateMachine.transition(lpeStateId,
        lpe::EVENT_TYPE_TRANSMISSION, lpe::EVENT_SCATTERING_TYPE_DIFFUSE, labelId);
}

int
LightAovs::emissionEventTransition(pbr::TLState *pbrTls, int lpeStateId, const shading::Bsdf &bsdf) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    // labels
    const int labelId = computeScatterEventLabelId<shading::BsdfLobe>(bsdf, nullptr);

    return mLpeStateMachine.transition(lpeStateId, lpe::EVENT_TYPE_EMISSION,
                                       lpe::EVENT_SCATTERING_TYPE_NONE,
                                       labelId);
}

int
LightAovs::emissionEventTransition(pbr::TLState *pbrTls, int lpeStateId, int volumeLabelId) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    return mLpeStateMachine.transition(lpeStateId, lpe::EVENT_TYPE_EMISSION,
                                       lpe::EVENT_SCATTERING_TYPE_NONE,
                                       volumeLabelId);
}

int
LightAovs::volumeEventTransition(pbr::TLState *pbrTls, int lpeStateId) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return -1;
    if (lpeStateId < 0) return lpeStateId;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    // Set scattering type to none is a bit confusing. This is a volume
    // scattering event but not a surface scattering event, thus
    // reflection/transmission both not make sense to set.
    // We basically follow the industry convention for this notation
    // (use V token alone to represent a volume scattering event)
    return mLpeStateMachine.transition(lpeStateId, lpe::EVENT_TYPE_VOLUME,
                                       lpe::EVENT_SCATTERING_TYPE_NONE,
                                       lpe::sNoLabel);
}

bool
LightAovs::isValid(pbr::TLState *pbrTls, int lpeStateId, int aovSchemaId) const
{
    MNRY_ASSERT(mFinalized);
    if (!hasEntries()) return false;
    if (lpeStateId < 0) return false;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);

    return mLpeStateMachine.isValid(lpeStateId, aovSchemaId);
}

std::string
LightAovs::expandLpeLabels(const std::string &lpe) const
{
    // Events that take place on bsdf and bssrdf lobes can
    // have a single, user defined label.  A label on a lobe is
    // built from 2 parts: a material_label and a lobe_label.  The
    // material_label is set by the end user as an attribute on the
    // material itself.  The lobe label is defined by the shader
    // writer and is set in code.  When only a lobe label is provided,
    // the label used to match in light path expressions is just 'lobe_label'.
    // simiarly, when only a material_label is provided, the lpe label is
    // 'material_label'.  When both are provided the matching label is
    // a concatenation of the two 'material_label.lobe_label'.
    //
    // Unfortunately, a consequence of this behavior is that specifying just
    // 'material_label' to match all lobes in a material or 'lobe_label'
    // to match all lobes with this label regardless of material does not work.
    // Once both a lobe label and material label exist, the concatendated label
    // must be used or else the event will not match.
    //
    // Consider a material that defines 3 lobe labels: clearcoat, diffuse, and specular.
    // Further assume the user assigns a material label, 'poppy'.  Then in order
    // to match all the lobes in material poppy, the lpe must be written as
    //    ['poppy.clearcoat''poppy.diffuse''poppy.specular']
    // this is verbose and error prone.  If the intention is to capture all of poppy,
    // then a shader writer just adding a single new label to the material will
    // break the expression.
    //
    // To solve this problem, this function searches the expression for labels that
    // match known material labels in the scene.  It then replaces this label with
    // the expanded set of labels.  In the previous example:
    //    'poppy' => ['poppy''poppy.clearcoat''poppy.diffuse''poppy.specular']
    // Note that we keep the original 'poppy' in the LPE  as well.  This is to
    // handle the case where some materials with the label 'poppy' define
    // lobes and some do not.  This could happen when instances of different
    // classes define the same material label.

    std::string newLpe = lpe;
    size_t tickOpen = newLpe.find_first_of('\'');
    while (tickOpen != std::string::npos) {
        size_t tickClose = newLpe.find_first_of('\'', tickOpen + 1);
        if (!tickClose) {
            // This indicates a parse error, it will be reported when
            // the lpe is parsed.  For now we just abandon the substitution
            // and return the unmodified input
            return lpe;
        }
        // get the label
        const std::string label = trim(newLpe.substr(tickOpen + 1, tickClose - tickOpen - 1));
        LabelSubstitutions::const_iterator itr = mLabelSubstitutions.find(label);
        if (itr != mLabelSubstitutions.end()) {
            newLpe.replace(tickOpen, tickClose - tickOpen + 1, itr->second);
            // search for next tick
            tickOpen = newLpe.find_first_of('\'', tickOpen + itr->second.length());
        } else {
            // label is fine as is
            tickOpen = newLpe.find_first_of('\'', tickClose + 1);
        }
    }

    // DEBUG
    if (newLpe != lpe) {
        scene_rdl2::logging::Logger::info("[MCRT-RENDER] LPE label expansion: ", lpe, " ==> ", newLpe);
    }

    return newLpe;
}

void
LightAovs::buildLabelSubstitutions(const scene_rdl2::rdl2::Layer &layer)
{
    // build an unordered map that maps material labels and lobe
    // labels to strings that match all possible material_label.lobe_label
    // combinations.  For example:
    //    mat0 -> ['mat0''mat0.clearcoat''mat0.diffuse''mat0.specular']
    mLabelSubstitutions.clear();

    // FIXME: no direct way to access the materials from a const Layer
    // need to lookup the secret "surface shaders" attribute.
    // this call will throw an exception, which we don't catch if
    // "surface shaders" does not exist as an attribute
    const scene_rdl2::rdl2::SceneObjectVector &materialVector = layer.get<scene_rdl2::rdl2::SceneObjectVector>(
        "surface_shaders");
    std::unordered_set<const scene_rdl2::rdl2::Material *> materials;
    for (const scene_rdl2::rdl2::SceneObject *s : materialVector) {
        if (!s) {
            continue;
        }
        scene_rdl2::rdl2::ConstSceneObjectSet b;
        s->getBindingTransitiveClosure(b);
        for (const scene_rdl2::rdl2::SceneObject * obj : b) {
            if (obj && obj->isA<scene_rdl2::rdl2::Material>()) {
                materials.insert(obj->asA<scene_rdl2::rdl2::Material>());
            }
        }
    }

    // all possible lobe labels for a material label
    typedef std::unordered_map<std::string, std::unordered_set<std::string>> MatLabel2LobeLabels;
    MatLabel2LobeLabels matLabel2LobeLabels;

    // all possible material labels for a lobe label
    typedef std::unordered_map<std::string, std::unordered_set<std::string>> LobeLabels2MatLabels;
    LobeLabels2MatLabels lobeLabels2MatLabels;

    // collect the needed mappings
    for (const scene_rdl2::rdl2::Material *mat : materials) {
        // get the material label and lobe labels for this material
        const scene_rdl2::rdl2::String &matLabel = mat->get(scene_rdl2::rdl2::Material::sLabel);
        const char * const *lobeLabels = mat->getSceneClass().getDataPtr<const char *>("labels");
        if (lobeLabels && !matLabel.empty()) {
            for (int i = 0; lobeLabels[i] != nullptr; ++i) {
                const std::string lobeLabel = lobeLabels[i];
                matLabel2LobeLabels[matLabel].insert(lobeLabel);
                lobeLabels2MatLabels[lobeLabel].insert(matLabel);
            }
        }
    }

    // insert the material label substitutions
    for (const auto &e : matLabel2LobeLabels) {
        const std::string matLabel = e.first;
        const std::unordered_set<std::string> &lobeLabels = e.second;
        std::string target = "['" + matLabel + "'";
        for (const std::string &lobeLabel : lobeLabels) {
            target += "'" + matLabel + "." + lobeLabel + "'";
        }
        target += "]";
        mLabelSubstitutions.insert({ matLabel, target });
    }

    // insert the lobe label substitutions
    for (const auto &e : lobeLabels2MatLabels) {
        const std::string lobeLabel = e.first;
        // it is possible that a user may create a material label that matches
        // a lobe label.  in that case, we favor the material label substitutions
        // over the lobe label substitutions
        if (mLabelSubstitutions.find(lobeLabel) == mLabelSubstitutions.end()) {
            const std::unordered_set<std::string> &matLabels = e.second;
            std::string target = "['" + lobeLabel + "'";
            for (const std::string &matLabel : matLabels) {
                target += "'" + matLabel + "." + lobeLabel + "'";
            }
            target += "]";
            mLabelSubstitutions.insert({ lobeLabel, target });
        }
    }

    // DEBUG
    // uncomment to print the substituion tabel
    // for (const auto &e : mLabelSubstitutions) {
    //    std::cerr << "labelSubstitution: " << '\'' << e.first << '\'' << " ==> " << e.second << '\n';
    // }
}

// Accumulates two values into separate aovs depending on whether or not they match with a certain flag or not.
template<AovType type, typename VALUE>
bool
aovAccumLpeAovs(pbr::TLState *pbrTls,
                const AovSchema &aovSchema,
                const LightAovs &lightAovs,
                const VALUE &matchValue,          // the value used for avg filters if aov matches prefixFlags
                const VALUE &matchSampleValue,    // the value used for sum, max and min filters if aov matches 
                                                  // prefixFlags
                const VALUE *nonMatchValue,       // if null, the value does not exist and will not be accumulated
                const VALUE *nonMatchSampleValue, // if null, the sample value does not exist and will not be accumulated
                int prefixFlags,
                int lpeStateId,
                bool lpePassthrough,              // disregard lpe when adding samples
                float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    bool success = false;
    for (const auto &entry: aovSchema) {
        if (entry.type() == type) {
            if (lpePassthrough || lightAovs.isValid(pbrTls, lpeStateId, entry.id())) {
                // Make sure to keep (== prefixFlags) as this statement should always hold true when prefixFlags == 0.
                const bool aovMatchPrefixFlags = (entry.lpePrefixFlags() & prefixFlags) == prefixFlags;
                if (aovMatchPrefixFlags || (nonMatchValue && nonMatchSampleValue)) {
                    const VALUE &value = aovMatchPrefixFlags ? matchValue : *nonMatchValue;
                    const VALUE &sampleValue = aovMatchPrefixFlags ? matchSampleValue : *nonMatchSampleValue;

                    for (size_t i = 0; i < entry.numChannels(); ++i) {
                        switch (entry.filter()) {
                        // Only "extra aovs" support non-avg math filters.  For extra aovs,
                        // we are computing and accumulating results at potentially every
                        // path vertex.  If the filter is "avg" then we just accumulate the value
                        // normally.  This basically just means to add the value (which should
                        // already take the path throughput into account) to the frame buffer.
                        // For min, max, and sum we want to use the raw sample value.  With min and
                        // max we need to check against the current value in dest and replace
                        // if appropriate.
                        case AOV_FILTER_SUM:
                            *(dest + i) += sampleValue[i];
                            break;
                        case AOV_FILTER_MAX:
                            *(dest + i) = (*(dest + i) < sampleValue[i]) ? sampleValue[i] : *(dest + i);
                            break;
                        case AOV_FILTER_MIN:
                            *(dest + i) = (*(dest + i) < sampleValue[i]) ? *(dest + i) : sampleValue[i];
                            break;
                        case AOV_FILTER_AVG:
                        default:
                            *(dest + i) += value[i];
                            break;
                        }
                    }
                    success = true;
                }
            }
        }

        // onto the next one
        dest += entry.numChannels();
    }

    return success;
}

bool
aovAccumLightAovs(pbr::TLState *pbrTls,
                  const AovSchema &aovSchema,
                  const LightAovs &lightAovs,
                  const scene_rdl2::math::Color &matchValue,
                  const scene_rdl2::math::Color *nonMatchValue,
                  int prefixFlags,
                  int lpeStateId,
                  float *dest)
{
    return aovAccumLpeAovs<AOV_TYPE_LIGHT_AOV, Color>(pbrTls, aovSchema,
        lightAovs, matchValue, matchValue, nonMatchValue, nonMatchValue, prefixFlags, 
        lpeStateId, /* lpePassthrough */ false, dest);
}

bool
aovAccumVisibilityAovs(pbr::TLState *pbrTls,
                       const AovSchema &aovSchema,
                       const LightAovs &lightAovs,
                       const Vec2f &value,
                       int lpeStateId,
                       float *dest)
{
    return aovAccumLpeAovs<AOV_TYPE_VISIBILITY_AOV, Vec2f>(pbrTls, aovSchema,
        lightAovs, value, value, /* nonMatchValue = */ nullptr, /* nonMatchSampleValue = */nullptr, 
        AovSchema::sLpePrefixNone, lpeStateId, /* lpePassthrough */ false, dest);
}

/// This function is a wrapper around aovAccumLpeAovs -- it adds a specified number of "misses" to the visibility
/// aov, disregarding the lpe
bool
aovAccumVisibilityAttempts(pbr::TLState *pbrTls,
                           const AovSchema &aovSchema,
                           const LightAovs &lightAovs,
                           const float value,
                           float *dest)
{
    scene_rdl2::math::Vec2f vecValue(0.f, value);
    return aovAccumLpeAovs<AOV_TYPE_VISIBILITY_AOV, Vec2f>(pbrTls, aovSchema,
        lightAovs, vecValue, vecValue, /* nonMatchValue = */ nullptr, /* nonMatchSampleValue = */nullptr, 
        AovSchema::sLpePrefixNone, /* lpeStateId */ -1, /* lpePassthrough */ true, dest);
}

template<AovType type, typename VALUE>
bool
aovAccumLpeAovsBundled(pbr::TLState *pbrTls,
                       const AovSchema &aovSchema,
                       const LightAovs &lightAovs,
                       const VALUE &matchValue,            // used for avg filters
                       const VALUE &matchSampleValue,      // used for sum, max, and min filters
                       const VALUE *nonMatchValue,         // if null, the value does not exist and will not be
                                                           // accumulated
                       const VALUE *nonMatchSampleValue,   // if null, the sample value does not exist and will not be
                                                           // accumulated
                       int prefixFlags,
                       int lpeStateId,
                       uint32_t pixel,
                       uint32_t deepDataHandle,
                       bool lpePassthrough)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    bool success = false;
    // Ideally, this function would make use of addToBundledQueue, but for
    // performance reasons, we check the lpeStateId and queue passing results
    // immediately, without the need to create a temporary aov buffer.
    // Sorry about the duplication.
    unsigned aovIdx = 0;
    unsigned aov = 0;

    BundledAov bundledAov(pixel, pbr::nullHandle);
    for (const auto &entry: aovSchema) {

        if (entry.type() == type) {
            if (lpePassthrough || lightAovs.isValid(pbrTls, lpeStateId, entry.id())) {
                // Make sure to keep (== prefixFlags) as this statement should always hold true when prefixFlags == 0.
                const bool aovMatchPrefixFlags = (entry.lpePrefixFlags() & prefixFlags) == prefixFlags;
                if (aovMatchPrefixFlags || (nonMatchValue && nonMatchSampleValue)) {
                    const VALUE &value = aovMatchPrefixFlags ? matchValue : *nonMatchValue;
                    const VALUE &sampleValue = aovMatchPrefixFlags ? matchSampleValue : *nonMatchSampleValue;

                    // need to process each color channel, we only
                    // send non-zero values
                    for (unsigned c = 0; c < entry.numChannels(); ++c ) {
                        switch (entry.filter()) {
                        // Only "extra aovs" support non-avg math filters.  For extra aovs,
                        // we are computing and accumulating results at potentially every
                        // path vertex.  If the filter is "avg" then we just accumulate the value
                        // normally.  This basically just means to add the value (which should
                        // already take the path throughput into account) to the frame buffer.
                        // For min, max, and sum we want to use the raw sample value.  As an
                        // optimization, we only queue meaningful values based on the filter.
                        // For avg and sum this means non-zero values.  For min and max this
                        // means non +/-inf to which the frame buffer is already cleared.
                        case AOV_FILTER_SUM:
                            if (sampleValue[c] != 0.f) {
                                bundledAov.setAov(aov++, sampleValue[c], aovIdx + c);
                            }
                            break;
                        case AOV_FILTER_MAX:
                        case AOV_FILTER_MIN:
                            if (scene_rdl2::math::isfinite(sampleValue[c])) {
                                bundledAov.setAov(aov++, sampleValue[c], aovIdx + c);
                            }
                            break;
                        case AOV_FILTER_AVG:
                        default:
                            if (value[c] != 0.f) {
                                bundledAov.setAov(aov++, value[c], aovIdx + c);
                            }
                            break;
                        }
                        if (aov == BundledAov::MAX_AOV) {
                            bundledAov.mDeepDataHandle = pbrTls->acquireDeepData(deepDataHandle);
                            pbrTls->addAovQueueEntries(1, &bundledAov);
                            bundledAov.init(pixel, pbr::nullHandle);
                            aov = 0;
                        }
                    }
                    success = true;
                }
            }
        }

        // advance aovIdx
        aovIdx += entry.numChannels();
    }

    // queue any remainders
    if (aov > 0) {
        bundledAov.mDeepDataHandle = pbrTls->acquireDeepData(deepDataHandle);
        pbrTls->addAovQueueEntries(1, &bundledAov);
    }

    return success;
}

bool
aovAccumLightAovsBundled(pbr::TLState *pbrTls,
                         const AovSchema &aovSchema,
                         const LightAovs &lightAovs,
                         const Color &matchValue,
                         const Color *nonMatchValue,
                         int prefixFlags,
                         int lpeStateId,
                         uint32_t pixel,
                         uint32_t deepDataHandle)
{
    return aovAccumLpeAovsBundled<AOV_TYPE_LIGHT_AOV, Color>(pbrTls, aovSchema,
        lightAovs, matchValue, matchValue, nonMatchValue, nonMatchValue, prefixFlags,
        lpeStateId, pixel, deepDataHandle, /* lpePassthrough */ false);
}

bool
aovAccumVisibilityAovsBundled(pbr::TLState *pbrTls,
                              const AovSchema &aovSchema,
                              const LightAovs &lightAovs,
                              const Vec2f &value,
                              int lpeStateId,
                              uint32_t pixel,
                              uint32_t deepDataHandle,
                              bool lpePassthrough)
{
    return aovAccumLpeAovsBundled<AOV_TYPE_VISIBILITY_AOV, Vec2f>(
        pbrTls, aovSchema, lightAovs, value, value, /* nonMatchValue = */ nullptr,
        /* nonMatchSampleValue = */ nullptr, AovSchema::sLpePrefixNone, lpeStateId, pixel, deepDataHandle, 
        lpePassthrough);
}

extern "C" bool
CPP_aovAccumVisibilityAovsBundled(pbr::TLState *pbrTls,
                                  const AovSchema &aovSchema,
                                  const LightAovs &lightAovs,
                                  const Vec2f &value,
                                  int lpeStateId,
                                  uint32_t pixel,
                                  uint32_t deepDataHandle,
                                  bool lpePassthrough)
{
    return aovAccumVisibilityAovsBundled(pbrTls, aovSchema, lightAovs, value, lpeStateId, pixel, 
                                         deepDataHandle, lpePassthrough);
}

/// This function is a wrapper around aovAccumLpeAovs -- it adds a specified number of "misses" to the visibility
/// aov, disregarding the lpe
bool aovAccumVisibilityAttemptsBundled(pbr::TLState *pbrTls,
                                       const AovSchema &aovSchema,
                                       const LightAovs &lightAovs,
                                       int attempts,
                                       uint32_t pixel,
                                       uint32_t deepDataHandle)
{
    scene_rdl2::math::Vec2f value(0.f, attempts);
    return aovAccumLpeAovsBundled<AOV_TYPE_VISIBILITY_AOV, Vec2f>(
        pbrTls, aovSchema, lightAovs, value, value, /* nonMatchValue = */ nullptr,
        /* nonMatchSampleValue = */ nullptr, AovSchema::sLpePrefixNone, /* lpeStateId */-1, pixel, 
        deepDataHandle, /* lpePassthrough */ true);
}

// Computes and accumulates extra aovs
void
aovAccumExtraAovs(pbr::TLState *pbrTls,
                  const FrameState &fs,
                  const PathVertex &pv,
                  const shading::Intersection &isect,
                  const scene_rdl2::rdl2::Material *mat,
                  float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    if (pv.lpeStateId == -1) return;
    const shading::Material &ext = mat->get<shading::Material>();
    const std::vector<shading::Material::ExtraAov> &extraAovs = ext.getExtraAovs();
    if (extraAovs.empty()) return;
    const LightAovs &lightAovs = *fs.mLightAovs;
    if (!lightAovs.hasEntries()) return;
    const AovSchema &aovSchema = *fs.mAovSchema;
    shading::TLState *shadingTls = pbrTls->mTopLevelTls->mShadingTls.get();
    const shading::State state(&isect);
    for (const shading::Material::ExtraAov &ea : extraAovs) {
        const int labelId = ea.mLabelId;
        MNRY_ASSERT(labelId != -1);
        const scene_rdl2::rdl2::Map *map = ea.mMap;
        int lpeStateId = lightAovs.extraAovEventTransition(pbrTls, pv.lpeStateId, labelId);
        if (lpeStateId == -1) continue; // no matches
        // at least one aov matches, compute the result
        scene_rdl2::math::Color value;
        map->sample(shadingTls, state, &value);
        // Now stuff the value in the right aov dest slots.
        // The assert is to verify that we did not waste time
        // computing this aov, which should not be the case with a >= 0
        // lpeStateId.
        MNRY_VERIFY((aovAccumLpeAovs<AOV_TYPE_LIGHT_AOV, Color>(pbrTls,
                                                               aovSchema,
                                                               lightAovs,
                                                               value * pv.pathThroughput,
                                                               value,
                                                               /* nonMatchValue = */ nullptr,
                                                               /* nonMatchSampleValue = */ nullptr,
                                                               AovSchema::sLpePrefixNone,
                                                               lpeStateId,
                                                               /* lpePassthrough */ false,
                                                               dest)));
    }
}

// Computes and accumulates extra aovs
void
aovAccumExtraAovsBundled(pbr::TLState *pbrTls,
                         const FrameState &fs,
                         RayState const * const *rayStates,
                         const float *presences,
                         const shading::Intersectionv *isectvs,
                         const scene_rdl2::rdl2::Material *mat,
                         unsigned int numRays)
{
    // rayStates are AOS, isectvs are AOSOA.
    const unsigned int numBlocks = (numRays + VLEN_MASK) / VLEN;

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    const shading::Material &ext = mat->get<shading::Material>();
    const std::vector<shading::Material::ExtraAov> &extraAovs = ext.getExtraAovs();
    if (extraAovs.empty()) return;
    const LightAovs &lightAovs = *fs.mLightAovs;
    if (!lightAovs.hasEntries()) return;
    const AovSchema &aovSchema = *fs.mAovSchema;
    shading::TLState *shadingTls = pbrTls->mTopLevelTls->mShadingTls.get();
    // We need to setup attribute offsets so maps that depend on primitive
    // attributes will work.  This matches what happens in shadev. It is
    // unfortunate that we assume all map shader evaluations are associated
    // with a root material shader.
    shadingTls->getAttributeOffsetsFromRootShader(*mat);
    for (const shading::Material::ExtraAov &ea : extraAovs) {
        const int labelId = ea.mLabelId;
        MNRY_ASSERT(labelId != -1);
        const scene_rdl2::rdl2::Map *map = ea.mMap;
        for (unsigned int block = 0; block < numBlocks; ++block) {
            int lpeStateIds[VLEN];
            bool hasValidAov = false;
            std::fill(&lpeStateIds[0], &lpeStateIds[VLEN - 1], -1);
            for (unsigned int ray = block * VLEN; ray < (block + 1) * VLEN && ray < numRays; ++ray) {
                const PathVertex &pv = rayStates[ray]->mPathVertex;
                const unsigned int idx = ray - block * VLEN;
                lpeStateIds[idx] = lightAovs.extraAovEventTransition(pbrTls, pv.lpeStateId, labelId);
                hasValidAov |= (lpeStateIds[idx] >= 0);
            }
            if (!hasValidAov) continue; // no matches
            // at least one aov matches, compute the result
            shading::Colorv valuev;
            shading::samplev(map, shadingTls, (const shading::Statev *)(&isectvs[block]), &valuev);
            for (unsigned int ray = block * VLEN; ray < (block + 1) * VLEN && ray < numRays; ++ray) {
                const unsigned int idx = ray - block * VLEN;
                if (lpeStateIds[idx] >= 0) {
                    const PathVertex &pv = rayStates[ray]->mPathVertex;
                    const uint32_t pixel = rayStates[ray]->mSubpixel.mPixel;
                    const uint32_t deepDataHandle = rayStates[ray]->mDeepDataHandle;
                    Color value(valuev.r[idx], valuev.g[idx], valuev.b[idx]);
                    // Now stuff the value in the right aov dest slots.
                    // The assert is to verify that we did not waste time
                    // computing this aov, which should not be the case with a >= 0
                    // lpeStateId.
                    // Unlike scalar mode, we need to multiply the presence in because
                    // it has not been multiplied into the path throughput yet.
                    MNRY_VERIFY((aovAccumLpeAovsBundled<AOV_TYPE_LIGHT_AOV, Color>(pbrTls,
                                                                                  aovSchema,
                                                                                  lightAovs,
                                                                                  value * pv.pathThroughput * presences[ray],
                                                                                  value,
                                                                                  /* nonMatchValue = */ nullptr,
                                                                                  /* nonMatchSampleValue = */ nullptr,
                                                                                  AovSchema::sLpePrefixNone,
                                                                                  lpeStateIds[idx],
                                                                                  pixel,
                                                                                  deepDataHandle,
                                                                                  /* lpePassthrough */ false)));
                }
            }
        }
    }
    shadingTls->clearAttributeOffsets();
}

// Accumultes post scatter extra aovs
void
aovAccumPostScatterExtraAovs(pbr::TLState *pbrTls,
                             const FrameState &fs,
                             const PathVertex &pv,
                             const Bsdf &bsdf,
                             float *dest)
{
    const Bsdf::ExtraAovs &extraAovs = bsdf.getPostScatterExtraAovs();
    const int numExtraAovs = extraAovs.mNum;
    if (numExtraAovs > 0 && pv.lpeStateId != -1) {
        const LightAovs &lightAovs = *fs.mLightAovs;
        const AovSchema &aovSchema = *fs.mAovSchema;
        const int *labelIds = extraAovs.mLabelIds;
        const Color *colors = extraAovs.mColors;
        for (int i = 0; i < numExtraAovs; ++i) {
            const Color value = colors[i];
            const int lpeStateId = lightAovs.extraAovEventTransition(pbrTls, pv.lpeStateId, labelIds[i]);
            if (lpeStateId != -1) {
                MNRY_VERIFY((aovAccumLpeAovs<AOV_TYPE_LIGHT_AOV, Color>(pbrTls,
                                                                       aovSchema,
                                                                       lightAovs,
                                                                       value * pv.pathThroughput,
                                                                       value,
                                                                       /* nonMatchValue = */ nullptr,
                                                                       /* nonMatchSampleValue = */ nullptr,
                                                                       AovSchema::sLpePrefixNone,
                                                                       lpeStateId,
                                                                       /* lpePassthrough */ false,
                                                                       dest)));
            }
        }
    }
}

// Accumulates background extra aovs
// Since background events have no intersection point or material, all
// we can do is emit a color.  An example use case is emit white to create
// a transparency aov.
void
aovAccumBackgroundExtraAovs(pbr::TLState *pbrTls,
                            const FrameState &fs,
                            const PathVertex &pv,
                            float *dest)
{
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    if (pv.lpeStateId == -1) return;
    const LightAovs &lightAovs = *fs.mLightAovs;
    if (!lightAovs.hasEntries()) return;

    for (int i = 0; i < LightAovs::BackgroundExtraAovs::sNum; ++i) {
        const int bgLabelId = lightAovs.getBackgroundExtraAovs()[i].mLabelId;
        if (bgLabelId == -1) continue;
        const int lpeStateId = lightAovs.extraAovEventTransition(pbrTls, pv.lpeStateId, bgLabelId);
        if (lpeStateId == -1) continue;

        const AovSchema &aovSchema = *fs.mAovSchema;
        const scene_rdl2::math::Color &value = lightAovs.getBackgroundExtraAovs()[i].mColor;
        MNRY_VERIFY((aovAccumLpeAovs<AOV_TYPE_LIGHT_AOV, Color>(pbrTls,
                                                               aovSchema,
                                                               lightAovs,
                                                               value * pv.pathThroughput,
                                                               value,
                                                               /* nonMatchValue = */ nullptr,
                                                               /* nonMatchSampleValue = */ nullptr,
                                                               AovSchema::sLpePrefixNone,
                                                               lpeStateId,
                                                               /* lpePassthrough */ false,
                                                               dest)));
    }
}

void
aovAccumBackgroundExtraAovsBundled(pbr::TLState *pbrTls,
                                   const FrameState &fs,
                                   const RayState *rs)
{

    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_AOVS);
    const PathVertex &pv = rs->mPathVertex;
    if (pv.lpeStateId == -1) return;
    const LightAovs &lightAovs = *fs.mLightAovs;
    if (!lightAovs.hasEntries()) return;

    for (int i = 0; i < LightAovs::BackgroundExtraAovs::sNum; ++i) {
        const int bgLabelId = lightAovs.getBackgroundExtraAovs()[i].mLabelId;
        if (bgLabelId == -1) continue;
        const int lpeStateId = lightAovs.extraAovEventTransition(pbrTls, pv.lpeStateId, bgLabelId);
        if (lpeStateId == -1) continue;

        const uint32_t pixel = rs->mSubpixel.mPixel;
        const uint32_t deepDataHandle = rs->mDeepDataHandle;
        scene_rdl2::math::Color pt = pv.pathThroughput;
        if (rs->mVolHit) {
            // Volume transparency needs to be taken into account.  Unlike the scalar code,
            // the RayHandler code will not have multiplied the volume transmittance into
            // the pathThroughput.  So we need to do that in this function as well.
            // Technically, we should do this:
            //     VolumeTransmittance vt;
            //     vt.reset();
            //     vt.update(rs.mVolTr, rs.mVolTh);
            //     pt *= vt.transmittance();
            // But we already have the results of update() stored in the ray state,
            // so we just mimic the implementation of vt.transmittance() here
            pt *= rs->mVolTr * rs->mVolTh;
        }

        const AovSchema &aovSchema = *fs.mAovSchema;
        const scene_rdl2::math::Color &value = lightAovs.getBackgroundExtraAovs()[i].mColor;
        MNRY_VERIFY((aovAccumLpeAovsBundled<AOV_TYPE_LIGHT_AOV, Color>(pbrTls,
                                                                      aovSchema,
                                                                      lightAovs,
                                                                      value * pt,
                                                                      value,
                                                                      /* nonMatchValue = */ nullptr,
                                                                      /* nonMatchSampleValue = */ nullptr,
                                                                      AovSchema::sLpePrefixNone,
                                                                      lpeStateId,
                                                                      pixel,
                                                                      deepDataHandle,
                                                                      /* lpePassthrough */ false)));
    }
}

// keep in sync with declaration in Aov.ispc
extern "C" void
CPP_aovAccumLightAovs(pbr::TLState *pbrTls,
                      const AovSchema &aovSchema,
                      const LightAovs &lightAovs,
                      const Color &value,
                      int lpeStateId,
                      uint32_t pixel,
                      uint32_t deepDataHandle)
{
    // TODO: fix accumulator parameter.
    // CPP_aovAccumLightAovs doesn't have to take into account flags, so we won't include them.
    aovAccumLightAovsBundled(pbrTls, aovSchema, lightAovs, value, nullptr, 
                             AovSchema::sLpePrefixNone, lpeStateId, pixel, deepDataHandle);
}

extern "C" void
CPP_aovAccumPostScatterExtraAov(pbr::TLState *pbrTls,
                                const AovSchema &aovSchema,
                                const LightAovs &lightAovs,
                                const Color &value,
                                const Color &sampleValue,
                                int pvStateId,
                                int labelId,
                                uint32_t pixel,
                                uint32_t deepDataHandle)
{
    MNRY_ASSERT(pvStateId >= 0);
    MNRY_ASSERT(labelId >= 0);
    int lpeStateId = lightAovs.extraAovEventTransition(pbrTls, pvStateId, labelId);
    if (lpeStateId != -1) {
        MNRY_VERIFY((aovAccumLpeAovsBundled<AOV_TYPE_LIGHT_AOV, Color>(pbrTls,
                                                                      aovSchema,
                                                                      lightAovs,
                                                                      value,
                                                                      sampleValue,
                                                                      /* nonMatchValue = */ nullptr,
                                                                      /* nonMatchSampleValue = */ nullptr,
                                                                      AovSchema::sLpePrefixNone,
                                                                      lpeStateId,
                                                                      pixel,
                                                                      deepDataHandle,
                                                                      /* lpePassthrough */ false)));
    }
}
 
} // namespace pbr
} // namespace moonray
