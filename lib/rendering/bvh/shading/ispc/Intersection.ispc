// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Intersection.isph"
#include <moonray/rendering/bvh/shading/ispc/AttributeTable.isph>

inline uniform uint32_t
intersectionCrc(uniform bool verbose)
{
    INTERSECTION_VALIDATION(VLEN);
}

export uniform uint32_t
Intersection_hvdValidation(uniform bool verbose)
{
    return intersectionCrc(verbose);
}

//----------------------------------------------------------------------------

varying Vec3f
Intersection_adaptNormal(const varying Intersection &isect, const varying Vec3f &Ns)
{
    // Ns is the result of a normal mapping operation.
    // The result may or may not be physically plausible from
    // our particular view point.  This function detects this
    // case and minimally bends the shading normal back towards
    // the physically plausible geometric normal.  This unfortunately
    // breaks bi-directional algorithms.  See "Microfacet-based
    // Normal Mapping for Robust Monte Carlo Path Tracing" for
    // an apporach to normal mapping that can work bi-directionally.
    //
    // The solution employed here is descrbied in
    // "The Iray Light Transport System" Section A.3 "Local Shading
    // Normal Adaption".
    //
    const Vec3f &wo = isect.mWo;
    const Vec3f &Ng = isect.mNg;

    Vec3f result = Ns;

    // Compute reflection vector r, which is wo reflected about Ns
    const Vec3f r = 2.f * dot(Ns, wo) * Ns - wo;

    // If r is above the horizon of Ng, we're in good shape
    const float rDotNg = dot(r, Ng);

    // compute a bent reflection vector that is just
    // slightly (based on eps) above the Ng horizon
    const float eps = 0.01f;
    if (rDotNg < eps) {

        // this appears to work, but has the wrong normalization
        // if this becomes a problem, the correct normalization can be
        // computed via:
        //   const float s = math::sqrt(math::max(0.f, (1 - rDotNg * rDotNg)) / (1 - eps * eps));
        //   const float alpha = eps * s - rDotNg;
        //   math::Vec3f rBent = r + alpha * Ng;
        // 
        Vec3f rBent = r + (eps - rDotNg) * Ng;

        // Normalize it.
        // Worth noting: since we are going to normalize(wo + rBent)
        // we could alternatively scale wo via wo = wo * |rBent| and obtain
        // the same result for bentNs.
        rBent = normalize(rBent);

        // Compute bent Ns as halfway vector between wo and rBent
        Vec3f bentNs = normalize(wo + rBent);
        result = bentNs;
    }

    return result;
}

varying Vec3f
Intersection_adaptToonNormal(const varying Intersection &isect, const varying Vec3f &Ns)
{
    if (!Flags_get(isect.mFlags, (uniform uint32_t)INTERSECTION_FLAGS_USE_ADAPT_NORMAL)) {
        return Ns;
    }
    return Intersection_adaptNormal(isect, Ns);
}

//----------------------------------------------------------------------------
varying bool
isProvided(const varying Intersection &me, const uniform int32_t key)
{
    const uniform uint8_t *varying base =
        (const uniform uint8_t *varying)Address64_get(me.mData);
    return (key < me.mNumKeys) && (key >= 0) && (base != NULL) &&
        ((*(base + me.mValidTableOffset + key) & ATTRIBUTE_INITIALIZED) != 0);
}

varying bool
isProvided(const varying Intersection &me, const varying int32_t key)
{
    const uniform uint8_t *varying base =
        (const uniform uint8_t *varying)Address64_get(me.mData);
    return (key < me.mNumKeys) && (key >= 0) && (base != NULL) &&
        ((*(base + me.mValidTableOffset + key) & ATTRIBUTE_INITIALIZED) != 0);
}

varying bool
isdsProvided(const varying Intersection &me, const varying int32_t key)
{
    const uniform uint8_t *varying base =
        (const uniform uint8_t *varying)Address64_get(me.mData);
    return (key < me.mNumKeys) && (key >= 0) && (base != NULL) &&
        ((*(base + me.mValidTableOffset + key) & ATTRIBUTE_DS_INITIALIZED) != 0);
}

varying bool
isdtProvided(const varying Intersection &me, const varying int32_t key)
{
    const uniform uint8_t *varying base =
        (const uniform uint8_t *varying)Address64_get(me.mData);
    return (key < me.mNumKeys) && (key >= 0) && (base != NULL) &&
        ((*(base + me.mValidTableOffset + key) & ATTRIBUTE_DT_INITIALIZED) != 0);
}

// Return nullptr if attribute is not provided, otherwise return a pointer
// to the location of the attribute data.  Calling code will need to cast it
// to the appropriate type.
const uniform uint8_t * varying
getAttributeLocation(const varying Intersection &isect, uniform int32_t key)
{
    const uniform uint8_t * varying result = nullptr;
    if (isProvided(isect, key)) {
        // The attribute is provided, now get the location.
        // This requires us to look into the AttributeTable structure and extract
        // the mKeyOffset member.
        const uniform uint8_t * varying base = (const uniform uint8_t * varying) Address64_get(isect.mData);
        const uniform AttributeTable * varying table =
            (const uniform AttributeTable * varying) Address64_get(isect.mTable);
        const varying Address64 ptrToKeyOffsets = table->mKeyOffset;
        const uniform int32_t * varying keyOffsets =
            (const uniform int32_t * varying) Address64_get(ptrToKeyOffsets);
        const varying int32_t offset = keyOffsets[key];
        if (offset >= 0) {
            result = base + offset;
        }
    }
    return result;
}

