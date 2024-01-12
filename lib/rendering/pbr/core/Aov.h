// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// @file Aov.h
#pragma once

#include "Aov.hh"
#include <moonray/rendering/pbr/integrator/BsdfSampler.h>
#include "RayState.h"

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/lpe/StateMachine.h>
#include <moonray/rendering/pbr/core/Aov_ispc_stubs.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/Material.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/scene/rdl2/Layer.h>

#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace moonray {
namespace pbr {

class TLState;

// TODO: move to math lib, then should be
// typedef scene_rdl2::math::Colorv Colorv;
// in pbr/Types.h
ISPC_UTIL_TYPEDEF_STRUCT(Col3f, Colorv);

// ---------------------------------------------------------------------------
// AovSchema Basics

static const int AOV_MAX_RANGE_TYPE = 100;

// An aovSchema is an array of integers/AovSchemaIds that tells us what value
// goes in a particular float array location.  Some aov schema IDs are known
// at compile time (e.g. AOV_SCHEMA_ID_STATE_P - the
// state's position variable).  Some are not (e.g. some particular primitive
// color attribute index).  Those that are not known
// at compile time are known as a "RANGE_TYPE".  Their integer values occupy
// a reserved range of IDs.  For example, any ID in the range
// of [100, 199) corresponds to a primitive attribute float.
// In the code, an AovSchemaId variable can be represented as an int or as the
// AovSchemaId type.  We often use an int for the general case because the
// range type values aren't known at compile time.
// Multiple AOVs can use the same AovSchemaId if they output the same value.

// The actual value of these IDs is not important, but the order is
// assumed.  Do not re-arrange them without extensive inspection of Aov.cc.
enum AovSchemaId {
    AOV_SCHEMA_ID_UNKNOWN = 0,
    AOV_SCHEMA_ID_BEAUTY,
    AOV_SCHEMA_ID_ALPHA,
    // state position (vec3f)
    AOV_SCHEMA_ID_STATE_P,
    // state geometric normal (vec3f)
    AOV_SCHEMA_ID_STATE_NG,
    // state shading normal (vec3f)
    AOV_SCHEMA_ID_STATE_N,
    // state St coord (vec2f)
    AOV_SCHEMA_ID_STATE_ST,
    // state dpds (vec3f)
    AOV_SCHEMA_ID_STATE_DPDS,
    // state dpdt (vec3f)
    AOV_SCHEMA_ID_STATE_DPDT,
    // state ds, dt (floats)
    AOV_SCHEMA_ID_STATE_DSDX,
    AOV_SCHEMA_ID_STATE_DSDY,
    AOV_SCHEMA_ID_STATE_DTDX,
    AOV_SCHEMA_ID_STATE_DTDY,
    // wireframe output is handled like a state attribute aov
    AOV_SCHEMA_ID_WIREFRAME,
    // state world position (vec3f)
    AOV_SCHEMA_ID_STATE_WP,
    // state depth (-1.f * P.z)
    AOV_SCHEMA_ID_STATE_DEPTH,
    // motion vectors are handled like a state attribute aov
    AOV_SCHEMA_ID_STATE_MOTION,

    // Beyond here, everything should be a range type.
    // Danger! We only have room for AOV_MAX_RANGE_TYPE of each of these!
    // E.g. We can only support 100 different light AOV schema IDs.

    // primitive attributes
    AOV_SCHEMA_ID_PRIM_ATTR_FLOAT    =  1 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_PRIM_ATTR_VEC2F    =  2 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_PRIM_ATTR_VEC3F    =  3 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_PRIM_ATTR_RGB      =  4 * AOV_MAX_RANGE_TYPE,
    // material aovs
    AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT =  5 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_MATERIAL_AOV_VEC2F =  6 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_MATERIAL_AOV_VEC3F =  7 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_MATERIAL_AOV_RGB   =  8 * AOV_MAX_RANGE_TYPE,
    // lpe aovs
    AOV_SCHEMA_ID_VISIBILITY_AOV     =  9 * AOV_MAX_RANGE_TYPE,
    AOV_SCHEMA_ID_LIGHT_AOV          = 10 * AOV_MAX_RANGE_TYPE,
};

std::string aovSchemaIdToString(const AovSchemaId &aovSchemaId); // for debug

// LightAovs_hasEntries() in Aov.ispc depends on this
MNRY_STATIC_ASSERT(AOV_SCHEMA_ID_VISIBILITY_AOV == 900);
MNRY_STATIC_ASSERT(AOV_SCHEMA_ID_LIGHT_AOV == 1000);

// aliases
static constexpr int AOV_START_STATE_VAR      = AOV_SCHEMA_ID_STATE_P;
static constexpr int AOV_START_PRIM_ATTR      = AOV_SCHEMA_ID_PRIM_ATTR_FLOAT;
static constexpr int AOV_START_MATERIAL_AOV   = AOV_SCHEMA_ID_MATERIAL_AOV_FLOAT;
static constexpr int AOV_START_VISIBILITY_AOV = AOV_SCHEMA_ID_VISIBILITY_AOV;
static constexpr int AOV_START_LIGHT_AOV      = AOV_SCHEMA_ID_LIGHT_AOV;

// An AOV can be of a certain type:
// These values can be bitwise combined into a mask
enum AovType {
    AOV_TYPE_UNKNOWN        = 0,
    AOV_TYPE_BEAUTY         = (1 << 0),
    AOV_TYPE_ALPHA          = (1 << 1),
    AOV_TYPE_STATE_VAR      = (1 << 3),
    AOV_TYPE_PRIM_ATTR      = (1 << 4),
    AOV_TYPE_MATERIAL_AOV   = (1 << 5),
    AOV_TYPE_VISIBILITY_AOV = (1 << 6),
    AOV_TYPE_LIGHT_AOV      = (1 << 7),
    AOV_TYPE_ALL            = (AOV_TYPE_BEAUTY         |
                               AOV_TYPE_ALPHA          |
                               AOV_TYPE_STATE_VAR      |
                               AOV_TYPE_PRIM_ATTR      |
                               AOV_TYPE_MATERIAL_AOV   |
                               AOV_TYPE_VISIBILITY_AOV |
                               AOV_TYPE_LIGHT_AOV),
};

std::string aovTypeToString(const AovType &aovType); // for debug

enum AovFilter {
    AOV_FILTER_AVG = 0,
    AOV_FILTER_SUM,
    AOV_FILTER_MIN,
    AOV_FILTER_MAX,
    AOV_FILTER_FORCE_CONSISTENT_SAMPLING,
    AOV_FILTER_CLOSEST
};

std::string aovFilterToString(const AovFilter &aovFilter); // for debug

enum class AovStorageType {
    UNSPECIFIED,
    FLOAT,
    VEC2,
    VEC3,
    VEC4,
    RGB,
    RGB4,
    VISIBILITY
};

std::string aovStorageTypeToString(const AovStorageType &aovStorageType); // for debug

enum class AovOutputType {
    FLOAT,
    VEC2,
    VEC3,
    RGB
};

AovType aovType(int aovSchemaId);

// AOVs are currently limited to float channels.  Returns the number of float channels in an AOV.
unsigned aovNumChannels(int aovSchemaId);


class AovSchema
{
public:
    static constexpr int sLpePrefixNone       = 0;
    static constexpr int sLpePrefixUnoccluded = 1 << 0;

    struct EntryData
    {
        EntryData() noexcept :
            schemaID(-1),
            filter(AovFilter::AOV_FILTER_AVG),
            storageType(AovStorageType::UNSPECIFIED),
            lpePrefixFlags(sLpePrefixNone),
            stateAovId(0)
        {
        }

        int schemaID;
        AovFilter filter;
        AovStorageType storageType;
        int lpePrefixFlags;
        int stateAovId;

        std::string toString() const; // for debug
    };

    class Entry {
    public:
        Entry(const EntryData& data) noexcept :
            mId(data.schemaID),
            mType(aovType(data.schemaID)),
            mNumChannels(aovNumChannels(data.schemaID)),
            mFilter(data.filter),
            mStorageType(data.storageType),
            mLpePrefixFlags(data.lpePrefixFlags),
            mStateAovId(data.stateAovId)
        {
        }

        int            id()               const noexcept { return mId; }
        AovType        type()             const noexcept { return mType; }
        unsigned int   numChannels()      const noexcept { return mNumChannels; }
        AovFilter      filter()           const noexcept { return mFilter; }
        AovStorageType storageType()      const noexcept { return mStorageType; }
        int            lpePrefixFlags()   const noexcept { return mLpePrefixFlags; }
        int            stateAovId()       const noexcept { return mStateAovId; }
 
        float defaultValue() const noexcept {
            if (filter() == AOV_FILTER_MIN ||
                filter() == AOV_FILTER_CLOSEST) {
                return scene_rdl2::math::pos_inf;
            }
            if (filter() == AOV_FILTER_MAX) return scene_rdl2::math::neg_inf;
            return 0.f;
        }

        std::string toString() const; // for debug

    private:
        int            mId;
        AovType        mType;
        unsigned int   mNumChannels;
        AovFilter      mFilter;
        AovStorageType mStorageType;
        int            mLpePrefixFlags;
        int            mStateAovId;
    };

    // HUD validation
    static uint32_t hudValidation(bool verbose) { AOV_SCHEMA_VALIDATION; }

    AovSchema();

    // build the aov schema from an array of EntryData
    void init(const std::vector<EntryData> &data);

    // initialize a float array with values appropriate to filter type.
    // AVG,SUM,CLOSEST = 0, MIN = +inf, MAX = -inf
    void initFloatArray(float *aovs) const;

    // total number of (float) channels used by this aov schema
    // note: this is not mEntries.size()!
    unsigned int numChannels() const { return mNumChannels; }

    unsigned int size() const      { return mEntries.size(); }
    bool empty() const             { return mEntries.empty(); }

    // If one aov filter is not average or sum, then this schema
    // has a special math filter. In this case mHasAovFilter is true.
    bool hasAovFilter() const      { return mHasAovFilter; }

    // If any of the aovs have a closest filter, this returns true.
    bool hasClosestFilter() const  { return mHasClosestFilter; }

    // Checks if any of the AovSchema has the set of flags set. No guarantee that there exists
    // one AovSchema with all of the flags passed set.
    bool hasLpePrefixFlags(int flags) const { return (mAllLpePrefixFlags & flags) == flags; }

    const Entry& operator[] (int i) const { return mEntries[i]; }
    Entry &operator[](int i) { return mEntries[i]; }

    bool requiresScaledByWeight(unsigned int idx) const noexcept
    {
        return mEntries[idx].filter() == AOV_FILTER_AVG;
    }

    std::vector<Entry>::const_iterator begin() const { return mEntries.begin(); }
    std::vector<Entry>::const_iterator end()   const { return mEntries.end(); }

private:
    AOV_SCHEMA_MEMBERS;
};

// given a schema ID, return its type
AovType aovType(int aovSchemaId);

// given a schema ID, return its size in floats
unsigned aovNumChannels(int aovSchemaId);

// For a range type aov schemaId, what is the base aov schema id?
AovSchemaId aovToRangeTypeSchemaId(int aovSchemaId);

// From a range type aov schemaId, determine the offset to its
// base schema id?
int aovToRangeTypeOffset(int aovSchemaId);

// Set the relevant beauty and alpha AOV values in the *dest buffer.
// Skips values in *dest that aren't beauty and alpha.
void aovSetBeautyAndAlpha(pbr::TLState *pbrTls,
                          const AovSchema &aovSchema,
                          const scene_rdl2::math::Color &c,
                          float alpha,
                          float pixelWeight,
                          float *dest);

// ---------------------------------------------------------------------------
// State Variable Aovs

// Set the relevant state AOV values in the *dest buffer.
// Skips values in *dest that aren't state AOVs.
// The state AOV values come from an intersection structure.
// dest size = aovSchema.numChannels()
void aovSetStateVars(pbr::TLState *pbrTls,
                     const AovSchema &aovSchema,
                     const shading::Intersection &isect,
                     float volumeT,
                     const mcrt_common::RayDifferential &ray,
                     const Scene &scene,
                     float pixelWeight,
                     float *dest);

// Same as above but called when we have volume position/depth AOVs
//  but no hard-surface isect.
void aovSetStateVarsVolumeOnly(pbr::TLState *pbrTls,
                     const AovSchema &aovSchema,
                     float volumeT,
                     const mcrt_common::RayDifferential &ray,
                     const Scene &scene,
                     float pixelWeight,
                     float *dest);

// ---------------------------------------------------------------------------
// Primitive attribute aovs

// convert an integer code that represents a primitive
// attribute to a geom attribute key index.  Ensure that
// aovType(aov) == AOV_TYPE_PRIM_ATTR before calling.
int aovToGeomIndex(int aovSchemaId);

// convert a geom index into an aov code
int aovFromGeomIndex(int geomIndex);

// Set the relevant primitive attribute AOV values in the *dest buffer.
// The primitive attribute values come from the isect.
// Skips values in *dest that aren't primitive attribute AOVs.
// activeFlags determine which primitive attributes in aovSchema are 
// active for this isect - not all materials require all primitive attributes.
// dest size = aovSchema.numChannels()
void aovSetPrimAttrs(pbr::TLState *pbrTls,
                     const AovSchema &aovSchema,
                     const std::vector<char> &activeFlags,
                     const shading::Intersection &isect,
                     float pixelWeight,
                     float *dest);

// ---------------------------------------------------------------------------
// Material Aovs

// A class to manage a set of material aovs (owned by a render output driver).
// Note that the name of the material aov is its expression.
// A material aov:
//    - has a name/expression  e.g "RD.albedo"
//    - has an aovSchemaId
//    - holds a vectorized and non-vectorized function pointer
//      that is used to compute the aov
class MaterialAovs
{
public:
    struct Entry;

    // input parameters passed to scalar compute functions
    struct ComputeParams
    {
        ComputeParams(const Entry                        &entry,
                      const shading::Intersection        &isect,
                      const mcrt_common::RayDifferential &ray,
                      const Scene                        &scene,
                      const shading::Bsdf                &bsdf,
                      const scene_rdl2::math::Color      &ssAov,
                      const BsdfSampler                  *bSampler,
                      const BsdfSample                   *bsmps,
                      const shading::BsdfSlice           *bsdfSlice,
                      float                              pixelWeight):
            mEntry(entry),
            mIsect(isect),
            mRay(ray),
            mScene(scene),
            mBsdf(bsdf),
            mSsAov(ssAov),
            mBSampler(bSampler),
            mBsmps(bsmps),
            mBsdfSlice(bsdfSlice),
            mPixelWeight(pixelWeight)
        {
        }

            
        const Entry                        &mEntry;
        const shading::Intersection        &mIsect;
        const mcrt_common::RayDifferential &mRay;
        const Scene                        &mScene;
        const shading::Bsdf                &mBsdf;
        const scene_rdl2::math::Color      &mSsAov;
        const BsdfSampler                  *mBSampler;
        const BsdfSample                   *mBsmps;
        const shading::BsdfSlice           *mBsdfSlice;
        float                              mPixelWeight;
    };

    // input parameters passed to vector compute functions
    struct ComputeParamsv
    {
        ComputeParamsv(const Entry                         &entry,
                       const shading::Intersectionv        &isect,
                       const mcrt_common::RayDifferentialv &ray,
                       const Scene                         &scene,
                       const shading::Bsdfv                &bsdf,
                       const Colorv                        &ssAov,
                       const BsdfSamplerv                  *bSampler,
                       const BsdfSamplev                   *bsmps,
                       const shading::BsdfSlicev           *bsdfSlice,
                       const float                         *pixelWeight,
                       uint32_t                            lanemask):
            mEntry(entry),
            mIsect(isect),
            mRay(ray),
            mScene(scene),
            mBsdf(bsdf),
            mSsAov(ssAov),
            mBSampler(bSampler),
            mBsmps(bsmps),
            mBsdfSlice(bsdfSlice),
            mPixelWeight(pixelWeight),
            mLanemask(lanemask)
        {
        }

        const Entry                         &mEntry;
        const shading::Intersectionv        &mIsect;
        const mcrt_common::RayDifferentialv &mRay;
        const Scene                         &mScene;
        const shading::Bsdfv                &mBsdf;
        const Colorv                        &mSsAov;
        const BsdfSamplerv                  *mBSampler;
        const BsdfSamplev                   *mBsmps;
        const shading::BsdfSlicev           *mBsdfSlice;
        const float                         *mPixelWeight;
        uint32_t                            mLanemask;
    };

    // pointer to function that computes a material aov from
    //    - aovSchemaId
    //    - bsdf
    //    - computed subsurface material aov value
    //    - bsdf sampler (can be null)
    //    - bsdf samples (can be null)
    // not all material aov computation functions will use all
    // of these inputs, but they are available to all of them.
    typedef void (*ComputeFn)(const ComputeParams &params, float *dest);
    typedef void (*ComputeFnv)(const ComputeParamsv &params, float *dest);

    struct Entry
    {
        Entry(const std::string &name, int aovSchemaId,
              ComputeFn computeFn, ComputeFnv computeFnv,
              const std::vector<int> &geomLabelIndices,
              const std::vector<int> &materialLabelIndices,
              const std::vector<int> &labelIndices,
              int lobeFlags, AovFilter filter,
              int primAttrKey, AovSchemaId stateAovId,
              AovSchemaId lpeSchemaId, int lpeLabelId, bool subsurface);

        std::string      mName;
        int              mAovSchemaId;
        ComputeFn        mComputeFn;
        ComputeFnv       mComputeFnv;
        std::vector<int> mGeomLabelIndices;
        std::vector<int> mMaterialLabelIndices;
        std::vector<int> mLabelIndices;
        int              mLobeFlags;
        AovFilter        mFilter;

        // Used if this material aov is a primitive attribute
        int              mPrimAttrKey; // -1 if not in use

        // Used if this material aov is a state variable
        AovSchemaId      mStateAovId;  // AOV_SCHEMA_ID_UNKNOWN if not in use

        // Used when this material aov has an LPE.
        // We actually create an entry in the LightAovs as well for this
        // material AOV so we can use the LightAovs' LPE state machine.
        // Note that we still only write the material AOV to the channels
        // that match the mAovSchemaId.  The mLpeSchemaId is only used for
        // matching/filtering based on the LPE (in MaterialAovs::computeScalar/Vector()).
        // It doesn't actually have any real channels.
        AovSchemaId      mLpeSchemaId; // AOV_SCHEMA_ID_UNKNOWN if not in use
        int              mLpeLabelId;  // -1 if not in use

        bool             mSubsurface;
    };


    // HUD validation.
    static uint32_t hudValidation(bool verbose) { MATERIAL_AOVS_VALIDATION; }

    MaterialAovs();

    // MaterialAov expressions can use labels.  This function returns
    // a global index for a particular label.  The label will be known
    // only in the case that it is used in some existing, parsed expression.
    // @return the label index or < 0 if not found
    int getLabelIndex(const std::string &label) const;
    int getMaterialLabelIndex(const std::string &label) const;
    int getGeomLabelIndex(const std::string &label) const;

    // Create a material aov entry.
    // Return the aovSchemaId for the created entry.
    // A material aov entry can require a geom primitive attribute.  If the
    // created entry requires a primitive attribute, the key is written to the
    // primAttrKey parameter.  If the material aov is a state variable, the
    // schema id of the underlying state variable is written to stateVariable
    int createEntry(const std::string &name,
                    AovFilter filter,
                    AovSchemaId lpeSchemaId,
                    int lpeLabelId,
                    AovSchemaId &stateVar,
                    int &primAttrKey);

    // return the aovSchemaId for this aov, returns AOV_SCHEMA_ID_UNKNOWN
    // if not found.
    int findEntry(const std::string &name, AovSchemaId lpeSchemaId) const;

    // Compute a material aov, store results in dest.
    void computeScalar(pbr::TLState *pbrTls,
                       int aovSchemaId,
                       const LightAovs& lightAovs,
                       const shading::Intersection &isect,
                       const mcrt_common::RayDifferential &ray,
                       const Scene &scene, const shading::Bsdf &bsdf,
                       const scene_rdl2::math::Color &ssAov,
                       const BsdfSampler *bSampler,
                       const BsdfSample *bsmps,
                       const shading::BsdfSlice *bsdfSlice,
                       float pixelWeight,
                       int lpeStateId,
                       float *dest) const;

    uint32_t computeVector(pbr::TLState *pbrTls, 
                       int aovSchemaId,
                       const LightAovs &lightAovs,
                       const shading::Intersectionv &isectv,
                       const mcrt_common::RayDifferentialv &ray,
                       const Scene &scene,
                       const shading::Bsdfv &bsdfv,
                       const Colorv &ssAov,
                       const BsdfSamplerv *bSampler,
                       const BsdfSamplev *bsmps,
                       const shading::BsdfSlicev *bsdfSlice,
                       const float *pixelWeight,
                       const int *lpeStateId,
                       const uint32_t *isPrimaryRay,
                       float *dest,
                       uint32_t lanemask) const;

private:

    static AovSchemaId parseExpression(const std::string &expression,
                                       ComputeFn &computeFn,
                                       ComputeFnv &computeFnv,
                                       std::vector<std::string> &geomLabels,
                                       std::vector<std::string> &materialLabels,
                                       std::vector<std::string> &labels,
                                       int &lobeFlags,
                                       int &primAttrKey,
                                       AovSchemaId &stateAovId,
                                       bool &subsurface);

    MATERIAL_AOVS_MEMBERS;
};

// Parsing

// The result of a MaterialAov parse
// Semantic checking is still needed after parsing
struct ParsedMaterialExpression
{
    void init(const std::string &expression);

    // List of possible material aov properties
    // Semantic compatibility between these and the selector
    // must be checked post-parse.
    enum Property {
        PROPERTY_UNKNOWN = 0,
        PROPERTY_ALBEDO,
        PROPERTY_COLOR,
        PROPERTY_FACTOR,
        PROPERTY_EMISSION,
        PROPERTY_NORMAL,
        PROPERTY_RADIUS,
        PROPERTY_ROUGHNESS,
        PROPERTY_MATTE,
        PROPERTY_PBR_VALIDITY,
        PROPERTY_STATE_VARIABLE_P,
        PROPERTY_STATE_VARIABLE_N,
        PROPERTY_STATE_VARIABLE_NG,
        PROPERTY_STATE_VARIABLE_ST,
        PROPERTY_STATE_VARIABLE_DPDS,
        PROPERTY_STATE_VARIABLE_DPDT,
        PROPERTY_STATE_VARIABLE_DSDX,
        PROPERTY_STATE_VARIABLE_DSDY,
        PROPERTY_STATE_VARIABLE_DTDX,
        PROPERTY_STATE_VARIABLE_DTDY,
        PROPERTY_STATE_VARIABLE_WP,
        PROPERTY_STATE_VARIABLE_DEPTH,
        PROPERTY_STATE_VARIABLE_MOTION,
        PROPERTY_PRIMITIVE_ATTRIBUTE_FLOAT,
        PROPERTY_PRIMITIVE_ATTRIBUTE_VEC2,
        PROPERTY_PRIMITIVE_ATTRIBUTE_VEC3,
        PROPERTY_PRIMITIVE_ATTRIBUTE_RGB,
    };

    // Lobes (or subsurface) selector
    // These bits simply tell us what was present
    // in the selector section of the string.  The
    // semantics can vary depending on the property.
    static const int EMPTY        = 0;
    static const int REFLECTION   = 1 << 0;
    static const int TRANSMISSION = 1 << 1;
    static const int DIFFUSE      = 1 << 2;
    static const int GLOSSY       = 1 << 3;
    static const int MIRROR       = 1 << 4;
    static const int SUBSURFACE   = 1 << 5;

    // What property are we outputing?
    Property mProperty;

    // A "fresnel" keyword can qualify any property. This field is true
    // if the parsed property contained the fresnel qualifier
    bool mFresnelProperty;

    // After the labels, we can optionally have a set of
    // lobe selectors or the subsurface selector. [RTGDM] | SS
    // If empty, ALL_LOBES is the interpretation.
    // These selections are stored in the following bits
    int mSelector;

    // We can parse a hierarchical set of labels separated by
    // the '.' character.  These are interepreted as
    // [<geomLabels>.][<materialLabels>.][<lobeLabels>]
    std::stack<std::vector<std::string> > mLabels;

    // Primitive attribute properties are named by an
    // arbitrary string, e.g. ref_P
    std::string mPrimitiveAttribute;

    // If the parser encounters a syntax error, the error string
    // is placed in this member.  Empty means no error.
    std::string mError;

    // This is the expression string set in init
    std::string mExpression;

    // This is the index of the next char to lex
    std::size_t mNextLex;
};

// pass in an initialized ParsedMaterialExpression structure
// function parses and fills in results.
void aovParseMaterialExpression(ParsedMaterialExpression *m);

// Set the relevant material AOV values in the *dest buffer.
// Skips values in *dest that aren't material AOVs.
// dest size = aovSchema.numChannels()
void aovSetMaterialAovs(pbr::TLState *pbrTls,
                        const AovSchema &aovSchema,
                        const LightAovs &lightAovs,
                        const MaterialAovs &materialAovs,
                        const shading::Intersection &isect,
                        const mcrt_common::RayDifferential &ray,
                        const Scene &scene,
                        const shading::Bsdf &bsdf,
                        const scene_rdl2::math::Color &ssAov,
                        const BsdfSampler *bSampler,
                        const BsdfSample *bsmps,
                        float pixelWeight,
                        int lpeStateId,
                        float *dest);

// Set the relevant material AOV values in the *dest buffer.
// Skips values in *dest that aren't material AOVs.
// dest size = aovSchema.numChannels()
void aovSetMaterialAovs(pbr::TLState *pbrTls,
                        const AovSchema &aovSchema,
                        const LightAovs &lightAovs,
                        const MaterialAovs &materialAovs,
                        const shading::Intersection &isect,
                        const mcrt_common::RayDifferential &ray,
                        const Scene &scene,
                        const shading::Bsdf &bsdf,
                        const scene_rdl2::math::Color &ssAov,
                        const shading::BsdfSlice *bsdfSlice,
                        float pixelWeight,
                        int lpeStateId,
                        float *dest);

// exposed to ispc
extern "C" void
CPP_aovSetMaterialAovs(pbr::TLState *pbrTls,
                       const AovSchema &aovSchema,
                       const LightAovs &lightAovs,
                       const MaterialAovs &materialAovs,
                       const shading::Intersectionv &isectv,
                       const mcrt_common::RayDifferentialv &ray,
                       intptr_t scene,
                       const shading::Bsdfv &bsdf,
                       const Colorv &ssAov,
                       const BsdfSamplerv *bSampler,
                       const BsdfSamplev *bsmps,
                       const shading::BsdfSlicev *bsdfSlice,
                       const float pixelWeight[],
                       const uint32_t pixel[],
                       const uint32_t deepDataHandle[],
                       const int lpeStateId[],
                       const uint32_t isPrimaryRay[],
                       uint32_t lanemask);

// ---------------------------------------------------------------------------
// Light Aovs

// A class to manage a set of light aovs (owned by a render output driver)
class LightAovs
{
public:
    // Background extra aovs are defined by a label and a color.
    // These aovs are triggered when an indirect ray misses geometry.
    struct BackgroundExtraAov
    {
        const char *mLabel;
        scene_rdl2::math::Color mColor;
        int mLabelId;
    };
    
    struct BackgroundExtraAovs
    {
        static const int sNum = 1;
        BackgroundExtraAov &operator[](int i) { return mAovs[i]; }
        const BackgroundExtraAov &operator[](int i) const { return mAovs[i]; }
        BackgroundExtraAov mAovs[sNum] { { "U:bg_white", scene_rdl2::math::sWhite, -1 } };
    };

    // HUD validation
    static uint32_t hudValidation(bool verbose) { LIGHT_AOVS_VALIDATION; }

    explicit LightAovs(const scene_rdl2::rdl2::Layer *layer);

    // @param visibility: is this a visibility aov?
    // @param flags: stores any resulting lpe prefix flags
    // @return an aov schema id for this light aov
    int createEntry(const std::string &lpe, bool visibility, int &flags);

    // @return a light aov label id for this label, -1 if
    // it does not exist.
    int getLabelId(const std::string &label) const;

    // @return background extra aov structures
    const BackgroundExtraAovs &getBackgroundExtraAovs() const { return mBackgroundExtraAovs; }

    // @return a light aov label id for this material label, -1
    // if it does not exist.
    // @note by convention, material labels should be prefixed
    // with "mat:" when appearing in light path expressions.
    // without the prefix they are interpreted as lobe or light
    // labels.
    int getMaterialLabelId(const std::string &label) const;

    // finalize the state machine.  No further calls to createEntry are legal
    void finalize();

    // The "transition" event functions below are called by the integrator
    // when various things happen.  They are convenience wrappers that call
    // mLpeStateMachine.transition().

    // transition the LPE state machine to a ray leaves camera event
    // this is the initial event and requires no lpeStateId.
    int cameraEventTransition(pbr::TLState *pbrTls) const;

    // given a Bsdf and BsdfLobe describing a scattering event,
    // transition the LPE state machine to this event.  Return the new lpeStateId.
    int scatterEventTransition(pbr::TLState *pbrTls,
                               int lpeStateId,
                               const shading::Bsdf &bsdf,
                               const shading::BsdfLobe &lobe) const;
    int scatterEventTransitionVector(pbr::TLState *pbrTls,
                                     int lpeStateId,
                                     const shading::Bsdfv &bsdfv,
                                     const shading::BsdfLobev &lobev) const;

    int straightEventTransition(pbr::TLState *pbrTls, int lpeStateId) const;

    // transition the LPE state machine to a hit light event
    // this is typically a final transition.
    int lightEventTransition(pbr::TLState *pbrTls, int lpeStateId, const Light *light) const;

    // transition the LPE state machine to an "extra aov" event
    // this is typically a final transition.
    int extraAovEventTransition(pbr::TLState *pbrTls, int lpeStateId, int labelId) const;

    // transition the LPE state machine to a hit emissive surface event
    // this is typically a final transition
    int emissionEventTransition(pbr::TLState *pbrTls, int lpeStateId, const shading::Bsdf &bsdf) const;

    // transition the LPE state machine to a hit emissive volume event
    // this is typically a final transition
    int emissionEventTransition(pbr::TLState *pbrTls, int lpeStateId, int volumeLabelId) const;

    // transition the LPE state machine to a "material aov" event
    // this is typically a final transition.
    int materialAovEventTransition(pbr::TLState *pbrTls, int lpeStateId, int lpeLabelId) const;

    // transition the LPE state machine to a hit on a bssrdf
    int subsurfaceEventTransition(pbr::TLState *pbrTls, int lpeStateId, const shading::Bsdf &bsdf) const;

    // transition the LPE state machine to a volume scattering event
    int volumeEventTransition(pbr::TLState *pbrTls, int lpeStateId) const;


    // check if the given aovSchemaId (returned from createEntry)
    // is valid at this lpeStateId
    bool isValid(pbr::TLState *pbrTls, int lpeStateId, int aovSchemaId) const;

    // @return true if we have light or visibility aovs, false otherwise
    bool hasEntries() const {
        return mNextLightAovSchemaId > AOV_SCHEMA_ID_LIGHT_AOV ||
               mNextVisibilityAovSchemaId > AOV_SCHEMA_ID_VISIBILITY_AOV;
    }

    // @return true if we have visibility aovs, false otherwise
    bool hasVisibilityEntries() const {
        return mNextVisibilityAovSchemaId > AOV_SCHEMA_ID_VISIBILITY_AOV;
    }

private:
    typedef std::unordered_map<std::string, std::string> LabelSubstitutions;
    // must match member size in LIGHT_AOVS_MEMBERS
    MNRY_STATIC_ASSERT(sizeof(lpe::StateMachine) == SIZEOF_LPE_STATEMACHINE);
    MNRY_STATIC_ASSERT(sizeof(LabelSubstitutions) == SIZEOF_LABELSUBSTITUTIONS);

    // compute an LPE state machine label for a scattering event.
    template<typename LobeType>
    static int computeScatterEventLabelId(const shading::Bsdf &bsdf, const LobeType *lobe);
    static int computeScatterEventLabelIdVector(const shading::Bsdfv &bsdf, const shading::BsdfLobev *lobe);

    // replace an LPE alias with a normal LPE
    static std::string replaceLpeAliases(const std::string &lpe);

    // expand material and lobe labels as needed
    std::string expandLpeLabels(const std::string &lpe) const;

    // construct the label substitutions from the render layer
    void buildLabelSubstitutions(const scene_rdl2::rdl2::Layer &layer);

    LIGHT_AOVS_MEMBERS;
};


// add matchValue to aov dest locations that match lpeStateId and prefixFlags.
// If nonMatchValue is not NULL, adds it to aov dest locations that match lpeStateId and not prefixFlags.
// @returns true if there is a dest that matches lpeStateId, false otherwise
bool aovAccumLightAovs(pbr::TLState *pbrTls,
                       const AovSchema &aovSchema,
                       const LightAovs &lightAovs,
                       const scene_rdl2::math::Color &matchValue,
                       const scene_rdl2::math::Color *nonMatchValue,
                       int prefixFlags,
                       int lpeStateId,
                       float *dest);

bool aovAccumVisibilityAovs(pbr::TLState *pbrTls,
                            const AovSchema &aovSchema,
                            const LightAovs &lightAovs,
                            const scene_rdl2::math::Vec2f &value,
                            int lpeStateId,
                            float *dest);

/// Adds the specified number of "misses" to the visibility
/// aov, disregarding the lpe
bool aovAccumVisibilityAttempts(pbr::TLState *pbrTls,
                                const AovSchema &aovSchema,
                                const LightAovs &lightAovs,
                                const float value,
                                float *dest);

// queue aov results that match lpeStateId
// @returns true if there is a result that matches lpeStateId, false otherwise
bool aovAccumLightAovsBundled(pbr::TLState *pbrTls,
                              const AovSchema &aovSchema,
                              const LightAovs &lightAovs,
                              const scene_rdl2::math::Color &matchValue,
                              const scene_rdl2::math::Color *nonMatchValue,
                              int prefixFlags,
                              int lpeStateId,
                              uint32_t pixel,
                              uint32_t deepDataHandle);

bool aovAccumVisibilityAovsBundled(pbr::TLState *pbrTls,
                                   const AovSchema &aovSchema,
                                   const LightAovs &lightAovs,
                                   const scene_rdl2::math::Vec2f &value,
                                   int lpeStateId,
                                   uint32_t pixel,
                                   uint32_t deepDataHandle,
                                   bool lpePassthrough);

/// Adds the specified number of "misses" to the visibility
/// aov, disregarding the lpe
bool aovAccumVisibilityAttemptsBundled(pbr::TLState *pbrTls,
                                       const AovSchema &aovSchema,
                                       const LightAovs &lightAovs,
                                       int attempts,
                                       uint32_t pixel,
                                       uint32_t deepDataHandle);

// "Extra Aovs"
// Extra aovs are a type of light aov, but rather than accumulating radiance
// when a light or emissive object is hit, they are accumulated after
// material shading.  They are called "extra" aovs because they are additonal
// shader computations that are not otherwise run as part of the main render.
// These functions both compute and then accumulate extra aovs.
void aovAccumExtraAovs(pbr::TLState *pbrTls,
                       const FrameState &fs,
                       const PathVertex &pv,
                       const shading::Intersection &isect,
                       const scene_rdl2::rdl2::Material *mat,
                       float *dest);

void aovAccumExtraAovsBundled(pbr::TLState *pbrTls,
                              const FrameState &fs,
                              RayState const * const *rayStates,
                              const float *presences,
                              const shading::Intersectionv *isectvs,
                              const scene_rdl2::rdl2::Material *mat,
                              unsigned int numRays);
                       
// "Post Scatter" extra aovs are extra aovs that are computed at the
// ray intersection point, but are accumulated relative to the
// scattered, outgoing indirect rays.  This means that the LPE scatter transition
// event and path throughput reduction are applied before the result
// is accumulated.  This function just accumulates the aovs.  It relies on the
// bsdf object to already hold stored extra aov evaluations.
void aovAccumPostScatterExtraAovs(pbr::TLState *pbrTls,
                                  const FrameState &fs,
                                  const PathVertex &pv,
                                  const shading::Bsdf &bsdf,
                                  float *dest);

// Another type of extra aov is a background aov.  This is called when an
// indirect ray exits the scene without intersecting geometry.
void aovAccumBackgroundExtraAovs(pbr::TLState *pbrTls,
                                 const FrameState &fs,
                                 const PathVertex &pv,
                                 float *dest);

void aovAccumBackgroundExtraAovsBundled(pbr::TLState *pbrTls,
                                        const FrameState &fs,
                                        const RayState *rs);

// ---------------------------------------------------------------------------
// Bundling

// Bundling support - send an array worth of aov float values
// to the aov queue. Only adds non-zero values for aovs
// using sum or avg filters.  Does not pass along +/-inf
// for aovs with min or max filters.
void aovAddToBundledQueue(pbr::TLState *pbrTls,
                          const AovSchema &aovSchema,
                          const shading::Intersection &isect,
                          const mcrt_common::RayDifferential &ray,
                          const uint32_t aovTypeMask,
                          const float *aovValues,
                          uint32_t pixel,
                          uint32_t deepDataHandle);

// Same as above but called when we have volume position/depth AOVs
//  but no hard-surface isect.
void aovAddToBundledQueueVolumeOnly(pbr::TLState *pbrTls,
                                    const AovSchema &aovSchema,
                                    const mcrt_common::RayDifferential &ray,
                                    const uint32_t aovTypeMask,
                                    const float *aovValues,
                                    uint32_t pixel,
                                    uint32_t deepDataHandle);

} // namespace pbr
} // namespace moonray

