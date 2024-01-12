// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Intersection
/// $Id$
///

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/InstanceAttributes.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/bvh/shading/ispc/Intersection.hh>
#include <moonray/rendering/bvh/shading/MipSelector.h>

#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/scene/rdl2/Layer.h>
#include <scene_rdl2/scene/rdl2/Material.h>
#include <scene_rdl2/scene/rdl2/Types.h>

namespace moonray {
namespace shading {

class CACHE_ALIGN Intersection
{
public:
    enum IntersectionFlags
    {
        HasGeometryDerivatives  = 1,

        GeomInitialized         = 1 << 1,   // Is the data fully initialized for a geom intersection.
        DisplacementInitialized = 1 << 2,   // Is the data fully initialized for a displacement intersection.

        Entering                = 1 << 3,
        SubsurfaceAllowed       = 1 << 4,  // Should we calculate subsurface on this path?
        CausticPath             = 1 << 5,  // Is this on a caustic path ?

        // If you move this, be sure to adjust sPathTypeOffset accordingly
        PathTypeLoBit           = 1 << 6,  // |
        PathTypeMidBit          = 1 << 7,  // | Three bits to encode PathType
        PathTypeHiBit           = 1 << 8,  // |

        HasAllRequiredAttributes= 1 << 9,

        UseAdaptNormal          = 1 << 10,

        // Determines whether or not we'll test the normal 
        // for backfacing points ( see MOONSHINE-1562 )
        IsFlatPoint             = 1 << 11
    };


    // Warning: Keep this enum in sync with enum in Intersection.isph
    enum PathType {
        // Camera paths
        Primary = 0,        // Primary ray hit
        IndirectMirror,     // At least one mirror bounce
        IndirectGlossy,     // At least one glossy or mirror bounce
        IndirectDiffuse,    // At least one diffuse or glossy or mirror bounce

        // Light path not yet connected to the camera (bi-directional integrators)
        LightPath,

        PathTypeCount
    };

    // cppcheck-suppress uninitMemberVar // (some members are left intentionally uninitialized)
    Intersection()
    {
#pragma warning(push)
#pragma warning(disable:1875)   // #1875: offsetof applied to non-POD (Plain Old Data) types is nonstandard
        MNRY_STATIC_ASSERT(offsetof(Intersection, mP) == 64);
        MNRY_STATIC_ASSERT(offsetof(Intersection, mdPdt) == 112);
        MNRY_STATIC_ASSERT(offsetof(Intersection, mdSdx) == 136);
        MNRY_STATIC_ASSERT(offsetof(Intersection, mWo) == 192);
#pragma warning(pop)
        MNRY_STATIC_ASSERT(sizeof(Intersection) == CACHE_LINE_SIZE * 4);

        init(); // This init is redundant and could go.
    }

    void init(const scene_rdl2::rdl2::Geometry *rdlGeometryObject = nullptr)
    {
        // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
        memset(this, 0, sizeof(Intersection));
        mGeometryObject = rdlGeometryObject;
        mLayerAssignmentId = -1;
        mFlags.clearAll();
    }

    // set members that are based on layer and assignmentId
    void setLayerAssignments(int assignmentId, const scene_rdl2::rdl2::Layer *layer)
    {
        MNRY_ASSERT(assignmentId != -1, "unassigned part");
        mLayerAssignmentId = assignmentId;

        mMaterial = MNRY_VERIFY(layer->lookupMaterial(mLayerAssignmentId));

        const scene_rdl2::rdl2::Layer::GeometryPartPair gpPair =
            layer->lookupGeomAndPart(mLayerAssignmentId);
        mGeometryObject = MNRY_VERIFY(gpPair.first);
    }

    // Set ids
    finline void setIds(uint32_t id1, uint32_t id2, uint32_t id3) {
        mId1 = id1;  mId2 = id2;  mId3 = id3;
    }

    // Set epsilon hint
    finline void setEpsilonHint(float value) {
        mEpsilonHint = value;
    }

    // Set shadow epsilon hint
    finline void setShadowEpsilonHint(float value) {
        mShadowEpsilonHint = value;
    }

    finline void setDifferentialGeometry(const scene_rdl2::math::Vec3f &Ng, const scene_rdl2::math::Vec3f &N,
            const scene_rdl2::math::Vec2f &St, const scene_rdl2::math::Vec3f &dPds, const scene_rdl2::math::Vec3f &dPdt,
            bool hasDerivatives, bool isFlatPoint=false)
    {
        mNg = Ng;
        mN = N;
        mSt = St;
        mdPds = dPds;
        mdPdt = dPdt;

        mFlags.set(HasGeometryDerivatives, hasDerivatives);
        mFlags.set(IsFlatPoint, isFlatPoint);
    }

    void setTable(scene_rdl2::alloc::Arena *arena, const AttributeTable * const table) {
        MNRY_ASSERT(arena != nullptr);
        mValidTableOffset = table->getAttributesSize();
        mNumKeys = table->getNumKeys();
        mTable = table;
        // the memory layout for allocated mData:
        // |                                       |    |
        // |---------------------------------------|----|
        // |buffer to store primitive attributes   |    |
        //                                          table to mark whether a
        //                                          specific attribute exists
        int size = mValidTableOffset + mNumKeys;
        mData = nullptr;
        if (size) {
            mData = (char *)arena->alloc(size, sizeof(void*));
            memset(mData, 0, size);
        }
    }

    const AttributeTable *getTable() const {
        return MNRY_VERIFY(mTable);
    }

    bool isProvided(AttributeKey key) const {
        return (key < mNumKeys) && (key >= 0) && (mData != nullptr) &&
            ((*(mData + mValidTableOffset + key) & ATTRIBUTE_INITIALIZED) != 0);
    }

    bool isdsProvided(AttributeKey key) const {
        return (key < mNumKeys) && (key >= 0) && (mData != nullptr) &&
            ((*(mData + mValidTableOffset + key) & ATTRIBUTE_DS_INITIALIZED) != 0);
    }

    bool isdtProvided(AttributeKey key) const {
        return (key < mNumKeys) && (key >= 0) && (mData != nullptr) &&
            ((*(mData + mValidTableOffset + key) & ATTRIBUTE_DT_INITIALIZED) != 0);
    }

    template <typename T>
    inline const T &getAttribute(TypedAttributeKey<T> key) const {
        MNRY_ASSERT(mData);
        char *readLoc = mData + keyOffset(key);
        return *(reinterpret_cast<T *>(readLoc));
    }

    template <typename T>
    inline T getdAttributeds(TypedAttributeKey<T> key) const {
        if (isdsProvided(key)) {
            char *readLoc = mData + keyOffset(key) + key.getSize();
            return *(reinterpret_cast<T *>(readLoc));
        } else {
            return T(scene_rdl2::math::zero);
        }
    }

    template <typename T>
    inline T getdAttributedt(TypedAttributeKey<T> key) const {
        MNRY_ASSERT(mData && key.hasDerivatives());
        if (isdtProvided(key)) {
            char *readLoc = mData + keyOffset(key) + 2 * key.getSize();
            return *(reinterpret_cast<T *>(readLoc));
        } else {
            return T(scene_rdl2::math::zero);
        }
    }

    template <typename T>
    inline void setAttribute(TypedAttributeKey<T> key, Interpolator &interpolator) {
        MNRY_ASSERT(mData);
        char *writeLoc = mData + keyOffset(key);
        if (interpolator.interpolate(key, writeLoc)) {
            // mark this attribute is correctly provided in this Intersection
            *(mData + mValidTableOffset + key) |= ATTRIBUTE_INITIALIZED;
        }
    }

    template <typename T>
    inline void setAttribute(TypedAttributeKey<T> key,
            const InstanceAttributes* instanceAttributes) {
        MNRY_ASSERT(mData);
        char *writeLoc = mData + keyOffset(key);
        instanceAttributes->getAttribute(key, writeLoc);
        *(mData + mValidTableOffset + key) |= ATTRIBUTE_INITIALIZED;
    }

    template <typename T>
    inline void setAttribute(TypedAttributeKey<T> key, const T &t)
    {
        MNRY_ASSERT(mData);
        char *writeLoc = mData + keyOffset(key);
        *(reinterpret_cast<T *>(writeLoc)) = t;
       *(mData + mValidTableOffset + key) |= ATTRIBUTE_INITIALIZED;
    }

    template <typename T>
    inline void setdAttributeds(TypedAttributeKey<T> key, const T &dfds)
    {
        MNRY_ASSERT(mData && key.hasDerivatives());
        char *writeLoc = mData + keyOffset(key) + key.getSize();
        *(reinterpret_cast<T *>(writeLoc)) = dfds;
        *(mData + mValidTableOffset + key) |= ATTRIBUTE_DS_INITIALIZED;
    }

    template <typename T>
    inline void setdAttributedt(TypedAttributeKey<T> key, const T &dfdt)
    {
        MNRY_ASSERT(mData && key.hasDerivatives());
        char *writeLoc = mData + keyOffset(key) + 2 * key.getSize();
        *(reinterpret_cast<T *>(writeLoc)) = dfdt;
        *(mData + mValidTableOffset + key) |= ATTRIBUTE_DT_INITIALIZED;
    }

    inline void setAttribute(AttributeKey key, Interpolator &interpolator) {
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_BOOL:
            setAttribute(TypedAttributeKey<bool>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_INT:
            setAttribute(TypedAttributeKey<int>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_LONG:
            setAttribute(TypedAttributeKey<long>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_FLOAT:
            setAttribute(TypedAttributeKey<float>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_STRING:
            setAttribute(TypedAttributeKey<std::string>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            setAttribute(TypedAttributeKey<scene_rdl2::math::Color>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_RGBA:
            setAttribute(TypedAttributeKey<scene_rdl2::math::Color4>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            setAttribute(TypedAttributeKey<scene_rdl2::math::Vec2f>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            setAttribute(TypedAttributeKey<scene_rdl2::math::Vec3f>(key), interpolator);
            break;
        case scene_rdl2::rdl2::TYPE_MAT4F:
            setAttribute(TypedAttributeKey<scene_rdl2::math::Mat4f>(key), interpolator);
            break;
        default:
            MNRY_ASSERT(false, (std::string("unsupported attribute type ") +
                std::string(attributeTypeName(key.getType())) +
                std::string(" for atttribute ") + std::string(key.getName())).c_str());
            break;
        }
    }

    inline void setRequiredAttributes(Interpolator &interpolator) {
        for (const auto k : getTable()->getRequiredAttributes()) {
            setAttribute(k, interpolator);
        }
        for (const auto k : getTable()->getOptionalAttributes()) {
            setAttribute(k, interpolator);
        }
    }

    inline void fillInstanceAttributes(
        const InstanceAttributes* instanceAttributes,
        const std::vector<AttributeKey>& keys) {
        for (auto key : keys) {
            if (!instanceAttributes->isSupported(key)) {
                continue;
            }
            switch (key.getType()) {
            case scene_rdl2::rdl2::TYPE_BOOL:
                setAttribute(TypedAttributeKey<bool>(key), instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_INT:
                setAttribute(TypedAttributeKey<int>(key), instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_LONG:
                setAttribute(TypedAttributeKey<long>(key), instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_FLOAT:
                setAttribute(TypedAttributeKey<float>(key), instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_STRING:
                setAttribute(TypedAttributeKey<std::string>(key),
                    instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_RGB:
                setAttribute(TypedAttributeKey<scene_rdl2::math::Color>(key),
                    instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_RGBA:
                setAttribute(TypedAttributeKey<scene_rdl2::math::Color4>(key),
                    instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2F:
                setAttribute(TypedAttributeKey<scene_rdl2::math::Vec2f>(key), instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3F:
                setAttribute(TypedAttributeKey<scene_rdl2::math::Vec3f>(key), instanceAttributes);
                break;
            case scene_rdl2::rdl2::TYPE_MAT4F:
                setAttribute(TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                    instanceAttributes);
                break;
            default:
                MNRY_ASSERT(false, (std::string("unsupported attribute type ") +
                    std::string(attributeTypeName(key.getType())) +
                    std::string(" for atttribute ") +
                    std::string(key.getName())).c_str());
                break;
            }
        }
    }

    inline void setInstanceAttributesOverride(
            const InstanceAttributes* instanceAttributes) {
        if (instanceAttributes == nullptr) {
            return;
        }
        fillInstanceAttributes(instanceAttributes,
                               getTable()->getRequiredAttributes());
        fillInstanceAttributes(instanceAttributes,
                               getTable()->getOptionalAttributes());

        // Explicitly override attributes used for explicit shading
        fillInstanceAttributes(instanceAttributes,
                               { StandardAttributes::sExplicitShading,
                                 StandardAttributes::sNormal,
                                 StandardAttributes::sdPds,
                                 StandardAttributes::sdPdt });
    }

    /// This must be called after all calls to setInstanceAttributesOverride()
    // and before any calls to hasAllRequiredAttributes()
    inline void validateRequiredAttributes() {
        // Check whether all the required attributes from shading network is
        // properly filled in this Intersection
        bool hasAllRequiredAttributes = true;
        for (auto key : getTable()->getRequiredAttributes()) {
            if (!isProvided(key)) {
                hasAllRequiredAttributes = false;
                break;
            }
        }

        mFlags.set(HasAllRequiredAttributes, hasAllRequiredAttributes);
    }

    /// What is this intersection about
    finline const scene_rdl2::rdl2::Geometry* getGeometryObject() const {  return mGeometryObject;  }
    finline const scene_rdl2::rdl2::Material* getMaterial() const   {  return mMaterial;  }
    finline int32_t getLayerAssignmentId() const            {  return mLayerAssignmentId;  }
    finline void getIds(uint32_t &id1, uint32_t &id2, uint32_t &id3) const {
        id1 = mId1;  id2 = mId2;  id3 = mId3;
    }

    /// What should the next ray epsilon be
    finline float getEpsilonHint() const           {  return mEpsilonHint;  }

    /// What should the next shadow ray epsilon be
    finline float getShadowEpsilonHint() const           {  return mShadowEpsilonHint;  }

    /// Returns true if derivatives have been computed.
    finline bool hasGeometryDerivatives() const  { return mFlags.get(HasGeometryDerivatives); }

    /// Returns true if rendering flat points
    finline bool isFlatPoint() const  { return mFlags.get(IsFlatPoint); }

    /// Differential Geometry
    finline const scene_rdl2::math::Vec3f &getP() const        {  return mP;  }
    finline const scene_rdl2::math::Vec3f &getNg() const       {  return mNg; }
    finline const scene_rdl2::math::Vec3f &getN() const        {  return mN;  }
    finline const scene_rdl2::math::Vec2f &getSt() const       {  return reinterpret_cast<scene_rdl2::math::Vec2f const &>(mSt); }
    finline const scene_rdl2::math::Vec3f &getdPds() const     {  return mdPds;  }
    finline const scene_rdl2::math::Vec3f &getdPdt() const     {  return mdPdt;  }

    finline void setN(const scene_rdl2::math::Vec3f& n) { mN = n; }
    finline void setNg(const scene_rdl2::math::Vec3f& ng) { mNg = ng; }

    finline void setSt(const scene_rdl2::math::Vec2f& st) {
        mSt.x = st.x;
        mSt.y = st.y;
    }
    finline void setP(const scene_rdl2::math::Vec3f& p) { mP = p; }

    /// Initialize an intersection for evaluating a map shader.
    void initMapEvaluation(scene_rdl2::alloc::Arena *arena,
        const AttributeTable *const table,
        const scene_rdl2::rdl2::Geometry *rdlGeometry,
        const scene_rdl2::rdl2::Layer *rdlLayer, int assignmentId,
        const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &N,
        const scene_rdl2::math::Vec3f &dPds, const scene_rdl2::math::Vec3f &dPdt, const scene_rdl2::math::Vec2f &st,
        float dSdx, float dSdy, float dTdx, float dTdy)
    {
        init(rdlGeometry);
        mP = P;
        mNg = N;
        mN = N;
        mSt = st;
        mdPds = dPds;
        mdPdt = dPdt;
        mdSdx = dSdx;
        mdTdx = dTdx;
        mdSdy = dSdy;
        mdTdy = dTdy;
        mMinRoughness = scene_rdl2::math::Vec2f(0.0f);
        mWo = scene_rdl2::math::Vec3f(0.0f);
        if (rdlLayer && assignmentId >= 0) {
            setLayerAssignments(assignmentId, rdlLayer);
        }
        if (table) {
            setTable(arena, table);
        }
    }

    /// Initialize an intersection for displacement mapping. Only a subset
    /// of the members will be initialized.
    void initDisplacement(mcrt_common::ThreadLocalState *tls,
                          const AttributeTable *const table,
                          const scene_rdl2::rdl2::Geometry *rdlGeometry,
                          const scene_rdl2::rdl2::Layer *rdlLayer, int assignmentId,
                          const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec3f &N,
                          const scene_rdl2::math::Vec3f &dPds, const scene_rdl2::math::Vec3f &dPdt, const scene_rdl2::math::Vec2f &st,
                          float dSdx, float dSdy, float dTdx, float dTdy)
    {
        initMapEvaluation(&tls->mArena, table, rdlGeometry, rdlLayer, assignmentId, P, N,
            dPds, dPdt, st, dSdx, dSdy, dTdx, dTdy);
        mFlags.set(DisplacementInitialized);
    }


    /// This function calls RayDifferential::transfer and then computes the texture
    /// derivatives and scales them with textureDiffScale. It also takes care of the
    /// case where we previously had valid ray differentials but they became
    /// invalid during the transfer operation.
    /// See comments in implementation for more details.
    void transferAndComputeDerivatives(mcrt_common::ThreadLocalState *tls,
            mcrt_common::RayDifferential *ray, float textureDiffScale);

    //
    // transferAndComputeDerivatives must be called before it's valid to call
    // this function.
    //
    // Sort uvs to gain coherency when adding to the shade queue.
    // Bits  0-13   quantized swizzled uv coordinates, 14 bits = 16,384 mini-tiles.
    // Bits 14-17   mip level, lower resolution mips sorted earlier, 4 bits = 16 mip levels.
    // Bits 18-24   udim tile, lower resolution idx sorted earlier, 7 bits = 128 tiles.
    // Bits 25-31   lightset index, 7 bits = 128 light sets.
    //
    // If we exceed any of the limits in the number of bit allocated for each
    // category, then we clamp the value rather than wrap it.
    //
    finline uint32_t computeShadingSortKey() const;

    /// Normal derivatives
    finline scene_rdl2::math::Vec3f getdNdx() const
    {
        // mN is already flipped at this point, unflip it if necessary to compute
        // aux normals
        float sign = isEntering() ? 1.f : -1.f;
        scene_rdl2::math::Vec3f auxNormalX = normalize(mN * sign + mdNds * mdSdx + mdNdt * mdTdx);
        return (auxNormalX * sign) - mN;
    }

    finline scene_rdl2::math::Vec3f getdNdy() const
    {
        // mN is already flipped at this point, unflip it if necessary to compute
        // aux normals
        float sign = isEntering() ? 1.f : -1.f;
        scene_rdl2::math::Vec3f auxNormalY = normalize(mN * sign + mdNds * mdSdy + mdNdt * mdTdy);
        return (auxNormalY * sign) - mN;
    }

    /// Get texture coordinate differentials wrt. dx and dy. This is the ray
    /// footprint in texture space after scaling. Just what we need for texturing
    finline float getdSdx() const           { return mdSdx; }
    finline float getdSdy() const           { return mdSdy; }
    finline float getdTdx() const           { return mdTdx; }
    finline float getdTdy() const           { return mdTdy; }

    finline scene_rdl2::math::Vec3f const &getdNds() const    { return mdNds; }
    finline scene_rdl2::math::Vec3f const &getdNdt() const    { return mdNdt; }

    finline void setdPds(const scene_rdl2::math::Vec3f &dPds) {  mdPds = dPds;  }
    finline void setdPdt(const scene_rdl2::math::Vec3f &dPdt) {  mdPdt = dPdt;  }

    finline void setdSdx(float dsdx) { mdSdx = dsdx; }
    finline void setdSdy(float dsdy) { mdSdy = dsdy; }
    finline void setdTdx(float dtdx) { mdTdx = dtdx; }
    finline void setdTdy(float dtdy) { mdTdy = dtdy; }

    finline void setdNds(const scene_rdl2::math::Vec3f &dNds) { mdNds = dNds; }
    finline void setdNdt(const scene_rdl2::math::Vec3f &dNdt) { mdNdt = dNdt; }

    template <typename T>
    const T getdAttributedx(TypedAttributeKey<T> key) const {
        return getdAttributeds(key) * getdSdx() +
            getdAttributedt(key) * getdTdx();
    }

    template <typename T>
    const T getdAttributedy(TypedAttributeKey<T> key) const {
        return getdAttributeds(key) * getdSdy() +
            getdAttributedt(key) * getdTdy();
    }

    /// Is this "intersection" used for evaluating displacement ?
    finline bool isDisplacement() const  { return mFlags.get(DisplacementInitialized); }

    // Returns true if all shader required primitive attributes are provided.
    // validateRequiredAttributes() should be called first during initialization
    // of this Intersection
    finline bool hasAllRequiredAttributes() const {
        return mFlags.get(HasAllRequiredAttributes);
    }

    /// Are we entering the hit surface or leaving ?
    finline bool isEntering() const        { return mFlags.get(Entering); }
    finline void setIsEntering(bool entering) { mFlags.set(Entering, entering); }

    /// Is this on a caustic path ?
    finline bool isCausticPath() const   { return mFlags.get(CausticPath); }

    /// Is this on a subsurface path ?
    finline bool isSubsurfaceAllowed() const   { return mFlags.get(SubsurfaceAllowed); }

    /// What type of path leads to this intersection ?
    finline PathType getPathType() const    { return PathType((mFlags.getAll() & sPathTypeMask) >> sPathTypeOffset); }

    /// Is this indirect shading ?
    finline bool isIndirect() const      { return getPathType() != Primary; }

    /// Get the minimum roughness to be used for roughness clamping. We always
    /// guarantee that minRoughness.x <= minRoughness.y
    finline const scene_rdl2::math::Vec2f &getMinRoughness() const   {  return mMinRoughness;  }

    /// Convenience method to enforce convention on hi / low fidelity
    finline bool isHifi() const {
        const PathType pathType = getPathType();
        return (pathType == Intersection::PathType::Primary  ||
                pathType == Intersection::PathType::IndirectMirror  ||
                pathType == Intersection::PathType::LightPath);
    }

    /// Using 'wo' to create view-dependent effects in the shader is generally
    /// frowned upon as it likely breaks any future bi-directional techniques.
    /// However, we are forced to expose it in order to provide certain shading
    /// effects to production. Currently the only use case is the DirectionalMap
    finline const scene_rdl2::math::Vec3f &getWo() const { return mWo; }

    finline void setMediumIor(float ior) { mMediumIor = ior; }
    finline float getMediumIor() const { return mMediumIor; }

    /// Adapt a normal mapped normal to the current view direction
    finline void setUseAdaptNormal(bool useAdaptNormal) { mFlags.set(UseAdaptNormal, useAdaptNormal); }
    scene_rdl2::math::Vec3f adaptNormal(const scene_rdl2::math::Vec3f &Ns) const;
    scene_rdl2::math::Vec3f adaptToonNormal(const scene_rdl2::math::Vec3f &Ns) const;

    finline void setPathType(PathType type) {
        mFlags.set((uint32_t(type) << sPathTypeOffset) & sPathTypeMask);
    }

    finline void setIsCausticPath(bool causticPath) { mFlags.set(CausticPath, causticPath); }
    finline void setIsSubsurfaceAllowed(bool allowed) { mFlags.set(SubsurfaceAllowed, allowed); }
    finline void setMinRoughness(const scene_rdl2::math::Vec2f &minRoughness) {  mMinRoughness = minRoughness;  }
    finline void setWo(const scene_rdl2::math::Vec3f &wo) { mWo = wo; }

    // called at the end of geometry initializaion
    finline bool getGeomInitialized() { return mFlags.get(GeomInitialized); }
    finline void setGeomInitialized() { mFlags.set(GeomInitialized); }

    std::ostream& print(std::ostream& outs) const;

    // HVD validation.
    static uint32_t hvdValidation(bool verbose) { INTERSECTION_VALIDATION(VLEN); }

    // Invert Intersection Normals
    // One use-case is in the PathTracedSubsurface.cc to invert intersections
    // to be able to compute normal-mapping correctly
    finline void invertIntersectionNormals() {
        mNg = -mNg;
        mN  = -mN;
    }

private:
    int keyOffset(AttributeKey key) const {
        return getTable()->keyOffset(key);
    }

    static uint32_t intersectionHvdCrc(bool verbose)    { INTERSECTION_VALIDATION(VLEN); }

    /// Copy is disabled
    DISALLOW_COPY_OR_ASSIGNMENT(Intersection);

    typedef mcrt_common::Flags Flags;
    INTERSECTION_MEMBERS;

private:
    static const uint32_t sPathTypeMask = PathTypeLoBit
                                        | PathTypeMidBit
                                        | PathTypeHiBit;
    static const uint32_t sPathTypeOffset = 6;
};

namespace {

// Branchless int32_t min/max/clamp functions.
// TODO: These don't really belong here, move them into BitUtils.h next time
// scene_rdl2 is being updated.
finline int32_t minI32(int32_t a, int32_t b)
{
    int32_t diff = b - a;
    return a + (diff & (diff >> 31));
}

finline int32_t maxI32(int32_t a, int32_t b)
{
    int32_t diff = a - b;
    return a - (diff & (diff >> 31));
}

finline int32_t clampI32(int32_t v, int32_t min, int32_t max)
{
    return minI32(maxI32(v, min), max);
}

} // end anonymous namespace

template <>
inline const std::string&
Intersection::getAttribute(TypedAttributeKey<std::string> key) const
{
    MNRY_ASSERT(mData);
    char *readLoc = mData + keyOffset(key);
    return *(*(reinterpret_cast<std::string **>(readLoc)));
}

finline uint32_t
Intersection::computeShadingSortKey() const
{
    MNRY_ASSERT(mFlags.get(GeomInitialized));
    MNRY_ASSERT(getLayerAssignmentId() >= 0);

    // Divide up uvs into quads such that there are n quads over the 0-1 domain
    // in each dimension. Quads are themselves stored in Morton order.

    // n = 128 quads per dimension.
    uint64_t x = ((uint64_t)(mSt.x * 127.9999f)) & 0x7f;
    uint64_t y = ((uint64_t)(mSt.y * 127.9999f)) & 0x7f;

    // Interleave bits with 64-bit multiplies.
    // http://graphics.stanford.edu/~seander/bithacks.html#Interleave64bitOps
    // This interleaved multiply is what converts the coorindates from linear
    // to morton ordering.
    uint32_t uv = ((x * 0x0101010101010101ULL & 0x8040201008040201ULL) *
                  0x0102040810204081ULL >> 49) & 0x5555 |
                  ((y * 0x0101010101010101ULL & 0x8040201008040201ULL) *
                  0x0102040810204081ULL >> 48) & 0xAAAA;

    float mipSelector = computeMipSelector(mdSdx, mdTdx, mdSdy, mdTdy);

    uint32_t ms = (uint32_t)minI32(int(mipSelector), 15);

    // Convert uv coordinates into a udim index.
    int32_t tileU = clampI32(int(mSt.x), 0, 9);
    int32_t tileV = maxI32(int(mSt.y), 0);
    uint32_t udim = (uint32_t)minI32(tileV * 10 + tileU, 127);

    // Entries without light set assignments are placed at the end of the
    // sorted list.
    uint32_t ls = (uint32_t)minI32(getLayerAssignmentId(), 127);

    MNRY_ASSERT(uv < 16384);
    MNRY_ASSERT(ms < 16);
    MNRY_ASSERT(udim < 128);
    MNRY_ASSERT(ls < 128);

    return (ls << 25) | (udim << 18) | (ms << 14) | uv;
}


finline std::ostream& operator<<(std::ostream& outs, const Intersection& i)
{
    return i.print(outs);
}


MNRY_STATIC_ASSERT(Intersection::PathTypeCount == 5);

struct CACHE_ALIGN Intersectionv
{
    uint8_t mPlaceholder[sizeof(Intersection) * VLEN];
};

MNRY_STATIC_ASSERT(sizeof(Intersection) * VLEN == sizeof(Intersectionv));

} // namespace shading
} // namespace moonray



