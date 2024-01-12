// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Bsdf.h
/// $Id$
///

#pragma once

#include "Fresnel.h"

#include <moonray/rendering/shading/ispc/bsdf/Bsdf_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>
#include <scene_rdl2/render/util/Arena.h>
#include <moonray/common/mcrt_util/StaticVector.h>


namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

// General Convention:
// - All directions below wo and wi are in the global shading coordinate system
//   of the caller (typically render space, but no assumption is made about this).
//   Internally, a BsdfLobe can use a ReferenceFrame class to
//   transform directions from/to global coordinate system to/from the local
//   reference frame. See the ReferenceFrame for more details.
// - All directions wo and wi are guaranteed to be normalized and pointing
//   away from the surface.
// - We follow the convention that the global space N and Ng are flipped
//   together so that Ng is pointing towards the observer direction wo -- i.e.
//   reversed incoming eye ray direction / reversed incoming photon direction
//   in the adjoint case.
//   If no flipping happens, the slice entering flag must be set to true, and
//   must be set to false otherwise. This flag indicates if we are entering or
//   leaving the volume inside an object.
// - By the above conventions, the passed-in wo is guaranteed to be in the
//   positive geometric hemisphere (wrt. Ng). Part of that hemisphere may be in
//   the negative shading hemisphere (wrt. N) when bump mapping or/and
//   interpolated vertex normals are in effect.
// - See more specific rules in the API below.


// Forward decl.
class BsdfSlice;
class Bssrdf;
class VolumeSubsurface;

//----------------------------------------------------------------------------

///
/// @class BsdfLobe BsdfLobe.h <shading/BsdfLobe.h>
/// @brief Define the BsdfLobe interface. You can derive this class to define
/// specific (single) Bsdf lobes.
///
class BsdfLobe
{
public:

    /// This is used to select bsdfs that contribute to reflection / transmission
    /// or/and to select bsdfs that are categorized as diffuse / glossy / ideal mirror.
    /// They are organized such that they can easily be translated into ray masks
    /// (rdl2::Geometry::VisibilityType) by lobeTypeToRayMask()(PathIntegratorUtil.h).
    enum Type {
        NONE = 0,

        // Surface side categories. Most lobes belong to only one category, but
        // there are some exceptions.
        REFLECTION   = 1 << 0,
        TRANSMISSION = 1 << 1,

        // Lobe categories. These ARE mutually exclusive. A lobe can only
        // belong to one of the categories below.
        // The bit shift assignments are for easy determination of ray masking based on
        // lobe type.
        DIFFUSE      = 1 << 2,
        GLOSSY       = 1 << 4,
        MIRROR       = 1 << 6,

        // Mixed bags
        ALL_SURFACE_SIDES = REFLECTION | TRANSMISSION,
        ALL_LOBES         = DIFFUSE | GLOSSY | MIRROR,
        ALL_REFLECTION    = ALL_LOBES | REFLECTION,
        ALL_TRANSMISSION  = ALL_LOBES | TRANSMISSION,
        ALL_DIFFUSE       = ALL_SURFACE_SIDES | DIFFUSE,
        ALL_GLOSSY        = ALL_SURFACE_SIDES | GLOSSY,
        ALL_MIRROR        = ALL_SURFACE_SIDES | MIRROR,
        ALL               = ALL_REFLECTION | ALL_TRANSMISSION
    };

    enum DifferentialFlags {
        // Set this flag if the differentials function doesn't take the incoming
        // differentials into account. There is already some footprint scaling
        // information built into the differentials, so if this function ignores
        // them, we need to re-apply those scales again.
        IGNORES_INCOMING_DIFFERENTIALS = 1 << 1, 
    };


    enum Property {
        PROPERTY_NONE         = 0,
        PROPERTY_NORMAL       = 1 << 0,
        PROPERTY_ROUGHNESS    = 1 << 1,
        PROPERTY_COLOR        = 1 << 2,
        PROPERTY_PBR_VALIDITY = 1 << 3
    };

    /// Constructor / Destructor
    /// isSpherical tells us if the bsdf scatters in the lobe's hemisphere, of
    /// if it scatters in the full spherical domain. Surface BsdfLobes should
    /// pass false for isSpherical and curve / hair BsdfLobes should pass true.
    BsdfLobe(Type type, DifferentialFlags diffFlags, bool isSpherical, int32_t propertyFlags);
    virtual ~BsdfLobe();

    
    
    // TODO: add a flag so lobes can contribute to direct / indirect / both


    /// Return the bsdf's type and spherical flags
    finline Type getType() const                    {  return mType;  }
    finline void setType(Type type)                 {  mType = type;  }
    finline DifferentialFlags getDifferentialFlags() const {  return mDifferentialFlags;  }
    finline bool getIsSpherical() const             {  return mIsSpherical;  }
    finline void setIsSpherical(bool isSpherical)   {  mIsSpherical = isSpherical;  }

    // Does this lobe belong to hair material?
    finline void setIsHair(bool isHair) { mIsHair  = isHair; }
    finline bool getIsHair() const { return mIsHair; }

    /// Does this bsdf type match the given set of flags ? Returns true if at
    /// least one of the "surface side" bits matches the flags and at least one
    /// of the "lobe" bits matches the flags.
    finline bool matchesFlags(Type flags) const  {
        return matchesFlags(static_cast<int>(flags));
    }
    finline bool matchesFlags(int flags) const {
        return ((mType & ALL_SURFACE_SIDES & flags)  &&
                (mType & ALL_LOBES & flags));
    }

    /// Does this bsdf type match the given flag ? This is useful for testing
    /// flags other than boolean unions (i.e. flags other than ALL_*)
    finline bool matchesFlag(Type flag) const  {  return (mType & flag);  }

    /// A bsdf lobe can be scaled by a color (and by a fresnel reflectance).
    /// It is the responsibility of the derived class' implementation to include
    /// the color scaling (and fresnel if any) in the result of eval() and sample()
    finline void setScale(const scene_rdl2::math::Color &scale)     {  mScale = scale;  }
    finline const scene_rdl2::math::Color &getScale() const         {  return mScale;  }

    /// The bsdf lobe DOES take ownership of the Fresnel object
    void setFresnel(Fresnel *fresnel);
    finline const Fresnel *getFresnel() const       {  return mFresnel;  }

    /// A label can be set on a lobe
    finline void setLabel(int label) { mLabel = label; }
    finline int  getLabel() const { return mLabel; }

    /// lobes can have certain properties (e.g. Normals, Specular Roughness)
    /// this API is used to query if a lobe has a particular property and to
    /// evaluate that property value
    bool hasProperty(Property property) const { return mPropertyFlags & property; }
    int32_t getPropertyFlags() const { return mPropertyFlags; }
    virtual bool getProperty(Property property, float *dest) const;

    scene_rdl2::math::Color getScale() { return mScale; }

    /// Convenience method for derived classes to include color and fresnel
    /// scaling contributions
    finline scene_rdl2::math::Color computeScaleAndFresnel(float cosWi) const {
        return mScale * computeFresnel(cosWi);
    }

    finline scene_rdl2::math::Color computeFresnel(float cosWi) const
            {  return (mFresnel) ? (mFresnel->eval(cosWi)) : scene_rdl2::math::sWhite;  }

    /// Evaluate the bsdf and (optionaly) the probability density of sampling the
    /// direction wi (density measured wrt. solid angle) for the pair of
    /// directions. wo is the observer or fixed direction and wi is the
    /// light / varying direction. This function should behave consistently with
    /// the sample() method below.
    /// Rules:
    /// - The returned bsdf should or should not include the cosine term of the
    ///   rendering equation, according to the slice.getIncludeCosineTerm() setting
    /// - The bsdf should include the fresnel effect if any
    /// - The returned pdf should always be positive. Zero should only be
    ///   returned if the sample() method actually has zero probability of
    ///   choosing wi.
    /// - This function shouldn't need to test if wo and wi are in the same
    ///   hemisphere. If that is not the case, the function should still
    ///   evaluate the bsdf return the probability density of choosing wi (unless
    ///   this is required for numerical stability of the underlying bsdf model).
    /// - Perfect mirror reflection / transmission Bsdfs should always return 0
    ///   for the bsdf and the pdf.
    /// THREAD-SAFETY WARNING!!!
    /// This API is not thread safe because it allows BsdfLobe to cache
    /// computations that are redundant across lobes inside the Bsdf
    /// class. This is only an issue if multiple threads need to sample / eval
    /// the same hair Bsdf (lobe) instance. We only really need to do this in
    /// our unittests, so there we're careful to use a copy of the bsdf per thread
    virtual scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi,
            float *pdf = nullptr) const = 0;

    /// Sample the bsdf for the given observer (fixed) direction wo.
    /// This function should sample an incident direction wi and return
    /// the probability density of choosing this direction given wo (density
    /// measured wrt. solid angle). The returned pdf and evaluated color must
    /// match the value returned by eval(slice, wi).
    /// Rules:
    /// - This function may return a wi direction that may be anywhere in the
    ///   full spherical domain. It is up to the caller to make sure this
    ///   wi is tested before calling eval() further.
    /// - This function shouldn't need to test if wo and wi are in the same
    ///   geometric hemisphere. If that is the case, the function should still
    ///   return the probability density of choosing wi.
    /// - A zero color sample may be returned if that is the result
    ///   of eval(slice, wi).
    /// - A sample with zero probability may be returned if the function is
    ///   unable to sample given its inputs, in which case wi may be left
    ///   un-initialized. However, the pdf() should behave consistently and still
    ///   integrate to one over the hemisphere, otherwise this will result in
    ///   biased importance sampling.
    /// - Perfect mirror reflection / transmission Bsdfs should sample the
    ///   mirror direction and return the proper contribution and a
    ///   probability of 1.
    /// - Do not apply russian-roulette inside sample(), this should be done by
    ///   higher-level code.
    /// - Bsdfs that rely on sampling half-vectors should make sure that
    ///   dot(wo, H) is less than 90 degrees, so the re-constructed H vector
    ///   in eval() is consistently oriented with the sampled H vector
    /// THREAD-SAFETY WARNING!!!
    /// This API is not thread safe because it allows BsdfLobe to cache
    /// computations that are redundant across lobes inside the Bsdf
    /// class. This is only an issue if multiple threads need to sample / eval
    /// the same hair Bsdf (lobe) instance. We only really need to do this in
    /// our unittests, so there we're careful to use a copy of the bsdf per thread
    virtual scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const = 0;

    /// Compute a fast approximation of the albedo. The implementation can
    /// assume the bsdf lobe is normalized, which means the un-scaled bsdf
    /// function should always integrate to 1. Therefore the
    /// implementation can just return the scale color * fresnel. The fresnel
    /// term can also be computed assuming an ideal mirror reflection direction
    /// as a further approximation.
    /// This is only used for driving lobe sampling decisions and should only
    /// affect the noise in the image, never the actual lighting intensity.
    virtual scene_rdl2::math::Color albedo(const BsdfSlice &slice) const = 0;

    /// Update the ray direction differentials (dDdx, dDdy), for a given
    /// scattering event (wo, wi). H is the normalized half-vector.
    /// The normal derivatives (dNdx, dNdy) are also provided to
    /// take into account the surface curvature. The differentials are
    /// normally computed by differentiating the sampling routine recursively
    /// along the path wrt. the random variables used to sample the path.
    /// This is sometimes overly complex for "blurry" scattering events and
    /// a simpler approximation is enough (i.e. taking only the diffuse
    /// sampling into account for diffuse scattering). The routine should assume
    /// only one sample is taken over the entire domain and the renderer will
    /// scale the differentials automatically when path splitting happens.
    //
    /// TODO: Implement all BsdfLobes
    /// TODO: It's unclear at the moment if this math should be part of the
    /// sample() method. Right now it looks like a lot of redundant math which
    /// could be avoided, but for the sake of expediency and for an initial test,
    /// I will stick to this for now.
    virtual void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const = 0;

    // prints out a description of this lobe with the provided indentation
    // prepended.
    virtual void show(std::ostream& os, const std::string& indent) const = 0;

private:

    Type mType;
    const DifferentialFlags mDifferentialFlags;
    bool mIsSpherical;
    bool mIsHair;
    scene_rdl2::math::Color mScale;
    Fresnel *mFresnel;
    int mLabel;
    int32_t mPropertyFlags;
};


/// Utility functions to compute the minimum roughness to be used during
/// roughness clamping
finline scene_rdl2::math::Vec2f
computeMinRoughness(const BsdfLobe &lobe, float roughnessClampingFactor,
                    const scene_rdl2::math::Vec2f &previousMinRoughness)
{
    scene_rdl2::math::Vec2f minRoughness(0.0f);

    if (roughnessClampingFactor > 0.0f) {
        scene_rdl2::math::Vec2f roughness;
        if (lobe.getProperty(BsdfLobe::PROPERTY_ROUGHNESS, &roughness[0])) {
            minRoughness = roughness * roughnessClampingFactor;
        }

        // Keep smaller of the two min roughnesses in x
        if (minRoughness.x > minRoughness.y) {
            std::swap(minRoughness.x, minRoughness.y);
        }
    }

    minRoughness = scene_rdl2::math::clamp(minRoughness, previousMinRoughness, scene_rdl2::math::Vec2f(1.0f));
    return minRoughness;
}

finline scene_rdl2::math::Vec2f
computeMinRoughness(float lobeRoughness, float roughnessClampingFactor,
        const scene_rdl2::math::Vec2f &previousMinRoughness)
{
    scene_rdl2::math::Vec2f minRoughness(0.0f);

    if (roughnessClampingFactor > 0.0f) {
        minRoughness = scene_rdl2::math::Vec2f(lobeRoughness * roughnessClampingFactor);
    }

    minRoughness = scene_rdl2::math::clamp(minRoughness, previousMinRoughness, scene_rdl2::math::Vec2f(1.0f));
    return minRoughness;
}


/// Convenience functions to set/clear flags
finline void setFlag(BsdfLobe::Type &type, BsdfLobe::Type flag)
        {  type = BsdfLobe::Type(type | flag);  }
finline void clearFlag(BsdfLobe::Type &type, BsdfLobe::Type flag)
        {  type = BsdfLobe::Type(type & ~flag);  }


//----------------------------------------------------------------------------

///
/// @class Bsdf Bsdf.h <shading/Bsdf.h>
/// @brief Define the Bsdf object, used to represent multi-lobe bsdfs.
///
class Bsdf final
{
public:
    static const std::size_t maxLobes = 16;

    // storage for extra aov evaluations that are accumulated
    // after ray scattering.
    struct ExtraAovs
    {
        ExtraAovs() : mNum(0), mLabelIds(nullptr), mColors(nullptr) {}
        ExtraAovs(int num, const int *labelIds, const scene_rdl2::math::Color *colors):
            mNum(num), mLabelIds(labelIds), mColors(colors)
        {
        }

        int mNum;
        const int *mLabelIds;
        const scene_rdl2::math::Color *mColors;
    };

    /// Constructor / Destructor
    Bsdf() : mLobeArray(), mBssrdf(nullptr), mVolumeSubsurface(nullptr),
        mSelfEmission(scene_rdl2::math::sBlack),
        mEarlyTermination(false),
        mType(BsdfLobe::Type(0)), mIsSpherical(false),
        mMaterialLabelId(-1), mLpeMaterialLabelId(-1),
        mGeomLabelId(-1)
    {}
    ~Bsdf();

    /// Add Lobes. The Bsdf DOES take ownership of the lobes. Adding a lobe
    /// instance more than once is not supported, it will cause it to be deleted
    /// more than once in ~Bsdf().
    finline void addLobe(BsdfLobe *lobe) {
        MNRY_ASSERT(mLobeArray.size() < maxLobes);
        mType = BsdfLobe::Type(unsigned(mType) | unsigned(lobe->getType()));
        mIsSpherical |= lobe->getIsSpherical();
        mLobeArray.push_back(lobe);
    }

    /// How many lobes do we have (that match the given flags)
    finline int getLobeCount() const    {  return mLobeArray.size();  }
    int getLobeCount(BsdfLobe::Type flags) const;

    /// Get a specific lobe
    finline const BsdfLobe *getLobe(int index) const
            {  return mLobeArray[index];  }
    finline BsdfLobe *getLobe(int index)
            {  return mLobeArray[index];  }

    /// Set/get a Bssrdf. The Bssrdf is owned by the Bsdf and will be destroyed
    /// in the Bsdf destructor.
    /// TODO: This is a temporary shortcut. We probably want this to be returned
    /// by a surface shader separately from the Bsdf.
    finline void setBssrdf(Bssrdf *bssrdf)     {  mBssrdf = bssrdf;  }
    finline const Bssrdf *getBssrdf() const    {  return mBssrdf;  }
    finline Bssrdf *getBssrdf()                {  return mBssrdf;  }

    finline void setVolumeSubsurface(VolumeSubsurface* volumeSubsurface) {
        mVolumeSubsurface = volumeSubsurface;
    }

    finline const VolumeSubsurface* getVolumeSubsurface() const {
        return mVolumeSubsurface;
    }
    finline VolumeSubsurface* getVolumeSubsurface() {
        return mVolumeSubsurface;
    }

    finline bool hasSubsurface() const {
        return (mBssrdf != nullptr ||
                mVolumeSubsurface != nullptr);
    }

    /// Set/get self emission
    /// TODO: This is a temporary shortcut. We probably want this to be returned
    /// by a surface shader separately from the Bsdf.
    finline void setSelfEmission(const scene_rdl2::math::Color &color)   {  mSelfEmission = color;  }
    finline const scene_rdl2::math::Color &getSelfEmission() const       {  return mSelfEmission;  }

    /// Signals that the BSDF should not be sampled, and tracing should
    /// immediately terminate.
    finline void setEarlyTermination() {
        mEarlyTermination = true;
    }
    finline const bool& getEarlyTermination() const { return mEarlyTermination; }

    /// Returns the type of all contained BsdfLobes or'ed together.
    BsdfLobe::Type getType() const  {  return mType;  }

    /// Returns true if at least one lobe is spherical and false otherwise.
    bool getIsSpherical() const     {  return mIsSpherical;  }

    /// The renderer must call this function after shading and before
    /// integration.
    /// @param materialLabelId a single value that maps the label attribute value
    ///                        of the material that produced this Bsdf to the
    ///                        label id used in material aov expressions.
    ///
    /// @param lpeMaterialLabelId a single value that maps the label attribute value
    ///                           of the material that produced this Bsdf to the
    ///                           label id used in light path aov expressions.
    ///
    /// @param geomLabelId a single value that maps the label attribute value
    ///                    of the geom that produced this Bsdf to the
    ///                    label id used in material aov expressions
    void setLabelIds(int materialLabelId, int lpeMaterialLabelId,
                     int geomLabelId)
    {
        mMaterialLabelId    = materialLabelId;
        mLpeMaterialLabelId = lpeMaterialLabelId;
        mGeomLabelId        = geomLabelId;
    }

    /// @return id of material label used in material aov expressions
    int  getMaterialLabelId() const { return mMaterialLabelId; }

    /// @return id of material label used in light path aov expressions
    int  getLpeMaterialLabelId() const { return mLpeMaterialLabelId; }

    /// @return id of geom label used in material aov expressions
    int getGeomLabelId() const { return mGeomLabelId; }

    /// The following evaluation API follows mostly the same rules and conventions
    /// from the BsdfLobe class. The pdf returned by eval assumes equiprobability
    /// of all lobes within the Bsdf.
    scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float &pdf) const;
    scene_rdl2::math::Color albedo(const BsdfSlice &slice) const;

    // prints out a description of this Bsdf
    void show(const std::string &sceneClass,
              const std::string &name,
              std::ostream& os) const;

    void setPostScatterExtraAovs(int numExtraAovs, const int * labelIds, const scene_rdl2::math::Color *colors);
    const ExtraAovs &getPostScatterExtraAovs() const { return mPostScatterExtraAovs; }

private:
    typedef util::StaticVector<BsdfLobe *, maxLobes> BsdfLobePtrArray;

    Bsdf(const Bsdf &other) = delete;
    Bsdf &operator=(const Bsdf &other) = delete;

    BsdfLobePtrArray        mLobeArray;
    Bssrdf *                mBssrdf;
    VolumeSubsurface *      mVolumeSubsurface;
    scene_rdl2::math::Color mSelfEmission;
    bool                    mEarlyTermination;

    /// The type of all contained lobes or'ed together.
    BsdfLobe::Type mType;

    bool mIsSpherical;

    /// See setLabelIds() comment for explanation
    int mMaterialLabelId;
    int mLpeMaterialLabelId;
    int mGeomLabelId;

    ExtraAovs mPostScatterExtraAovs;
};


//----------------------------------------------------------------------------

// ispc vector types
ISPC_UTIL_TYPEDEF_STRUCT(Bsdf, Bsdfv)
ISPC_UTIL_TYPEDEF_STRUCT(BsdfLobe, BsdfLobev)
typedef ispc::BsdfLobeName BsdfLobeName;

} // namespace shading
} // namespace moonray


