// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Distribution.hh"
#include <moonray/rendering/geom/prim/Util.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/scene/rdl2/Light.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat3.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/platform/HybridUniformData.h>
#include <OpenImageIO/imageio.h>

#include <string>

// Forward declaration of the ISPC types
namespace ispc {
    struct GuideDistribution1D;
    struct Distribution2D;
    struct ImageDistribution;
}


namespace moonray {
namespace pbr {


// Keep this in sync with scene_rdl2/lib/scene/rdl2/Light.h
enum TextureFilterType {
    TEXTURE_FILTER_ENUM
};
MNRY_STATIC_ASSERT(static_cast<int>(TEXTURE_FILTER_NEAREST) ==
                  static_cast<int>(scene_rdl2::rdl2::TextureFilterType::TEXTURE_FILTER_NEAREST));
MNRY_STATIC_ASSERT(static_cast<int>(TEXTURE_FILTER_BILINEAR) ==
                  static_cast<int>(scene_rdl2::rdl2::TextureFilterType::TEXTURE_FILTER_BILINEAR));
MNRY_STATIC_ASSERT(static_cast<int>(TEXTURE_FILTER_NEAREST_MIP_NEAREST) ==
                  static_cast<int>(scene_rdl2::rdl2::TextureFilterType::TEXTURE_FILTER_NEAREST_MIP_NEAREST));
MNRY_STATIC_ASSERT(static_cast<int>(TEXTURE_FILTER_BILINEAR_MIP_NEAREST) ==
                  static_cast<int>(scene_rdl2::rdl2::TextureFilterType::TEXTURE_FILTER_BILINEAR_MIP_NEAREST));
MNRY_STATIC_ASSERT(static_cast<int>(TEXTURE_FILTER_NUM_TYPES) ==
                  static_cast<int>(scene_rdl2::rdl2::TextureFilterType::TEXTURE_FILTER_NUM_TYPES));

//----------------------------------------------------------------------------

///
/// @class GuideDistribution1D Distribution.h <pbr/core/Distribution.h>
/// @brief A utility object that can sample according to a a piecewise constant
/// 1D distribution. This class uses the method of guide table from Chen and
/// Asau (1974):
///     Chen, H.-C. and Asau, Y. (1974). On generating random variates from an
///     empirical distribution. AIEE transactions, 6:153–166.
/// We store a cdf table and a guide table of the same size.
/// 
class GuideDistribution1D
{
public:
    typedef uint32_t size_type;

    /// Constructor / Destructor
    /// Weights and cdf are left un-initialized
    explicit GuideDistribution1D(const size_type size);
    explicit GuideDistribution1D(const size_type size, float *const cdf, uint32_t *const guide);
    GuideDistribution1D(GuideDistribution1D&& other);
    GuideDistribution1D& operator=(GuideDistribution1D&& other);

    ~GuideDistribution1D();


    /// HUD validation and type casting
    static uint32_t hudValidation(const bool verbose)
    {
        GUIDE_DISTRIBUTION_1D_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(GuideDistribution1D);


    /// Accessors
    finline size_type getSize() const
    {
        return mSizeCdf;
    }

    finline float getInvSize() const
    {
        return mInvSizeCdf;
    }

    finline float getTotalWeight() const
    {
        return mTotalWeight;
    }

    finline float getThresholdLow() const
    {
        return mThresholdLow;
    }

    finline float getThresholdHigh() const
    {
        return mThresholdHigh;
    }

    finline float getLinearCoeffLow() const
    {
        return mLinearCoeffLow;
    }

    finline float getLinearCoeffHigh() const
    {
        return mLinearCoeffHigh;
    }

    finline size_type getGuideIndex(const float r) const
    {
        MNRY_ASSERT(r >= 0.0f  &&  r <= 1.0f);
        const float rj = r * mSizeGuide;
        size_type j = static_cast<size_type>(rj);
        j = scene_rdl2::math::min(j, mSizeGuide - 1);
        return mGuide[j];
    }

    finline const float *getCdf() const
    {
        return mCdf;
    }


    /// Setup the distribution weight function. Make sure to initialize each
    /// weight with a value strictly > 0.0. The weights are normalized into a
    /// proper pdf during tabulateCdf().
    finline void setWeight(const size_type index, const float weight)
    {
        mCdf[index] = weight;
    }

    /// Helper to scale an existing weight.
    finline void scaleWeight(const size_type index, const float scale)
    {
        mCdf[index] *= scale;
    }

    /// Make sure to compute the cdf when you're done setting ALL weights.
    /// Returns the integral of the weight function.
    float tabulateCdf();

    float pdfDiscrete(const size_type index) const;

    float pdfContinuous(const float u) const;
    
    float pdfLinear(const float u) const;

    /// Discrete sampling: returns the index of the segment in the distribution
    /// corresponding to the uniform parameter r in [0,1)
    /// Optionally returns the pdf of sampling this segment of the distribution
    /// Returns the index of the interval in the discrete distribution where the value r falls.
    /// rRemapped represents the fractional distance of r along that interval. 
    size_type sampleDiscrete(const float r, float *const pdf = nullptr, float *const rRemapped = nullptr) const;

    /// Continuous sampling: returns the continuous variable u in [0,1), with
    /// density proportional to the distribution, that corresponds to the
    /// uniform parameter r in [0,1)
    /// Optionally returns the pdf of sampling the value u
    /// Optionally returns the index of the sampled segment
    float sampleContinuous(const float r, float *const pdf = nullptr, size_type *const index = nullptr) const;

    /// Sample from a linearly filtered distribution
    float sampleLinear(const float r, float *const pdf = nullptr) const;

private:
    /// Copy is disabled
    GuideDistribution1D(const GuideDistribution1D &other);
    const GuideDistribution1D &operator=(const GuideDistribution1D &other);

    // Members
    GUIDE_DISTRIBUTION_1D_MEMBERS;
};


//----------------------------------------------------------------------------

// WARNING: Make sure this matches the ISPC Distribution2D implementation
typedef GuideDistribution1D Distribution1D;

//----------------------------------------------------------------------------

///
/// @class Distribution2D Distribution.h <pbr/core/Distribution.h>
/// @brief A utility object that can sample according to a piecewise constant
/// 2D distribution.
///
class Distribution2D
{
public:
    enum Mapping {
        NONE,           ///< Undefined / un-initialized
        PLANAR,         ///< Weighted by the image luminance, no-op.
        CIRCULAR,       ///< Used for disk light texture sampling, pixels outside of 
                        ///< a maximal inscribed circle are set to black so we don't
                        ///< sample them.
        HEMISPHERICAL,  ///< Assuming a latlong input, the upper half of image is
                        ///< weighted by math::sin(2pi * v), see (see pbrt section 14.6.5).
                        ///< The bottom half of image is set to zero so it's not
                        ///< accessed. (Currently the memory for the bottom half
                        ///< of the image weights goes to waste)
        SPHERICAL,      ///< Weighted by the image luminance * math::sin(2pi * v) over
                        ///< the full image.
    };

    typedef Distribution1D::size_type size_type;

    /// Constructor / Destructor
    /// Weights and cdf are left un-initialized
    Distribution2D(size_type sizeU, size_type sizeV);
    ~Distribution2D();

    /// HUD validation and type casting
    static uint32_t hudValidation(const bool verbose)
    {
        DISTRIBUTION_2D_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(Distribution2D);

    /// Accessors
    finline size_type getSizeU() const
    {
        return mConditional[0]->getSize();
    }

    finline size_type getSizeV() const
    {
        return mSizeV;
    }

    /// Setup the distribution weight function. Make sure to initialize each
    /// weight with a value strictly > 0.0. The weights are normalized into a
    /// proper pdf during tabulateCdf().
    finline void setWeight(const size_type u, const size_type v, const float weight)
    {
        mConditional[v]->setWeight(u, weight);
    }

    /// Helper to scale an existing weight.
    finline void scaleWeight(const size_type u, const size_type v, const float scale)
    {
        mConditional[v]->scaleWeight(u, scale);
    }

    /// Make sure to compute the cdf when you're done setting ALL weights.
    /// At this point a optional mapping can also be applied.
    void tabulateCdf(const Mapping mapping = PLANAR);

    float pdfNearest(const float u, const float v) const;
    float pdfBilinear(const float u, const float v) const;

    /// Continuous sampling: returns the continuous variable u in [0,1), with
    /// density proportional to the distribution, that corresponds to the
    /// uniform parameter ru in [0,1) (and similarly for v & rv).
    /// These (u,v) values are passed back in *uv. The pointer uv must be non-null.
    /// Optionally returns the pdf of sampling this segment of the distribution
    finline void
    sample(const float ru, const float rv, scene_rdl2::math::Vec2f *const uv, float *const pdf,
           const TextureFilterType texFilter) const
    {
        MNRY_ASSERT(uv != nullptr);
        switch (texFilter) {
        case TEXTURE_FILTER_NEAREST:
            sampleNearest(ru, rv, uv, pdf);
            break;
        case TEXTURE_FILTER_BILINEAR:
            sampleBilinear(ru, rv, uv, pdf);
            break;
        default:
            sampleNearest(ru, rv, uv, pdf);
            break;
        }
    }

    void sampleNearest(const float ru, const float rv, scene_rdl2::math::Vec2f *const uv, float *const pdf) const;
    void sampleBilinear(const float ru, const float rv, scene_rdl2::math::Vec2f *const uv, float *const pdf) const;

private:
    /// Copy is disabled
    Distribution2D(const Distribution2D &other);
    const Distribution2D &operator=(const Distribution2D &other);

    // Members
    DISTRIBUTION_2D_MEMBERS;
};


//----------------------------------------------------------------------------

///
/// @class ImageDistribution Distribution.h <pbr/core/Distribution.h>
/// @brief A utility object that can sample according to an image
///
class ImageDistribution
{
public:
    typedef Distribution2D::size_type size_type;

    /// Constructor / Destructor
    /// Pass in a .exr image filename, it's sampling weighting for the desired
    /// mapping.
    ImageDistribution(const std::string &mapFilename,
                      const Distribution2D::Mapping mapping);

    /// This Constructor takes all Image Based Lighting Color Correction (IBLCC) inputs,
    /// and performs the color corrections *before* determining the luminence / energy
    /// distribution.
    /// Most inputs will be familiar to users but TME (TMI) is explained here:
    /// TME (TMI) Temperature Color Correction
    /// One parameter adjusts the color temperature by moving red and blue
    /// in different directions (you don t have to specify a proper temperature
    /// like 5300K). The second parameter is a shift along the magenta/green axis
    /// Color Temperature (T): Red Gain = 1   T/2,
    ///                        Blue Gain = 1 + T/2
    /// Magenta/Green (M): Red Gain = 1 + M/3,
    ///                    Green Gain = 1   M*2/3,
    ///                    Blue Gain = 1 + M/3
    /// Exposure (E): RGB Gain *= 2.0^E
    ImageDistribution(const std::string &mapFilename,
                      const Distribution2D::Mapping mapping,
                      const scene_rdl2::math::Color& gamma,
                      const scene_rdl2::math::Color& contrast, 
                      const scene_rdl2::math::Color& saturation,
                      const scene_rdl2::math::Color& gain,
                      const scene_rdl2::math::Color& offset,
                      const scene_rdl2::math::Vec3f& temperature,
                      const float  rotationAngle,
                      const scene_rdl2::math::Vec2f& translation,
                      const scene_rdl2::math::Vec2f& coverage,
                      const float  repsU,
                      const float  repsV,
                      const bool   mirrorU,
                      const bool   mirrorV,
                      const scene_rdl2::math::Color& borderColor);
    ~ImageDistribution();

    /// HUD validation and type casting
    static uint32_t hudValidation(const bool verbose)
    {
        IMAGE_DISTRIBUTION_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(ImageDistribution);

    /// If this returns true then you can use the rest of the API below
    finline bool isValid() const
    {
        return mDistribution != nullptr;
    }

    int getWidth()  const { return mWidth;  }
    int getHeight() const { return mHeight; }

    /// Return the pdf of sampling the given (u, v) value
    /// Note: the mapping only affects the weight of the distribution
    /// It's up to the caller to transform the returned pdf wrt. the proper
    /// measure.
    float pdf(const float u, const float v, const float mipLevel, const TextureFilterType texFilter) const;

    /// Return the color in the image at the given uv location
    scene_rdl2::math::Color eval(const float u, const float v, const float mipLevel, const TextureFilterType texFilter) const;

    /// Continuous sampling: returns the continuous variable u in [0,1), with
    /// density proportional to the distribution, that corresponds to the
    /// uniform parameter ru in [0,1) (and similarly for v & rv).
    /// These (u,v) values are passed back in *uv. The pointer uv must be non-null.
    /// Optionally returns the pdf of sampling this segment of the distribution
    void sample(const float ru, const float rv, const float mipLevel, scene_rdl2::math::Vec2f *const uv, float *const pdf,
                const TextureFilterType texFilter) const;


private:
    /// Copy is disabled
    ImageDistribution(const ImageDistribution &other);
    const ImageDistribution &operator=(const ImageDistribution &other);

    void init(const std::string&             mapFilename,
              const Distribution2D::Mapping  mapping,
              const scene_rdl2::math::Color& gamma,
              const scene_rdl2::math::Color& contrast,
              const scene_rdl2::math::Color& saturation,
              const scene_rdl2::math::Color& gain,
              const scene_rdl2::math::Color& offset,
              const scene_rdl2::math::Vec3f& temperature,
              const float                    rotationAngle,
              const scene_rdl2::math::Vec2f& translation,
              const scene_rdl2::math::Vec2f& coverage,
              const float                    repsU,
              const float                    repsV,
              const bool                     mirrorU,
              const bool                     mirrorV,
              const scene_rdl2::math::Color& borderColor);

    scene_rdl2::math::Color lookup(const int xi, const int yi, const int mipLevel) const;
    scene_rdl2::math::Color filterNearest(const float u, const float v) const;
    scene_rdl2::math::Color filterBilinear(const float u, const float v) const;
    scene_rdl2::math::Color filterNearestMipNearest(const float u, const float v, const float mipLevel) const;
    scene_rdl2::math::Color filterBilinearMipNearest(const float u, const float v, const float mipLevel) const;
    scene_rdl2::math::Color applyTexFilter(const float u, const float v, const float mipLevel,
                               const TextureFilterType texFilter) const;
    scene_rdl2::math::Color textureLookupDirect(const float u, const float v, const float mipLevel,
                                    const TextureFilterType texFilter) const;
    scene_rdl2::math::Color textureLookupTransformed(const float uUntransformed, const float vUntransformed, const float mipLevel, 
                                         const TextureFilterType texFilterT) const;
    float pdfNearest(const float u, const float v) const;
    float pdfBilinear(const float u, const float v) const;
    float pdfNearestMipNearest(const float u, const float v, const float mipLevel) const;
    float pdfBilinearMipNearest(const float u, const float v, const float mipLevel) const;

    void sampleNearest(const float ru, const float rv, scene_rdl2::math::Vec2f *uv, float *pdf) const;
    void sampleBilinear(const float ru, const float rv, scene_rdl2::math::Vec2f *uv, float *pdf) const;
    void sampleNearestMipNearest(const float ru, const float rv, float mipLevel, scene_rdl2::math::Vec2f *uv, float *pdf) const;
    void sampleBilinearMipNearest(const float ru, const float rv, float mipLevel, scene_rdl2::math::Vec2f *uv, float *pdf) const;

    // Members
    IMAGE_DISTRIBUTION_MEMBERS;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


