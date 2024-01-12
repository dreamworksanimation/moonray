// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_DEEPFLAGS_H
#define INCLUDED_DCX_DEEPFLAGS_H

//=============================================================================
//
//  class   DeepFlags
//
//=============================================================================

#include "DcxAPI.h"

#include <iostream>

//-------------------------
//!rst:cpp:begin::
//DeepFlags
//=========
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER

#if 0
typedef uint32_t DeepFlag; // old flag type
#endif

//====================
//
//  class DeepFlags
//
//====================
//--------------------------------------------------------------------------------
//
//  Per-deep sample metadata flag-bits class.
//
//  Stores surface-type flags and the partial subpixel-coverage bin count which
//  indicates the subpixel-coverage weight applied to the deep channel values.
//
//  .. note::
//      Important backwards compatibility note!
//      Flags now require 32-bit storage. In 2.2.1 the 4 defined flags would fit
//      easily inside the mantissa of a 16-bit half float, but with the addition
//      of the partial subpixel-coverage weight it's prudent to switch to 32-bit
//      floats with 23 bits worth of mantissa space. This way additional bits can
//      be allocated later on without changing the channel format (again.)
//
//--------------------------------------------------------------------------------

class DCX_EXPORT DeepFlags
{
  public:

    static GNU_CONST_DECL uint32_t   ALL_BITS_OFF       = 0x0ul;
    static GNU_CONST_DECL uint32_t   ALL_BITS_ON        = 0xfffffffful;


    //-------------------------------------------------------------------
    //!rst:left-align::
    //  **Surface-Type Flags**
    //      ``bits 0x01..0xff``
    //-------------------------------------------------------------------
    static GNU_CONST_DECL uint32_t   LINEAR_INTERP      = 0x00001;  // Linear surface sample interpolation (not volumetric)
    static GNU_CONST_DECL uint32_t   MATTE_OBJECT       = 0x00002;  // Matte sample that cuts-out (blackens) other samples
    static GNU_CONST_DECL uint32_t   ADDITIVE           = 0x00004;  // Additive sample which plusses with adjacent additive samples
    static GNU_CONST_DECL uint32_t   FLAG4              = 0x00008;  // Placeholder
    static GNU_CONST_DECL uint32_t   FLAG5              = 0x00010;  // Placeholder
    static GNU_CONST_DECL uint32_t   FLAG6              = 0x00020;  // Placeholder
    static GNU_CONST_DECL uint32_t   FLAG7              = 0x00040;  // Placeholder
    static GNU_CONST_DECL uint32_t   FLAG8              = 0x00080;  // Placeholder
    //
    static GNU_CONST_DECL uint32_t   SURFACE_FLAG_BITS  = 0x000ff;


    //-----------------------------------------------------------------------------------------
    //!rst:left-align::
    //
    //  **Partial Subpixel-Coverage Weight**
    //      ``bits 0x0100..0xff00``
    //
    //  If any of the 8 partial spcoverage bits are enabled the deep sample values include
    //  partial-coverage weighting equal to the binary value of the coverage bits. These bits
    //  work in conjunction with the ADDITIVE flag.
    //
    //  Partial-coverage count to partial-coverage weight conversion is biased by 1 so that
    //  0xff00=0.996, 0x8000=0.5, 0x4000=0.25, 0x0000=0.0, etc.
    //  i.e::
    //
    //       weight = float((bits & PARTIAL_SPCOVERAGE_BITS) >> 8) / maxSpCoverageScale   or
    //       weight = float(partialSpCoverageBits()) / 65536.0f                           or just use
    //       weight = DeepFlags::getSpCoverageWeight()
    //
    //  Maximum coverage count is 256 (0x10000) indicating complete subpixel coverage and is
    //  logically the same as all bits off. Since both 0x00000 and 0x10000 effectively mean the
    //  same thing we don't require the 9th bit to store the actual value 256. The methods to
    //  retrieve the bin count check for count==0 and ADDITIVE=1 and return maxSpCoverageCount instead.
    //
    //  So, when accumulating partial-coverage weight counts and maxSpCoverageCount is reached or
    //  exceeded, *clear* the coverage value to 0x00000 and clear the ADDITIVE flags to indicate
    //  full-coverage.
    //-----------------------------------------------------------------------------------------
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE0     = 0x00100;  // Partial sp-coverage weight bit 0
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE1     = 0x00200;  // Partial sp-coverage weight bit 1
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE2     = 0x00400;  // Partial sp-coverage weight bit 2
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE3     = 0x00800;  // Partial sp-coverage weight bit 3
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE4     = 0x01000;  // Partial sp-coverage weight bit 4
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE5     = 0x02000;  // Partial sp-coverage weight bit 5
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE6     = 0x04000;  // Partial sp-coverage weight bit 6
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE7     = 0x08000;  // Partial sp-coverage weight bit 7
    //
    static GNU_CONST_DECL uint32_t  PARTIAL_SPCOVERAGE_BITS = 0x0ff00;
    //
    static GNU_CONST_DECL uint32_t  minSpCoverageCount      = 0;
    static GNU_CONST_DECL uint32_t  maxSpCoverageCount      = 256;
    static GNU_CONST_DECL float     maxSpCoverageScale      = 256.0f;


    //-------------------------------------------------------------------
    //!rst:left-align::
    //  **Reserved Bits (undefined)**
    //      ``bits 0x10000..0xf0000``
    //-------------------------------------------------------------------
    static GNU_CONST_DECL uint32_t   RESERVED0              = 0x10000;
    static GNU_CONST_DECL uint32_t   RESERVED1              = 0x20000;
    static GNU_CONST_DECL uint32_t   RESERVED2              = 0x40000;
    static GNU_CONST_DECL uint32_t   RESERVED3              = 0x80000;
    //
    static GNU_CONST_DECL uint32_t   RESERVED_BITS          = 0xf0000;


  public:

    DeepFlags (); // sets all flags to 0
    DeepFlags (uint32_t);

    operator uint32_t () const;

    bool operator [] (int bit) const;  // read a bit by index

    void    clearAll (); // clear all bits (set value to 0x0)

    //---------------------------------------------------
    // Set or clear the surface flag bits.
    // Partial subpixel-coverage bits are left untouched.
    //---------------------------------------------------
    void    setSurfaceFlags (const DeepFlags&);
    void    setSurfaceFlags (uint32_t);
    void    clearSurfaceFlags (const DeepFlags&);
    void    clearSurfaceFlags (uint32_t);

    //---------------------------------
    // Get a subset range of flag bits.
    //---------------------------------

    uint32_t    surfaceFlags () const;              // bits & SURFACE_FLAG_BITS
    uint32_t    partialSpCoverageBits () const;     // bits & PARTIAL_SPCOVERAGE_BITS

    //-----------------------------------------------------------------
    // Does the segment represent a solid (hard) or volumetric surface?
    // If hard-surface - use linear interpolation between Zf/Zb.
    // If volumetric - use log interpolation between Zf/Zb.
    //-----------------------------------------------------------------

    bool    isHardSurface () const;
    bool    isVolumetric () const;
    void    setHardSurface ();
    void    setVolumetric ();

    //----------------------------------------------------------------------
    // Should the segment cutout (act as a matte of) the segments behind it?
    //----------------------------------------------------------------------

    bool    isMatte () const;
    void    setMatte ();
    void    clearMatte ();

    //-------------------------------------------------------------------
    // Sample should be added to surrounding samples rather than under-ed
    // This is used primarily for partial subpixel coverage.
    //-------------------------------------------------------------------

    bool    isAdditive () const;
    void    setAdditive ();
    void    clearAdditive ();


    //-----------------------------------------------------------------------------
    // Does the sample have partial subpixel coverage baked into color, alpha, etc?
    // If so the sample is handled as additive when composited.
    // This normally indicates filtering or resampling of subpixel masks has been
    // applied.
    //-----------------------------------------------------------------------------

    bool            hasPartialSpCoverage () const;
    bool            hasFullSpCoverage () const;

    uint32_t        getSpCoverageCount () const;
    float           getSpCoverageWeight () const;
    static float    getSpCoverageWeightForCount (uint32_t count); // float(count) / maxSpCoverageScale

    void            setSpCoverageCount (uint32_t count);
    void            setSpCoverageWeight (float weight);
    uint32_t        addToSpCoverageCount (uint32_t count); // returns the added result
    void            clearSpCoverageCount ();


    //------------------------------------------------------------
    // Bitwise flag ops - operates only on the surface flags bits.
    // Partial subpixel-coverage bits are left untouched.
    //------------------------------------------------------------

    DeepFlags  operator &  (uint32_t rhs) const;
    DeepFlags  operator &  (const DeepFlags& rhs) const;
    DeepFlags  operator |  (uint32_t rhs) const;
    DeepFlags  operator |  (const DeepFlags& rhs) const;

    DeepFlags& operator &= (uint32_t rhs);
    DeepFlags& operator &= (const DeepFlags& rhs);
    DeepFlags& operator |= (uint32_t rhs);
    DeepFlags& operator |= (const DeepFlags& rhs);

    bool     operator == (int rhs) const;
    bool     operator == (uint32_t rhs) const;
    bool     operator == (const DeepFlags& rhs) const;
    bool     operator != (int rhs) const;
    bool     operator != (uint32_t rhs) const;
    bool     operator != (const DeepFlags& rhs) const;


    //-------------------------------------------------------------------
    // Convert to/from floats.
    // Does a simple int/float conversion and assumes storage in 32-bit
    // floats which has 23 bits worth of mantissa capacity.
    // Note: if stored in a 16-bit half the partial spcoverage bits will
    // get truncated!
    //-------------------------------------------------------------------

    void    fromFloat (float);
    float   toFloat () const;


    //--------------------------------
    // Print the list of enabled flags
    //--------------------------------

    void    print (std::ostream&) const;
    friend  std::ostream& operator << (std::ostream&,
                                       const DeepFlags&);


  public:

    uint32_t   bits; // the flag bits

};


//--------------
//!rst:cpp:end::
//--------------

//-----------------
// Inline Functions
//-----------------

inline DeepFlags::DeepFlags () : bits(0) {}
inline DeepFlags::DeepFlags (uint32_t v) : bits(v) {}
inline DeepFlags::operator uint32_t () const { return bits; }
inline bool DeepFlags::operator [] (int bit) const { return (bits & (1ul << bit))!=0; }
inline void DeepFlags::clearAll () { bits = 0ul; }
//
inline uint32_t DeepFlags::surfaceFlags () const { return (bits & SURFACE_FLAG_BITS); }
inline uint32_t DeepFlags::partialSpCoverageBits () const { return (bits & PARTIAL_SPCOVERAGE_BITS); }
//
inline void DeepFlags::setSurfaceFlags (const DeepFlags& b) { bits |= b.surfaceFlags(); }
inline void DeepFlags::setSurfaceFlags (uint32_t v)  { bits |= (v & SURFACE_FLAG_BITS); }
inline void DeepFlags::clearSurfaceFlags (const DeepFlags& b) { bits &= ~b.surfaceFlags(); }
inline void DeepFlags::clearSurfaceFlags (uint32_t v)  { bits &= ~(v & SURFACE_FLAG_BITS); }
//
inline bool DeepFlags::isHardSurface () const { return (bits & LINEAR_INTERP)!=0; }
inline bool DeepFlags::isVolumetric () const { return !isHardSurface(); }
inline void DeepFlags::setHardSurface () { setSurfaceFlags(LINEAR_INTERP); }
inline void DeepFlags::setVolumetric () { clearSurfaceFlags(LINEAR_INTERP); }
inline bool DeepFlags::isMatte () const { return (bits & MATTE_OBJECT)!=0; }
inline void DeepFlags::setMatte () { setSurfaceFlags(MATTE_OBJECT); }
inline void DeepFlags::clearMatte () { clearSurfaceFlags(MATTE_OBJECT); }
inline bool DeepFlags::isAdditive () const { return (bits & ADDITIVE)!=0; }
inline void DeepFlags::setAdditive () { setSurfaceFlags(ADDITIVE); }
inline void DeepFlags::clearAdditive () { clearSurfaceFlags(ADDITIVE); }
//
inline bool DeepFlags::hasPartialSpCoverage () const { return (bits & PARTIAL_SPCOVERAGE_BITS)!=0; }
inline bool DeepFlags::hasFullSpCoverage () const { return (bits & PARTIAL_SPCOVERAGE_BITS)==0; }
inline uint32_t DeepFlags::getSpCoverageCount () const {
    const uint32_t count = (bits & PARTIAL_SPCOVERAGE_BITS);
    return (count == 0 && isAdditive())?maxSpCoverageCount:(count >> 8); // translate 0 to max count
}
/*static*/ inline float DeepFlags::getSpCoverageWeightForCount (uint32_t count) {
    return float(count) / maxSpCoverageScale;
}
inline float DeepFlags::getSpCoverageWeight () const {
    return getSpCoverageWeightForCount(getSpCoverageCount());
}
//
inline void DeepFlags::setSpCoverageCount (uint32_t count) {
    bits = surfaceFlags();
    if (count > 0 && count < maxSpCoverageCount)
        bits |= ((count << 8) | ADDITIVE);
}
inline void DeepFlags::setSpCoverageWeight (float weight) {
    setSpCoverageCount(int(weight * maxSpCoverageScale));
}
inline void DeepFlags::clearSpCoverageCount () {
    bits = surfaceFlags();
    clearSurfaceFlags(ADDITIVE);
}
inline uint32_t DeepFlags::addToSpCoverageCount (uint32_t count) {
    const uint32_t new_count = getSpCoverageCount() + count;
    if (new_count >= maxSpCoverageCount)
        clearSpCoverageCount(); // saturated  - clear the weight, disable additive
    else
        setSpCoverageCount(new_count);
    return new_count;
}
//
inline DeepFlags  DeepFlags::operator &  (uint32_t rhs) const { return DeepFlags(bits & (rhs | ~SURFACE_FLAG_BITS)); }
inline DeepFlags  DeepFlags::operator &  (const DeepFlags& rhs) const { return DeepFlags(bits & (rhs.bits | ~SURFACE_FLAG_BITS)); }
inline DeepFlags  DeepFlags::operator |  (uint32_t rhs) const { return DeepFlags(bits | (rhs & SURFACE_FLAG_BITS)); }
inline DeepFlags  DeepFlags::operator |  (const DeepFlags& rhs) const { return DeepFlags(bits | rhs.surfaceFlags()); }
//
inline DeepFlags& DeepFlags::operator &= (uint32_t rhs) { bits &= (rhs | ~SURFACE_FLAG_BITS); return *this; }
inline DeepFlags& DeepFlags::operator &= (const DeepFlags& rhs) { bits &= (rhs.bits | ~SURFACE_FLAG_BITS); return *this; }
inline DeepFlags& DeepFlags::operator |= (uint32_t rhs) { bits |= (rhs & SURFACE_FLAG_BITS); return *this; }
inline DeepFlags& DeepFlags::operator |= (const DeepFlags& rhs) { bits |= rhs.surfaceFlags(); return *this; }
//
inline bool DeepFlags::operator == (int rhs) const { return (surfaceFlags() == ((uint32_t)rhs & SURFACE_FLAG_BITS)); }
inline bool DeepFlags::operator == (uint32_t rhs) const { return (surfaceFlags() == (rhs & SURFACE_FLAG_BITS)); }
inline bool DeepFlags::operator == (const DeepFlags& rhs) const { return (surfaceFlags() == rhs.surfaceFlags()); }
inline bool DeepFlags::operator != (int rhs) const { return (surfaceFlags() != ((uint32_t)rhs & SURFACE_FLAG_BITS)); }
inline bool DeepFlags::operator != (uint32_t rhs) const { return (surfaceFlags() != (rhs & SURFACE_FLAG_BITS)); }
inline bool DeepFlags::operator != (const DeepFlags& rhs) const { return (surfaceFlags() != rhs.surfaceFlags()); }
//
inline void DeepFlags::fromFloat (float v) { bits = (uint32_t)floor(v); }
inline float DeepFlags::toFloat () const { return (float)bits; }


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_DEEPFLAGS_H
