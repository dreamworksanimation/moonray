// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_SPMASK_H
#define INCLUDED_DCX_SPMASK_H

//=============================================================================
//
//  class   SpMask8
//
//=============================================================================

#include "DcxAPI.h"

#include <cstddef>
#include <stdint.h> // for uint64_t
#include <math.h> // for floorf
#include <iostream>


//-------------------------
//!rst:cpp:begin::
//.. _spmask8_class:
//
//SpMask8
//=======
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER

#if 0
typedef uint64_t SpMask8; // old bitmask type
#endif


//=====================
//
//  class   SpMask8
//
//=====================
//-----------------------------------------------------------------------------
//
//  An 8x8 A-buffer bitmask class which wraps a 64-bit unsigned int.
//  It offers convenience methods for x/y bit access and translating
//  to/from floats for I/O.
//
//  The 8x8 fixed size (64 grey values) is a compromise between per-sample memory
//  use and providing enough resolution to resolve spatial alignments. A larger
//  size such as 16x16 would consume 8 floats per sample vs. just 2 for 8x8, while
//  still not providing enough resolution to perform precision transform filtering.
//  To improve subpixel resolution the 8x8 mask can be used in conjunction with
//  8 additional bits in the DeepFlags class representing partial subpixel-coverage
//  weight which allows each sample to represent 16384 grey values rather than just
//  64.
//
//  The pattern is spatially interpreted with the lowest bit 0 in the
//  lower-left corner and the highest bit 63 in the upper-right corner
//  and always oriented Y-up.  This matches most current applications and
//  renderers (c 2016) - even though OpenEXR's pixel coordinate axis is
//  Y-down:
//  ::
//
//           -------------------
//        7  |56 . . . . . . 63|  +Y
//        6  | . . . . . . . . |
//        5  | . . . . . . . . |
//        4  | . . . . . . . . |
//        3  | . . . . . . . . |
//        2  | . . . . . . . . |
//        1  | . . . . . . . . |
//        0  | 0 . . . . . . 7 |  -Y
//           -------------------
//      -X     0 1 2 3 4 5 6 7    +X
//
//-----------------------------------------------------------------------------

class DCX_EXPORT SpMask8
{
  public:

    static GNU_CONST_DECL int       width           =  8;
    static GNU_CONST_DECL int       height          =  8;
    static GNU_CONST_DECL int       numBits         = 64;
    static GNU_CONST_DECL uint64_t  allBitsOff      = 0x0ull;
    static GNU_CONST_DECL uint64_t  allBitsOn       = 0xffffffffffffffffull;
    static const SpMask8            zeroCoverage;   // all bits off
    static const SpMask8            fullCoverage;   // all bits on


  public:

    SpMask8 (); // sets mask to zero
    SpMask8 (uint64_t);
    SpMask8 (const SpMask8&);

    SpMask8& operator = (uint64_t rhs);
    SpMask8& operator = (const SpMask8&);

    operator uint64_t() const;

    uint64_t   value() const;

    bool operator [] (int bit) const;  // read a bit by index

    static uint32_t getSubpixelIndex (int sx,
                                      int sy);
    static SpMask8  getMask (int sx,
                             int sy);

    void    clear (); // clear all bits

    //
    // Bitwise ops
    //

    SpMask8  operator &  (const SpMask8& rhs) const;
    SpMask8  operator |  (const SpMask8& rhs) const;
    SpMask8  operator ^  (const SpMask8& rhs) const;
    SpMask8  operator ~  () const;

    SpMask8& operator &= (const SpMask8& rhs);
    SpMask8& operator |= (const SpMask8& rhs);
    SpMask8& operator ^= (const SpMask8& rhs);

    SpMask8  operator << (int rhs) const;
    SpMask8  operator >> (int rhs) const;

    SpMask8& operator <<= (int rhs);
    SpMask8& operator >>= (int rhs);

    SpMask8& operator ++ (int); // postfix++, same as <<=
    SpMask8& operator ++ ();    //  ++prefix, same as <<=
    SpMask8& operator -- (int); // postfix--, same as >>=
    SpMask8& operator -- ();    //  --prefix, same as >>=

    bool     operator == (int rhs) const;
    bool     operator == (uint64_t rhs) const;
    bool     operator == (const SpMask8& rhs) const;
    bool     operator != (int rhs) const;
    bool     operator != (uint64_t rhs) const;
    bool     operator != (const SpMask8& rhs) const;


    //
    // Convert to/from floats
    // TODO: handle endianness!
    //

    void    fromFloat (float sp1,
                       float sp2);
    void    toFloat (float& sp1,
                     float& sp2) const;


    //
    // Read/write subpixel bits
    //

    bool    isSubpixelOn (int sx,
                          int sy) const;
    void    setSubpixel (int sx,
                         int sy);
    void    setSubpixels (int sx,
                          int sy,
                          int sr,
                          int st);
    void    unsetSubpixel (int sx,
                           int sy);
    void    unsetSubpixels (int sx,
                            int sy,
                            int sr,
                            int st);


    //
    // Map X/Y coord from a different mask size into SpMask8 range.
    // Output coords can specify an overlap range (inclusive.)
    // (The X&Y methods are functionally the same since the mask size is
    //  now fixed at 8x8)
    //
    // ex. set the bits covered by subpixel 3,11 of a 16x16 source grid:
    //  int outSpX, outSpY, outSpR, outSpT;
    //  SpMask8::mapXCoord( 3, 16, outSpX, outSpR);
    //  SpMask8::mapYCoord(11, 16, outSpY, outSpT);
    //  SpMask8 spmask;
    //  spmask.setSubpixels(outSpX, outSpY, outSpR, outSpT);
    //

    static void mapXCoord (int inX,
                           int inW,
                           int& outX,
                           int& outR);
    static void mapYCoord (int inY,
                           int inH,
                           int& outY,
                           int& outT);


    //
    // Bit count / coverage
    //

    int     bitsOn () const;
    int     bitsOff() const;
    float   toCoverage() const;


    //
    // Print the bit pattern as a text grid
    //

    void    printPattern (std::ostream&,
                          const char* prefix) const;

    //
    // Print the mask as a hex value
    //

    friend  std::ostream& operator << (std::ostream&,
                                       const SpMask8&);


  private:
    union floatUnion
    {
        uint64_t as_mask;
        float    as_float[2];
    };

    uint64_t    m;  // the bitmask

};



//----------
//!rst:cpp:end::
//----------

//-----------------
// Inline Functions
//-----------------

inline SpMask8::SpMask8 () : m(0ull) {}
inline SpMask8::SpMask8 (uint64_t spmask) : m(spmask) {}
inline SpMask8::SpMask8 (const SpMask8& b) : m(b.m) {}
//
inline SpMask8& SpMask8::operator = (uint64_t rhs) { m = rhs; return *this; }
inline SpMask8& SpMask8::operator = (const SpMask8& rhs) { if (this != &rhs) m = rhs.m; return *this; }
//
inline SpMask8::operator uint64_t() const { return m; }
//
inline SpMask8  SpMask8::operator &  (const SpMask8& rhs) const { return SpMask8(m & rhs.m); }
inline SpMask8  SpMask8::operator |  (const SpMask8& rhs) const { return SpMask8(m | rhs.m); }
inline SpMask8  SpMask8::operator ^  (const SpMask8& rhs) const { return SpMask8(m ^ rhs.m); }
inline SpMask8  SpMask8::operator ~  () const { return SpMask8(~m); }
//
inline SpMask8& SpMask8::operator &= (const SpMask8& rhs) { m &= rhs.m; return *this; }
inline SpMask8& SpMask8::operator |= (const SpMask8& rhs) { m |= rhs.m; return *this; }
inline SpMask8& SpMask8::operator ^= (const SpMask8& rhs) { m ^= rhs.m; return *this; }
//
inline SpMask8  SpMask8::operator << (int rhs) const { return SpMask8(m << rhs); }
inline SpMask8  SpMask8::operator >> (int rhs) const { return SpMask8(m >> rhs); }
//
inline SpMask8& SpMask8::operator <<= (int rhs) { m <<= rhs; return *this; }
inline SpMask8& SpMask8::operator >>= (int rhs) { m >>= rhs; return *this; }
//
inline SpMask8& SpMask8::operator ++ (int) { m <<= 1; return *this; };
inline SpMask8& SpMask8::operator ++ ()    { m <<= 1; return *this; };
inline SpMask8& SpMask8::operator -- (int) { m >>= 1; return *this; };
inline SpMask8& SpMask8::operator -- ()    { m >>= 1; return *this; };
//
inline bool     SpMask8::operator == (int rhs) const { return (m == (uint64_t)rhs); }
inline bool     SpMask8::operator == (uint64_t rhs) const { return (m == rhs); }
inline bool     SpMask8::operator == (const SpMask8& rhs) const { return (m == rhs.m); }
inline bool     SpMask8::operator != (int rhs) const { return (m != (uint64_t)rhs); }
inline bool     SpMask8::operator != (uint64_t rhs) const { return (m != rhs); }
inline bool     SpMask8::operator != (const SpMask8& rhs) const { return (m != rhs.m); }
//
inline void SpMask8::fromFloat (float sp1, float sp2)
{
    floatUnion mask_union;
    mask_union.as_float[0] = sp1;
    mask_union.as_float[1] = sp2;
    m = mask_union.as_mask;
}
inline void SpMask8::toFloat (float& sp1, float& sp2) const
{
    floatUnion mask_union;
    mask_union.as_mask = m;
    sp1 = mask_union.as_float[0];
    sp2 = mask_union.as_float[1];
}
inline uint64_t SpMask8::value() const { return m; }
inline bool SpMask8::operator [] (int bit) const { return (m & (1ull << bit))!=0; }
inline /*static*/ uint32_t SpMask8::getSubpixelIndex (int sx, int sy) { return ((sy << 3) + sx); }
inline /*static*/ SpMask8 SpMask8::getMask (int sx, int sy) { return SpMask8(1ull << getSubpixelIndex(sx, sy)); }
inline void SpMask8::clear () { m = 0ull; }
inline bool SpMask8::isSubpixelOn (int sx, int sy) const { return (m & getSubpixelIndex(sx, sy))!=0; }
inline void SpMask8::setSubpixel (int sx, int sy) { m |= (1ull << ((sy << 3) + sx)); }
inline void SpMask8::setSubpixels (int sx, int sy, int sr, int st)
{
    for (int y=sy; y <= st; ++y)
        for (int x=sx; x <= sr; ++x)
            setSubpixel(x, y);
}
inline void SpMask8::unsetSubpixel (int sx, int sy) { m &= ~(1ull << ((sy << 3) + sx)); }
inline void SpMask8::unsetSubpixels (int sx, int sy, int sr, int st)
{
    for (int y=sy; y <= st; ++y)
        for (int x=sx; x <= sr; ++x)
            unsetSubpixel(x, y);
}
inline int SpMask8::bitsOn () const
{
    if (m==allBitsOff || m==allBitsOn)
        return numBits;
#if 1
#define BX_(x) ((x) - (((x)>>1)&0x77777777)  \
                    - (((x)>>2)&0x33333333)  \
                    - (((x)>>3)&0x11111111))
#define BITCOUNT(x) (((BX_(x)+(BX_(x)>>4)) & 0x0F0F0F0F) % 255)
    const uint32_t lo = uint32_t(m);
    const uint32_t hi = uint32_t(m >> 32);
    return BITCOUNT(lo) + BITCOUNT(hi);
#else
    int numOnBits = 0;
    SpMask8 sp(1ull);
    for (uint32_t sp_bin=0; sp_bin < numBits; ++sp_bin, ++sp)
        if (m & sp)
            ++numOnBits;
    return numOnBits;
#endif
}
inline int SpMask8::bitsOff() const { return (numBits - bitsOn()); }
inline /*static*/ void SpMask8::mapXCoord (int inX, int inW, int& outX, int& outR)
{
    if (inW == OPENDCX_INTERNAL_NAMESPACE::SpMask8::width)
        outX = outR = inX;
    else
    {
        const float fX = float(inX);
        const float scale = float(OPENDCX_INTERNAL_NAMESPACE::SpMask8::width) / float(inW);
        outX = (int)floorf(fX * scale);
        outR = (int)floorf((fX + 0.999f) * scale);
    }
}
inline /*static*/ void SpMask8::mapYCoord (int inY, int inH, int& outY, int& outT)
{
    if (inH == OPENDCX_INTERNAL_NAMESPACE::SpMask8::height)
        outY = outT = inY;
    else
    {
        const float fY = float(inY);
        const float scale = float(OPENDCX_INTERNAL_NAMESPACE::SpMask8::height) / float(inH);
        outY = (int)floorf(fY * scale);
        outT = (int)floorf((fY + 0.999f) * scale);
    }
}
inline float SpMask8::toCoverage() const
{
    if (m==allBitsOff || m==allBitsOn)
        return 1.0f; // Zero mask indicates source doesn't have subpixel masks so weight is also 1
    const int numOnBits = bitsOn();
    return (numOnBits == 0)?0.0f:float(numOnBits)/float(numBits);
}
inline void SpMask8::printPattern (std::ostream& os,
                                   const char* prefix) const
{
    for (int sy=(int)(SpMask8::width-1); sy >= 0; --sy)
    {
        SpMask8 sp(1ull << (sy << 3));
        os << prefix;
        for (int sx=0; sx < (int)SpMask8::width; ++sx, ++sp)
            if (m & sp) os << "1 "; else os << ". ";
        os << std::endl;
    }
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_SPMASK_H
