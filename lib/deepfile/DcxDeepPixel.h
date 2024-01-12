// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_DEEPPIXEL_H
#define INCLUDED_DCX_DEEPPIXEL_H

//=============================================================================
//
//  struct  DeepMetadata
//
//  class   DeepSegment
//
//  class   DeepPixel
//  struct  DeepPixel::CombineContext
//
//=============================================================================

#include "DcxChannelSet.h"
#include "DcxChannelDefs.h"
#include "DcxPixel.h"
#include "DcxSpMask.h"
#include "DcxDeepFlags.h"

#include <iostream>
#include <vector>
#include <math.h> // for log1p, expm1
#include <set>


//-------------------------
//!rst:cpp:begin::
//DeepPixel
//=========
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


//-------------------------
//!rst:left-align::
//.. _deepmetadata_class:
//
//DeepMetadata
//************
//-------------------------

//========================
//
//  struct DeepMetadata
//
//========================
//-------------------------------------------------------------------------------
//
//  Stores extra information about a DeepSegment.
//  For the moment it only stores the subpixel mask and the flags, but this
//  could be made to hold arbitrary attributes.
//
//  These values are packed & unpacked via float & half channels for file IO.
//
//  TODO: support arbitrary attributes?
//
//-------------------------------------------------------------------------------

struct DCX_EXPORT DeepMetadata
{


    SpMask8     spmask;         // Subpixel 8x8 bitmask array (A-buffer)
    DeepFlags   flags;          // Flags - interpolation-type, matte-mode, etc.


    //-----------------------------------------
    // Default constructor leaves junk in vars.
    //-----------------------------------------

    inline DeepMetadata ();


    //------------
    // Constructor
    //------------

    inline DeepMetadata (const SpMask8& _mask,
                         const DeepFlags& _flags);


    //-------------------------------------------------------
    // Is Z-depth of segment zero (thin) or non-zero (thick)?
    //-------------------------------------------------------

    bool    isThin () const;
    bool    isThick () const;
    float   thickness () const;


    //-----------------------------------------------------------------
    // Does the segment represent a solid (hard) or volumetric surface?
    // If hard-surface - use linear interpolation between Zf/Zb.
    // If volumetric - use log interpolation between Zf/Zb.
    //-----------------------------------------------------------------

    bool    isHardSurface () const;
    bool    isVolumetric () const;


    //-------------------------------------------------------------------
    // Should the segment cutout (act as a matte) the segments behind it?
    //-------------------------------------------------------------------

    bool    isMatte () const;


    //-------------------------------------------------------------------
    // Sample should be added to surrounding samples rather than under-ed
    // This is used primarily for partial subpixel-coverage.
    //-------------------------------------------------------------------

    bool    isAdditive () const;


    //-----------------------------------------------------------------------------
    // Does the sample have partial subpixel coverage baked into color, alpha, etc?
    // If so the sample is handled as additive when composited.
    // This normally indicates filtering or resampling of subpixel masks has been
    // applied.
    //-----------------------------------------------------------------------------

    uint32_t    surfaceFlags () const;
    uint32_t    partialCoverageBits () const;

    bool        hasPartialSpCoverage () const;
    bool        hasFullSpCoverage () const;

    uint32_t    getSpCoverageCount () const;
    float       getSpCoverageWeight () const;

    void        setSpCoverageCount (uint32_t count);
    void        setSpCoverageWeight (float weight);
    void        clearSpCoverageCount ();


    //--------------------------------
    // Print the list of enabled flags
    //--------------------------------

    void    printFlags (std::ostream&) const;
    friend  std::ostream& operator << (std::ostream&,
                                       const DeepMetadata&);

};


//-------------------------
//!rst:left-align::
//.. _deepsegment_class:
//
//DeepSegment
//***********
//-------------------------

//=====================
//
//  class DeepSegment
//
//=====================
//-------------------------------------------------------------------------------------
//
//  A single deep sample describing a linear segment of Z-space where Zf <= Zb.
//
//  The color channel data describes the values at Zb, so finding values between
//  Zf and Zb requires linear or log interpolation depending on the
//  interpolation-type flag in the metadata.
//
//  Note that this class does not store the actual channel data so that the adding,
//  deleting, sorting, etc of DeepSegments is lightweight and fast.
//
//-------------------------------------------------------------------------------------

class DCX_EXPORT DeepSegment
{
  public:

    float           Zf, Zb;         // Z-front / Z-back depth positions
    int             index;          // Index into an array of channel data storage. -1 == non-assignment!
    DeepMetadata    metadata;       // Flags - interpolation-type, matte-mode, etc.


  public:
    //-----------------------------------------
    // Default constructor leaves junk in vars.
    //-----------------------------------------
    DeepSegment ();


    //---------------------------------------
    // Constructor
    // If Zf > Zb, Zf is clamped to Zb.
    //---------------------------------------

    DeepSegment (float _Zf, float _Zb,
                 int _index = -1,
                 const DeepMetadata& _metadata = DeepMetadata(SpMask8::fullCoverage,
                                                              DeepFlags::ALL_BITS_OFF));


    //--------------------------------------------
    // Set both Zf & Zb - checks for bad Z's and
    // tries to find reasonable solution...
    //--------------------------------------------

    void    setDepths (float _Zf, float _Zb);

    void    transformDepths (float translate,
                             float scale,
                             bool reverse=false); // translate then scale, or scale then translate if reverse=true

    //-------------------------
    // Used by the sort routine
    //-------------------------

    bool operator < (const DeepSegment&) const;


    //-------------------------
    // DeepFlags metadata access
    //-------------------------

    const DeepFlags&    flags () const;
    uint32_t            surfaceFlags () const;


    //-----------------------------------------------------------------
    // Does the segment represent a solid (hard) or volumetric surface?
    // If hard-surface use linear interpolation between Zf/Zb.
    // If volumetric use log interpolation between Zf/Zb.
    //-----------------------------------------------------------------

    bool    isHardSurface () const;
    bool    isVolumetric () const;


    //-------------------------------------------------------
    // Is Z-depth of segment zero (thin) or non-zero (thick)?
    //-------------------------------------------------------

    bool    isThin () const;
    bool    isThick () const;
    float   thickness () const;

    //-------------------------------------------------------
    // Should the segment cutout (act as a matte) the segments behind it?
    //-------------------------------------------------------

    bool    isMatte () const;


    //-------------------------------------------------------------------
    // Sample should be added to surrounding samples rather than under-ed
    //-------------------------------------------------------------------

    bool    isAdditive () const;


    //-----------------------------------------------------------------------------
    // Does the sample have partial subpixel coverage baked-in to color, alpha, etc
    // values? If so the sample is handled as additive when composited.
    // This normally indicates filtering or resampling of subpixel masks has been
    // applied.
    //-----------------------------------------------------------------------------

    bool        hasPartialSpCoverage () const;
    bool        hasFullSpCoverage () const;

    uint32_t    getSpCoverageCount () const;
    float       getSpCoverageWeight () const;

    void        setSpCoverageCount (uint32_t count);
    void        setSpCoverageWeight (float weight);
    void        clearSpCoverageCount ();


    //---------------------
    // Subpixel mask access
    //---------------------

    SpMask8&        spMask ();
    const SpMask8&  spMask () const;


    //---------------------------------------------------------------------------
    // Are all subpixel mask bits off or on?
    //
    // If all are OFF this usually indicates a 'legacy' deep sample containing no
    // subpixel mask data which is interpreted as volumetric.
    //
    // If all bits are ON this can simplify and speed up sample operations,
    // especially compositing (flattening) as iterating through subpixels is
    // uneccessary.
    //---------------------------------------------------------------------------

    bool    zeroCoverage () const;

    bool    fullCoverage () const;

    bool    hasSpMasks () const; // same as !zeroCoverage()


    //------------------------------------------------------------------
    // Return true if specific bits are enabled in the subpixel mask.
    // Note that all-bits-off is considered all-bits-on for these tests.
    //------------------------------------------------------------------

    bool    maskBitsEnabled (const SpMask8& check_bits) const;

    bool    maskBitEnabled (int bit_index) const;


    //--------------------------------------------------------------
    // Convert the subpixel mask to coverage weight.
    // This is the same as float(spMask8BitsOn(metadata.mask))/64.0f
    // Note that zero-bits-on is considered all-bits-on for this.
    //--------------------------------------------------------------

    float   getCoverage () const;


    //-----------------------------------------------------------
    // Print info about DeepSegment to output stream
    //-----------------------------------------------------------

    void    printInfo (std::ostream&,
                       bool show_mask=true) const;
    void    printFlags (std::ostream&) const;
    friend  std::ostream& operator << (std::ostream&,
                                       const DeepSegment&);

};


//-------------------------
//!rst:left-align::
//.. _deeppixel_class:
//
//DeepPixel
//*********
//-------------------------

//=====================
//
//  class DeepPixel
//
//=====================
//----------------------------------------------------------------------------------------
//
//  Contains a series of DeepSegments and their associated Pixels (containing float
//  channel data,) which combined together comprise a series of deep samples.
//
//  Supports metadata storage for each deep sample and offers methods to aid the
//  manipulation of samples within the deep pixel, including compositing (flattening,)
//  and sorting.
//  Because a :ref:`deepsegment_class` is lightweight the list can be rearranged and
//  sorted very quickly.  The list of large Pixel channel data structures is kept
//  static.
//
//  ::
//
//      TODO: investigate cost of using varying-sized Pixel channel array
//      TODO: extend flatten methods to accept near/far depth range to flatten within
//
//----------------------------------------------------------------------------------------

class DCX_EXPORT DeepPixel
{
  public:

    //------------------------------------------------------
    // Segment interpolation modes for flattening operations
    //------------------------------------------------------

    enum InterpolationMode
    {
        INTERP_OFF,         // Disable interpolation
        INTERP_AUTO,        // Determine interpolation from per-sample metadata (DeepFlags)
        INTERP_LOG,         // Use log interpolation for all samples
        INTERP_LIN,         // Use linear interpolation for all samples

        NUM_INTERPOLATIONMODES
    };

    //---------------------------------------------------
    // Return a string version of the interpolation mode.
    //---------------------------------------------------

    static const char* interpolationModeString (InterpolationMode mode);


    //---------------------------------------------------------------------
    //  struct DeepPixel::CombineContext
    //
    //    Threshold values for combining similar segments together, passed
    //    to the DeepPixel combiner methods. If color values are
    //    within the +/- threshold range the values are considered
    //    matched.
    //
    //    color_threshold is used for all color channels - default is 0.001
    //    id_threshold is used for ID channels - default is 0.495
    //
    //    TODO: change color_threshold to a Pixel so that each channel can
    //          be thresholded individually? Non-color channels may require
    //          larger values than color channels...
    //---------------------------------------------------------------------

    struct CombineContext
    {
        float       color_threshold;    // Combiner threshold for color channel values - default is 0.001
        float       id_threshold;       // Combiner threshold for id channel values - default is 0.495
        ChannelSet  id_channels;        // ID channels used to combine like segments - default is None

        inline CombineContext ();
        inline CombineContext (const CombineContext&);
        inline CombineContext (float             _color_threshold,
                               float             _id_threshold,
                               const ChannelSet& _id_channels);
    };


  public:

    //-----------------------------------------
    // Constructors
    //-----------------------------------------
    DeepPixel (const ChannelSet& channels);
    DeepPixel (const ChannelIdx z);
    DeepPixel (const DeepPixel& b);

    //---------------------------------------------------------
    // Read-only ChannelSet access
    //      This ChannelSet is shared between all DeepSegments.
    //---------------------------------------------------------

    const   ChannelSet& channels () const;


    //---------------------------------------------------------
    // Assign a ChannelSet
    //      This ChannelSet is shared between all DeepSegments.
    //---------------------------------------------------------

    void    setChannels (const ChannelSet& c);


    //---------------------------------------------------------
    // Get/set the xy location
    //      Currently used for debugging
    //---------------------------------------------------------

    int     x () const;
    int     y () const;
    void    getXY (int& x, int& y);

    void    setXY (int x, int y);


    //---------------------------------------------------------
    // Transform the Zf & Zb coords for all segments.
    //---------------------------------------------------------

    void    transformDepths (float translate,
                             float scale,
                             bool reverse=false); // translate then scale, or scale then translate if reverse=true


    //-----------------------------------------------------
    // Empty the segment list and clear most shared values.
    // The shared ChannelSet is unaffected.
    //-----------------------------------------------------

    void    clear ();


    //----------------------------
    // DeepSegment list management
    //----------------------------

    bool    empty () const;
    size_t  size () const;
    size_t  capacity () const;
    void    reserve (size_t n);


    //---------------------------------------------------------
    // DeepSegment Handling
    //---------------------------------------------------------

    DeepSegment& operator [] (size_t segment);
    const DeepSegment& operator [] (size_t segment) const;

    DeepSegment& getSegment (size_t segment);
    const DeepSegment& getSegment (size_t segment) const;

    // Sort the segments.  If the sorted flag is already true this returns quickly.
    void    sort (bool force=false);
    void    invalidateSort ();

    // Return the index of the DeepSegment nearest to Z and inside the distance of Z +- maxDistance, or -1 if nothing found.
    // (TODO: finish implementing this!)
    //int     nearestSegment (double Z, double maxDistance=0.0);

    // Check for overlaps between samples and return true if so.
    bool    hasOverlaps (const SpMask8& spmask=SpMask8::fullCoverage, bool force=false);

    // Returns true if at least one segment has spmasks (same as !allZeroCoverage())
    bool    hasSpMasks ();

    // Returns true if all segment spmasks are zero-coverage and not hard-surface/matte/additive tagged.
    bool    isLegacyDeepPixel ();

    // Returns true if all segment spmasks are zero-coverage - this may indicate a legacy deep pixel.
    bool    allZeroCoverage ();

    // Returns true if all segment spmasks are full-coverage, zero-coverage or a mix of full OR zero coverage.
    bool    allFullCoverage ();

    // Returns true is all segments are volumetric (log interp.)
    bool    allVolumetric ();
    bool    anyVolumetric ();

    // Returns true is all segments are hard-surface (lin interp.)
    bool    allHardSurface ();
    bool    anyHardSurface ();

    // Returns true is all segments are matte.
    bool    allMatte ();
    bool    anyMatte ();

    // Asssigns a subpixel mask to a range of segments
    void    setSegmentMask (const SpMask8& mask, size_t start=0, size_t end=10000);

    // Add an empty DeepSegment to the end of the list, returning its index.
    size_t  append ();
    // Add a DeepSegment to the end of the list.
    size_t  append (const DeepSegment& segment);
    // Add a DeepSegment to the end of the list.
    size_t  append (const DeepSegment& segment, const Pixelf& pixel);
    // Copy one segment from the second DeepPixel.
    size_t  append (const DeepPixel& b, size_t segment_index);
    // Combine all the segments of two DeepPixels.
    void    append (const DeepPixel& b);


    //
    // Remove a DeepSegment from the segment list, deleting its referenced Pixel.
    // Note that this method will possibly reorder some of the Pixel indices in the
    // DeepSegments, so a previously referenced Pixel index may become invalid and
    // need to be reaquired from its DeepSegment.
    //
    void    removeSegment (size_t segment_index);


    //----------------------------------------------
    // Accumulated spmask and flags for all segments
    //----------------------------------------------

    SpMask8     getAccumOrMask () const;
    SpMask8     getAccumAndMask () const;

    DeepFlags   getAccumOrFlags () const;
    DeepFlags   getAccumAndFlags () const;


    //---------------------------------------------------------
    // Arithmetic ops
    // Note that there are no *, /, -, + operators to avoid the
    // high cost of constructing & destroying DeepPixels.
    // Try to use these in-place modifiers.
    //---------------------------------------------------------

    DeepPixel& operator += (float val);
    DeepPixel& operator -= (float val);
    DeepPixel& operator *= (float val);
    DeepPixel& operator /= (float val);


    //------------------------------------
    // Read/Write DeepSegment Pixel access
    //------------------------------------

    const Pixelf&   getPixel (size_t pixel_index) const;
    Pixelf&         getPixel (size_t pixel_index);
    const Pixelf&   getSegmentPixel (size_t segment_index) const;
    Pixelf&         getSegmentPixel (size_t segment_index);
    const Pixelf&   getSegmentPixel (const DeepSegment& segment) const;
    Pixelf&         getSegmentPixel (const DeepSegment& segment);

    //------------------------------------------------------------------------
    // These Pixel read methods also copy metadata values from the DeepSegment
    // into the output Pixel's predefined channels: Chan_ZFront, Chan_ZBack,
    // Chan_SpBits1, Chan_SpBits2, Chan_DeepFlags and Chan_SpCoverage.
    //------------------------------------------------------------------------

    void    getSegmentPixelWithMetadata (size_t segment_index,
                                         Pixelf& out) const;
    void    getSegmentPixelWithMetadata (size_t segment_index,
                                         const ChannelSet& get_channels, // restrict copied channels to this
                                         Pixelf& out) const;
    void    getSegmentPixelWithMetadata (const DeepSegment& segment,
                                         Pixelf& out) const;
    void    getSegmentPixelWithMetadata (const DeepSegment& segment,
                                         const ChannelSet& get_channels, // restrict copied channels to this
                                         Pixelf& out) const;


    //---------------------------------------------
    // Read-only channel value access by ChannelIdx
    //---------------------------------------------

    float   getChannel (size_t segment,
                        ChannelIdx z) const;
    float   getChannel (const DeepSegment& segment,
                        ChannelIdx z) const;


    //---------------------------------------------------------
    // Print info about DeepPixel to output stream
    //---------------------------------------------------------

    void    printInfo (std::ostream&,
                       const char* prefix,
                       int padding=2,
                       bool show_mask=true);
    friend  std::ostream& operator << (std::ostream&,
                                       DeepPixel&);


  protected:

    ChannelSet                  m_channels;         // ChannelSet shared by all segments
    std::vector<DeepSegment>    m_segments;         // List of deep sample segments
    std::vector<Pixelf>         m_pixels;           // List of Pixels referenced by DeepSegment.index
    //
    int                         m_x, m_y;           // Pixel's xy location
    bool                        m_sorted;           // Have the segments been Z-sorted?
    bool                        m_overlaps;         // Are there any Z overlaps between segments?
    //
    SpMask8                     m_accum_or_mask;    // Subpixel bits that are on for ANY segment
    SpMask8                     m_accum_and_mask;   // Subpixel bits that are on for ALL segments
    DeepFlags                   m_accum_or_flags;   // Deep flags that are on for ANY segment
    DeepFlags                   m_accum_and_flags;  // Deep flags that are on for ALL segments
};



//--------------
//!rst:cpp:end::
//--------------

//-----------------
// Inline Functions
//-----------------

inline DeepMetadata::DeepMetadata () {}
inline DeepMetadata::DeepMetadata (const SpMask8& _spmask, const DeepFlags& _flags) :
    spmask(_spmask),
    flags(_flags)
{
    //
}
inline bool DeepMetadata::isHardSurface () const { return flags.isHardSurface(); }
inline bool DeepMetadata::isVolumetric () const { return flags.isVolumetric(); }
inline bool DeepMetadata::isMatte () const { return flags.isMatte(); }
inline bool DeepMetadata::isAdditive () const { return flags.isAdditive(); }
inline bool DeepMetadata::hasFullSpCoverage () const { return flags.hasFullSpCoverage(); }
inline bool DeepMetadata::hasPartialSpCoverage () const { return flags.hasPartialSpCoverage(); }
inline float DeepMetadata::getSpCoverageWeight () const { return flags.getSpCoverageWeight(); }
inline uint32_t DeepMetadata::getSpCoverageCount () const { return flags.getSpCoverageCount(); }
inline void DeepMetadata::setSpCoverageCount (uint32_t count) { flags.setSpCoverageCount(count); }
inline void DeepMetadata::setSpCoverageWeight (float weight) { flags.setSpCoverageWeight(weight); }
inline void DeepMetadata::clearSpCoverageCount () { flags.clearSpCoverageCount(); }
inline void DeepMetadata::printFlags (std::ostream& os) const { flags.print(os); }
//--------------------------------------------------------
inline void DeepSegment::setDepths (float _Zf, float _Zb)
{
    Zf = (_Zf <= _Zb)?_Zf:_Zb;
    Zb = _Zb;
    if (Zf < EPSILONf)
    {
        Zf = Zb;
        if (Zf < EPSILONf)
            Zf = Zb = EPSILONf;
    }
}
inline DeepSegment::DeepSegment () {}
inline DeepSegment::DeepSegment (float _Zf, float _Zb, int _index, const DeepMetadata& _metadata) :
    index(_index),
    metadata(_metadata)
{
    setDepths(_Zf, _Zb);
}
//
inline void DeepSegment::transformDepths (float translate, float scale, bool reverse) {
    if (!reverse)
    {
        Zf = (Zf + translate)*scale;
        Zb = (Zb + translate)*scale;
    }
    else
    {
        Zf = Zf*scale + translate;
        Zb = Zb*scale + translate;
    }
}
//
inline const DeepFlags& DeepSegment::flags () const { return metadata.flags; }
inline uint32_t DeepSegment::surfaceFlags () const { return metadata.flags.surfaceFlags(); }
inline bool DeepSegment::isHardSurface () const { return metadata.isHardSurface(); }
inline bool DeepSegment::isVolumetric () const { return metadata.isVolumetric(); }
inline bool DeepSegment::isThin () const { return (Zb <= Zf); }
inline bool DeepSegment::isThick () const { return !isThin(); }
inline float DeepSegment::thickness () const { return (Zb - Zf); }
inline bool DeepSegment::isMatte () const { return metadata.isMatte(); }
inline bool DeepSegment::isAdditive () const { return metadata.isAdditive(); }
inline bool DeepSegment::hasPartialSpCoverage () const { return metadata.hasPartialSpCoverage(); }
inline bool DeepSegment::hasFullSpCoverage () const { return metadata.hasFullSpCoverage(); }
inline float DeepSegment::getSpCoverageWeight () const { return metadata.getSpCoverageWeight(); }
inline uint32_t DeepSegment::getSpCoverageCount () const { return metadata.getSpCoverageCount(); }
inline void DeepSegment::setSpCoverageCount (uint32_t count) { metadata.setSpCoverageCount(count); }
inline void DeepSegment::setSpCoverageWeight (float weight) { metadata.setSpCoverageWeight(weight); }
inline void DeepSegment::clearSpCoverageCount () { metadata.clearSpCoverageCount(); }
//
inline SpMask8& DeepSegment::spMask () { return metadata.spmask; }
inline const SpMask8& DeepSegment::spMask () const { return metadata.spmask; }
inline bool DeepSegment::zeroCoverage () const { return (metadata.spmask==SpMask8::zeroCoverage); }
inline bool DeepSegment::fullCoverage () const {
    return (metadata.spmask==SpMask8::fullCoverage || metadata.spmask==SpMask8::zeroCoverage);
}
inline bool DeepSegment::hasSpMasks () const { return !zeroCoverage(); }
inline bool DeepSegment::maskBitsEnabled (const SpMask8& check_bits) const {
    return (zeroCoverage() || (metadata.spmask & check_bits));
}
inline bool DeepSegment::maskBitEnabled (int bit_index) const {
    return (zeroCoverage() || ((metadata.spmask & (SpMask8(1ull) << bit_index)) != 0));
}
inline float DeepSegment::getCoverage () const {
    if (fullCoverage())
        return 1.0f;
    // Count the on pixels:
    uint32_t on_bits = 0;
    SpMask8 spmask = 1ull;
    for (int sp_bin=0; sp_bin < SpMask8::numBits; ++sp_bin, ++spmask)
        if (metadata.spmask & spmask)
            ++on_bits;
    return (on_bits == 0)?0.0f:float(on_bits)/float(SpMask8::numBits);
}
inline bool DeepSegment::operator < (const DeepSegment& b) const {
    if      (Zf < b.Zf) return true;
    else if (Zf > b.Zf) return false;
    // If both Zfronts are equal check partial subpixel-coverage first, then Zbacks
    else if (getSpCoverageCount() < b.getSpCoverageCount()) return true;
    return (Zb < b.Zb);
}
inline void DeepSegment::printFlags (std::ostream& os) const { metadata.printFlags(os); }
//--------------------------------------------------------
inline DeepPixel::CombineContext::CombineContext () :
    color_threshold(0.001f),
    id_threshold(0.495f),
    id_channels(Mask_None)
{
    //
}
inline DeepPixel::CombineContext::CombineContext (const DeepPixel::CombineContext& b)
{
    if (this == &b)
        return;
    memcpy(this, &b, sizeof(*this));
    id_channels = b.id_channels;
}
inline DeepPixel::CombineContext::CombineContext (float _color_threshold,
                                                  float _id_threshold,
                                                  const ChannelSet& _id_channels) :
    color_threshold(_color_threshold),
    id_threshold(_id_threshold),
    id_channels(_id_channels)
{
    //
}
//--------------------------------------------------------
inline DeepPixel::DeepPixel (const ChannelSet& channels) :
    m_channels(channels),
    m_x(0), m_y(0),
    m_sorted(false),
    m_overlaps(false),
    m_accum_or_mask(SpMask8::zeroCoverage),
    m_accum_and_mask(SpMask8::zeroCoverage),
    m_accum_or_flags(DeepFlags::ALL_BITS_OFF),
    m_accum_and_flags(DeepFlags::ALL_BITS_OFF)
{}
inline DeepPixel::DeepPixel (const ChannelIdx channel) :
    m_channels(channel),
    m_x(0), m_y(0),
    m_sorted(false),
    m_overlaps(false),
    m_accum_or_mask(SpMask8::zeroCoverage),
    m_accum_and_mask(SpMask8::zeroCoverage),
    m_accum_or_flags(DeepFlags::ALL_BITS_OFF),
    m_accum_and_flags(DeepFlags::ALL_BITS_OFF)
{}
inline DeepPixel::DeepPixel (const DeepPixel& b) {
    m_channels        = b.m_channels;
    m_segments        = b.m_segments;
    m_pixels          = b.m_pixels;
    m_x               = b.m_x;
    m_y               = b.m_y;
    m_sorted          = b.m_sorted;
    m_overlaps        = b.m_overlaps;
    m_accum_or_mask   = b.m_accum_or_mask;
    m_accum_and_mask  = b.m_accum_and_mask;
    m_accum_or_flags  = b.m_accum_or_flags;
    m_accum_and_flags = b.m_accum_and_flags;
}
//
inline const ChannelSet& DeepPixel::channels () const { return m_channels; }
inline void DeepPixel::setChannels (const ChannelSet& channels) {
    const size_t nPixels = m_pixels.size();
    for (size_t i=0; i < nPixels; ++i)
        m_pixels[i].channels = channels;
    m_channels = channels;
}
//
inline int DeepPixel::x () const { return m_x; }
inline int DeepPixel::y () const { return m_y; }
inline void DeepPixel::getXY (int& x, int& y) { x = m_x; y = m_y; }
inline void DeepPixel::setXY (int x, int y) { m_x = x; m_y = y; }
//
inline void DeepPixel::transformDepths (float translate, float scale, bool reverse) {
    const size_t nSegments = m_segments.size();
    for (size_t i=0; i < nSegments; ++i)
        m_segments[i].transformDepths(translate, scale, reverse);
}
//
inline SpMask8 DeepPixel::getAccumOrMask () const { return m_accum_or_mask; }
inline SpMask8 DeepPixel::getAccumAndMask () const { return m_accum_and_mask; }
inline DeepFlags DeepPixel::getAccumOrFlags () const { return m_accum_or_flags; }
inline DeepFlags DeepPixel::getAccumAndFlags () const { return m_accum_and_flags; }
//
inline bool DeepPixel::empty () const { return (m_segments.size() == 0); }
inline size_t DeepPixel::size () const { return m_segments.size(); }
inline size_t DeepPixel::capacity () const { return m_segments.capacity(); }
inline void DeepPixel::reserve (size_t n) { m_segments.reserve(n); m_pixels.reserve(n); }
//
inline DeepSegment& DeepPixel::operator [] (size_t segment_index) { return m_segments[segment_index]; }
inline const DeepSegment& DeepPixel::operator [] (size_t segment_index) const { return m_segments[segment_index]; }
inline DeepSegment& DeepPixel::getSegment (size_t segment_index) { return m_segments[segment_index]; }
inline const DeepSegment& DeepPixel::getSegment (size_t segment_index) const { return m_segments[segment_index]; }
//
inline bool DeepPixel::hasSpMasks () { return !allZeroCoverage(); }
//
inline const Pixelf& DeepPixel::getPixel (size_t pixel_index) const { return m_pixels[pixel_index]; }
inline Pixelf& DeepPixel::getPixel (size_t pixel_index) { return m_pixels[pixel_index]; }
inline const Pixelf& DeepPixel::getSegmentPixel (size_t segment_index) const { return m_pixels[m_segments[segment_index].index]; }
inline Pixelf& DeepPixel::getSegmentPixel (size_t segment_index) { return m_pixels[m_segments[segment_index].index]; }
inline const Pixelf& DeepPixel::getSegmentPixel (const DeepSegment& segment) const { return m_pixels[segment.index]; }
inline Pixelf& DeepPixel::getSegmentPixel (const DeepSegment& segment) { return m_pixels[segment.index]; }
// Copy Pixel channels in ChannelSet 'get_channels' and copy/extract metadata
// values from the DeepSegment into the output Pixel's predefined channels.
inline void DeepPixel::getSegmentPixelWithMetadata (const DeepSegment& segment, const ChannelSet& get_channels, Pixelf& out) const
{
    out.copy(m_pixels[segment.index], get_channels);
    out.channels = get_channels;
    out[OPENDCX_INTERNAL_NAMESPACE::Chan_ZFront] = segment.Zf;
    out[OPENDCX_INTERNAL_NAMESPACE::Chan_ZBack ] = segment.Zb;
    segment.metadata.spmask.toFloat(out[OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits1],
                                    out[OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits2]);
    out[OPENDCX_INTERNAL_NAMESPACE::Chan_DeepFlags] = segment.metadata.flags.toFloat();
    if (segment.hasPartialSpCoverage())
        out[OPENDCX_INTERNAL_NAMESPACE::Chan_SpCoverage] = segment.getSpCoverageWeight();
    else
        out[OPENDCX_INTERNAL_NAMESPACE::Chan_SpCoverage] = 1.0f;
    if (segment.isMatte())
        out[OPENDCX_INTERNAL_NAMESPACE::Chan_ACutout] = 0.0f;
    else
        out[OPENDCX_INTERNAL_NAMESPACE::Chan_ACutout] = out[OPENDCX_INTERNAL_NAMESPACE::Chan_A];
}
inline void DeepPixel::getSegmentPixelWithMetadata (size_t segment_index, Pixelf& out) const {
    const DeepSegment& segment = m_segments[segment_index];
    getSegmentPixelWithMetadata(segment, m_pixels[segment.index].channels, out);
}
inline void DeepPixel::getSegmentPixelWithMetadata (size_t segment_index, const ChannelSet& get_channels, Pixelf& out) const {
    getSegmentPixelWithMetadata(m_segments[segment_index].index, get_channels, out);
}
inline void DeepPixel::getSegmentPixelWithMetadata (const DeepSegment& segment, Pixelf& out) const {
    getSegmentPixelWithMetadata(segment, m_pixels[segment.index].channels, out);
}
inline float DeepPixel::getChannel (size_t segment, ChannelIdx z) const { return m_pixels[m_segments[segment].index][z]; }
inline float DeepPixel::getChannel (const DeepSegment& segment, ChannelIdx z) const { return m_pixels[segment.index][z]; }
//
inline void DeepPixel::setSegmentMask (const SpMask8& spmask, size_t start, size_t end)
{
    ++end;
    const size_t nSegments = m_segments.size();
    if (end < start || start >= nSegments)
        return;
    if (end >= nSegments)
        end = nSegments;
    for (size_t i=start; i < end; ++i)
        m_segments[i].metadata.spmask = spmask;
    m_accum_or_mask = m_accum_and_mask = SpMask8::zeroCoverage;
    m_sorted = false; // force it to re-evaluate
}

//
inline DeepPixel& DeepPixel::operator += (float val) {
    const size_t nSegments = m_segments.size();
    for (size_t i=0; i < nSegments; ++i)
        getSegmentPixel(i) += val;
    return *this;
}
inline DeepPixel& DeepPixel::operator -= (float val) {
    const size_t nSegments = m_segments.size();
    for (size_t i=0; i < nSegments; ++i)
        getSegmentPixel(i) -= val;
    return *this;
}
inline DeepPixel& DeepPixel::operator *= (float val) {
    const size_t nSegments = m_segments.size();
    for (size_t i=0; i < nSegments; ++i)
        getSegmentPixel(i) *= val;
    return *this;
}
inline DeepPixel& DeepPixel::operator /= (float val) {
    const float ival = 1.0f / val;
    const size_t nSegments = m_segments.size();
    for (size_t i=0; i < nSegments; ++i)
        getSegmentPixel(i) *= ival;
    return *this;
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_DEEPPIXEL_H
