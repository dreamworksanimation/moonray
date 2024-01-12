// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DcxDeepPixel.h"

#include <algorithm> // for std::sort in some compilers

// Handle the Z accumulation with coverage weighting - i.e. pick Z from
// the sample with largest coverage.
// Finishing this code will likely produce more accurate flattened Z's:
#define DCX_USE_LARGEST_COVERAGE_Z 1


// Uncomment this to get debug info about DeepPixel:
//#define DCX_DEBUG_DEEPPIXEL 1
//#define DCX_DEBUG_COLLAPSING 1
//#define DCX_DEBUG_FLATTENER 1

#if defined(DCX_DEBUG_DEEPPIXEL) || defined(DCX_DEBUG_COLLAPSING) || defined(DCX_DEBUG_FLATTENER)
#  include <assert.h>
#  define SAMPLER_X -100000 // special debug coord
#  define SAMPLER_Y -100000 // special debug coord
#  define SAMPLER_Z 0
#  define SAMPLINGXY(A, B) (A==SAMPLER_X && B==SAMPLER_Y)
#  define SAMPLINGXYZ(A, B, C) (A==SAMPLER_X && B==SAMPLER_Y && C==SAMPLER_Z)
#endif

#include <map>

OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


// For now we're using a fixed number of steps:
#ifdef DCX_DEBUG_FLATTENER
#  define SEGMENT_SAMPLE_STEPS 5
#else
#  define SEGMENT_SAMPLE_STEPS 5//15
#endif

// Maxium absorbance is what we consider an opaque surface.
// For this we are using an alpha of 0.999999999999
// So absorbance = -log( 1.0 - alpha )
// So absorbance = more or less 12.0
#define MAX_ABSORBANCE  12.0

// Clamp 0...1
template <class T>
inline
T CLAMP(T v)
{
    return std::max((T)0, std::min((T)1, v));
}

//----------------------------------------------------------------------------------------

/*static*/ const SpMask8 SpMask8::zeroCoverage(SpMask8::allBitsOff);
/*static*/ const SpMask8 SpMask8::fullCoverage(SpMask8::allBitsOn);

/*friend*/
std::ostream&
operator << (std::ostream& os,
             const SpMask8& mask)
{
    os << "0x";
    const std::ios_base::fmtflags flags = os.flags();
    const char fill = os.fill();
    const int w = (int)os.width();
    os.fill('0');
    os.width(16);
    os << std::hex << mask.value();
    os.flags(flags);
    os.fill(fill);
    os.width(w);
    return os;
}

//----------------------------------------------------------------------------------------

/*friend*/
std::ostream&
operator << (std::ostream& os,
             const DeepFlags& flags)
{
    os << "0x";
    const std::ios_base::fmtflags osflags = os.flags();
    const char fill = os.fill();
    const int w = (int)os.width();
    os.fill('0');
    os.width(16);
    os << std::hex << flags.bits;
    os.flags(osflags);
    os.fill(fill);
    os.width(w);

    return os;
}


void
DeepFlags::print (std::ostream& os) const
{
    if (bits == DeepFlags::ALL_BITS_OFF)
    {
        os << "none(Log,NoMatte)";
        return;
    }
    if (bits & OPENDCX_INTERNAL_NAMESPACE::DeepFlags::LINEAR_INTERP)
        os << "Linear";
    else
        os << "Log";
    if (bits & OPENDCX_INTERNAL_NAMESPACE::DeepFlags::MATTE_OBJECT)
        os << ",Matte";
    if (bits & OPENDCX_INTERNAL_NAMESPACE::DeepFlags::ADDITIVE)
        os << ",Additive";
    if (hasPartialSpCoverage())
    {
        os.precision(3);
        os << ",spCvg(" << std::fixed << getSpCoverageWeight() << ")";
    }
}

//----------------------------------------------------------------------------------------

//
// Prints segment info and metadata, but no channel data.
//

void
DeepSegment::printInfo (std::ostream& os,
                        bool show_mask) const
{
    const std::streamsize prec = os.precision();
    os.precision(8);
    os << "Zf=" << Zf << ", Zb=" << Zb << ", flags=["; metadata.flags.print(os); os << "]";
    if (show_mask) {
        os << std::endl;
        metadata.spmask.printPattern(os, "  ");
    } else
        os << ", spMask=" << metadata.spmask;
    os.precision(prec);
}

//
// Print a nicely formatted list of sample info including flags and subpixel mask patterns.
//

void
DeepPixel::printInfo (std::ostream& os,
                      const char* prefix,
                      int padding,
                      bool show_mask)
{
    if (prefix && prefix[0])
        os << prefix;
    os << "{ xy[" << m_x << " " << m_y << "]";
    if (m_segments.size() > 0)
    {
        const std::ios_base::fmtflags flags = os.flags();
        const std::streamsize prec = os.precision();
        this->sort(true/*force*/); // Get global info up to date
        os << " overlaps=" << m_overlaps << " allFullCoverage=" << allFullCoverage();
        os << " ANDmask=" << m_accum_and_mask << " ORmask=" << m_accum_or_mask;
        os << " ANDflags=0x" << std::hex << m_accum_and_flags << " ORflags=0x" << m_accum_or_flags << std::dec;
        os << std::endl;
        padding = std::max(0, std::min((int)padding, 1024));
        std::vector<char> spaces(padding*2+1, ' ');
        spaces[spaces.size()-1] = 0;
        for (uint32_t i=0; i < m_segments.size(); ++i) {
            const DeepSegment& segment = m_segments[i];
            os.precision(8);
            os << &spaces[padding] << i << ": Zf=" << std::fixed << segment.Zf << " Zb=" << segment.Zb;
            //
            os << " flags=["; segment.printFlags(os); os << "]";
            //
            if (!show_mask)
                os << " spMask=" << segment.spMask();
            //
            os << " chans=" << getSegmentPixel(i) << std::endl;
            if (show_mask)
                segment.spMask().printPattern(os, &spaces[0]);
        }
        os.flags(flags);
        os.precision(prec);
    }
    else
    {
        os << " empty ";
    }
    os << "}" << std::endl;
}


//----------------------------------------------------------------------------------------

/*friend*/
std::ostream&
operator << (std::ostream& os,
             const DeepMetadata& metadata)
{
    os << "[flags=" << metadata.flags;
    os << ", spCvg=" << metadata.getSpCoverageWeight();
    os << ", spMask=" << std::hex << metadata.spmask << std::dec << "]";
    return os;
}


//
// Outputs a raw list of values suitable for parsing.
//

/*friend*/
std::ostream&
operator << (std::ostream& os,
             const DeepSegment& ds)
{
    os << "[Zf=" << ds.Zf << ", Zb=" << ds.Zb << ", flags=" << ds.metadata.flags;
    os << ", spCvg=" << ds.metadata.getSpCoverageWeight();
    os << ", spMask=" << std::hex << ds.metadata.spmask << std::dec << ", coverage=" << ds.getCoverage() << "]";
    return os;
}

//
// Outputs a raw list of samples and channel contents that's suitable for parsing.
//

std::ostream&
operator << (std::ostream& os,
             DeepPixel& dp)
{
    dp.sort(true/*force*/); // Get global info up to date
    os << "{";
    if (dp.size() > 0)
    {
        for (uint32_t i=0; i < dp.size(); ++i)
        {
            if (i > 0)
                os << " ";
            os << dp[i].Zf << ":" << dp[i].Zb << "[";
            foreach_channel(z, dp.channels())
                os << " " << dp.getChannel(i, z);
            os << " ]";
        }
    }
    os << "}";
    return os;
}


/*static*/
const char*
DeepPixel::interpolationModeString (InterpolationMode mode)
{
    switch (mode)
    {
    case INTERP_OFF:  return "off";     // Disable interpolation
    case INTERP_AUTO: return "auto";    // Determine interpolation from per-sample metadata (DeepFlags)
    case INTERP_LOG:  return "log";     // Use log interpolation for all samples
    case INTERP_LIN:  return "linear";  // Use linear interpolation for all samples
    default: return "invalid";
    }
}


//
//  Empty the segment list and clear shared values, except for channel set.
//

void
DeepPixel::clear ()
{
    m_segments.clear();
    m_pixels.clear();

    invalidateSort(); // force sort to re-run
}


//
//  Add a DeepSegment to the end of the list.
//

size_t
DeepPixel::append (const DeepSegment& bs,
                   const Pixelf& bp)
{
    const size_t new_segment_index = m_segments.size();
    const size_t new_pixel_index   = m_pixels.size();

    if (m_segments.capacity() < m_segments.size()+1)
        m_segments.reserve(m_segments.size() + (size_t)int(float(m_segments.size())/1.5f));
    if (m_pixels.capacity() < m_pixels.size()+1)
        m_pixels.reserve(m_pixels.size() + (size_t)int(float(m_pixels.size())/1.5f));
    m_segments.push_back(bs);
    m_pixels.push_back(bp);

    m_pixels[new_pixel_index].channels  = m_channels;
    m_segments[new_segment_index].index = (int)new_pixel_index;

    invalidateSort(); // force sort to re-run

    return new_segment_index;
}


//
//  Add an empty DeepSegment to the end of the list, returning its index.
//

size_t
DeepPixel::append ()
{
    return this->append(DeepSegment(), Pixelf(m_channels));
}


//
//  Add a DeepSegment to the end of the list.
//

size_t
DeepPixel::append (const DeepSegment& bs)
{
    return this->append(bs, Pixelf(m_channels));
}


//
//  Copy one segment from a second DeepPixel.
//

size_t
DeepPixel::append (const DeepPixel& b,
                   size_t segment_index)
{
#ifdef DCX_DEBUG_DEEPPIXEL
    assert(segment_index < b.m_segments.size());
#endif
    const DeepSegment& bs = b[segment_index];
    const Pixelf& bp = b.m_pixels[bs.index];
    return this->append(bs, bp);
}


//
//  Combine the segments of two DeepPixels.
//

void
DeepPixel::append (const DeepPixel& b)
{
    const size_t nCurrent = m_segments.size();
    // Clear any new channels that are being added by the new set
    // so we don't end up with junk left in the memory:
    foreach_channel(z, b.m_channels)
    {
        if (!m_channels.contains(z))
        {
            for (size_t i=0; i < nCurrent; ++i)
                m_pixels[i][z] = 0.0f;
        }
    }
    m_channels += b.m_channels; // OR the channel masks together
    // Copy the segments:
    const size_t nAdded = b.m_segments.size();
    m_segments.reserve(m_segments.size() + nAdded);
    m_pixels.reserve(m_pixels.size() + nAdded);
    for (size_t i=0; i < nAdded; ++i)
    {
        m_segments.push_back(b.m_segments[i]);
        m_pixels.push_back(b.m_pixels[i]);
        m_segments[m_segments.size()-1].index = (int)(m_pixels.size()-1);
        m_pixels[m_pixels.size()-1].channels = m_channels;
    }

    invalidateSort(); // force sort to re-run
}

//-------------------------------------------------------------------------------------

//
//  Remove a DeepSegment from the segment list, deleting its referenced Pixel.
//  Note that this method will possibly reorder some of the Pixel indices in the
//  DeepSegments, so a previously referenced Pixel index may become invalid and
//  need to be reaquired from its DeepSegment.
//

void DeepPixel::removeSegment (size_t segment_index)
{
    if (segment_index >= m_segments.size())
        return;
    const int overwritePixelIndex = m_segments[segment_index].index;
    const int lastPixelIndex = (int)(m_pixels.size()-1);
    if (overwritePixelIndex < lastPixelIndex)
    {
        // Find the DeepSegment pointing at the last Pixel, then
        // copy the last Pixel in the list over the one being deleted
        // and update the index in the changed DeepSegment:
        const size_t nSegments = m_segments.size();
        size_t lastPixelSegmentIndex = 0;
        for (; lastPixelSegmentIndex < nSegments; ++lastPixelSegmentIndex)
            if (m_segments[lastPixelSegmentIndex].index == lastPixelIndex)
                break;
#ifdef DCX_DEBUG_DEEPPIXEL
        assert(lastPixelSegmentIndex < nSegments); // shouldn't happen...
#endif
        m_pixels[overwritePixelIndex] = m_pixels[lastPixelIndex];
        m_segments[lastPixelSegmentIndex].index = overwritePixelIndex;
    }
    // Delete the DeepSegment and the last Pixel:
    m_segments.erase(m_segments.begin()+segment_index);
    m_pixels.pop_back();
#ifdef DCX_DEBUG_DEEPPIXEL
    assert(m_segments.size() == m_pixels.size());
#endif
    invalidateSort(); // force sort to re-run
}


//-------------------------------------------------------------------------------------

//
//  Return the index of the DeepSegment nearest to Z and inside the distance
//  of Z + or - maxDistance. Return -1 if nothing found.
//

// TODO: finish implementing this!
#if 0
int
DeepPixel::nearestSegment (double Z,
                           double maxDistance)
{
    if (m_segments.size() == 0)
        return -1;
    this->sort();
    DeepSegment ds;
    ds.Zf = Z;
    std::vector<DeepSegment >::const_iterator i = lower_bound(m_segments.begin(), m_segments.end(), ds);
    if (i == m_segments.end())
        i = m_segments.end()-1;
    printf("Z=%f, i=%d[%f]\n", Z, (int)(i - m_segments.begin()), i->Zf);
    return (int)(i - m_segments.begin());
}
#endif


//
//  Sort the segments.  If the sorted flag is true this returns quickly.
//  This also updates global overlap and coverage flags.
//

void
DeepPixel::sort (bool force)
{
    if (m_sorted && !force)
        return;
    m_sorted = true;
    m_overlaps = false;
    const size_t nSegments = m_segments.size();
    if (nSegments == 0)
    {
        m_accum_or_mask  = m_accum_and_mask  = SpMask8::zeroCoverage;
        m_accum_or_flags = m_accum_and_flags = DeepFlags::ALL_BITS_OFF;
        return;
    }

    // Z-sort the segments:
    std::sort(m_segments.begin(), m_segments.end());

    const DeepSegment& segment0 = m_segments[0];

    float prev_Zf = segment0.Zf;
    float prev_Zb = segment0.Zb;
    // Make sure mixed legacy full-coverage(allBitsOff) and true full-coverage(allBitsOn)
    // segments are both considered full-coverage:
    m_accum_or_mask  = m_accum_and_mask  = (segment0.fullCoverage())?SpMask8::fullCoverage:segment0.spMask();
    m_accum_or_flags = m_accum_and_flags = segment0.flags();

    // Determine global overlap and coverage status.
    for (size_t i=1; i < nSegments; ++i)
    {
        const DeepSegment& segment = m_segments[i];
        if (segment.Zf < prev_Zf || segment.Zb < prev_Zf ||
            segment.Zf < prev_Zb || segment.Zb < prev_Zb)
            m_overlaps = true;

        const SpMask8 spmask = (segment.fullCoverage())?SpMask8::fullCoverage:segment.spMask();
        m_accum_or_mask   |= spmask;
        m_accum_and_mask  &= spmask;
        m_accum_or_flags  |= segment.flags();
        m_accum_and_flags &= segment.flags();

        prev_Zf = segment.Zf;
        prev_Zb = segment.Zb;
    }
}

//
// Force sort to re-run.
//

void DeepPixel::invalidateSort ()
{
    m_sorted = m_overlaps = false;
    m_accum_or_mask  = m_accum_and_mask  = SpMask8::zeroCoverage;
    m_accum_or_flags = m_accum_and_flags = DeepFlags::ALL_BITS_OFF;
}


//
//  Check for overlaps between samples and return true if so.
//  If the spmask arg is not full coverage then determine overlap
//  for the specific subpixel mask.
//

bool
DeepPixel::hasOverlaps (const SpMask8& spmask,
                        bool force)
{
    const size_t nSegments = m_segments.size();
    if (nSegments == 0)
        return false;
    // Sorting calculates the global overlap:
    if (!m_sorted || force)
        this->sort(force);
    // If full coverage then we can return the global overlap indicator:
    if (spmask == SpMask8::fullCoverage || allFullCoverage() || isLegacyDeepPixel())
        return m_overlaps;
    // Determine the overlap status for the spmask:
    float prev_Zf = -INFINITYf;
    float prev_Zb = -INFINITYf;
    for (size_t i=0; i < nSegments; ++i)
    {
        const DeepSegment& segment = m_segments[i];
        if (segment.maskBitsEnabled(spmask))
        {
            if (segment.Zf < prev_Zf || segment.Zb < prev_Zf ||
                segment.Zf < prev_Zb || segment.Zb < prev_Zb)
                return true;
            prev_Zf = segment.Zf;
            prev_Zb = segment.Zb;
        }
    }
    return false;
}

bool
DeepPixel::allZeroCoverage ()
{
    sort(); // updates flags
    return (m_accum_or_mask == SpMask8::zeroCoverage);
}

bool
DeepPixel::allFullCoverage ()
{
    sort(); // updates flags
    return (m_accum_and_mask == SpMask8::fullCoverage || m_accum_or_mask == SpMask8::zeroCoverage);
}

bool
DeepPixel::allVolumetric ()
{
    sort(); // updates flags
    return ((m_accum_and_flags & DeepFlags::LINEAR_INTERP)==0 &&
            (m_accum_or_flags  & DeepFlags::LINEAR_INTERP)==0);
}
bool
DeepPixel::anyVolumetric ()
{
    sort(); // updates flags
    return (m_accum_or_flags & DeepFlags::LINEAR_INTERP)==0;
}

bool
DeepPixel::allHardSurface ()
{
    sort(); // updates flags
    return (m_accum_and_flags & DeepFlags::LINEAR_INTERP)!=0;
}
bool
DeepPixel::anyHardSurface ()
{
    sort(); // updates flags
    return (m_accum_or_flags & DeepFlags::LINEAR_INTERP)!=0;
}

bool
DeepPixel::allMatte ()
{
    sort(); // updates flags
    return (m_accum_and_flags & DeepFlags::MATTE_OBJECT)!=0;
}
bool
DeepPixel::anyMatte ()
{
    sort(); // updates flags
    return (m_accum_or_flags & DeepFlags::MATTE_OBJECT)!=0;
}

bool
DeepPixel::isLegacyDeepPixel ()
{
    sort(); // updates flags
    return (allZeroCoverage() && allVolumetric());
}


//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------


//----------------------------------------------------------------------------

OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT
