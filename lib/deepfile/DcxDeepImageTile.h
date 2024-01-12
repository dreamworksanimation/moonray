// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_DEEPIMAGETILE_H
#define INCLUDED_DCX_DEEPIMAGETILE_H

//=============================================================================
//
//  class  DeepImageOutputTile
//
//=============================================================================

#include "DcxDeepTile.h"
#include "DcxImageFormat.h"

#ifdef __ICC
// disable icc remark #1572: 'floating-point equality and inequality comparisons are unreliable'
//   this is coming from Imath/half.h...
#  pragma warning(disable:2557)
#endif
#include <OpenEXR/ImfDeepImage.h>
#include <OpenEXR/ImfDeepImageLevel.h>
#include <OpenEXR/ImfDeepScanLineOutputFile.h>

#ifdef DEBUG
#  include <assert.h>
#endif


//-----------------
//!rst:cpp:begin::
//DeepImageTile
//=============
//-----------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER

//------------------------------
//!rst:left-align::
//.. _deepimageoutputtile_class:
//
//DeepImageOutputTile
//===================
//------------------------------

//==============================
//
//  class DeepImageOutputTile
//
//==============================
//-----------------------------------------------------------------------------
//
//  Adapter class for an output DeepImage tile.
//
//
//  TODO: Disconnect this from DeepScanLineOutputFile so we can better
//        manage the deep IO.
//  TODO: support multiple DeepImages/Headers so that multiple Parts
//        can be combined into a single DeepPixel.
//
//-----------------------------------------------------------------------------


class DCX_EXPORT DeepImageOutputTile : public DeepTile, public ImageFormat
{
  public:

    typedef std::vector<half>     HalfVec;
    typedef std::vector<float>    FloatVec;
    typedef std::vector<uint32_t> UintVec;
    typedef std::vector<void*>    PtrVec;

    struct DeepLine
    {
        ChannelSet            channels;             // Channels which are in packed array
        std::vector<FloatVec> channel_arrays;       // Packed channel data for entire line
        std::vector<uint32_t> samples_per_pixel;    // Per-pixel sample count

        DeepLine (uint32_t width, const ChannelSet& _channels);

        uint32_t floatOffset (uint32_t xoffset) const;   // Get offset into channel_arrays for line x-offset

        void get (int xoffset,
                  OPENDCX_INTERNAL_NAMESPACE::DeepPixel& deep_pixel) const;
        void getMetadata(int xoffset,
                         size_t sample,
                         OPENDCX_INTERNAL_NAMESPACE::DeepMetadata& metadata) const;
        void set (int xoffset,
                  const OPENDCX_INTERNAL_NAMESPACE::DeepPixel& deep_pixel);
        void clear (int xoffset);
    };


  public:

    //
    // Sets resolution and channel info.
    //

    DeepImageOutputTile (const IMATH_NAMESPACE::Box2i& display_window,
                         const IMATH_NAMESPACE::Box2i& data_window,
                         bool sourceWindowsYup,
                         const ChannelSet& channels,
                         ChannelContext& channel_ctx,
                         bool yAxisUp=true);

    //
    ~DeepImageOutputTile ();


    //
    // Return the ImageFormat object
    //

    /*virtual*/ ImageFormat* format ();


    //
    // Change the set of channels.
    // Possibly destructive! If 'force'==true or the new channel set is
    // different than the current one all existing deep data will be deleted.
    //

    /*virtual*/ void    setChannels (const OPENDCX_INTERNAL_NAMESPACE::ChannelSet&,
                                     bool force=false);


    //
    // Change the active data window.
    // Possibly destructive! If 'force'==true or the new dataWindow is
    // different than the current one all existing deep data will be deleted.
    //

    /*virtual*/ void    setDataWindow (const IMATH_NAMESPACE::Box2i& data_window,
                                       bool sourceWindowYAxisUp=true,
                                       bool force=false);


    //
    // Returns the number of total bytes used for the entire tile.
    //

    size_t      bytesUsed () const;


    //
    // Get the DeepLine object for pixel-space line y
    //

    DeepLine*   getLine (int y) const;


    //
    // Returns the number of deep samples at pixel x,y.
    //

    /*virtual*/ size_t getNumSamplesAt (int x, int y) const;


    //
    // Reads deep samples from a pixel-space location (x, y) into a deep pixel.
    // If xy is out of bounds the deep pixel is left empty and false is returned.
    // If write access is sequential
    //

    /*virtual*/ bool getDeepPixel (int x,
                                   int y,
                                   OPENDCX_INTERNAL_NAMESPACE::DeepPixel& pixel) const;

    /*virtual*/ bool getSampleMetadata (int x,
                                        int y,
                                        size_t sample,
                                        OPENDCX_INTERNAL_NAMESPACE::DeepMetadata& metadata) const;


    //
    // Writes a DeepPixel to a pixel-space location (x, y) in the deep channels.
    // If xy is out of bounds false is returned.
    //

    /*virtual*/ bool setDeepPixel (int x,
                                   int y,
                                   const OPENDCX_INTERNAL_NAMESPACE::DeepPixel& pixel);


    //
    // Writes an empty DeepPixel (0 samples) at a pixel-space location (x, y).
    // If xy is out of bounds false is returned.
    //

    /*virtual*/ bool clearDeepPixel (int x,
                                     int y);

    //
    // Create an output deep file linked to this tile - destructive!
    // Will allocate a new Imf::DeepScanLineOutputFile and assign its
    // channels, destroying any current file.
    //
    // The Imf::Header argument variant allows arbitrary attributes
    // (metadata) to be copied into the output file. However some
    // attributes are overwritten by ones defined by this class. You
    // should override this method to change that behavior.
    //   Overwritten attributes:
    //      channels
    //      dataWindow
    //      displayWindow
    //      lineOrder
    //      pixelAspectRatio
    //      screenWindowCenter
    //      screenWindowWidth
    //

    virtual void    setOutputFile (const char* filename,
                                   const Imf::Header& headerToCopyAttributes);

    //
    // Write a DeepLine to the output file.  Must be called in the line_order
    // specified in setOutputFile otherwise image will be upside-down.  ex. if
    // write line order is 0-100 use INCREASING_Y.  RANDOM_Y writing is only
    // supported for tiled images.
    // If flush_line is true the DeepLine memory is freed.
    //

    virtual void    writeScanline (int y,
                                   bool flush_line=true);


    //
    // Write entire tile to output file.
    // If flush_tile is true all DeepLine memory is freed.
    //

    virtual void    writeTile (bool flush_tile=true);


  protected:

    void        deleteDeepLines ();
    DeepLine*   createDeepLine (int y);


    std::vector<DeepLine*>          m_deep_lines;           // Channel data storage
    std::string                     m_filename;
    OPENEXR_IMF_NAMESPACE::DeepScanLineOutputFile* m_file;  // Output file, if assigned

};



//--------------
//!rst:cpp:end::
//--------------

//-----------------
// Inline Functions
//-----------------

/*virtual*/ inline ImageFormat* DeepImageOutputTile::format () { return this; }

//-------------------------------------------------------
inline
DeepImageOutputTile::DeepLine* DeepImageOutputTile::getLine(int y) const {
    return (y < m_dataWindow.min.y || y > m_dataWindow.max.y)?0:m_deep_lines[y - m_dataWindow.min.y]; }
//-------------------------------------------------------
inline
uint32_t
DeepImageOutputTile::DeepLine::floatOffset (uint32_t xoffset) const
{
#ifdef DEBUG
    assert(xoffset < samples_per_pixel.size());
#endif
    uint32_t offset = 0;
    const uint32_t* p = samples_per_pixel.data();
    for (uint32_t i=0; i < xoffset; ++i)
        offset += *p++;
    return offset;
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_DEEPIMAGETILE_H
