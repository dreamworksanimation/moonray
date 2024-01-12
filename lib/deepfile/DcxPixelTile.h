// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_PIXELTILE_H
#define INCLUDED_DCX_PIXELTILE_H

//=============================================================================
//
//  class  PixelTile
//
//=============================================================================

#include "DcxDeepPixel.h"
#include "DcxChannelContext.h"

#include <OpenEXR/ImfHeader.h>


//-------------------------
//!rst:cpp:begin::
//.. _pixeltile_class:
//
//PixelTile
//=========
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


//=====================
//
//  class PixelTile
//
//=====================
//-----------------------------------------------------------------------------
//
//  Abstract class to manage a rectangular region of pixels.
//
//  This class is intended for simplifing common pixel-region loops that
//  step from pixel to pixel processing multiple channels simultaneously,
//  vs. using the IlmImfUtil classes which appear organized more for
//  per-plane processing and targeted primarily for texture use.
//
//-----------------------------------------------------------------------------

class DCX_EXPORT PixelTile
{
  public:

    //-----------------------------------------------------------------------------
    // Must provide a ChannelContext at a minimum. When sharing pixel data between
    // multiple PixelTiles they must share the same global ChannelContext to make
    // sure the ChannelIdx's align.
    // Data and display windows are set to invalid value [0 0 -1 -1]
    //-----------------------------------------------------------------------------

    PixelTile (ChannelContext& channel_ctx,
               bool yAxisUp=true);

    //----------------------------------------------------------------------------------
    // Build from an exr header. This will search for PixelTile related attributes like
    // formatName, apertureWindow, originalDisplayWindow, etc. If those are missing it
    // assigns intuitive defaults (apertureWindow=displayWindow, pa=1.0)
    //----------------------------------------------------------------------------------

    PixelTile (const OPENEXR_IMF_NAMESPACE::Header&,
               ChannelContext& channel_ctx,
               bool yAxisUp=true);


    //----------------------------------------------------------------------------
    // Assigns resolution and channel set.
    // The set of ChannelAliases is used to construct a map of ChannelIdx's to
    // ChannelAliases.  If multiple input ChannelAliases have the same ChannelIdx
    // destination only the first one is accepted and the others are ignored.
    //
    // display_window is only used to set the window-top reference.
    //----------------------------------------------------------------------------

    PixelTile (const IMATH_NAMESPACE::Box2i& data_window,
               const IMATH_NAMESPACE::Box2i& display_window,
               bool sourceWindowsYAxisUp,
               const ChannelSet& channels,
               ChannelContext& channel_ctx,
               bool yAxisUp=true);
    PixelTile (const IMATH_NAMESPACE::Box2i& data_window,
               int top_reference,
               bool sourceWindowsYAxisUp,
               const ChannelSet& channels,
               ChannelContext& channel_ctx,
               bool yAxisUp=true);

    virtual ~PixelTile ();


    //----------------------------------------------------------------------
    // Are the data/display windows coordinates Y-up?  If true pixel access
    // methods with Y-coordinate args are interpreted as Y-up.
    //
    // This is intended to help ease the translation of coordinates
    // between most modern applications which use Y-up and OpenEXR which
    // is Y-down.
    //----------------------------------------------------------------------

    bool    yAxisUp() const;

    int                     flipY(int y) const;
    IMATH_NAMESPACE::Box2i  flipY(const IMATH_NAMESPACE::Box2i& bbox) const;


    //--------------------------------------------------------------------------
    //  dataWindow: Bbox of active pixel area - *** possibly flipped in Y! ***
    //              Check yAxisUp() for direction. Requires a window-top
    //              reference, normally taken from display window args to ctors.
    //--------------------------------------------------------------------------

    const IMATH_NAMESPACE::Box2i&   dataWindow () const;
    int     minX () const;
    int     minY () const;
    int     maxX () const;
    int     maxY () const;
    int     width () const;
    int     height () const;
    int     w () const;
    int     h () const;


    //--------------------------------------------------------------------------
    // Change the active data window.
    // Virtual so that subclasses can reallocate memory, etc.
    //--------------------------------------------------------------------------

    virtual void    setDataWindow (const IMATH_NAMESPACE::Box2i& data_window,
                                   bool sourceWindowYAxisUp=true,
                                   bool force=false);


    //----------------------------
    // Global ChannelSet for tile.
    //----------------------------

    const OPENDCX_INTERNAL_NAMESPACE::ChannelSet&  channels () const;

    //-------------------------------------------------------
    // Assign the active ChannelSet.
    // Virtual so that subclasses can reallocate memory, etc.
    //-------------------------------------------------------

    virtual void setChannels (const ChannelSet&,
                              bool force=false);

    //------------------------------------------------
    // Number of color/aov channels in the ChannelSet.
    //------------------------------------------------

    size_t  numChannels () const;

    //---------------------------------------------
    // Global ChannelContext assigned to this tile.
    // Use this to access available ChannelAliases.
    //---------------------------------------------

    const ChannelContext* channelContext () const;


    //-------------------------------------------------------------
    // Return a ChannelAlias pointer corresponding to a ChannelIdx.
    //-------------------------------------------------------------

    const ChannelAlias* getChannelAlias(ChannelIdx) const;


    //-------------------------------------------------
    // Returns true if pixel x,y is inside data window.
    //-------------------------------------------------

    bool isActivePixel (int x, int y) const;
    bool isInside (int x, int y) const;


  protected:
    //--------------------------------------
    // Copy constructor only for subclasses.
    //--------------------------------------

    PixelTile (const PixelTile&);


    // Assigned vars:
    bool                    m_yaxis_up;             // Is Y-axis of region pointing up(industry-norm) or down(exr-std)?
    int                     m_top_reference;        // Window-top reference allowing Y-coords to be flipped by it
    IMATH_NAMESPACE::Box2i  m_dataWindow;           // Bbox of active pixel area, *** possibly flipped in Y! ***
    ChannelContext*         m_channel_ctx;          // Context for channels

    // Derived:
    OPENDCX_INTERNAL_NAMESPACE::ChannelSet         m_channels;             // ChannelSet shared by all pixels in region

};



//--------------
//!rst:cpp:end::
//--------------

//-----------------
// Inline Functions
//-----------------

inline bool PixelTile::yAxisUp () const { return m_yaxis_up; }
inline int  PixelTile::flipY(int y) const { return (m_yaxis_up)?(m_top_reference - y):y; }
inline IMATH_NAMESPACE::Box2i PixelTile::flipY(const IMATH_NAMESPACE::Box2i& bbox) const {
    IMATH_NAMESPACE::Box2i flipped(bbox);
    if (m_yaxis_up) {
        flipped.max.y = m_top_reference - bbox.min.y;
        flipped.min.y = m_top_reference - bbox.max.y;
    }
    return flipped;
}
inline const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& PixelTile::channels () const { return m_channels; }
inline const ChannelContext* PixelTile::channelContext () const { return m_channel_ctx; }
inline const ChannelAlias* PixelTile::getChannelAlias(ChannelIdx z) const { return m_channel_ctx->findChannelAlias(z); }
inline
bool PixelTile::isActivePixel (int x, int y) const { return !(x < m_dataWindow.min.x || y < m_dataWindow.min.y ||
                                                              x > m_dataWindow.max.x || y > m_dataWindow.max.y); }
inline bool PixelTile::isInside (int x, int y) const { return isActivePixel(x, y); }
inline size_t PixelTile::numChannels () const { return m_channels.size(); }
//-------------------------------------------------------
inline const IMATH_NAMESPACE::Box2i& PixelTile::dataWindow () const { return m_dataWindow; }
inline int PixelTile::minX () const { return m_dataWindow.min.x; }
inline int PixelTile::minY () const { return m_dataWindow.min.y; }
inline int PixelTile::maxX () const { return m_dataWindow.max.x; }
inline int PixelTile::maxY () const { return m_dataWindow.max.y; }
inline int PixelTile::width () const { return (m_dataWindow.max.x - m_dataWindow.min.x + 1); }
inline int PixelTile::height () const { return (m_dataWindow.max.y - m_dataWindow.min.y + 1); }
inline int PixelTile::w () const { return width(); }
inline int PixelTile::h () const { return height(); }

OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_PIXELTILE_H
