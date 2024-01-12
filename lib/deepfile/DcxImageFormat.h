// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_IMAGEFORMAT_H
#define INCLUDED_DCX_IMAGEFORMAT_H

#include "DcxAPI.h"

#include <OpenEXR/ImfHeader.h>
#include <Imath/ImathMatrix.h>

#include <string>
#include <iostream>


//-------------------------
//!rst:cpp:begin::
//.. _imageformat_class:
//
//ImageFormat
//===========
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


const std::string FORMATNAME_ATTRIBUTE      = "formatName";
const std::string FORMATREFERENCE_ATTRIBUTE = "formatReference";
const std::string APERTUREWINDOW_ATTRIBUTE  = "displayAperture";


//=======================
//
// class ImageFormat
//
//=======================
//---------------------------------------------------------------------------------------------------
//
//  Image pixel resolution description class containing most of the info required
//  for handling pixel space windows and how they map into & out of common camera
//  projections.
//
//  For an explanation of the basic terminology see the OpenEXR TechnicalIntroduction.pdf,
//  pages 4 & 5.
//  Extensions to the OpenEXR basics and reasons why they were required:
//
//  **apertureWindow**:
//      This rectangle is not in the base OpenEXR definition and is intended to clearly define
//      the extent of the *original* camera aperture in pixel space before any expansions, and
//      in the same coordinate context (scaling) as the display and data windows. For cg renders
//      the lower-left corner of apertureWindow will be 0,0 allowing the displayWindow to expand
//      the camera aperture into negative coords without losing the alignment of the original
//      aperture as commonly happens in renderers.
//      Using projection matrices to determine the alignment is often impossible due to the
//      inclusion of stereo offsets or other lens related aperture manipulations.
//
//      Depending on the usage context of the image (layout/render, composite, delivery, etc)
//      apertureWindow can be thought of as:
//
//        * Shooting camera aperture. i.e. the original, unexpanded camera aperture rectangle
//          in pixel coordinates for a cg render before any filmback expansions.
//        * Final delivery rectangle. i.e. what gets shown to a viewer or audience. Often images
//          are worked on with additional padding to allow for stereo repositioning, shake, blur,
//          etc, and this extra padding is typically trimmed off before final delivery. Quite often
//          the final delivery rectangle is the same as the shooting camera aperture.
//        * Sub-composition rectangle: ex. a special project might require multiple camera
//          compositions to be combined into a single image - like an old multi-media slide show.
//          Each image could have a different apertureWindow defining that camera composition's
//          location within the final image.
//
//      OpenEXR does provide the attributes 'screenWindowCenter' and 'screenWindowWidth' as a way
//      of defining the camera aperture but these are inadequate in practice for several reasons:
//
//        1) They are floating-point values and subject to rounding when converted to pixel
//           coordinates. Because OpenEXR does not explicitly define the rounding math each
//           implementation is free to round differently resulting in resolution differences.
//        2) Does not fully define camera aperture expansions. While setting screenWindowWidth
//           allows the horizontal aperture expansion to be clearly set, the reliance on pixel
//           aspect ratio to derive screenWindowHeight means that vertical aperture expansions
//           with square pixels are impossible to define. There are plenty of reasons to expand
//           the vertical aperture while not affecting pixel aspect ratio!
//        3) Most (all?) applications completely ignore these variables, perhaps because of
//           reasons 1 & 2.
//
//  **formatReference**:
//      Typically the non-proxy or unscaled format used to determine the current proxy scaling of
//      the format and to support absolute scaling of pixel coords. While the term 'proxy' is
//      commonly thought of as a reduction in scale of an image to speed up iterations, it can
//      also be used to scale up an image or completely change the shape of an image, as is often
//      done when producing publicity content or print-res versions of imagery.
//      Managing proxy resolutions is difficult with the current OpenEXR standard as there is no
//      defined or suggested handling methodology. Rather than storing the proxy value as a scalar
//      which is subject to accuracy and rounding issues when applied to inetger pixel rectangles,
//      and is difficult to apply to all image parameters like pixelAspectRatio, the unscaled,
//      original, or 'reference' format definition is stored in string form.
//
//  **Example stereo-window format with negative displayWindow offset:**
//
//  .. image:: ../images/image_format_ex01a.jpg
//
//  **Example stereo-window format with positive apertureWindow offset:**
//
//  .. image:: ../images/image_format_ex01b.jpg
//
//  **Example format with negative xy displayWindow offset:**
//
//  .. image:: ../images/image_format_ex02.jpg
//
//---------------------------------------------------------------------------------------------------

class DCX_EXPORT ImageFormat
{
  public:

    //-----------------------------------------------
    // Resizing fit modes for reformatting operations
    //-----------------------------------------------

    enum FitTarget
    {
        FIT_WIDTH,       //
        FIT_HEIGHT,      //
        FIT_BOTH,        //
        FIT_SMALLER,     //
        FIT_LARGER       //
    };


  public:

    //-----------------------------------------------------------
    // Default ctor leaves name="", windows=[0,0,-1,-1] and pa=1.
    //-----------------------------------------------------------

    ImageFormat ();


    //-----------------------------------------------------------------------------------
    // Build from an exr header. This will search for ImageFormat related attributes like
    // formatName, apertureWindow, originalDisplayWindow, etc. If those are missing it
    // assigns intuitive defaults like apertureWindow=displayWindow.
    //-----------------------------------------------------------------------------------

    ImageFormat (const OPENEXR_IMF_NAMESPACE::Header&);

    //-----------------------------------------
    // Assumes apertureWindow == displayWindow
    //-----------------------------------------

    ImageFormat (const std::string& name,
                 const IMATH_NAMESPACE::Box2i& display_window,
                 float pixel_aspect=1.0f);

    //-----------------------------------------
    // Assumes apertureWindow == displayWindow
    //-----------------------------------------

    ImageFormat (const std::string& name,
                 int display_window[4],
                 float pixel_aspect=1.0f);

    //-----------------------------------------------------------------------------
    // Display win must be >= aperture win so any corners that fall inside aperture
    // will be clamped to it.
    // Int array args are defined as [minX, minY, maxX, maxY]
    //-----------------------------------------------------------------------------

    ImageFormat (const std::string& name,
                 const IMATH_NAMESPACE::Box2i& display_window,
                 const IMATH_NAMESPACE::Box2i& aperture_window,
                 float pixel_aspect=1.0f);
    ImageFormat (const std::string& name,
                 int disp[4],
                 int aper[4],
                 float pixel_aspect=1.0f);
    ImageFormat (const std::string& name,
                 int dispMinX, int dispMinY, int dispMaxX, int dispMaxY,
                 int aperMinX, int aperMinY, int aperMaxX, int aperMaxY,
                 float pixel_aspect=1.0f);


    virtual ~ImageFormat ();


  public:

    //----------------------------------------------
    // Return the ImageFormat object for subclasses.
    // Subclasses should return 'this'.
    //----------------------------------------------

    virtual ImageFormat* format ();


    //-----------------------------------------------------------------------------
    //
    // Pixel-space windows.
    //
    //
    //      displayWindow:  'Working' image region. Can include aperture expansions
    //                      for stereo offsets, shake, blur, etc.
    //      apertureWindow: Camera aperture region, prior to any expansions. Always
    //                      equal to or smaller than displayWindow.
    //
    //-----------------------------------------------------------------------------

    const IMATH_NAMESPACE::Box2i& displayWindow () const;
    const IMATH_NAMESPACE::Box2i& apertureWindow() const;


    //-----------------------------------------------------------------------------
    // formatReference string.
    // Typically the non-proxy or unscaled format used to determine the current
    // proxy scaling of the format and to support absolute scaling of pixel coords.
    //-----------------------------------------------------------------------------

    const std::string&  formatReference () const;

    //-------------------------------------------------------------------------------
    //
    // Display window values.
    //
    // The displayWindow in OpenEXR is commonly defined as the extent of the working
    // image area, as opposed to the dataWindow which is the region of active pixels.
    // See OpenEXR TechnicalIntroduction.pdf, pages 4 & 5
    //
    //-------------------------------------------------------------------------------

    int                     displayMinX () const;
    int                     displayMinY () const;
    int                     displayMaxX () const;
    int                     displayMaxY () const;
    int                     displayW () const;
    int                     displayH () const;
    int                     displayWidth ()  const;
    int                     displayHeight () const;
    float                   displayCenterX () const;
    float                   displayCenterY () const;
    IMATH_NAMESPACE::V2i    displayMin () const;
    IMATH_NAMESPACE::V2i    displayMax () const;
    IMATH_NAMESPACE::V2f    displayMin2f () const;
    IMATH_NAMESPACE::V2f    displayMax2f () const;
    IMATH_NAMESPACE::V2f    displayCenter2f () const;
    IMATH_NAMESPACE::V3f    displayMin3f () const;
    IMATH_NAMESPACE::V3f    displayMax3f () const;
    IMATH_NAMESPACE::V3f    displayCenter3f () const;


    //--------------------------------------------------------------
    // Returns true if format has been initialized successfully.
    // Formats with zero displayWindow w/h are not considered valid.
    //--------------------------------------------------------------

    virtual bool    isValid () const;


    //-------------------------------------------------------------------------------------------
    //
    // Aperture window values.
    //
    // Defines the extent of the *original* camera aperture in pixel space before any expansions,
    // and is in the same coordinate context (scaling) as the display and data windows. Can also
    // be used to define the final delivery region of an image.
    //
    // For typical cg renders the lower-left corner of apertureWindow should be 0,0 allowing an
    // expanded displayWindow to shift into negative coords without losing the alignment of the
    // original camera aperture. However in practice most renderers will keep the expanded
    // displayWindow's lower-left cornder at 0,0 causing the camera aperture to shift up and to
    // the right.
    //
    // This offset is typically not captured in metadata so it's difficult or impossible to
    // automatically align images rendered with same camera aperture but with aperture expansions.
    // Using projection matrices to determine the offset is often impossible due to the inclusion
    // of stereo offsets or other lens related aperture manipulations.
    //
    // See the ImageFormat class notes above for more info.
    //
    //-------------------------------------------------------------------------------------------

    int                     apertureMinX () const;
    int                     apertureMinY () const;
    int                     apertureMaxX () const;
    int                     apertureMaxY () const;
    int                     apertureW () const;
    int                     apertureH () const;
    int                     apertureWidth () const;
    int                     apertureHeight () const;
    float                   apertureCenterX () const;
    float                   apertureCenterY () const;
    IMATH_NAMESPACE::V2i    apertureMin () const;
    IMATH_NAMESPACE::V2i    apertureMax () const;
    IMATH_NAMESPACE::V2f    apertureMin2f () const;
    IMATH_NAMESPACE::V2f    apertureMax2f () const;
    IMATH_NAMESPACE::V2f    apertureCenter2f () const;
    IMATH_NAMESPACE::V3f    apertureMin3f () const;
    IMATH_NAMESPACE::V3f    apertureMax3f () const;
    IMATH_NAMESPACE::V3f    apertureCenter3f () const;


    //------------------------------------------------
    // Returns true if apertureWindow == displayWindow
    //------------------------------------------------

    bool    apertureMatchesDisplay () const;


    //--------------------------------
    // Returns the pixel aspect ratio.
    //--------------------------------

    float   pixelAspectRatio () const;


    //----------------------------------------------------------------
    // ScreenWindow parameters derived from aperture & display windows
    // scaled to match the OpenEXR spec of [-0.5...+0.5].
    // screenWindowWidth is always fitted to aperture width while
    // screenWindowHeight includes pixel aspect ratio scaling.
    //----------------------------------------------------------------

    IMATH_NAMESPACE::V2f    screenWindowCenter () const;
    float                   screenWindowWidth () const;
    float                   screenWindowHeight () const;


    //--------------------------------------------------------------------
    // Transform a pixel space coordinate to/from NDC [-1.0...+1.0] space.
    //--------------------------------------------------------------------

    void                    pixelToNdc (float x, float y,
                                        float& u, float& v) const;
    IMATH_NAMESPACE::V2f    pixelToNdc (float x, float y) const;
    IMATH_NAMESPACE::V2f    pixelToNdc (const IMATH_NAMESPACE::V2f& pixelXY) const;

    void                    ndcToPixel (float u, float v,
                                        float& x, float& y) const;
    IMATH_NAMESPACE::V2f    ndcToPixel (float u, float v) const;
    IMATH_NAMESPACE::V2f    ndcToPixel (const IMATH_NAMESPACE::V2f& ndcUV) const;


    //=======================================================
    //
    // Header I/O
    //
    //=======================================================

    //-------------------------------------------------------------------------
    // Get the ImageFormat from an Imf::Header.  If header is missing the extra
    // format attributes like 'apertureWindow' they are intuitively built using
    // the standard windows.
    //-------------------------------------------------------------------------

    virtual void    fromHeader (const OPENEXR_IMF_NAMESPACE::Header&);

    //----------------------------------------------------------------------------
    // Sets the Imf::Header to the info from the ImageFormat.
    // Sets displayWindow, pixelAspectRatio, and adds the extra format attributes.
    // Does not affect the data window.
    //----------------------------------------------------------------------------

    virtual void    toHeader (OPENEXR_IMF_NAMESPACE::Header&) const;


    //=======================================================
    //
    // String I/O
    //
    //=======================================================

    //--------------------------------------------------------------
    // Serialize the format parameters to a string or output stream.
    // If brackets=true wrap it in '[]'
    //--------------------------------------------------------------

    virtual void            toString (std::ostream& os,
                                      bool brackets=false) const;
    virtual std::string     toString (bool brackets=false) const;

    virtual void            printInfo (std::ostream& os) const;


    //------------------------------------------------------------------------
    //!rst:left-align::
    // **Format name handling**
    //
    // If a format is constructed without being assigned an explicit name
    // an ad-hoc name of the display window's width x height is constructed.
    // i.e. '2048x1556'.
    //
    // If the format is a proxy scaled version of a 'full-res' format
    // then the end of the name may contain a proxy scale factor in
    // parentheses.
    // For example if a format with the name 'hd_2.35_hvp24' is proxy scaled
    // by one-third (0.3333...) then the resulting name would be
    // 'hd_2.35_hvp24(s1/3)', but if the scale factor is 3.0 then the name is
    // 'hd_2.35_hvp24(s3)'. This is to avoid floating-point accuracy issues.
    //
    // Use getScaleFromName() to get the name of the original, full-res
    // format.
    //
    //------------------------------------------------------------------------

    //----------------------------
    // Format's identifier string.
    //----------------------------

    const std::string&  name () const;


    //---------------------------------------------------------------
    // Builds a new format name for this format from the scale value.
    //---------------------------------------------------------------

    std::string     buildScaledName (double scale) const;

    //--------------------------------------------------------------
    // Build a new format name from the provided name & scale value,
    // replacing any existing scale value.
    // Or, strip scale notation off the end of a format name.
    //--------------------------------------------------------------

    static std::string  appendScaleToName (const std::string& name,
                                           double scale);
    static std::string  stripScaleFromName (const std::string& name_with_scale,
                                            double* scale_val=NULL);

    //-------------------------------------------------------------------------------
    // Warning - the text name scale notation may not be very accurate - get the
    // scale from proxyScale() which is displayWindow / formatReference.displayWindow.
    //-------------------------------------------------------------------------------

    static double   getScaleFromName (const std::string& name_with_scale);


    //---------------------------------------------------------
    // Scale notation string is of the form 's###' or 's1/###'.
    // ex. 's2' 's1/2' 's4.23' 's1/3.33'
    //---------------------------------------------------------

    static std::string  toScaleNotation (double scale);
    static double       fromScaleNotation (const std::string& s);


    //=======================================================
    //
    // Format transforms
    //
    //=======================================================

    //void    createProxyFormat (double scale,
    //                           ImageFormat& proxy_format);

    //void    createProxyFormat (double scale,
    //                           ImageFormat& proxy_format);


    //=======================================================
    //
    // Pixel/NDC space transforms
    //
    //=======================================================

    //--------------------------------------------------------------------
    // Transforms a window to ndc coords using the ImageFormat's aperture.
    //--------------------------------------------------------------------

    IMATH_NAMESPACE::Box2f  windowToNdc (const IMATH_NAMESPACE::Box2i& in) const;


    //-----------------------------------------------------------------------------------------------
    // Multiply the provided matrix by a transform to project ndc coords to ImageFormat pixel coords.
    // Puts ndc x=-1 at the left edge of the apertureWindow and x=+1 at the right edge.
    // Y is centered vertically within the apertureWindow and pixel aspect ratio applied.
    //-----------------------------------------------------------------------------------------------

    void    ndcToPixelTransform (IMATH_NAMESPACE::M44d& m) const;


    //-----------------------------------------------------------------------------------------------
    // Multiply the provided matrix by a transform to project ImageFormat pixel coords to ndc coords.
    // Puts the apertureWindow left edge at ndc x=-1 and the right edge at ndc x=+1.
    // The y-center of the apertureWindow is placed at ndc y=0 and pixel aspect ratio removed.
    //-----------------------------------------------------------------------------------------------

    void    pixelToNdcTransform (IMATH_NAMESPACE::M44d& m) const;


    //------------------------------------------------------
    // Returns a pixel window scaled and optionally padding.
    // Pad amounts are also scaled.
    //------------------------------------------------------

    static IMATH_NAMESPACE::Box2i scaleWindow (const IMATH_NAMESPACE::Box2i& in,
                                               double scale,
                                               int xpad=0, int ypad=0);
    static IMATH_NAMESPACE::Box2i scaleWindow (const IMATH_NAMESPACE::Box2i& in,
                                               double scale,
                                               int xpad, int ypad,
                                               int rpad, int tpad);


  protected:
    //-------------------------------------
    // Copy constructor only for subclasses
    //-------------------------------------

    ImageFormat (const ImageFormat&);

    //----------------------------------------------------------------------------------
    // Any display win corners inside aperture win will be moved to the aperture's edge.
    //----------------------------------------------------------------------------------
    void clampDisplayToAperture ();

    //-----------------------------------------------------------------------------------
    // Any aperture win corners outside display win will be moved to the displays's edge.
    //-----------------------------------------------------------------------------------

    void clampApertureToDisplay ();


  protected:
    std::string             m_name;                 // Format's identifying name - should be kept unique
    //
    IMATH_NAMESPACE::Box2i  m_apertureWindow;       // Camera aperture region, prior to any expansions, in pixels
    IMATH_NAMESPACE::Box2i  m_displayWindow;        // Target working region, always equal to or larger than apertureWindow
    //
    float                   m_pixelAspectRatio;     // Aspect ratio of a pixel (pixel-height / pixel-width)
    //
    std::string             m_formatReference;      // Reference format used to determine proxy scaling


};


// Serializes parameters to an output stream
std::ostream& operator << (std::ostream& os, const ImageFormat& f);



//----------
//!rst:cpp:end::
//----------

//-----------------
// Inline Functions
//-----------------

/*virtual*/ inline ImageFormat* ImageFormat::format () { return this; };
//
inline const IMATH_NAMESPACE::Box2i& ImageFormat::displayWindow () const { return m_displayWindow; }
inline int   ImageFormat::displayMinX () const { return m_displayWindow.min.x; }
inline int   ImageFormat::displayMinY () const { return m_displayWindow.min.y; }
inline int   ImageFormat::displayMaxX () const { return m_displayWindow.max.x; }
inline int   ImageFormat::displayMaxY () const { return m_displayWindow.max.y; }
inline int   ImageFormat::displayWidth () const { return (m_displayWindow.max.x - m_displayWindow.min.x)+1; }
inline int   ImageFormat::displayHeight () const { return (m_displayWindow.max.y - m_displayWindow.min.y)+1; }
inline int   ImageFormat::displayW ()  const { return displayWidth(); }
inline int   ImageFormat::displayH () const { return displayHeight(); }
inline float ImageFormat::displayCenterX () const { return float(m_displayWindow.max.x - m_displayWindow.min.x + 1)/2.0f; }
inline float ImageFormat::displayCenterY () const { return float(m_displayWindow.max.y - m_displayWindow.min.y + 1)/2.0f; }
inline IMATH_NAMESPACE::V2i ImageFormat::displayMin () const { return m_displayWindow.min; }
inline IMATH_NAMESPACE::V2i ImageFormat::displayMax () const { return m_displayWindow.max; }
inline IMATH_NAMESPACE::V2f ImageFormat::displayMin2f () const { return IMATH_NAMESPACE::V2f(float(m_displayWindow.min.x), float(m_displayWindow.min.y)); }
inline IMATH_NAMESPACE::V2f ImageFormat::displayMax2f () const { return IMATH_NAMESPACE::V2f(float(m_displayWindow.max.x), float(m_displayWindow.max.y)); }
inline IMATH_NAMESPACE::V2f ImageFormat::displayCenter2f () const { return IMATH_NAMESPACE::V2f(displayCenterX(), displayCenterY()); }
inline IMATH_NAMESPACE::V3f ImageFormat::displayMin3f () const { return IMATH_NAMESPACE::V3f(float(m_displayWindow.min.x), float(m_displayWindow.min.y), 0.0f); }
inline IMATH_NAMESPACE::V3f ImageFormat::displayMax3f () const { return IMATH_NAMESPACE::V3f(float(m_displayWindow.max.x), float(m_displayWindow.max.y), 0.0f); }
inline IMATH_NAMESPACE::V3f ImageFormat::displayCenter3f () const { return IMATH_NAMESPACE::V3f(displayCenterX(), displayCenterY(), 0.0f); }
//-------------------------------------------------------
inline const IMATH_NAMESPACE::Box2i& ImageFormat::apertureWindow() const { return m_apertureWindow; }
inline int   ImageFormat::apertureMinX () const { return m_apertureWindow.min.x; }
inline int   ImageFormat::apertureMinY () const { return m_apertureWindow.min.y; }
inline int   ImageFormat::apertureMaxX () const { return m_apertureWindow.max.x; }
inline int   ImageFormat::apertureMaxY () const { return m_apertureWindow.max.y; }
inline int   ImageFormat::apertureWidth () const { return (m_apertureWindow.max.x - m_apertureWindow.min.x)+1; }
inline int   ImageFormat::apertureHeight () const { return (m_apertureWindow.max.y - m_apertureWindow.min.y)+1; }
inline int   ImageFormat::apertureW () const { return apertureWidth(); }
inline int   ImageFormat::apertureH () const { return apertureHeight(); }
inline float ImageFormat::apertureCenterX () const { return float(m_apertureWindow.max.x + m_apertureWindow.min.x + 1)/2.0f; }
inline float ImageFormat::apertureCenterY () const { return float(m_apertureWindow.max.y + m_apertureWindow.min.y + 1)/2.0f; }
inline IMATH_NAMESPACE::V2i ImageFormat::apertureMin () const { return m_apertureWindow.min; }
inline IMATH_NAMESPACE::V2i ImageFormat::apertureMax () const { return m_apertureWindow.max; }
inline IMATH_NAMESPACE::V2f ImageFormat::apertureMin2f () const { return IMATH_NAMESPACE::V2f(float(m_apertureWindow.min.x), float(m_apertureWindow.min.y)); }
inline IMATH_NAMESPACE::V2f ImageFormat::apertureMax2f () const { return IMATH_NAMESPACE::V2f(float(m_apertureWindow.max.x), float(m_apertureWindow.max.y)); }
inline IMATH_NAMESPACE::V2f ImageFormat::apertureCenter2f () const { return IMATH_NAMESPACE::V2f(apertureCenterX(), apertureCenterY()); }
inline IMATH_NAMESPACE::V3f ImageFormat::apertureMin3f () const { return IMATH_NAMESPACE::V3f(float(m_apertureWindow.min.x), float(m_apertureWindow.min.y), 0.0f); }
inline IMATH_NAMESPACE::V3f ImageFormat::apertureMax3f () const { return IMATH_NAMESPACE::V3f(float(m_apertureWindow.max.x), float(m_apertureWindow.max.y), 0.0f); }
inline IMATH_NAMESPACE::V3f ImageFormat::apertureCenter3f () const { return IMATH_NAMESPACE::V3f(apertureCenterX(), apertureCenterY(), 0.0f); }
inline bool ImageFormat::apertureMatchesDisplay () const {
    return !(m_apertureWindow.min.x > m_displayWindow.min.x ||
             m_apertureWindow.min.y > m_displayWindow.min.y ||
             m_apertureWindow.max.x < m_displayWindow.max.x ||
             m_apertureWindow.max.y < m_displayWindow.max.y);
}
//-------------------------------------------------------
inline const std::string& ImageFormat::formatReference () const { return m_formatReference; }

//-------------------------------------------------------
inline void ImageFormat::pixelToNdc (float x, float y, float& u, float& v) const
{
    const float sx = float(apertureWidth()) / 2.0f;
    u = (x - apertureCenterX()) * sx;
    v = (y - apertureCenterY()) * (sx / m_pixelAspectRatio);
}
inline IMATH_NAMESPACE::V2f ImageFormat::pixelToNdc (float x, float y) const
{
    IMATH_NAMESPACE::V2f uv;
    pixelToNdc(x, y, uv.x, uv.y);
    return uv;
}
inline IMATH_NAMESPACE::V2f ImageFormat::pixelToNdc (const IMATH_NAMESPACE::V2f& pixelXY) const
{
    IMATH_NAMESPACE::V2f uv;
    pixelToNdc(pixelXY.x, pixelXY.y, uv.x, uv.y);
    return uv;
}
inline void ImageFormat::ndcToPixel (float u, float v, float& x, float& y) const
{
    const float sx = float(apertureWidth()) / 2.0f;
    x = floorf((u * sx) + apertureCenterX() + 0.5f);
    y = floorf((v * sx * m_pixelAspectRatio) + apertureCenterX() + 0.5f);
}
inline IMATH_NAMESPACE::V2f ImageFormat::ndcToPixel (float u, float v) const
{
    IMATH_NAMESPACE::V2f xy;
    ndcToPixel(u, v, xy.x, xy.y);
    return xy;
}
inline IMATH_NAMESPACE::V2f ImageFormat::ndcToPixel (const IMATH_NAMESPACE::V2f& ndcUV) const
{
    IMATH_NAMESPACE::V2f xy;
    ndcToPixel(ndcUV.x, ndcUV.y, xy.x, xy.y);
    return xy;
}
//-------------------------------------------------------
inline float ImageFormat::pixelAspectRatio () const { return m_pixelAspectRatio; }
inline IMATH_NAMESPACE::V2f ImageFormat::screenWindowCenter () const { return pixelToNdc(apertureCenter2f()); }
inline float ImageFormat::screenWindowWidth () const { return (float(displayWidth()) / float(apertureWidth())); }
inline float ImageFormat::screenWindowHeight () const { return (float(displayHeight()) / float(apertureHeight()))/m_pixelAspectRatio; }
/*virtual*/
inline bool ImageFormat::isValid () const { return (displayMaxX() > displayMinX() && displayMaxY() > displayMinY()); }
//-------------------------------------------------------
inline void ImageFormat::ndcToPixelTransform (IMATH_NAMESPACE::M44d& m) const
{
    m.translate(apertureCenter3f());
    const double sx = double(apertureWidth()) / 2.0;
    m.scale(IMATH_NAMESPACE::V3d(sx, sx*m_pixelAspectRatio, 1.0));
}
inline void ImageFormat::pixelToNdcTransform (IMATH_NAMESPACE::M44d& m) const
{
    const double sx = 2.0 / double(apertureWidth());
    m.scale(IMATH_NAMESPACE::V3d(sx, sx/m_pixelAspectRatio, 1.0));
    m.translate(-apertureCenter3f());
}
inline IMATH_NAMESPACE::Box2f ImageFormat::windowToNdc (const IMATH_NAMESPACE::Box2i& in) const
{
    IMATH_NAMESPACE::Box2f out;
    const float sx = float(apertureWidth()) / 2.0f;
    const float sy = sx / m_pixelAspectRatio;
    out.min.x = (float(in.min.x  ) - apertureCenterX())*sx;
    out.min.y = (float(in.min.y  ) - apertureCenterY())*sy;
    out.max.x = (float(in.max.x+1) - apertureCenterX())*sx;
    out.max.y = (float(in.max.y+1) - apertureCenterY())*sy;
    return out;
}
//-------------------------------------------------------
/*static*/
inline IMATH_NAMESPACE::Box2i ImageFormat::scaleWindow (const IMATH_NAMESPACE::Box2i& in, double scale, int xpad, int ypad)
{
    return scaleWindow(in, scale, xpad, ypad, xpad, ypad);
}
inline const std::string& ImageFormat::name () const { return m_name; }
inline std::string ImageFormat::buildScaledName (double scale) const { return appendScaleToName(m_name, scale); }

OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_IMAGEFORMAT_H
