// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DcxImageFormat.h"

#include <OpenEXR/ImfBoxAttribute.h>
#include <OpenEXR/ImfStringAttribute.h>

#include <stdio.h> // for sscanf, snprintf
#include <stdlib.h> // for strtod

// Uncomment this to get debug info:
//#define DCX_DEBUG_IMAGEFORMAT 1


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


ImageFormat::ImageFormat () :
    m_apertureWindow(IMATH_NAMESPACE::V2i(0, 0), IMATH_NAMESPACE::V2i(-1, -1)),
    m_displayWindow(IMATH_NAMESPACE::V2i(0, 0), IMATH_NAMESPACE::V2i(-1, -1)),
    m_pixelAspectRatio(1.0)
{
    //
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") empty "; printInfo(std::cout); std::cout << std::endl;
#endif
}

ImageFormat::ImageFormat (const ImageFormat& b) :
    m_name(b.m_name),
    m_apertureWindow(b.m_apertureWindow),
    m_displayWindow(b.m_displayWindow),
    m_pixelAspectRatio(b.m_pixelAspectRatio),
    m_formatReference(b.m_formatReference)
{
    //
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") copy "; printInfo(std::cout); std::cout << std::endl;
    printInfo(std::cout);
#endif
}


//
// Build from a header
//

ImageFormat::ImageFormat (const OPENEXR_IMF_NAMESPACE::Header& header) :
    m_apertureWindow(IMATH_NAMESPACE::V2i(0, 0), IMATH_NAMESPACE::V2i(-1, -1)),
    m_displayWindow(IMATH_NAMESPACE::V2i(0, 0), IMATH_NAMESPACE::V2i(-1, -1)),
    m_pixelAspectRatio(1.0)
{
    fromHeader(header);
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") fromHeader "; printInfo(std::cout); std::cout << std::endl;
#endif
}

//
// Assumes displayWindow == apertureWindow
//

ImageFormat::ImageFormat (const std::string& name,
                          const IMATH_NAMESPACE::Box2i& disp,
                          float pixel_aspect) :
    m_name(name),
    m_displayWindow(disp),
    m_pixelAspectRatio(pixel_aspect)
{
    m_apertureWindow = m_displayWindow;
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") name/disp/pa "; printInfo(std::cout); std::cout << std::endl;
#endif
}

//
// Assumes displayWindow == apertureWindow
//

ImageFormat::ImageFormat (const std::string& name,
                          int disp[4],
                          float pixel_aspect) :
    m_name(name),
    m_displayWindow(IMATH_NAMESPACE::V2i(disp[0], disp[1]), IMATH_NAMESPACE::V2i(disp[2], disp[3])),
    m_pixelAspectRatio(pixel_aspect)
{
    m_apertureWindow = m_displayWindow;
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") name/disp/pa "; printInfo(std::cout); std::cout << std::endl;
#endif
}

//
// Display must be >= aperture so any corners inside aperture will be clamped to its edges.
//

ImageFormat::ImageFormat (const std::string& name,
                          const IMATH_NAMESPACE::Box2i& disp,
                          const IMATH_NAMESPACE::Box2i& aper,
                          float pixel_aspect) :
    m_name(name),
    m_apertureWindow(aper),
    m_displayWindow(disp),
    m_pixelAspectRatio(pixel_aspect)
{
    // Make sure display window is legal:
    clampDisplayToAperture();
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") name/disp/aper/pa "; printInfo(std::cout); std::cout << std::endl;
#endif
}

//
// Display must be >= aperture so any corners inside aperture will be clamped to its edges.
//

ImageFormat::ImageFormat (const std::string& name,
                          int disp_x, int disp_y, int disp_r, int disp_t,
                          int aper_x, int aper_y, int aper_r, int aper_t,
                          float pixel_aspect) :
    m_name(name),
    m_apertureWindow(IMATH_NAMESPACE::V2i(aper_x, aper_y), IMATH_NAMESPACE::V2i(aper_r, aper_t)),
    m_displayWindow(IMATH_NAMESPACE::V2i(disp_x, disp_y), IMATH_NAMESPACE::V2i(disp_r, disp_t)),
    m_pixelAspectRatio(pixel_aspect)
{
    // Make sure display window is legal:
    clampDisplayToAperture();
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") name/disp/pa "; printInfo(std::cout); std::cout << std::endl;
#endif
}

//
// Display must be >= aperture so any corners inside aperture will be clamped to its edges.
//

ImageFormat::ImageFormat (const std::string& name,
                          int disp[4],
                          int aper[4],
                          float pixel_aspect) :
    m_name(name),
    m_apertureWindow(IMATH_NAMESPACE::V2i(aper[0], aper[1]), IMATH_NAMESPACE::V2i(aper[2], aper[3])),
    m_displayWindow(IMATH_NAMESPACE::V2i(disp[0], disp[1]), IMATH_NAMESPACE::V2i(disp[2], disp[3])),
    m_pixelAspectRatio(pixel_aspect)
{
    // Make sure display window is legal:
    clampDisplayToAperture();
#ifdef DCX_DEBUG_IMAGEFORMAT
    std::cout << "ImageFormat::ctor(" << this << ") name/disp/pa "; printInfo(std::cout); std::cout << std::endl;
#endif
}

/*virtual*/
ImageFormat::~ImageFormat ()
{
    //
}


//
// Any displayWindow corners inside apertureWindow will be moved to the aperture's edge.
//

void
ImageFormat::clampDisplayToAperture ()
{
    m_displayWindow.min.x = std::min(m_apertureWindow.min.x, m_displayWindow.min.x);
    m_displayWindow.min.y = std::min(m_apertureWindow.min.y, m_displayWindow.min.y);
    m_displayWindow.max.x = std::max(m_displayWindow.max.x, m_apertureWindow.max.x);
    m_displayWindow.max.y = std::max(m_displayWindow.max.y, m_apertureWindow.max.y);
}

//
// Any aperture win corners outside display win will be moved to the displays's edge.
//

void
ImageFormat::clampApertureToDisplay ()
{
    m_apertureWindow.min.x = std::max(m_displayWindow.min.x, m_apertureWindow.min.x);
    m_apertureWindow.min.y = std::max(m_displayWindow.min.y, m_apertureWindow.min.y);
    m_apertureWindow.max.x = std::min(m_apertureWindow.max.x, m_displayWindow.max.x);
    m_apertureWindow.max.y = std::min(m_apertureWindow.max.y, m_displayWindow.max.y);
}

//-----------------------------------------------------------------------------

//
// Serialize the format params to a string.  If brackets is true wrap it in '[]'
//

/*virtual*/
std::string
ImageFormat::toString (bool brackets) const
{
    std::stringstream ss;
    toString(ss, brackets);
    return ss.str();
}

//
// Serialize the format params to a string.  If brackets is true wrap it in '[]'
//    Outputs in this formatting:
//       '<name> [<disp x y r t>] [<aper x y r t>] <pa>'
//    or this is aperture == display:
//       '<name> [<disp x y r t>] <pa>'
//    ex. 'hd_2.35_hvp24 [-24 0 1967 815] [0 0 1919 815] 1.0'
//

/*virtual*/
void
ImageFormat::toString (std::ostream& os,
                       bool brackets) const
{
    if (brackets)
        os << "[";
    os << m_name;
    os << " [" << m_displayWindow.min.x;
    os << " " << m_displayWindow.min.y;
    os << " " << m_displayWindow.max.x;
    os << " " << m_displayWindow.max.y << "]";
    if (!apertureMatchesDisplay())
    {
        os << " [" << m_apertureWindow.min.x;
        os << " " << m_apertureWindow.min.y;
        os << " " << m_apertureWindow.max.x;
        os << " " << m_apertureWindow.max.y << "]";
    }
    os << " " << m_pixelAspectRatio;
    if (brackets)
        os << "]";
}

/*virtual*/
void
ImageFormat::printInfo (std::ostream& os) const
{
    os << "name '" << m_name << "'";
    os << ", disp [" << m_displayWindow.min << " " << m_displayWindow.max << "]";
    os << ", aper [" << m_apertureWindow.min << " " << m_apertureWindow.max << "]";
    os << ", ref ['" << m_formatReference << "']";
    os << ", pa " << m_pixelAspectRatio;
}

//-----------------------------------------------------------------------------

//
// Scale a pixel rectangle with rounding and scaled padding.
//

/*static*/
IMATH_NAMESPACE::Box2i
ImageFormat::scaleWindow (const IMATH_NAMESPACE::Box2i& in,
                          double scale,
                          int xpad, int ypad,
                          int rpad, int tpad)
{
    IMATH_NAMESPACE::Box2i out;

    if (scale < EPSILONd || isinf(scale))
    {
        // Ignore illegal scale values but still allow padding:
        scale = 1.0;
    }
    else if (scale > 1.0)
    {
        out.min.x = int(floor(in.min.x / scale));
        out.min.y = int(floor(in.min.y / scale));
        out.max.x = int(floor(((in.max.x + 1)*scale) - 1));
        out.max.y = int(floor(((in.max.y + 1)*scale) - 1));
    }
    else
    {
        out.min.x = int(in.min.x / scale);
        out.min.y = int(in.min.y / scale);
        out.max.x = int(in.min.x + ((in.max.x - in.min.x + 1)*scale) - 1);
        out.max.y = int(in.min.y + ((in.max.y - in.min.y + 1)*scale) - 1);
    }

    // Pad by the scaled expansion values:
    xpad = int(double(xpad)*scale);
    ypad = int(double(ypad)*scale);
    rpad = int(double(rpad)*scale);
    tpad = int(double(tpad)*scale);
    out.min.x -= xpad;
    out.min.y -= ypad;
    out.max.x += rpad;
    out.max.y += tpad;

    return out;
}


/*virtual*/
void
ImageFormat::fromHeader (const OPENEXR_IMF_NAMESPACE::Header& header)
{
    m_name             = "";
    m_displayWindow    = header.displayWindow();
    m_apertureWindow   = header.displayWindow();
    m_pixelAspectRatio = header.pixelAspectRatio();
    m_formatReference  = "";

    OPENEXR_IMF_NAMESPACE::Header::ConstIterator it;

    // formatName attribute
    it = header.find(FORMATNAME_ATTRIBUTE);
    if (it != header.end() && strcmp(it.attribute().typeName(), "string")==0)
        m_name = (static_cast<const OPENEXR_IMF_NAMESPACE::StringAttribute*>(&it.attribute()))->value();

    // apertureWindow attribute
    it = header.find(APERTUREWINDOW_ATTRIBUTE);
    if (it != header.end() && strcmp(it.attribute().typeName(), "box2i")==0)
    {
        m_apertureWindow = (static_cast<const OPENEXR_IMF_NAMESPACE::Box2iAttribute*>(&it.attribute()))->value();
        clampDisplayToAperture();
    }

    // formatReference attribute
    it = header.find(FORMATREFERENCE_ATTRIBUTE);
    if (it != header.end() && strcmp(it.attribute().typeName(), "string")==0)
        m_formatReference = (static_cast<const OPENEXR_IMF_NAMESPACE::StringAttribute*>(&it.attribute()))->value();

}


/*virtual*/
void
ImageFormat::toHeader (OPENEXR_IMF_NAMESPACE::Header& header) const
{
    // Set standard header parameters:
    header.displayWindow()      = displayWindow();
    header.pixelAspectRatio()   = (float)pixelAspectRatio();
    header.screenWindowCenter() = screenWindowCenter();
    header.screenWindowWidth()  = screenWindowWidth();

    // Add custom ImageFormat attributes:
    if (m_apertureWindow.min.x > m_displayWindow.min.x || m_apertureWindow.min.y > m_displayWindow.min.y ||
        m_apertureWindow.max.x < m_displayWindow.max.x || m_apertureWindow.max.y < m_displayWindow.max.y)
        header.insert(APERTUREWINDOW_ATTRIBUTE, OPENEXR_IMF_NAMESPACE::Box2iAttribute(m_apertureWindow));
    if (!m_name.empty())
        header.insert(FORMATNAME_ATTRIBUTE, OPENEXR_IMF_NAMESPACE::StringAttribute(m_name));
    if (!m_formatReference.empty())
        header.insert(FORMATREFERENCE_ATTRIBUTE, OPENEXR_IMF_NAMESPACE::StringAttribute(m_formatReference));

}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//
//
//

/*static*/
std::string ImageFormat::toScaleNotation (double scale)
{
    char buf[32];
    if (scale > (1.0 + EPSILONd))
        snprintf(buf, 32, "s%g", scale);
    else if (scale < (1.0 - EPSILONd))
        snprintf(buf, 32, "s1/%g", (1.0 / scale));
    else
        return std::string("");
    return std::string(buf);
}


//
// Assumes input string is of the form 's###' or 's1/###'.
//

/*static*/
double ImageFormat::fromScaleNotation (const std::string& s)
{
    if (s[0] != 's')
        return 1.0;
    else if (s.size() > 3 && s[1] == '1' && s[2] == '/') // 1/scale:
    {
        char* e;
        double scale = strtod(&s[3], &e);
        if (e > &s[3])
            return 1.0/scale;
    }
    else if (s.size() > 1)
    {
        char* e;
        double scale = strtod(&s[1], &e);
        if (e > &s[1])
            return scale;
    }
    return 1.0;
}


//
// Builds a new format name from the provided name & scale value by appending
// '(s##)' to the end of the format name.  For example a name of 'hd_2.35_hvp24'
// with a scale of 0.5 will return 'hd_2.35_hvp24(s1/2)' and a scale of 2.0 will
// return 'hd_2.35_hvp24(s2)'.
//
// If there's an existing scale value on the end of the name it will be replaced.
//

/*static*/
std::string
ImageFormat::appendScaleToName (const std::string& name,
                                double scale)
{
    double s = 1.0;
    std::string n = stripScaleFromName(name, &s);
    if (n.empty() || fabs(s - 1.0) < EPSILONd)
        return name;
    n += "("; n += toScaleNotation(scale); n += ")";
    return n;
}


//
// Strips any scale notation off the end of the format name.
//

/*static*/
std::string
ImageFormat::stripScaleFromName (const std::string& name,
                                 double* scale_val)
{
    if (scale_val)
        *scale_val = 1.0;
    if (name.empty())
        return name;

    // Check for notations using parentheses 'foo(s###)' or a
    // leading-underscore 'foo_s####':
#if 1
    size_t a = name.find_last_of("(");
    if (a != std::string::npos && name[a+1] == 's' && isdigit(name[a+2]))
    {
        size_t b = name.find_first_of(")", a+2);
        if (b != std::string::npos)
        {
            if (scale_val)
                *scale_val = fromScaleNotation(name.substr(a+1, b-a-2));
            return name.substr(0, a);
        }
    }
    else
    {
        // If no parentheses also check for leading-underscore suffix:
        a = name.find_last_of("_");
        if (a != std::string::npos && name[a+1] == 's' && isdigit(name[a+2]))
        {
            if (scale_val)
                *scale_val = fromScaleNotation(name.substr(a+1));
            return name.substr(0, a);
        }
    }
    return name;
#else
    std::string n(name);
    size_t a = name.find_last_of("(");
    if (a != std::string::npos && n[a+1] == 's')
    {
        size_t b = n.find_first_of(")", a+2);
        if (b != std::string::npos)
            // Try to convert into a number:
            char* endptr;
            double s = strtod(&n[a+2], &endptr);
            if (endptr > &n[a+2])
            {
                n[a] = 0; // trim off '(s##)'
                if (scale_val)
                    *scale_val = (n[a+1] == 'p')?(1.0 / s):s;
            }
        }
    }


    else
    {
        // If no parantheses also check for leading-underscore suffix:
        a = n.find_last_of("_");
        if (a != std::string::npos && n[a+1] == 's')
        {
            // Try to convert into a number:
            char* endptr;
            double s = strtod(&n[a+2], &endptr);
            if (endptr > &n[a+2])
            {
                n[a] = 0; // trim off '_s##':
                if (scale_val)
                    *scale_val = (n[a+1] == 'p')?(1.0 / s):s;
            }
        }
    }
    return n;
#endif
}


/*static*/
double
ImageFormat::getScaleFromName (const std::string& name_with_scale)
{
    double s;
    stripScaleFromName(name_with_scale, &s);
    return s;
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT
