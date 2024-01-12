// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_PIXEL_H
#define INCLUDED_DCX_PIXEL_H

//=============================================================================
//
//  class  Pixel
//
//=============================================================================

#include "DcxChannelSet.h"

#ifdef __ICC
// disable icc remark #1572: 'floating-point equality and inequality comparisons are unreliable'
//   this is coming from Imath/half.h...
#   pragma warning(disable:1572)
#endif
#include <Imath/half.h>  // For Pixelh

#include <string.h> // for memset in some compilers
#include <iostream>


//-------------------------
//!rst:cpp:begin::
//.. _pixel_class:
//
//Pixel
//=====
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


//==================
//
//  class  Pixel
//
//==================
//-------------------------------------------------------------------------------------
//
//  Contains a fixed-size array of data and a ChannelSet
//  defining the active channels.
//
//  Intentionally similar to Nuke's DD::Image::Pixel class.
//
//  NOTE: This class is solely intended for use in image processing algorithms and
//  *not* for multi-pixel data storage (ex. a line or tile's worth.)
//
//  (TODO: is it worth it to change this class to a packed channel array? Likely not
//   as managing the data packing/unpacking may be far more trouble than it's worth)
//
//-------------------------------------------------------------------------------------

template <class T>
class DCX_EXPORT Pixel
{
  public:
    ChannelSet  channels;                   // Set of enabled channels
    T           chan[OPENDCX_INTERNAL_NAMESPACE::Chan_Max+1];      // Fixed-sized array of channel values


  public:

    //-------------------------------------------------
    // Channel mask is empty, leaves junk in channels
    //-------------------------------------------------

    Pixel ();


    //-------------------------------------------------
    // Assigns channel mask but leaves junk in channels
    //-------------------------------------------------

    Pixel (const ChannelSet&);


    //-----------------------------------------------------------------
    // Assigns channel mask and sets enabled channels to a single value
    //-----------------------------------------------------------------

    Pixel (const ChannelSet&, T val);


    //-----------------
    // Copy constructor
    //-----------------

    Pixel (const Pixel&);


    //-----------------------------
    // Set channels to zero (black)
    //-----------------------------

    void    erase ();
    void    erase (const ChannelSet&);
    void    erase (ChannelIdx);


    //---------------------------
    // Replace values in channels
    //---------------------------

    void    replace (const Pixel&);
    void    replace (const Pixel&,
                     const ChannelSet&);


    //-------------------------------------------
    // Pointer to beginning of channel data array
    //-------------------------------------------

    T* array();


    //----------------
    // Channel access
    //----------------

    T& operator [] (ChannelIdx);
    const T& operator [] (ChannelIdx) const;


    //-----------
    // Assignment
    //-----------

    Pixel& operator = (const Pixel&); // copies all floats
    Pixel& operator = (T val);
    void set (T val);
    void set (ChannelIdx,
              T val);
    void set (const ChannelSet&,
              T val);
    void copy (const Pixel&); // same as replace()
    void copy (const Pixel&,
               const ChannelSet&); // same as replace()


    //---------
    // Multiply
    //---------

    Pixel  operator *  (const Pixel&) const;
    Pixel& operator *= (const Pixel&);
    Pixel  operator *  (T val) const;
    Pixel& operator *= (T val);


    //-------
    // Divide
    //-------

    Pixel  operator /  (const Pixel&) const;
    Pixel& operator /= (const Pixel&);
    Pixel  operator /  (T val) const;
    Pixel& operator /= (T val);


    //---------
    // Addition
    //---------

    Pixel  operator +  (const Pixel&) const;
    Pixel& operator += (const Pixel&);
    Pixel  operator +  (T val) const;
    Pixel& operator += (T val);

    //------------
    // Subtraction
    //------------

    Pixel  operator -  (const Pixel&) const;
    Pixel& operator -= (const Pixel&);
    Pixel  operator -  (T val) const;
    Pixel& operator -= (T val);


    //-------------
    // Print values
    //-------------

    void    print (std::ostream&,
                   const char* prefix,
                   int precision=6,
                   const ChannelSet& channels=Mask_All) const;
    template <class S>
    friend  std::ostream& operator << (std::ostream&,
                                       const Pixel<S>&);

};

// Predefined types:
typedef Pixel<int32_t>  Pixeli;
typedef Pixel<uint32_t> Pixelu;  // Imf::UINT
typedef Pixel<half>     Pixelh;  // Imf::HALF
typedef Pixel<float>    Pixelf;  // Imf::FLOAT
typedef Pixel<double>   Pixeld;






//----------
//!rst:cpp:end::
//----------

//-----------------
// Inline Functions
//-----------------

template <class T>
inline Pixel<T>::Pixel () {}
template <class T>
inline Pixel<T>::Pixel (const ChannelSet& set) : channels(set) {}
template <class T>
inline Pixel<T>::Pixel (const ChannelSet& set, T val) : channels(set) { this->set(val); }
template <class T>
inline void Pixel<T>::erase () { memset(chan, 0, sizeof(T)*OPENDCX_INTERNAL_NAMESPACE::Chan_Max); }
template <class T>
inline void Pixel<T>::erase (const ChannelSet& set)
{
    foreach_channel(z, set)
        chan[z] = 0;
}
template <class T>
inline void Pixel<T>::erase (ChannelIdx channel) { chan[channel] = 0; }
//---------------------------------------------------
template <class T>
inline void Pixel<T>::replace (const Pixel<T>& b, const ChannelSet& set)
{
    if (&b != this)
    {
        foreach_channel(z, set)
            chan[z] = b.chan[z];
    }
}
template <class T>
inline void Pixel<T>::replace (const Pixel<T>& b) { replace(b, b.channels); }
template <class T>
inline Pixel<T>& Pixel<T>::operator = (const Pixel<T>& b) {
    if (&b != this)
    {
        channels = b.channels;
        memcpy(chan, b.chan, sizeof(T)*OPENDCX_INTERNAL_NAMESPACE::Chan_Max);
    }
    return *this;
}
template <class T>
inline Pixel<T>::Pixel (const Pixel<T>& b) { *this = b; }
//---------------------------------------------------
template <class T>
inline T& Pixel<T>::operator [] (ChannelIdx channel) { return chan[channel]; }
template <class T>
inline const T& Pixel<T>::operator [] (ChannelIdx channel) const { return chan[channel]; }
template <class T>
inline T* Pixel<T>::array () { return chan; }
//---------------------------------------------------
template <class T>
inline Pixel<T>& Pixel<T>::operator = (T val)
{
    foreach_channel(z, channels)
        chan[z] = val;
    return *this;
}
template <class T>
inline void Pixel<T>::set (ChannelIdx channel, T val) { chan[channel] = val; }
template <class T>
inline void Pixel<T>::set (T val)
{
    foreach_channel(z, channels)
        chan[z] = val;
}
template <class T>
inline void Pixel<T>::set (const ChannelSet& _set, T val)
{
    channels = _set;
    this->set(val);
}
template <class T>
inline void Pixel<T>::copy (const Pixel<T>& b, const ChannelSet& set) { replace(b, set); }
template <class T>
inline void Pixel<T>::copy (const Pixel<T>& b) { replace(b); }
//---------------------------------------------------
template <class T>
inline Pixel<T> Pixel<T>::operator * (T val) const
{
    Pixel<T> ret(channels);
    foreach_channel(z, channels)
        ret.chan[z] = chan[z] * val;
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator *= (T val)
{
    foreach_channel(z, channels)
        chan[z] *= val;
    return *this;
}
template <class T>
inline Pixel<T> Pixel<T>::operator * (const Pixel<T>& b) const
{
    Pixel<T> ret(channels);
    foreach_channel(z, b.channels)
        ret.chan[z] = (chan[z] * b.chan[z]);
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator *= (const Pixel<T>& b)
{
    foreach_channel(z, b.channels)
        chan[z] *= b.chan[z];
    return *this;
}
//---------------------------------------------------
template <class T>
inline Pixel<T> Pixel<T>::operator / (T val) const
{
    const T ival = (T)1 / val;
    Pixel<T> ret(channels);
    foreach_channel(z, channels)
        ret.chan[z] = chan[z]*ival;
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator /= (T val)
{
    const T ival = (T)1 / val;
    foreach_channel(z, channels)
        chan[z] *= ival;
    return *this;
}
template <class T>
inline Pixel<T> Pixel<T>::operator / (const Pixel<T>& b) const
{
    Pixel<T> ret(channels);
    foreach_channel(z, b.channels)
        ret.chan[z] = (chan[z] / b.chan[z]);
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator /= (const Pixel<T>& b)
{
    foreach_channel(z, b.channels)
        chan[z] /= b.chan[z];
    return *this;
}
//---------------------------------------------------
template <class T>
inline Pixel<T> Pixel<T>::operator + (T val) const
{
    Pixel<T>ret(channels);
    foreach_channel(z, channels)
        ret.chan[z] = (chan[z] + val);
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator += (T val)
{
    foreach_channel(z, channels)
        chan[z] += val;
    return *this;
}
template <class T>
inline Pixel<T> Pixel<T>::operator + (const Pixel<T>& b) const
{
    Pixel<T>ret(channels);
    foreach_channel(z, b.channels)
        ret.chan[z] = (chan[z] + b.chan[z]);
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator += (const Pixel<T>& b)
{
    foreach_channel(z, b.channels)
        chan[z] += b.chan[z];
    return *this;
}
//---------------------------------------------------
template <class T>
inline Pixel<T> Pixel<T>::operator - (T val) const
{
    Pixel<T> ret(channels);
    foreach_channel(z, channels)
        ret.chan[z] = (chan[z] - val);
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator -= (T val)
{
    foreach_channel(z, channels)
        chan[z] -= val;
    return *this;
}
template <class T>
inline Pixel<T> Pixel<T>::operator - (const Pixel<T>& b) const
{
    Pixel<T> ret(channels);
    foreach_channel(z, b.channels)
        ret.chan[z] = (chan[z] - b.chan[z]);
    return ret;
}
template <class T>
inline Pixel<T>& Pixel<T>::operator -= (const Pixel<T>& b)
{
    foreach_channel(z, b.channels)
        chan[z] -= b.chan[z];
    return *this;
}
//---------------------------------------------------
template <class T>
inline void Pixel<T>::print (std::ostream& os, const char* prefix, int precision, const ChannelSet& do_channels) const
{
    os << prefix << "[";
    const std::streamsize sprec = std::cout.precision();
    os.precision(precision);
    if (do_channels.all())
    {
        foreach_channel(z, channels)
            std::cout << " " << z << "=" << std::fixed << chan[z];
    }
    else
    {
        foreach_channel(z, do_channels)
            std::cout << " " << z << "=" << std::fixed << chan[z];
    }
    os << " ]";
    std::cout.precision(sprec);
}
template <class T>
/*friend*/
inline std::ostream& operator << (std::ostream& os, const Pixel<T>& pixel)
{
    os << "[";
    const std::streamsize sprec = std::cout.precision();
    os.precision(6);
    foreach_channel(z, pixel.channels)
        std::cout << " " << z << "=" << std::fixed << pixel[z];
    std::cout.precision(sprec);
    os << " ]";
    return os;
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_PIXEL_H
