// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_CHANNELSET_H
#define INCLUDED_DCX_CHANNELSET_H

//=============================================================================
//
//  typedef  ChannelIdx
//  typedef  ChannelIdxSet
//
//  class    ChannelSet
//
//=============================================================================

#include "DcxAPI.h"

#include <OpenEXR/ImfPixelType.h>

#include <stdint.h> // for uint32_t
#include <vector>
#include <bitset>
#include <iostream>


//-------------------------
//!rst:cpp:begin::
//ChannelSet
//==========
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER

//---------------------
//!rst:left-align::
//.. _channelidx_type:
//
//ChannelIdx
//**********
//---------------------
//----------------------------------------
// ``uint_32_t``
// Global channel index type
//----------------------------------------
typedef uint32_t ChannelIdx;


//------------------------------------------
// Reserved value for a non-existant channel
//------------------------------------------
static GNU_CONST_DECL ChannelIdx  Chan_Invalid = 0;

//----------------------------------------------------
// Reserved value for the first channel index allowed.
//----------------------------------------------------
static GNU_CONST_DECL ChannelIdx  Chan_First = 1;

//-----------------------------------------------------------------
// Reserved value for the maximum channel index allowed.
// (TODO: remove this in favor of varying-sized packed arrays...?)
//-----------------------------------------------------------------
static GNU_CONST_DECL ChannelIdx  Chan_Max = 1024;

//-------------------------------------------------------------------
// Reserved value. If m_last in a ChannelSet is set to this it
// indicates a set with all channels enabled (i.e. Mask_All)
//-------------------------------------------------------------------
static GNU_CONST_DECL ChannelIdx  Chan_All = 0xffffffff;


typedef std::bitset<Chan_Max> ChannelIdxSet;


class ChannelContext;

//---------------------
//!rst:left-align::
//.. _channelset_class:
//
//ChannelSet
//**********
//---------------------


//======================
//
//  class ChannelSet
//
//======================
//---------------------------------------------------------------------------
//
//  A std::set wrapper class that acts similar to DD::Image::ChannelSet.
//
//  The set contains ChannelIdx indices.  Use the foreach_channel()
//  macro to iterate through the set (similar to Nuke's
//  DD::Image foreach() macro.)
//
//---------------------------------------------------------------------------

class DCX_EXPORT ChannelSet
{
  public:

    //-----------------------------------------
    // Default constructor creates an empty set
    //-----------------------------------------

    ChannelSet ();

    //-----------------
    // Copy constructor
    //-----------------

    ChannelSet (const ChannelSet&);

    //-----------------------------------------------------
    // Constructor to initialize from a set of ChannelIdx's
    // (TODO: change to a C++11 initializer list)
    //-----------------------------------------------------
    ChannelSet (ChannelIdx a,
                ChannelIdx b=Chan_Invalid, ChannelIdx c=Chan_Invalid,
                ChannelIdx d=Chan_Invalid, ChannelIdx e=Chan_Invalid);


    //---------------
    // Copy operators
    //---------------

    ChannelSet& operator = (const ChannelSet&);
    ChannelSet& operator = (ChannelIdx);


    //---------------------------------
    // Remove all channels from the set
    //---------------------------------

    void    clear ();


    //--------------------------------------
    // Number of enabled channels in the set
    //--------------------------------------

    size_t  size () const;


    //-------------------------------
    // Is set empty?
    //-------------------------------

    bool    empty () const;


    //------------------------------------------------------
    // Mask applies to all assigned ChannelIdxs.
    // This tests for a special case where the set contains
    // the Chan_All ChannelIdx (other channels are ignored.)
    //------------------------------------------------------

    bool    all () const;
    void    setAll();

    //-------------------------------------------------------------------
    // Read/write access to the wrapped std::bitset - use with caution!
    // Clearing bits without updating the m_last var will break the
    // foreach_channel() loops, so call update() after any change.
    //-------------------------------------------------------------------

    ChannelIdxSet&  mask ();
    void            update ();


    //----------------------------
    // ChannelSet::iterator access
    //----------------------------

    ChannelIdx  first () const;
    ChannelIdx  last () const;
    ChannelIdx  prev (ChannelIdx z) const;
    ChannelIdx  next (ChannelIdx z) const;


    //--------------------------------------------------
    // Return true if the mask includes the ChannelIdx's
    //--------------------------------------------------

    bool    contains (const ChannelSet&) const;
    bool    contains (ChannelIdx) const;


    //-------------------
    // Equality operators
    //-------------------

    bool operator == (const ChannelSet&) const;
    bool operator != (const ChannelSet&) const;


    //--------------------------------------------------------------------
    // Add a ChannelIdx or ChannelSet to the mask.
    // There will only be one instance of each ChannelIdx value in the set
    //--------------------------------------------------------------------

    void    insert (const ChannelSet&);
    void    insert (ChannelIdx);
    void    operator += (const ChannelSet&);
    void    operator += (ChannelIdx);


    //------------------------------------------------
    // Remove a ChannelIdx or ChannelSet from the mask
    //------------------------------------------------
    void    erase (const ChannelSet&);
    void    erase (ChannelIdx);
    void    operator -= (const ChannelSet&);
    void    operator -= (ChannelIdx);


    //-----------------------------
    // Bitwise operators on the set
    //-----------------------------
    ChannelSet operator | (const ChannelSet&);
    ChannelSet operator & (const ChannelSet&);


    //---------------------------------------------------
    // Intersect a ChannelIdx or ChannelSet with the mask
    //---------------------------------------------------
    void    intersect (const ChannelSet&);
    void    intersect (ChannelIdx);
    void    operator &= (const ChannelSet&);
    void    operator &= (ChannelIdx);


    //--------------------------------------------------------------------
    // Print info about the set to an output stream.
    // If the ChannelContext is NULL only the ChannelIdx number will be
    // printed, otherwise the channel names will.
    //--------------------------------------------------------------------
    void print (const char* prefix,
                std::ostream&,
                const ChannelContext* ctx=0) const;

    //------------------------------------------------------
    // Outputs the ChannelIdx of the channels to the stream.
    //------------------------------------------------------

    friend std::ostream& operator << (std::ostream&,
                                      const ChannelSet&);

  protected:

    ChannelIdxSet   m_mask;     // Set of channel bits, one per ChannelIdx from 1..Chan_Max
    ChannelIdx      m_last;     // Highest ChannelIdx added to set, to reduce iteration loop cost. If Chan_All all-mode in on.

};



//
//  Convenience macro for iterating through a ChannelSet.
//
//      This is similar to Nuke's DD::Image::ChannelSet foreach() macro.
//      (must have a different name to avoid clashes when building Nuke plugins)
//
//      ex.
//          ChannelSet my_channels(Mask_RGBA);
//          Pixel<float> my_pixel(my_channels);
//          my_pixel.erase();
//          foreach_channel(z, my_channels)
//          {
//              my_pixel[z] += 1.0f;
//          }
//

#undef  foreach_channel
#define foreach_channel(CHAN, CHANNELS) \
    for (OPENDCX_INTERNAL_NAMESPACE::ChannelIdx CHAN=CHANNELS.first(); \
            CHAN != OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid; CHAN = CHANNELS.next(CHAN))


//
// ChannelSet bitwise operators
//

ChannelSet operator | (const ChannelSet&,
                       const ChannelSet&);
ChannelSet operator & (const ChannelSet&,
                       const ChannelSet&);




//--------------
//!rst:cpp:end::
//--------------

//-----------------
// Inline Functions
//-----------------


//------------------------------------------------------------------------------------------
// It's likely slow to call bitset::all(), so like Nuke's ChannelSet we use m_last == Chan_All to indicate all-mode.
inline bool   ChannelSet::all () const { return (m_last == Chan_All); }
inline void   ChannelSet::setAll()
{
    m_mask.reset();
    m_last = Chan_All;
}
inline ChannelIdxSet& ChannelSet::mask () { return m_mask; }
//------------------------------------------------------------------------------------------
inline ChannelIdx ChannelSet::prev (ChannelIdx channel) const
{
    if (channel > Chan_First)
    {
        --channel;
        if (this->all())
            return channel;
        for (; channel > Chan_Invalid; --channel)
            if (m_mask[channel])
                return channel;
    }
    return Chan_Invalid; // off end
}
inline ChannelIdx ChannelSet::next (ChannelIdx channel) const
{
    if (channel < Chan_Max)
    {
        ++channel;
        if (this->all())
            return channel;
        for (; channel <= m_last; ++channel)
            if (m_mask[channel])
                return channel;
    }
    return Chan_Invalid; // off end
}
inline ChannelIdx ChannelSet::first () const { return next(Chan_Invalid); }
inline ChannelIdx ChannelSet::last ()  const { return prev(m_last+1); }
inline void       ChannelSet::update () { m_last = this->prev(Chan_Max+1); }
//------------------------------------------------------------------------------------------
inline void ChannelSet::insert (ChannelIdx channel)
{
    if (channel == Chan_All)
    {
        this->setAll();
        return;
    }
    if (channel < Chan_First || channel > Chan_Max)
        return;
    m_mask.set(channel);
    if (channel > m_last)
        m_last = channel;
}
//------------------------------------------------------------------------------------------
inline ChannelSet::ChannelSet () : m_mask(), m_last(0) {}
inline ChannelSet::ChannelSet (const ChannelSet& b) : m_mask(b.m_mask), m_last(b.m_last) {}
inline ChannelSet::ChannelSet (ChannelIdx a, ChannelIdx b, ChannelIdx c, ChannelIdx d, ChannelIdx e)
{
    this->insert(a); this->insert(b); this->insert(c); this->insert(d); this->insert(e);
}
//------------------------------------------------------------------------------------------
inline void   ChannelSet::clear () { m_mask.reset(); m_last = 0; }
inline size_t ChannelSet::size () const
{
    if (this->all())
        return Chan_Max;
    size_t n = 0;
    const ChannelSet& set = *this;
    foreach_channel(z, set)
        ++n;
    return n;
}
inline bool   ChannelSet::empty () const { return (m_last == 0); }
//------------------------------------------------------------------------------------------
inline ChannelSet& ChannelSet::operator = (const ChannelSet& b)
{
    m_mask = b.m_mask;
    m_last = b.m_last;
    return *this;
}
inline ChannelSet& ChannelSet::operator = (ChannelIdx channel)
{
    this->clear();
    this->insert(channel);
    return *this;
}
//------------------------------------------------------------------------------------------
inline bool   ChannelSet::contains (ChannelIdx channel) const
{
    if (this->all())
        return true;
    if (channel < Chan_First || channel > m_last)
        return false;
    return m_mask[channel];
}
inline bool   ChannelSet::contains (const ChannelSet& b) const
{
    if (this->all())
        return true;
    foreach_channel(z, b)
        if (!m_mask[z])
            return false; // one of the b channels is not in this set
    return true;
}
//------------------------------------------------------------------------------------------
inline void   ChannelSet::insert (const ChannelSet& b)
{
    if (b.all())
    {
        this->setAll();
        return;
    }
    m_mask |= b.m_mask;
    if (b.m_last > m_last)
        m_last = b.m_last;
}
inline void   ChannelSet::operator += (const ChannelSet& b) { this->insert(b); }
inline void   ChannelSet::operator += (ChannelIdx channel) { this->insert(channel); }
//------------------------------------------------------------------------------------------
inline void   ChannelSet::erase (ChannelIdx channel)
{
    if (channel == Chan_All)
    {
        this->clear();
        return;
    }
    if (channel < Chan_First || channel > Chan_Max)
        return;
    m_mask.reset(channel);
    if (m_last == channel)
        m_last = this->prev(channel);
}
inline void   ChannelSet::erase (const ChannelSet& b)
{
    if (b.all())
    {
        this->clear();
        return;
    }
    foreach_channel(z, b)
        m_mask.reset(z);
    m_last = this->last();
}
inline void   ChannelSet::operator -= (const ChannelSet& b) { this->erase(b); }
inline void   ChannelSet::operator -= (ChannelIdx channel) { this->erase(channel); }
//
inline bool   ChannelSet::operator == (const ChannelSet& b) const { return (m_mask == b.m_mask); }
inline bool   ChannelSet::operator != (const ChannelSet& b) const { return (m_mask != b.m_mask); }
//
inline void   ChannelSet::intersect (const ChannelSet& b)
{
    if (b.all())
        return;
    m_mask &= b.m_mask;
    m_last = this->last();
}
inline void   ChannelSet::intersect (ChannelIdx channel)
{
    if (channel == Chan_All)
        return;
    if (channel < Chan_First || channel > Chan_Max)
        return;
    if (m_mask[channel])
    {
        m_mask.reset();
        m_mask.set(channel);
        m_last = channel;
    }
    else
        this->clear();
}
inline void   ChannelSet::operator &= (const ChannelSet& b) { this->intersect(b); }
inline void   ChannelSet::operator &= (ChannelIdx channel) { this->intersect(channel); }
//
inline ChannelSet ChannelSet::operator | (const ChannelSet& b) { this->insert(b); return *this; }
inline ChannelSet ChannelSet::operator & (const ChannelSet& b) { this->intersect(b); return *this; }
//------------------------------------------------------------------------------------------
inline ChannelSet operator | (const ChannelSet& a, const ChannelSet& b)
{
    ChannelSet c(a);
    c.insert(b);
    return c;
}
inline ChannelSet operator & (const ChannelSet& a, const ChannelSet& b)
{
    ChannelSet c(a);
    c.intersect(b);
    return c;
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_CHANNELSET_H
