// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_CHANNELCONTEXT_H
#define INCLUDED_DCX_CHANNELCONTEXT_H

//=============================================================================
//
//  class    ChannelContext
//
//=============================================================================

#include "DcxChannelAlias.h"


//-------------------------
//!rst:cpp:begin::
//.. _channelcontext_class:
//
//ChannelContext
//==============
//-------------------------


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER

//-------------------------------------------------------------------
//
// Split the name into separate layer & chan strings, if possible,
// returning true if successful.
// This splits at the last period ('.') in the string. If the
// name contains more than one period then those become part of the
// layer string.
//
//-------------------------------------------------------------------

DCX_EXPORT
void    splitName (const char* name,
                   std::string& layer,
                   std::string& chan);


//-------------------------------------------------------------------
//
// Returns the best matching standard ChannelIdx and layer name for
// the channel name (with no layer prefix,) or false if no match.
// If there's a match it also sets the channel's default name and
// PixelType for file I/O.
//
//-------------------------------------------------------------------

DCX_EXPORT
bool    matchStandardChannel (const char* channel_name,
                              std::string&    std_layer_name,
                              std::string&    std_chan_name,
                              ChannelIdx&     std_channel,
                              std::string&    std_io_name,
                              OPENEXR_IMF_NAMESPACE::PixelType& std_io_type);


//--------------------------------------------------------------
//
// If the kind of channel is one of the predefined ones, return
// the common position that channel occupies in a layer.
// i.e::
//
//     if kind==Chan_R -> rgba position 0
//     if kind==Chan_A -> rgba position 3
//
//--------------------------------------------------------------

DCX_EXPORT
int     getLayerPositionFromKind (ChannelIdx kind);


//============================
//
//  class ChannelContext
//
//============================
//---------------------------------------------------------------------------
//
//  Context structure for storing the global ChannelIdx assignment slots
//  and maps for quick access to/from ChannelAliases and ChannelIdxs.
//
//  A :ref:`channelalias_class` contains layer and channel strings which map
//  to unique :ref:`channelidx_type` indice.
//
//---------------------------------------------------------------------------

class DCX_EXPORT ChannelContext
{
  public:

    struct ChanOrder
    {
        ChannelIdx  channel;    // ChannelIdx
        uint32_t    order;      // Order in layer (TODO: not implemented - required anymore?)
    };

    struct Layer
    {
        std::string             name;
        std::vector<ChanOrder>  channels;
    };


    typedef std::map<std::string, int>      AliasNameToListMap;
    typedef std::map<ChannelIdx,  int>      ChannelIdxToListMap;
    typedef std::map<std::string, int>      LayerNameToListMap;


  public:

    //--------------------------------------------------------------
    //
    // Default ctor adds the predefined standard ChannelAliases
    // to the context.
    // Pass in false if a subclass of ChannelContext wants to define
    // a custom ChannelAlias type and manage the predefined set
    // itself.
    //
    //--------------------------------------------------------------

    ChannelContext(bool addStandardChans=true);

    virtual ~ChannelContext();


    //----------------------------------------------------
    // Get the ChannelSet for all channels in the context.
    //----------------------------------------------------

    ChannelSet getChannels ();


    //---------------------------------------------------------------
    // Get the ChannelSet from a ChannelAliasList or ChannelAliasSet.
    //---------------------------------------------------------------

    ChannelSet getChannelSetFromAliases (const ChannelAliasPtrList&);
    ChannelSet getChannelSetFromAliases (const ChannelAliasPtrSet&);


    //---------------------------------------------------------------
    //
    // Returns the last assigned ChannelIdx.
    //
    // This value can be used to size channel arrays as it represents
    // the current maximum channel count for this context.
    // Note that this value does not represent *active* channels,
    // only channel definitions assigned to ChannelIdx slots, whereas
    // a ChannelSet is used to define the set of active channels.
    //
    // If no arbitrary channels have been added to this context the
    // value will be Chan_ArbitraryStart-1.
    //
    //---------------------------------------------------------------

    ChannelIdx  lastAssignedChannel () const;


    //---------------------------------------------------------------------
    //
    // Get and/or create a channel, returning a ChannelIdx or ChannelAlias.
    // Returns Chan_Invalid or NULL if there was an error creating it.
    //
    // Note that multiple channel names can map to the same ChannelIdx but
    // each name will have a unique ChannelAlias.
    // ex. 'R', 'rgba.R', 'rgb.red' will all be mapped to Chan_R with each
    // getting a unique ChannelAlias.
    //
    // Unrecognized channel names like 'mylayer.foo' and 'mylayer.F' will
    // each be assigned unique ChannelIdxs unless the duplicate
    // ChannelAliases were added using addChannelAlias() with the same
    // ChannelIdx from the first created alias.
    //
    //---------------------------------------------------------------------

    ChannelIdx      getChannel (const char* name);
    ChannelIdx      getChannel (const std::string& name);

    ChannelAlias*   getChannelAlias (const char* name);
    ChannelAlias*   getChannelAlias (const std::string& name);


    //---------------------------------------------------------------------
    //
    // Get channel or layer.channel name from a ChannelIdx.
    // Returns 'unknown' if the ChannelIdx doesn't exist (is out of range.)
    //
    //---------------------------------------------------------------------

    const char*     getChannelName (ChannelIdx channel) const;
    std::string     getChannelFullName (ChannelIdx channel) const;


    //----------------------------------------------------------------------
    //
    // Find channel by name or ChannelIdx and return a ChannelAlias pointer,
    // or NULL if not found.
    //
    //----------------------------------------------------------------------

    ChannelAlias*   findChannelAlias (const char* name) const;
    ChannelAlias*   findChannelAlias (const std::string& name) const;
    ChannelAlias*   findChannelAlias (ChannelIdx channel) const;


    //--------------------------------------------------------------------------------
    //
    // Add the predefined standard channels to the context.
    //
    //--------------------------------------------------------------------------------

    void    addStandardChannels();


    //--------------------------------------------------------------------------------
    //
    // Add a new ChannelAlias to the context, either by passing in a pre-allocated
    // ChannelAlias or having the context construct it.
    // In both cases the ChannelContext takes ownership of the pointer and
    // deletes the pointers in the destructor.
    //
    // If the assigned channel is Chan_Invalid then no specfic channel slot is
    // being requested so the next available ChannelIdx is assigned, incrementing
    // lastAssignedChannel().
    //
    // * chan_name   user-facing name, not including the layer ('R', 'G', 'red',
    //               'green', 'alpha', 'Z', 'ZBack', etc)
    // * layer_name  user-facing layer name ('rgba', 'beauty', 'beauty.diffuse')
    // * channel     absolute ChannelIdx - if Chan_Invalid a new ChannelIdx is assigned
    // * position    sorted position index within layer (i.e. 0,1,2)
    // * io_name     name to use for exr file I/O ('R' or 'AR'  vs. 'rgba.red' or
    //               'opacity.R')
    // * io_type     PixelType to use for file I/O
    //
    //--------------------------------------------------------------------------------

    ChannelAlias*   addChannelAlias (ChannelAlias* alias);

    ChannelAlias*   addChannelAlias (const std::string&                 chan_name,
                                     const std::string&                 layer_name,
                                     ChannelIdx                         channel,
                                     uint32_t                           position,
                                     const std::string&                 io_name,
                                     OPENEXR_IMF_NAMESPACE::PixelType   io_type,
                                     int                                io_part=0);


    //---------------------------------------------
    //
    // Read-only access to the shared lists & maps.
    //
    //---------------------------------------------

    const ChannelAliasPtrList&  channelAliasList () const;
    const AliasNameToListMap&   channelNameToAliasListMap () const;
    const ChannelIdxToListMap&  channelAliasToChannelMap () const;


    //---------------------------------------------------------------
    //
    // Print channel or '<layer>.<channel>' name to an output stream.
    //
    //---------------------------------------------------------------

    void    printChannelName (std::ostream&, const ChannelIdx&) const;
    void    printChannelFullName (std::ostream&, const ChannelIdx&) const;


  protected:

    ChannelIdx              m_last_assigned;                // Most recently assigned custom channel
    //
    ChannelAliasPtrList     m_channelalias_list;            // List of all added ChannelAliases
    AliasNameToListMap      m_channelalias_name_map;        // Map of channel names -> m_channelalias_list index
    ChannelIdxToListMap     m_channelalias_channel_map;     // Map of ChannelIdxs -> m_channelalias_list index
    //
    std::vector<Layer>      m_layers;                       // List of Layers
    LayerNameToListMap      m_layer_name_map;               // Map of layer names -> m_layers index

};



//--------------
//!rst:cpp:end::
//--------------

//-----------------
// Inline Functions
//-----------------

//-----------------
inline ChannelAlias* ChannelContext::getChannelAlias (const std::string& name) { return getChannelAlias(name.c_str()); }
inline ChannelIdx ChannelContext::getChannel (const char* name)
{
    ChannelAlias* chan = getChannelAlias(name);
    if (chan)
        return chan->channel();
    return Chan_Invalid;
}
inline ChannelIdx ChannelContext::getChannel (const std::string& name) { return getChannel(name.c_str()); }
inline ChannelIdx ChannelContext::lastAssignedChannel () const { return m_last_assigned; }
//
inline const ChannelAliasPtrList&
ChannelContext::channelAliasList () const { return m_channelalias_list; }
inline const ChannelContext::AliasNameToListMap&
ChannelContext::channelNameToAliasListMap () const { return m_channelalias_name_map; }
inline const ChannelContext::ChannelIdxToListMap&
ChannelContext::channelAliasToChannelMap () const { return m_channelalias_channel_map; }
//----------------
inline void
ChannelContext::printChannelName (std::ostream& os, const ChannelIdx& channel) const { os << getChannelName(channel); }
inline void
ChannelContext::printChannelFullName (std::ostream& os, const ChannelIdx& channel) const { os << getChannelFullName(channel); }


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_DCX_CHANNELCONTEXT_H
