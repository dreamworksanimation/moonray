// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DcxDeepTile.h"


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


DeepTile::DeepTile (ChannelContext& channel_ctx,
                    WriteAccessMode write_access_mode,
                    bool yAxisUp) :
    PixelTile(channel_ctx,
              yAxisUp),
    m_write_access_mode(write_access_mode),
    m_flags_channel(OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid)
{
    m_spmask_channel[0] = m_spmask_channel[1] = OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid;
}

DeepTile::DeepTile (const Imf::Header& header,
                    ChannelContext& channel_ctx,
                    WriteAccessMode write_access_mode,
                    bool yAxisUp) :
    PixelTile(header,
              channel_ctx,
              yAxisUp),
    m_write_access_mode(write_access_mode),
    m_flags_channel(OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid)
{
    m_spmask_channel[0] = m_spmask_channel[1] = OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid;
}

DeepTile::DeepTile (const IMATH_NAMESPACE::Box2i& display_window,
                    const IMATH_NAMESPACE::Box2i& data_window,
                    bool sourceWindowsYup,
                    const ChannelSet& channels,
                    ChannelContext& channel_ctx,
                    WriteAccessMode write_access_mode,
                    bool yAxisUp) :
    PixelTile(display_window,
              data_window,
              sourceWindowsYup,
              channels,
              channel_ctx,
              yAxisUp),
    m_write_access_mode(write_access_mode),
    m_flags_channel(OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid)
{
    m_spmask_channel[0] = m_spmask_channel[1] = OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid;
}

DeepTile::DeepTile (const IMATH_NAMESPACE::Box2i& data_window,
                    int top_reference,
                    bool sourceWindowsYup,
                    const ChannelSet& channels,
                    ChannelContext& channel_ctx,
                    WriteAccessMode write_access_mode,
                    bool yAxisUp) :
    PixelTile(data_window,
              top_reference,
              sourceWindowsYup,
              channels,
              channel_ctx,
              yAxisUp),
    m_write_access_mode(write_access_mode),
    m_flags_channel(OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid)
{
    m_spmask_channel[0] = m_spmask_channel[1] = OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid;
}

DeepTile::DeepTile (const DeepTile& b) :
    PixelTile(b),
    m_write_access_mode(b.m_write_access_mode),
    m_flags_channel(b.m_flags_channel)
{
    m_spmask_channel[0] = b.m_spmask_channel[0];
    m_spmask_channel[1] = b.m_spmask_channel[1];
}


/*virtual*/
void
DeepTile::setChannels (const ChannelSet& channels,
                       bool force)
{
    PixelTile::setChannels(channels, force);
    // Get the spmask and flag channel assignments:
    m_spmask_channel[0] = m_spmask_channel[1] = OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid;
    m_flags_channel = OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid;
    foreach_channel(z, channels)
    {
        if      (z == Chan_SpBits1 && m_spmask_channel[0] == OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid)
            m_spmask_channel[0] = Chan_SpBits1;
        else if (z == Chan_SpBits2 && m_spmask_channel[1] == OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid)
            m_spmask_channel[1] = Chan_SpBits2;
        else if (z == Chan_DeepFlags && m_flags_channel == Chan_Invalid)
            m_flags_channel = Chan_DeepFlags;
    }
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT
