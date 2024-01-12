// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DcxPixelTile.h"


OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


PixelTile::PixelTile (ChannelContext& channel_ctx,
                      bool yAxisUp) :
    m_yaxis_up(yAxisUp),
    m_top_reference(0),
    m_dataWindow(IMATH_NAMESPACE::V2i(0,0), IMATH_NAMESPACE::V2i(-1,-1)),
    m_channel_ctx(&channel_ctx)
{
    //
}

PixelTile::PixelTile (const OPENEXR_IMF_NAMESPACE::Header& header,
                      ChannelContext& channel_ctx,
                      bool yAxisUp) :
    m_yaxis_up(yAxisUp),
    m_top_reference(header.displayWindow().max.y),
    m_channel_ctx(&channel_ctx)
{
    setDataWindow(header.dataWindow(), true/*sourceWindowsYAxisUp*/, true/*force*/); // May flip the data window
}

PixelTile::PixelTile (const IMATH_NAMESPACE::Box2i& display_window,
                      const IMATH_NAMESPACE::Box2i& data_window,
                      bool sourceWindowsYAxisUp,
                      const ChannelSet& channels,
                      ChannelContext& channel_ctx,
                      bool yAxisUp) :
    m_yaxis_up(yAxisUp),
    m_top_reference(display_window.max.y),
    m_dataWindow(data_window),
    m_channel_ctx(&channel_ctx)
{
    setDataWindow(data_window, sourceWindowsYAxisUp, true/*force*/); // May flip the data window
    setChannels(channels, true/*force*/);
}

PixelTile::PixelTile (const IMATH_NAMESPACE::Box2i& data_window,
                      int top_reference,
                      bool sourceWindowsYAxisUp,
                      const ChannelSet& channels,
                      ChannelContext&  channel_ctx,
                      bool yAxisUp) :
    m_yaxis_up(yAxisUp),
    m_top_reference(top_reference),
    m_dataWindow(data_window),
    m_channel_ctx(&channel_ctx)
{
    setDataWindow(data_window, sourceWindowsYAxisUp, true/*force*/); // May flip the data window
    setChannels(channels, true/*force*/);
}

PixelTile::PixelTile (const PixelTile& b) :
    m_yaxis_up(b.m_yaxis_up),
    m_top_reference(b.m_top_reference),
    m_dataWindow(b.m_dataWindow),
    m_channel_ctx(b.m_channel_ctx),
    //
    m_channels(b.m_channels)
{
    //
}


/*virtual*/
void
PixelTile::setDataWindow (const IMATH_NAMESPACE::Box2i& data_window,
                          bool sourceWindowYAxisUp,
                          bool /*force*/)
{
    m_dataWindow = data_window;
    // Flip data window vertically if source window is flipped:
    if (m_yaxis_up != sourceWindowYAxisUp)
    {
        m_dataWindow.min.y = m_top_reference - data_window.max.y;
        m_dataWindow.max.y = m_top_reference - data_window.min.y;
    }
}


/*virtual*/
void
PixelTile::setChannels (const ChannelSet& channels,
                        bool /*force*/)
{
    m_channels.clear();
    if (!m_channel_ctx)
        return;
    // Verify each channel is valid in this context:
    foreach_channel(z, channels)
    {
        if (m_channel_ctx->findChannelAlias(z))
            m_channels += z;
    }
}


/*virtual*/
PixelTile::~PixelTile ()
{
   //
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT
