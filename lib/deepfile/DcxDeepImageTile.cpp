// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DcxDeepImageTile.h"
#include "DcxChannelContext.h"

#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfDeepImage.h>

OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER


DeepImageOutputTile::DeepImageOutputTile (const IMATH_NAMESPACE::Box2i& display_window,
                                          const IMATH_NAMESPACE::Box2i& data_window,
                                          bool sourceWindowsYup,
                                          const ChannelSet& channels,
                                          ChannelContext& channel_ctx,
                                          bool yAxisUp) :
    DeepTile(data_window,
             display_window.max.y,
             sourceWindowsYup,
             channels,
             channel_ctx,
             WRITE_RANDOM,
             yAxisUp),
    ImageFormat(std::string(""), display_window, 1.0f/*pa*/),
    m_file(0)
{
#if 0
    // Make sure output channels have Z's, metadata enabled:
    m_channels += OPENDCX_INTERNAL_NAMESPACE::Mask_Deep;
    m_channels += m_spmask_channel[0];
    m_channels += m_spmask_channel[1];
    m_channels += m_flags_channel;
#endif
    setDataWindow(data_window, sourceWindowsYup, true/*force*/);
}


DeepImageOutputTile::~DeepImageOutputTile ()
{
    const size_t nLines = m_deep_lines.size();
    for (size_t y=0; y < nLines; ++y)
        delete m_deep_lines[y];
    delete m_file;
}


/*virtual*/
void
DeepImageOutputTile::setChannels (const ChannelSet& channels,
                                  bool force)
{
    if (!force && channels == m_channels)
        return; // no change, do nothing
    deleteDeepLines();
    DeepTile::setChannels(channels, force);
}


/*virtual*/
void
DeepImageOutputTile::setDataWindow (const IMATH_NAMESPACE::Box2i& data_window,
                                    bool sourceWindowYAxisUp,
                                    bool force)
{
    if (!force && data_window == m_dataWindow)
        return; // no change, do nothing
    deleteDeepLines();
    DeepTile::setDataWindow(data_window, sourceWindowYAxisUp);
    size_t nLines = std::max(0, h());
    m_deep_lines.resize(nLines);
    memset(&m_deep_lines[0], 0, nLines*sizeof(DeepLine*));
}


size_t
DeepImageOutputTile::bytesUsed () const
{
    size_t count = 0;
    size_t nLines = m_deep_lines.size();
    for (size_t j=0; j < nLines; ++j)
    {
        const DeepLine* dl = m_deep_lines[j];
        if (dl)
        {
            int chan_index = 0;
            foreach_channel(z, m_channels)
                count += dl->channel_arrays[chan_index].size();
        }
    }
    return (count * sizeof(float));
}


void
DeepImageOutputTile::deleteDeepLines ()
{
    const size_t nLines = m_deep_lines.size();
    for (size_t j=0; j < nLines; ++j)
    {
        delete m_deep_lines[j];
        m_deep_lines[j] = 0;
    }
}

/*virtual*/
size_t
DeepImageOutputTile::getNumSamplesAt (int x, int y) const
{
    const DeepLine* dl = getLine(y);
    if (!dl)
        return 0;
    x -= m_dataWindow.min.x;
    return (x < 0 || x >= (int)dl->samples_per_pixel.size())?0:dl->samples_per_pixel[x];
}


//----------------------------------------------------------


DeepImageOutputTile::DeepLine::DeepLine (uint32_t width,
                                         const ChannelSet& _channels) :
    channels(_channels)
{
    channel_arrays.resize(channels.size());
    samples_per_pixel.resize(width, 0);
}


void
DeepImageOutputTile::DeepLine::get (int xoffset,
                                    OPENDCX_INTERNAL_NAMESPACE::DeepPixel& deep_pixel) const
{
#ifdef DEBUG
    assert(xoffset >= 0 && xoffset < (int)samples_per_pixel.size());
#endif

    const uint32_t nSegments = samples_per_pixel[xoffset];

    ChannelSet out_channels(channels);
    out_channels -= Mask_DeepMetadata;

    deep_pixel.clear();
    deep_pixel.setChannels(out_channels);
    if (nSegments == 0)
        return;
    deep_pixel.reserve(nSegments);

    const uint32_t foffset = floatOffset(xoffset);

    OPENDCX_INTERNAL_NAMESPACE::DeepSegment ds;
    OPENDCX_INTERNAL_NAMESPACE::Pixelf dp(deep_pixel.channels());
    float sp1, sp2;

    for (uint32_t i=0; i < nSegments; ++i)
    {
        ds.Zf = ds.Zb = 0.0f;
        sp1 = sp2 = 0.0f;
        ds.metadata.flags.clearAll();

        int chan_index = 0;
        foreach_channel(z, channels)
        {
            const float v = channel_arrays[chan_index][foffset + i];
            if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_ZFront)
                ds.Zf = v;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_ZBack)
                ds.Zb = v;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits1)
                sp1 = v;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits2)
                sp2 = v;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_DeepFlags)
                ds.metadata.flags.fromFloat(v);
            else
                dp[z] = v;
            ++chan_index;
        }

        ds.metadata.spmask.fromFloat(sp1, sp2);
        if (ds.Zb < ds.Zf)
            ds.Zb = ds.Zf;
        ds.index = -1; // gets updated when added to deep pixel

        deep_pixel.append(ds, dp);
    }
}


void
DeepImageOutputTile::DeepLine::getMetadata(int xoffset,
                                           size_t sample,
                                           OPENDCX_INTERNAL_NAMESPACE::DeepMetadata& metadata) const
{
    const uint32_t nSegments = samples_per_pixel[xoffset];
#ifdef DEBUG
    assert(xoffset >= 0 && xoffset < (int)samples_per_pixel.size());
    assert(sample < nSegments);
#endif
    if (nSegments == 0)
        return;

    const uint32_t foffset = floatOffset(xoffset) + (uint32_t)sample;

    float sp1=0.0f, sp2=0.0f;
    metadata.flags.clearAll();

    int chan_index = 0;
    foreach_channel(z, channels)
    {
        if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits1)
            sp1 = channel_arrays[chan_index][foffset];
        else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits2)
            sp2 = channel_arrays[chan_index][foffset];
        else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_DeepFlags)
            metadata.flags = (int)floorf(channel_arrays[chan_index][foffset]);
        ++chan_index;
    }

    metadata.spmask.fromFloat(sp1, sp2);
}


void
DeepImageOutputTile::DeepLine::set (int xoffset,
                                    const OPENDCX_INTERNAL_NAMESPACE::DeepPixel& deep_pixel)
{
#ifdef DEBUG
    assert(xoffset >= 0 && xoffset < (int)samples_per_pixel.size());
#endif
    const uint32_t nWriteSegments = (uint32_t)deep_pixel.size();
    if (nWriteSegments == 0)
    {
        clear(xoffset);
        return;
    }

    const uint32_t foffset = floatOffset(xoffset);
    const uint32_t nCurrSegments = samples_per_pixel[xoffset];

    if (nCurrSegments < nWriteSegments)
    {
        // Add segments:
        const size_t nAdd = (nWriteSegments - nCurrSegments);
        int chan_index = 0;
        foreach_channel(z, channels)
        {
            FloatVec& values = channel_arrays[chan_index++];
#ifdef DEBUG
            assert(foffset <= values.size()); // shouldn't happen...
#endif
            // Round the memory reserve up in chunks to avoid constantly resizing/copying:
            if (values.capacity() < values.size()+nAdd)
                values.reserve(values.size() + (size_t)int(float(values.size())/1.5f));
            values.insert(values.begin() + foffset, nAdd, 0.0f);
        }
    }
    else if (nCurrSegments > nWriteSegments)
    {
        // Remove segments:
        const size_t nRemove = (nCurrSegments - nWriteSegments);
        int chan_index = 0;
        foreach_channel(z, channels)
        {
            FloatVec& values = channel_arrays[chan_index++];
            values.erase(values.begin() + foffset,
                         values.begin() + foffset + nRemove);
        }
    }

    float sp1, sp2, flags;
    // Copy segment channel data:
    for (uint32_t i=0; i < nWriteSegments; ++i)
    {
        const DeepSegment& segment = deep_pixel[i];
        segment.spMask().toFloat(sp1, sp2);
        flags = segment.flags().toFloat();
        const Pixelf& pixel = deep_pixel.getSegmentPixel(segment);

        float v;
        int chan_index = 0;
        foreach_channel(z, channels)
        {
            if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_ZFront)
                v = segment.Zf;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_ZBack)
                v = segment.Zb;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits1)
                v = sp1;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits2)
                v = sp2;
            else if (z == OPENDCX_INTERNAL_NAMESPACE::Chan_DeepFlags)
                v = flags;
            else if (deep_pixel.channels().contains(z))
                v = pixel[z];
            else
                v = 0.0f;
            channel_arrays[chan_index++][foffset + i] = v;
        }
    }

    samples_per_pixel[xoffset] = nWriteSegments;
}

void
DeepImageOutputTile::DeepLine::clear (int xoffset)
{
#ifdef DEBUG
    assert(xoffset >= 0 && xoffset < (int)samples_per_pixel.size());
#endif
    const uint32_t nCurrSegments = samples_per_pixel[xoffset];
    if (nCurrSegments == 0)
        return;

    const uint32_t foffset = floatOffset(xoffset);

    int chan_index = 0;
    foreach_channel(z, channels)
    {
        FloatVec& values = channel_arrays[chan_index++];
        values.erase(values.begin() + foffset,
                     values.begin() + foffset + nCurrSegments);
    }
    samples_per_pixel[xoffset] = 0;
}


//----------------------------------------------------------


DeepImageOutputTile::DeepLine*
DeepImageOutputTile::createDeepLine (int y)
{
    if (y < m_dataWindow.min.y || y > m_dataWindow.max.y)
        return 0; // don't crash...  TODO: throw exception?
    y -= m_dataWindow.min.y;
#ifdef DEBUG
    assert(m_deep_lines.size() > 0);
    assert(y >= 0 && y < (int)m_deep_lines.size());
#endif
    DeepLine* dl = m_deep_lines[y];
    if (!dl)
        m_deep_lines[y] = dl = new DeepLine(w(), channels());
    if (!dl)
        return 0; // don't crash... TODO: how to best handle memory alloc error...?
    return dl;
}


/*virtual*/
bool
DeepImageOutputTile::getDeepPixel (int x,
                                   int y,
                                   OPENDCX_INTERNAL_NAMESPACE::DeepPixel& pixel) const
{
    pixel.clear();
    if (!isActivePixel(x, y))
        return false;
    if (m_channels.empty())
        return true;

    DeepLine* dl = (const_cast<DeepImageOutputTile*>(this))->createDeepLine(y);
    if (!dl)
        return false; // don't crash...

    dl->get(x - m_dataWindow.min.x, pixel);

    return true;
}

/*virtual*/
bool
DeepImageOutputTile::getSampleMetadata (int x,
                                        int y,
                                        size_t sample,
                                        OPENDCX_INTERNAL_NAMESPACE::DeepMetadata& metadata) const
{
    if (!isActivePixel(x, y))
        return false;
    if (m_channels.empty())
        return true;

    DeepLine* dl = (const_cast<DeepImageOutputTile*>(this))->createDeepLine(y);
    if (!dl)
        return false; // don't crash...

    dl->getMetadata(x - m_dataWindow.min.x, sample, metadata);

    return true;
}



/*virtual*/
bool
DeepImageOutputTile::setDeepPixel (int x,
                                   int y,
                                   const OPENDCX_INTERNAL_NAMESPACE::DeepPixel& deep_pixel)
{
    DeepLine* dl = createDeepLine(y);
    if (!dl || x < m_dataWindow.min.x || x > m_dataWindow.max.x)
        return false; // don't crash...

    // Copy DeepPixel data into packed DeepLine arrays (offseting x into array range):
    dl->set(x - m_dataWindow.min.x, deep_pixel);

    return true;
}


/*virtual*/
bool
DeepImageOutputTile::clearDeepPixel (int x,
                                     int y)
{
    DeepLine* dl = createDeepLine(y);
    if (!dl || x < m_dataWindow.min.x || x > m_dataWindow.max.x)
        return false; // don't crash...

    dl->clear(x - m_dataWindow.min.x); // Offset x into DeepLine array

    return true;
}


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

/*virtual*/
void
DeepImageOutputTile::setOutputFile (const char* filename,
                                    const Imf::Header& headerToCopyAttributes)
{
    if (!filename || !filename[0] || m_filename == filename)
        return;
    m_filename = filename;

    Imf::Header header(headerToCopyAttributes);

    // Update the header params from the ImageFormat:
    //      header.displayWindow()      = displayWindow();
    //      header.pixelAspectRatio()   = (float)pixelAspectRatio();
    //      header.screenWindowCenter() = screenWindowCenter();
    //      header.screenWindowWidth()  = screenWindowWidth();
    ImageFormat::toHeader(header);

    header.dataWindow() = flipY(m_dataWindow); // Flip data window back to Y-down if tile is Y-up

    // Make line order best align with Y-up mode:
    if (headerToCopyAttributes.lineOrder() == Imf::DECREASING_Y)
        header.lineOrder() = (m_yaxis_up)?Imf::INCREASING_Y:Imf::DECREASING_Y;
    else if (headerToCopyAttributes.lineOrder() == Imf::INCREASING_Y)
        header.lineOrder() = (m_yaxis_up)?Imf::DECREASING_Y:Imf::INCREASING_Y;
    else
        header.lineOrder() = (m_yaxis_up)?Imf::DECREASING_Y:Imf::INCREASING_Y;

    // Ignore incoming compression for now and always use ZIPS: (TODO do this always...?)
    header.compression() = Imf::ZIPS_COMPRESSION;

    ChannelSet write_channels(m_channels);
    //write_channels -= Mask_SpMask8;
    //write_channels -= Mask_DeepFlags;

    foreach_channel(z, write_channels)
    {
        const ChannelAlias* c = getChannelAlias(z);
#ifdef DEBUG
        assert(c); // shouldn't happen...
#endif

        // Use the fileIOName for EXR output channel name:
        header.channels().insert(c->fileIOName(), Imf::Channel(c->fileIOPixelType()));
    }

#if 0
    if (m_channels.contains(Mask_SpMask8))
    {
        header.channels().insert(spMask8Channel1Name, Imf::Channel(Imf::FLOAT));
        header.channels().insert(spMask8Channel2Name, Imf::Channel(Imf::FLOAT));
    }
    if (m_channels.contains(Mask_DeepFlags))
        header.channels().insert(flagsChannelName, Imf::Channel(Imf::HALF));
#endif

    delete m_file;
    m_file = new Imf::DeepScanLineOutputFile(filename, header);
    deleteDeepLines();
}


//
// Write the DeepLine to the assigned output file.
// If flush_line is true the DeepLine memory is freed.
//

typedef std::vector<DeepImageOutputTile::HalfVec>  HalfSamples;
typedef std::vector<DeepImageOutputTile::FloatVec> FloatSamples;
typedef std::vector<DeepImageOutputTile::UintVec>  UintSamples;


/*virtual*/
void
DeepImageOutputTile::writeScanline (int y,
                                    bool flush_line)
{
    if (!m_file || y < m_dataWindow.min.y || y > m_dataWindow.max.y)
        return; // don't crash...  TODO: throw exception?

    y -= m_dataWindow.min.y;
    DeepLine* dl = m_deep_lines[y];
    DeepLine empty_dl(w(), channels()); // dummy DeepLine for empty line
    if (!dl)
        dl = &empty_dl;
#ifdef DEBUG
    assert(dl); // shouldn't happen...
#endif

    // Unpack the floats to arrays of the appropriate pixel types and
    // assign the frambuffer slices to them:
    const size_t nChannels = m_channels.size();
    if (nChannels == 0)
        return;

    const size_t nPixels = dl->samples_per_pixel.size();
#ifdef DEBUG
    assert((int)nPixels == this->w());
#endif

    // Unpacked sample data storage (only some of these actually get filled in):
    std::vector<HalfSamples>   half_samples(nChannels);
    std::vector<FloatSamples> float_samples(nChannels);
    std::vector<UintSamples>   uint_samples(nChannels);
    std::vector<PtrVec>           data_ptrs(nChannels);

    Imf::DeepFrameBuffer fb;
    fb.insertSampleCountSlice(Imf::Slice(Imf::UINT,
                                         (char*)(dl->samples_per_pixel.data() - m_dataWindow.min.x),
                                         sizeof(uint32_t)/*xStride*/,
                                         0/*yStride*/));

    int chan_index = 0;
    size_t sample_stride = 0;
    foreach_channel(z, m_channels)
    {
        const ChannelAlias* c = getChannelAlias(z);
#ifdef DEBUG
        assert(c); // shouldn't happen...
#endif

        const float* IN = dl->channel_arrays[chan_index].data();

        PtrVec& ptrs = data_ptrs[chan_index];
        ptrs.resize(nPixels);

        switch (c->fileIOPixelType())
        {
            case Imf::HALF:
            {
                HalfSamples& value_lists = half_samples[chan_index];
                value_lists.resize(nPixels);
                for (size_t i=0; i < nPixels; ++i)
                {
                    const size_t nSamples = dl->samples_per_pixel[i];
                    HalfVec& samples = value_lists[i];
                    samples.reserve(nSamples);
                    for (size_t s=0; s < nSamples; ++s)
                        samples.push_back(half(*IN++));
                    ptrs[i] = samples.data(); // always point to valid data, even for 0 samples...
                }
                sample_stride = sizeof(half);
                break;
            }

            case Imf::FLOAT:
            {
                FloatSamples& value_lists = float_samples[chan_index];
                value_lists.resize(nPixels);
                for (size_t i=0; i < nPixels; ++i)
                {
                    const size_t nSamples = dl->samples_per_pixel[i];
                    FloatVec& samples = value_lists[i];
                    samples.reserve(nSamples);
                    for (size_t s=0; s < nSamples; ++s)
                        samples.push_back(*IN++);
                    ptrs[i] = samples.data(); // always point to valid data, even for 0 samples...
                }
                sample_stride = sizeof(float);
                break;
            }

            case Imf::UINT:
            {
                UintSamples& value_lists = uint_samples[chan_index];
                value_lists.resize(nPixels);
                for (size_t i=0; i < nPixels; ++i)
                {
                    const size_t nSamples = dl->samples_per_pixel[i];
                    UintVec& samples = value_lists[i];
                    samples.reserve(nSamples);
                    for (size_t s=0; s < nSamples; ++s, ++IN)
                        samples.push_back(int(floorf((*IN < 0.0f)?0.0f:*IN)));
                    ptrs[i] = samples.data();
                }
                sample_stride = sizeof(uint32_t);
                break;
            }

            default:
#ifdef DEBUG
                assert(0); // TODO: throw exception instead?
#endif
                break;
        }

        fb.insert(c->fileIOName(), Imf::DeepSlice(c->fileIOPixelType(),
                                                  (char*)(ptrs.data() - m_dataWindow.min.x),
                                                  sizeof(void*)/*xStride*/,
                                                  0/*yStride*/,
                                                  sample_stride/*sampleStride*/));

        ++chan_index;
    }

    // Write line to file:
    m_file->setFrameBuffer(fb);
    m_file->writePixels(1/*nLines*/);

    // Free the DeepLine:
    if (flush_line)
    {
        delete m_deep_lines[y];
        m_deep_lines[y] = 0;
    }
}

//
// Write entire tile to output file.
// If flush_tile is true all DeepLine memory is freed.
//

/*virtual*/
void
DeepImageOutputTile::writeTile (bool flush_tile)
{
    for (int y=m_dataWindow.min.y; y < m_dataWindow.max.y; ++y)
        writeScanline(y, flush_tile);
}


OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT
