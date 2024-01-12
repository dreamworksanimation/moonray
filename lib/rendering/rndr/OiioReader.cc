// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "OiioReader.h"

namespace moonray {
namespace rndr {

bool
OiioReader::readData(const int subImageId)
{
    if (!mIn) return false;

    if (!mIn->seek_subimage(subImageId, mSpec)) {
        return false;           // failed to seek
    }

    const OIIO::ImageSpec &spec = mSpec;
    mPixels.resize(spec.width * spec.height * spec.nchannels);
    if (mPixels.empty()) {
        return false;           // internal pixels buffer resize failed
    }

    mIn->read_image(OIIO::TypeDesc::FLOAT, mPixels.data());
    return true;
}

std::string
OiioReader::showSpec(const std::string &hd) const
{
    std::ostringstream ostr;

    if (!mIn) {
        ostr << "file open failed.";
        return ostr.str();
    }

    ostr << OiioUtils::showImageSpec(hd, mSpec);

    return ostr.str();
}

} // namespace rndr
} // namespace moonray

