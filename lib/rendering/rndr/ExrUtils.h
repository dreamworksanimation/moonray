// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/scene/rdl2/Metadata.h>
#include <OpenImageIO/imageio.h>

namespace moonray {
namespace rndr {

/**
 * Writes metadata to an exr header by parsing lists of strings.
 * Must convert the string into the appropriate data type.
 * Data types supported:
 *      * box2i
 *      * box2f
 *      * chromaticities
 *      * double
 *      * float
 *      * int
 *      * m33f
 *      * m44f
 *      * string
 *      * v2i
 *      * v2f
 *      * v3i
 *      * v3f
 *
 * @param   spec      The ImageSpec. We add the attributes to the ImageSpec,
 *                    which then get written to the exr header.
 * @param   metadata  The metadata object contains the list of attributes.
 */

void writeExrHeader(OIIO::ImageSpec& spec, const scene_rdl2::rdl2::Metadata *metadata);

void writeExrHeader(OIIO::ImageSpec &spec,
                    const std::vector<std::string> &attrNames,
                    const std::vector<std::string> &attrTypes,
                    const std::vector<std::string> &attrValues,
                    const std::string &metadataName);

void writeExrHeader(OIIO::ImageSpec &spec,
                    const std::string &attrNames,
                    const std::string &attrTypes,
                    const std::string &attrValues,
                    const std::string &metadataName);

} // namespace rndr
} // namespace moonray

