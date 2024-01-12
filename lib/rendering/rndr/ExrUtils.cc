// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ExrUtils.h"

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/common/except/exceptions.h>

namespace moonray {
namespace rndr {

template <typename T, std::size_t N>
void makeArray(std::array<T, N>& array, const std::string& values)
{
    std::istringstream ins(values);
    ins.exceptions(std::ios_base::badbit | std::ios::failbit);

    try {
        std::copy_n(std::istream_iterator<T>(ins), N, array.begin());
    } catch (...) {
        throw scene_rdl2::except::ValueError("error converting to array of values.\n"
            "\tMake sure input string has correct number of elements."
            " Input string format should be \"elem0 elem1 ...\"\n");
    }
}

void
writeExrHeader(OIIO::ImageSpec& spec, const scene_rdl2::rdl2::Metadata *metadata)
{
    // metadata does not exist
    if (!metadata) {
        return;
    }

    // get list of attributes
    const std::vector<std::string>& attrNames = metadata->getAttributeNames();
    const std::vector<std::string>& attrTypes = metadata->getAttributeTypes();
    const std::vector<std::string>& attrValues = metadata->getAttributeValues();

    writeExrHeader(spec, attrNames, attrTypes, attrValues, metadata->getName());
}

void
writeExrHeader(OIIO::ImageSpec &spec,
               const std::vector<std::string> &attrNames,
               const std::vector<std::string> &attrTypes,
               const std::vector<std::string> &attrValues,
               const std::string &metadataName)
{
    // convert each attribute to appropriate data type
    for (size_t i = 0; i < attrNames.size(); ++i) {
        writeExrHeader(spec, attrNames[i], attrTypes[i], attrValues[i], metadataName);
    }
}

void
writeExrHeader(OIIO::ImageSpec &spec,
               const std::string &attrNames,
               const std::string &attrTypes,
               const std::string &attrValues,
               const std::string &metadataName)
{
    try {
        // scalar data types
        if (attrTypes == "float") {
            spec.attribute(attrNames, std::stof(attrValues));

        } else if (attrTypes == "int") {
            spec.attribute(attrNames, std::stoi(attrValues));

        } else if (attrTypes == "string") {
            spec.attribute(attrNames, attrValues);

        } else if (attrTypes == "double") {
            // must specify type for double
            OIIO::TypeDesc type(OIIO::TypeDesc::DOUBLE, OIIO::TypeDesc::SCALAR);
            spec.attribute(attrNames, type, attrValues);

        } else if (attrTypes == "chromaticities") {
            // array of 8 floats
            std::array<float,8> chrom;
            makeArray(chrom, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::SCALAR, chrom.size());
            spec.attribute(attrTypes, type, chrom.data());
        }

        // vector data types
        else if (attrTypes == "v2i") {
            std::array<int, 2> v2i;
            makeArray(v2i, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::INT, OIIO::TypeDesc::VEC2);
            spec.attribute(attrNames, type, v2i.data());

        } else if (attrTypes == "v2f") {
            std::array<float, 2> v2f;
            makeArray(v2f, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::VEC2);
            spec.attribute(attrNames, type, v2f.data());

        } else if (attrTypes == "v3i") {
            std::array<int, 3> v3i;
            makeArray(v3i, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::INT, OIIO::TypeDesc::VEC3);
            spec.attribute(attrNames, type, v3i.data());

        } else if (attrTypes == "v3f") {
            std::array<float, 3> v3f;
            makeArray(v3f, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::VEC3);
            spec.attribute(attrNames, type, v3f.data());
        }

        // matrix data types
        else if (attrTypes == "m33f") {
            std::array<float, 9> m33f;
            makeArray(m33f, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::MATRIX33);
            spec.attribute(attrNames, type, m33f.data());

        } else if (attrTypes == "m44f") {
            std::array<float, 16> m44f;
            makeArray(m44f, attrValues);
            OIIO::TypeDesc type(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::MATRIX44);
            spec.attribute(attrNames, type, m44f.data());
        }

        // box data types
        else if (attrTypes == "box2i") {
            // array of 2 vec2s
            std::array<int, 4> box2i;
            makeArray(box2i, attrValues);

            // make sure min < max
            if (box2i[0] > box2i[2]) {
                std::swap(box2i[0], box2i[2]);
            }
            if (box2i[1] > box2i[3]) {
                std::swap(box2i[1], box2i[3]);
            }

            OIIO::TypeDesc type(OIIO::TypeDesc::INT, OIIO::TypeDesc::VEC2, 2);
            spec.attribute(attrNames, type, box2i.data());

        } else if (attrTypes == "box2f") {
            // array of 2 vec2s
            std::array<float, 4> box2f;
            makeArray(box2f, attrValues);

            // make sure min < max
            if (box2f[0] > box2f[2]) {
                std::swap(box2f[0], box2f[2]);
            }
            if (box2f[1] > box2f[3]) {
                std::swap(box2f[1], box2f[3]);
            }

            OIIO::TypeDesc type(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::VEC2, 2);
            spec.attribute(attrNames, type, box2f.data());
        }

        // unkown data types
        else {
            throw scene_rdl2::except::TypeError("datatype " + attrTypes + " is not supported");
        }

    } catch (const std::exception& e) {
        scene_rdl2::logging::Logger::error("Metadata(\"" + metadataName + "\") : \"" + attrNames + "\": " + e.what());
    }
}

} // namespace rndr
} // namespace moonray

