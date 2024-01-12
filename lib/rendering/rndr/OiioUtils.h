// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <moonray/rendering/mcrt_common/Util.h>

// Make sure atomic<float> template specialization is defined
// before including OIIO headers
#include <scene_rdl2/render/util/AtomicFloat.h>
#include <OpenImageIO/imageio.h>
#include <memory>

namespace moonray {
namespace rndr {

class OiioUtils
{
public:
    using ImageInputUqPtr = OIIO::ImageInput::unique_ptr;

    inline static ImageInputUqPtr openFile(const std::string &filename);
    inline static bool getMetadata(const OIIO::ImageSpec &spec, const std::string &name, int &out);
    inline static bool getMetadata(const OIIO::ImageSpec &spec, const std::string &name, std::string &out);
    inline static bool getMetadata(const OIIO::ImageSpec &spec, const std::string &name, float out[3]);
    inline static int getPixChanOffset(const OIIO::ImageSpec &spec, const std::string &channelName); // error return -1

    //
    // Useful debug/test APIs
    //
    static std::string showImageSpec(const std::string &hd, const OIIO::ImageSpec &spec);
    static std::string showImageSpecAllMetadata(const std::string &hd, const OIIO::ImageSpec &spec);
    static std::string showImageSpecMetadata(const std::string &hd, const OIIO::ParamValue &param);
    static std::string showImageSpecMetadataTypeDesc(const OIIO::TypeDesc &typeDesc);
    static std::string showImageSpecMetadataData(const OIIO::ParamValue &param);
};


// static function
inline OiioUtils::ImageInputUqPtr
OiioUtils::openFile(const std::string &filename)
{
    return OIIO::ImageInput::open(filename);
}



// static function
inline bool
OiioUtils::getMetadata(const OIIO::ImageSpec &spec, const std::string &name, int &out)
{
    const OIIO::ParamValue *p = spec.find_attribute(name, OIIO::TypeDesc::INT);
    if (!p) return false;
    out = *(const int *)p->data();
    return true;
}

// static function
inline bool
OiioUtils::getMetadata(const OIIO::ImageSpec &spec, const std::string &name, std::string &out)
{
    const OIIO::ParamValue *p = spec.find_attribute(name, OIIO::TypeDesc::STRING);
    if (!p) return false;
    out = *(const char * const *)p->data();
    return true;
}

// static function
inline bool
OiioUtils::getMetadata(const OIIO::ImageSpec &spec, const std::string &name, float out[3])
{
    // We need to set all of BASETYPE, AGGREGATE and VECSEMANTICS for TypeDesc in this case somehow.
    const OIIO::ParamValue *p = 
        spec.find_attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::VEC3, OIIO::TypeDesc::VECTOR));
    if (!p) return false;
    out[0] = ((const float *)p->data())[0];
    out[1] = ((const float *)p->data())[1];
    out[2] = ((const float *)p->data())[2];
    return true;
}

// static function
inline int
OiioUtils::getPixChanOffset(const OIIO::ImageSpec &spec, const std::string &channelName)
{
    for (int offset = 0; offset < spec.nchannels; ++offset) {
        if (spec.channelnames[offset] == channelName) {
            return offset;
        }
    }
    return -1; // error : Could not find target channel
}

} // namespace rndr
} // namespace moonray

