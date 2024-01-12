// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "OiioUtils.h"

#include <iomanip>

namespace moonray {
namespace rndr {

// static function
std::string
OiioUtils::showImageSpec(const std::string &hd, const OIIO::ImageSpec &spec)
{
    std::ostringstream ostr;
    ostr << hd << "ImageSpec {\n";
    ostr << hd << "  width:" << spec.width << '\n';
    ostr << hd << "  height:" << spec.height << '\n';
    ostr << hd << "  nchannels:" << spec.nchannels << '\n';
    ostr << hd << "  alpha_channel:" << spec.alpha_channel << '\n';
    ostr << hd << "  z_channel:" << spec.z_channel << '\n';

    ostr << hd << "  channels {\n";
    for (int i = 0; i < spec.nchannels; ++i) {
        ostr << hd << "    chanId:" << std::setw(2) << i << " name:" << spec.channelnames[i] << '\n';
    }
    ostr << hd << "  }\n";

    ostr << showImageSpecAllMetadata(hd + "  ", spec) << '\n';

    ostr << hd << "}";
    return ostr.str();
}

// static function
std::string
OiioUtils::showImageSpecAllMetadata(const std::string &hd, const OIIO::ImageSpec &spec)
{
    std::ostringstream ostr;
    ostr << hd << "metadata {\n";
    for (size_t i = 0; i < spec.extra_attribs.size(); ++i) {
        const OIIO::ParamValue &p(spec.extra_attribs[i]);
        ostr << showImageSpecMetadata(hd + "  ", p);
        ostr << '\n';
    }
    ostr << hd << "}";
    return ostr.str();
}

// static function
std::string
OiioUtils::showImageSpecMetadata(const std::string &hd, const OIIO::ParamValue &param)
{
    std::ostringstream ostr;
    ostr << hd << "name:" << param.name() << ' '
         << showImageSpecMetadataTypeDesc(param.type()) << ' '
         << showImageSpecMetadataData(param);
    return ostr.str();
}

// static function
std::string
OiioUtils::showImageSpecMetadataTypeDesc(const OIIO::TypeDesc &typeDesc)
{
    std::ostringstream ostr;

    switch (typeDesc.aggregate) {
    case OIIO::TypeDesc::SCALAR   : ostr << "(scalar)";   break;
    case OIIO::TypeDesc::VEC2     : ostr << "(vec2)";     break;
    case OIIO::TypeDesc::VEC3     : ostr << "(vec3)";     break;
    case OIIO::TypeDesc::VEC4     : ostr << "(vec4)";     break;
    case OIIO::TypeDesc::MATRIX33 : ostr << "(matrix33)"; break;
    case OIIO::TypeDesc::MATRIX44 : ostr << "(matrix44)"; break;
    }
    ostr << ' ';
    
    switch (typeDesc.basetype) {
    case OIIO::TypeDesc::UNKNOWN   : ostr << "<unknown>"; break;
    case OIIO::TypeDesc::NONE      : ostr << "<none>";    break;
    case OIIO::TypeDesc::UCHAR     : ostr << "<uint8>";   break;
    case OIIO::TypeDesc::CHAR      : ostr << "<int8>";    break;
    case OIIO::TypeDesc::USHORT    : ostr << "<uint16>";  break;
    case OIIO::TypeDesc::SHORT     : ostr << "<int16>";   break;
    case OIIO::TypeDesc::UINT      : ostr << "<uint32>";  break;
    case OIIO::TypeDesc::INT       : ostr << "<int32>";   break;
    case OIIO::TypeDesc::ULONGLONG : ostr << "<uint64>";  break;
    case OIIO::TypeDesc::LONGLONG  : ostr << "<int64>";   break;
    case OIIO::TypeDesc::HALF      : ostr << "<half>";    break;
    case OIIO::TypeDesc::FLOAT     : ostr << "<float>";   break;
    case OIIO::TypeDesc::DOUBLE    : ostr << "<double>";  break;
    case OIIO::TypeDesc::STRING    : ostr << "<string>";  break;
    case OIIO::TypeDesc::PTR       : ostr << "<ptr>";     break;
    }
    ostr << ' ';

    switch (typeDesc.vecsemantics) {
    case OIIO::TypeDesc::NOXFORM  : ostr << "{noxform}";  break;
    case OIIO::TypeDesc::COLOR    : ostr << "{color}";    break;
    case OIIO::TypeDesc::POINT    : ostr << "{point}";    break;
    case OIIO::TypeDesc::VECTOR   : ostr << "{vector}";   break;
    case OIIO::TypeDesc::NORMAL   : ostr << "{normal}";   break;
    case OIIO::TypeDesc::TIMECODE : ostr << "{timecode}"; break;
    case OIIO::TypeDesc::KEYCODE  : ostr << "{keycode}";  break;
    }
    ostr << ' ';

    ostr << "arraylen:" << typeDesc.arraylen << ' ';
    
    ostr << "basesize:" << typeDesc.basesize()
         << " elementsize:" << typeDesc.elementsize()
         << " numelements:" << typeDesc.numelements()
         << " >" << typeDesc.c_str() << "<";

    return ostr.str();
}

// static function
std::string
OiioUtils::showImageSpecMetadataData(const OIIO::ParamValue &param)
{
    std::ostringstream ostr;

    int numelements = param.type().numelements();
    size_t basesize = param.type().basesize();
    int vecLen = (int)param.type().aggregate;

    uintptr_t dataAddr = (uintptr_t)param.data();

    ostr << '<';
    for (int elemId = 0; elemId < numelements; ++elemId) {
        ostr << '{';
        for (int vecId = 0; vecId < vecLen; ++vecId) {
            switch (param.type().basetype) {
            case OIIO::TypeDesc::UNKNOWN   : ostr << "?";                             break;
            case OIIO::TypeDesc::NONE      :                                          break;
            case OIIO::TypeDesc::UCHAR     : ostr << *(unsigned char *)dataAddr;      break;
            case OIIO::TypeDesc::CHAR      : ostr << *(char *)dataAddr;               break;
            case OIIO::TypeDesc::USHORT    : ostr << *(unsigned short *)dataAddr;     break;
            case OIIO::TypeDesc::SHORT     : ostr << *(short *)dataAddr;              break;
            case OIIO::TypeDesc::UINT      : ostr << *(unsigned int *)dataAddr;       break;
            case OIIO::TypeDesc::INT       : ostr << *(int *)dataAddr;                break;
            case OIIO::TypeDesc::ULONGLONG : ostr << *(unsigned long long *)dataAddr; break;
            case OIIO::TypeDesc::LONGLONG  : ostr << *(long long *)dataAddr;          break;
            case OIIO::TypeDesc::HALF      : ostr << "<half>";                        break;
            case OIIO::TypeDesc::FLOAT     : ostr << *(float *)dataAddr;              break;
            case OIIO::TypeDesc::DOUBLE    : ostr << *(double *)dataAddr;             break;
            case OIIO::TypeDesc::STRING    : ostr << *(const char * const *)dataAddr; break;
            case OIIO::TypeDesc::PTR       : ostr << "<ptr>";                         break;
            }
            if (vecId != vecLen - 1) {
                ostr << ',';
            }

            dataAddr += (uintptr_t)basesize;
        } // vecId
        ostr << '}';
        if (elemId != numelements - 1) {
            ostr << ',';
        }
    } // elemId
    ostr << '>';

    return ostr.str();
}

} // namespace rndr
} // namespace moonray

