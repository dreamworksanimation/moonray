// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// Created by kjeffery on 5/10/16.
//

#pragma once

#include <array>
#include <ios>

namespace moonray_stats {

struct IOSBaseFlags
{
public:
    IOSBaseFlags() :
        // These are the defaults when std::basic_ios::init is called.
        mPrecision(6),
        mWidth(0),
        mFlags(std::ios_base::skipws | std::ios_base::dec),
        mLocale(std::locale())
    {
    }

    explicit IOSBaseFlags(const std::ios_base& ios) :
        mPrecision(ios.precision()),
        mWidth(ios.width()),
        mFlags(ios.flags()),
        mLocale(ios.getloc())
    {
    }

    void imbue(std::ios_base& ios) const
    {
        ios.flags(mFlags);
        ios.precision(mPrecision);
        ios.width(mWidth);
        ios.imbue(mLocale);
    }

    void imbue(std::locale mylocale) 
    {
        mLocale = mylocale;
    }

    std::ios_base::fmtflags flags(std::ios_base::fmtflags fmt)
    {
        const std::ios_base::fmtflags old = mFlags;
        mFlags = fmt;
        return old;
    }

    std::ios_base::fmtflags setf(std::ios_base::fmtflags fmt)
    {
        const std::ios_base::fmtflags old = mFlags;
        mFlags |= fmt;
        return old;
    }

    std::ios_base::fmtflags setf(std::ios_base::fmtflags fmt,
                                 std::ios_base::fmtflags mask)
    {
        const std::ios_base::fmtflags old = mFlags;
        mFlags &= ~mask;
        mFlags |= (fmt & mask);
        return old;
    }

    void unsetf(std::ios_base::fmtflags mask)
    {
        mFlags &= ~mask;
    }

    void dec()
    {
        setf(std::ios_base::dec, std::ios_base::basefield);
    }

    void hex()
    {
        setf(std::ios_base::hex, std::ios_base::basefield);
    }

    void oct()
    {
        setf(std::ios_base::oct, std::ios_base::basefield);
    }

    void left()
    {
        setf(std::ios_base::left, std::ios_base::adjustfield);
    }

    void right()
    {
        setf(std::ios_base::right, std::ios_base::adjustfield);
    }

    void internal()
    {
        setf(std::ios_base::internal, std::ios_base::adjustfield);
    }

    void fixed()
    {
        setf(std::ios_base::fixed, std::ios_base::floatfield);
    }

    void scientific()
    {
        setf(std::ios_base::scientific, std::ios_base::floatfield);
    }

    void hexfloat()
    {
        setf(std::ios_base::fixed | std::ios_base::scientific,
             std::ios_base::floatfield);
    }

    void defaultfloat()
    {
        unsetf(std::ios_base::floatfield);
    }

    void precision(std::streamsize p)
    {
        mPrecision = p;
    }

    std::streamsize precision() const
    {
        return mPrecision;
    }

    void width(std::streamsize w)
    {
        mWidth = w;
    }

    std::streamsize width() const
    {
        return mWidth;
    }

    std::locale getloc() const
    {
        return mLocale;
    }

private:
    std::streamsize mPrecision;
    std::streamsize mWidth;
    std::ios::fmtflags mFlags;
    std::locale mLocale;
};

template <class CharT, class Traits>
class IOSBasicFlags : private IOSBaseFlags
{
public:
    using IOSBaseFlags::flags;
    using IOSBaseFlags::setf;
    using IOSBaseFlags::unsetf;
    using IOSBaseFlags::dec;
    using IOSBaseFlags::hex;
    using IOSBaseFlags::oct;
    using IOSBaseFlags::left;
    using IOSBaseFlags::right;
    using IOSBaseFlags::internal;
    using IOSBaseFlags::fixed;
    using IOSBaseFlags::scientific;
    using IOSBaseFlags::hexfloat;
    using IOSBaseFlags::defaultfloat;
    using IOSBaseFlags::precision;
    using IOSBaseFlags::width;

    IOSBasicFlags() :
        IOSBaseFlags(),
        mFill(widen(' '))
    {
    }

    explicit IOSBasicFlags(const std::basic_ios<CharT, Traits>& ios) :
        IOSBaseFlags(ios),
        mFill(ios.fill())
    {
    }

    void imbue(std::basic_ios<CharT, Traits>& ios) const
    {
        IOSBaseFlags::imbue(ios);
        ios.fill(mFill);
    }

    void imbue(std::locale mylocale) 
    {
        IOSBaseFlags::imbue(mylocale);
    }

    CharT fill() const
    {
        return mFill;
    }

    CharT fill(CharT f)
    {
        const char old = mFill;
        mFill = f;
        return old;
    }

private:
    CharT widen(CharT c) const
    {
        return std::use_facet<std::ctype<CharT>>(getloc()).widen(c);
    }

    CharT mFill;
};

template <class CharT, class Traits>
class IOSBasicFlagsRAII
{
    std::basic_ios<CharT, Traits>& mIOS;
    IOSBasicFlags<CharT, Traits> mFlags;

public:
    explicit IOSBasicFlagsRAII(std::basic_ios<CharT, Traits>& ios) :
        mIOS(ios),
        mFlags(ios)
    {
    }

    ~IOSBasicFlagsRAII()
    {
        mFlags.imbue(mIOS);
    }
};

using IOSFlags      = IOSBasicFlags<char,    std::char_traits<char>>;
using wIOSFlags     = IOSBasicFlags<wchar_t, std::char_traits<wchar_t>>;
using IOSFlagsRAII  = IOSBasicFlagsRAII<char,    std::char_traits<char>>;
using wIOSFlagsRAII = IOSBasicFlagsRAII<wchar_t, std::char_traits<wchar_t>>;

template <std::size_t N>
using FormatList = std::array<IOSFlags, N>;

} //namespace moonray_stats

