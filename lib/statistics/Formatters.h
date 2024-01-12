// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// Created by kjeffery on 5/10/16.
//

#pragma once

#include "IOSFlags.h"
#include "Util.h"

#include <chrono>
#include <iomanip>
#include <ios>
#include <iterator>
#include <memory>
#include <ostream>
#include <ratio>
#include <sstream>
#include <type_traits>
#include <vector>

#include <cmath>

namespace moonray_stats {

inline std::locale getLocale()
{
    try {
        return std::locale("en_US.UTF-8");
    } catch (...) {
        return std::locale("");
    }
}

class DimensionlessTypeErasureConcept
{
public:
    virtual ~DimensionlessTypeErasureConcept() = default;
    virtual std::ostream& print(std::ostream& outs) const = 0;
};

template <typename T>
class DimensionlessTypeErasure : public DimensionlessTypeErasureConcept
{
public:
    explicit DimensionlessTypeErasure(T t) : mT(t) {}

    std::ostream& print(std::ostream& outs) const override
    {
        return outs << mT;
    }

private:
    T mT;
};

class ListTypeErasureConcept
{
public:
    virtual ~ListTypeErasureConcept() = default;
    virtual std::ostream& print(std::ostream& outs, const char* sep) const = 0;
};

template <typename T>
class ListTypeErasure : public ListTypeErasureConcept
{
public:
    template<typename FwdIter>
    ListTypeErasure(FwdIter first, FwdIter last) :
        mList(first, last)
    {
    }

    std::ostream& print(std::ostream& outs, const char* sep) const override
    {
        if (!mList.empty()) {
            auto it = mList.cbegin();
            outs << *it;
            for (++it; it != mList.cend(); ++it) {
                outs << sep;
                outs << *it;
            }
        }
        return outs;

    }

private:
    std::vector<T> mList;
};

class Dimensionless;
class Bytes;
class BytesPerSecond;
class Percentage;
class Time;
class Seconds;
class List;

class Formatter
{
public:
    virtual ~Formatter() = default;

    std::ostream& visitWrite(std::ostream& outs, const Dimensionless& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    std::ostream& visitWrite(std::ostream& outs, const Bytes& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    std::ostream& visitWrite(std::ostream& outs, const BytesPerSecond& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    std::ostream& visitWrite(std::ostream& outs, const Percentage& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    std::ostream& visitWrite(std::ostream& outs, const Time& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    std::ostream& visitWrite(std::ostream& outs, const Seconds& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    std::ostream& visitWrite(std::ostream& outs, const List& t) const
    {
        std::ostringstream oss;

        // Copy the formatting from the stream to the string stream, but with
        // no width. We'll let the ostream worry about that. However, we want to
        // write this value as a single unit to outs (so things like width don't
        // affect it).
        IOSFlags flags(outs);
        flags.width(0);
        flags.imbue(oss);

        visitWriteImpl(oss, t);
        return outs << oss.str();
    }

    const char* visitGetUnit(const Dimensionless& t) const
    {
        return visitGetUnitImpl(t);
    }

    const char* visitGetUnit(const Bytes& t) const
    {
        return visitGetUnitImpl(t);
    }

    const char* visitGetUnit(const BytesPerSecond& t) const
    {
        return visitGetUnitImpl(t);
    }

    const char* visitGetUnit(const Percentage& t) const
    {
        return visitGetUnitImpl(t);
    }

    const char* visitGetUnit(const Time& t) const
    {
        return visitGetUnitImpl(t);
    }

    const char* visitGetUnit(const Seconds& t) const
    {
        return visitGetUnitImpl(t);
    }

    const char* visitGetUnit(const List& t) const
    {
        return visitGetUnitImpl(t);
    }

private:
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const Dimensionless& t) const = 0;
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const Bytes& t) const = 0;
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const BytesPerSecond& t) const = 0;
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const Percentage& t) const = 0;
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const Time& t) const = 0;
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const Seconds& t) const = 0;
    virtual std::ostream& visitWriteImpl(std::ostream& outs, const List& t) const = 0;

    virtual const char* visitGetUnitImpl(const Dimensionless& t) const = 0;
    virtual const char* visitGetUnitImpl(const Bytes& t) const = 0;
    virtual const char* visitGetUnitImpl(const BytesPerSecond& t) const = 0;
    virtual const char* visitGetUnitImpl(const Percentage& t) const = 0;
    virtual const char* visitGetUnitImpl(const Time& t) const = 0;
    virtual const char* visitGetUnitImpl(const Seconds& t) const = 0;
    virtual const char* visitGetUnitImpl(const List& t) const = 0;
};

class Type
{
public:
    virtual ~Type() = default;
    virtual std::ostream& write(std::ostream& outs, const Formatter& formatter) const = 0;
    virtual const char* getUnit(const Formatter& formatter) const = 0;
};

class Dimensionless : public Type
{
public:
    template <typename T>
    explicit Dimensionless(T&& t) :
        mData(moonray_stats::make_unique<
            DimensionlessTypeErasure<
                typename std::decay<T>::type>>(std::forward<T>(t)))
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return formatter.visitWrite(outs, *this);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    std::ostream& print(std::ostream& outs) const
    {
        return mData->print(outs);
    }

private:
    std::unique_ptr<DimensionlessTypeErasureConcept> mData;
};

class Bytes : public Type
{
    template <typename T>
    static constexpr std::uint64_t sizeInBytes()
    {
        return T::num / T::den;
    };

public:
    // Technically, there's no reason for these to be ratios. We could just use
    // constant values.
    using Byte     = std::ratio<1ULL <<  0ULL, 1ULL>;
    using Kilobyte = std::ratio<1ULL << 10ULL, 1ULL>;
    using Megabyte = std::ratio<1ULL << 20ULL, 1ULL>;
    using Gigabyte = std::ratio<1ULL << 30ULL, 1ULL>;
    using Terabyte = std::ratio<1ULL << 40ULL, 1ULL>;
    using Petabyte = std::ratio<1ULL << 50ULL, 1ULL>;
    using Exabyte  = std::ratio<1ULL << 60ULL, 1ULL>;

    explicit Bytes(std::uint64_t b) :
        mBytes(b)
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return formatter.visitWrite(outs, *this);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    const char* getAutoUnit() const
    {
        if (mBytes >= sizeInBytes<Exabyte>()) {
            return "EB";
        } else if (mBytes >= sizeInBytes<Petabyte>()) {
            return "PB";
        } else if (mBytes >= sizeInBytes<Terabyte>()) {
            return "TB";
        } else if (mBytes >= sizeInBytes<Gigabyte>()) {
            return "GB";
        } else if (mBytes >= sizeInBytes<Megabyte>()) {
            return "MB";
        } else if (mBytes >= sizeInBytes<Kilobyte>()) {
            return "KB";
        } else {
            return "B";
        }
    }

    double autoConvert() const
    {
        const double dbytes = static_cast<double>(mBytes);
        if (mBytes >= sizeInBytes<Exabyte>()) {
            return dbytes / static_cast<double>(sizeInBytes<Exabyte>());
        } else if (mBytes >= sizeInBytes<Petabyte>()) {
            return dbytes  / static_cast<double>(sizeInBytes<Petabyte>());
        } else if (mBytes >= sizeInBytes<Terabyte>()) {
            return dbytes  / static_cast<double>(sizeInBytes<Terabyte>());
        } else if (mBytes >= sizeInBytes<Gigabyte>()) {
            return dbytes  / static_cast<double>(sizeInBytes<Gigabyte>());
        } else if (mBytes >= sizeInBytes<Megabyte>()) {
            return dbytes  / static_cast<double>(sizeInBytes<Megabyte>());
        } else if (mBytes >= sizeInBytes<Kilobyte>()) {
            return dbytes  / static_cast<double>(sizeInBytes<Kilobyte>());
        } else {
            return dbytes;
        }
    }

    template <typename T>
    double convert() const
    {
        return static_cast<double>(mBytes) / static_cast<double>(sizeInBytes<T>());
    }

private:
    std::uint64_t mBytes;
};

class BytesPerSecond : public Type
{
public:
    explicit BytesPerSecond(std::uint64_t b) :
        mBytesPerSec(b)
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return mBytesPerSec.write(outs, formatter);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    const char* getAutoUnit() const
    {
        std::string s(mBytesPerSec.getAutoUnit());
        s += "/s";
        return s.c_str();
    }

    double autoConvert() const
    {
        return mBytesPerSec.autoConvert();
    }

    template <typename T>
    double convert() const
    {
        return mBytesPerSec.convert<T>();
    }

private:
    Bytes mBytesPerSec;
};

class Percentage : public Type
{
public:
    explicit Percentage(float rawValue) :
        mRawValue(rawValue)
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return formatter.visitWrite(outs, *this);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    float asPercentage() const
    {
        return mRawValue * 100.0f;
    }

private:
    float mRawValue;
};

class Time : public Type
{
public:
    explicit Time(double seconds) :
        mSeconds(seconds)
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return formatter.visitWrite(outs, *this);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    double getSeconds() const
    {
        return mSeconds;
    }

private:
    double mSeconds;
};

class Seconds : public Type
{
public:
    explicit Seconds(double seconds) :
        mSeconds(seconds)
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return formatter.visitWrite(outs, *this);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    double getSeconds() const
    {
        return mSeconds;
    }

private:
    double mSeconds;
};

class List : public Type
{
public:
    template <typename FwdIter>
    List(FwdIter first, FwdIter last) :
        mData(moonray_stats::make_unique<ListTypeErasure<typename std::iterator_traits<FwdIter>::value_type>>(first, last))
    {
    }

    std::ostream& write(std::ostream& outs, const Formatter& formatter) const override
    {
        return formatter.visitWrite(outs, *this);
    }

    const char* getUnit(const Formatter& formatter) const override
    {
        return formatter.visitGetUnit(*this);
    }

    std::ostream& print(std::ostream& outs, const char* sep) const
    {
        return mData->print(outs, sep);
    }

private:
    std::unique_ptr<ListTypeErasureConcept> mData;
};

class FormatterHuman : public Formatter
{
private:
    std::ostream& visitWriteImpl(std::ostream& outs, const Dimensionless& t) const override
    {
        return t.print(outs);
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Bytes& t) const override
    {
        return outs << t.autoConvert() << ' ' << t.getAutoUnit();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const BytesPerSecond& t) const override
    {
        return outs << t.autoConvert() << ' ' << t.getAutoUnit();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Percentage& t) const override
    {
        return outs << t.asPercentage() << '%';
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Time& t) const override
    {
        using namespace std::chrono;

        IOSFlagsRAII raii(outs);

        using FpSeconds = duration<double, seconds::period>;
        const FpSeconds totalSeconds(t.getSeconds());

        // split into hours, minutes, seconds, and milliseconds
        milliseconds totalMS = duration_cast<milliseconds>(totalSeconds);
        hours        hh = duration_cast<hours>(totalMS);
        minutes      mm = duration_cast<minutes>(totalMS % hours(1));
        seconds      ss = duration_cast<seconds>(totalMS % minutes(1));
        milliseconds ms = duration_cast<milliseconds>(totalMS % seconds(1));

        const FpSeconds fss = ss + ms;

        IOSFlags flags(outs);
        flags.fill('0');
        flags.right();
        flags.fixed();
        flags.imbue(outs);

        const auto precision = flags.precision();
        // Like the other values, we want two digits for the integral part of
        // the seconds. If the precision is greater than 0, we have to account
        // for the decimal point.
        const int secWidth = (precision == 0) ? 2 : 3 + static_cast<int>(precision);
        outs << std::setw(2) << hh.count() << ':'
             << std::setw(2) << mm.count() << ':'
             << std::setw(secWidth) << fss.count();

        return outs;
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Seconds& t) const override
    {
        return outs << t.getSeconds() << 's';
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const List& t) const override
    {
        return t.print(outs, " ");
    }

    const char* visitGetUnitImpl(const Dimensionless&) const override
    {
        return "";
    }

    const char* visitGetUnitImpl(const Bytes& t) const override
    {
        return t.getAutoUnit();
    }

    const char* visitGetUnitImpl(const BytesPerSecond& t) const override
    {
        return t.getAutoUnit();
    }

    const char* visitGetUnitImpl(const Percentage&) const override
    {
        return "%";
    }

    const char* visitGetUnitImpl(const Time&) const override
    {
        return "";
    }

    const char* visitGetUnitImpl(const Seconds&) const override
    {
        return "s";
    }

    const char* visitGetUnitImpl(const List&) const override
    {
        return "";
    }
};

class FormatterCSV : public Formatter
{
private:
    std::ostream& visitWriteImpl(std::ostream& outs, const Dimensionless& t) const override
    {
        return t.print(outs);
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Bytes& t) const override
    {
        return outs << t.convert<Bytes::Megabyte>();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const BytesPerSecond& t) const override
    {
        return outs << t.convert<Bytes::Megabyte>();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Percentage& t) const override
    {
        return outs << t.asPercentage();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Time& t) const override
    {
        return outs << t.getSeconds();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const Seconds& t) const override
    {
        return outs << t.getSeconds();
    }

    std::ostream& visitWriteImpl(std::ostream& outs, const List& t) const override
    {
        return t.print(outs, ",");
    }

    const char* visitGetUnitImpl(const Dimensionless&) const override
    {
        return "";
    }

    const char* visitGetUnitImpl(const Bytes&) const override
    {
        return "MB";
    }

    const char* visitGetUnitImpl(const BytesPerSecond&) const override
    {
        return "MB/s";
    }

    const char* visitGetUnitImpl(const Percentage&) const override
    {
        return "%";
    }

    const char* visitGetUnitImpl(const Time&) const override
    {
        return "s";
    }

    const char* visitGetUnitImpl(const Seconds&) const override
    {
        return "s";
    }

    const char* visitGetUnitImpl(const List&) const override
    {
        return "";
    }
};

template <typename T>
struct IsOutputFunctionPtr : public std::integral_constant<bool, false>
{
};

// TODO: This isn't very robust. It's currently actually just a check for any
// unique_ptr. Since we'd be comparing against the base class, the template
// parameter doesn't match. We'd probably need some SFINAE.
template <typename T>
struct IsOutputFunctionPtr<std::unique_ptr<T>>
    : public std::integral_constant<bool, true>
{
};

template <typename T>
inline std::unique_ptr<Dimensionless> dimensionless(T&& t)
{
    return moonray_stats::make_unique<Dimensionless>(std::forward<T>(t));
}

template <typename T>
inline std::unique_ptr<List> list(T&& t)
{
    return moonray_stats::make_unique<List>(std::begin(t), std::end(t));
}

inline std::unique_ptr<Percentage> percentage(float t)
{
    return moonray_stats::make_unique<Percentage>(t);
}

inline std::unique_ptr<Time> time(float t)
{
    return moonray_stats::make_unique<Time>(t);
}

inline std::unique_ptr<Seconds> seconds(float t)
{
    return moonray_stats::make_unique<Seconds>(t);
}

inline std::unique_ptr<Bytes> bytes(std::uint64_t b)
{
    return moonray_stats::make_unique<Bytes>(b);
}

inline std::unique_ptr<BytesPerSecond> bytesPerSecond(std::uint64_t b)
{
    return moonray_stats::make_unique<BytesPerSecond>(b);
}

} // namespace moonray_stats

