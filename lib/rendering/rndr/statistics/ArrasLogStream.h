// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include <scene_rdl2/render/logging/logging.h>

#include <array>
#include <ostream>
#include <streambuf>

namespace moonray {
namespace stats {

// A buffered stream buffer that writes to the arras logging framework.
class LogStreamBuf : public std::streambuf
{
public:
    inline explicit LogStreamBuf(scene_rdl2::logging::LogLevel level);
    inline LogStreamBuf(const LogStreamBuf&) = delete;
    inline LogStreamBuf& operator=(const LogStreamBuf&) = delete;
    inline ~LogStreamBuf();

    inline int getPriority() const;
    inline int setPriority(scene_rdl2::logging::LogLevel priority);

protected:
    inline bool doFlush();
    inline int_type overflow(int_type c) override;
    inline int sync() override;

private:
    virtual void doLog() const = 0;

    std::array<char, 512> mBuffer;
    scene_rdl2::logging::LogLevel mPriority;
};

LogStreamBuf::LogStreamBuf(scene_rdl2::logging::LogLevel level) :
    mPriority(level)
{
    char* const base = mBuffer.data();

    // -1 to make room for terminating '/0'
    setp(base, base + mBuffer.size() - 1u);
}

LogStreamBuf::~LogStreamBuf()
{
}

scene_rdl2::logging::LogLevel LogStreamBuf::getPriority() const
{
    return mPriority;
}

scene_rdl2::logging::LogLevel LogStreamBuf::setPriority(scene_rdl2::logging::LogLevel priority)
{
    const scene_rdl2::logging::LogLevel old = mPriority;
    sync();
    mPriority = priority;
    return old;
}

int LogStreamBuf::sync()
{
    return doFlush() ? 0 : -1;
}

LogStreamBuf::int_type LogStreamBuf::overflow(int_type ch)
{
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
        MNRY_ASSERT(std::less_equal<char*>()(pptr(), epptr()));
        if (doFlush()) {
            traits_type::assign(*pptr(), traits_type::to_char_type(ch));
            pbump(1);
            return traits_type::not_eof(ch);
        }
    }

    return traits_type::eof();
}

bool LogStreamBuf::doFlush()
{
    const std::ptrdiff_t n = pptr() - pbase();
    MNRY_ASSERT(n < static_cast<std::ptrdiff_t>(mBuffer.size()));
    pbump(static_cast<int>(-n));
    mBuffer[n] = '\0';

    doLog();
    return true;
}

class ArrasLogStreamBuf : public LogStreamBuf
{
public:
    explicit ArrasLogStreamBuf(scene_rdl2::logging::LogLevel level = scene_rdl2::logging::INFO_LEVEL) :
        LogStreamBuf(level)
    {
    }

    ~ArrasLogStreamBuf()
    {
        pubsync();
    }

private:
    void doLog() const override
    {
        scene_rdl2::logging::Logger::log(getPriority(), pbase());
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Unfortunately, every time we send our output to the logging system, a newline
// is appended. This class removes the newlines from the input before passing
// the character off to the buffered stream buffer. If we encounter a newline,
// immediately flush, hopefully leaving enough room in the buffer for the next
// newline delimited string.
template <typename StreamBuf>
class NewLineLogStreamBuf : public std::streambuf
{
public:
    inline explicit NewLineLogStreamBuf(scene_rdl2::logging::LogLevel level = scene_rdl2::logging::INFO_LEVEL);
    inline NewLineLogStreamBuf(const NewLineLogStreamBuf&) = delete;
    inline NewLineLogStreamBuf& operator=(const NewLineLogStreamBuf&) = delete;
    inline ~NewLineLogStreamBuf();

    inline int getPriority() const;
    inline int setPriority(scene_rdl2::logging::LogLevel priority);

private:
    int_type overflow(int_type c) override;
    StreamBuf mBuf;
};

template <typename StreamBuf>
NewLineLogStreamBuf<StreamBuf>::NewLineLogStreamBuf(scene_rdl2::logging::LogLevel level) :
    mBuf(level)
{
}

template <typename StreamBuf>
NewLineLogStreamBuf<StreamBuf>::~NewLineLogStreamBuf()
{
    mBuf.pubsync();
    pubsync();
}

template <typename StreamBuf>
scene_rdl2::logging::LogLevel NewLineLogStreamBuf<StreamBuf>::getPriority() const
{
    return mBuf.getPriority();
}

template <typename StreamBuf>
scene_rdl2::logging::LogLevel NewLineLogStreamBuf<StreamBuf>::setPriority(scene_rdl2::logging::LogLevel priority)
{
    return mBuf.setPriority(priority);
}

template <typename StreamBuf>
typename NewLineLogStreamBuf<StreamBuf>::int_type NewLineLogStreamBuf<StreamBuf>::overflow(int_type ch)
{
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
        if (ch == '\n') {
            mBuf.pubsync();
        } else {
            return mBuf.sputc(traits_type::to_char_type(ch));
        }
        return traits_type::not_eof(ch);
    }

    return traits_type::eof();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename StreamBuf>
class LogStream : public std::ostream
{
public:
    LogStream() :
        std::ostream(nullptr),
        mBuf()
    {
    }

    void open(scene_rdl2::logging::LogLevel level = scene_rdl2::logging::INFO_LEVEL)
    {
        try {
            mBuf.reset(new NewLineLogStreamBuf<StreamBuf>(level));
            this->init(mBuf.get());
            this->clear();
        } catch (const std::exception&) {
            this->setstate(std::ios_base::failbit);
        } catch (...) {
            this->setstate(std::ios_base::failbit);
        }
    }

    bool is_open() const noexcept
    {
        return static_cast<bool>(mBuf);
    }

    scene_rdl2::logging::LogLevel setPriority(scene_rdl2::logging::LogLevel priority)
    {
        if (mBuf) {
            mBuf.setPriority(priority);
        }
        return priority;
    }

private:
    std::unique_ptr<NewLineLogStreamBuf<StreamBuf>> mBuf;
};

typedef LogStream<ArrasLogStreamBuf> ArrasLogStream;

} // namespace stats
} // namespace moonray

