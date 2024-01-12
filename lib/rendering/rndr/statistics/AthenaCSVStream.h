// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include <scene_rdl2/render/util/GUID.h>

#include <chrono>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

namespace moonray {
namespace stats {

class AthenaCSVStreamBuf : public std::streambuf
{
public:
    explicit AthenaCSVStreamBuf(const scene_rdl2::util::GUID& guid, bool debug = false);

    AthenaCSVStreamBuf() :
        AthenaCSVStreamBuf(scene_rdl2::util::GUID::uuid4(), false)
    {
    }

    explicit AthenaCSVStreamBuf(bool debug) :
        AthenaCSVStreamBuf(scene_rdl2::util::GUID::uuid4(), debug)
    {
    }

    ~AthenaCSVStreamBuf();

    void open(bool debug);

    bool is_open() const noexcept
    {
        return mOpened;
    }

private:
    using MSType = std::chrono::milliseconds::rep;
    int_type overflow(int_type ch) override;
    static MSType msSinceEpoch();
    void appendTimestamp();

    void appendBuffer(char c)
    {
        mSyslogBuffer.push_back(c);
    }

    void appendBuffer(const char* s, size_t size)
    {
        mSyslogBuffer.insert(mSyslogBuffer.end(), s, s + size);
    }

    void flushBuffer();

    static std::string getLocalHostName();

    const std::string mGUID;
    const std::string mHostname;
    std::vector<char> mStorage;
    std::vector<char> mSyslogBuffer;

    // MOONRAY-1648 Update RSyslog data to include a global correlation ID
    // This unique ID precedes all syslog communication so as to reassemble all
    // the asynchronous data it is receiving simultaneously from varying renders.
    std::string mDwaGacId;

    int mSocket;
    std::string mIdent;
    bool mOpened;
};

class AthenaCSVStream : public std::ostream
{
public:
    AthenaCSVStream() :
        std::ostream(nullptr),
        mBuf()
    {
        mOpened = false;
    }

    void open(bool debug)
    {
        do_open(debug);
    }

    void open(const scene_rdl2::util::GUID& guid, bool debug)
    {
        do_open(guid, debug);
    }

    bool is_open() const noexcept
    {
        // open attempted yet?
        return mOpened && mBuf->is_open();
    }
private:
    template <typename... Args>
    void do_open(Args&&... args)
    {
        try {
            mBuf.reset(new AthenaCSVStreamBuf(std::forward<Args>(args)...));
            this->init(mBuf.get());
            this->clear();
            mOpened = true;
        } catch (const std::exception&) {
            this->setstate(std::ios_base::failbit);
            mOpened = false;
        } catch (...) {
            this->setstate(std::ios_base::failbit);
            mOpened = false;
        }
    }
    
    std::unique_ptr<AthenaCSVStreamBuf> mBuf;
    bool mOpened;
};


} // namespace stats
} // namespace moonray

