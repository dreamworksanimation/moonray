// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "AthenaCSVStream.h"
#include <moonray/rendering/rndr/Error.h>
#include <scene_rdl2/render/util/GetEnv.h>

#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <climits> // HOST_NAME_MAX
#include <cstring> //strerror_r
#include <ctime> //strftime & localtime_r
#include <iostream>
#include <netdb.h> //getservbyname
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h> //gethostname

namespace {
constexpr unsigned int LOG_USER = 1;
constexpr unsigned int LOG_INFO = 6;
constexpr unsigned int SYSLOG_PRIO = (LOG_USER << 3) | LOG_INFO;
}

namespace moonray {
namespace stats {

AthenaCSVStreamBuf::AthenaCSVStreamBuf(const scene_rdl2::util::GUID& guid, bool debug) :
        mGUID(guid.asString()),
        mHostname(getLocalHostName()),
        mStorage(),
        mDwaGacId{scene_rdl2::util::getenv<std::string>("DWA_GACID")},
        mSocket{-1},
        mOpened(false)
{
    open(debug);
}

AthenaCSVStreamBuf::~AthenaCSVStreamBuf()
{
    if (mOpened) {
        ::shutdown(mSocket, SHUT_RDWR);
    }
}

void
AthenaCSVStreamBuf::open(bool debug)
{
    // From the Athena wiki:
    // The "athena." identifier followed by your producer or application name,
    // e.g.: "athena.moonray.stats"
    const char* athena_development = scene_rdl2::util::getenv<const char*>("MNRY_ATHENA_STATS_DEV");
    if (debug || athena_development != nullptr) {
        mIdent = std::string("dev-athena.moonray_stats");
    } else {
        mIdent = std::string("prod-athena.moonray_stats");
    }

    const auto serviceEntry = ::getservbyname("syslog", "udp");
    ::endservent();

    if (serviceEntry == nullptr) {
        std::cerr << "Failed to lookup syslog from services database" << std::endl;
        return;
    }

    mSocket = ::socket(AF_INET, SOCK_DGRAM|SOCK_NONBLOCK, 0);
    if (mSocket == -1) {
        std::cerr << "Failed to create socket due to: " << rndr::getErrorDescription() << std::endl;
        return;
    }

    struct sockaddr_in addr;
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    addr.sin_family = AF_INET;
    addr.sin_port = htons(serviceEntry->s_port);
    if (::connect(mSocket, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        mOpened = true;
    } else {
        std::cerr << "Failed to connect to socket due to: " << rndr::getErrorDescription() << std::endl;
    }
}

std::streambuf::int_type 
AthenaCSVStreamBuf::overflow(int_type ch)
{
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
        if (ch != traits_type::to_int_type('\n')) {
            mStorage.push_back(traits_type::to_char_type(ch));
        } else {
            if (!mStorage.empty()) {
                appendBuffer(mHostname.c_str(), mHostname.length());
                appendBuffer(',');
                appendBuffer(mGUID.c_str(), mGUID.length());
                appendBuffer(',');
                appendTimestamp();
                appendBuffer(',');
                appendBuffer(mStorage.data(), mStorage.size());
                appendBuffer('\0');  // string must be null terminated
                flushBuffer();
                mStorage.clear();
            }
        }
        return traits_type::not_eof(ch);
    }
    return traits_type::eof();
}

std::chrono::milliseconds::rep 
AthenaCSVStreamBuf::msSinceEpoch()
{
    const auto p = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(p.time_since_epoch()).count();
}

void
AthenaCSVStreamBuf::appendTimestamp()
{
    // digits10 is rounded down, because it represents the
    // number of digits that can be stored without loss of
    // information, so add one.
    constexpr auto digits = std::numeric_limits<MSType>::digits10 + 1;

    // Add one for terminator.
    char timestampString[digits + 1] = { 0 };

    const auto ms = msSinceEpoch();
    static_assert(sizeof(MSType) == sizeof(long),
                    "We're using the long conversion for sprintf");
    std::snprintf(timestampString, digits + 1, "%ld", ms);
    appendBuffer(timestampString, std::strlen(timestampString));
}

void
AthenaCSVStreamBuf::flushBuffer()
{
    if (mSyslogBuffer.size() <= 0) return;

    // TIMESTAMP
    char timestamp[256];
    time_t now = time(0);
    struct tm tmnow;
    localtime_r(&now, &tmnow);
    // Note timestamp must be in this old format, ISO standard format isn't accepted
    //rfc5424: strftime(timestamp, 256, "%FT%T%z", &tmnow);
    strftime(timestamp, 256, "%b %e %H:%M:%S", &tmnow);
    
    std::ostringstream msg;
    msg << "<" << SYSLOG_PRIO << "> "
        << timestamp << " "
        << mHostname << " "
        << mIdent << ": "
        << mDwaGacId << "," << mSyslogBuffer.data();

    std::string s = msg.str();
    if (::send(mSocket, s.c_str(), s.size(), 0) == -1) {
        std::cerr << "Failed to send log message to Athena due to: "
                  << rndr::getErrorDescription()
                  << " : " << s << std::endl;
    }

    mSyslogBuffer.clear();
}

std::string 
AthenaCSVStreamBuf::getLocalHostName()
{
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    return std::string(hostname);
}

} // namespace stats
} // namespace moonray

