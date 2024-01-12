// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#include "SocketStream.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace moonray {
namespace stats {

ListeningSocket::ListeningSocket(int port) :
    mFD(-1)
{
    addrinfo hints;
    memset(&hints, 0, sizeof(addrinfo));
    hints.ai_family    = AF_UNSPEC;  // Allow IPv4 or IPv6
    hints.ai_socktype  = SOCK_STREAM;
    hints.ai_flags     = AI_PASSIVE; // For wildcard IP address
    hints.ai_protocol  = 0;          // Any protocol
    hints.ai_canonname = nullptr;
    hints.ai_addr      = nullptr;
    hints.ai_next      = nullptr;

    constexpr int portdigits = std::numeric_limits<decltype(port)>::digits10;
    char portstring[portdigits + 1];
    snprintf(portstring, portdigits + 1, "%i", port);

    addrinfo* result;
    const int gairesult = getaddrinfo(nullptr, portstring, &hints, &result);
    if (gairesult != 0) {
        throw SocketException(std::string("Unable to get server information: ") +
                              gai_strerror(gairesult));
    }

    addrinfo* rp;
    int sfd = -1;
    for (rp = result; rp != nullptr; rp = rp->ai_next) {
        sfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sfd < 0) {
            continue;
        }

        if (bind(sfd, rp->ai_addr, rp->ai_addrlen) == 0) {
            // success!
            break;
        }

        close(sfd);
    }

    freeaddrinfo(result);

    if (rp == nullptr) {
        throw SocketException("Could not bind");
    }

    mFD = sfd;

    if (listen(mFD, 5) < 0) {
        throw SocketException("Error on listen: " + std::string(strerror(errno)));
    }
}

ListeningSocket::ListeningSocket() :
    ListeningSocket(0)
{
}

ListeningSocket::ListeningSocket(ListeningSocket&& other) noexcept :
    mFD(other.mFD)
{
    other.mFD = -1;
}

ListeningSocket& ListeningSocket::operator=(ListeningSocket&& other) noexcept
{
    this->swap(other);
    return *this;
}

ListeningSocket::~ListeningSocket()
{
    if (mFD >= 0) {
        close(mFD);
    }
}

ListeningSocket::operator bool() const noexcept
{
    return mFD >= 0;
}

int ListeningSocket::getPortNumber() const
{
    sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    std::memset(&addr, 0, addrlen);
    if (getsockname(mFD, reinterpret_cast<sockaddr*>(&addr), &addrlen) < 0) {
        throw SocketException("Error in getsockname");
    }
    return ntohs(addr.sin_port);
}

void ListeningSocket::swap(ListeningSocket& other) noexcept
{
    std::swap(other.mFD, mFD);
}

Socket::Socket() noexcept :
    mFD(-1)
{
}

Socket::~Socket()
{
    if (mFD >= 0) {
        close(mFD);
    }
}

Socket Socket::as_client(const std::string& hostname, int port)
{
    addrinfo hints;
    memset(&hints, 0, sizeof(addrinfo));
    hints.ai_family   = AF_UNSPEC;  // Allow IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = 0;
    hints.ai_protocol = 0;          // Any protocol

    constexpr int portdigits = std::numeric_limits<decltype(port)>::digits10;
    char portstring[portdigits + 1];
    snprintf(portstring, portdigits + 1, "%i", port);

    addrinfo* result;
    const int gairesult = getaddrinfo(hostname.c_str(), portstring, &hints, &result);
    if (gairesult != 0) {
        throw SocketException(std::string("Unable to get host address: ") +
                              gai_strerror(gairesult));
    }

    addrinfo* rp;
    int sfd = -1;
    for (rp = result; rp != nullptr; rp = rp->ai_next) {
        sfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sfd < 0) {
            continue;
        }

        if (connect(sfd, rp->ai_addr, rp->ai_addrlen) != -1) {
            // success!
            break;
        }

        close(sfd);
    }

    freeaddrinfo(result);

    if (rp == nullptr) {
        throw SocketException("Could not connect to host");
    }

    Socket ret;
    ret.mFD = sfd;
    return ret;
}

Socket Socket::as_server(const ListeningSocket& ls)
{
    Socket ret;
    sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    ret.mFD = accept(ls.mFD, reinterpret_cast<sockaddr*>(&cli_addr), &clilen);
    if (ret.mFD < 0) {
        throw SocketException("Error on accept: " + std::string(strerror(errno)));
    }
    return ret;
}

Socket::Socket(Socket&& other) noexcept :
    mFD(other.mFD)
{
    other.mFD = -1;
}

Socket& Socket::operator=(Socket&& other) noexcept
{
    this->swap(other);
    return *this;
}

Socket::operator bool() const noexcept
{
    return mFD >= 0;
}

ssize_t Socket::write(const char* buf, ssize_t count)
{
    ssize_t n = 0;
    while (n < count) {
        const ssize_t ret = ::write(mFD, buf + n, count - n);
        if (ret < 0) {
            if (errno == EINTR) {
                // Signal interruption; try again.
                continue;
            } else if (errno == EWOULDBLOCK || errno == EAGAIN) {
                // Blocking socket. Wait for it to free up.
                fd_set writefds;
                FD_ZERO(&writefds);
                FD_SET(mFD, &writefds);

                // TODO: set timeout? What do we do if we timeout?
                if (select(mFD + 1, nullptr, &writefds, nullptr, nullptr) < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    return n;
                }
                assert(FD_ISSET(mFD, &writefds));
            } else {
                return n;
            }
        } else {
            n += ret;
        }
    }
    return n;
}

ssize_t Socket::read(char* buf, ssize_t count)
{
    ssize_t rc = 0;
    for ( ; ; ) {
        const ssize_t ret = ::read(mFD, buf + rc, count - rc);
        if (ret >= 0) {
            return ret;
        } else {
            if (errno == EINTR) {
                // Signal interruption; try again.
                continue;
            } else if (errno == EWOULDBLOCK || errno == EAGAIN) {
                // Blocking socket. Wait for it to free up.
                fd_set readfds;
                FD_ZERO(&readfds);
                FD_SET(mFD, &readfds);

                // TODO: set timeout? What do we do if we timeout?
                if (select(mFD + 1, &readfds, nullptr, nullptr, nullptr) < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    return rc;
                }
                assert(FD_ISSET(mFD, &readfds));
            } else {
                return rc;
            }
        }
    }
}

void Socket::swap(Socket& other) noexcept
{
    std::swap(other.mFD, mFD);
}

SocketStreambuf::SocketStreambuf() :
    SocketStreambuf(Socket())
{
}

SocketStreambuf::SocketStreambuf(Socket&& socket) :
    mSock(std::move(socket))
{
#if 0 // Ugh. We need C++14 libraries for this to work.
    static_assert(sPutBack < mInputBuffer.size(),
                  "We need enough size for putback and writing.");
#endif

    char *const outBase = mOutputBuffer.data();
    setp(outBase, outBase + mOutputBuffer.size());

    char *const inEnd = mInputBuffer.data() + mInputBuffer.size();
    setg(inEnd, inEnd, inEnd);
}

SocketStreambuf::~SocketStreambuf()
{
    pubsync();
}

bool SocketStreambuf::is_open() const noexcept
{
    return static_cast<bool>(mSock);
}

void SocketStreambuf::open(const std::string& filename, int port)
{
    mSock = Socket::as_client(filename, port);
}

bool SocketStreambuf::write_to_socket()
{
    const std::ptrdiff_t count = pptr() - pbase();
    const ssize_t n = mSock.write(pbase(), count);
    if (n != count) {
        return false;
    }
    pbump(-n);
    return true;
}

SocketStreambuf::int_type SocketStreambuf::overflow(int_type ch)
{
    if (mSock && !traits_type::eq_int_type(ch, traits_type::eof())) {
        if (!write_to_socket()) {
            return traits_type::eof();
        }
        assert(std::less<char*>()(pptr(), epptr()));
        traits_type::assign(*pptr(), traits_type::to_char_type(ch));
        pbump(1);
        return traits_type::not_eof(ch);
    }
    return traits_type::eof();
}

int SocketStreambuf::sync()
{
    return write_to_socket() ? 0 : 1;
}

SocketStreambuf::int_type SocketStreambuf::underflow()
{
    if (gptr() < egptr()) { // Buffer not exhausted.
        return traits_type::not_eof(*gptr());
    }

    if (!mSock) {
        return traits_type::eof();
    }

    char* const base = mInputBuffer.data();
    char* start = base;

    if (eback() == base) { // Not our first fill
        // Move the putback characters to the beginning of the buffer.
        std::memmove(base, egptr() - sPutBack, sPutBack);
        start += sPutBack;
    }

    const std::size_t count = mInputBuffer.size() - (start - base);
    const ssize_t rc = mSock.read(start, count);
    if (rc <= 0) {
        return traits_type::eof();
    }
    setg(base, start, start + rc);
    return traits_type::not_eof(*gptr());
}

} // namespace stats
} // namespace moonray

