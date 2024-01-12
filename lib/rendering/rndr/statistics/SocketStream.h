// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include <array>
#include <chrono>
#include <iosfwd>
#include <istream>
#include <stdexcept>
#include <streambuf>
#include <vector>
#include <netinet/in.h>

namespace moonray {
namespace stats {

class SocketException : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class ListeningSocket
{
public:
    friend class Socket;

    explicit ListeningSocket(int port);
    ListeningSocket(); // Automatic port

    ListeningSocket(const ListeningSocket&) = delete;
    ListeningSocket(ListeningSocket&& other) noexcept;
    ListeningSocket& operator=(const ListeningSocket&) = delete;
    ListeningSocket& operator=(ListeningSocket&& other) noexcept;

    ~ListeningSocket();

    explicit operator bool() const noexcept;
    int getPortNumber() const;

    void swap(ListeningSocket& other) noexcept;

private:
    int mFD;
};

class Socket
{
public:
    // These calls will block as necessary.
    static Socket as_server(const ListeningSocket& ls);
    static Socket as_client(const std::string& hostname, int port);

    Socket() noexcept;
    Socket(const Socket&) = delete;
    Socket(Socket&& other) noexcept;
    Socket& operator=(const Socket&) = delete;
    Socket& operator=(Socket&& other) noexcept;

    ~Socket();

    explicit operator bool() const noexcept;

    ssize_t write(const char* buf, ssize_t count);
    ssize_t read(char* buf, ssize_t count);

    void swap(Socket& other) noexcept;

private:
    int mFD;
};

class SocketStreambuf : public std::streambuf
{
public:
    SocketStreambuf();
    explicit SocketStreambuf(Socket&& socket);

    SocketStreambuf(const SocketStreambuf&) = delete;
    SocketStreambuf operator=(const SocketStreambuf&) = delete;
    ~SocketStreambuf();

    bool is_open() const noexcept;
    void open(const std::string& filename, int port);

private:
    bool write_to_socket();
    int_type overflow(int_type ch) override;
    int sync() override;
    int_type underflow() override;

    static constexpr std::size_t sPutBack = 8;
    std::array<char, 2048> mOutputBuffer;
    std::array<char, 2048> mInputBuffer;

    Socket mSock;
};

class SocketStream : public std::iostream
{
public:
    explicit SocketStream(Socket&& socket) :
            std::iostream(nullptr),
            mBuf(std::move(socket))
    {
        this->init(&mBuf);
    }

    bool is_open() const noexcept
    {
        return mBuf.is_open();
    }

private:
    SocketStreambuf mBuf;
};

} // namespace stats
} // namespace moonray

