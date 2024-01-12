// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestSocketStream.h"
#include <moonray/rendering/rndr/statistics/SocketStream.h>

#include <iostream>

namespace moonray {
namespace rndr {
namespace unittest {

void
TestSocketStream::setUp()
{
}

void
TestSocketStream::tearDown()
{
}

void
TestSocketStream::testWrite()
{
    // Setting up server for automatic port.
    stats::ListeningSocket ls;
    const int port = ls.getPortNumber();

    // Setting up client.
    stats::SocketStream client(stats::Socket::as_client("localhost", port));

    // Accepting connections.
    stats::SocketStream server(stats::Socket::as_server(ls));

    std::string line;

    // We use std::endl here to ensure we're flushing the streams, otherwise we never send over the connection. :(
    client << "Line " << 1 << std::endl;
    std::getline(server, line);
    CPPUNIT_ASSERT(line == "Line 1");
    server << "Line " << 2 << std::endl;
    std::getline(client, line);
    CPPUNIT_ASSERT(line == "Line 2");
    client << "Line " << 3 << std::endl;
    std::getline(server, line);
    CPPUNIT_ASSERT(line == "Line 3");
}

} // namespace unittest
} // namespace rndr
} // namespace moonray

