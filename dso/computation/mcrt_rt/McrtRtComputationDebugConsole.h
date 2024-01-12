// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <moonray/dataio/telnet_server/TlSvr.h>

namespace moonray {
namespace mcrt_rt_computation {

class McrtRtComputation;

class McrtRtComputationDebugConsole {
public:
    McrtRtComputationDebugConsole() : mComputation(0x0) {}
    ~McrtRtComputationDebugConsole() { close(); }

    int open(const int portNumber, McrtRtComputation *computation);

    bool eval();

    void close();

protected:

    void cmdParse(const std::string &cmdLine);

    void cmdFps(const std::string &cmdLine);
    void cmdGeoUpdate(const std::string &cmdLine);
    void cmdHelp();
    void cmdResetCounter();
    void cmdShow();

    bool cmdCmp(const std::string &cmdName, const std::string &cmdline) const;

    McrtRtComputation *mComputation;
    moonray::telnet_server::TlSvr mTlSvr;
};

} // namespace mcrt_rt_computation
} // namespace moonray

