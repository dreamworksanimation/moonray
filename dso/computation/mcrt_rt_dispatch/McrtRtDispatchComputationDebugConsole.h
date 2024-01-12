// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/dataio/telnet_server/TlSvr.h>

namespace scene_rdl2 {
namespace mcrt_rt_dispatch_computation {

class McrtRtDispatchComputation;

class McrtRtDispatchComputationDebugConsole {
public:
    McrtRtDispatchComputationDebugConsole() : mDispatchComputation(0x0) {}
    ~McrtRtDispatchComputationDebugConsole() { close(); }

    int open(const int portNumber, McrtRtDispatchComputation *dispatchComputation);

    bool eval();

    void close();

protected:

    void cmdParse(const std::string &cmdLine);
    void cmdFps(const std::string &cmdLine);
    void cmdHelp();
    void cmdMocap(const std::string &cmdLine);
    void cmdGeoUpdate(const std::string &cmdLine);
    void cmdShow();
    bool cmdCmp(const std::string& cmdName, const std::string& cmdLine) const;

    McrtRtDispatchComputation *mDispatchComputation;
    scene_rdl2::telnet_server::TlSvr mTlSvr;
};

} // namespace mcrt_rt_dispatch_computation {
} // namespace scene_rdl2

