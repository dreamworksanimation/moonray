// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "McrtRtDispatchComputationDebugConsole.h"
#include "McrtRtDispatchComputation.h"

#include <scene_rdl2/common/log/logging.h>

#include <sstream>
#include <unistd.h>             // gethostname()

namespace moonray {
namespace mcrt_rt_dispatch_computation {

int
McrtRtDispatchComputationDebugConsole::open(const int portNumber,
                                            McrtRtDispatchComputation *dispatchComputation)
{
    mDispatchComputation = dispatchComputation;

    char hostname[128];
    gethostname(hostname, 128);

    int port = mTlSvr.open(portNumber);
    MOONRAY_LOG_INFO("DebugConsole host:%s port:%d\n", hostname, port);

    return port;
}

bool
McrtRtDispatchComputationDebugConsole::eval()
{
    if (!mDispatchComputation) return false;

    std::string cmdLine;

    int rt = mTlSvr.recv(cmdLine, 0.0f);
    if (rt < 0) {
        MOONRAY_LOG_ERROR("TlSvr error\n");
        return false;
    } else if (rt == 0) {
        // empty console input
    } else {
        MOONRAY_LOG_INFO("DebugConsole:%s\n", cmdLine.c_str());
        cmdParse(cmdLine);
    }

    return true;
}

void
McrtRtDispatchComputationDebugConsole::close()
{
    mTlSvr.close();
    mDispatchComputation = 0x0;
}

void    
McrtRtDispatchComputationDebugConsole::cmdHelp()
{
    std::ostringstream ostr;
    ostr << "help {\n"
         << "  fps <val>          : set fps rate\n"
         << "  help               : show this message\n"
         << "  mocap <on|off>     : mocap mode on or off\n"
         << "  geoUpdate <on|off> : geometry message send condition on or off\n"
         << "  show               : show internal info\n"
         << "}\n";

    mTlSvr.send(ostr.str());
}

void
McrtRtDispatchComputationDebugConsole::cmdParse(const std::string &cmdLine)
{
    if (cmdCmp("fps", cmdLine)) {
        cmdFps(cmdLine);
    } else if (cmdCmp("help", cmdLine)) {
        cmdHelp();
    } else if (cmdCmp("mocap", cmdLine)) {
        cmdMocap(cmdLine);
    } else if (cmdCmp("geoUpdate", cmdLine)) {
        cmdGeoUpdate(cmdLine);
    } else if (cmdCmp("show", cmdLine)) {
        cmdShow();
        return;
    } else {
        std::ostringstream ostr;
        ostr << "> unknown command>" << cmdLine;
        mTlSvr.send(ostr.str());
    }
}

void
McrtRtDispatchComputationDebugConsole::cmdFps(const std::string &cmdLine)
{
    std::istringstream istr(cmdLine);
    std::string token;

    float fps;

    istr >> token;              // skip "fps"
    istr >> token;
    std::istringstream(token) >> fps;

    mDispatchComputation->setFps(fps);

    std::ostringstream ostr;
    ostr << "> update fps rate ... " << fps << std::endl;
    mTlSvr.send(ostr.str());
}

void
McrtRtDispatchComputationDebugConsole::cmdMocap(const std::string &cmdLine)
{
    std::istringstream istr(cmdLine);
    std::string token;

    istr >> token;              // skip "app"
    istr >> token;

    std::ostringstream ostr;
    bool sw;
    if (token == "on") {
        ostr << "> mocap on\n";
        sw = true;
    } else {
        ostr << "> mocap off\n";
        sw = false;
    }
    mDispatchComputation->setMotionCaptureMode(sw);
    mTlSvr.send(ostr.str());
}

void
McrtRtDispatchComputationDebugConsole::cmdGeoUpdate(const std::string &cmdLine)
{
    std::istringstream istr(cmdLine);
    std::string token;

    istr >> token;              // skip "app"
    istr >> token;

    std::ostringstream ostr;
    bool sw;
    if (token == "on") {
        ostr << "> geoUpdate on\n";
        sw = true;
    } else {
        ostr << "> geoUpdate off\n";
        sw = false;
    }
    mDispatchComputation->setGeoUpdateMode(sw);
    mTlSvr.send(ostr.str());
}

void
McrtRtDispatchComputationDebugConsole::cmdShow()
{
    std::ostringstream ostr;
    ostr << "status {\n"
         << "        fps:" << mDispatchComputation->getFps() << std::endl
         << "      mocap:" << ((mDispatchComputation->getMotionCaptureMode())? "true": "false") << std::endl
         << "  geoUpdate:" << ((mDispatchComputation->getGeoUpdateMode())? "true": "false") << std::endl 
         << "}\n";
    mTlSvr.send(ostr.str());
}

bool
McrtRtDispatchComputationDebugConsole::cmdCmp(const std::string& cmdName, const std::string& cmdLine) const
{
    size_t lenCmdName = cmdName.length();
    size_t lenCmdLine = cmdLine.length();
    if ( lenCmdName > lenCmdLine ) {
        return false;
    }

    std::string ccmd = cmdLine.substr(0, lenCmdName);
    if (cmdName == ccmd) {
        return true;
    }
    return false;
}

} // namespace mcrt_rt_dispatch_computation {
} // namespace moonray

