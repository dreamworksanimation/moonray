// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "McrtRtComputationDebugConsole.h"
#include "McrtRtComputation.h"

#include <sstream>
#include <unistd.h>             // gethostname()

namespace moonray {
namespace mcrt_rt_computation {

int
McrtRtComputationDebugConsole::open(const int portNumber,
                                    McrtRtComputation *computation)
{
    mComputation = computation;

    char hostname[128];
    gethostname(hostname, 128);

    int port = mTlSvr.open(portNumber);
    std::cout << "DebugConsole host:" << hostname << " port:" << port << std::endl;

    return port;
}

bool    
McrtRtComputationDebugConsole::eval()
{
    if (!mComputation) return false;

    std::string cmdLine;

    int rt = mTlSvr.recv(cmdLine, 0.0f);
    if (rt < 0) {
        std::cout << "TlSvr error" << std::endl;
        return false;
    } else if (rt == 0) {
        // empty console input
    } else {
        std::cout << "DebugConsole:" << cmdLine.c_str() << std::endl;
        cmdParse(cmdLine);
    }

    return true;
}

void
McrtRtComputationDebugConsole::close()
{
    mTlSvr.close();
    mComputation = 0x0;
}

void
McrtRtComputationDebugConsole::cmdParse(const std::string &cmdLine)
{
    if (cmdCmp("fps", cmdLine)) {
        cmdFps(cmdLine);
    } else if (cmdCmp("geoUpdate", cmdLine)) {
        cmdGeoUpdate(cmdLine);
    } else if (cmdCmp("help", cmdLine)) {
        cmdHelp();
    } else if (cmdCmp("resetCounter", cmdLine)) {
        cmdResetCounter();
    } else if (cmdCmp("show", cmdLine)) {
        cmdShow();
    } else {
        std::ostringstream ostr;
        ostr << "> unknown command>" << cmdLine;
        mTlSvr.send(ostr.str());
    }
}

void
McrtRtComputationDebugConsole::cmdFps(const std::string &cmdLine)
{
    std::istringstream istr(cmdLine);
    std::string token;

    float fps;

    istr >> token;              // skip "fps"
    istr >> token;
    std::istringstream(token) >> fps;

    mComputation->setFps(fps);

    std::ostringstream ostr;
    ostr << "> update fps rate ... " << fps << std::endl;
    mTlSvr.send(ostr.str());
}

void
McrtRtComputationDebugConsole::cmdGeoUpdate(const std::string &cmdLine)
{
    std::istringstream istr(cmdLine);
    std::string token;

    istr >> token;              // skip "geoUpdate"
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
    mComputation->setGeoUpdateMode(sw);
    mTlSvr.send(ostr.str());
}

void
McrtRtComputationDebugConsole::cmdHelp()
{
    std::ostringstream ostr;
    ostr << "help {\n"
         << "  fps <val>          : set fps rate\n"
         << "  geoUpdate <on|off> : geometry message enable(on) or  not(off)\n"
         << "  help               : show this message\n"
         << "  resetCounter       : reset primary ray counter\n"
         << "  show               : show internal info\n"
         << "}\n";

    mTlSvr.send(ostr.str());
}

void
McrtRtComputationDebugConsole::cmdResetCounter()
{
    mComputation->setResetCounter();
    mTlSvr.send("resetCounter\n");
}

void
McrtRtComputationDebugConsole::cmdShow()
{
    std::ostringstream ostr;
    ostr << "status {\n"
         << "        fps:" << mComputation->getFps() << std::endl
         << "  geoUpdate:" << ((mComputation->getGeoUpdateMode())? "true": "false") << std::endl
         << "}\n";
    mTlSvr.send(ostr.str());
}

bool
McrtRtComputationDebugConsole::cmdCmp(const std::string &cmdName, const std::string &cmdLine) const
{
    size_t lenCmdName = cmdName.length();
    size_t lenCmdLine = cmdLine.length();
    if (lenCmdName > lenCmdLine) {
        return false;
    }

    std::string cCmd = cmdLine.substr(0, lenCmdName);
    if (cmdName == cCmd) {
        return true;
    }
    return false;
}

} // namespace mcrt_rt_computation
} // namespace moonray

