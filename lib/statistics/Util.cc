// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#include "Util.h"

#include <sys/ioctl.h>
#include <unistd.h>

namespace moonray_stats {

/// Split a line at a space (literally ' ') nearest the center of the string.
std::pair<std::string, std::string> splitMiddle(const std::string& s)
{
    if (s.empty()) {
        return std::make_pair(std::string(), std::string());
    }

    const auto center = s.size() / 2;
    const auto posl = s.rfind(' ', center);
    const auto posr = s.find(' ', center);

    if (posl == std::string::npos && posr == std::string::npos) {
        // No spaces found.
        return std::make_pair(s, std::string());
    } else if (posl == std::string::npos) {
        return std::make_pair(s.substr(0, posr), s.substr(posr + 1));
    } else if (posr == std::string::npos) {
        return std::make_pair(s.substr(0, posl), s.substr(posl + 1));
    } else {
        const auto distl = center - posl;
        const auto distr = posr - center;
        const auto pos = (distl < distr) ? posl : posr;
        return std::make_pair(s.substr(0, pos), s.substr(pos + 1));
    }
}

int computeWindowWidth()
{
    // If attached to a terminal...
    if (isatty(STDOUT_FILENO)) {
        // Get the terminal width.
        winsize win;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &win);
        return win.ws_col;
    } else {
        return 72;
    }
}

std::string createDashTitle(std::string s)
{
    s = ' ' + s + ' ';

    const std::size_t preLength = 10;
    const std::size_t overallLength = 58;
    const std::size_t maxStringRoom = overallLength - preLength;

    if (s.length() >= maxStringRoom) {
        return std::string(preLength, '-') + s;
    } else {
        return std::string(preLength, '-') +
               s +
               std::string(maxStringRoom - s.length(), '-');
    }
}

std::string createArrowTitle(std::string s)
{
    s = ' ' + s + ' ';

    const std::size_t numDashes = 10;
    const std::string dashString(numDashes, '-');
    return '<' + dashString + s + dashString + '>';
}

} // namespace moonray_stats

