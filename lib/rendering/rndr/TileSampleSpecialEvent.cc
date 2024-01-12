// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "TileSampleSpecialEvent.h"

#include <iomanip>
#include <sstream>

namespace moonray {
namespace rndr {

std::string    
TileSampleSpecialEvent::show() const
{
    auto showTbl = [&](const std::string &msg, const UIntTable tbl) -> std::string {
        std::ostringstream ostr;
        int w0 = std::to_string(tbl.size()).size();
        int w1 = std::to_string(tbl.back()).size();
        int w2 = std::to_string((tbl.back() + 1)/ 64).size();
        ostr << msg << " tbl (total:" << tbl.size() << ") {\n";
        for (size_t i = 0; i < tbl.size(); ++i) {
            ostr << "  i:" << std::setw(w0) << i
                 << " tileSamplesId:" << std::setw(w1) << tbl[i]
                 << " pixSamplesTotal:" << std::setw(w2) << (tbl[i] + 1) / 64 << '\n';
        }
        ostr << "}";
        return ostr.str();
    };

    return showTbl("TileSampleSpecialEvent tileSampleIdTable", mTileSampleIdTable);
}

} // namespace rndr
} // namespace moonray

