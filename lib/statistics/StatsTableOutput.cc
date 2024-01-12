// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "StatsTableOutput.h"

namespace moonray_stats {
namespace internal {

std::ostream& writeEqualityCSVTableContents(std::ostream& outs,
                                            const StatsTable<2>& table,
                                            bool athenaFormat,
                                            const Formatter& formatter,
                                            const TableFlags& flags,
                                            const char* sep)
{
    for (std::size_t row = 0; row < table.getNumRows(); ++row) {
        if (athenaFormat) {
            outs << table.getTitle() << CSVTableSep;
        }
        writeTableCell(outs, table, formatter, flags, row, 0);

        // Check the next column's unit. If it has one, append it to the first
        // column.
        const char* const unit = table(row, 1).getUnit(formatter);
        if (std::strcmp(unit, "") != 0) {
            outs << " (" << unit << ")";
        }
        outs << sep;
        writeTableCell(outs, table, formatter, flags, row, 1);
        outs.put('\n');
    }
    return outs;
}

} // namespace internal
} // namespace moonray_stats

