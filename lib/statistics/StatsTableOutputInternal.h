// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "CountingStream.h"
#include "Formatters.h"
#include "IOSFlags.h"
#include "integer_sequence.h"
#include "StatsTable.h"
#include "TableFlags.h"
#include "Util.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace moonray_stats {
namespace internal {

const char* const InfoTableSep = " ";
const char* const InfoEqualityTableSep = " = ";
const char* const CSVTableSep = ",";

// Allow for up to 2 rows of headers.
// cppcheck-suppress constStatement
template <std::size_t N> using HeaderRowContainer = std::vector<StringList<N>>;

std::ostream& writeEqualityCSVTableContents(std::ostream& outs,
                                            const StatsTable<2>& table,
                                            bool athenaFormat,
                                            const Formatter& formatter,
                                            const TableFlags& flags,
                                            const char* sep);

template <std::size_t N>
HeaderRowContainer<N> splitHeaders(const StringList<N>& basicHeaders)
{
    const std::size_t maxHeaderLength = 16;
    HeaderRowContainer<N> headers;

    if (std::any_of(basicHeaders.cbegin(), basicHeaders.cend(),
                    [](const std::string& h) {
                        return h.length() > maxHeaderLength;
                    })) {
        // Split headers into multiple lines
        headers.resize(2);
        for (std::size_t i = 0; i < N; ++i) {
            const auto p = splitMiddle(basicHeaders[i]);
            headers[0][i] = p.first;
            headers[1][i] = p.second;
        }
    } else {
        headers.resize(1);
        for (std::size_t i = 0; i < N; ++i) {
            headers[0][i] = basicHeaders[i];
        }
    }
    return headers;
}

template <std::size_t N>
std::ostream& writeTableCell(std::ostream& outs,
                             const StatsTable<N>& table,
                             const Formatter& formatter,
                             const TableFlags& flags,
                             std::size_t row,
                             std::size_t col)
{
    IOSFlagsRAII raii(outs);
    flags.get(row, col).imbue(outs);
    table(row, col).write(outs, formatter);
    return outs;
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeTableRow(std::ostream& outs,
                            const StatsTable<N>& table,
                            const Formatter& formatter,
                            const TableFlags& flags,
                            std::size_t row,
                            const char* sep)
{
    constexpr std::array<std::size_t, N> rtindices = {indices...};

    writeTableCell(outs, table, formatter, flags, row, rtindices[0]);
    for (std::size_t col = 1; col < table.getNumColumns(); ++col) {
        outs << sep;
        writeTableCell(outs, table, formatter, flags, row, rtindices[col]);
    }
    return outs;
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeTableRow(std::ostream& outs,
                            const StatsTable<N>& table,
                            const Formatter& formatter,
                            const TableFlags& flags,
                            std::size_t row,
                            const char* sep,
                            moonray_stats::index_sequence<indices...>)
{
    return writeTableRow<N, indices...>(outs, table, formatter, flags, row, sep);
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeHeaderLine(std::ostream& outs,
                              const StringList<N>& output,
                              std::array<std::streamsize, N> widths,
                              const char* sep)
{
    IOSFlagsRAII raii(outs);
    IOSFlags flags(outs);
    flags.left();

    constexpr std::array<std::size_t, N> rtindices = {indices...};

    // Output the first value.
    const auto firstIdx = rtindices[0];

    flags.width(widths[firstIdx]);
    flags.imbue(outs);
    outs << output[firstIdx];

    for (std::size_t i = 1; i < N; ++i) {
        const auto idx = rtindices[i];
        outs.width(0);
        outs << sep;
        flags.width(widths[idx]);
        flags.imbue(outs);
        outs << output[idx];
    }
    return outs;
}

template <std::size_t... indices, std::size_t N>
std::ostream& writeHeaderLine(std::ostream& outs,
                              const StringList<N>& output,
                              std::array<std::streamsize, N> widths,
                              const char* sep,
                              moonray_stats::index_sequence<indices...>)
{
    return writeHeaderLine<N, indices...>(outs, output, widths, sep);
};

template <std::size_t N, std::size_t... indices>
std::ostream& writeInfoTableHeaders(std::ostream& outs,
                                    const std::string& prefix,
                                    const StringList<N>& headers,
                                    const std::array<std::streamsize, N>& widths,
                                    const char* sep)
{
    const auto split = splitHeaders(headers);
    for (const auto& header : split) {
        outs << prefix;
        writeHeaderLine<N, indices...>(outs, header, widths, sep);
        outs.put('\n');
    }

    return outs;
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeInfoTableHeaders(std::ostream& outs,
                                    const std::string& prefix,
                                    const StringList<N>& headers,
                                    const std::array<std::streamsize, N>& widths,
                                    const char* sep,
                                    moonray_stats::index_sequence<indices...>)
{
    return writeInfoTableHeaders<N, indices...>(outs, prefix, headers, widths,
                                                sep);
}

// return overall row width
template <std::size_t N>
std::size_t autoSizeHumanColumnFlags(const std::ostream& outs,
                                     const StatsTable<N>& table,
                                     TableFlags& flags,
                                     std::size_t maxRows)
{
    CountingStream cs;
    IOSFlags fromStream(outs);
    fromStream.imbue(cs);

    const char sep[] = { CountingStream::sRecordSeparator, '\0' };
    constexpr std::array<std::streamsize, N> columnWidths = { 0 };
    internal::writeInfoTableHeaders<N>(cs, "", table.getHeaders(), columnWidths,
                                       sep, moonray_stats::make_index_sequence<N>{});

    const std::size_t numRows = std::min(table.getNumRows(), maxRows);
    for (std::size_t i = 0; i < numRows; ++i) {
        internal::writeTableRow<N>(cs, table, FormatterHuman(), flags, i, sep,
                                   moonray_stats::make_index_sequence<N>{});
        cs.put('\n');
    }

    for (std::size_t row = 0; row < table.getNumRows(); ++row) {
        for (std::size_t col = 0; col < table.getNumColumns(); ++col) {
            flags.get(row, col).width(cs.getColumnWidth(col));
        }
    }

    return cs.getWidth();
}

template <std::size_t N>
void trimLastColumn(const std::string& prefix,
                    std::size_t rowWidth,
                    TableFlags& flags,
                    const char* sep,
                    std::size_t numRows,
                    const std::array<std::size_t, N>& indices)
{

    // The total output width is:
    // width(prefix) + width(columns) + width(separator) * (columns - 1)
    //                                  (separation for each column)
    //
    // We want every row in every column the same width for proper left/right
    // justification. However, we want to prevent wrapping for columns where the
    // output isn't actually longer than the terminal width. So, to keep
    // everything aligned, we want to adjust only the last column so that it
    // only takes up the remaining terminal width.
    //
    // If terminal width >= total width, we're good.
    // Else
    // Overage = total width - terminal width
    // If last column width = max(0, last column width - overage)

    const std::size_t terminalWidth = computeWindowWidth();
    const std::size_t sepWidth = std::strlen(sep);
    const std::size_t totalWidth = prefix.length() + sepWidth * (indices.size() - 1) + rowWidth;

    if (terminalWidth < totalWidth) {
        const std::size_t lastColumn = indices.back();
        const std::size_t overage = totalWidth - terminalWidth;
        const std::size_t oldWidth = flags.get(0, lastColumn).width();
        const std::size_t newWidth = (overage < oldWidth) ? oldWidth - overage : 0;
        for (std::size_t row = 0; row < numRows; ++row) {
            flags.get(row, lastColumn).width(newWidth);
        }
    }
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeInfoTablePermutationImpl(std::ostream& outs,
                                            const std::string& prefix,
                                            const StatsTable<N>& table,
                                            const TableFlags& flags,
                                            const char* sep,
                                            std::size_t maxRows)
{
    constexpr std::array<std::size_t, N> rtindices = {indices...};

    auto flagsCpy = flags.clone();
    const auto rowWidth = autoSizeHumanColumnFlags(outs, table, *flagsCpy, maxRows);
    trimLastColumn(prefix, rowWidth, *flagsCpy, sep, table.getNumRows(), rtindices);

    std::array<std::streamsize, N> columnWidths;
    for (std::size_t i = 0; i < N; ++i) {
        columnWidths[i] = flagsCpy->get(0, i).width();
    }

    // Write separator
    const std::size_t sepWidth = std::strlen(sep);
    const std::size_t overallWidth = std::accumulate(columnWidths.cbegin(),
                                                      columnWidths.cend(), 0) +
                                                     (N - 1) * sepWidth;

    outs << prefix;
    outs << createDashTitle(table.getTitle());
    outs.put('\n');

    if (table.hasHeaders()) {
        const StringList<N> headers = table.getHeaders();
        const std::string headerSep(sepWidth, ' ');
        writeInfoTableHeaders<N, indices...>(outs, prefix, headers,
                                             columnWidths,
                                             headerSep.c_str());

        StringList<N> separators;
        std::generate(separators.begin(), separators.end(), [&, i=0]() mutable {
            return std::string(columnWidths[i++], '-');
        });
        outs << prefix;
        writeHeaderLine<N, indices...>(outs, separators, columnWidths,
                                       headerSep.c_str());
        outs.put('\n');
    }

    const std::size_t numRows = std::min(table.getNumRows(), maxRows);
    for (std::size_t i = 0; i < numRows; ++i) {
        if (table.isSeparator(i)) {
            outs << prefix;
            outs << std::string(overallWidth, '-');
            outs.put('\n');
        }
        outs << prefix;
        writeTableRow<N, indices...>(outs, table, FormatterHuman(), *flagsCpy, i,
                                     sep);
        outs.put('\n');
    }
    if (table.getNumRows() > maxRows) {
        outs << prefix << "...more\n";
    }
    return outs;
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeInfoTablePermutationImpl(std::ostream& outs,
                                            const std::string& prefix,
                                            const StatsTable<N>& table,
                                            const TableFlags& flags,
                                            const char* sep,
                                            std::size_t maxRows,
                                            moonray_stats::index_sequence<indices...>)
{
    return writeInfoTablePermutationImpl<N, indices...>(outs, prefix, table,
                                                        flags, sep, maxRows);
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeCSVTablePermutationImpl(std::ostream& outs,
                                           const StatsTable<N>& table,
                                           bool athenaFormat,
                                           const TableFlags& flags)
{
    if (!athenaFormat) {
        outs << createArrowTitle(table.getTitle());
        outs.put('\n');

        constexpr std::array<std::streamsize, N> widths = {0};
        if (table.hasHeaders()) {
            writeHeaderLine<N, indices...>(outs, table.getHeaders(), widths,
                                           CSVTableSep);
            outs.put('\n');
        }
    }
    for (std::size_t i = 0; i < table.getNumRows(); ++i) {
        if (athenaFormat) {
            outs << table.getTitle() << CSVTableSep;
        }
        writeTableRow<N, indices...>(outs, table, FormatterCSV(), flags, i,
                                     CSVTableSep);
        outs.put('\n');
    }
    if (!athenaFormat) {
        outs.put('\n'); // Add an extra line break between tables
    }
    return outs;
}

template <std::size_t N, std::size_t... indices>
std::ostream& writeCSVTablePermutationImpl(std::ostream& outs,
                                           const StatsTable<N>& table,
                                           bool athenaFormat,
                                           const TableFlags& flags,
                                           moonray_stats::index_sequence<indices...>)
{
    return writeCSVTablePermutationImpl<N, indices...>(outs, table, athenaFormat, flags);
}

} // namespace internal
} // namespace moonray_stats


