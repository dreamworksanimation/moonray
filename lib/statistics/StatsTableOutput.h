// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "CountingStream.h"
#include "Formatters.h"
#include "IOSFlags.h"
#include "integer_sequence.h"
#include "StatsTable.h"
#include "StatsTableOutputInternal.h"
#include "TableFlags.h"
#include "Util.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

#ifndef PRINT
#define PRINT(x) std::cout << #x << ": " << (x) << '\n'
#endif

namespace moonray_stats {

template <std::size_t N>
inline ColumnFlags<N> getHumanColumnFlags(const std::ostream& outs,
                                          const StatsTable<N>& /*table*/)
{
    return ColumnFlags<N>(outs);
}

template <std::size_t N>
inline FullFlags<N> getHumanFullFlags(const std::ostream& outs,
                                      const StatsTable<N>& table)
{
    return FullFlags<N>(getHumanColumnFlags<N>(outs, table), table.getNumRows());
}

inline ColumnFlags<2> getHumanEqualityColumnFlags(const std::ostream& outs,
                                                  const StatsTable<2>& table)
{
    ColumnFlags<2> flags = getHumanColumnFlags(outs, table);
    flags.set(0).left();
    flags.set(1).left();
    flags.set(0).width(std::max<std::streamsize>(flags.set(0).width(), 32));
    return flags;
}

inline FullFlags<2> getHumanEqualityFullFlags(const std::ostream& outs,
                                              const StatsTable<2>& table)
{
    return FullFlags<2>(getHumanEqualityColumnFlags(outs, table),
                        table.getNumRows());
}

template <std::size_t N>
ConstantFlags getCSVFlags(const std::ostream& outs,
                          const StatsTable<N>& /*table*/)
{
    ConstantFlags flags(outs);
    flags.set().width(0);
    flags.set().left();
    struct NoComma : public std::numpunct<char>
    {
        std::string do_grouping() const override { return ""; }
    };
    std::locale mylocale(outs.getloc(), new NoComma);
    flags.set().imbue( mylocale ); // Not a memory
    return flags;
}

// We templatize the table type so that we avoid the ambiguity of passing in
// simply the size. We want the indices to the first parameter, so that this is
// user-friendly and easier to read.
template <std::size_t... indices, typename TableType, typename FlagType>
std::ostream& writeInfoTablePermutation(std::ostream& outs,
                                        const std::string& prefix,
                                        const TableType& table,
                                        const FlagType& flags,
                                        std::size_t maxRows = std::numeric_limits<std::size_t>::max())
{
    return internal::writeInfoTablePermutationImpl<TableType::getNumColumns(), indices...>(
        outs, prefix, table, flags, internal::InfoTableSep, maxRows);
}

// We templatize the table type so that we avoid the ambiguity of passing in
// simply the size. We want the indices to the first parameter, so that this is
// user-friendly and easier to read.
template <std::size_t... indices, typename TableType>
std::ostream& writeInfoTablePermutation(std::ostream& outs,
                                        const std::string& prefix,
                                        const TableType& table,
                                        std::size_t maxRows = std::numeric_limits<std::size_t>::max())
{
    const ColumnFlags<TableType::getNumColumns()> flags(outs);
    return internal::writeInfoTablePermutationImpl<TableType::getNumColumns(), indices...>(
        outs, prefix, table, flags, internal::InfoTableSep, maxRows);
}

template <std::size_t N>
std::ostream& writeInfoTable(std::ostream& outs,
                             const std::string& prefix,
                             const StatsTable<N>& table,
                             const TableFlags& flags,
                             std::size_t maxRows = std::numeric_limits<std::size_t>::max())
{
    return internal::writeInfoTablePermutationImpl<N>(outs,
                                                      prefix,
                                                      table,
                                                      flags,
                                                      internal::InfoTableSep,
                                                      maxRows,
                                                      moonray_stats::make_index_sequence<N>{});
}

template <std::size_t N>
std::ostream& writeInfoTable(std::ostream& outs,
                             const std::string& prefix,
                             const StatsTable<N>& table,
                             std::size_t maxRows = std::numeric_limits<std::size_t>::max())
{
    const ColumnFlags<StatsTable<N>::getNumColumns()> flags(outs);
    return internal::writeInfoTablePermutationImpl<N>(outs,
                                                      prefix,
                                                      table,
                                                      flags,
                                                      internal::InfoTableSep,
                                                      maxRows,
                                                      moonray_stats::make_index_sequence<N>{});
}

inline std::ostream& writeEqualityInfoTable(std::ostream& outs,
                                            const std::string& prefix,
                                            const StatsTable<2>& table,
                                            const TableFlags& flags,
                                            std::size_t maxRows = std::numeric_limits<std::size_t>::max())
{
    return internal::writeInfoTablePermutationImpl<2, 0, 1>(outs,
                                                            prefix,
                                                            table,
                                                            flags,
                                                            internal::InfoEqualityTableSep,
                                                            maxRows);
}

inline std::ostream& writeEqualityInfoTable(std::ostream& outs,
                                            const std::string& prefix,
                                            const StatsTable<2>& table,
                                            std::size_t maxRows = std::numeric_limits<std::size_t>::max())
{
    const auto flags = getHumanEqualityColumnFlags(outs, table);
    return internal::writeInfoTablePermutationImpl<2, 0, 1>(outs,
                                                            prefix,
                                                            table,
                                                            flags,
                                                            internal::InfoEqualityTableSep,
                                                            maxRows);
}

// We templatize the table type so that we avoid the ambiguity of passing in
// simply the size. We want the indices to the first parameter, so that this is
// user-friendly and easier to read.
template <std::size_t... indices, typename TableType, typename FlagType>
std::ostream& writeCSVTablePermutation(std::ostream& outs,
                                       const TableType& table,
                                       const FlagType& flags)
{
    return internal::writeCSVTablePermutationImpl<TableType::getNumColumns(), indices...>(
        outs, table, flags);
}

// We templatize the table type so that we avoid the ambiguity of passing in
// simply the size. We want the indices to the first parameter, so that this is
// user-friendly and easier to read.
template <std::size_t... indices, typename TableType>
std::ostream& writeCSVTablePermutation(std::ostream& outs,
                                       const TableType& table)
{
    const auto flags = getCSVFlags(outs, table);
    return internal::writeCSVTablePermutationImpl<TableType::getNumColumns(), indices...>(
        outs, table, flags);
}

template <std::size_t N>
std::ostream& writeCSVTable(std::ostream& outs,
                            const StatsTable<N>& table,
                            bool athenaFormat,
                            const TableFlags& flags)
{
    return internal::writeCSVTablePermutationImpl<N>(outs, table, athenaFormat, flags,
                                                     moonray_stats::make_index_sequence<N>{});
}

template <std::size_t N>
std::ostream& writeCSVTable(std::ostream& outs,
                            const StatsTable<N>& table,
                            bool athenaFormat)
{
    const auto flags = getCSVFlags(outs, table);
    return internal::writeCSVTablePermutationImpl<N>(outs, table, athenaFormat, flags,
                                                     moonray_stats::make_index_sequence<N>{});
}

inline std::ostream& writeEqualityCSVTable(std::ostream& outs,
                                           const StatsTable<2>& table,
                                           bool athenaFormat,
                                           const TableFlags& flags)
{
    if (!athenaFormat) {
        outs << createArrowTitle(table.getTitle());
        outs.put('\n');

        constexpr std::array<std::streamsize, 2> widths = {0};
        if (table.hasHeaders()) {
            internal::writeHeaderLine<2, 0, 1>(outs, table.getHeaders(), widths,
                                               internal::CSVTableSep);
            outs.put('\n');
        }
    }
    internal::writeEqualityCSVTableContents(outs,
                                            table,
                                            athenaFormat,
                                            FormatterCSV(),
                                            flags,
                                            internal::CSVTableSep);
    if (!athenaFormat) {
        outs.put('\n'); // Add an extra line break between tables
    }
    return outs;
}

inline std::ostream& writeEqualityCSVTable(std::ostream& outs,
                                           const StatsTable<2>& table,
                                           bool athenaFormat)
{
    const auto flags = getCSVFlags(outs, table);
    return writeEqualityCSVTable(outs, table, athenaFormat, flags);
}

} // namespace moonray_stats


