// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace moonray_stats {

// TODO: replace this with the standard library version when possible.

namespace detail {
    template <std::size_t... Indices>
    struct IndexTuple
    {
        using next = IndexTuple<Indices..., sizeof...(Indices)>;
    };

    // Builds an IndexTuple<0, 1, 2, ..., Num - 1>
    template <std::size_t Num>
    struct BuildIndexTuple
    {
        using type = typename BuildIndexTuple<Num - 1>::type::next;
    };

    template <>
    struct BuildIndexTuple<0>
    {
        using type = IndexTuple<>;
    };
} // namespace detail

template <typename T, T... Ints>
struct integer_sequence
{
    using value_type = T;
    static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};

namespace detail {
    template <typename T, T Num, typename InSeq = typename BuildIndexTuple<Num>::type>
    struct MakeIntegerSequence;

    template <typename T, T Num, std::size_t... Idx>
    struct MakeIntegerSequence<T, Num, IndexTuple<Idx...>>
    {
        static_assert(Num >= 0, "Cannot make integer sequence of negative length");
        using type = integer_sequence<T, static_cast<T>(Idx)...>;
    };
} // namespace detail

template <typename T, T Num>
using make_integer_sequence = typename detail::MakeIntegerSequence<T, Num>::type;

template <std::size_t... Idx>
using index_sequence = integer_sequence<std::size_t, Idx...>;

template <std::size_t Num>
using make_index_sequence = make_integer_sequence<std::size_t, Num>;

template <typename... Types>
using index_sequence_for = make_index_sequence<sizeof...(Types)>;

} // namespace moonray_stats

