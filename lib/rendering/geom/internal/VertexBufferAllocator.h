// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VertexBufferAllocator.h
/// $Id$
///

#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace moonray {
namespace geom {

// Construct "size" elements into array "p" with arguments "args."
// If construction throws, all previously constructed elements are destroyed.
template <typename Allocator, typename... Args>
inline static void constructArraySamples(Allocator& alloc,
                                         typename std::allocator_traits<Allocator>::pointer p,
                                         typename std::allocator_traits<Allocator>::size_type size,
                                         Args&&... args)
{
    typedef std::allocator_traits<Allocator> traits;
    using size_type = typename traits::size_type;

    assert(p);
    if (!p) {
        return;
    }

    size_type n = 0;
    try {
        for (n = 0; n < size; ++n) {
            traits::construct(alloc, p + n,
                              std::forward<Args>(args)...);
        }
    } catch (...) {
        for (size_type i = 0; i < n; ++i) {
            traits::destroy(alloc, p + 1);
        }
        throw;
    }
}

// Construct "size" elements into array "p" by copying from "src."
// If construction throws, all previously constructed elements are destroyed.
template <typename Allocator, typename... Args>
inline static void copyArraySamples(Allocator& alloc,
                                    typename std::allocator_traits<Allocator>::pointer p,
                                    typename std::allocator_traits<Allocator>::const_pointer src,
                                    typename std::allocator_traits<Allocator>::size_type size)
{
    typedef std::allocator_traits<Allocator> traits;
    using size_type = typename traits::size_type;

    assert(p);
    if (!p || !src) {
        return;
    }

    size_type n = 0;
    try {
        for (n = 0; n < size; ++n) {
            traits::construct(alloc, p + n, src[n]);
        }
    } catch (...) {
        for (size_type i = 0; i < n; ++i) {
            traits::destroy(alloc, p + 1);
        }
        throw;
    }
}

// Construct "size" elements into array "p" by moving or copying from "src."
// If the type guarantees nothrow construction we move, otherwise we copy.
// If construction throws, all previously constructed elements are destroyed.
template <typename Allocator>
inline static void moveArraySamples(Allocator& alloc,
                                    typename std::allocator_traits<Allocator>::pointer p,
                                    typename std::allocator_traits<Allocator>::pointer src,
                                    typename std::allocator_traits<Allocator>::size_type size)
{
    typedef std::allocator_traits<Allocator> traits;
    using size_type = typename traits::size_type;

    assert(p);
    if (!p || !src) {
        return;
    }

    size_type n = 0;
    try {
        for (n = 0; n < size; ++n) {
            traits::construct(alloc, p + n, std::move_if_noexcept(src[n]));
        }
    } catch (...) {
        for (size_type i = 0; i < n; ++i) {
            traits::destroy(alloc, p + 1);
        }
        throw;
    }
}

// Destroy "size" elements in array "p."
template <typename Allocator>
inline static void destroyArraySamples(Allocator& alloc,
                                       typename std::allocator_traits<Allocator>::pointer p,
                                       typename std::allocator_traits<Allocator>::size_type size) noexcept
{
    typedef std::allocator_traits<Allocator> traits;
    using size_type  = typename traits::size_type;
    using pointer    = typename traits::pointer;

    if (!p) {
        return;
    }

    for (size_type i = 0; i < size; ++i) {
        pointer q = p + i;
        if (q) {
            traits::destroy(alloc, q);
        }
    }
}


} // namespace geom
} // namespace moonray

