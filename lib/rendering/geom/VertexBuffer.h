// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VertexBuffer.h
/// $Id$
///

#pragma once

#include <scene_rdl2/render/util/AlignedAllocator.h>
#include <cstdint>
#include <memory>
#include <utility>

namespace moonray {
namespace geom {

///
/// @class VertexBuffer
/// @brief Provides a vertex buffer in which the layout may vary at compile
/// time. The layout is (mostly) hidden to the user. The only difference between
/// layouts in the public API are the data calls.
///
template <typename T, template <typename, typename> class Traits, typename Allocator = scene_rdl2::alloc::AlignedAllocator<T, SIMD_MEMORY_ALIGNMENT>>
class VertexBuffer : private Traits<T, Allocator>
{
public:
    typedef Traits<T, Allocator> traits_type;

    /// Gets a raw float*. Argument types may vary from layout to layout.
    using traits_type::data;

    /// Gets number of elements in the VertexBuffer independent of time.
    /// E.g. If you have 4 elements in the VertexBuffer with 2 time samples
    /// (8 elements total), size returns 4.
    ///
    /// If you have 4 elements in the VertexBuffer with 3 time samples
    /// (12 elements total), size returns 4.
    using traits_type::size;
    using traits_type::capacity;
    using traits_type::reserve;
    using traits_type::shrink_to_fit;
    using traits_type::get_allocator;
    using traits_type::get_time_steps;

    /// Gets the number of floats in the float* returned by data()
    /// E.g. If you have 4 elements in the VertexBuffer with 2 time samples
    /// (8 elements total), and each element is a type composed of three floats,
    /// we return 4*2*3 == 24
    using traits_type::data_size;
    using traits_type::get_memory_usage;
    using traits_type::empty;
    using traits_type::clear;

    using value_type      = typename traits_type::value_type;
    using allocator_type  = typename traits_type::allocator_type;
    using size_type       = typename traits_type::size_type;
    using difference_type = typename traits_type::difference_type;
    using reference       = typename traits_type::reference;
    using const_reference = typename traits_type::const_reference;
    using pointer         = typename traits_type::pointer;
    using const_pointer   = typename traits_type::const_pointer;

    /// Construct an empty VertexBuffer for a single time sample.
    explicit VertexBuffer(allocator_type alloc = allocator_type()) :
        Traits<T, Allocator>(1, alloc)
    {
    }

    /// Construct a VertexBuffer with n default elements.
    explicit VertexBuffer(size_type n,
                          size_type timeSteps = 1,
                          allocator_type alloc = allocator_type()) :
        Traits<T, Allocator>(n, timeSteps, alloc)
    {
    }

    /// Construct a VertexBuffer with n element initialized to value.
    explicit VertexBuffer(size_type n,
                          const_reference value,
                          size_type timeSteps = 1,
                          allocator_type alloc = allocator_type()) :
        Traits<T, Allocator>(n, value, timeSteps, alloc)
    {
    }

    VertexBuffer(VertexBuffer&&) noexcept = default;

    VertexBuffer& operator=(const VertexBuffer&) = delete;
    VertexBuffer& operator=(VertexBuffer&& other)
        noexcept(noexcept(std::declval<VertexBuffer>().traits_type::operator=(std::move(other))))
    {
        traits_type::operator=(std::move(other));
        return *this;
    }

    // We don't have a public copy constructor because it was decided that we
    // didn't want to allow accidental copies of data. However, if you want a
    // copy, you can get one through an explicit copy call.
    VertexBuffer copy() const
    {
        return VertexBuffer(*this);
    }

    reference operator()(size_type pos, size_type time = 0)
    {
        auto p = this->getAddress(time, pos);
        return *p;
    }

    const_reference operator()(size_type pos, size_type time = 0) const
    {
        auto p = this->getAddress(time, pos);
        return *p;
    }

    // Follows strong-exception guarantee.
    void resize(size_type n)
    {
        if (n > size()) {
            ensure_capacity(n);
            // Just add n - size() values
            traits_type::append(n - size(), value_type());
        } else if (n < size()) {
            // Destroy last size() - n values.
            traits_type::destroy_last(size() - n);
        } // else if (n == size) do nothing.
    }

    // Follows strong-exception guarantee.
    void resize(size_type n, const_reference value)
    {
        if (n > size()) {
            ensure_capacity(n);
            // Just add n - size() values
            traits_type::append(n - size(), value);
        } else if (n < size()) {
            // Destroy last size() - n values.
            traits_type::destroy_last(size() - n);
        } // else if (n == size) do nothing.
    }

    /// This version of push_back is only to be used with a VertexBuffer with a
    /// single time sample.
    // Follows strong-exception guarantee.
    void push_back(const_reference u)
    {
        assert(get_time_steps() == 1);
        ensure_capacity(1);
        traits_type::push_back(u);
    }

    /// This version of push_back is only to be used with a VertexBuffer with a
    /// single time sample.
    // Follows strong-exception guarantee.
    void push_back(value_type&& u)
    {
        assert(get_time_steps() == 1);
        ensure_capacity(1);
        traits_type::push_back(std::move(u));
    }

    /// This version of push_back assumes that u has array access semantics,
    /// i.e. operator[] is defined for u. Time samples are accessed through u's
    /// operator[] for each of the time samples in the VertexBuffer.
    // Follows strong-exception guarantee.
    template <typename U>
    void push_back(const U& u)
    {
        ensure_capacity(1);
        traits_type::push_back(u);
    }

    // Follows strong-exception guarantee.
    void append(const VertexBuffer& other)
    {
        ensure_capacity(other.size());
        traits_type::append(other);
    }

    // Follows strong-exception guarantee.
    void append(VertexBuffer&& other)
    {
        ensure_capacity(other.size());
        traits_type::append(std::move(other));
    }

    void swap(VertexBuffer& other) noexcept(noexcept(std::declval<VertexBuffer>().traits_type::swap(other)))
    {
        traits_type::swap(other);
    }

private:
    // We don't have a public copy constructor because it was decided that we
    // didn't want to allow accidental copies of data. However, if you want a
    // copy, you can get one through an explicit copy call.
    VertexBuffer(const VertexBuffer& other) :
        Traits<T, Allocator>(other)
    {
    }

    void ensure_capacity(size_type nelements)
    {
        auto cap = capacity();
        auto sz = size();
        if (sz + nelements > cap) {
            // This has to be at least 2 for our integer math recurrence
            // relation to work, but we'll start out at four, as a VertexBuffer
            // will probably always hold at least three, and a good chance at
            // four.
            cap = std::max<decltype(cap)>(cap, 4);
            while (sz + nelements > cap) {
                cap = cap * 3 / 2;
            }
            traits_type::reserve(cap);
        }
    }
};

} // namespace geom
} // namespace moonray

namespace std {
template <typename T, template <typename, typename> class Traits, typename Allocator>
void swap(moonray::geom::VertexBuffer<T, Traits, Allocator>& a,
          moonray::geom::VertexBuffer<T, Traits, Allocator>& b) noexcept(noexcept(a.swap(b)))
{
    a.swap(b);
}
} // namespace std

