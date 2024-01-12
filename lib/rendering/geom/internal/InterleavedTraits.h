// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file InterleavedTraits.h
/// $Id$
///

#pragma once

#include <scene_rdl2/render/util/AlignedAllocator.h>
#include <scene_rdl2/render/util/type_traits.h>
#include "VertexBufferAllocator.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

namespace moonray {
namespace geom {

///
/// @class InterleavedTraits
/// @brief Provides a vertex buffer layout in which each time sample is
/// interleaved. E.g. (at0, at1, bt0, bt1, ct0, ct1, ..., zt0, zt1)
/// @remark Implementation detail for internal memory layout.
///
template <typename T, typename Allocator = scene_rdl2::alloc::AlignedAllocator<T, SIMD_MEMORY_ALIGNMENT>>
class InterleavedTraits : private Allocator // Empty base class optimization
{
protected:
    // This non-virtual destructor is protected because:
    // A: This class is supposed to be a base class
    // B: This class is not supposed to be used polymorphically
    ~InterleavedTraits()
    {
        destroy();
    }

protected:
    using traits                 = std::allocator_traits<Allocator>;
    using value_type             = T;
    using allocator_type         = typename traits::allocator_type;
    using size_type              = typename traits::size_type;
    using difference_type        = std::ptrdiff_t;
    using reference              = T&;
    using const_reference        = const T&;
    using pointer                = typename traits::pointer;
    using const_pointer          = typename traits::const_pointer;

    explicit InterleavedTraits(size_type timeSteps,
                               allocator_type alloc = allocator_type()) :
        allocator_type(alloc),
        mCapacity(0),
        mSize(0),
        mTimeSteps(timeSteps),
        mData(nullptr)
    {
    }

    InterleavedTraits(size_type n, size_type timeSteps,
                      allocator_type alloc = allocator_type()) :
        allocator_type(alloc),
        mCapacity(n),
        mSize(n),
        mTimeSteps(timeSteps),
        mData(create(getAllocatorInternal(), n, timeSteps))
    {
    }

    InterleavedTraits(size_type n, const_reference value, size_type timeSteps,
                      allocator_type alloc = allocator_type()) :
        allocator_type(alloc),
        mCapacity(n),
        mSize(n),
        mTimeSteps(timeSteps),
        mData(create(getAllocatorInternal(), n, timeSteps, value))
    {
    }

    InterleavedTraits(const InterleavedTraits& other) :
        allocator_type(traits::select_on_container_copy_construction(other)),
        mCapacity(other.mSize),
        mSize(other.mSize),
        mTimeSteps(other.mTimeSteps),
        mData(copyValue(getAllocatorInternal(), other.mData, other.mSize, other.mTimeSteps))
    {
    }

    InterleavedTraits(InterleavedTraits&& other) noexcept :
        allocator_type(std::move(other)),
        mCapacity(other.mCapacity),
        mSize(other.mSize),
        mTimeSteps(other.mTimeSteps),
        mData(other.mData)
    {
        other.mCapacity = 0;
        other.mTimeSteps = 0;
        other.mData = nullptr;
        other.mSize = 0;
    }

    InterleavedTraits& operator=(const InterleavedTraits& other) = delete;

    InterleavedTraits& operator=(InterleavedTraits&& other)
        noexcept(traits::propagate_on_container_move_assignment::value ||
                 fauxstd::is_always_equal<allocator_type>::value)
    {
        if (traits::propagate_on_container_move_assignment::value &&
            this->getAllocatorInternal() != other.getAllocatorInternal()) {
            // We get to move the allocator. This is good.
            allocator_type::operator=(std::move(other));
            mCapacity = other.mCapacity;
            mSize = other.mSize;
            mTimeSteps = other.mTimeSteps;
            mData = std::move(other.mData);

            other.mTimeSteps = 0;
            other.mCapacity = 0;
            other.mSize = 0;
            other.mData = nullptr;
        } else if (fauxstd::is_always_equal<allocator_type>::value ||
                   this->getAllocatorInternal() ==
                   other.getAllocatorInternal()) {
            // We don't have to move the allocator. This is good.
            mCapacity = other.mCapacity;
            mSize = other.mSize;
            mTimeSteps = other.mTimeSteps;
            mData = std::move(other.mData);

            other.mTimeSteps = 0;
            other.mCapacity = 0;
            other.mSize = 0;
            other.mData = nullptr;
        } else {
            // Our only option is to copy all of the elements, because we can't
            // do anything smart with the allocators. This is bad.
            auto newData = copyMove(getAllocatorInternal(), other.mData,
                                    other.mSize, other.mTimeSteps);
            destroy();
            mCapacity = other.mSize;
            mSize = other.mSize;
            mTimeSteps = other.mTimeSteps;
            mData = newData;
        }

        return *this;
    }

    bool empty() const noexcept
    {
        return mSize == 0;
    }

    void clear() noexcept
    {
        if (mData) {
            destroyArraySamples(getAllocatorInternal(), mData,
                                mSize * mTimeSteps);
        }
        mSize = 0;
    }

    void swap(InterleavedTraits& other)
        noexcept(std::allocator_traits<Allocator>::propagate_on_container_swap::value ||
                 fauxstd::is_always_equal<allocator_type>::value)
    {
        using std::swap; // Allow ADL
        if (traits::propagate_on_container_swap::value) {
            // We get to swap the allocators. This is good.
            swap(getAllocatorInternal(), other.getAllocatorInternal());
        } else {
            // If we're not swapping the allocators, they had better be equal!
            assert(fauxstd::is_always_equal<allocator_type>::value ||
                   getAllocatorInternal() == other.getAllocatorInternal());
        }

        swap(mCapacity, other.mCapacity);
        swap(mSize, other.mSize);
        swap(mTimeSteps, other.mTimeSteps);
        swap(mData, other.mData);
    }

    float* data()
    {
        static_assert(std::is_standard_layout<T>::value, "Assume standard layout"
            " for conversion to float pointer.");
        assert(mData);
        return reinterpret_cast<float*>(mData);
    }

    const float* data() const
    {
        static_assert(std::is_standard_layout<T>::value, "Assume standard layout"
            " for conversion to float pointer.");
        assert(mData);
        return reinterpret_cast<const float*>(mData);
    }

    size_type size() const
    {
        return mSize;
    }

    size_type capacity() const
    {
        return mCapacity;
    }

    size_type data_size() const
    {
        return mSize * mTimeSteps * sizeof(T) / sizeof(float);
    }

    size_type get_time_steps() const
    {
        return mTimeSteps;
    }

    allocator_type get_allocator() const
    {
        return getAllocatorInternal();
    }

    size_type get_memory_usage() const noexcept
    {
        size_type mem = sizeof(*this) +
                        sizeof(value_type) * mCapacity * mTimeSteps;
        return mem;
    }

    void push_back(const_reference u)
    {
        assert(mTimeSteps == 1);
        assert(mCapacity > mSize);
        pointer p = getAddress(0, mSize);
        traits::construct(getAllocatorInternal(), p, u);
        ++mSize;
    }

    void push_back(value_type&& u)
    {
        assert(mTimeSteps == 1);
        assert(mCapacity > mSize);
        pointer p = getAddress(0, mSize);
        traits::construct(getAllocatorInternal(), p, std::move(u));
        ++mSize;
    }

    template <typename U>
    void push_back(const U& u)
    {
        assert(mCapacity > mSize);
        for (size_type t = 0; t < mTimeSteps; ++t) {
            pointer p = getAddress(t, mSize);
            traits::construct(getAllocatorInternal(), p, u[t]);
        }
        ++mSize;
    }

    void append(const InterleavedTraits& other)
    {
        assert(mTimeSteps == other.mTimeSteps);
        assert(mCapacity >= mSize + other.mSize);
        copyArraySamples(getAllocatorInternal(),
                         mData + mSize * mTimeSteps,
                         other.mData,
                         other.mSize * other.mTimeSteps);
        mSize += other.mSize;
    }

    void append(InterleavedTraits&& other)
    {
        assert(mTimeSteps == other.mTimeSteps);
        assert(mCapacity >= mSize + other.mSize);
        moveArraySamples(getAllocatorInternal(),
                         mData + mSize * mTimeSteps,
                         other.mData,
                         other.mSize * other.mTimeSteps);
        mSize += other.mSize;
        other.destroy();
        other.mSize = 0;
        other.mCapacity = 0;
        other.mData = nullptr;
    }

    void append(size_type n, const_reference value)
    {
        assert(mCapacity >= mSize + n);

        // We don't have to worry about exceptions: constructArraySamples will
        // clean up after itself, and we already have the memory allocated.
        constructArraySamples(getAllocatorInternal(),
                              mData + mSize * mTimeSteps,
                              n * mTimeSteps,
                              value);
        mSize += n;
    }

    void destroy_last(size_type n)
    {
        assert(n <= mSize);
        destroyArraySamples(getAllocatorInternal(),
                            mData + mSize * mTimeSteps - n, n * mTimeSteps);
        mSize -= n;
    }

    void reserve(size_type cap)
    {
        if (cap > mCapacity) {
            change_memory_size(cap);
        }
    }

    void shrink_to_fit()
    {
        if (mCapacity != mSize) {
            change_memory_size(mSize);
        }
    }

    const_pointer getAddress(size_type time, size_type idx) const noexcept
    {
        assert(time < mTimeSteps);
        assert(mData);
        const_pointer base = mData;
        return base + idx * mTimeSteps + time;
    }

    pointer getAddress(size_type time, size_type idx) noexcept
    {
        assert(time < mTimeSteps);
        assert(mData);
        pointer base = mData;
        return base + idx * mTimeSteps + time;
    }

private:
    const allocator_type& getAllocatorInternal() const noexcept
    {
        return *this;
    }

    allocator_type& getAllocatorInternal() noexcept
    {
        return *this;
    }

    template <typename... Args>
    static pointer create(allocator_type& alloc, size_type size,
                          size_type timeSteps, Args&&... args)
    {
        pointer p = traits::allocate(alloc, size * timeSteps);
        try {
            constructArraySamples(alloc, p, size * timeSteps,
                                  std::forward<Args>(args)...);
        } catch (...) {
            // Any constructed elements will be destroyed in
            // constructArraySamples, but we're in charge of the memory.
            traits::deallocate(alloc, p, size * timeSteps);
            throw;
        }
        return p;
    }

    static pointer copyValue(allocator_type& alloc, pointer oldData,
                             size_type size, size_type timeSteps)
    {
        pointer p = traits::allocate(alloc, size * timeSteps);
        try {
            copyArraySamples(alloc, p, oldData, size * timeSteps);
        } catch (...) {
            traits::deallocate(alloc, p, size * timeSteps);
            throw;
        }
        return p;
    }

    static pointer copyMove(allocator_type& alloc, pointer oldData,
                            size_type size, size_type timeSteps)
    {
        pointer p = traits::allocate(alloc, size * timeSteps);
        if (!std::is_nothrow_move_constructible<value_type>::value) {
            try {
                // If a move fails, we can't unwind, so we copy if move can throw.
                copyArraySamples(alloc, p, oldData, size * timeSteps);
            } catch (...) {
                traits::deallocate(alloc, p, size * timeSteps);
                throw;
            }
        } else {
            for (size_type i = 0; i < size; ++i) {
                traits::construct(alloc, p + i, std::move(oldData[i]));
            }
        }
        return p;
    }

    void change_memory_size(size_type size)
    {
        using std::swap; // Allow ADL
        pointer p = traits::allocate(getAllocatorInternal(), size * mTimeSteps);
        try {
            moveArraySamples(getAllocatorInternal(), p, mData,
                             mSize * mTimeSteps);
        } catch (...) {
            // moveArraySamples may throw, in that case, we have to
            // deallocate the memory we've already allocated.
            traits::deallocate(getAllocatorInternal(), p, size * mTimeSteps);
            throw;
        }
        // Destroy the old values. We (justifiably) assume that
        // destruction does not throw.
        destroyArraySamples(getAllocatorInternal(), mData, mSize * mTimeSteps);
        // Now that all of the throwing operations are done, we can modify
        // the state of our object with non-throwing operations.
        swap(p, mData);
        traits::deallocate(getAllocatorInternal(), p, mCapacity * mTimeSteps);
        mCapacity = size;
    }

    void destroy() noexcept
    {
        if (mData) {
            destroyArraySamples(getAllocatorInternal(), mData,
                                mSize * mTimeSteps);
            traits::deallocate(getAllocatorInternal(), mData,
                               mCapacity * mTimeSteps);
        }
    }

    size_type mCapacity;
    size_type mSize;
    size_type mTimeSteps;
    pointer mData;
};

} // namespace geom
} // namespace moonray

