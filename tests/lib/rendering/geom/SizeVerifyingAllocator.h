// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cppunit/extensions/HelperMacros.h>

#include <cassert>
#include <map>
#include <memory>
#include <type_traits>

// This allocator (which should only be used for testing) keeps track of the
// size passed in for allocation and deallocation of memory, to make sure it's
// consistent (most allocators ignore the size on dealloation, making testing
// difficult).
template <typename T>
class SizeVerifyingAllocator
{
public:
    using value_type = T;
    using MapType = std::map<void*, std::size_t>;

    // We use a shared_ptr because we can construct other types of allocators
    // from this one, use it, and then discard it, only to do the same thing
    // later (i.e. rebind). We want to hold all of the pointers from these
    // various incarnations of the allocator.
    using MapTypePointer = std::shared_ptr<MapType>;

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    SizeVerifyingAllocator() :
        mMap(new MapType)
    {
    }

    template <typename U>
    SizeVerifyingAllocator(const SizeVerifyingAllocator<U>& other) :
        mMap(other.get_map())
    {
    }

    T* allocate(std::size_t n)
    {
        T* const p = std::allocator<T>().allocate(n);
        (*mMap)[p] = n;
        return p;
    }

    void deallocate(T *ptr, std::size_t n)
    {
        if (ptr) {
            const auto it = mMap->find(ptr);
            CPPUNIT_ASSERT(it != mMap->end());
            CPPUNIT_ASSERT(it->second == n);
            std::allocator<T>().deallocate(ptr, n);
        }
    }

    MapTypePointer get_map() const
    {
        return mMap;
    }

private:
    MapTypePointer mMap;

    template <typename T1, typename T2>
    friend bool operator==(const SizeVerifyingAllocator<T1> a, const SizeVerifyingAllocator<T2>&b);
};

template <typename T, typename U>
bool operator==(const SizeVerifyingAllocator<T> &a, const SizeVerifyingAllocator<U>&b)
{
    return a.mMap == b.mMap;
}

template <typename T, typename U>
bool operator!=(const SizeVerifyingAllocator<T>&a, const SizeVerifyingAllocator<U>&b)
{
    return !(a == b);
}


