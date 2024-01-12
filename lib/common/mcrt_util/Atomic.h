// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace moonray {
namespace util {

// TODO: This is duplicated from scene_rdl2 AtomicFloat
namespace atomic_detail {

// We do a lot of casting to integer values for the intrinsics interface. Let's make sure we are casting the values we
// think we are.
static_assert(__ATOMIC_RELAXED == static_cast<int>(std::memory_order_relaxed));
static_assert(__ATOMIC_CONSUME == static_cast<int>(std::memory_order_consume));
static_assert(__ATOMIC_ACQUIRE == static_cast<int>(std::memory_order_acquire));
static_assert(__ATOMIC_RELEASE == static_cast<int>(std::memory_order_release));
static_assert(__ATOMIC_ACQ_REL == static_cast<int>(std::memory_order_acq_rel));
static_assert(__ATOMIC_SEQ_CST == static_cast<int>(std::memory_order_seq_cst));

// In compare_exchange overloads where only one memory order is given, we have
// to decide on the other. These are what is laid out by the standard.
constexpr std::memory_order compare_exchange_duo(std::memory_order in) noexcept
{
    constexpr std::memory_order mapping[6] = {
            /* std::memory_order_relaxed -> */ std::memory_order_relaxed,
            /* std::memory_order_consume -> */ std::memory_order_consume,
            /* std::memory_order_acquire -> */ std::memory_order_acquire,
            /* std::memory_order_release -> */ std::memory_order_relaxed,
            /* std::memory_order_acq_rel -> */ std::memory_order_acquire,
            /* std::memory_order_seq_cst -> */ std::memory_order_seq_cst

    };
    return mapping[static_cast<int>(in)];
}
} // namespace atomic_detail

// A double-quad-word is 16 bytes == 128 bits.
// Aligning on the size of the data type guarantees that we are not splitting a cache-line. This does, however, mean
// that there may be false sharing. If you want to avoid false sharing, you should explicitly align on cache line
// size.
//
// Cache-line size is defined as CACHE_LINE_SIZE or std::hardware_destructive_interference_size (C++17).
constexpr std::size_t kDoubleQuadWordAtomicAlignment = 16u;

template <typename T>
constexpr std::size_t atomicAlignment() noexcept
{
    // Aligning on the size of the data type guarantees that we are not splitting a cache-line. This does, however, mean
    // that there may be false sharing. If you want to avoid false sharing, you should explicitly align on cache line
    // size.
    //
    // Cache-line size is defined as CACHE_LINE_SIZE or std::hardware_destructive_interference_size (C++17).
    return sizeof(T);
}

// Up to and including ICC 19/GCC 8.5, the library doesn't define the nested value_type for std:atomic (boo!). Write our
// own value_type helper.
template <typename T>
struct atomic_value_type
{
    using value_type = T;
};

template <typename T>
struct atomic_value_type<std::atomic<T>>
{
    using value_type = T;
};

template <typename T>
inline T
atomicLoad(const T* v, std::memory_order order = std::memory_order_seq_cst) noexcept
{
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), v));
    alignas(T) unsigned char buf[sizeof(T)];
    auto* const dest = reinterpret_cast<T*>(buf);
    __atomic_load(v, dest, static_cast<int>(order));
    return *dest;
}

template <typename T>
void
atomicStore(T* v, T val, std::memory_order order = std::memory_order_seq_cst) noexcept
{
    __atomic_store(v, std::addressof(val), static_cast<int>(order));
}

template <typename T>
inline T
atomicCompareAndSwapWeak(T* v, T& expected, T desired, std::memory_order success, std::memory_order failure) noexcept
{
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), v));
    return __atomic_compare_exchange(v,
                                     std::addressof(expected),
                                     std::addressof(desired),
                                     true,
                                     static_cast<int>(success),
                                     static_cast<int>(failure));
}

template <typename T>
inline bool
atomicCompareAndSwapWeak(T* v, T& expected, T desired, std::memory_order order = std::memory_order_seq_cst) noexcept
{
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), v));
    return atomicCompareAndSwapWeak(v, expected, desired, order, atomic_detail::compare_exchange_duo(order));
}

template <typename T>
inline T
atomicCompareAndSwapStrong(T* v, T& expected, T desired, std::memory_order success, std::memory_order failure) noexcept
{
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), v));
    return __atomic_compare_exchange(v,
                                     std::addressof(expected),
                                     std::addressof(desired),
                                     false,
                                     static_cast<int>(success),
                                     static_cast<int>(failure));
}

template <typename T>
inline T
atomicCompareAndSwapStrong(T* v, T& expected, T desired, std::memory_order order = std::memory_order_seq_cst) noexcept
{
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), v));
    return atomicCompareAndSwapStrong(v, expected, desired, order, atomic_detail::compare_exchange_duo(order));
}

template <typename T>
inline void
atomicAssignFloat(T* val, T newValue, std::memory_order order = std::memory_order_seq_cst) noexcept
{
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), val));
    __atomic_store(val, std::addressof(newValue), static_cast<int>(order));
}

// We want to store the min of the two values (_a_, _b_) in _a_
template <typename T>
inline void
atomicMin(T* a, T b) noexcept
{
    alignas(atomicAlignment<T>()) T x = atomicLoad(a, std::memory_order_relaxed);
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), a));
    do {
        // If _x_ is less than or equal to _b_, our work is done. If some other thread calls in, the value of _x_ is
        // only going to get lower, and b is still greater.
        if (x <= b) {
            break;
        }
        // If we get to the CAS, we know that _b_ is less than _a_ (unless another thread preempted us). Update _a_ to
        // the value of _b_. If it succeeds, we're done! If it fails, _x_ is updated to the new value of _a_, and we
        // will continue and check if the new value is less than _b_.
    } while (!atomicCompareAndSwapWeak(a, x, b, std::memory_order_relaxed));
}

// We want to store the max of the two values (_a_, _b_) in _a_
template <typename T>
inline void
atomicMax(T* a, T b) noexcept
{
    alignas(atomicAlignment<T>()) T x = atomicLoad(a, std::memory_order_relaxed);
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), a));
    do {
        // If _x_ is greater than or equal to _b_, our work is done. If some other thread calls in, the value of x is
        // only going to get higher, and b is still smaller.
        if (b <= x) {
            break;
        }
        // If we get to the CAS, we know that _b_ is greater than _a_ (unless another thread preempted us). Update _a_
        // to the value of _b_. If it succeeds, we're done! If it fails, _x_ is updated to the new value of _a_, and we
        // will continue and check if the new value is greater than _b_.
    } while (!atomicCompareAndSwapWeak(a, x, b, std::memory_order_relaxed));
}

// We want to store the sum of the two values (_a_, _b_) in _a_
template <typename T>
inline void
atomicAdd(T* a, T b) noexcept
{
    alignas(atomicAlignment<T>()) T oldVal = atomicLoad(a, std::memory_order_relaxed);
    alignas(atomicAlignment<T>()) T newVal = oldVal + b;
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(T), a));
    while (!atomicCompareAndSwapWeak(a, oldVal, newVal, std::memory_order_relaxed)) {
        newVal = oldVal + b;
    }
}

inline void
atomicLoadFloat4(float* __restrict dst, const float* __restrict src)
{
    // We don't allow our array of four floats to overlap since we're using restrict.
    MNRY_ASSERT(dst + 4 <= src || src + 4 <= dst);

    MNRY_ASSERT(reinterpret_cast<uintptr_t>(dst) % kDoubleQuadWordAtomicAlignment == 0);
    MNRY_ASSERT(reinterpret_cast<uintptr_t>(src) % kDoubleQuadWordAtomicAlignment == 0);

    // assumes 1) dst and src point to aligned float[4] types
    //         2) These types should produce lock-free atomics

    // We don't define this struct as aligned, because that changes the size.
    // So, for instance, if somebody decides that we should align all
    // double-quad words on 64-bytes (to avoid false sharing), that will make
    // the structure 64-bytes, and probably no longer atomic.
    struct Float4Aligned
    {
        float x;
        float y;
        float z;
        float w;
    };

    static_assert(sizeof(Float4Aligned) == 16,
                  "If it's bigger, our architecture (at the time of this writing) won't make it atomic");

    const auto dstStruct = static_cast<Float4Aligned*>(
        __builtin_assume_aligned(reinterpret_cast<void*>(dst), kDoubleQuadWordAtomicAlignment));
    const auto srcStruct = static_cast<const Float4Aligned*>(
        __builtin_assume_aligned(reinterpret_cast<const void*>(src), kDoubleQuadWordAtomicAlignment));

    MNRY_ASSERT(reinterpret_cast<uintptr_t>(dstStruct) % kDoubleQuadWordAtomicAlignment == 0);
    MNRY_ASSERT(reinterpret_cast<uintptr_t>(srcStruct) % kDoubleQuadWordAtomicAlignment == 0);
    MNRY_ASSERT(__atomic_is_lock_free(sizeof(Float4Aligned), srcStruct));

    // This may not be true, because it's architecture dependent, which the compiler doesn't know.
    // static_assert(__atomic_always_lock_free(sizeof(Float4Aligned), nullptr), "");

    __atomic_load(srcStruct, dstStruct, __ATOMIC_RELAXED);
}

inline void
atomicAssignIfClosest(float* __restrict val, const float* __restrict newVal)
{
    // We don't allow our array of four floats to overlap since we're using restrict.
    MNRY_ASSERT(val + 4 <= newVal || newVal + 4 <= val);

    MNRY_ASSERT(reinterpret_cast<uintptr_t>(val) % kDoubleQuadWordAtomicAlignment == 0);
    MNRY_ASSERT(reinterpret_cast<uintptr_t>(newVal) % kDoubleQuadWordAtomicAlignment == 0);

    // assumes 1) val and newVal point to aligned float[4] types
    //         2) val[3] and newVal[3] store the depth
    //         3) These types should produce lock-free atomics

    // We don't define this struct as aligned, because that changes the size.
    // So, for instance, if somebody decides that we should align all
    // double-quad words on 64-bytes (to avoid false sharing), that will make
    // the structure 64-bytes, and probably no longer atomic.
    struct Float4Aligned
    {
        float x, y, z, d;
    };

    static_assert(sizeof(Float4Aligned) == 16,
                  "If it's bigger, our architecture (at the time of this writing) won't make it atomic");

    const auto dest = static_cast<std::atomic<Float4Aligned>*>(
        __builtin_assume_aligned(reinterpret_cast<void*>(val), kDoubleQuadWordAtomicAlignment));
    const auto srcVal = *static_cast<const Float4Aligned*>(
        __builtin_assume_aligned(reinterpret_cast<const void*>(newVal), kDoubleQuadWordAtomicAlignment));

    MNRY_ASSERT(std::atomic_is_lock_free(dest));

    // This fails on GCC 9.3, but succeeds on GCC 6.3 and ICC 19.
    // However, the run-time lock-free check above succeeds.
    // There are valid reasons why this may be the case: it's just interesting
    // that later compilers are less strict here.
    // static_assert(__atomic_always_lock_free(sizeof(Float4Aligned), dest), "");

    auto observed = dest->load(std::memory_order_relaxed);
    do {
        // If _observed.d_ is less than or equal to _srcVal.d_, our work is done. If some other thread calls in, the
        // value of _dest.d_ is only going to get lower, so _srcVal.d_ is still greater.
        if (observed.d <= srcVal.d) {
            break;
        }
        // If we get to the CAS, we know that _srcVal.d_ is less than _dest->d_ (unless another thread preempted us).
        // Update _dest_ to the value of _srcVal_. If it succeeds, we're done! If it fails, _observed_ is updated to the
        // new value of _dest_ and we will continue and check if the new value is less than _srcVal_.
    } while (!dest->compare_exchange_weak(observed, srcVal, std::memory_order_relaxed));
}

} // namespace util
} // namespace moonray

