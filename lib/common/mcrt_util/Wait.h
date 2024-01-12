// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "CPPSupport.h"

#include <scene_rdl2/common/platform/Platform.h>

#include <atomic>
#include <cstring>
#include <system_error>
#include <thread>
#include <type_traits>

#if defined(_MSC_FULL_VER)
#include <windows.h>
#elif defined (__ICC)
#include <immintrin.h>
#endif

// Much of this code mimics the code in GCC 12.1's atomic_wait code. The general problem is that futexes are only
// defined on 32-bit integers, and we have to make it work on the general types supported by std::atomic.

#if defined(__GNUC__)
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if !defined(_GLIBCXX_HAVE_LINUX_FUTEX)
#error Expecting platform wait support
#endif

inline void do_pause()
{
#if defined(_MSC_FULL_VER)
    YieldProcessor();
#elif defined (__ICC)
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
}

namespace wait_impl
{
#if defined(_GLIBCXX_HAVE_LINUX_FUTEX)
using platform_wait_t = int;
constexpr size_t platform_wait_alignment = 4;
#else
using platform_wait_t = uint64_t;
constexpr size_t platform_wait_alignment = alignof(platform_wait_t);
#endif

#if defined(_GLIBCXX_HAVE_LINUX_FUTEX)
template <typename T>
constexpr bool platform_wait_uses_type = std::is_scalar<T>::value && ((sizeof(T) == sizeof(platform_wait_t)) &&
                                                                      (alignof(T*) >= platform_wait_alignment));
#else
constexpr bool platform_wait_uses_type = false;
#endif

// This function isn't an atomic function, but it does the comparison required for atomic wait comparisons (which is
// bitwise excluding padding bits).
template <typename T>
bool atomic_compare(const T& a, const T& b) noexcept
{
    // This doesn't ignore the padding bits, but, to be fair, neither does GCC 12.1.
    return std::memcmp(std::addressof(a), std::addressof(b), sizeof(T)) == 0;
}

struct DefaultSpinPolicy
{
    bool operator()() const noexcept { return false; }
};

template <typename Pred, typename Spin = DefaultSpinPolicy>
inline bool atomic_spin(Pred pred, Spin spin = Spin{}) noexcept
{
    constexpr int k_pause_iter = 12;
    constexpr int k_yield_iter =  4;

    for (int i = 0; i < k_pause_iter; ++i) {
        if (pred()) {
            return true;
        }
        do_pause();
    }

    for (int i = 0; i < k_yield_iter; ++i) {
        if (pred()) {
            return true;
        }
        std::this_thread::yield();
    }

    while (spin()) {
        if (pred()) {
            return true;
        }
    }

    return false;
}

template <typename Pred>
inline bool spin_lock(Pred pred) noexcept
{
    return atomic_spin(pred, []{ return true; });
}

#if defined(__GNUC__)
template <typename T>
inline bool wait_aligned(const T* ptr) noexcept
{
    return reinterpret_cast<std::uintptr_t>(ptr) % platform_wait_alignment == 0;
}

template <typename T>
void platform_wait(const T* addr, platform_wait_t old)
{
    MNRY_ASSERT(wait_aligned(addr));
    const auto e = syscall(SYS_futex, static_cast<const void*>(addr), FUTEX_WAIT_PRIVATE, old, nullptr, nullptr, 0);
    if (e == 0 || errno == EAGAIN) {
        return;
    }
    if (e != EINTR) {
        throw std::system_error(errno, std::generic_category(), "Failure in futex wait");
    }
}

template <typename T>
void platform_notify_all(const T* addr) noexcept
{
    MNRY_ASSERT(wait_aligned(addr));
    syscall(SYS_futex, static_cast<const void*>(addr), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

template <typename T>
void platform_notify_one(const T* addr) noexcept
{
    MNRY_ASSERT(wait_aligned(addr));
    syscall(SYS_futex, static_cast<const void*>(addr), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

struct WaiterPoolBase
{
#if defined(__cpp_lib_hardware_interference_size)
    static constexpr size_t s_align = std::hardware_destructive_interference_size;
#else
    static constexpr size_t s_align = 64;
#endif

    alignas(s_align) platform_wait_t m_wait = 0;
    alignas(s_align) platform_wait_t m_ver = 0;

    WaiterPoolBase() = default;

    static WaiterPoolBase& s_for(const void* addr) noexcept
    {
        constexpr uintptr_t ct = 16;
        static WaiterPoolBase w[ct];
        const auto key = (reinterpret_cast<uintptr_t>(addr) >> 2) % ct;
        return w[key];
    }

    void enter_wait() noexcept { __atomic_fetch_add(&m_wait, 1, __ATOMIC_SEQ_CST); }

    void leave_wait() noexcept { __atomic_fetch_sub(&m_wait, 1, __ATOMIC_RELEASE); }

    bool waiting() const noexcept
    {
        platform_wait_t res;
        __atomic_load(std::addressof(m_wait), std::addressof(res), __ATOMIC_SEQ_CST);
        return res != 0;
    }

    void notify(const platform_wait_t* addr, bool all) noexcept
    {
        if (!waiting()) {
            return;
        }

        if (all) {
            platform_notify_all(addr);
        } else {
            platform_notify_one(addr);
        }
    }

protected:
    ~WaiterPoolBase() = default;
};

#pragma warning push
#pragma warning disable 444 // destructor for base class isn't virtual
struct WaiterPool : WaiterPoolBase
{
    void do_wait(const platform_wait_t* addr, platform_wait_t old) { platform_wait(addr, old); }
};
#pragma warning pop

template <bool uses_platform_type>
struct SpinHelper;

template <>
struct SpinHelper<true /*uses platform type*/>
{
    template <typename U, typename ValFn>
    static bool do_spin_v(platform_wait_t* /*addr*/, const U& old, ValFn vfn, platform_wait_t& val)
    {
        const auto pred = [old, vfn] { return !atomic_compare(old, vfn()); };
        std::memcpy(std::addressof(val), std::addressof(old), sizeof(val));
        return atomic_spin(pred);
    }
};

template <>
struct SpinHelper<false /*uses platform type*/>
{
    template <typename U, typename ValFn>
    static bool do_spin_v(platform_wait_t* addr, const U& old, ValFn vfn, platform_wait_t& val)
    {
        const auto pred = [old, vfn] { return !atomic_compare(old, vfn()); };
        __atomic_load(addr, &val, __ATOMIC_ACQUIRE);
        return atomic_spin(pred);
    }
};

template <bool uses_platform_type>
struct WaitHelper;

template <>
struct WaitHelper<true /*uses platform type*/>
{
    template <typename U>
    static platform_wait_t* s_wait_addr(const U* a, platform_wait_t* /*b*/)
    {
        return reinterpret_cast<platform_wait_t*>(const_cast<U*>(a));
    }
};

template <>
struct WaitHelper<false /*uses platform type*/>
{
    template <typename U>
    static platform_wait_t* s_wait_addr(const U* /*a*/, platform_wait_t* b)
    {
        return b;
    }
};

template <typename T>
struct WaiterBase
{
    using WaiterType = T;

    WaiterType& m_w;
    platform_wait_t* m_addr;

    template <typename U>
    explicit WaiterBase(const U* addr) noexcept
    : m_w(s_for(addr))
    , m_addr(s_wait_addr(addr, std::addressof(m_w.m_ver)))
    {
    }

    template <typename U>
    static platform_wait_t* s_wait_addr(const U* a, platform_wait_t* b)
    {
        // We can't do "if constexpr" in our version of C++, so we have to dispatch. :'(
        return WaitHelper<platform_wait_uses_type<U>>::s_wait_addr(a, b);
    }

    static WaiterType& s_for(const void* addr) noexcept
    {
        static_assert(sizeof(WaiterType) == sizeof(WaiterPoolBase));
        auto& res = WaiterPoolBase::s_for(addr);
        return reinterpret_cast<WaiterType&>(res);
    }

    template <typename U, typename ValFn>
    static bool do_spin_v(platform_wait_t* addr, const U& old, ValFn vfn, platform_wait_t& val)
    {
        // We can't do "if constexpr" in our version of C++, so we have to dispatch. :'(
        return SpinHelper<platform_wait_uses_type<U>>::do_spin_v(addr, old, vfn, val);
    }

    template <typename U, typename ValFn>
    bool do_spin_v(const U& old, ValFn vfn, platform_wait_t& val)
    {
        return WaiterBase::do_spin_v(m_addr, old, vfn, val);
    }

    bool laundered() const noexcept
    {
        return m_addr == std::addressof(m_w.m_ver);
    }

    void notify(bool all) noexcept
    {
        if (laundered()) {
            __atomic_fetch_add(m_addr, 1, __ATOMIC_SEQ_CST);
            all = true;
        }
        m_w.notify(m_addr, all);
    }

protected:
    WaiterBase() = default;
};

#pragma warning push
#pragma warning disable 444 // destructor for base class isn't virtual
template <typename EntersWait>
struct Waiter : WaiterBase<WaiterPool>
{
    using BaseType = WaiterBase<WaiterPool>;

    template <typename T>
    explicit Waiter(const T* addr) noexcept
    : BaseType(addr)
    {
        IF_CONSTEXPR (EntersWait::value) {
            m_w.enter_wait();
        }
    }

    ~Waiter()
    {
        IF_CONSTEXPR (EntersWait::value) {
            m_w.leave_wait();
        }
    }

    template <typename T, typename ValFn>
    void do_wait_v(T old, ValFn vfn)
    {
        do {
            platform_wait_t val;
            if (BaseType::do_spin_v(old, vfn, val)) {
                return;
            }
            BaseType::m_w.do_wait(BaseType::m_addr, val);
        } while (atomic_compare(old, vfn()));
    }
};
#pragma warning pop

using enters_wait = Waiter<std::true_type>;
using bare_wait = Waiter<std::false_type>;

template <typename T, typename ValFn>
inline void atomic_wait_address_v(const T* addr, T old, ValFn vfn) noexcept
{
#pragma warning push
#pragma warning disable 444 // destructor for base class isn't virtual
    enters_wait w(addr);
    w.do_wait_v(old, vfn);
#pragma warning pop
}

template <typename T>
void atomic_notify_address(const T* addr, bool all) noexcept
{
#pragma warning push
#pragma warning disable 444 // destructor for base class isn't virtual
    bare_wait w(addr);
    w.notify(all);
#pragma warning pop
}

#endif // #if defined(__GNUC__)
} // namespace wait_impl

template <typename T>
inline void wait(const std::atomic<T>& a, T old, std::memory_order order = std::memory_order_seq_cst) noexcept
{
#if defined(__cpp_lib_atomic_wait)
    a.wait(old, order);
#elif defined(__GNUC__)
    const T* const addr = reinterpret_cast<const T*>(std::addressof(a));
    wait_impl::atomic_wait_address_v(addr, old, [order, &a] { return a.load(order); });
#else
    wait_impl::spin_lock([&a, old, order]() { return a.load(order) != old; });
#endif
}

template <typename T>
inline void notify_one(std::atomic<T>& a) noexcept
{
#if defined(__cpp_lib_atomic_wait)
    a.notify_one();
#elif defined(__GNUC__)
    const T* const addr = reinterpret_cast<const T*>(std::addressof(a));
    wait_impl::atomic_notify_address(addr, false);
#endif
}

template <typename T>
inline void notify_all(std::atomic<T>& a) noexcept
{
#if defined(__cpp_lib_atomic_wait)
    a.notify_all();
#elif defined(__GNUC__)
#pragma warning push
#pragma warning disable 444 // destructor for base class isn't virtual
    const T* const addr = reinterpret_cast<const T*>(std::addressof(a));
    wait_impl::atomic_notify_address(addr, true);
#pragma warning pop
#endif
}

