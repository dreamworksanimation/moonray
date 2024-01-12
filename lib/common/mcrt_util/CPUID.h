// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <type_traits>

#if defined(__WIN32__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#pragma warning push
#pragma warning(disable:1711) // Assignment to static variable

namespace moonray {
namespace util {

namespace cpuid_detail
{
enum class CPUFeatures : std::uint64_t
{
    NONE                   = 0,
    CPU_FEATURE_SSE        = 0b0000'0000'0000'0000'0000'0001,
    CPU_FEATURE_SSE2       = 0b0000'0000'0000'0000'0000'0010,
    CPU_FEATURE_SSE3       = 0b0000'0000'0000'0000'0000'0100,
    CPU_FEATURE_SSSE3      = 0b0000'0000'0000'0000'0000'1000,
    CPU_FEATURE_SSE41      = 0b0000'0000'0000'0000'0001'0000,
    CPU_FEATURE_SSE42      = 0b0000'0000'0000'0000'0010'0000,
    CPU_FEATURE_POPCNT     = 0b0000'0000'0000'0000'0100'0000,

    CPU_FEATURE_AVX        = 0b0000'0000'0000'0000'1000'0000,
    CPU_FEATURE_F16C       = 0b0000'0000'0000'0001'0000'0000,
    CPU_FEATURE_RDRAND     = 0b0000'0000'0000'0010'0000'0000,
    CPU_FEATURE_AVX2       = 0b0000'0000'0000'0100'0000'0000,
    CPU_FEATURE_FMA4       = 0b0000'0000'0000'1000'0000'0000,
    CPU_FEATURE_LZCNT      = 0b0000'0000'0001'0000'0000'0000,
    CPU_FEATURE_BMI1       = 0b0000'0000'0010'0000'0000'0000,
    CPU_FEATURE_BMI2       = 0b0000'0000'0100'0000'0000'0000,

    CPU_FEATURE_AVX512F    = 0b0000'0000'1000'0000'0000'0000,
    CPU_FEATURE_AVX512DQ   = 0b0000'0001'0000'0000'0000'0000,
    CPU_FEATURE_AVX512PF   = 0b0000'0010'0000'0000'0000'0000,
    CPU_FEATURE_AVX512ER   = 0b0000'0100'0000'0000'0000'0000,
    CPU_FEATURE_AVX512CD   = 0b0000'1000'0000'0000'0000'0000,
    CPU_FEATURE_AVX512BW   = 0b0001'0000'0000'0000'0000'0000,
    CPU_FEATURE_AVX512IFMA = 0b0010'0000'0000'0000'0000'0000,
    CPU_FEATURE_AVX512VL   = 0b0100'0000'0000'0000'0000'0000,
    CPU_FEATURE_AVX512VBMI = 0b1000'0000'0000'0000'0000'0000
};

CPUFeatures operator|(CPUFeatures a, CPUFeatures b) noexcept
{
    using underlying_type = typename std::underlying_type<CPUFeatures>::type;
    return static_cast<CPUFeatures>(static_cast<underlying_type>(a) | static_cast<underlying_type>(b));
}

CPUFeatures& operator|=(CPUFeatures& a, CPUFeatures b) noexcept
{
    return a = a | b;
}

CPUFeatures operator&(CPUFeatures a, CPUFeatures b) noexcept
{
    using underlying_type = typename std::underlying_type<CPUFeatures>::type;
    return static_cast<CPUFeatures>(static_cast<underlying_type>(a) & static_cast<underlying_type>(b));
}

CPUFeatures& operator&=(CPUFeatures& a, CPUFeatures b) noexcept
{
    return a = a & b;
}

bool is_set(CPUFeatures collection, CPUFeatures individual) noexcept
{
    return (collection & individual) == individual;
}

} // namespace cpuid_detail

class CPUID
{
public:
    CPUID()
    {
        static std::once_flag flag;
        std::call_once(flag, &CPUID::init);
    }

    const std::string& vendor() const noexcept
    {
        return s_vendor;
    }

    bool atomic_8() const noexcept
    {
        return s_atomic_sizes.a8;
    }

    bool atomic_16() const noexcept
    {
        return s_atomic_sizes.a16;
    }

    bool atomic_32() const noexcept
    {
        return s_atomic_sizes.a32;
    }

    bool atomic_64() const noexcept
    {
        return s_atomic_sizes.a64;
    }

    bool atomic_128() const noexcept
    {
        return s_atomic_sizes.a128;
    }

    bool sse() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_SSE);
    }

    bool sse2() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_SSE2);
    }

    bool sse3() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_SSE3);
    }

    bool sse41() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_SSE41);
    }

    bool sse42() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_SSE42);
    }

    bool avx() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_AVX);
    }

    bool avx2() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_AVX2);
    }

    bool avx512() const noexcept
    {
        return cpuid_detail::is_set(s_features, cpuid_detail::CPUFeatures::CPU_FEATURE_AVX512F);
    }

private:
    static constexpr std::size_t log2(std::size_t n) noexcept
    {
        std::size_t r{0};
        while (n >>= 1ULL) {
            ++r;
        }
        return r;
    }

    template<std::size_t size>
    static bool check_atomic()
    {
        constexpr std::size_t index = log2(size) - 3ULL;// We start at 8 == 2^3
        using Tuple = std::tuple<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, __uint128_t>;
        using Type = std::tuple_element_t<index, Tuple>;
        return std::atomic<Type>{}.is_lock_free();
    }

    static std::string get_vendor()
    {
        std::array<unsigned, 4> cpuinfo;
        __get_cpuid(0, &cpuinfo[0], &cpuinfo[1], &cpuinfo[2], &cpuinfo[3]);
        const std::array<unsigned, 4> name = {cpuinfo[1], cpuinfo[3], cpuinfo[2], 0};
        return reinterpret_cast<const char *>(name.data());
    }

    static cpuid_detail::CPUFeatures get_cpu_features()
    {
        using cpuid_detail::CPUFeatures;

        CPUFeatures features = CPUFeatures::NONE;

        //constexpr unsigned EAX = 0U;
        constexpr unsigned EBX = 1U;
        constexpr unsigned ECX = 2U;
        constexpr unsigned EDX = 3U;

        const auto max_ids = __get_cpuid_max(0x00000000U, nullptr);
        if (max_ids == 0) {
            // TODO: warn: cpuid not supported
            return features;
        }
        const auto max_extended_ids = __get_cpuid_max(0x80000000U, nullptr);

        /* get CPUID leaves for EAX = 1,7, and 0x80000001 */
        std::array<unsigned, 4> cpuid_leaf_1  = { 0U, 0U, 0U, 0U };
        std::array<unsigned, 4> cpuid_leaf_7  = { 0U, 0U, 0U, 0U };
        std::array<unsigned, 4> cpuid_leaf_e1 = { 0U, 0U, 0U, 0U };

        if (max_ids >= 1) {
            __get_cpuid(0x00000001U, &cpuid_leaf_1[0], &cpuid_leaf_1[1], &cpuid_leaf_1[2], &cpuid_leaf_1[3]);
        }
        if (max_ids >= 7) {
            __get_cpuid_count(0x00000007, 0x00000000U, &cpuid_leaf_7[0], &cpuid_leaf_7[1], &cpuid_leaf_7[2], &cpuid_leaf_7[3]);
        }
        if (max_extended_ids >= 0x80000001) {
            __get_cpuid(0x80000001U, &cpuid_leaf_e1[0], &cpuid_leaf_e1[1], &cpuid_leaf_e1[2], &cpuid_leaf_e1[3]);
        }

        if (cpuid_leaf_1[EDX] & bit_SSE   )     features |= CPUFeatures::CPU_FEATURE_SSE;
        if (cpuid_leaf_1[EDX] & bit_SSE2  )     features |= CPUFeatures::CPU_FEATURE_SSE2;
        if (cpuid_leaf_1[ECX] & bit_SSE3  )     features |= CPUFeatures::CPU_FEATURE_SSE3;
        if (cpuid_leaf_1[ECX] & bit_SSSE3 )     features |= CPUFeatures::CPU_FEATURE_SSSE3;
        if (cpuid_leaf_1[ECX] & bit_SSE4_1)     features |= CPUFeatures::CPU_FEATURE_SSE41;
        if (cpuid_leaf_1[ECX] & bit_SSE4_2)     features |= CPUFeatures::CPU_FEATURE_SSE42;
        if (cpuid_leaf_1[ECX] & bit_POPCNT)     features |= CPUFeatures::CPU_FEATURE_POPCNT;

        if (cpuid_leaf_1[ECX] & bit_AVX   )     features |= CPUFeatures::CPU_FEATURE_AVX;
        if (cpuid_leaf_1[ECX] & bit_F16C  )     features |= CPUFeatures::CPU_FEATURE_F16C;
        if (cpuid_leaf_1[ECX] & bit_RDRND)      features |= CPUFeatures::CPU_FEATURE_RDRAND;
        if (cpuid_leaf_7[EBX] & bit_AVX2  )     features |= CPUFeatures::CPU_FEATURE_AVX2;
        if (cpuid_leaf_1[ECX] & bit_FMA4  )     features |= CPUFeatures::CPU_FEATURE_FMA4;
        if (cpuid_leaf_e1[ECX] & bit_LZCNT)     features |= CPUFeatures::CPU_FEATURE_LZCNT;
        if (cpuid_leaf_7 [EBX] & bit_BMI )      features |= CPUFeatures::CPU_FEATURE_BMI1;
        if (cpuid_leaf_7 [EBX] & bit_BMI2 )     features |= CPUFeatures::CPU_FEATURE_BMI2;

        if (cpuid_leaf_7[EBX] & bit_AVX512F   ) features |= CPUFeatures::CPU_FEATURE_AVX512F;
        if (cpuid_leaf_7[EBX] & bit_AVX512DQ  ) features |= CPUFeatures::CPU_FEATURE_AVX512DQ;
        if (cpuid_leaf_7[EBX] & bit_AVX512PF  ) features |= CPUFeatures::CPU_FEATURE_AVX512PF;
        if (cpuid_leaf_7[EBX] & bit_AVX512ER  ) features |= CPUFeatures::CPU_FEATURE_AVX512ER;
        if (cpuid_leaf_7[EBX] & bit_AVX512CD  ) features |= CPUFeatures::CPU_FEATURE_AVX512CD;
        if (cpuid_leaf_7[EBX] & bit_AVX512BW  ) features |= CPUFeatures::CPU_FEATURE_AVX512BW;
        if (cpuid_leaf_7[EBX] & bit_AVX512IFMA) features |= CPUFeatures::CPU_FEATURE_AVX512IFMA;
        if (cpuid_leaf_7[EBX] & bit_AVX512VL  ) features |= CPUFeatures::CPU_FEATURE_AVX512VL;
        if (cpuid_leaf_7[ECX] & bit_AVX512VBMI) features |= CPUFeatures::CPU_FEATURE_AVX512VBMI;
        return features;
    }

    static void init()
    {
        s_vendor            = get_vendor();
        s_features          = get_cpu_features();
        s_atomic_sizes.a8   = check_atomic<8>();
        s_atomic_sizes.a16  = check_atomic<16>();
        s_atomic_sizes.a32  = check_atomic<32>();
        s_atomic_sizes.a64  = check_atomic<64>();
        s_atomic_sizes.a128 = check_atomic<128>();
    }

    struct AtomicSize
    {
        constexpr AtomicSize() noexcept
        : a8(false)
        , a16(false)
        , a32(false)
        , a64(false)
        , a128(false)
        {
        }

        bool a8   : 1;
        bool a16  : 1;
        bool a32  : 1;
        bool a64  : 1;
        bool a128 : 1;
    };

    static std::string s_vendor;
    static AtomicSize s_atomic_sizes;
    static cpuid_detail::CPUFeatures s_features;
};

} // namespace util
} // namespace moonray

#pragma warning pop

