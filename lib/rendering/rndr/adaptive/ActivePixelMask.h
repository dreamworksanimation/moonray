// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/fb_util/Tiler.h>

#include <cstdint>
#include <ostream>

namespace moonray {
namespace rndr {

class ActivePixelMask;

inline std::ostream &operator<<(std::ostream &outs, const ActivePixelMask &mask);

class ActivePixelMask
{
public:
    using size_type                   = unsigned;
    static constexpr size_type width  = COARSE_TILE_SIZE;
    static constexpr size_type height = COARSE_TILE_SIZE;

    constexpr ActivePixelMask() noexcept
    : mActiveMask(sNone)
    {
    }

    static constexpr ActivePixelMask all() noexcept
    {
        return ActivePixelMask(sAll);
    }

    static constexpr ActivePixelMask none() noexcept
    {
        return ActivePixelMask(sNone);
    }

    void clear() noexcept
    {
        mActiveMask = sNone;
    }

    void set(size_type x, size_type y) noexcept
    {
        mActiveMask |= (sOne << offset(x, y));
    }

    void unset(size_type x, size_type y) noexcept
    {
        mActiveMask &= ~(sOne << offset(x, y));
    }

    void set(size_type x, size_type y, bool value) noexcept
    {
        if (value) {
            set(x, y);
        } else {
            unset(x, y);
        }
    }

    bool get(size_type x, size_type y) const noexcept
    {
        return (mActiveMask & (sOne << offset(x, y))) != 0;
    }

    bool operator()(size_type x, size_type y) const noexcept
    {
        return get(x, y);
    }

    bool full() const noexcept
    {
        return *this == all();
    }

    bool empty() const noexcept
    {
        return *this == none();
    }

    explicit operator bool() const noexcept
    {
        return !empty();
    }

    // Fill in any gaps in the axis-aligned bounding-box of the selection set.
    // +----------+
    // |..........|
    // |..x.......|
    // |.....x....|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+
    //
    // becomes
    //
    // +----------+
    // |..........|
    // |..xxxxx...|
    // |..xxxxx...|
    // |..xxxxx...|
    // |..........|
    // |..........|
    // +----------+
    void fillGaps() noexcept
    {
        const auto lbRow = lowerBoundRow();
        const auto ubRow = upperBoundRow();
        const auto lbCol = lowerBoundCol();
        const auto ubCol = upperBoundCol();
        mActiveMask      = lbRow & ubRow & lbCol & ubCol;
    }

    // Add neighboring pixels of selection to selection set. This does not include diagonal neighbors.
    // +----------+
    // |..........|
    // |x.x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+
    //
    // becomes
    //
    // +----------+
    // |x.x.......|
    // |xxxx......|
    // |x.x...x...|
    // |.....xxx..|
    // |......x...|
    // |..........|
    // +----------+
    void dilate() noexcept
    {
        // clang-format off
        mActiveMask |= mActiveMask << 1ULL & 0xFEFEFEFEFEFEFEFEULL |
                       mActiveMask >> 1ULL & 0x7F7F7F7F7F7F7F7FULL |
                       mActiveMask << 8ULL                         |
                       mActiveMask >> 8ULL;
        // clang-format on
    }

    friend ActivePixelMask operator|(const ActivePixelMask& a, const ActivePixelMask& b) noexcept
    {
        return ActivePixelMask(a.mActiveMask | b.mActiveMask);
    }

    friend ActivePixelMask operator&(const ActivePixelMask& a, const ActivePixelMask& b) noexcept
    {
        return ActivePixelMask(a.mActiveMask & b.mActiveMask);
    }

    friend bool operator==(const ActivePixelMask& a, const ActivePixelMask& b) noexcept
    {
        return a.mActiveMask == b.mActiveMask;
    }

private:
    // Returns a mask starting at the numerically lowest row (inclusive) of selection set.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Scan from the top, masking off the first occupied column and subsequent columns.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Results in:
    // +----------+
    // |..........|
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // +----------+
    std::uint64_t lowerBoundRow() const noexcept
    {
        static_assert(height == 8, "This function hard-codes an assumption that we have eight rows");
        // Mask out each row in ascending order.
        // We only care if ANY cell in the row is set.
        // clang-format off

        // With AVX or AVX2, we can store 4 64-bit numbers in a __m256i and can do this in two bitwise ands.
        // With AVX512, we can store 8 64-bit numbers in a __m512i and can do this in one bitwise and.
        // a <- broadcast(mActiveMask)
        // b <- mask literals below
        // c <- a & b
        // However, we still have all of the conditionals to look through. Looking through the assembly, it looks like
        // the compiler (ICC) might already be condensing this down to one bitwise and and a collection of conditionals.

        if (mActiveMask & 0x00000000000000ffULL) return 0xffffffffffffffffULL;
        if (mActiveMask & 0x000000000000ff00ULL) return 0xffffffffffffff00ULL;
        if (mActiveMask & 0x0000000000ff0000ULL) return 0xffffffffffff0000ULL;
        if (mActiveMask & 0x00000000ff000000ULL) return 0xffffffffff000000ULL;
        if (mActiveMask & 0x000000ff00000000ULL) return 0xffffffff00000000ULL;
        if (mActiveMask & 0x0000ff0000000000ULL) return 0xffffff0000000000ULL;
        if (mActiveMask & 0x00ff000000000000ULL) return 0xffff000000000000ULL;
        if (mActiveMask & 0xff00000000000000ULL) return 0xff00000000000000ULL;
        // clang-format on
        return sNone;
    }

    // Returns a mask starting at the numerically highest row (inclusive) of selection set.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Scan from the bottom, masking off the first occupied column and subsequent columns.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Results in:
    // +----------+
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // |xxxxxxxxxx|
    // |..........|
    // |..........|
    // +----------+
    std::uint64_t upperBoundRow() const noexcept
    {
        static_assert(height == 8, "This function hard-codes an assumption that we have eight rows");
        // Mask out each row in descending order.
        // We only care if ANY cell in the row is set.
        // clang-format off
        if (mActiveMask & 0xff00000000000000ULL) return 0xffffffffffffffffULL;
        if (mActiveMask & 0x00ff000000000000ULL) return 0x00ffffffffffffffULL;
        if (mActiveMask & 0x0000ff0000000000ULL) return 0x0000ffffffffffffULL;
        if (mActiveMask & 0x000000ff00000000ULL) return 0x000000ffffffffffULL;
        if (mActiveMask & 0x00000000ff000000ULL) return 0x00000000ffffffffULL;
        if (mActiveMask & 0x0000000000ff0000ULL) return 0x0000000000ffffffULL;
        if (mActiveMask & 0x000000000000ff00ULL) return 0x000000000000ffffULL;
        if (mActiveMask & 0x00000000000000ffULL) return 0x00000000000000ffULL;
        // clang-format on
        return sNone;
    }

    // Returns a mask starting at the numerically lowest column (inclusive) of selection set.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Scan from the left, masking off the first occupied column and subsequent columns.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Results in:
    // +----------+
    // |..xxxxxxxx|
    // |..xxxxxxxx|
    // |..xxxxxxxx|
    // |..xxxxxxxx|
    // |..xxxxxxxx|
    // |..xxxxxxxx|
    // +----------+
    std::uint64_t lowerBoundCol() const noexcept
    {
        static_assert(width == 8, "This function hard-codes an assumption that we have eight columns");
        // Mask out each column in ascending order.
        // We only care if ANY cell in the column is set.
        // clang-format off
        if (mActiveMask & 0x0101010101010101ULL) return 0xffffffffffffffffULL;
        if (mActiveMask & 0x0202020202020202ULL) return 0xfefefefefefefefeULL;
        if (mActiveMask & 0x0404040404040404ULL) return 0xfcfcfcfcfcfcfcfcULL;
        if (mActiveMask & 0x0808080808080808ULL) return 0xf8f8f8f8f8f8f8f8ULL;
        if (mActiveMask & 0x1010101010101010ULL) return 0xf0f0f0f0f0f0f0f0ULL;
        if (mActiveMask & 0x2020202020202020ULL) return 0xe0e0e0e0e0e0e0e0ULL;
        if (mActiveMask & 0x4040404040404040ULL) return 0xc0c0c0c0c0c0c0c0ULL;
        if (mActiveMask & 0x8080808080808080ULL) return 0x8080808080808080ULL;
        // clang-format on
        return sNone;
    }

    // Returns a mask starting at the numerically highest column (inclusive) of selection set.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Scan from the right, masking off the first occupied column and subsequent columns.
    // +----------+
    // |..........|
    // |..x.......|
    // |..........|
    // |......x...|
    // |..........|
    // |..........|
    // +----------+

    // Results in:
    // +----------+
    // |xxxxxxx...|
    // |xxxxxxx...|
    // |xxxxxxx...|
    // |xxxxxxx...|
    // |xxxxxxx...|
    // |xxxxxxx...|
    // +----------+
    std::uint64_t upperBoundCol() const noexcept
    {
        static_assert(width == 8, "This function hard-codes an assumption that we have eight columns");
        // Mask out each column in descending order.
        // We only care if ANY cell in the column is set.
        // clang-format off
        if (mActiveMask & 0x8080808080808080ULL) return 0xffffffffffffffffULL;
        if (mActiveMask & 0x4040404040404040ULL) return 0x7f7f7f7f7f7f7f7fULL;
        if (mActiveMask & 0x2020202020202020ULL) return 0x3f3f3f3f3f3f3f3fULL;
        if (mActiveMask & 0x1010101010101010ULL) return 0x1f1f1f1f1f1f1f1fULL;
        if (mActiveMask & 0x0808080808080808ULL) return 0x0f0f0f0f0f0f0f0fULL;
        if (mActiveMask & 0x0404040404040404ULL) return 0x0707070707070707ULL;
        if (mActiveMask & 0x0202020202020202ULL) return 0x0303030303030303ULL;
        if (mActiveMask & 0x0101010101010101ULL) return 0x0101010101010101ULL;
        // clang-format on
        return sNone;
    }

    static bool checkPreconditions(size_type x, size_type y)
    {
        if (x >= width || y >= height) {
            return false;
        }
        return true;
    }

    static std::uint32_t offset(size_type x, size_type y) noexcept
    {
        MNRY_ASSERT(checkPreconditions(x, y));
        return y * width + x;
    }

    constexpr explicit ActivePixelMask(std::uint64_t mask) noexcept
    : mActiveMask(mask)
    {
    }

    static_assert(width * height <= 64, "We're storing our values in a 64-bit value");

    std::uint64_t                  mActiveMask;
    static constexpr std::uint64_t sOne  = 1ULL;
    static constexpr std::uint64_t sNone = 0ULL;
    static constexpr std::uint64_t sAll  = ~0ULL;
};

inline std::ostream& operator<<(std::ostream& outs, const ActivePixelMask& mask)
{
    const auto flags = outs.setf(std::ios_base::fmtflags(0), std::ios_base::boolalpha);
    for (ActivePixelMask::size_type y = 0; y < ActivePixelMask::height; ++y) {
        for (ActivePixelMask::size_type x = 0; x < ActivePixelMask::width; ++x) {
            outs << mask(x, y);
        }
        outs << '\n';
    }
    outs.setf(flags);
    return outs;
}
} // namespace rndr
} // namespace moonray


