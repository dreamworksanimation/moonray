// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once

#include <scene_rdl2/common/math/Constants.h>

namespace moonray {
namespace mcrt_common {

struct BaseTLState;
class ThreadLocalState;

void threadSleep();
void threadYield();

//-----------------------------------------------------------------------------

finline bool
isLittleEndian()
{
    unsigned uno = 1;
    return (*((char *)&uno) == 1) ? true : false;
}

// Fill a block of data with NaNs. Used for debugging.
finline void
debugFillWithNaNs(void *buffer, unsigned numBytes)
{
    MNRY_ASSERT((((intptr_t)buffer) & 3) == 0);

    unsigned numFloats = numBytes / sizeof(float);

    if (numFloats) {
        float *fltBuf = (float *)buffer;
        for (unsigned i = 0; i < numFloats; ++i) {
            fltBuf[i] = scene_rdl2::math::NaNTy();
        }

        // A NaN is the only float that when compared to itself will return false.
        MNRY_ASSERT(fltBuf[0] != fltBuf[0]);
    }
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

//
// This function can be called on either a BaseTLState or ThreadLocalState object
// without seeing the concrete layout of the objects. This saves us from having
// to include ThreadLocalState.h any time we want to profile something using
// profile accumulators.
//
// In ThreadLocalState.h we statically assert that the offset of the thread index
// is 4 bytes long and is always placed 8 bytes into both the ThreadLocalState
// and the TLState objects.
//
finline uint32_t
getThreadIdx(const void *tls)
{
    return ((const uint32_t *)tls)[2];
}

finline uint32_t
getThreadIdx(const BaseTLState *tls)
{
    return getThreadIdx((const void *)tls);
}

finline uint32_t
getThreadIdx(const ThreadLocalState *tls)
{
    return getThreadIdx((const void *)tls);
}

// This is just a debugging aid. Note: the thread ID is distinct from the thread
// index stored in our TLS objects.
void debugPrintThreadID(const char *contextString);

// This is just a debugging aid. Display the current callstack when called.
void debugPrintCallstack(const char *contextString);

// Special versions without string args for use with ISPC.
extern "C"
{
// This function isn't useful until ISPC starts providing correct debug information.
void CPP_debugPrintCallstack();
void CPP_debugPrintThreadID();
}
// This function round a floating point number to a certain lowest significant bit from the right
// Rounding is away from zero
finline float roundFloat(const float in, const uint8_t lsb)
{
    float out = in;
    unsigned int *outInt = reinterpret_cast<unsigned int*>(&out);
    if (((*outInt) & 0x7f800000) == 0) return 0; // make all denormalized coding zero
    if (((*outInt) | 0x807fffff) == 0xffffffff) return out; // Inf and NaN remains the same
    *outInt += 1<<lsb;
    *outInt &= ((unsigned int)(-1))<<lsb;
    return out;
}

// This function performs Kahan algorithm to accumulate the data in serial
// c is the running compensation for lost low-order bits
finline void kahanSum(float& sum, float& c, const float in)
{
    float y = in - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// This class exists to aid fast and safe transposition of AOS to SOA data.
// - By making this object 32-bit exactly, we can use SIMD intrinsics to do the
//   transposition directly without needing to write another custom code path.
// - By being able to store 32 flags in a single 32-bit value, it cuts down on
//   storage required compared to bools, and hence saves SOA transposition bandwidth.
// - Bools are ill defined in ISPC so we don't want to be using ISPC structures
//   with bools, which implies removing them from C++ structures which we want
//   to transpose.
//
class Flags
{
public:
    explicit Flags(uint32_t initFlags = 0) : mBits(initFlags)
    {}

    finline bool get(uint32_t flag) const
    {
        return (mBits & flag) != 0;
    }

    finline uint32_t getBits(uint32_t mask) const
    {
        return mBits & mask;
    }

    finline uint32_t getAll() const
    {
        return mBits;
    }

    finline void set(uint32_t flag)
    {
        mBits |= flag;
    }

    finline void set(uint32_t flag, bool on)
    {
        if (on) {
            set(flag);
        } else {
            clear(flag);
        }
    }

    finline void setBits(uint32_t bits)
    {
        mBits = bits;
    }

    finline void setAll()
    {
        mBits = 0xffffffff;
    }

    finline void toggle(uint32_t flag)
    {
        mBits ^= flag;
    }

    finline void toggleAll()
    {
        mBits = ~mBits;
    }

    finline void clear(uint32_t flag)
    {
        mBits &= ~flag;
    }

    finline void clearAll()
    {
        mBits = 0;
    }

protected:
    uint32_t mBits;
};

//-----------------------------------------------------------------------------

} // namespace mcrt_common

using mcrt_common::getThreadIdx;

inline unsigned
getAccumulatorThreadIndex(mcrt_common::BaseTLState *tls)
{
    return getThreadIdx(tls);
}

inline unsigned
getAccumulatorThreadIndex(mcrt_common::ThreadLocalState *tls)
{
    return getThreadIdx(tls);
}

} // namespace moonray

