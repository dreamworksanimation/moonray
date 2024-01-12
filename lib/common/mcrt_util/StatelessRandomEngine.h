// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <scene_rdl2/render/util/BitUtils.h>

#include <scene_rdl2/common/math/sse.h>
#include <Random123/threefry.h>

#include <array>
#include <cstring>

namespace moonray {
namespace util {

class StatelessRandomEngine
{
public:
    typedef r123::Threefry4x32 EngineType;
    typedef EngineType::ctr_type CounterType;
    typedef EngineType::key_type KeyType;
    typedef std::array<float, 4> FloatType;

    explicit StatelessRandomEngine(uint32_t key) :
        mEngine(),
        mKey{key+0, key+1, key+2, key+3}
    {
    }

    CounterType asUint32(uint32_t n)
    {
        const CounterType c = { n, n, n, n };
        return mEngine(c, mKey);
    }

    FloatType asFloat(uint32_t n)
    {
        const CounterType c = asUint32(n);
        const simd::ssef f = scene_rdl2::util::bitsToFloat(simd::ssei(c[0], c[1], c[2], c[3]));
        FloatType ret;
        std::memcpy(ret.data(), f.f, sizeof(float) * 4);
        return ret;
    }

private:
    EngineType mEngine;
    KeyType mKey;
};

} // namespace util
} // namespace moonray

