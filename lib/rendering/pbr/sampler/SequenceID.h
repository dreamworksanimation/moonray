// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gphash.h"
#include "SamplingConstants.h"

#include <scene_rdl2/common/platform/Platform.h>

namespace moonray {
namespace pbr {

enum class SequenceType : utype
{
    Bsdf,
    Light,
    RussianRouletteBsdf,
    RussianRouletteLight,
    BssrdfLocal,
    BssrdfLocalChannel,
    BssrdfLocalBsdf,
    BssrdfLocalLight,
    BssrdfGlobal,
    BssrdfGlobalBsdf,
    BssrdfGlobalLight,
    VolumeAttenuation,
    VolumeScattering,
    VolumeEmission,
    VolumeDistance,
    VolumeEquiAngular,
    VolumePhase,
    IndexSelection,
    NextEventEstimation,
    IndirectLighting,
    LightFilter,
    LightFilter3D
};

// Normally, I would be all about type-safety. However, the only purpose for this class' existence is to supply
// consistent, but random, hash values based on certain state. I'm not too concerned with the types used to specify that
// state.
class SequenceID
{
public:
    template <typename... T>
    explicit SequenceID(T... t) noexcept
    : mHash(computeHash(t...))
    {
    }

    utype operator()(utype seed) const noexcept
    {
        return getHash(seed);
    }

    utype getHash(utype seed) const noexcept
    {
        return gphash(mHash, seed);
    }

    utype private_test() const noexcept
    {
        return mHash;
    }

private:
    static constexpr utype computeHashImpl() noexcept
    {
        // A Mersenne prime. We don't really expect a lot of calls to this level, but a Mersenne prime is probably as
        // good a choice as any: it's prime, so it will give fairly good strides if used in that fashion, but it's
        // Mersenne: A smart compiler can optimize any multiplications.
        return 2147483647u;
    }

    template <typename T>
    static constexpr utype computeHashImpl(T t) noexcept
    {
        // We could do something clever here, like break 64-bit types into two 32-bit types, but we want this to be
        // the same in ISPC, where we only use 32-bit types. So we static_cast.
        const utype mersennePrime7Exp = 19u;
        const utype mersennePrime8Exp = 31u;
        const auto h = hashCombine(0x31e149a2u, static_cast<utype>(t));
        return mersenneMultiply(mersennePrime8Exp, mersenneMultiply(mersennePrime7Exp, h));
    }

    // TODO: C++17: re-write with fold expressions
    template <typename T, typename... Rest>
    static constexpr utype computeHashImpl(T t, Rest... rest) noexcept
    {
        // We sandwich the args with the first element so that we get good distributions whether we are increments on
        // the first or the last elements. E.g.,
        // SequenceID(2, 1, 3);
        // SequenceID(2, 1, 4);
        // SequenceID(2, 1, 5);
        // vs
        // SequenceID(2, 1, 3);
        // SequenceID(3, 1, 3);
        // SequenceID(4, 1, 3);
        return hashCombine(hashCombine(computeHashImpl(t), computeHashImpl(rest...)), computeHashImpl(t));
    }

    template <typename T>
    static constexpr utype computeHash(T t) noexcept
    {
        // We overload the computeHash function for one argument instead of
        // using the variadic template version. If given one argument, the
        // variadic template concatenates the value with itself, which gives a
        // rather poor distribution.
        const utype mersennePrime7Exp = 19u;
        const utype mersennePrime8Exp = 31u;
        const auto h = hashCombine(static_cast<utype>(t), 0x72407263u);
        return mersenneMultiply(mersennePrime8Exp, mersenneMultiply(mersennePrime7Exp, h));
    }

    template <typename... T>
    static constexpr utype computeHash(T... t) noexcept
    {
        return computeHashImpl(t...);
    }

    utype mHash;
};

typedef SequenceID SequenceIDRR;
typedef SequenceID SequenceIDBssrdf;
typedef SequenceID SequenceIDIntegrator;

} // namespace pbr
} // namespace moonray

