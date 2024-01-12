// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <iostream>
#include <typeinfo>
#define prints(v) std::cout << #v << ": " << (v) << '\n'

template <typename TConstReturn, class TObj, typename... TArgs>
typename std::remove_const<TConstReturn>::type SameAsConstVersion(
        TObj const* obj,
        TConstReturn (TObj::* memFun)(TArgs...) const,
        TArgs... args)
{
    //typedef typename std::remove_const<TConstReturn>::type NonConstReturnType;
    //return const_cast<NonConstReturnType>((obj->*memFun)(std::forward<TArgs>(args)...));
    return (obj->*memFun)(std::forward<TArgs>(args)...);
}

template <typename T, std::size_t S, std::size_t... OtherS>
class Array;

namespace detail {
template <typename T, std::size_t... OtherS>
struct ContainedType
{
    typedef Array<T, OtherS...> Type;
};

template <typename T>
struct ContainedType<T>
{
    typedef T Type;
};

template <std::size_t a, std::size_t b>
struct AreEqual
{
    static constexpr bool value = (a == b);
};
} // namespace detail

/*
template <typename T, std::size_t dims>
struct Point
{
    T[dims] mData;
};
*/

template <typename T, std::size_t S, std::size_t... OtherS>
class Array
{
    typedef typename detail::ContainedType<T, OtherS...>::Type Type;
    //Type mData[S];
    Type* mData;

public:
    /*
    Array() = default;

    explicit Array(const T& t)
    {
        auto f = [t](T& in) { in = t; };
        visitEach(f);
    }
    */
    Array() : mData(new Type[S])
    {
        //std::cout << "Allocating " << typeid(Type).name() << '\n';
    }

    explicit Array(const T& t) : Array()
    {
        auto f = [t](T& in) { in = t; };
        visitEach(f);
    }

    ~Array()
    {
        delete[] mData;
    }

    static constexpr std::size_t sDims = 1u + sizeof...(OtherS);

    Type& operator[](std::size_t i)
    {
        assert(i < S);
        return mData[i];
    }

    const Type& operator[](std::size_t i) const
    {
        assert(i < S);
        return mData[i];
    }

    template <typename F>
    void visitNeighbors(F f, const std::size_t* indices) const
    {
        visitNeighborsImpl<F, sDims>(f, indices);
    }

    template <typename F>
    void visitEach(F f) const
    {
        visitEachImpl<F, sDims>(f);
    }

    template <typename F>
    void visitNeighbors(F f, const std::size_t* indices)
    {
        return SameAsConstVersion(this, &Array<T, S, OtherS...>::visitNeighborsImpl<F, sDims>, f, indices);
    }

    template <typename F>
    void visitEach(F f)
    {
        return SameAsConstVersion(this, &Array<T, S, OtherS...>::visitEachImpl<F, sDims>, f);
    }

    /*
    template <typename F, std::size_t car>
    void visitNeighbors(F f)
    {
        f(mData[car-1]);
        f(mData[car+0]);
        f(mData[car+1]);
    }

    template <typename F, std::size_t car, std::size_t carcdr, std::size_t... cdr>
    void visitNeighbors(F f)
    {
        mData[car-1].visitNeighbors<F, carcdr, cdr...>(f);
        mData[car+0].visitNeighbors<F, carcdr, cdr...>(f);
        mData[car+1].visitNeighbors<F, carcdr, cdr...>(f);
    }
    */

    /*
    template <typename F>
    void visitNeighbors(F f, const Point<unsigned, sDims>& p)
    {
        const unsigned mid  = p.mData[0];
        const unsigned lft = (mid == 0)      ? S - 1 : mid - 1;
        const unsigned rht = (mid) == S - 1) ? 0     : mid + 1;

        if (sDims == 1) {
            f(mData[left]);
            f(mData[mid]);
            f(mData[right]);
        } else {
            visitNeighbors(f, cdr(p));
        }
    }
    */

private:
    std::size_t left(std::size_t i) const
    {
        return (i == 0) ? S - 1u : i - 1u;
    }

    std::size_t rght(std::size_t i) const
    {
        return (i == S - 1u) ? 0 : i + 1u;
    }

    std::size_t cntr(std::size_t i) const
    {
        return i;
    }

    // We have to make this decision at compile-time, otherwise the compiler
    // complains that we try to call "visitNeighbors" on a type that's not an
    // array. We can't use specialization, since that would require partial
    // function specialization. Yay SFINAE!
    template <typename F, std::size_t dim>
    typename std::enable_if<detail::AreEqual<dim, 1>::value>::type visitNeighborsImpl(F f, const std::size_t* indices) const
    {
        const std::size_t idx = *indices;
        const std::size_t l = left(idx);
        const std::size_t c = cntr(idx);
        const std::size_t r = rght(idx);

        assert(l < S);
        assert(c < S);
        assert(r < S);

        //prints(idx);
        f(mData[l]);
        f(mData[c]);
        f(mData[r]);
    }

    template <typename F, std::size_t dim>
    typename std::enable_if<!detail::AreEqual<dim, 1>::value>::type visitNeighborsImpl(F f, const std::size_t* indices) const
    {
        const std::size_t idx = *indices;
        const std::size_t l = left(idx);
        const std::size_t c = cntr(idx);
        const std::size_t r = rght(idx);

        assert(l < S);
        assert(c < S);
        assert(r < S);

        //prints(idx);
        mData[l].template visitNeighbors<F>(f, indices+1);
        mData[c].template visitNeighbors<F>(f, indices+1);
        mData[r].template visitNeighbors<F>(f, indices+1);
    }

    // We have to make this decision at compile-time, otherwise the compiler
    // complains that we try to call "visitNeighbors" on a type that's not an
    // array. We can't use specialization, since that would require partial
    // function specialization. Yay SFINAE!
    template <typename F, std::size_t dim>
    typename std::enable_if<detail::AreEqual<dim, 1>::value>::type visitEachImpl(F f) const
    {
        for (std::size_t i = 0; i < S; ++i) {
            f(mData[i]);
        }
    }

    template <typename F, std::size_t dim>
    typename std::enable_if<!detail::AreEqual<dim, 1>::value>::type visitEachImpl(F f) const
    {
        for (std::size_t i = 0; i < S; ++i) {
            mData[i].template visitEach<F>(f);
        }
    }
};
