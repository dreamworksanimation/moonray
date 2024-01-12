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

template <typename T, std::size_t D>
class PerfectPowerArray;

namespace ppadetail {

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

template <typename T, std::size_t D>
struct ContainedType
{
    typedef PerfectPowerArray<T, D - 1u> Type;
};

template <typename T>
struct ContainedType<T, 1u>
{
    typedef T Type;
};

template <std::size_t a, std::size_t b>
struct AreEqual
{
    static constexpr bool value = (a == b);
};

} // namespace ppadetail

template <typename T, std::size_t D>
class PerfectPowerArray
{
    typedef typename ppadetail::ContainedType<T, D>::Type Type;
    Type* mData;
    std::size_t mSize;

public:
    explicit PerfectPowerArray(std::size_t size) :
        mData(reinterpret_cast<Type*>(::operator new(sizeof(Type) * size))),
        mSize(size)
    {
        //std::cout << __PRETTY_FUNCTION__ << '\n';
#ifndef NDEBUG
        const Type* const minAddress = mData;
        const Type* const maxAddress = mData + size;
#endif
        for (std::size_t i = 0; i < size; ++i) {
            void* addr = reinterpret_cast<void*>(&mData[i]);
            assert(addr >= minAddress);
            assert(addr <=  maxAddress);
            new (addr) Type(size);
        }
        //std::cout << "Allocating " << typeid(Type).name() << '\n';
    }

    explicit PerfectPowerArray(std::size_t size, const T& t) : PerfectPowerArray(size)
    {
        auto f = [t](T& in) { in = t; };
        visitEach(f);
    }

    ~PerfectPowerArray()
    {
        for (std::size_t i = 0; i < mSize; ++i) {
            mData[i].~Type();
        }
        ::operator delete(mData);
    }

    static constexpr std::size_t sDims = D;

    std::size_t size() const
    {
        return mSize;
    }

    Type& operator[](std::size_t i)
    {
        assert(i < size());
        return mData[i];
    }

    const Type& operator[](std::size_t i) const
    {
        assert(i < size());
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
        return ppadetail::SameAsConstVersion(this, &PerfectPowerArray<T, D>::visitNeighborsImpl<F, sDims>, f, indices);
    }

    template <typename F>
    void visitEach(F f)
    {
        return ppadetail::SameAsConstVersion(this, &PerfectPowerArray<T, D>::visitEachImpl<F, sDims>, f);
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
        return (i == 0) ? size() - 1u : i - 1u;
    }

    std::size_t rght(std::size_t i) const
    {
        return (i == size() - 1u) ? 0 : i + 1u;
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
    typename std::enable_if<ppadetail::AreEqual<dim, 1>::value>::type visitNeighborsImpl(F f, const std::size_t* indices) const
    {
        const std::size_t idx = *indices;
        const std::size_t l = left(idx);
        const std::size_t c = cntr(idx);
        const std::size_t r = rght(idx);

        assert(l < size());
        assert(c < size());
        assert(r < size());

        //prints(idx);
        f(mData[l]);
        f(mData[c]);
        f(mData[r]);
    }

    template <typename F, std::size_t dim>
    typename std::enable_if<!ppadetail::AreEqual<dim, 1>::value>::type visitNeighborsImpl(F f, const std::size_t* indices) const
    {
        const std::size_t idx = *indices;
        const std::size_t l = left(idx);
        const std::size_t c = cntr(idx);
        const std::size_t r = rght(idx);

        assert(l < size());
        assert(c < size());
        assert(r < size());

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
    typename std::enable_if<ppadetail::AreEqual<dim, 1>::value>::type visitEachImpl(F f) const
    {
        for (std::size_t i = 0; i < size(); ++i) {
            f(mData[i]);
        }
    }

    template <typename F, std::size_t dim>
    typename std::enable_if<!ppadetail::AreEqual<dim, 1>::value>::type visitEachImpl(F f) const
    {
        for (std::size_t i = 0; i < size(); ++i) {
            mData[i].template visitEach<F>(f);
        }
    }
};
