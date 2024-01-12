// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveAttribute.h
/// $Id$
///

#pragma once

#include "AttributeKey.h"

#include <moonray/common/mcrt_util/StringPool.h>
#include <type_traits>
#include <unordered_map>

namespace moonray {
namespace shading {

enum AttributeRate
{
    RATE_UNKNOWN,
    RATE_CONSTANT,
    RATE_UNIFORM,
    RATE_VARYING,
    RATE_FACE_VARYING,
    RATE_VERTEX,
    RATE_PART,
    RATE_LAST
};

template <typename T>
class PrimitiveAttribute;

///
/// @class PrimitiveAttributeBase
/// @brief base class for all primitive attribute types
///
class PrimitiveAttributeBase
{
public:
    PrimitiveAttributeBase(AttributeRate rate) : mRate(rate) {}

    virtual ~PrimitiveAttributeBase() = 0;

    /// get the AttributeRate for this primitive attribute
    AttributeRate getRate() const
    {
        return mRate;
    }

    /// set the AttributeRate for this primitive attribute
    void setRate(AttributeRate rate)
    {
        mRate = rate;
    }

    /// down cast to typed PrimitiveAttribute
    template <typename T>
    PrimitiveAttribute<T>& as()
    {
        return dynamic_cast<PrimitiveAttribute<T>&>(*this);
    }

    /// down cast to typed PrimitiveAttribute
    template <typename T>
    const PrimitiveAttribute<T>& as() const
    {
        return dynamic_cast<const PrimitiveAttribute<T>&>(*this);
    }

    /// Return size
    virtual size_t size() const = 0;

    /// Change size
    virtual void resize(size_t n) = 0;

    virtual void fetchData(size_t offset, char* data) const = 0;

    virtual void copyInPlace(size_t src, size_t dst) = 0;

protected:
    AttributeRate mRate;
};

inline PrimitiveAttributeBase::~PrimitiveAttributeBase()
{
}

template <typename T>
constexpr bool noexceptMemberSwap()
{
    return noexcept(std::declval<T&>().swap(std::declval<T&>()));
}

///
/// @class PrimitiveAttribute
/// @brief template vector that is used to store arbitrary primitive attribute
///
template <typename T>
class PrimitiveAttribute : public PrimitiveAttributeBase
{
public:
    typedef std::vector<T>                             list_type;
    typedef typename list_type::value_type             value_type;
    typedef typename list_type::size_type              size_type;
    typedef typename list_type::reference              reference;
    typedef typename list_type::const_reference        const_reference;
    typedef typename list_type::pointer                pointer;
    typedef typename list_type::const_pointer          const_pointer;
    typedef typename list_type::iterator               iterator;
    typedef typename list_type::const_iterator         const_iterator;
    typedef typename list_type::reverse_iterator       reverse_iterator;
    typedef typename list_type::const_reverse_iterator const_reverse_iterator;

    explicit PrimitiveAttribute(AttributeRate rate, list_type&& list = list_type{}):
            PrimitiveAttributeBase(rate), mList(list)
    {
    }

    /// Swap content
    void swap(PrimitiveAttribute& other) noexcept(noexceptMemberSwap<list_type>())
    {
        using std::swap; // Allow ADL
        swap(mRate, other.mRate);
        mList.swap(other.mList);
    }

    /// Add element at the end
    void push_back(const T& value)
    {
        mList.push_back(value);
    }

    /// Add element at the end
    void push_back(T&& value)
    {
        mList.push_back(std::forward<T>(value));
    }

    /// Construct and insert element at the end
    template<class... Args>
    void emplace_back(Args&&... args)
    {
        mList.emplace_back(std::forward<Args>(args)...);
    }

    iterator insert(const_iterator pos, const T& value)
    {
        return mList.insert(toNonConst(mList, pos), value);
    }

    iterator insert(const_iterator pos, T&& value)
    {
        return mList.insert(toNonConst(mList, pos), std::forward<T>(value));
    }

    // TODO: C++11 feature-complete library: this should return iterator.
    void insert(const_iterator pos, size_type count, const T& value)
    {
        return mList(toNonConst(mList, pos), count, value);
    }

    // TODO: C++11 feature-complete library: this should return iterator.
    template <typename InputIt>
#if __cplusplus > 201103L
    iterator
#else
    void
#endif
    insert(const_iterator pos, InputIt first, InputIt last)
    {
        return mList.insert(toNonConst(mList, pos), first, last);
    }

    iterator insert(const_iterator pos, std::initializer_list<T> ilist)
    {
        return mList.insert(toNonConst(mList, pos), ilist);
    }

    iterator erase(const_iterator pos)
    {
        return mList.erase(toNonConst(mList, pos));
    }

    iterator erase(const_iterator first, const_iterator last)
    {
        return mList.erase(toNonConst(mList, first), toNonConst(mList, last));
    }

    void reserve(size_type size)
    {
        mList.reserve(size);
    }

    void clear() noexcept
    {
        mList.clear();
    }

    /// Return iterator to beginning
    iterator begin()
    {
        return mList.begin();
    }

    /// Return const_iterator to beginning
    const_iterator begin() const
    {
        return mList.begin();
    }

    /// Return const_iterator to beginning
    const_iterator cbegin() const
    {
        return mList.cbegin();
    }

    /// Return iterator to end
    iterator end()
    {
        return mList.end();
    }

    /// Return const_iterator to end
    const_iterator end() const
    {
        return mList.end();
    }

    /// Return const_iterator to end
    const_iterator cend() const
    {
        return mList.cend();
    }

    /// Return reverse_iterator to reverse beginning
    reverse_iterator rbegin()
    {
        return mList.rbegin();
    }

    /// Return const_reverse_iterator to reverse beginning
    const_reverse_iterator rbegin() const
    {
        return mList.rbegin();
    }

    /// Return const_reverse_iterator to reverse beginning
    const_reverse_iterator crbegin() const
    {
        return mList.crbegin();
    }

    /// Return reverse_iterator to reverse end
    reverse_iterator rend()
    {
        return mList.rend();
    }

    /// Return const_reverse_iterator to reverse end
    const_reverse_iterator rend() const
    {
        return mList.rend();
    }

    /// Return const_reverse_iterator to reverse end
    const_reverse_iterator crend() const
    {
        return mList.crend();
    }

    /// Return size
    virtual size_t size() const noexcept override
    {
        return mList.size();
    }

    /// Change size
    virtual void resize(size_t n) override
    {
        mList.resize(n);
    }

    /// Access element
    reference operator[] (size_type n)
    {
        return mList[n];
    }

    /// Access element
    const_reference operator[] (size_type n) const
    {
        return mList[n];
    }

    /// @brief fetch data at offset location to raw memory chunk
    ///     this is used internally to build interleaved Attributes
    virtual void fetchData(size_t offset, char* data) const override;

    virtual void copyInPlace(size_t src, size_t dst) override
    {
        mList[dst] = mList[src];
    }

private:
    static iterator toNonConst(const list_type& list, const_iterator citer)
    {
        // We have to jump through hoops because C++11 allows its operations to
        // work on const_iterators, but our standard library doesn't support
        // the interface yet.
        const auto diff = citer - list.cbegin();
        return const_cast<list_type&>(list).begin() + diff;
    }

    list_type mList;
};

template <typename T>
inline void swap(PrimitiveAttribute<T>& a, PrimitiveAttribute<T>& b) noexcept(noexceptMemberSwap<PrimitiveAttribute<T>>())
{
    a.swap(b);
}

template <typename T>
void insert(PrimitiveAttribute<T>& pa, typename PrimitiveAttribute<T>::size_type pos, const T& value)
{
    pa.insert(pa.cbegin() + pos, value);
}

template <typename T>
void insert(PrimitiveAttribute<T>& pa, typename PrimitiveAttribute<T>::size_type pos, T&& value)
{
    pa.insert(pa.cbegin() + pos, std::forward<T>(value));
}

template <typename T>
void insert(PrimitiveAttribute<T>& pa, typename PrimitiveAttribute<T>::size_type pos, typename PrimitiveAttribute<T>::size_type count, const T& value)
{
    pa(pa.cbegin() + pos, count, value);
}

template <typename T, typename InputIt>
void insert(PrimitiveAttribute<T>& pa, typename PrimitiveAttribute<T>::size_type pos, InputIt first, InputIt last)
{
    pa.insert(pa.cbegin() + pos, first, last);
}

template <typename T>
void insert(PrimitiveAttribute<T>& pa, typename PrimitiveAttribute<T>::size_type pos, std::initializer_list<T> ilist)
{
    pa.insert(pa.cbegin() + pos, ilist);
}

template <typename T>
void erase(PrimitiveAttribute<T>& pa, typename PrimitiveAttribute<T>::size_type pos)
{
    pa.erase(pa.cbegin() + pos);
}

template <typename T>
void
PrimitiveAttribute<T>::fetchData(size_t offset, char* data) const
{
    *(reinterpret_cast<T *>(data)) = mList[offset];
}

template <>
inline void
PrimitiveAttribute<std::string>::fetchData(size_t offset, char* data) const
{
    const std::string* strPtr = util::getStringPool().get(mList[offset]);
    *(reinterpret_cast<const std::string **>(data)) = strPtr;
}

template <typename T>
struct is_string : public std::integral_constant<bool, false>
{
};

template <typename CharT, typename Traits, typename Allocator>
struct is_string<std::basic_string<CharT, Traits, Allocator>> : public std::integral_constant<bool, true>
{
};

///
/// @class PrimitiveAttributeTable
/// @brief a PrimitiveAttribute hashtable using AttributeKey as lookup index
///
class PrimitiveAttributeTable
{
public:
    typedef std::unordered_map<AttributeKey,
            std::vector<std::unique_ptr<PrimitiveAttributeBase>>,
            AttributeKeyHash> map_type;
    typedef typename map_type::key_type key_type;
    typedef typename map_type::mapped_type mapped_type;
    typedef typename map_type::iterator iterator;
    typedef typename map_type::const_iterator const_iterator;

    PrimitiveAttributeTable() = default;
    ~PrimitiveAttributeTable() = default;
    PrimitiveAttributeTable(PrimitiveAttributeTable&&) = default;
    PrimitiveAttributeTable(const PrimitiveAttributeTable&) = delete;
    PrimitiveAttributeTable& operator=(const PrimitiveAttributeTable&) = delete;

    /// @brief submit a vector data to build a PrimitiveAttribute and stored in
    ///     PrimitiveAttributeTable
    /// @param key TypedAttributeKey for indexing submitted primitive attribute
    /// @param rate specify the frequency of submitted primitive attribute
    /// @param data the content of submitted primitive attribute
    /// @return whether the attribute is correctly submitted
    ///     into PrimitiveAttributeTable
    template <typename T>
    bool addAttribute(TypedAttributeKey<T> key, AttributeRate rate,
            std::vector<T>&& data)
    {
        std::vector<std::unique_ptr<PrimitiveAttributeBase>> dataList;
        dataList.emplace_back(new PrimitiveAttribute<T>(rate, std::move(data)));
        if (mMap.find(key) == mMap.end()) {
            return mMap.emplace(key, std::move(dataList)).second;
        } else {
            return false;
        }
    }

    /// @brief submit a set of vector data to build a PrimitiveAttribute
    ///     and stored in PrimitiveAttributeTable. Each vector data in the set
    ///     stands for a time sample of this PrimitiveAttribute
    /// @param key TypedAttributeKey for indexing submitted primitive attribute
    /// @param rate specify the frequency of submitted primitive attribute
    /// @param data the content of submitted primitive attribute
    /// @return whether the attribute is correctly submitted
    ///     into PrimitiveAttributeTable
    template <typename T>
    bool addAttribute(TypedAttributeKey<T> key, AttributeRate rate,
            std::vector<std::vector<T>>&& data)
    {
        static_assert(!std::is_integral<T>::value,
            "integral type is not motionblur interpolatable");
        static_assert(!is_string<T>::value,
            "string type is not motionblur interpolatable");

        MNRY_ASSERT_REQUIRE(data.size() <= 2,
            "only support two time samples for motionblur at this moment");

        std::vector<std::unique_ptr<PrimitiveAttributeBase>> dataList;
        dataList.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            dataList.emplace_back(
                new PrimitiveAttribute<T>(rate, std::move(data[i])));
        }
        if (mMap.find(key) == mMap.end()) {
            return mMap.emplace(key, std::move(dataList)).second;
        } else {
            return false;
        }
    }

    /// return the typed PrimitiveAttribute reference with specified key
    template <typename T>
    PrimitiveAttribute<T>& getAttribute(TypedAttributeKey<T> key,
            size_t timeIndex = 0)
    {
        MNRY_ASSERT(timeIndex < mMap.find(key)->second.size());
        return mMap[key][timeIndex]->template as<T>();
    }

    template <typename T>
    const PrimitiveAttribute<T>& getAttribute(TypedAttributeKey<T> key,
            size_t timeIndex = 0) const
    {
        MNRY_ASSERT(hasAttribute(key));
        MNRY_ASSERT(timeIndex < mMap.find(key)->second.size());
        return (mMap.find(key)->second[timeIndex])->template as<T>();
    }

    /// return whether the PrimitiveAttribute with specified key exists
    bool hasAttribute(AttributeKey key) const;

    /// return the AttributeRate of PrimitiveAttribute with apecified key
    AttributeRate getRate(AttributeKey key) const;

    /// return the count of time sample for PrimitiveAttribute
    /// with specified key
    size_t getTimeSampleCount(AttributeKey key) const;

    /// return iterator to beginning
    iterator begin();

    /// return const_iterator to beginning
    const_iterator begin() const;

    /// return iterator to end
    iterator end();

    /// return const_iterator to end
    const_iterator end() const;

    /// Test whether container is empty
    bool empty() const;

    /// Get iterator to element
    iterator find(const key_type& k);

    /// Get const_iterator to element
    const_iterator find(const key_type& k) const;

    /// Erase element indexed by key
    size_t erase(const key_type& key);

    /// Copy the contents to the  passed in PrimitiveAttributeTable
    void copy(PrimitiveAttributeTable& result) const;

    /// Erase all elements
    void clear()
    {
        mMap.clear();
    }

private:

    template <typename T>
    mapped_type copyAttribute(TypedAttributeKey<T> key) const
    {
        mapped_type attr;
        size_t timeSampleCount = getTimeSampleCount(key);
        attr.reserve(timeSampleCount);
        for (size_t t = 0; t < timeSampleCount; ++t) {
            attr.emplace_back(new PrimitiveAttribute<T>(getAttribute(key, t)));
        }
        return attr;
    }

private:
    map_type mMap;
};

} // namespace shading
} // namespace moonray


