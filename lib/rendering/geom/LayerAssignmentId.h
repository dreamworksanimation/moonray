// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file LayerAssignmentId.h
/// $Id$
///

#pragma once
#include <scene_rdl2/render/util/Memory.h>

#include <initializer_list>
#include <vector>

namespace moonray {
namespace geom {

namespace detail {
template<typename T>
void destroy(T& t)
{
    t.~T();
}
} // namespace detail

///
/// @class LayerAssignmentId
/// @brief This class represents the mapping from each sub-primitive
///     (face/curve) to a corresponding layer assignment id.
/// Each primitive is constructed with one LayerAssignmentId so that it can
/// locate the material shader assignment for each sub-primitive. If the whole
/// primitive shares one material assignment, one ValueType is used
/// to represent the id (CONSTANT). If the primitive is composed of several
/// parts, a std::vector<ValueType> with size of sub-primitive count
/// (for example, curves count in Curves primitive) is used to represent
/// the id (VARYING)
class LayerAssignmentId
{
public:
    typedef int ValueType;
    typedef std::vector<ValueType> ContainerType;

    enum class Type
    {
        CONSTANT, ///<all sub-primitives share one assignment id
        VARYING   ///<each sub-primitive maps to its corresponding assignment id
    };

    /// construct CONSTANT rate LayerAssignmentId
    explicit LayerAssignmentId(ValueType id) noexcept :
        mType(Type::CONSTANT),
        mConstId(id)
    {
    }

    /// construct VARYING rate LayerAssignmentId
    explicit LayerAssignmentId(const ContainerType& id) :
        mType(Type::VARYING),
        mVaryingId(id)
    {
    }

    /// construct VARYING rate LayerAssignmentId
    explicit LayerAssignmentId(ContainerType&& id) :
        mType(Type::VARYING),
        mVaryingId(std::move(id))
    {
    }

    /// construct VARYING rate LayerAssignmentId
    LayerAssignmentId(std::initializer_list<ValueType> list) :
        mType(Type::VARYING),
        mVaryingId(list.begin(), list.end())
    {
    }

    /// copy constructor
    LayerAssignmentId(const LayerAssignmentId& other) :
        mType(other.mType)
    {
        switch (mType) {
            case Type::CONSTANT:
                new (&mConstId) ValueType(other.mConstId);
                break;
            case Type::VARYING:
                new (&mVaryingId) ContainerType(other.mVaryingId);
                break;
        }
    }

    /// move constructor
    LayerAssignmentId(LayerAssignmentId&& other) noexcept :
        mType(other.mType)
    {
        switch (mType) {
            case Type::CONSTANT:
                new (&mConstId) ValueType(std::move(other.mConstId));
                break;
            case Type::VARYING:
                new (&mVaryingId) ContainerType(std::move(other.mVaryingId));
                break;
        }
    }

    // Meets strong exception guarantee. Wheeee!
    LayerAssignmentId& operator=(const LayerAssignmentId& other)
    {
        LayerAssignmentId lid(other);
        swap(lid);
        return *this;
    }

    LayerAssignmentId& operator=(LayerAssignmentId&& other) noexcept
    {
        LayerAssignmentId lid(std::move(other));
        swap(lid);
        return *this;
    }

    ~LayerAssignmentId()
    {
        switch (mType) {
            case Type::CONSTANT:
                detail::destroy(mConstId);
                break;
            case Type::VARYING:
                detail::destroy(mVaryingId);
                break;
        }
    }

    void swap(LayerAssignmentId& other) noexcept
    {
        using std::swap; // Allow ADL

        const Type thisType = this->getType();
        const Type otherType = other.getType();

        if (thisType == Type::CONSTANT && otherType == Type::CONSTANT) {
            swap(mConstId, other.mConstId);
        } else if (thisType == Type::VARYING && otherType == Type::VARYING) {
            swap(mVaryingId, other.mVaryingId);
        } else if (thisType == Type::CONSTANT && otherType == Type::VARYING) {
            swapConstVarying(*this, other);
        } else if (thisType == Type::VARYING && otherType == Type::CONSTANT) {
            swapConstVarying(other, *this);
        }

        swap(mType, other.mType);
    }

    Type getType() const noexcept
    {
        return mType;
    }

    ValueType getConstId() const noexcept
    {
        MNRY_ASSERT(mType == Type::CONSTANT);
        return mConstId;
    }

    const ContainerType& getVaryingId() const noexcept
    {
        MNRY_ASSERT(mType == Type::VARYING);
        return mVaryingId;
    }

    ContainerType& getVaryingId() noexcept
    {
        MNRY_ASSERT(mType == Type::VARYING);
        return mVaryingId;
    }

    size_t getMemory() const {
        return getType() == Type::CONSTANT ?
            sizeof(LayerAssignmentId) :
            scene_rdl2::util::getVectorMemory(getVaryingId());
    }

    bool hasValidAssignment() const {
        if (mType == Type::CONSTANT) {
            return mConstId >= 0;
        } else {
            return std::any_of(mVaryingId.begin(), mVaryingId.end(),
                [](ValueType id) { return id >= 0; });
        }
    }

private:
    // Initially, "a" is const, and "b" is varying.
    static void swapConstVarying(
            LayerAssignmentId& a, LayerAssignmentId& b) noexcept
    {
        // Technically, we don't have to destroy an int, but it doesn't hurt.
        // Let's be consistent.
        auto id = std::move(a.mConstId);
        detail::destroy(a.mConstId);

        // Move the vector from b to a, making sure to call the constructor of
        // the new vector.
        new (&a.mVaryingId) ContainerType(std::move(b.mVaryingId));
        detail::destroy(b.mVaryingId);

        // Move the saved int to b.
        new (&b.mConstId) ValueType(std::move(id));
    }

    Type mType;
    union
    {
        ValueType mConstId;
        ContainerType mVaryingId;
    };
};

} // namespace geom
} // namespace moonray

namespace std
{
    inline void swap(moonray::geom::LayerAssignmentId& a,
            moonray::geom::LayerAssignmentId& b) noexcept
    {
        a.swap(b);
    }
}

