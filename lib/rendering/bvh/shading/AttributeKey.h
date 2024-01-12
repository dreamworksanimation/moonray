// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file AttributeKey.h
/// $Id$
///

#pragma once

#include <scene_rdl2/scene/rdl2/Types.h>
#include <tbb/mutex.h>
#include <unordered_set>
#include <map>

using scene_rdl2::rdl2::AttributeType;
using scene_rdl2::rdl2::attributeType;

namespace moonray {
namespace shading {

enum class Vec3Type
{
    POINT,
    VECTOR,
    NORMAL
};

/// @class AttributeKey
/// @brief a light weight index with type/name/size info for fast lookup
///     primitive attributes
///
/// An AttributeKey is an index into a table of primitive attributes.
/// It is very lightweight (just an int) and can be cheaply copied.
/// The index points into a table of names, data types and data sizes for each key.
/// The table is a set of static member variables which should only be
/// modified by StandardAttributes::init() sometime at startup.
/// It would be desirable to merge this class with scene_rdl2::rdl2::AttributeKey, as they
/// serve very similar purposes.
class AttributeKey
{
 public:
    /// Creates an invalid AttributeKey. In general, this should
    /// only be used when the key will be initialized at a later point.
    finline AttributeKey();
    finline AttributeKey(int idx);
    finline AttributeKey(const AttributeKey &other);
    finline bool operator==(const AttributeKey &other) const;

    finline int getIndex() const;

    // The following methods look up into the static table via the index
    // stored in this AttributeKey
    finline size_t getSize() const;
    finline const char *getName() const;
    finline AttributeType getType() const;
    finline bool hasDerivatives() const;

    finline operator int() const;

    // Invalid AttributeKeys are represented by a -1 index.
    finline bool isValid() const;

    // Involves locking! only call this on render prepare stage
    finline bool requestDerivatives() const;

protected:
    int mIndex;

    // Inserts a key into the table with the given name, type
    // and size (derived from the type).
    // if the key already exists, smiply returns index of existing key.
    template <typename T>
    static finline int insertKey(const std::string &name, bool requestDerivatives);

    // Looks up key based on name and type and returns index for
    // that key.
    // If the key is not found -1 is returned (and asserts in debug mode)
    template <typename T>
    static finline int lookup(const std::string &name);

    // Static versions of previous functions
    static finline bool isValid(int index);
    static finline const char *getName(AttributeKey key);
    static finline AttributeType getType(AttributeKey key);
    static finline size_t getSize(AttributeKey key);
    static finline bool hasDerivatives(AttributeKey key);

private:
    static tbb::mutex sRegisterMutex;
    static std::vector<std::string> sKeyNames;
    static std::vector<AttributeType> sKeyTypes;
    static std::vector<size_t> sKeySizes;
    static std::vector<int8_t> sHasDerivatives;
    static std::map<std::pair<std::string, AttributeType>, int> sTable;
    static int sNumKeys;
};

/// @class TypedAttributeKey
/// @brief A templated version of AttributeKey for type safety when possible
template <typename T>
class TypedAttributeKey : public AttributeKey
{
public:
    finline TypedAttributeKey();
    finline TypedAttributeKey(AttributeKey k);
    // This constructor involves lock operation so it's better to
    // construct it once and reused it later instead of construct a new
    // one everytime with attribute name string
    finline explicit TypedAttributeKey(const std::string &name,
        bool requestDerivatives = false);
};

typedef std::pair<TypedAttributeKey<scene_rdl2::math::Vec3f>, Vec3Type> Vec3KeyType;

/// @class StandardAttributes
/// @brief an enumeration of standard primitive attributes that can
///     be used directly without further declaration
///
/// This is an enumeration of all the standard primitive attributes
/// that can be used. Additional keys should be added as needed
/// (with a corresponding change to init()).
/// This list of keys is initialized in init() and init() should be
/// called before using any of these keys.
class StandardAttributes
{
public:
    // Initializes the standard attributes. This function should
    // be called soon after program start. The standard attributes
    // are not valid until init() is called.
    static finline void init();

    // Maximum number of polyvertices in an intersection
    static const size_t MAX_NUM_POLYVERTICES = 16;
    // Polyvertices can define different types of primitives
    enum PolyVertexType {
        POLYVERTEX_TYPE_UNKNOWN = 0,  // likely a program error
        POLYVERTEX_TYPE_POLYGON,      // polyvertices define a triangle, quad, etc...
        POLYVERTEX_TYPE_CUBIC_SPLINE, // polyvertices define a cubic spline span
        POLYVERTEX_TYPE_LINE          // polyvertices define a line segment
    };

    // Standard attributes
    static TypedAttributeKey<scene_rdl2::math::Vec2f> sUv;
    // hair coordinates for the point closest to it on the surface geo
    static TypedAttributeKey<scene_rdl2::math::Vec2f> sClosestSurfaceST;
    // hair coordinates for the root of the hair fiber on surface geo
    static TypedAttributeKey<scene_rdl2::math::Vec2f> sSurfaceST;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sNormal;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sRefP;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sRefN;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sdPds;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sRefdPds;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sdPdt;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sVelocity;      // primitive specified velocity attribute
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sP0;            // P at first time step
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sAcceleration;  // primitive specified acceleration attribute
    static TypedAttributeKey<scene_rdl2::math::Mat4f> sInstanceObjectTransform;
    static TypedAttributeKey<scene_rdl2::math::Mat4f> sInstanceTransformLevel0;
    static TypedAttributeKey<scene_rdl2::math::Mat4f> sInstanceTransformLevel1;
    static TypedAttributeKey<scene_rdl2::math::Mat4f> sInstanceTransformLevel2;
    static TypedAttributeKey<scene_rdl2::math::Mat4f> sInstanceTransformLevel3;
    static TypedAttributeKey<scene_rdl2::math::Mat4f> sInstanceTransformLevel4;
    static TypedAttributeKey<float> sScatterTag;
    static TypedAttributeKey<float> sShadowRayEpsilon;
    // sMotion is computed in postIntersect.  It contains
    // the instaneous velocity of the intersection point
    // in render space units per shutter interval. It accounts
    // for motion caused by geometry node transforms,
    // instancing, vertex slice information, and user supplied
    // velocity attributes.  It does not include relative camera
    // or projection motion.
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sMotion;
    static TypedAttributeKey<scene_rdl2::math::Vec3f> sPolyVertices[MAX_NUM_POLYVERTICES];
    static TypedAttributeKey<int> sNumPolyVertices;
    static TypedAttributeKey<int> sPolyVertexType; // PolyVertexType
    static TypedAttributeKey<int> sId;
    static TypedAttributeKey<bool> sReversedNormals;
    static TypedAttributeKey<bool> sExplicitShading;
};

/// @class AttributeKeyHash
/// @brief Hashing class for inserting AttributeKeys into std::unordered_sets
class AttributeKeyHash
{
 public:
    size_t operator()(AttributeKey k) const {
        return (size_t)k.getIndex();
    }
};

typedef std::unordered_set<AttributeKey, AttributeKeyHash> AttributeKeySet;
typedef std::unordered_map<const scene_rdl2::rdl2::Geometry *, AttributeKeySet> PerGeometryAttributeKeySet;

// Public
AttributeKey::AttributeKey()
{
    mIndex = -1;
}

AttributeKey::AttributeKey(int idx)
{
    MNRY_ASSERT(isValid(idx));
    mIndex = idx;
}

AttributeKey::AttributeKey(const AttributeKey &other)
{
    mIndex = other.mIndex;
}

bool
AttributeKey::operator==(const AttributeKey &other) const
{
    return mIndex == other.mIndex;
}

int
AttributeKey::getIndex() const
{
    return mIndex;
}

size_t
AttributeKey::getSize() const
{
    return getSize(*this);
}

const char *
AttributeKey::getName() const
{
    return getName(*this);
}

AttributeType
AttributeKey::getType() const
{
    return getType(*this);
}

bool
AttributeKey::hasDerivatives() const
{
    return hasDerivatives(*this);
}

AttributeKey::operator int() const
{
    return mIndex;
}

bool
AttributeKey::isValid() const
{
    return mIndex != -1;
}

bool
AttributeKey::requestDerivatives() const
{
    if (!isValid()) {
        return false;
    }
    // unable to compute differential for non float type attributes
    AttributeType type = getType();
    if (type == scene_rdl2::rdl2::TYPE_BOOL || type == scene_rdl2::rdl2::TYPE_INT ||
        type == scene_rdl2::rdl2::TYPE_LONG || type == scene_rdl2::rdl2::TYPE_STRING) {
        return false;
    }
    {
        tbb::mutex::scoped_lock lock(sRegisterMutex);
        sHasDerivatives[mIndex] = 1;
    }
    return true;
}

// Protected
template <typename T>
size_t
getAttributeTypeSize()
{
    AttributeType type = attributeType<T>();
    // string need special treatment since it's not a plain old data types.
    // manipulate it in raw memory chunk will cause unwanted destructor call
    // and seg fault. We use a util pool to manage string object and store the
    // string pointer instead of string in chunk to avoid this issue
    // (and save memory usage since we may have repetitive attribute values)
    if (type == scene_rdl2::rdl2::TYPE_STRING) {
        return sizeof(std::string*);
    } else {
        return sizeof(T);
    }
}

template <typename T>
int
AttributeKey::insertKey(const std::string &name, bool requestDerivatives)
{
    AttributeType type = attributeType<T>();
    size_t size = getAttributeTypeSize<T>();
    std::pair<std::string, AttributeType> lookup(name, type);
    int index = -1;
    {
        tbb::mutex::scoped_lock lock(sRegisterMutex);
        auto it = sTable.find(lookup);
        if (it == sTable.end()) {
            index = static_cast<int>(sKeyNames.size());
            sTable[lookup] = index;
            sKeyNames.push_back(name);
            sKeySizes.push_back(size);
            sKeyTypes.push_back(type);
            sHasDerivatives.push_back(0);
        } else {
            index = it->second;
        }
        if (requestDerivatives &&
            type != scene_rdl2::rdl2::TYPE_BOOL && type != scene_rdl2::rdl2::TYPE_INT &&
            type != scene_rdl2::rdl2::TYPE_LONG && type != scene_rdl2::rdl2::TYPE_STRING) {
            sHasDerivatives[index] = 1;
        }
    }
    return index;
}

template <typename T>
int
AttributeKey::lookup(const std::string &name)
{
    AttributeType type = attributeType<T>();
    std::pair<std::string, AttributeType> lu(name, type);

    auto it = sTable.find(lu);
    if (it == sTable.end()) {
        std::stringstream ss;
        ss << "Key not found:" << name << " Type:(" << type << "). Add key before using.";
        MNRY_ASSERT(0, ss.str().c_str());
        return -1;
    } else {
        return it->second;
    }
}

bool
AttributeKey::isValid(int index)
{
    return index >= 0 && index < (int)sKeyNames.size();
}


const char *
AttributeKey::getName(AttributeKey key)
{
    if (key.isValid()) {
        return sKeyNames[key].c_str();
    } else {
        return "INVALID";
    }
}

AttributeType
AttributeKey::getType(AttributeKey key)
{
    if (key.isValid()) {
        return sKeyTypes[key];
    } else {
        return scene_rdl2::rdl2::TYPE_UNKNOWN;
    }
}

size_t
AttributeKey::getSize(AttributeKey key)
{
    if (key.isValid()) {
        return sKeySizes[key];
    } else {
        return 0;
    }
}

bool
AttributeKey::hasDerivatives(AttributeKey key)
{
    if (key.isValid()) {
        return sHasDerivatives[key];
    } else {
        return false;
    }
}

template <typename T>
TypedAttributeKey<T>::TypedAttributeKey()
{
}

template <typename T>
TypedAttributeKey<T>::TypedAttributeKey(AttributeKey k)
{
    MNRY_ASSERT(attributeType<T>() == k.getType());
    mIndex = k.getIndex();
}

template <typename T>
TypedAttributeKey<T>::TypedAttributeKey(const std::string &name,
        bool requestDerivatives)
{
    mIndex = insertKey<T>(name, requestDerivatives);
}

void
StandardAttributes::init()
{
    // Disabling this warning. This method should only be called in
    // program startup time
#pragma warning push
#pragma warning disable 1711
    sUv = TypedAttributeKey<scene_rdl2::math::Vec2f>("uv");
    sClosestSurfaceST = TypedAttributeKey<scene_rdl2::math::Vec2f>("closest_surface_uv");
    sSurfaceST = TypedAttributeKey<scene_rdl2::math::Vec2f>("surface_st");
    sInstanceObjectTransform = TypedAttributeKey<scene_rdl2::math::Mat4f>("instance_object_transform");
    sInstanceTransformLevel0 = TypedAttributeKey<scene_rdl2::math::Mat4f>("instance_transform_0");
    sInstanceTransformLevel1 = TypedAttributeKey<scene_rdl2::math::Mat4f>("instance_transform_1");
    sInstanceTransformLevel2 = TypedAttributeKey<scene_rdl2::math::Mat4f>("instance_transform_2");
    sInstanceTransformLevel3 = TypedAttributeKey<scene_rdl2::math::Mat4f>("instance_transform_3");
    sInstanceTransformLevel4 = TypedAttributeKey<scene_rdl2::math::Mat4f>("instance_transform_4");
    sShadowRayEpsilon = TypedAttributeKey<float>("shadow_ray_epsilon");
    sScatterTag = TypedAttributeKey<float>("scatter_tag");
    sNormal = TypedAttributeKey<scene_rdl2::math::Vec3f>("normal");
    sRefP = TypedAttributeKey<scene_rdl2::math::Vec3f>("ref_P", true); // Request derivatives
    sRefN = TypedAttributeKey<scene_rdl2::math::Vec3f>("ref_N", true); // Request derivatives
    sRefdPds = TypedAttributeKey<scene_rdl2::math::Vec3f>("ref_dPds");
    sdPds = TypedAttributeKey<scene_rdl2::math::Vec3f>("dPds");
    sdPdt = TypedAttributeKey<scene_rdl2::math::Vec3f>("dPdt");
    sVelocity = TypedAttributeKey<scene_rdl2::math::Vec3f>("velocity");
    sP0 = TypedAttributeKey<scene_rdl2::math::Vec3f>("p0");
    sAcceleration = TypedAttributeKey<scene_rdl2::math::Vec3f>("acceleration");
    sMotion = TypedAttributeKey<scene_rdl2::math::Vec3f>("motionvec");
    for (size_t i = 0; i < MAX_NUM_POLYVERTICES; ++i) {
        std::stringstream ss;
        ss << "polyvertex" << i;
        sPolyVertices[i] = TypedAttributeKey<scene_rdl2::math::Vec3f>(ss.str());
    }
    sNumPolyVertices = TypedAttributeKey<int>("numpolyvertices");
    sPolyVertexType = TypedAttributeKey<int>("polyvertex_type");
    sReversedNormals = TypedAttributeKey<bool>("reversed_normals");
    sExplicitShading = TypedAttributeKey<bool>("explicit_shading");
    sId = TypedAttributeKey<int>("id");
#pragma warning pop
}


} // namespace shading
} // namespace moonray



