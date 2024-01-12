// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Attributes.h
/// $Id$
///

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>

#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/render/util/stdmemory.h>
#include <scene_rdl2/scene/rdl2/Types.h>

using scene_rdl2::rdl2::AttributeType;
using scene_rdl2::rdl2::attributeType;

namespace moonray {
namespace shading {

/**
 * This class represents attribute data associated with each primitive.
 * It is implemented as several pools of raw data with some simple accessors,
 * under the expectation that this data will need to be accessed by ISPC at
 * some point in the future.
 * 
 * The constructor for this class is not client-friendly, and it is expected
 * that clients pass in a PrimitiveAttributeTable (which is basically a map
 * with AttributeKey as key and vector of data as value) with topology info
 * to the util helper method Attributes::interleave to build Attributes objects.
 * The primary members of this class are five memory chunks: mConstants,
 * mUniforms, mVaryings, mFaceVaryings and mVertices. These represent the
 * different rates at which attributes may be stored in a primitive.
 * All other member fields are indicators of either size or how to iterate
 * over these chunks. Below we describe the layout of each rate
 *
 * The general layout is that attributes are laid out one after the other in a 
 * block within their chunk (here we're using "block" to denote one set of 
 * adjacent attributes, and chunk to refer to all the memory to store that type 
 * of variation e.g. mUniforms).  So, for example, in the mUniforms chunk you 
 * will see first attrA,  then attrB, then attrC,  where A, B and C are all 
 * attributes that have been stored at an uniform rate (an attribute  can only 
 * be stored at one of the four rates in a primitive). Face and varying indices 
 * are used to index into which chunk of contiguous attributes you are interested 
 * in. Attributes that vary temporally, have their temporal samples adjacent. 
 * 
 *
 * mConstants
 * mConstants represent data that does not vary (except temporally) within each
 * primitive. Given three attributes, the first varying temporally the second not,
 * and the third varying temporally, the data for these attributes would be laid out 
 * like this in the mConstants chunk:
 *
 * attr0(t=0) | | | | | - mKeyOffset[mNumKeys*0 + attr0]
 * attr0(t=1) | | | | - mKeyOffset[mNumKeys*1 + attr0]
 * attr1(t=0) | | | - mKeyOffset[mNumKeys*0 + attr1]
 * attr2(t=0) | | - mKeyOffset[mNumKeys*0 + attr2]
 * attr2(t=1) | - mKeyOffset[mNumKeys*1 + attr2] 
 *
 * The mKeyOffset member is a two dimensional array (attributes x time) that tracks where 
 * the start of each attribute time sample is within each block. 
 * Thus, mKeyOffset[mNumKeys*1 + attr6] would indicate where the second time sample
 * for attribute 6 is within each block. Attributes that do not have multiple samples
 * will have the same value for both mKeyOffsets so the renderer will see the same
 * values for both time samples without any additional checks.
 * For example, if attribute 5 does not vary temporally, 
 * mKeyOffset[mNumKeys*0 + attr5] == mKeyOffset[mNumKeys*1 + attr5]
 * While mKeyOffset is used for all the data chunks, not just mConstant, I will not be
 * including it in the subsequent diagrams for the sake of clarity.
 * 
 * mUniforms
 * mUniforms represent data that varies by face (or, e.g. curve in the case of curve primitives)
 * Given two attributes, the first varying temporally and the second one not, 
 * the data for these attributes would be laid out like this in the mUniforms chunk:
 *
 * attr0(t=0,f=0)  |
 * attr0(t=1,f=0)  | - mUniformStride
 * attr1(t=0,f=0)  |
 * attr0(t=0,f=1)
 * attr0(t=1,f=1)
 * attr1(t=0,f=1)
 * attr0(t=0,f=2)
 * attr0(t=1,f=2)
 * attr1(t=0,f=2)
 * ...
 *
 * mVaryings
 * mVaryings represent data that varies by vertex (for mesh) or by span (for curve), 
 * it is stored similarly to mUniforms.
 * Given two attributes, the first varying temporally and the second one not,
 * the data for these attributes would be laid out like this in the mVaryings chunk:
 * 
 * attr0(t=0,v=0)  | 
 * attr0(t=1,v=0)  | - mVaryingStride
 * attr1(t=0,v=0)  |
 * attr0(t=0,v=1)
 * attr0(t=1,v=1)
 * attr1(t=0,v=1)
 * attr0(t=0,v=2)
 * attr0(t=1,v=2)
 * attr1(t=0,v=2)
 * ...
 * 
 * mFaceVaryings
 * mFaceVaryings represent data that varies by both face and vertex(for mesh)/span(for curve). 
 * Given two attributes, the first varying temporally and the second one not,
 * the data for these attributes would be laid out like this in the mFaceVaryings chunk:
 * 
 * attr0(t=0,f=0,v=0)  | | |
 * attr0(t=1,f=0,v=0)  | | | - mFaceVaryingStride
 * attr1(t=0,f=0,v=0)  | | |
 * attr0(t=0,f=0,v=1)  | |
 * attr0(t=1,f=0,v=1)  | |
 * attr1(t=0,f=0,v=1)  | |
 * attr0(t=0,f=0,v=2)  | | - mFaceVaryingStride*mBeginFaceVarying[1]
 * attr0(t=1,f=0,v=2)  | | 
 * attr1(t=0,f=0,v=2)  | |
 * attr0(t=0,f=0,v=3)  | |
 * attr0(t=1,f=0,v=3)  | |
 * attr1(t=0,f=0,v=3)  | |
 * attr0(t=0,f=1,v=0)  |
 * attr0(t=1,f=1,v=0)  |
 * attr1(t=0,f=1,v=0)  |
 * attr0(t=0,f=1,v=1)  | - mFaceVaryingStride*mBeginFaceVarying[2]
 * attr0(t=1,f=1,v=1)  |
 * attr1(t=0,f=1,v=1)  |
 * attr0(t=0,f=1,v=2)  |
 * attr0(t=1,f=1,v=2)  |
 * attr1(t=0,f=1,v=2)  |
 * attr0(t=0,f=2,v=0)
 * ...
 * 
 * mFaceVaryingStride is used to denote the size of each face varying attribute 
 * block, so adjacent vertices/spans are one mFaceVaryingStride apart.
 * Each face(curve) may have a varying number of vertices(spans), so we do not have a fixed
 * distance between the beginning of each face. Instead, we maintain an array
 * called mBeginFaceVarying that indicates at which block of attributes the first 
 * vertex/span for a given face(curve) begins.
 * 
 */

class Attributes 
{
 public:
    Attributes(size_t numKeys,
            std::vector<size_t>&& keyOffset,
            std::vector<size_t>&& timeSampleCount,
            std::vector<AttributeRate>&& keyRate,
            size_t maxNumTimes,
            size_t numParts,
            size_t numFaces,
            size_t numVaryings,
            size_t numFaceVaryings,
            std::vector<size_t>&& beginFaceVarying,
            size_t numVertices,
            size_t partStride,
            size_t uniformStride,
            size_t varyingStride,
            size_t faceVaryingStride,
            size_t vertexStride,
            Vector<char>&& constants,
            Vector<char>&& parts,
            Vector<char>&& uniforms,
            Vector<char>&& varyings,
            Vector<char>&& faceVaryings,
            Vector<char>&& vertices):
            mNumKeys(numKeys),
            mKeyOffset(std::move(keyOffset)),
            mTimeSampleCount(std::move(timeSampleCount)),
            mKeyRate(std::move(keyRate)),
            mMaxNumTimes(maxNumTimes),
            mNumParts(numParts),
            mNumFaces(numFaces),
            mNumVaryings(numVaryings),
            mNumFaceVaryings(numFaceVaryings),
            mBeginFaceVarying(std::move(beginFaceVarying)),
            mNumVertices(numVertices),
            mPartStride(partStride),
            mUniformStride(uniformStride),
            mVaryingStride(varyingStride),
            mFaceVaryingStride(faceVaryingStride),
            mVertexStride(vertexStride),
            mConstants(std::move(constants)),
            mParts(std::move(parts)),
            mUniforms(std::move(uniforms)),
            mVaryings(std::move(varyings)),
            mFaceVaryings(std::move(faceVaryings)),
            mVertices(std::move(vertices))
    {
    }

    std::unique_ptr<Attributes> copy() const
    {
        std::vector<size_t> keyOffset = mKeyOffset;
        std::vector<size_t> timeSampleCount = mTimeSampleCount;
        std::vector<AttributeRate> keyRate = mKeyRate;
        std::vector<size_t> beginFaceVarying = mBeginFaceVarying;
        Vector<char> constants = mConstants;
        Vector<char> parts = mParts;
        Vector<char> uniforms = mUniforms;
        Vector<char> varyings = mVaryings;
        Vector<char> faceVaryings = mFaceVaryings;
        Vector<char> vertices = mVertices;

        return fauxstd::make_unique<Attributes>(
            mNumKeys,
            std::move(keyOffset),
            std::move(timeSampleCount),
            std::move(keyRate),
            mMaxNumTimes,
            mNumParts,
            mNumFaces,
            mNumVaryings,
            mNumFaceVaryings,
            std::move(beginFaceVarying),
            mNumVertices,
            mUniformStride,
            mPartStride,
            mVaryingStride,
            mFaceVaryingStride,
            mVertexStride,
            std::move(constants),
            std::move(parts),
            std::move(uniforms),
            std::move(varyings),
            std::move(faceVaryings),
            std::move(vertices)
        );
    }
    
    finline size_t keyOffset(int key, int time) const;

    finline size_t getTimeSampleCount(int key) const;

    finline size_t getMaxTimeSampleCount() const;

    template <typename T>
    finline void getConstant(TypedAttributeKey<T> key,
            char* data, int time = 0) const;

    template <typename T>
    finline const T &getConstant(TypedAttributeKey<T> key,
            int time = 0) const;

    template <typename T>
    finline void getPart(TypedAttributeKey<T> key,
            size_t part, char* data, int time = 0) const;

    template <typename T>
    finline const T &getPart(TypedAttributeKey<T> key,
            size_t part, int time = 0) const;

    template <typename T>
    finline void getUniform(TypedAttributeKey<T> key,
            size_t face, char* data, int time = 0) const;

    template <typename T>
    finline const T &getUniform(TypedAttributeKey<T> key,
            size_t face, int time = 0) const;

    template <typename T>
    finline const T &getVarying(TypedAttributeKey<T> key,
            size_t varying, int time = 0) const;

    template <typename T>
    finline void setVertex(TypedAttributeKey<T> key,
            const T& data, size_t vertex, int time = 0);

    template <typename T>
    finline void setFaceVarying(TypedAttributeKey<T> key, const T& data,
            size_t face, size_t varying, int time = 0);

    template <typename T>
    finline const T &getFaceVarying(TypedAttributeKey<T> key,
            size_t face, size_t varying, int time = 0) const;

    template <typename T>
    finline const T &getVertex(TypedAttributeKey<T> key,
            size_t vertex, int time = 0) const;

    template <typename T>
    finline T getMotionBlurConstant(TypedAttributeKey<T> key,
            float time) const;

    template <typename T>
    finline T getMotionBlurUniform(TypedAttributeKey<T> key,
            size_t face, float time) const;

    template <typename T>
    finline T getMotionBlurVarying(TypedAttributeKey<T> key,
            size_t varying, float time) const;

    template <typename T>
    finline T getMotionBlurFaceVarying(TypedAttributeKey<T> key,
            size_t face, size_t varying, float time) const;

    template <typename T>
    finline T getMotionBlurVertex(TypedAttributeKey<T> key,
            size_t vertex, float time) const;


    finline bool isSupported(AttributeKey key) const;
    finline AttributeRate getRate(AttributeKey key) const;

    template<typename size_type>
    static std::unique_ptr<Attributes> interleave(const PrimitiveAttributeTable&,
            size_t numParts, size_t numUniforms, size_t numVaryings,
            const std::vector<size_type>& numFaceVaryings, size_t numVertices);

    void transformAttributes(const XformSamples& xforms,
            float shutterOpenDelta, float shutterCloseDelta,
            const std::vector<Vec3KeyType>& keyTypePairs);

    void negateNormal();

    void reverseFaceVaryingAttributes();

    size_t getMemory() const;

    finline bool hasAttribute(AttributeKey key) const {
        return (0 <= key && key < (int)mNumKeys && keyOffset(key, 0) != (size_t)-1);
    }

    finline bool hasVaryingAttributes() const {
        return !mVaryings.empty();
    }

    finline bool hasFaceVaryingAttributes() const {
        return !mFaceVaryings.empty();
    }

    finline bool hasVertexAttributes() const {
        return !mVertices.empty();
    }

    // for varying/vertex rate tessellation, we offer following low level
    // accesors that tessellator can resize/tessellate/update attributes data.
    // The clients of accessors are primitives that would need to tessellate
    // the varying/vertex rate primitive attribute.
    finline void resizeVaryingAttributes(size_t newVaryingCount) {
        mNumVaryings = newVaryingCount;
        mVaryings.resize(newVaryingCount * mVaryingStride);
    }

    finline void resizeVertexAttributes(size_t newVertexCount) {
        mNumVertices = newVertexCount;
        mVertices.resize(newVertexCount * mVertexStride);
    }

    finline size_t getVaryingAttributesStride() const {
        return mVaryingStride;
    }

    finline size_t getVertexAttributesStride() const {
        return mVertexStride;
    }

    finline float* getVaryingAttributesData() {
        return reinterpret_cast<float*>(mVaryings.data());
    }

    finline float* getVertexAttributesData() {
        return reinterpret_cast<float*>(mVertices.data());
    }

private:

    template <typename T>
    finline T& getConstantInternal(TypedAttributeKey<T> key,
            int time = 0);

    template <typename T>
    finline T& getUniformInternal(TypedAttributeKey<T> key,
            size_t face, int time = 0);

    template <typename T>
    finline T& getVaryingInternal(TypedAttributeKey<T> key,
            size_t varying, int time = 0);

    template <typename T>
    finline T& getFaceVaryingInternal(TypedAttributeKey<T> key,
            size_t faceVarying, int time = 0);

    template <typename T>
    finline T& getVertexInternal(TypedAttributeKey<T> key,
            size_t vertex, int time = 0);

    // interpolate primitive attribute from frame time coordinate
    // (where user supply data) to shutter time coordinate
    // (where shutter open time = 0 and shutter close time = 1)
    // since all the rendering calculation is in shutter time coordinate
    void motionBlurInterpolate(
            float shutterOpenDelta, float shutterCloseDelta);

    template <typename T>
    void motionBlurInterpolate(TypedAttributeKey<T> key,
            float shutterOpenDelta, float shutterCloseDelta);

private:
    size_t mNumKeys;
    // TODO: This table is somewhat memory-wasteful in the service
    // of performance (it allows a direct lookup).
    // The table memory use is proportional to the total number of
    // AttributeKeys and will become a problem if there's a large
    // number of primitives and the number of AttributeKeys grows too large.
    // Two solutions:
    // 1) Do an indirect lookup, so that only offsets for the keys actually
    //    used (as opposed to all keys) are stored here.
    // 2) Most primitives of the same type will likely have the exact
    //    same entries in this table and the table is not modified after
    //    construction, so we could share tables among primitives.
    //   (this would be my recommended solution)
    std::vector<size_t> mKeyOffset;
    std::vector<size_t> mTimeSampleCount;
    std::vector<AttributeRate> mKeyRate;

    size_t mMaxNumTimes;
    size_t mNumParts;
    size_t mNumFaces;
    size_t mNumVaryings;
    size_t mNumFaceVaryings;
    std::vector<size_t> mBeginFaceVarying;
    size_t mNumVertices;

    size_t mPartStride;
    size_t mUniformStride;
    size_t mVaryingStride;
    size_t mFaceVaryingStride;
    size_t mVertexStride;

    Vector<char> mConstants;
    Vector<char> mParts;
    Vector<char> mUniforms;
    Vector<char> mVaryings;
    Vector<char> mFaceVaryings;
    Vector<char> mVertices;
};


size_t
Attributes::keyOffset(int key, int time) const 
{
    return mKeyOffset[time*mNumKeys+key];
}

size_t
Attributes::getTimeSampleCount(int key) const
{
    MNRY_ASSERT(hasAttribute(key));
    return mTimeSampleCount[key];
}

size_t
Attributes::getMaxTimeSampleCount() const
{
    return mMaxNumTimes;
}

template <typename T>
void
Attributes::getConstant(TypedAttributeKey<T> key, char* data, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* constData = const_cast<char*>(&mConstants[offset]);
    *(reinterpret_cast<T *>(data)) = *(reinterpret_cast<T *>(constData));
}

template <>
inline void
Attributes::getConstant(TypedAttributeKey<std::string> key, char* data, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* constData = const_cast<char*>(&mConstants[offset]);
    *(reinterpret_cast<std::string **>(data)) =
        *(reinterpret_cast<std::string **>(constData));
}

template <typename T>
const T&
Attributes::getConstant(TypedAttributeKey<T> key, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* constData = const_cast<char*>(&mConstants[offset]);
    return *(reinterpret_cast<T *>(constData));
}

template <>
inline const std::string&
Attributes::getConstant(TypedAttributeKey<std::string> key, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* constData = const_cast<char*>(&mConstants[offset]);
    return *(*(reinterpret_cast<std::string **>(constData)));
}

template <typename T>
void
Attributes::getPart(TypedAttributeKey<T> key, size_t part, char* data, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* partData = const_cast<char*>(&mParts[part * mPartStride + offset]);
    MNRY_ASSERT(partData);
    *(reinterpret_cast<T *>(data)) = *(reinterpret_cast<T *>(partData));
}

template <>
inline void
Attributes::getPart(TypedAttributeKey<std::string> key, size_t part, char* data, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* partData = const_cast<char*>(&mParts[part * mPartStride + offset]);
    *(reinterpret_cast<std::string **>(data)) =
        *(reinterpret_cast<std::string **>(partData));
}

template <typename T>
const T&
Attributes::getPart(TypedAttributeKey<T> key, size_t part, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* partData = const_cast<char*>(&mParts[part * mPartStride + offset]);
    return *(reinterpret_cast<T *>(partData));
}

template <>
inline const std::string&
Attributes::getPart(TypedAttributeKey<std::string> key, size_t part, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* partData = const_cast<char*>(&mParts[part * mPartStride + offset]);
    return *(*(reinterpret_cast<std::string **>(partData)));
}

template <typename T>
void
Attributes::getUniform(TypedAttributeKey<T> key, size_t face, char* data, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* uniformData = const_cast<char*>(&mUniforms[face * mUniformStride + offset]);
    *(reinterpret_cast<T *>(data)) = *(reinterpret_cast<T *>(uniformData));
}

template <>
inline void
Attributes::getUniform(TypedAttributeKey<std::string> key, size_t face, char* data, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* uniformData = const_cast<char*>(&mUniforms[face * mUniformStride + offset]);
    *(reinterpret_cast<std::string **>(data)) = 
        *(reinterpret_cast<std::string **>(uniformData));
}

template <typename T>
const T&
Attributes::getUniform(TypedAttributeKey<T> key, size_t face, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* uniformData = const_cast<char*>(&mUniforms[face * mUniformStride + offset]);
    return *(reinterpret_cast<T *>(uniformData));
}

template <>
inline const std::string&
Attributes::getUniform(TypedAttributeKey<std::string> key, size_t face, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* uniformData = const_cast<char*>(&mUniforms[face * mUniformStride + offset]);
    return *(*(reinterpret_cast<std::string **>(uniformData)));
}

template <typename T>
const T &
Attributes::getVarying(TypedAttributeKey<T> key, size_t varying, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* varyingData = const_cast<char*>(&mVaryings[varying * mVaryingStride + offset]);
    return *(reinterpret_cast<T *>(varyingData));
}

template <typename T>
void
Attributes::setVertex(TypedAttributeKey<T> key, const T& data, size_t vertex, int time)
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* vertexData = const_cast<char*>(&mVertices[vertex * mVertexStride + offset]);
    *(reinterpret_cast<T *>(vertexData)) = data;
}

template <typename T>
void
Attributes::setFaceVarying(TypedAttributeKey<T> key, const T& data,
        size_t face, size_t varying, int time)
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    size_t faceVaryingIndex = mBeginFaceVarying[face]+varying;
    char* faceVaryingData = const_cast<char*>(
        &mFaceVaryings[faceVaryingIndex * mFaceVaryingStride + offset]);
    *(reinterpret_cast<T *>(faceVaryingData)) = data;
}

template <typename T>
const T &
Attributes::getFaceVarying(TypedAttributeKey<T> key, size_t face, size_t varying, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    size_t faceVaryingIndex = mBeginFaceVarying[face]+varying;
    char* faceVaryingData = const_cast<char*>(
        &mFaceVaryings[faceVaryingIndex * mFaceVaryingStride + offset]);
    return *(reinterpret_cast<T *>(faceVaryingData));
}

template <typename T>
const T &
Attributes::getVertex(TypedAttributeKey<T> key, size_t vertex, int time) const
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    char* vertexData = const_cast<char*>(&mVertices[vertex * mVertexStride + offset]);
    return *(reinterpret_cast<T *>(vertexData));
}

template <typename T>
T
Attributes::getMotionBlurConstant(TypedAttributeKey<T> key, float time) const
{
    const T& c0 = getConstant(key, 0);
    const T& c1 = getConstant(key, 1);
    return (1.0f - time) * c0 + time * c1;
}

template <typename T>
T
Attributes::getMotionBlurUniform(TypedAttributeKey<T> key, size_t face,
        float time) const
{
    const T& u0 = getUniform(key, face, 0);
    const T& u1 = getUniform(key, face, 1);
    return (1.0f - time) * u0 + time * u1;
}

template <typename T>
T
Attributes::getMotionBlurVarying(TypedAttributeKey<T> key, size_t varying,
        float time) const
{
    const T& v0 = getVarying(key, varying, 0);
    const T& v1 = getVarying(key, varying, 1);
    return (1.0f - time) * v0 + time * v1;
}

template <typename T>
T
Attributes::getMotionBlurFaceVarying(TypedAttributeKey<T> key, size_t face,
        size_t varying, float time) const
{
    const T& fv0 = getFaceVarying(key, face, varying, 0);
    const T& fv1 = getFaceVarying(key, face, varying, 1);
    return (1.0f - time) * fv0 + time * fv1;
}

template <typename T>
T
Attributes::getMotionBlurVertex(TypedAttributeKey<T> key, size_t vertex,
        float time) const
{
    const T& v0 = getVertex(key, vertex, 0);
    const T& v1 = getVertex(key, vertex, 1);
    return (1.0f - time) * v0 + time * v1;
}

bool 
Attributes::isSupported(AttributeKey key) const
{
    return 0 <= key && key < (int)mNumKeys && mKeyRate[key] != RATE_UNKNOWN;
}

AttributeRate 
Attributes::getRate(AttributeKey key) const
{
    MNRY_ASSERT(0 <= key && key < (int)mNumKeys);
    return mKeyRate[key];
}

template <typename T>
T&
Attributes::getConstantInternal(TypedAttributeKey<T> key, int time)
{
    return const_cast<T&>(static_cast<const Attributes&>(*this).getConstant(
        key, time));
}

template <typename T>
T&
Attributes::getUniformInternal(TypedAttributeKey<T> key, size_t face, int time)
{
    return const_cast<T&>(static_cast<const Attributes&>(*this).getUniform(
        key, face, time));
}

template <typename T>
T&
Attributes::getVaryingInternal(TypedAttributeKey<T> key, size_t varying, int time)
{
    return const_cast<T&>(static_cast<const Attributes&>(*this).getVarying(
        key, varying, time));
}

template <typename T>
T&
Attributes::getFaceVaryingInternal(TypedAttributeKey<T> key, size_t faceVarying,
        int time)
{
    MNRY_ASSERT(hasAttribute(key));
    size_t offset = keyOffset(key, time);
    return *(reinterpret_cast<T *>(mFaceVaryings.data() +
        offset + faceVarying * mFaceVaryingStride));
}

template <typename T>
T&
Attributes::getVertexInternal(TypedAttributeKey<T> key, size_t vertex, int time)
{
    return const_cast<T&>(static_cast<const Attributes&>(*this).getVertex(
        key, vertex, time));
}

template <typename T>
void
Attributes::motionBlurInterpolate(TypedAttributeKey<T> key,
        float shutterOpenDelta, float shutterCloseDelta)
{
    AttributeRate keyRate = getRate(key);
    MNRY_ASSERT_REQUIRE(getTimeSampleCount(key) == 2,
        "only support two time samples for motionblur at this moment");

    switch (keyRate) {
    case RATE_CONSTANT:
        {
            T& t0 = getConstantInternal(key, 0);
            T& t1 = getConstantInternal(key, 1);
            T newT0 = scene_rdl2::math::lerp(t0, t1, shutterOpenDelta);
            T newT1 = scene_rdl2::math::lerp(t0, t1, shutterCloseDelta);
            t0 = newT0;
            t1 = newT1;
        }
        break;
    case RATE_UNIFORM:
        {
            for (size_t f = 0; f < mNumFaces; ++f) {
                T& t0 = getUniformInternal(key, f, 0);
                T& t1 = getUniformInternal(key, f, 1);
                T newT0 = scene_rdl2::math::lerp(t0, t1, shutterOpenDelta);
                T newT1 = scene_rdl2::math::lerp(t0, t1, shutterCloseDelta);
                t0 = newT0;
                t1 = newT1;
            }
        }
        break;
    case RATE_VARYING:
        {
            for (size_t v = 0; v < mNumVaryings; ++v) {
                T& t0 = getVaryingInternal(key, v, 0);
                T& t1 = getVaryingInternal(key, v, 1);
                T newT0 = scene_rdl2::math::lerp(t0, t1, shutterOpenDelta);
                T newT1 = scene_rdl2::math::lerp(t0, t1, shutterCloseDelta);
                t0 = newT0;
                t1 = newT1;
            }
        }
        break;
    case RATE_FACE_VARYING:
        {
            for (size_t v = 0; v < mNumFaceVaryings; ++v) {
                T& t0 = getFaceVaryingInternal(key, v, 0);
                T& t1 = getFaceVaryingInternal(key, v, 1);
                T newT0 = scene_rdl2::math::lerp(t0, t1, shutterOpenDelta);
                T newT1 = scene_rdl2::math::lerp(t0, t1, shutterCloseDelta);
                t0 = newT0;
                t1 = newT1;
            }
        }
        break;
    case RATE_VERTEX:
        {
            for (size_t v = 0; v < mNumVertices; ++v) {
                T& t0 = getVertexInternal(key, v, 0);
                T& t1 = getVertexInternal(key, v, 1);
                T newT0 = scene_rdl2::math::lerp(t0, t1, shutterOpenDelta);
                T newT1 = scene_rdl2::math::lerp(t0, t1, shutterCloseDelta);
                t0 = newT0;
                t1 = newT1;
            }
        }
        break;
    default:
        MNRY_ASSERT(false, "Rate unknown");
        break;
    }
}

} // namespace shading
} // namespace rendering



