// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Interpolator
/// $Id$
///

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>

#include <array>

namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

class Interpolator
{
public:
    template <typename T>
    bool interpolate(TypedAttributeKey<T> key, char* data) const {
        if (unlikely(!mAttr->isSupported(key))) {
            return false;
        }
        bool isValidInterpolation = true;
        AttributeRate rate = mAttr->getRate(key);
        switch (rate) {
        case RATE_CONSTANT:
            interpolateConstant(key, data);
            break;
        case RATE_PART:
            interpolatePart(key, data);
            break;
        case RATE_UNIFORM:
            interpolateUniform(key, data);
            break;
        case RATE_VARYING:
            interpolateVaryings(key, data);
            break;
        case RATE_FACE_VARYING:
            interpolateFaceVaryings(key, data);
            break;
        case RATE_VERTEX:
            interpolateVertices(key, data);
            break;
        default:
            isValidInterpolation = false;
            break;
        }
        return isValidInterpolation;
    }

protected:
    Interpolator(const shading::Attributes *attr, float time, int part,
                 int coarseFace, int numVaryings,
                 const int *varyings,
                 const float *varyingWeights,
                 int numFaceVaryings,
                 const int *faceVaryings,
                 const float *faceVaryingWeights,
                 int tessellatedFace, int numVertices,
                 const int *vertices,
                 const float *vertexWeights) :
        mAttr(attr),
        mTime(time),
        mPart(part),
        mCoarseFace(coarseFace), 
        mNumVaryings(numVaryings),
        mVaryings(varyings),
        mVaryingWeights(varyingWeights),
        mNumFaceVaryings(numFaceVaryings),
        mFaceVaryings(faceVaryings),
        mFaceVaryingWeights(faceVaryingWeights),
        mTessellatedFace(tessellatedFace), 
        mNumVertices(numVertices), 
        mVertices(vertices),
        mVertexWeights(vertexWeights)
    {}

    template <typename T>
    void interpolateConstant(TypedAttributeKey<T> key, char* data) const {
        int timeSampleCount = mAttr->getTimeSampleCount(key);
        if (timeSampleCount == 1) {
            mAttr->getConstant(key, data);
        } else {
            // TODO only support two time samples for motion blur case right now
            MNRY_ASSERT(timeSampleCount == 2);
            T& result = *(reinterpret_cast<T*>(data));
            result = (T)(
                (1.0f - mTime) * mAttr->getConstant(key, 0) +
                (       mTime) * mAttr->getConstant(key, 1));
        }
    }

    template <typename T>
    void interpolatePart(TypedAttributeKey<T> key, char* data) const {
        int timeSampleCount = mAttr->getTimeSampleCount(key);
        if (timeSampleCount == 1) {
            mAttr->getPart(key, mPart, data);
        } else {
            // TODO only support two time samples for motion blur case right now
            MNRY_ASSERT(timeSampleCount == 2);
            T& result = *(reinterpret_cast<T*>(data));
            result = (T)(
                (1.0f - mTime) * mAttr->getPart(key, mPart, 0) +
                (       mTime) * mAttr->getPart(key, mPart, 1));
        }
    }

    template <typename T>
    void interpolateUniform(TypedAttributeKey<T> key, char* data) const {
        int timeSampleCount = mAttr->getTimeSampleCount(key);
        if (timeSampleCount == 1) {
            mAttr->getUniform(key, mCoarseFace, data);
        } else {
            // TODO only support two time samples for motion blur case right now
            MNRY_ASSERT(timeSampleCount == 2);
            T& result = *(reinterpret_cast<T*>(data));
            result = (T)(
                (1.0f - mTime) * mAttr->getUniform(key, mCoarseFace, 0) +
                (       mTime) * mAttr->getUniform(key, mCoarseFace, 1));
        }
    }

    template <typename T>
    void interpolateVaryings(TypedAttributeKey<T> key, char* data) const {
        T& result = *(reinterpret_cast<T*>(data));
        result = T(scene_rdl2::math::zero);
        int timeSampleCount = mAttr->getTimeSampleCount(key);
        if (timeSampleCount == 1) {
            for (int i = 0; i < mNumVaryings; i++) {
                int v = mVaryings[i];
                float w = mVaryingWeights[i];
                result = result + (T)(w * mAttr->getVarying(key, v));
            }
        } else {
            // TODO only support two time samples for motion blur case right now
            MNRY_ASSERT(timeSampleCount == 2);
            for (int i = 0; i < mNumVaryings; i++) {
                int v = mVaryings[i];
                float w = mVaryingWeights[i];
                result = result + (T)(w * (
                    (1.0f - mTime) * mAttr->getVarying(key, v, 0) +
                    (       mTime) * mAttr->getVarying(key, v, 1)));
            }
        }
    }

    template <typename T>
    void interpolateFaceVaryings(TypedAttributeKey<T> key, char* data) const {
        T& result = *(reinterpret_cast<T*>(data));
        result = T(scene_rdl2::math::zero);
        int timeSampleCount = mAttr->getTimeSampleCount(key);
        if (timeSampleCount == 1) {
            for (int i = 0; i < mNumFaceVaryings; i++) {
                int v = mFaceVaryings[i];
                float w = mFaceVaryingWeights[i];
                result = result + (T)(w *
                    mAttr->getFaceVarying(key, mCoarseFace, v));
            }
        } else {
            // TODO only support two time samples for motion blur case right now
            MNRY_ASSERT(timeSampleCount == 2);
            for (int i = 0; i < mNumFaceVaryings; i++) {
                int v = mFaceVaryings[i];
                float w = mFaceVaryingWeights[i];
                result = result + (T)(w * (
                    (1.0f - mTime) * mAttr->getFaceVarying(key, mCoarseFace, v, 0) +
                    (       mTime) * mAttr->getFaceVarying(key, mCoarseFace, v, 1)));
            }
        }
    }

    template <typename T>
    void interpolateVertices(TypedAttributeKey<T> key, char* data) const {
        T& result = *(reinterpret_cast<T*>(data));
        result = T(scene_rdl2::math::zero);
        int timeSampleCount = mAttr->getTimeSampleCount(key);
        if (timeSampleCount == 1) {
            for (int i = 0; i < mNumVertices; i++) {
                int v = mVertices[i];
                float w = mVertexWeights[i];
                result = result + (T)(w * mAttr->getVertex(key, v));
            }
        } else {
            // TODO only support two time samples for motion blur case right now
            MNRY_ASSERT(timeSampleCount == 2);
            for (int i = 0; i < mNumVertices; i++) {
                int v = mVertices[i];
                float w = mVertexWeights[i];
                result = result + (T)(w * (
                    (1.0f - mTime) * mAttr->getVertex(key, v, 0) +
                    (       mTime) * mAttr->getVertex(key, v, 1)));
            }
        }
    }

    const shading::Attributes *mAttr;
    float mTime;
    int mPart;
    int mCoarseFace;
    int mNumVaryings;
    const int *mVaryings;
    const float *mVaryingWeights;
    int mNumFaceVaryings;
    const int *mFaceVaryings;
    const float *mFaceVaryingWeights;
    int mTessellatedFace;
    int mNumVertices;
    const int *mVertices;
    const float *mVertexWeights;
};

template <>
inline bool Interpolator::interpolate(TypedAttributeKey<bool> key, char* data) const {
    if (unlikely(!mAttr->isSupported(key))) {
        return false;
    }
    bool isValidInterpolation = true;
    MNRY_ASSERT(mAttr->getTimeSampleCount(key) == 1);
    AttributeRate rate = mAttr->getRate(key);
    switch (rate) {
    case RATE_CONSTANT:
        mAttr->getConstant(key, data);
        break;
    case RATE_PART:
        mAttr->getPart(key, mPart, data);
        break;
    case RATE_UNIFORM:
        mAttr->getUniform(key, mCoarseFace, data);
        break;
    default:
        MNRY_ASSERT(false, (std::string("Bool Atttribute ") + std::string(key.getName()) +
            std::string(" only supports constant/uniform rate\n")).c_str());
        isValidInterpolation = false;
        break;
    }
    return isValidInterpolation;
}

template <>
inline bool Interpolator::interpolate(TypedAttributeKey<std::string> key, char* data) const {
    if (unlikely(!mAttr->isSupported(key))) {
        return false;
    }
    bool isValidInterpolation = true;
    MNRY_ASSERT(mAttr->getTimeSampleCount(key) == 1);
    AttributeRate rate = mAttr->getRate(key);
    switch (rate) {
    case RATE_CONSTANT:
        mAttr->getConstant(key, data);
        break;
    case RATE_PART:
        mAttr->getPart(key, mPart, data);
        break;
    case RATE_UNIFORM:
        mAttr->getUniform(key, mCoarseFace, data);
        break;
    default:
        MNRY_ASSERT(false, (std::string("String Atttribute ") + std::string(key.getName()) +
            std::string(" only supports constant/uniform rate\n")).c_str());
        isValidInterpolation = false;
        break;
    }
    return isValidInterpolation;
}


class QuadricInterpolator : public Interpolator
{
public:
    QuadricInterpolator(const shading::Attributes *attr, float time, float u , float v):
        Interpolator(attr, time,
        0,               // part
        0,               // coarseFace
        4,               // numVaryings
        mIndices.data(), // varyings
        mWeights.data(), // varyingWeights
        4,               // numFaceVaryings
        mIndices.data(), // faceVaryings
        mWeights.data(), // faceVaryingWeights
        0,               // tessellatedFace
        4,               // numVertices
        mIndices.data(), // vertices
        mWeights.data()) // vertexWeights
        {
            mWeights[0] = (1.0f - u) * (1.0f - v);
            mWeights[1] = (       u) * (1.0f - v);
            mWeights[2] = (       u) * (       v);
            mWeights[3] = (1.0f - u) * (       v);
        }

private:
    std::array<float, 4> mWeights;
    static constexpr std::array<int, 4> mIndices = { {0, 1, 2, 3} };
};

} // namespace shading
} // namespace moonray



