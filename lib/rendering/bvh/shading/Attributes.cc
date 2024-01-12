// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Attributes.h"

#include <moonray/rendering/mcrt_common/Types.h>

#include <numeric>

using namespace scene_rdl2;
using scene_rdl2::math::Vec2f;
using scene_rdl2::math::Vec3f;


namespace moonray {
namespace shading {

size_t
Attributes::getMemory() const
{
    size_t mem = sizeof(Attributes) +
        scene_rdl2::util::getVectorElementsMemory(mKeyOffset) +
        scene_rdl2::util::getVectorElementsMemory(mTimeSampleCount) +
        scene_rdl2::util::getVectorElementsMemory(mKeyRate) +
        scene_rdl2::util::getVectorElementsMemory(mBeginFaceVarying);

    // 5 vectors for actual primitive attributes data
    mem += scene_rdl2::util::getVectorElementsMemory(mConstants);
    mem += scene_rdl2::util::getVectorElementsMemory(mParts);
    mem += scene_rdl2::util::getVectorElementsMemory(mUniforms);
    mem += scene_rdl2::util::getVectorElementsMemory(mVaryings);
    mem += scene_rdl2::util::getVectorElementsMemory(mFaceVaryings);
    mem += scene_rdl2::util::getVectorElementsMemory(mVertices);
    return mem;
}

template <typename size_type>
std::unique_ptr<Attributes>
Attributes::interleave(const PrimitiveAttributeTable& primitiveAttributeTable,
        size_t numParts, size_t numUniforms, size_t numVaryings,
        const std::vector<size_type>& numFaceVaryings, size_t numVertices)
{
    int largestKey = -1;
    for (const auto& kv : primitiveAttributeTable) {
        AttributeKey key = kv.first;
        if (!kv.second.empty() && key > largestKey) {
            largestKey = key;
        }
    }
    size_t attrNumKeys = largestKey + 1;
    size_t attrMaxNumTimes = 0;
    std::vector<size_t> attrTimeSampleCount(attrNumKeys, 0);
    for (const auto& kv : primitiveAttributeTable) {
        size_t timeSampleCount =
            primitiveAttributeTable.getTimeSampleCount(kv.first);
        attrTimeSampleCount[kv.first] = timeSampleCount;
        if (timeSampleCount > attrMaxNumTimes) {
            attrMaxNumTimes = timeSampleCount;
        }
    }
    std::vector<size_t> attrKeyOffset(attrNumKeys * attrMaxNumTimes, (size_t)-1);
    std::vector<AttributeRate> attrKeyRate(attrNumKeys, RATE_UNKNOWN);
    size_t attrNumFaces = numUniforms;
    std::vector<size_t> attrBeginFaceVarying;
    attrBeginFaceVarying.reserve(attrNumFaces);
    size_t attrNumFaceVaryings = 0;
    for (size_t f = 0; f < numFaceVaryings.size(); ++f) {
        attrBeginFaceVarying.push_back(attrNumFaceVaryings);
        attrNumFaceVaryings += numFaceVaryings[f];
    }

    // make sure data in allocated buffer aligned
    // we sort the key by attribute size, and build key->offset table
    // with size descending order (to avoid wasted padding space)
    size_t dataSize[RATE_LAST] = {};
    std::map<size_t, std::vector<AttributeKey> > attrSizeMap;
    for (const auto& kv : primitiveAttributeTable) {
        AttributeKey key = kv.first;
        attrSizeMap[key.getSize()].push_back(key);
    }

    for (auto rit = attrSizeMap.rbegin(); rit != attrSizeMap.rend(); ++rit) {
        size_t attrSize = rit->first;
        for (auto k : rit->second) {
            size_t nT = primitiveAttributeTable.getTimeSampleCount(k);
            AttributeRate rate = primitiveAttributeTable.getRate(k);
            attrKeyRate[k] = rate;
            // Set the key offsets for each time sample. Time samples
            // are adjacent to each other and separated by the key data size.
            for (size_t t = 0; t < nT; ++t) {
                attrKeyOffset[t * attrNumKeys + k] =
                    dataSize[rate] + t * attrSize;
            }
            // When a key has fewer time samples that attrMaxNumTimes,
            // set the offsets of all subsequent times to point to the last one.
            // This allows us to take time samples of non-motion-blurred
            // attributes without requiring a check for MB on the attribute.
            for (size_t t = nT; t < attrMaxNumTimes; ++t) {
                attrKeyOffset[t * attrNumKeys + k] =
                    dataSize[rate] + (nT - 1) * attrSize;
            }
            dataSize[rate] += attrSize * nT;
        }
    }
    for (size_t i = 0; i < RATE_LAST; i++) {
        dataSize[i] = scene_rdl2::util::alignUp(dataSize[i], sizeof(float));
    }
    // vertex rate alignment is a bit weird since the OSD primitive attribute
    // tessellator require data to be Vec2f aligned
    dataSize[RATE_VERTEX] = scene_rdl2::util::alignUp(dataSize[RATE_VERTEX], sizeof(Vec2f));

    size_t attrNumParts = numParts;
    size_t attrNumVaryings = numVaryings;
    size_t attrNumVertices = numVertices;
    size_t attrPartStride = dataSize[RATE_PART];
    size_t attrUniformStride = dataSize[RATE_UNIFORM];
    size_t attrVaryingStride = dataSize[RATE_VARYING];
    size_t attrFaceVaryingStride = dataSize[RATE_FACE_VARYING];
    size_t attrVertexStride = dataSize[RATE_VERTEX];

    size_t constantSize = dataSize[RATE_CONSTANT];
    size_t partSize = dataSize[RATE_PART]*attrNumParts;
    size_t uniformSize = dataSize[RATE_UNIFORM]*attrNumFaces;
    size_t varyingSize = dataSize[RATE_VARYING]*attrNumVaryings;
    size_t faceVaryingSize = dataSize[RATE_FACE_VARYING]*attrNumFaceVaryings;
    size_t vertexSize = dataSize[RATE_VERTEX]*attrNumVertices;

    Vector<char> attrConstants(constantSize);
    Vector<char> attrParts(partSize);
    Vector<char> attrUniforms(uniformSize);
    Vector<char> attrVaryings(varyingSize);
    Vector<char> attrFaceVaryings(faceVaryingSize);
    Vector<char> attrVertices(vertexSize);

    // Fill in data
    for (const auto& kv : primitiveAttributeTable) {
        AttributeKey key = kv.first;
        AttributeRate keyRate = primitiveAttributeTable.getRate(key);
        for (size_t t = 0; t < kv.second.size(); ++t) {
            const auto& primitiveAttribute = kv.second[t];
            switch (keyRate) {
            case RATE_CONSTANT:
                {
                    size_t offset = attrKeyOffset[t * attrNumKeys + key];
                    primitiveAttribute->fetchData(0, &attrConstants[offset]);
                }
                break;
            case RATE_PART:
                {
                    for (size_t p = 0; p < attrNumParts; p++) {
                        size_t offset = p * attrPartStride +
                            attrKeyOffset[t * attrNumKeys + key];
                        primitiveAttribute->fetchData(p, &attrParts[offset]);
                    }
                }
                break;
            case RATE_UNIFORM:
                {
                    for (size_t f = 0; f < attrNumFaces; f++) {
                        size_t offset = f * attrUniformStride +
                            attrKeyOffset[t * attrNumKeys + key];
                        primitiveAttribute->fetchData(f, &attrUniforms[offset]);
                    }
                }
                break;
            case RATE_VARYING:
                {
                    for (size_t v = 0; v < attrNumVaryings; v++) {
                        size_t offset = v * attrVaryingStride +
                            attrKeyOffset[t * attrNumKeys + key];
                        primitiveAttribute->fetchData(v, &attrVaryings[offset]);
                    }
                }
                break;
            case RATE_FACE_VARYING:
                {
                    size_t fv = 0;
                    for (size_t f = 0; f < attrNumFaces; f++) {
                        size_t nV = numFaceVaryings[f];
                        size_t beginVarying = attrBeginFaceVarying[f];
                        for (size_t v = 0; v < nV; v++) {
                            size_t offset =
                                (v + beginVarying) * attrFaceVaryingStride +
                                attrKeyOffset[t * attrNumKeys + key];
                            primitiveAttribute->fetchData(fv,
                                &attrFaceVaryings[offset]);
                            fv++;
                        }
                    }
                }
                break;
            case RATE_VERTEX:
                {
                    for (size_t v = 0; v < attrNumVertices; v++) {
                        size_t offset = v * attrVertexStride +
                            attrKeyOffset[t * attrNumKeys + key];
                        primitiveAttribute->fetchData(v, &attrVertices[offset]);
                    }
                }
                break;
            default:
                MNRY_ASSERT(false, "Rate unknown");
                break;
            }
        }
    }

    std::unique_ptr<Attributes> result(new Attributes(attrNumKeys, std::move(attrKeyOffset),
        std::move(attrTimeSampleCount), std::move(attrKeyRate),
        attrMaxNumTimes, attrNumParts, attrNumFaces, attrNumVaryings,
        attrNumFaceVaryings, std::move(attrBeginFaceVarying),
        attrNumVertices, attrPartStride, attrUniformStride, attrVaryingStride,
        attrFaceVaryingStride, attrVertexStride,
        std::move(attrConstants), std::move(attrParts), std::move(attrUniforms),
        std::move(attrVaryings), std::move(attrFaceVaryings),
        std::move(attrVertices)));
    return result;
}

void Attributes::negateNormal()
{
    TypedAttributeKey<Vec3f> key = StandardAttributes::sNormal;
    if (!isSupported(key)) {
        return;
    }

    size_t timeSampleCount = getTimeSampleCount(key);
    AttributeRate keyRate = getRate(key);
    switch (keyRate) {
    case RATE_CONSTANT:
        {
            for (size_t t = 0; t < timeSampleCount; ++t) {
                Vec3f& vec3 = getConstantInternal(key, t);
                vec3 = -vec3;
            }
        }
        break;
    case RATE_UNIFORM:
        {
            for (size_t t = 0; t < timeSampleCount; ++t) {
                for (size_t f = 0; f < mNumFaces; ++f) {
                    Vec3f& vec3 = getUniformInternal(key, f, t);
                    vec3 = -vec3;
                }
            }
        }
        break;
    case RATE_VARYING:
        {
            for (size_t t = 0; t < timeSampleCount; ++t) {
                for (size_t v = 0; v < mNumVaryings; ++v) {
                    Vec3f& vec3 = getVaryingInternal(key, v, t);
                    vec3 = -vec3;
                }
            }
        }
        break;
    case RATE_FACE_VARYING:
        {
            for (size_t t = 0; t < timeSampleCount; ++t) {
                for (size_t v = 0; v < mNumFaceVaryings; ++v) {
                    Vec3f& vec3 = getFaceVaryingInternal(key, v, t);
                    vec3 = -vec3;
                }
            }
        }
        break;
    case RATE_VERTEX:
        {
            for (size_t t = 0; t < timeSampleCount; ++t) {
                for (size_t v = 0; v < mNumVertices; ++v) {
                    Vec3f& vec3 = getVertexInternal(key, v, t);
                    vec3 = -vec3;
                }
            }
        }
        break;
    default:
        MNRY_ASSERT(false, "Rate unknown");
        break;
    }
}

void Attributes::reverseFaceVaryingAttributes()
{
    if (mFaceVaryings.empty()) {
        return;
    }

    Vector<char> tmp(mFaceVaryingStride);
    for (size_t f = 0; f < mNumFaces; f++) {
        size_t begin = mFaceVaryingStride * mBeginFaceVarying[f];
        size_t end = f == mNumFaces - 1 ? mFaceVaryings.size() :
                  mFaceVaryingStride * mBeginFaceVarying[f+1];
        size_t nV = (end - begin) / mFaceVaryingStride;

        for (size_t v = 0; v < nV / 2; v++) {
            memcpy(tmp.data(), &mFaceVaryings[begin + v*mFaceVaryingStride], mFaceVaryingStride);
            memcpy(&mFaceVaryings[begin + v*mFaceVaryingStride],
                   &mFaceVaryings[end - (v+1)*mFaceVaryingStride], mFaceVaryingStride);
            memcpy(&mFaceVaryings[end - (v+1)*mFaceVaryingStride], tmp.data(), mFaceVaryingStride);
        }
    }
}

static void
transformVec3(Vec3f& vec3, Vec3Type type, const mcrt_common::Mat43& xform,
        const mcrt_common::Mat43& invXform)
{
    if (type == Vec3Type::POINT) {
        vec3 = transformPoint(xform, vec3);
    } else if (type == Vec3Type::VECTOR) {
        vec3 = transformVector(xform, vec3);
    } else if (type == Vec3Type::NORMAL) {
        vec3 = transformNormal(invXform, vec3);
    } else {
        MNRY_ASSERT(false, "unknown Vec3Type");
    }
}

void
Attributes::transformAttributes(const shading::XformSamples& xforms,
        float shutterOpenDelta, float shutterCloseDelta,
        const std::vector<Vec3KeyType>& keyTypePairs)
{
    shading::XformSamples invXforms;
    invXforms.reserve(xforms.size());
    for (size_t i = 0; i < xforms.size(); ++i) {
        invXforms.push_back(xforms[i].inverse());
    }

    // In rare cases the number of xforms can be less than the number of attribute
    // time samples.  This can happen if there is a problem with the motion blur
    // data on the vertices and we fell back to the static motion blur case.
    // In that case we choose the closest xform available as we can still interpolate
    // the attribute values even though they use the same xform.
    // i.e. xt = std::min(xforms.size() - 1, t);

    for (auto keyTypePair : keyTypePairs) {
        TypedAttributeKey<Vec3f> key = keyTypePair.first;
        Vec3Type type = keyTypePair.second;
        if (!isSupported(key)) {
            continue;
        }
        size_t timeSampleCount = getTimeSampleCount(key);
        AttributeRate keyRate = getRate(key);
        switch (keyRate) {
        case RATE_CONSTANT:
            {
                for (size_t t = 0; t < timeSampleCount; ++t) {
                    size_t xt = std::min(xforms.size() - 1, t);
                    Vec3f& vec3 = getConstantInternal(key, t);
                    transformVec3(vec3, type, xforms[xt], invXforms[xt]);
                }
            }
            break;
        case RATE_UNIFORM:
            {
                for (size_t t = 0; t < timeSampleCount; ++t) {
                    size_t xt = std::min(xforms.size() - 1, t);
                    for (size_t f = 0; f < mNumFaces; ++f) {
                        Vec3f& vec3 = getUniformInternal(key, f, t);
                        transformVec3(vec3, type, xforms[xt], invXforms[xt]);
                    }
                }
            }
            break;
        case RATE_VARYING:
            {
                for (size_t t = 0; t < timeSampleCount; ++t) {
                    size_t xt = std::min(xforms.size() - 1, t);
                    for (size_t v = 0; v < mNumVaryings; ++v) {
                        Vec3f& vec3 = getVaryingInternal(key, v, t);
                        transformVec3(vec3, type, xforms[xt], invXforms[xt]);
                    }
                }
            }
            break;
        case RATE_FACE_VARYING:
            {
                for (size_t t = 0; t < timeSampleCount; ++t) {
                    size_t xt = std::min(xforms.size() - 1, t);
                    for (size_t v = 0; v < mNumFaceVaryings; ++v) {
                        Vec3f& vec3 = getFaceVaryingInternal(key, v, t);
                        transformVec3(vec3, type, xforms[xt], invXforms[xt]);
                    }
                }
            }
            break;
        case RATE_VERTEX:
            {
                for (size_t t = 0; t < timeSampleCount; ++t) {
                    size_t xt = std::min(xforms.size() - 1, t);
                    for (size_t v = 0; v < mNumVertices; ++v) {
                        Vec3f& vec3 = getVertexInternal(key, v, t);
                        transformVec3(vec3, type, xforms[xt], invXforms[xt]);
                    }
                }
            }
            break;
        default:
            MNRY_ASSERT(false, "Rate unknown");
            break;
        }
    }

    motionBlurInterpolate(shutterOpenDelta, shutterCloseDelta);
}

void
Attributes::motionBlurInterpolate(
        float shutterOpenDelta, float shutterCloseDelta)
{
    for (size_t k = 0; k < mNumKeys; ++k) {
        AttributeKey key(k);
        if (!isSupported(key)) {
            continue;
        }
        // no need to interpolate static primitive attributes
        if (getTimeSampleCount(key) < 2) {
            continue;
        }
        switch (key.getType()) {
        case rdl2::TYPE_FLOAT:
            motionBlurInterpolate(TypedAttributeKey<float>(key),
                shutterOpenDelta, shutterCloseDelta);
            break;
        case rdl2::TYPE_RGB:
            motionBlurInterpolate(TypedAttributeKey<math::Color>(key),
                shutterOpenDelta, shutterCloseDelta);
            break;
        case rdl2::TYPE_RGBA:
            motionBlurInterpolate(TypedAttributeKey<math::Color4>(key),
                shutterOpenDelta, shutterCloseDelta);
            break;
        case rdl2::TYPE_VEC2F:
            motionBlurInterpolate(TypedAttributeKey<Vec2f>(key),
                shutterOpenDelta, shutterCloseDelta);
            break;
        case rdl2::TYPE_VEC3F:
            motionBlurInterpolate(TypedAttributeKey<Vec3f>(key),
                shutterOpenDelta, shutterCloseDelta);
            break;
        case rdl2::TYPE_MAT4F:
            motionBlurInterpolate(TypedAttributeKey<math::Mat4f>(key),
                shutterOpenDelta, shutterCloseDelta);
            break;
        default:
            break;
        }
    }
}

template std::unique_ptr<Attributes>
Attributes::interleave(const PrimitiveAttributeTable& primitiveAttributeTable,
        size_t numParts, size_t numUniforms, size_t numVaryings,
        const std::vector<size_t>& numFaceVaryings, size_t numVertices);

template std::unique_ptr<Attributes>
Attributes::interleave(const PrimitiveAttributeTable& primitiveAttributeTable,
        size_t numParts, size_t numUniforms, size_t numVaryings,
        const std::vector<uint32_t>& numFaceVaryings, size_t numVertices);

} // namespace shading
} // namespace rendering


