// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/VertexBuffer.h>
#include <moonray/rendering/mcrt_common/Ray.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/render/logging/logging.h>

namespace moonray{
namespace geom {

// Helper function adapted from PBRT3 that emulates the behavior of
// std::upper_bound(). The implementation here adds some bounds checking
// for corner cases (e.g., making sure that a valid interval is selected
// even in the case the predicate evaluates to true or false for all entries),
// which would normally have to follow a call to std::upper_bound()
template <typename Predicate>
int findInterval(int size, const Predicate& pred)
{
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    const int result = first - 1;
    const int maxRes = size - 2;
    return result < 0 ? 0 :
           result > maxRes ? maxRes : result;
}

namespace internal {

class Instance;

static const float gIllConditioned  = 1000.0f;

float tessCondNumber2x2SVD(float a, float b,
                           float c, float d,
                           float aTolerance);

// The following two function should belong to core math lib
bool computePartialsWithRespect2Texture(const Vec3f &dP_du,
                                        const Vec3f &dP_dv,
                                        const Vec2f &dT_du,
                                        const Vec2f &dT_dv,
                                              Vec3f &dP_ds,
                                              Vec3f &dP_dt);

void writeToObj(const std::string& outputPath,
        const std::vector<uint32_t>& faceVertexCount,
        const VertexBuffer<Vec3fa, InterleavedTraits>& vertices,
        const std::vector<uint32_t>& indices,
        const shading::Vector<Vec2f>& textureVertices,
        const std::vector<uint32_t>& textureIndices);

// Computes instantaneous motion vector of hit point in render space
// units per shutter interval.
//
// pos0:     Hit point position.  If pos1 is not null, pos0 contains the hit
//           point position at rayTime - halfDt.  If instance is not null
//           then pos0 is in local space, otherwise it is in render space.
// pos1:     If not null, this contains the hit point position at rayTime + halfDt.
//           If instance is not null, then pos1 is in local space, otherwise it is
//           in render space.
// rayTime:  Hit time (0 = shutter open, 1 = shutter close)
// instance: If not null, apply instancing transform when computing motion.
//
// TODO:     this function only works with single-level instancing
Vec3f computePrimitiveMotion(const Vec3f &pos0, const Vec3f *pos1, float rayTime,
        const Instance *instance);

// Used when computing motion vectors.  The value is
// in rayTime and represents how much to the left and how
// much too the right of the ray time position value
// should be computed
static const float sHalfDt = 0.01f;

struct Distribution3D
{
    struct Distribution1D
    {
        Distribution1D() = default;

        Distribution1D(const float* func, int n) :
            mFunc(func, func + n), mCdf(n + 1)
        {
            // Compute integral of step function
            mCdf[0] = 0;
            for (int i = 1; i < n + 1; ++i) {
                mCdf[i] = mCdf[i - 1] + mFunc[i - 1] / n;
            }
            // Transform step function integral into CDF
            mFuncInt = mCdf[n];
            if (mFuncInt == 0) {
                for (int i = 1; i < n + 1; ++i) {
                    mCdf[i] = float(i) / float(n);
                }
            } else {
                for (int i = 1; i < n + 1; ++i) {
                    mCdf[i] /= mFuncInt;
                }
            }
        }

        int count() const
        {
            return mFunc.size();
        }

        int sampleDiscrete(float u, float *pdf, float *uRemapped) const
        {
            // Find surrounding CDF segments and offset
            int offset = findInterval(mCdf.size(),
                [&](int index) {
                    return mCdf[index] <= u;
                });
            *pdf = (mFuncInt > 0) ? mFunc[offset] / (mFuncInt * count()) : 0;
            *uRemapped = (u - mCdf[offset]) / (mCdf[offset + 1] - mCdf[offset]);
            return offset;
        }

        float pdfDiscrete(int index) const
        {
            return (mFuncInt > 0) ? mFunc[index] / (mFuncInt * count()) : 0;
        }

        // Distribution1D Public Data
        std::vector<float> mFunc;
        std::vector<float> mCdf;
        float mFuncInt;
    };

    Distribution3D(const float* func, int nx, int ny, int nz) :
        mNx(nx), mNy(ny), mNz(nz)
    {
        mConditionalX.reserve(ny * nz);
        mMarginalY.reserve(nz);
        std::vector<float> xyIntegrals(nz, 0.0f);
        for (int z = 0; z < nz; ++z) {
            std::vector<float> xIntegrals(ny, 0.0f);
            for (int y = 0; y < ny; ++y) {
                int offset = z * (ny * nx) + y * nx;
                mConditionalX.emplace_back(&func[offset], nx);
                xIntegrals[y] = mConditionalX[z * ny + y].mFuncInt;
            }
            mMarginalY.emplace_back(xIntegrals.data(), ny);
            xyIntegrals[z] = mMarginalY[z].mFuncInt;
        }
        mMarginalZ = Distribution1D(xyIntegrals.data(), nz);
    }

    int count() const
    {
        return mMarginalZ.count();
    }

    scene_rdl2::math::Vec3i sampleDiscrete(float u1, float u2, float u3, float *pdf,
        float *uxRemapped, float* uyRemapped, float* uzRemapped) const
    {
        float pdfX, pdfY, pdfZ;
        int z = mMarginalZ.sampleDiscrete(u1, &pdfZ, uzRemapped);
        int y = mMarginalY[z].sampleDiscrete(u2, &pdfY, uyRemapped);
        int x = mConditionalX[z * mNy + y].sampleDiscrete(u3, &pdfX, uxRemapped);
        *pdf = pdfX * pdfY * pdfZ;
        return scene_rdl2::math::Vec3i(x, y, z);
    }

    float pdfDiscrete(const scene_rdl2::math::Vec3i& index) const
    {
        // out of index boundary
        if (index[0] < 0 || index[0] >= mNx ||
            index[1] < 0 || index[1] >= mNy ||
            index[2] < 0 || index[2] >= mNz) {
            return 0.0f;
        }
        float pdfZ = mMarginalZ.pdfDiscrete(index[2]);
        float pdfY = mMarginalY[index[2]].pdfDiscrete(index[1]);
        float pdfX = mConditionalX[index[2] * mNy + index[1]].pdfDiscrete(
            index[0]);
        return pdfX * pdfY * pdfZ;
    }

    int mNx, mNy, mNz;
    std::vector<Distribution1D> mConditionalX;
    std::vector<Distribution1D> mMarginalY;
    Distribution1D mMarginalZ;
};


scene_rdl2::math::Vec3f transformPoint(const scene_rdl2::math::Mat4f mat[2],
                                       const scene_rdl2::math::Vec3f& point,
                                       const float time, bool isMotionBlurOn);

scene_rdl2::math::Vec3f transformVector(const scene_rdl2::math::Mat4f mat[2],
                                        const scene_rdl2::math::Vec3f& vec, 
                                        const float time, bool isMotionBlurOn);

void overrideInstanceAttrs(const mcrt_common::Ray& ray, shading::Intersection& intersection);

bool hasInstanceAttr(shading::AttributeKey key,
                     const mcrt_common::Ray& ray);

template <typename T>
T getInstanceAttr(shading::TypedAttributeKey<T> key,
                  const mcrt_common::Ray& ray,
                  T defaultValue);

template <typename AttributeType> bool
getAttribute(const shading::Attributes* attributes,
             const shading::TypedAttributeKey<AttributeType> key,
             AttributeType& value,
             const int primID,
             const uint32_t vertex=0);

} // namespace internal
} // namespace geom
} // namespace moonray

