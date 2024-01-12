// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Util.h"

#include <moonray/rendering/geom/prim/Instance.h>

#include <scene_rdl2/common/platform/Platform.h>

#include <fstream>

namespace moonray {
namespace geom {
namespace internal {

//analytic solution of SVD based condition number calculation
float tessCondNumber2x2SVD(
    float a, float b,
    float c, float d,
    float aTolerance
    )
{
    MNRY_ASSERT(aTolerance>0.0, "need a positive tolerance");

    //compute AA^T = AAT_j AAT_l
    //               AAT_l AAT_k
    float AAT_j = a * a + b * b;
    float AAT_l = a * c + b * d;
    float AAT_k = c * c + d * d;

    float jmk = AAT_j - AAT_k;
    float jpk = AAT_j + AAT_k;
    float s = jmk * jmk + 4 * AAT_l * AAT_l;
    s = sqrt(s);
    float d1 = sqrt(0.5 * (jpk + s));

    float diff=0.5*(jpk - s);
    float d2;
    if(diff>0.0) {
        d2 = sqrt(diff);
    } else {
        // mathematically diff is always positive or null
        // but due to numerical precision, it can be negative
        d2 = 0.0;
    }

    if (d1 >= d2 && fabs(d2) > aTolerance) {
        return d1 / d2;
    }
    if (d1 < d2  && fabs(d1) > aTolerance) {
        return d2 / d1;
    }

    return gIllConditioned; //failed, almost singular or singular matrix
}

/*
 * Convert the u, v partial surface derivatives to derivatives
 * with respect to the texture coordinates s, t.
 *
 * Given:    dP/d(u,v)   P(u,v) -> [x,y,z]
 *           dT/d(u,v)   T(u,v) -> [s,t]
 *
 * We want:  dP/d(s,t)
 *
 * Notice that we want to take P composed with T:
 *
 *                    P(u,v) = P(T^{-1}(T(u,v)))
 *                           = P(T^{-1}(s,t))
 *
 * What we actually want is the partials if P o T{^-1}
 * Which turns out to be the column vectors of the Jacobian:
 *
 *   Jacobian:   J_{P o T^{-1}}(s,t)
 *
 * Note we also have (from inputs of the function) the following Jacobians:
 *
 *               J_P(u,v) from dP/d(u,v)
 *        and    J_T(u,v) from dT/d(u,v)
 *
 * Using the Chain Rule we get:
 *
 *   J_{P o T^{-1}}(s,t) = J_P(T^{-1}(s,t)) * J_T^{-1}(s,t)
 *
 *                 = J_P(u,v) * J_T^{-1}(s,t)
 *
 * Using the Inverse Function thereom:
 *
 *      J_T^{-1}(s,t) = [J_T(u, v)]^{-1}
 *
 * I.E. The Jacobian of the function F^{-1} is simply the matrix inverse of
 * the Jacobian of the function F.
 *
 * With substitution we get:
 *
 *      J_{P o T^{-1}}(s,t) = J_P(u,v) * [J_T(u, v)]^{-1}
 *
 * Thus to get the partials dP/d(s,t), we just have to invert J_T(u,v)
 * and evaluate the matrix multiplies and take the column vectors of
 * J_{P o T^{-1}}(s,t).
 *
 *
 */
bool computePartialsWithRespect2Texture(const Vec3f &dP_du,
                                        const Vec3f &dP_dv,
                                        const Vec2f &dT_du,
                                        const Vec2f &dT_dv,
                                              Vec3f &dP_ds,
                                              Vec3f &dP_dt)
{

    float detInv;
    float det = dT_du[0] * dT_dv[1] - dT_du[1] * dT_dv[0];

    const float tolerance=1.e-12;

    //use condition number to decide if matrix is ill conditioned
    //note 0 is an estimate of condition number Manny put in
    //1 is a SVD based, accurate calculation
    //we are keeping both versions just to be able to switch between them if there is performance concerns
#if 0
    const float condnum = condNumber2x2(
            dT_du[0], dT_du[1],     // a, b
            dT_dv[0], dT_dv[1]);    // c, d
#else
    const float condnum = tessCondNumber2x2SVD(
            dT_du[0], dT_du[1],     // a, b
            dT_dv[0], dT_dv[1],
            tolerance);    // c, d
#endif

    if (condnum >= gIllConditioned || fabs(det) < tolerance) {
//      if(!scene_rdl2::math::isnormal(det)) {

        dP_ds[0] = dP_du[0];
        dP_ds[1] = dP_du[1];
        dP_ds[2] = dP_du[2];
        dP_dt[0] = dP_dv[0];
        dP_dt[1] = dP_dv[1];
        dP_dt[2] = dP_dv[2];

        return false;
    }

    detInv = 1.0 / det;


    dP_ds[0] = detInv * (dP_du[0] * dT_dv[1] - dP_dv[0] * dT_du[1]);
    dP_ds[1] = detInv * (dP_du[1] * dT_dv[1] - dP_dv[1] * dT_du[1]);
    dP_ds[2] = detInv * (dP_du[2] * dT_dv[1] - dP_dv[2] * dT_du[1]);

    dP_dt[0] = detInv * (-dP_du[0] * dT_dv[0] + dP_dv[0] * dT_du[0]);
    dP_dt[1] = detInv * (-dP_du[1] * dT_dv[0] + dP_dv[1] * dT_du[0]);
    dP_dt[2] = detInv * (-dP_du[2] * dT_dv[0] + dP_dv[2] * dT_du[0]);

    return true;
}

void writeToObj(const std::string& outputPath,
        const std::vector<uint32_t>& faceVertexCount,
        const VertexBuffer<Vec3fa, InterleavedTraits>& vertices,
        const std::vector<uint32_t>& indices,
        const shading::Vector<Vec2f>& textureVertices,
        const std::vector<uint32_t>& textureIndices)
{
    std::ofstream filestream;
    filestream.open(outputPath);
    filestream << "# vertices" << std::endl;
    for (size_t v = 0; v < vertices.size(); ++v) {
        filestream << "v " <<
            vertices(v).x << " " <<
            vertices(v).y << " " <<
            vertices(v).z << std::endl;
    }
    filestream << std::endl;
    filestream << "# texture coordinates" << std::endl;
    for (size_t v = 0; v < textureVertices.size(); ++v) {
        filestream << "vt " <<
            textureVertices[v].x << " " <<
            textureVertices[v].y << std::endl;
    }
    filestream << std::endl;
    filestream << "# face indices" << std::endl;
    size_t offset = 0;
    for (size_t f = 0; f < faceVertexCount.size(); ++f) {
        uint32_t nFv = faceVertexCount[f];
        filestream << "f ";
        for (size_t v = 0; v < nFv; ++v) {
            filestream <<
                (indices[offset + v] + 1) << "/" <<
                (textureIndices[offset + v] + 1) << " ";
        }
        filestream << std::endl;
        offset += nFv;
    }
    filestream << std::endl;
    filestream.close();
}

Vec3f
computePrimitiveMotion(const Vec3f &pos0, const Vec3f *pos1, float rayTime,
    const Instance *instance)
{
    if (!pos1 && !instance) return Vec3f(0.0f); // no motion

    Vec3f p0 = pos0;
    Vec3f p1 = pos1? *pos1 : p0;

    if (instance) {
        const MotionTransform &xform = instance->getLocal2Parent();
        if (xform.isStatic()) {
            if (!pos1) return Vec3f(0.0f); // no motion

            const Mat43 l2p = xform.getStaticXform();
            p0 = transformPoint(l2p, p0);
            p1 = transformPoint(l2p, p1);
        } else {
            const Mat43 l2p0 = xform.eval(rayTime - sHalfDt);
            const Mat43 l2p1 = xform.eval(rayTime + sHalfDt);
            p0 = transformPoint(l2p0, p0);
            p1 = transformPoint(l2p1, p1);
        }
    }

    const Vec3f motion = (p1 - p0) / (2.0f * sHalfDt);
    return motion;
}

Vec3f transformPoint(const scene_rdl2::math::Mat4f mat[2], const scene_rdl2::math::Vec3f& point, const float time, bool isMotionBlurOn)
{
    if (isMotionBlurOn) {
        scene_rdl2::math::Vec3f point0 = scene_rdl2::math::transformPoint(mat[0], point);
        scene_rdl2::math::Vec3f point1 = scene_rdl2::math::transformPoint(mat[1], point);
        return scene_rdl2::math::lerp(point0, point1, time);
    } else {
        return scene_rdl2::math::transformPoint(mat[0], point);
    }
}

Vec3f transformVector(const scene_rdl2::math::Mat4f mat[2], const scene_rdl2::math::Vec3f& vec, const float time, bool isMotionBlurOn)
{
    if (isMotionBlurOn) {
        scene_rdl2::math::Vec3f vec0 = scene_rdl2::math::transformVector(mat[0], vec);
        scene_rdl2::math::Vec3f vec1 = scene_rdl2::math::transformVector(mat[1], vec);
        return scene_rdl2::math::lerp(vec0, vec1, time);
    } else {
        return scene_rdl2::math::transformVector(mat[0], vec);
    }
}

bool attrIsSupported(const void* instancex,                       \
                     shading::AttributeKey key)
{
    if (instancex) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(instancex);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            return instanceAttrs->isSupported(key);
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool hasInstanceAttr(shading::AttributeKey key,
                     const mcrt_common::Ray& ray)
{
    return attrIsSupported(ray.ext.instance3, key) ||
           attrIsSupported(ray.ext.instance2, key) ||
           attrIsSupported(ray.ext.instance1, key) ||
           attrIsSupported(ray.ext.instance0OrLight, key);
}

void overrideInstanceAttrs(const mcrt_common::Ray& ray, shading::Intersection& intersection)
{
    // Instance attributes override.  Start with the lowest instance
    // and overwrite attrs as we go up the chain, as higher level
    // instances have attr priority.
    if (ray.ext.instance3) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance3);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            intersection.setInstanceAttributesOverride(instanceAttrs);
        }
    }
    if (ray.ext.instance2) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance2);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            intersection.setInstanceAttributesOverride(instanceAttrs);
        }
    }
    if (ray.ext.instance1) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance1);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            intersection.setInstanceAttributesOverride(instanceAttrs);
        }
    }
    if (ray.ext.instance0OrLight) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance0OrLight);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            intersection.setInstanceAttributesOverride(instanceAttrs);
        }
    }
}

template <typename T>
T getInstanceAttr(shading::TypedAttributeKey<T> key,
                  const mcrt_common::Ray& ray,
                  T defaultValue)
{
    if (ray.ext.instance3) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance3);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            return instanceAttrs->getAttribute(key);
        }
    }
    if (ray.ext.instance2) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance2);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            return instanceAttrs->getAttribute(key);
        }
    }
    if (ray.ext.instance1) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance1);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            return instanceAttrs->getAttribute(key);
        }
    }
    if (ray.ext.instance0OrLight) {
        const geom::internal::Instance* instance =
            static_cast<const geom::internal::Instance*>(ray.ext.instance0OrLight);
        const shading::InstanceAttributes* instanceAttrs = instance->getAttributes();
        if (instanceAttrs) {
            return instanceAttrs->getAttribute(key);
        }
    }
    return defaultValue;
}

template float getInstanceAttr<float>(shading::TypedAttributeKey<float> key,
                                      const mcrt_common::Ray& ray,
                                      float defaultValue);

template <typename AttributeType> bool
getAttribute(const shading::Attributes* attributes,
             const shading::TypedAttributeKey<AttributeType> key,
             AttributeType& value,
             const int primID,
             const uint32_t vertex)
{
    if (!attributes->isSupported(key)) {
        return false;
    }

    // Data for points is typically stored as constant
    // (one value for all points) or varying (one value
    // per point).
    shading::AttributeRate rate = attributes->getRate(key);
    switch (rate) {
    case shading::RATE_CONSTANT:
        value = attributes->getConstant(key);
        break;
    case shading::RATE_VARYING:
        value = attributes->getVarying(key, primID);
        break;
    case shading::RATE_VERTEX:
        value = attributes->getVertex(key, vertex);
        break;
    default:
        return false;
    }

    return true;
}

template bool getAttribute<scene_rdl2::math::Vec3f>(
             const shading::Attributes* attributes,
             const shading::TypedAttributeKey<scene_rdl2::math::Vec3f> key,
             scene_rdl2::math::Vec3f& value,
             const int primID,
             const uint32_t vertex);


} // end namespace internal
} // end namespace geom
} // end namespace moonray

