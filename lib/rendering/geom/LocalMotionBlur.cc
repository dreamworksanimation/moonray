// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LocalMotionBlur.h"

#include <iomanip>

using namespace moonray::shading;
using namespace moonray::geom;
using namespace scene_rdl2::rdl2;
using namespace scene_rdl2::math;

namespace moonray {
namespace local_motion_blur {

LocalMotionBlur::LocalMotionBlur(const GenerateContext& generateContext,
                                 const std::vector<XformSamples>& regionXforms,
                                 const PrimitiveAttributeTable& pointsAttributes,
                                 const bool useLocalCameraMotionBlur,
                                 const float strengthMult,
                                 const float radiusMult) :
    mRdlGeometry(generateContext.getRdlGeometry()),
    mFps(24.0f),
    mUseLocalCameraMotionBlur(useLocalCameraMotionBlur)
{
    // Get this geometry's node_xform unblended values
    const Mat4d m0 = mRdlGeometry->get(Node::sNodeXformKey, TIMESTEP_BEGIN);
    const Mat4d m1 = mRdlGeometry->get(Node::sNodeXformKey, TIMESTEP_END);
    const Xform3f x0(m0.vx.x, m0.vx.y, m0.vx.z,
                     m0.vy.x, m0.vy.y, m0.vy.z,
                     m0.vz.x, m0.vz.y, m0.vz.z,
                     m0.vw.x, m0.vw.y, m0.vw.z);
    const Xform3f x1(m1.vx.x, m1.vx.y, m1.vx.z,
                     m1.vy.x, m1.vy.y, m1.vy.z,
                     m1.vz.x, m1.vz.y, m1.vz.z,
                     m1.vw.x, m1.vw.y, m1.vw.z);
    mNodeXform = { x0, x1 };

    const SceneVariables& sv = mRdlGeometry->getSceneClass().getSceneContext()->getSceneVariables();
    mFps = sv.get(SceneVariables::sFpsKey);

    if (useLocalCameraMotionBlur) {
        // Get the inverse of the scene camera's node_xform
        const SceneObject* cameraObject = sv.getCamera();
        const Camera* camera = (cameraObject) ?  cameraObject->asA<Camera>() : nullptr;
        mCameraXform = { Xform3f(scene_rdl2::math::one) };
        if (camera) {
            const Mat4d m0 = camera->get(Node::sNodeXformKey, TIMESTEP_BEGIN).inverse();
            const Mat4d m1 = camera->get(Node::sNodeXformKey, TIMESTEP_END).inverse();
            const Xform3f x0(m0.vx.x, m0.vx.y, m0.vx.z,
                             m0.vy.x, m0.vy.y, m0.vy.z,
                             m0.vz.x, m0.vz.y, m0.vz.z,
                             m0.vw.x, m0.vw.y, m0.vw.z);
            const Xform3f x1(m1.vx.x, m1.vx.y, m1.vx.z,
                             m1.vy.x, m1.vy.y, m1.vy.z,
                             m1.vz.x, m1.vz.y, m1.vz.z,
                             m1.vw.x, m1.vw.y, m1.vw.z);
            mCameraXform = {x0, x1};
        }
    }

    for (size_t i = 0; i < regionXforms.size(); ++i) {
        float radius = 1.0f;
        float inner_radius = 1.0f;
        float multiplier = 1.0f;
        for (const auto& kv : pointsAttributes) {
            moonray::shading::AttributeKey key = kv.first;
            const PrimitiveAttributeBase& attr = *kv.second[0];
            if (key.getType() == scene_rdl2::rdl2::TYPE_FLOAT &&
                (attr.getRate() == RATE_VERTEX || attr.getRate() == RATE_VARYING)) {

                if (strcmp(key.getName(), "radius") == 0) {
                    radius = attr.as<float>()[i];
                } else if (strcmp(key.getName(), "inner_radius") == 0) {
                    inner_radius = attr.as<float>()[i];
                } else if (strcmp(key.getName(), "multiplier") == 0) {
                    multiplier = attr.as<float>()[i];
                }
            }
        }

        mRegions.emplace_back(regionXforms[i][0],
                              radius * max(0.0f, radiusMult),
                              inner_radius * max(0.0f, radiusMult),
                              1.0f + (multiplier - 1.0f) * clamp(strengthMult));
    }
}

float
LocalMotionBlur::getMultiplier(const Vec3f& P,
                               const XformSamples& parent2root) const
{
    // If there are 2 motion steps use the second one
    float mbMult = 1.0f;
    for (size_t r = 0; r < mRegions.size(); ++r) {
        const Vec3f localP = transformPoint(mRegions[r].mXform.inverse(), P);
        const float distSqr = lengthSqr(localP);
        const float r2 = mRegions[r].mRadius * mRegions[r].mRadius;
        float ir2 = mRegions[r].mInnerRadius * mRegions[r].mInnerRadius;
        if (ir2 > r2) ir2 = r2 - sEpsilon;
        const float t = 1.0f - clamp((distSqr - ir2) / (r2 - ir2));
        mbMult = min(mbMult, lerp(1.0f,  mRegions[r].mMultiplier, t));
    }
    return mbMult;
}

template <> void
LocalMotionBlur::apply(const MotionBlurType mbType,
                       const XformSamples& parent2root,
                       VertexBuffer<Vec3fa, InterleavedTraits>& vertices,
                       PrimitiveAttributeTable& primitiveAttributeTable) const
{
    const size_t numTimeSteps = vertices.get_time_steps();
    VertexBuffer<Vec3f, InterleavedTraits> vertices3f(vertices.size(), numTimeSteps);
    std::vector<float> widths0(vertices.size(), 0.f);
    std::vector<float> widths1(vertices.size(), 0.f);
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices3f(i, 0) = Vec3f(vertices(i, 0).x, vertices(i, 0).y, vertices(i, 0).z);
        widths0[i] = vertices(i, 0).w;
        if (numTimeSteps > 1) {
            vertices3f(i, 1) = Vec3f(vertices(i, 1).x, vertices(i, 1).y, vertices(i, 1).z);
            widths1[i] = vertices(i, 1).w;
        }
    }

    apply(mbType,
          parent2root,
          vertices3f,
          primitiveAttributeTable);

    // compute the curve radius scaling that is applied to the w component
    XformSamples local2world = concatenate(parent2root, mNodeXform);
    float radiusScale0 = (length(local2world[0].l.vx) + length(local2world[0].l.vy) + length(local2world[0].l.vz)) / 3.f;
    float radiusScale1 = (length(local2world[1].l.vx) + length(local2world[1].l.vy) + length(local2world[1].l.vz)) / 3.f;

    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices(i, 0) = Vec3fa(vertices3f(i, 0).x,
                                vertices3f(i, 0).y,
                                vertices3f(i, 0).z,
                                vertices(i, 0).w * radiusScale0);
        if (numTimeSteps > 1) {
            vertices(i, 1) = Vec3fa(vertices3f(i, 1).x,
                                    vertices3f(i, 1).y,
                                    vertices3f(i, 1).z,
                                    vertices(i, 1).w * radiusScale1);
        }

    }
}

template <typename AttributeType> void
LocalMotionBlur::apply(const MotionBlurType mbType,
                       const XformSamples& parent2root,
                       VertexBuffer<AttributeType, InterleavedTraits>& vertices,
                       PrimitiveAttributeTable& primitiveAttributeTable) const
{
    std::vector<float> localMbMask(vertices.size());

    // Vertices are transformed into world space using the object's
    // node_xform parameter and the local transforms in parent2root.
    XformSamples local2world = concatenate(parent2root, mNodeXform);

    switch (mbType) {

    case MotionBlurType::STATIC_DUPLICATE:
    case MotionBlurType::FRAME_DELTA:
    {
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices(i, 0) = transformPoint(local2world[0], vertices(i, 0));
            vertices(i, 1) = transformPoint(local2world[1], vertices(i, 1));

            const Vec3f& p0 = vertices(i, 0);
            const Vec3f& p1 = vertices(i, 1);
            const float mbMult = getMultiplier(p1, parent2root);
            localMbMask[i] = 1.0f - mbMult;
            if (!isOne(mbMult)) {
                // Account for motion in the camera's node_xform.
                Vec3f cameraXformDiff(0.f);
                if (mUseLocalCameraMotionBlur) {
                    if (mCameraXform.size() > 1) {
                        cameraXformDiff = transformPoint(mCameraXform[0].inverse(), p1) -
                                          transformPoint(mCameraXform[1].inverse(), p1);
                    }
                }

                // Translate the first motion step position towards the second
                // motion step position(p1) then add the xform difference.
                vertices(i, 0) = p1 + (p0 - p1) * mbMult  +
                                 cameraXformDiff * (1.0f - mbMult);

            }
        }
        break;
    }

    case MotionBlurType::VELOCITY:
    {
        if (!primitiveAttributeTable.hasAttribute(StandardAttributes::sVelocity)) {
            mRdlGeometry->error("Missing velocity attribute for local motion blur");
            return;
        }
        PrimitiveAttribute<Vec3f>& velocityAttr =
            primitiveAttributeTable.getAttribute(StandardAttributes::sVelocity);
        if (velocityAttr.getRate() != AttributeRate::RATE_VERTEX &&
            velocityAttr.getRate() != AttributeRate::RATE_VARYING) {
            mRdlGeometry->error("Velocity must be defined at vertex or varying rate for local motion blur");
            return;
        }

        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices(i, 0) = transformPoint(local2world[0], vertices(i, 0));

            const float mbMult = getMultiplier(vertices(i, 0), parent2root);
            localMbMask[i] = 1.0f - mbMult;
            if (!isOne(mbMult)) {

                const Vec3f& p0 = vertices(i, 0);

                // Account for motion in the camera's node_xform.
                Vec3f cameraXformDiff(0.f);
                if (mCameraXform.size() > 1) {
                    cameraXformDiff = transformPoint(mCameraXform[1], p0) -
                                      transformPoint(mCameraXform[0], p0);
                }

                velocityAttr[i] = velocityAttr[i] * mbMult -
                                  cameraXformDiff * (1.0f - mbMult) * mFps;
            }
        }

        break;
    }

    case MotionBlurType::ACCELERATION:
    {
        if (!primitiveAttributeTable.hasAttribute(StandardAttributes::sVelocity)) {
            mRdlGeometry->error("Missing velocity attribute for local motion blur");
            return;
        }
        PrimitiveAttribute<Vec3f>& velocityAttr =
            primitiveAttributeTable.getAttribute(StandardAttributes::sVelocity);
        if (velocityAttr.getRate() != AttributeRate::RATE_VERTEX &&
            velocityAttr.getRate() != AttributeRate::RATE_VARYING) {
            mRdlGeometry->error("Velocity must be defined at vertex or varying rate for local motion blur");
            return;
        }

        if (!primitiveAttributeTable.hasAttribute(TypedAttributeKey<Vec3f>("accel"))) {
            mRdlGeometry->error("Missing 'accel' acceleration attribute for local motion blur");
            return;
        }
        PrimitiveAttribute<Vec3f>& accelAttr =
            primitiveAttributeTable.getAttribute(TypedAttributeKey<Vec3f>("accel"));
        if (accelAttr.getRate() != AttributeRate::RATE_VERTEX &&
            accelAttr.getRate() != AttributeRate::RATE_VARYING) {
            mRdlGeometry->error("Acceleration must be defined at vertex or varying rate for local motion blur");
            return;
        }

        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices(i, 0) = transformPoint(local2world[0], vertices(i, 0));

            const float mbMult = getMultiplier(vertices(i, 0), parent2root);
            localMbMask[i] = 1.0f - mbMult;
            if (!isOne(mbMult)) {

                const Vec3f& p0 = vertices(i, 0);

                // Account for motion in the camera's node_xform.
                Vec3f cameraXformDiff(0.f);
                if (mCameraXform.size() > 1) {
                    cameraXformDiff = transformPoint(mCameraXform[1], p0) -
                                      transformPoint(mCameraXform[0], p0);
                }

                velocityAttr[i] = velocityAttr[i] * mbMult -
                                  cameraXformDiff * (1.0f - mbMult) * mFps;

                accelAttr[i] = accelAttr[i] * mbMult -
                               cameraXformDiff * (1.0f - mbMult) * mFps;
            }
        }
        break;
    }

    case scene_rdl2::rdl2::MotionBlurType::HERMITE:
    {
        if (!primitiveAttributeTable.hasAttribute(StandardAttributes::sVelocity)) {
            mRdlGeometry->error("Missing velocity attribute for local motion blur");
            return;
        }
        PrimitiveAttribute<Vec3f>& velocity0Attr =
            primitiveAttributeTable.getAttribute(StandardAttributes::sVelocity, 0);
        if (velocity0Attr.getRate() != AttributeRate::RATE_VERTEX &&
            velocity0Attr.getRate() != AttributeRate::RATE_VARYING) {
            mRdlGeometry->error("Velocity must be defined at vertex or varying rate for local motion blur");
            return;
        }
        PrimitiveAttribute<Vec3f>& velocity1Attr =
            primitiveAttributeTable.getAttribute(StandardAttributes::sVelocity, 1);

        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices(i, 0) = transformPoint(local2world[0], vertices(i, 0));
            vertices(i, 1) = transformPoint(local2world[1], vertices(i, 1));

            const Vec3f& p1 = vertices(i, 1);
            const float mbMult = getMultiplier(p1, parent2root);
            localMbMask[i] = 1.0f - mbMult;
            if (!isOne(mbMult)) {
                const Vec3f& p0 = vertices(i, 0);

                // Account for motion in the camera's node_xform.
                Vec3f cameraXformDiff(0.f);
                if (mCameraXform.size() > 1) {
                    cameraXformDiff = transformPoint(mCameraXform[1], p0) -
                                      transformPoint(mCameraXform[0], p0);
                }

                velocity0Attr[i] = velocity0Attr[i] * mbMult -
                                   cameraXformDiff * (1.0f - mbMult) * mFps;

                velocity1Attr[i] = velocity1Attr[i] * mbMult -
                                   cameraXformDiff * (1.0f - mbMult) * mFps;

                // Translate the first motion step position towards the second
                // motion step position(p1) then add the xform difference.
                vertices(i, 0) = p1 + (p0 - p1) * mbMult  +
                                 cameraXformDiff * (1.0f - mbMult);
            }
        }
        break;
    }

    }

    primitiveAttributeTable.addAttribute(TypedAttributeKey<float>("local_motion_blur"),
                                         RATE_VERTEX,
                                         std::move(localMbMask));
}

template void LocalMotionBlur::apply(const MotionBlurType mbType,
                                     const XformSamples& parent2root,
                                     VertexBuffer<Vec3f, InterleavedTraits>& vertices,
                                     PrimitiveAttributeTable& primitiveAttributeTable) const;

template void LocalMotionBlur::apply(const MotionBlurType mbType,
                                     const XformSamples& parent2root,
                                     VertexBuffer<Vec3fa, InterleavedTraits>& vertices,
                                     PrimitiveAttributeTable& primitiveAttributeTable) const;

} // end local_motion_blur
} // end namespace moonray
