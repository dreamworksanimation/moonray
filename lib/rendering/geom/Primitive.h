// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Primitive.h
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/VertexBuffer.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>
#include <moonray/rendering/mcrt_common/Frustum.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

namespace moonray {

namespace geom {

namespace internal {
class Primitive;
class PrimitivePrivateAccess;
}

class PrimitiveVisitor;
class ProceduralContext;
class State;

typedef tbb::concurrent_vector<internal::Primitive*> InternalPrimitiveList;

/// @class Primitive
/// @brief base class for all primitive types.
class Primitive
{
    // expose private method transformPrimitive/getPrimitiveImpl
    // for internal renderer use
    friend class internal::PrimitivePrivateAccess;

public:
    typedef uint32_t size_type;
    typedef uint32_t IndexType;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief enum for specifiying how the Primitive can be modified during
    ///     Procedural::update stage
    enum class Modifiability: uint8_t
    {
        STATIC,    ///<all primitive data can't be modified after creation
        DEFORMABLE,///<transform and geometry vertices may change between frames
        DYNAMIC    ///<topology of the geometry may changes between frames
    };

    /// enum for specifying the primitive data (VertexBuffer, IndexBuffer,
    /// shading::PrimitiveAttribute) validness
    enum class DataValidness
    {
        VALID,
        INVALID_VERTEX_BUFFER,
        INVALID_TOPOLOGY,
        INVALID_CONSTANT_ATTRIBUTE,
        INVALID_UNIFORM_ATTRIBUTE,
        INVALID_VARYING_ATTRIBUTE,
        INVALID_FACEVARYING_ATTRIBUTE,
        INVALID_VERTEX_ATTRIBUTE,
    };

    Primitive();
    virtual ~Primitive();
    Primitive(const Primitive&) = delete;
    Primitive& operator=(const Primitive&) = delete;
    Primitive& operator=(Primitive&&) = delete;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    virtual void accept(PrimitiveVisitor& v);

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    void setUpdated(bool updated = true);

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    bool getUpdated() const;

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief flag primitive's modifiability to STATIC/DYNAMIC/DEFORMABLE,
    ///     do not call this method unless the procedural plugin contains
    /// update() method implementation
    void setModifiability(Modifiability m = Modifiability::STATIC);

    /// @remark For realtime rendering use. Feature film shader development
    ///     does not require this functionality.
    /// @brief query primitive's modifiability (STATIC/DYNAMIC/DEFORMABLE),
    ///     do not call this method unless the procedural plugin contains
    ///     update() method implementation
    Modifiability getModifiability() const;

    /// return the memory usage of this primitive in bytes
    virtual size_type getMemory() const = 0;

    /// return the number of motion samples
    virtual size_type getMotionSamplesCount() const = 0;

protected:



    // Use available positions and velocities to generate Bezier control polygons,
    // replacing vertex buffer with one containing the control points.
    template <class VecType, template <typename, typename> class Traits>
    void generateBezierPolygons(VertexBuffer<VecType, Traits>& vertices,
                                const MotionBlurParams& motionBlurParams,
                                const shading::PrimitiveAttributeTable* primAttrTab) {

        // Get vertex info
        int numVertexSets = vertices.get_time_steps();
        MNRY_ASSERT_REQUIRE(numVertexSets >= 1  &&  numVertexSets <= 2);
        size_t numVertices = vertices.size();

        // Get velocity info
        int numVelocitySets = 0;
        if (primAttrTab && primAttrTab->hasAttribute(shading::StandardAttributes::sVelocity)) {
            numVelocitySets = primAttrTab->getTimeSampleCount(shading::StandardAttributes::sVelocity);
        }
        MNRY_ASSERT_REQUIRE(numVelocitySets <= 2);

        // Get acceleration info
        int numAccelerationSets = 0;
        if (primAttrTab) {
            // "acceleration" for RdlGeometry primitive attributes,
            // "accel" for Alembic primitive attributes
            if (primAttrTab->hasAttribute(shading::StandardAttributes::sAcceleration) ||
                primAttrTab->hasAttribute(shading::TypedAttributeKey<Vec3f>("accel"))) {
                numAccelerationSets = 1;
            }
        }
        MNRY_ASSERT_REQUIRE(numAccelerationSets <= 1);

        // Short-cut: Can we use vertex buffer(s) as-is?
        if (numVelocitySets == 0) {
            // Yes.
            return;
        }

        // Get times corresponding to motion steps
        float t0, t1;
        motionBlurParams.getMotionStepTimes(t0, t1);

        // Set up aux vertex buffer for Bezier control points
        int numBezierControlPoints = numVertexSets + numVelocitySets + numAccelerationSets;
        VertexBuffer<VecType, InterleavedTraits> bezierControlPoints(numVertices, numBezierControlPoints);

        // We make use of parallel_for to loop over vertices
        tbb::blocked_range<size_t> range = tbb::blocked_range<size_t>(0, numVertices);

        if (numVelocitySets == 1) {
            // If there's 1 velocity set, we support either linear or quadratic motion.
            MNRY_ASSERT_REQUIRE(numVertexSets == 1);
            const shading::PrimitiveAttribute<Vec3f>& vel =
                primAttrTab->getAttribute(shading::StandardAttributes::sVelocity, 0);
            if (numAccelerationSets == 1) {
                // Acceleration-based, quadratic motion
                const shading::PrimitiveAttribute<Vec3f>& acc =
                    primAttrTab->hasAttribute(shading::StandardAttributes::sAcceleration)    ?
                    primAttrTab->getAttribute(shading::StandardAttributes::sAcceleration, 0) :
                    primAttrTab->getAttribute(shading::TypedAttributeKey<Vec3f>("accel"));
                tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        VecType p = vertices(i, 0);
                        VecType v = vel[i];
                        VecType a = acc[i];
                        VecType p0 = p + t0 * v + 0.5f * t0 * t0 * a;
                        VecType p1 = p + t1 * v + 0.5f * t1 * t1 * a;
                        VecType v0 = v + t0 * a;
                        bezierControlPoints(i, 0) = p0;
                        bezierControlPoints(i, 1) = p0 + 0.5f * (t1 - t0) * v0;
                        bezierControlPoints(i, 2) = p1;
                    }
                });
            } else {
                // Linear motion
                tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        VecType p = vertices(i, 0);
                        VecType v = vel[i];
                        bezierControlPoints(i, 0) = p + t0 * v;
                        bezierControlPoints(i, 1) = p + t1 * v;
                    }
                });
            }
        } else {  // numVelocitySets == 2
            // The 2 positions and 2 velocities supplied for each vertex defines the Hermite form of a cubic curve.
            // Here we convert to a Bezier representation, using the Hermite data to construct the 4 Bezier control points.
            // Suppose the supplied positions are p0 and p1, and the corresponding velocities v0 and v1, the subscripts
            // denoting the motion step indices.
            // We must determine the 2 intermediate Bezier cps, say pA and pB, so that the Bezier control polygon will
            // consist of the 4 points (p0, pA, pB, p1) in that order.
            // Let's suppose also that the first motion step corresponds to a time t=t0 and the second to a time t=t1.
            // Since, in general, we will not have [t0,t1] = [0,1], we remap t linearly to a parameter u whose value
            // ranges over [0,1], i.e. u=0 when t=t0, and u=1 when t=t1:
            //
            //   u = (t-t0) / (t1-t0)
            //
            // Note that du/dt = 1/(t1-t0) = 1/deltaT.
            // Write down the standard cubic Bezier form for the position p as a funciton of u:
            //
            //   p(u) = (1-u)^3*p0 + 3u(1-u)^2*pA + 3u^2(1-u)*pB + u^3*p1
            //
            // Differentiating with respect to u,
            //
            //   dp/du = -3(1-u)^2*p0 + 3((1-u)^2-2u(1-u))*pA + 3(2u(1-u)-u^2)*pB + 3u^2*p1
            //
            // Using the chain rule gives us dp/dt = dp/du * du/dt = (1/deltaT) * dp/du. Thus we can now write down
            // an expression for the velocity dp/dt of the point p:
            //
            //   dp/dt = (1/deltaT) * { -3(1-u)^2*p0 + 3((1-u)^2-2u(1-u))*pA + 3(2u(1-u)-u^2)*pB + 3u^2*p1 }
            //
            // After simplification, this becomes
            //
            //   dp/dt = (3/deltaT) * { (1-u)^2*(pA-p0) + 2u(1-u)*(pB-pA) + u^2*(p1-pB) }
            //
            // The Hermite form provides the velocities v0 and v1 at u=0 and u=1 respectively.
            // Substituting these values for u and equating to the given velocities, we find
            //
            //   v0 = (3/deltaT) * (pA-p0)
            //   v1 = (3/deltaT) * (p1-pB)
            //
            // Rearranging these expressions, we recover the unknown positions pA and pB:
            //
            //   pA = p0 + (deltaT/3) * v0
            //   pB = p1 - (deltaT/3) * v1

            MNRY_ASSERT_REQUIRE(numVertexSets == 2);
            const shading::PrimitiveAttribute<Vec3f>& vel0 =
                primAttrTab->getAttribute(shading::StandardAttributes::sVelocity, 0);
            const shading::PrimitiveAttribute<Vec3f>& vel1 =
                primAttrTab->getAttribute(shading::StandardAttributes::sVelocity, 1);
            float oneThirdDt = (1.0f / 3.0f) * motionBlurParams.getDt();
            tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    bezierControlPoints(i, 0) = vertices(i, 0);
                    bezierControlPoints(i, 1) = vertices(i, 0) + oneThirdDt * vel0[i];
                    bezierControlPoints(i, 2) = vertices(i, 1) - oneThirdDt * vel1[i];
                    bezierControlPoints(i, 3) = vertices(i, 1);
                }
            });
        }

        vertices = std::move(bezierControlPoints);
    }

    template <template <typename, typename> class Traits>
    void generateBezierPolygons(VertexBuffer<Vec3fa, Traits>& vertices,
                                const MotionBlurParams& motionBlurParams,
                                const shading::PrimitiveAttributeTable* primAttrTab) {

        // Get vertex info
        int numVertexSets = vertices.get_time_steps();
        MNRY_ASSERT_REQUIRE(numVertexSets >= 1  &&  numVertexSets <= 2);
        size_t numVertices = vertices.size();

        // Get velocity info
        int numVelocitySets = 0;
        if (primAttrTab && primAttrTab->hasAttribute(shading::StandardAttributes::sVelocity)) {
            numVelocitySets = primAttrTab->getTimeSampleCount(shading::StandardAttributes::sVelocity);
        }
        MNRY_ASSERT_REQUIRE(numVelocitySets <= 2);

        // Get acceleration info
        int numAccelerationSets = 0;
        if (primAttrTab) {
            if (primAttrTab->hasAttribute(shading::StandardAttributes::sAcceleration) || // accelerations coming from RdlMesh
                primAttrTab->hasAttribute(shading::TypedAttributeKey<Vec3f>("accel"))) { // accelerations coming from Alembic
                numAccelerationSets = 1;
            }
        }
        MNRY_ASSERT_REQUIRE(numAccelerationSets <= 1);

        // Short-cut: Can we use vertex buffer(s) as-is?
        if (numVelocitySets == 0) {
            // Yes.
            return;
        }

        // Get times corresponding to motion steps
        float t0, t1;
        motionBlurParams.getMotionStepTimes(t0, t1);

        // Set up aux vertex buffer for Bezier control points
        int numBezierControlPoints = numVertexSets + numVelocitySets + numAccelerationSets;
        VertexBuffer<Vec3fa, InterleavedTraits> bezierControlPoints(numVertices, numBezierControlPoints);

        // We make use of parallel_for to loop over vertices
        tbb::blocked_range<size_t> range = tbb::blocked_range<size_t>(0, numVertices);

        if (numVelocitySets == 1) {
            // If there's 1 velocity set, we support either linear or quadratic motion.
            MNRY_ASSERT_REQUIRE(numVertexSets == 1);
            const shading::PrimitiveAttribute<Vec3f>& vel = primAttrTab->getAttribute(shading::StandardAttributes::sVelocity, 0);
            if (numAccelerationSets == 1) {
                // Acceleration-based, quadratic motion
                const shading::PrimitiveAttribute<Vec3f>& acc =
                    primAttrTab->hasAttribute(shading::StandardAttributes::sAcceleration)    ?
                    primAttrTab->getAttribute(shading::StandardAttributes::sAcceleration, 0) :
                    primAttrTab->getAttribute(shading::TypedAttributeKey<Vec3f>("accel"));
                tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        Vec3fa p = vertices(i, 0);
                        Vec3fa v = Vec3fa(vel[i], 0.f);
                        Vec3fa a = Vec3fa(acc[i], 0.f);
                        Vec3fa p0 = p + t0 * v + 0.5f * t0 * t0 * a;
                        Vec3fa p1 = p + t1 * v + 0.5f * t1 * t1 * a;
                        Vec3fa v0 = v + t0 * a;
                        bezierControlPoints(i, 0) = p0;
                        bezierControlPoints(i, 1) = p0 + 0.5f * (t1 - t0) * v0;
                        bezierControlPoints(i, 2) = p1;
                    }
                });
            } else {
                // Linear motion
                tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        Vec3fa p = vertices(i, 0);
                        Vec3fa v = Vec3fa(vel[i], 0.f);
                        bezierControlPoints(i, 0) = p + t0 * v;
                        bezierControlPoints(i, 1) = p + t1 * v;
                    }
                });
            }
        } else {  // numVelocitySets == 2
            // The 2 positions and 2 velocities supplied for each vertex defines the Hermite form of a cubic curve.
            // Here we convert to a Bezier representation, using the Hermite data to construct the 4 Bezier control points.
            // Suppose the supplied positions are p0 and p1, and the corresponding velocities v0 and v1, the subscripts
            // denoting the motion step indices.
            // We must determine the 2 intermediate Bezier cps, say pA and pB, so that the Bezier control polygon will
            // consist of the 4 points (p0, pA, pB, p1) in that order.
            // Let's suppose also that the first motion step corresponds to a time t=t0 and the second to a time t=t1.
            // Since, in general, we will not have [t0,t1] = [0,1], we remap t linearly to a parameter u whose value
            // ranges over [0,1], i.e. u=0 when t=t0, and u=1 when t=t1:
            //
            //   u = (t-t0) / (t1-t0)
            // 
            // Note that du/dt = 1/(t1-t0) = 1/deltaT.
            // Write down the standard cubic Bezier form for the position p as a funciton of u:
            // 
            //   p(u) = (1-u)^3*p0 + 3u(1-u)^2*pA + 3u^2(1-u)*pB + u^3*p1
            //
            // Differentiating with respect to u,
            //
            //   dp/du = -3(1-u)^2*p0 + 3((1-u)^2-2u(1-u))*pA + 3(2u(1-u)-u^2)*pB + 3u^2*p1
            //
            // Using the chain rule gives us dp/dt = dp/du * du/dt = (1/deltaT) * dp/du. Thus we can now write down
            // an expression for the velocity dp/dt of the point p:
            //
            //   dp/dt = (1/deltaT) * { -3(1-u)^2*p0 + 3((1-u)^2-2u(1-u))*pA + 3(2u(1-u)-u^2)*pB + 3u^2*p1 }
            //
            // After simplification, this becomes
            //
            //   dp/dt = (3/deltaT) * { (1-u)^2*(pA-p0) + 2u(1-u)*(pB-pA) + u^2*(p1-pB) }
            //
            // The Hermite form provides the velocities v0 and v1 at u=0 and u=1 respectively.
            // Substituting these values for u and equating to the given velocities, we find
            //
            //   v0 = (3/deltaT) * (pA-p0)
            //   v1 = (3/deltaT) * (p1-pB)
            //
            // Rearranging these expressions, we recover the unknown positions pA and pB:
            //
            //   pA = p0 + (deltaT/3) * v0
            //   pB = p1 - (deltaT/3) * v1

            MNRY_ASSERT_REQUIRE(numVertexSets == 2);
            const shading::PrimitiveAttribute<Vec3f>& vel0 = primAttrTab->getAttribute(shading::StandardAttributes::sVelocity, 0);
            const shading::PrimitiveAttribute<Vec3f>& vel1 = primAttrTab->getAttribute(shading::StandardAttributes::sVelocity, 1);
            float oneThirdDt = (1.0f / 3.0f) * motionBlurParams.getDt();
            tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    bezierControlPoints(i, 0) = vertices(i, 0);
                    bezierControlPoints(i, 1) = vertices(i, 0) + oneThirdDt * Vec3fa(vel0[i], 0.f);
                    bezierControlPoints(i, 2) = vertices(i, 1) - oneThirdDt * Vec3fa(vel1[i], 0.f);
                    bezierControlPoints(i, 3) = vertices(i, 1);
                }
            });
        }

        vertices = std::move(bezierControlPoints);
    }

    // Use Bezier control points to generate sufficient time steps;
    // replace vertex buffer with one containing the interpolated positions.
    template <class VecType, template <typename, typename> class Traits>
    void generateMotionSamples(VertexBuffer<VecType, Traits>& vertices, const MotionBlurParams& motionBlurParams,
                               size_t numTimeSteps) {

        size_t numVertices = vertices.size();
        size_t numControlPoints = vertices.get_time_steps();
        MNRY_ASSERT_REQUIRE(numControlPoints >= 1  &&  numControlPoints <= 4);
        float du = 1.0f / (float)(numTimeSteps - 1);
        float ds0, ds1;
        motionBlurParams.getMotionBlurDelta(ds0, ds1);
        VertexBuffer<VecType, InterleavedTraits> tempVertices;

        // We make use of parallel_for to loop over vertices
        tbb::blocked_range<size_t> range = tbb::blocked_range<size_t>(0, numVertices);

        switch (numControlPoints) {

        case 1: 
            // Use single vertex
            break;

        case 2:
            // Keep 2 vertices but adjust their positions to shutter duration
            tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    VecType p0 = vertices(i, 0);
                    VecType p1 = vertices(i, 1);
                    vertices(i, 0) = scene_rdl2::math::lerp(p0, p1, ds0);
                    vertices(i, 1) = scene_rdl2::math::lerp(p0, p1, ds1);
                }
            });
            break;

        case 3:
            // Generate time steps from quadratic Bezier spline
            tempVertices = VertexBuffer<VecType, InterleavedTraits> (numVertices, numTimeSteps);
            for (size_t j = 0; j < numTimeSteps; j++) {
                float u = (float)j * du;
                float t = scene_rdl2::math::lerp(ds0, ds1, u);

                // Evaluate quadratic Bezier basis funcs at t
                float s = 1.0f - t;
                float c0 = s * s;
                float c1 = 2.0f * s * t;
                float c2 = t * t;

                tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        tempVertices(i, j) = c0 * vertices(i, 0) +
                                             c1 * vertices(i, 1) +
                                             c2 * vertices(i, 2);
                    }
                });
            }
            vertices = std::move(tempVertices);
            break;

        case 4:
            // Generate time steps from cubic Bezier spline
            tempVertices = VertexBuffer<VecType, InterleavedTraits> (numVertices, numTimeSteps);
            for (size_t j = 0; j < numTimeSteps; j++) {
                float u = (float)j * du;
                float t = scene_rdl2::math::lerp(ds0, ds1, u);

                // Evaluate cubic Bezier basis funcs at t
                float s = 1.0f - t;
                float t2 = t * t;
                float s2 = s * s;
                float c0 = s * s2;
                float c1 = 3.0f * t * s2;
                float c2 = 3.0f * s * t2;
                float c3 = t * t2;

                tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        tempVertices(i, j) = c0 * vertices(i, 0) +
                                             c1 * vertices(i, 1) +
                                             c2 * vertices(i, 2) +
                                             c3 * vertices(i, 3);
                    }
                });
            }
            vertices = std::move(tempVertices);
            break;

        default:
            break;
        }
    }


    // Special version for Vec3fa, to preserve & scale curve radius
    template <template <typename, typename> class Traits>
    void applyTransformationToSingleTimeStep(VertexBuffer<Vec3fa, Traits>& vertices,
                                             const scene_rdl2::math::Xform3f &p2r, int timeStep) {

        // compute the curve radius scaling that is applied to the w component
        float radiusScale = (length(p2r.l.vx) + length(p2r.l.vy) + length(p2r.l.vz)) / 3.f;

        // Loop over vertices
        tbb::blocked_range<size_t> range = tbb::blocked_range<size_t>(0, vertices.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                Vec3fa p = vertices(i, timeStep);
                Vec3fa q = Vec3fa(transformPoint(p2r, p), 0.f);
                q.w = p.w * radiusScale;    // scale curve radius
                vertices(i, timeStep) = q;
            }
        });
    }

    // Generic version for other vector types
    template <class VecType, template <typename, typename> class Traits>
    void applyTransformationToSingleTimeStep(VertexBuffer<VecType, Traits>& vertices,
                                             const scene_rdl2::math::Xform3f &p2r, int timeStep) {
        // Loop over vertices
        tbb::blocked_range<size_t> range = tbb::blocked_range<size_t>(0, vertices.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                VecType p = vertices(i, timeStep);
                vertices(i, timeStep) = transformPoint(p2r, p);
            }
        });
    }


    // Apply the prim2render transformation to all positions;
    // if 2 transformations are supplied, lerp between them over the shutter duration.
    template <class VecType, template <typename, typename> class Traits>
    void applyTransformation(VertexBuffer<VecType, Traits>& vertices, const shading::XformSamples& prim2render,
                             const MotionBlurParams& motionBlurParams) {
 
        // Get shutter interval
        float ds0, ds1;
        motionBlurParams.getMotionBlurDelta(ds0, ds1);

        // We support up to 2 transformations in prim2render
        int numXforms = prim2render.size();
        MNRY_ASSERT_REQUIRE(numXforms <= 2);

        // Pad vertex buffer if 2 xforms but only 1 time step
        size_t numTimeSteps = vertices.get_time_steps();
        if (numXforms == 2  &&  numTimeSteps == 1) {
            size_t numVertices = vertices.size();
            VertexBuffer<VecType, InterleavedTraits> tempVertices(numVertices, 2);
            tbb::blocked_range<size_t> range = tbb::blocked_range<size_t>(0, vertices.size());
            tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    tempVertices(i, 0) = vertices(i);
                    tempVertices(i, 1) = vertices(i);
                }
            });
            vertices = std::move(tempVertices);
            numTimeSteps = 2;
        }

        scene_rdl2::math::Xform3f p2r(prim2render[0]);

        // Loop over time steps
        float du = numTimeSteps > 1 ? 1.0f / (float)(numTimeSteps - 1) : 0;
        for (size_t j = 0; j < numTimeSteps; j++) {
            float u = (float)j * du;
            float t = scene_rdl2::math::lerp(ds0, ds1, u);

            // Interpolate transformation matrices if 2 were supplied
            if (numXforms == 2) {
                p2r = scene_rdl2::math::Xform3f(scene_rdl2::math::lerp(prim2render[0], prim2render[1], t));
            }
 
            applyTransformationToSingleTimeStep(vertices, p2r, j);
        }
    }


    template <class VecType, template <typename, typename> class Traits>
    void transformVertexBuffer(VertexBuffer<VecType, Traits>& vertices,
                               const shading::XformSamples& prim2render,
                               const MotionBlurParams& motionBlurParams,
                               scene_rdl2::rdl2::MotionBlurType motionBlurType,
                               int   curvedMotionBlurSampleCount,
                               const shading::PrimitiveAttributeTable* primAttrTab) {

        if (vertices.size() == 0) {
            return;
        }

        // TO-DO:
        // This is a bad hack put in as a quick fix for MOONRAY-3275.
        // The issue is that curved motion blur was implemented under the assumption that transformVertexBuffer()
        // would only ever be called once, which is not the case when instancing is used together with curved
        // motion blur. In this case it tries to do a double expansion on the vertex buffer, ending up with
        // an out-of-range number of time steps, or a dataset that no longer represents any valid motion blur type.
        // The longer-term plan is to prevent the double application of transformVertexBuffer in the first place.
        {
            int numVertexSets = vertices.get_time_steps();
            int numVelocitySets = 0;
            if (primAttrTab && primAttrTab->hasAttribute(shading::StandardAttributes::sVelocity)) {
                numVelocitySets = primAttrTab->getTimeSampleCount(shading::StandardAttributes::sVelocity);
            }
            if (numVertexSets > 2 || (numVertexSets == 2 && numVelocitySets == 1)) {
                // numVertexSets > 2 catches the case where curved motion blur has been applied
                // The other case, (numVertexSets == 2 && numVelocitySets == 1), catches the case where
                // velocity motion blur has been applied, expanding the data from 1 pos + 1 vel to
                // 2 pos + 1 vel, which is an illegal input state.
                return;
            }
        }

        // For multi-segment motion blur, Embree supports from 2 to 129 vertex buffers
        if (motionBlurParams.isMotionBlurOn()) {
            curvedMotionBlurSampleCount = scene_rdl2::math::clamp(curvedMotionBlurSampleCount, 2, 129);
        } else {
            curvedMotionBlurSampleCount = 1;
        }

        // If more than 2 output samples are requested for a linear blur type, clamp to 2
        if (curvedMotionBlurSampleCount > 2  &&  (motionBlurType == scene_rdl2::rdl2::MotionBlurType::VELOCITY ||
                                                  motionBlurType == scene_rdl2::rdl2::MotionBlurType::FRAME_DELTA)) {
            curvedMotionBlurSampleCount = 2;
        }

        generateBezierPolygons(vertices, motionBlurParams, primAttrTab);
        generateMotionSamples(vertices, motionBlurParams, curvedMotionBlurSampleCount);
        applyTransformation(vertices, prim2render, motionBlurParams);
    }


private:
    /// @remark For renderer internal use, procedural should never call this
    virtual internal::Primitive* getPrimitiveImpl() = 0;

    /// @brief transform local space primitive to render space
    /// @remark For renderer internal use, procedural should never call this
    ///
    /// transform the Primitive to rendering space and linear interpolate
    /// primitive data based on motion steps/shutter info.
    /// The motion steps and shutter open/close time are in
    /// frame relative coordinate
    /// (-1 stands for previous frame, and 0 stands for current frame)
    ///
    /// @param proceduralContext per procedural/layer context that is passed in
    ///     from Procedural::generate()/Procedural::update() method
    /// @param prim2render primitive space to rendering space transform
    virtual void transformPrimitive(
            const MotionBlurParams &motionBlurParams,
            const shading::XformSamples& prim2render) = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};


} // namespace geom
} // namespace moonray


