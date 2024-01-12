// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BakeCamera.cc

#include <moonray/rendering/geom/prim/Mesh.h>

#include "BakeCamera.h"

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>
#include <moonray/rendering/geom/TransformedPrimitive.h>
#include <moonray/rendering/geom/prim/Primitive.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/render/util/Memory.h>

// DEBUG
// #include <OpenImageIO/imageio.h>
// #include <OpenImageIO/imagebuf.h>
// #include <OpenImageIO/imagebufalgo.h>

#include <string.h>

namespace scene_rdl2 {

using scene_rdl2::math::Vec2f;
using scene_rdl2::math::Vec3fa;
using scene_rdl2::math::Vec3f;
using scene_rdl2::math::Vec3d;
using scene_rdl2::math::Mat4d;
using scene_rdl2::math::Mat4f;
using scene_rdl2::math::Mat3d;
}

namespace moonray {

namespace {

class PosMapBuilder: public geom::PrimitiveVisitor
{
public:
    PosMapBuilder(unsigned int udim,
                  unsigned int width,
                  unsigned int height,
                  const std::string &stKey,
                  scene_rdl2::math::Vec3fa *posResult,
                  scene_rdl2::math::Vec3f *nrmResult);

    void visitPrimitive(geom::Primitive& p) override { MNRY_ASSERT(0, "visitPrimitive NYI"); }
    void visitCurves(geom::Curves& c) override { MNRY_ASSERT(0, "visitCurves NYI"); }
    void visitInstance(geom::Instance& i) override { MNRY_ASSERT(0, "visitInstance NYI"); }
    void visitPoints(geom::Points& p) override { MNRY_ASSERT(0, "visitPoints NYI"); }
    void visitPolygonMesh(geom::PolygonMesh &p) override;
    void visitPrimitiveGroup(geom::PrimitiveGroup &pg) override;
    void visitSphere(geom::Sphere& s) override { MNRY_ASSERT(0, "visitSphere NYI"); }
    void visitBox(geom::Box& b) override { MNRY_ASSERT(0, "visitBox NYI"); }
    void visitSubdivisionMesh(geom::SubdivisionMesh &s) override;
    void visitTransformedPrimitive(geom::TransformedPrimitive& t) override;
    void visitVdbVolume(geom::VdbVolume& v) override { MNRY_ASSERT(0, "visitVdbVolume NYI"); }

private:
    unsigned int mWidth;
    unsigned int mHeight;
    unsigned int mUdim;
    shading::TypedAttributeKey<scene_rdl2::math::Vec2f> mStKey;
    scene_rdl2::math::Vec3fa *mPosResult;
    scene_rdl2::math::Vec3f *mNrmResult;
};

PosMapBuilder::PosMapBuilder(unsigned int width,
                             unsigned int height,
                             unsigned int udim,
                             const std::string &stKey,
                             scene_rdl2::math::Vec3fa *posResult,
                             scene_rdl2::math::Vec3f *nrmResult):
    mWidth(width),
    mHeight(height),
    mUdim(udim),
    mStKey(stKey.empty()?
           shading::StandardAttributes::sSurfaceST :
           shading::TypedAttributeKey<scene_rdl2::math::Vec2f>(stKey)),
    mPosResult(posResult),
    mNrmResult(nrmResult)
{
    MNRY_ASSERT(posResult);
    memset(posResult, 0, width * height * sizeof(scene_rdl2::math::Vec3fa));
    if (nrmResult) {
        memset(nrmResult, 0, width * height * sizeof(scene_rdl2::math::Vec3f));
    }
}

void
PosMapBuilder::visitTransformedPrimitive(geom::TransformedPrimitive &t)
{
    t.getPrimitive()->accept(*this);
}

void
PosMapBuilder::visitPolygonMesh(geom::PolygonMesh &p)
{
    geom::internal::Primitive* pImpl =
        geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
    MNRY_ASSERT(pImpl->getType() == geom::internal::Primitive::POLYMESH);
    auto pMesh = static_cast<geom::internal::Mesh*>(pImpl);
    pMesh->bakePosMap(mWidth, mHeight, mUdim, mStKey, mPosResult, mNrmResult);
}

void
PosMapBuilder::visitPrimitiveGroup(geom::PrimitiveGroup &pg)
{
    pg.forEachPrimitive(*this, /* parallel = */ false);
}

void
PosMapBuilder::visitSubdivisionMesh(geom::SubdivisionMesh &s)
{
    geom::internal::Primitive *pImpl =
        geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s);
    auto primitiveType = pImpl->getType();
    if (primitiveType == geom::internal::Primitive::POLYMESH) {
        auto pMesh = static_cast<geom::internal::Mesh*>(pImpl);
        pMesh->bakePosMap(mWidth, mHeight, mUdim, mStKey, mPosResult, mNrmResult);
    }
    // no support from embree subd mesh
}

} // namespace anonymous


namespace pbr {

BakeCamera::BakeCamera(const scene_rdl2::rdl2::Camera *rdlCamera):
    Camera(rdlCamera),
    mWidth(0),
    mHeight(0),
    mPosMap(nullptr),
    mNrmMap(nullptr),
    mPosMapWidth(0),
    mPosMapHeight(0)
{
    mAttrGeometry = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::SceneObject *>("geometry");
    mAttrUdim = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::Int>("udim");
    mAttrUvAttribute = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::String>("uv_attribute");
    mAttrMode = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::Int>("mode");
    mAttrBias = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::Float>("bias");
    mAttrUseRelativeBias = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::Bool>("use_relative_bias");
    mAttrMapFactor = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::Float>("map_factor");
    mAttrNormalMap = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::String>("normal_map");
    mAttrNormalMapSpace = rdlCamera->getSceneClass().getAttributeKey<scene_rdl2::rdl2::Int>("normal_map_space");
}

BakeCamera::~BakeCamera()
{
    if (mPosMap) {
        scene_rdl2::util::alignedFreeArray(mPosMap);
        mPosMap = nullptr;
    }

    if (mNrmMap) {
        scene_rdl2::util::alignedFreeArray(mNrmMap);
        mNrmMap = nullptr;
    }
}

bool
BakeCamera::getIsDofEnabledImpl() const
{
    return false;
}

void
BakeCamera::bakeUvMapsImpl()
{
    const scene_rdl2::rdl2::Camera *rdlCamera = getRdlCamera();

    scene_rdl2::rdl2::SceneObject *so = rdlCamera->get(mAttrGeometry);
    if (so && so->isA<scene_rdl2::rdl2::Geometry>()) {
        auto rdlGeom = so->asA<scene_rdl2::rdl2::Geometry>();
        MNRY_ASSERT(rdlGeom != nullptr);
        geom::Procedural *p = rdlGeom->getProcedural();
        MNRY_ASSERT(p != nullptr);
        PosMapBuilder builder(mPosMapWidth, mPosMapHeight, rdlCamera->get(mAttrUdim),
                              rdlCamera->get(mAttrUvAttribute), mPosMap, mNrmMap);
        p->forEachPrimitive(builder, /* parallel = */ false);
    }
    // DEBUG, write out our mNrmMap using OIIO
    // if (mNrmMap) {
    //     const char *filename = "debug_normals.exr";
    //     std::unique_ptr<OIIO::ImageOutput> out(OIIO::ImageOutput::create(filename));
    //     OIIO::ImageSpec spec(mPosMapWidth, mPosMapHeight, 3, OIIO::TypeDesc::FLOAT);
    //     OIIO::ImageBuf srcBuffer(filename, spec, reinterpret_cast<float *>(mNrmMap));
    //     OIIO::ImageBuf flippedBuffer(filename, spec);
    //     OIIO::ImageBufAlgo::flip(flippedBuffer, srcBuffer);
    //     out->open(filename, spec);
    //     flippedBuffer.write(out.get());
    //     out->close();
    // }
    // DEBUG, write out mPosMap using OIIO
    // if (mPosMap) {
    //     const char *filename = "debug_position.exr";
    //     std::unique_ptr<OIIO::ImageOutput> out(OIIO::ImageOutput::create(filename));
    //     OIIO::ImageSpec spec(mPosMapWidth, mPosMapHeight, 4, OIIO::TypeDesc::FLOAT);
    //     OIIO::ImageBuf srcBuffer(filename, spec, reinterpret_cast<float *>(mPosMap));
    //     OIIO::ImageBuf flippedBuffer(filename, spec);
    //     OIIO::ImageBufAlgo::flip(flippedBuffer, srcBuffer);
    //     out->open(filename, spec);
    //     flippedBuffer.write(out.get());
    //     out->close();
    // }
}

void
BakeCamera::getRequiredPrimAttributesImpl(shading::PerGeometryAttributeKeySet &keys) const
{
    const scene_rdl2::rdl2::Camera *rdlCamera = getRdlCamera();
    const std::string attrName = rdlCamera->get(mAttrUvAttribute);
    if (!attrName.empty()) {
        shading::TypedAttributeKey<scene_rdl2::math::Vec2f> key(attrName);

        const scene_rdl2::rdl2::SceneObject *so = rdlCamera->get(mAttrGeometry);
        if (so && so->isA<scene_rdl2::rdl2::Geometry>()) {
            const scene_rdl2::rdl2::Geometry *rdlGeom = so->asA<scene_rdl2::rdl2::Geometry>();
            shading::AttributeKeySet keySet;
            keySet.insert(key);
            keys.insert(std::pair<const scene_rdl2::rdl2::Geometry *, shading::AttributeKeySet>(rdlGeom, keySet));
        }
    }
}

void
BakeCamera::updateImpl(const scene_rdl2::math::Mat4d &world2render)
{
    const scene_rdl2::rdl2::Camera *rdlCamera = getRdlCamera();

    // width and height of the map we are baking
    // i think it is likely a setup error if we
    // are using aperture or region windows at all when baking.
    const int width = getApertureWindowWidth();
    const int height = getApertureWindowHeight();
    mWidth = width;
    mHeight = height;

    // has the user supplied a normal map?
    // TODO: handle udim string subst.
    const std::string filename = rdlCamera->get(mAttrNormalMap);
    if (filename != mNormalMap.mFilename) {
        // cleanup previously allocated texture handle
        mNormalMap.mFilename = "";
        mNormalMap.mTextureHandle = nullptr;
        mNormalMap.mTextureSystem = nullptr;

        if (!filename.empty() && needNormals()) {
            // allocate new handle
            texture::TextureSampler *textureSampler = texture::getTextureSampler();
            OIIO::TextureSystem *textureSystem = textureSampler->getTextureSystem();
            std::string errorString;
            texture::TextureHandle *textureHandle = 
                textureSampler->getHandle(filename, errorString, textureSystem->get_perthread_info());
            if (textureHandle) {
                // all good
                mNormalMap.mFilename = filename;
                mNormalMap.mTextureHandle = textureHandle;
                mNormalMap.mTextureSystem = textureSystem;
            } else {
                rdlCamera->error("failed to open normal map file ", filename, " ", errorString);
            }
        }
    }

    // allocate position and optional normal map
    int posMapWidth  = static_cast<int>(floor(width * rdlCamera->get(mAttrMapFactor)));
    int posMapHeight = static_cast<int>(floor(height * rdlCamera->get(mAttrMapFactor)));

    if (posMapWidth != mPosMapWidth || posMapHeight != mPosMapHeight) {
        if (mPosMap) {
            scene_rdl2::util::alignedFreeArray(mPosMap);
            mPosMap = nullptr;
        }
        if (mNrmMap) {
            scene_rdl2::util::alignedFreeArray(mNrmMap);
            mNrmMap = nullptr;
        }
        // we unconditionally have a pos map
        mPosMap = scene_rdl2::util::alignedMallocArray<scene_rdl2::math::Vec3fa>(posMapWidth * posMapHeight * 4);
        if (needNrmMap()) {
            mNrmMap = scene_rdl2::util::alignedMallocArray<scene_rdl2::math::Vec3f>(posMapWidth * posMapHeight * 3);
        }

        mPosMapWidth = posMapWidth;
        mPosMapHeight = posMapHeight;
    } else {
        // if our mode has changed, we might need to delete or allocate a norm map
        if (needNrmMap() && !mNrmMap) {
            mNrmMap = scene_rdl2::util::alignedMallocArray<scene_rdl2::math::Vec3f>(mPosMapWidth * mPosMapHeight * 3);
        } else if (!needNrmMap() && mNrmMap) {
            scene_rdl2::util::alignedFreeArray(mNrmMap);
            mNrmMap = nullptr;
        }
    }

    // Get the geometry of the mesh (not sure what to do if no such geometry is present):
    scene_rdl2::rdl2::SceneObject *so = rdlCamera->get(mAttrGeometry);
    if (so && so->isA<scene_rdl2::rdl2::Geometry>()) {
        auto rdlGeom = so->asA<scene_rdl2::rdl2::Geometry>();
        MNRY_ASSERT(rdlGeom != nullptr);

        // the decomposition for the matrix we want initially:
        const scene_rdl2::math::Mat4d geomToWorldOpen = rdlGeom->get(scene_rdl2::rdl2::Node::sNodeXformKey, 0.0);
        const scene_rdl2::math::Mat4d geomToWorldClose = rdlGeom->get(scene_rdl2::rdl2::Node::sNodeXformKey, 1.0);

        scene_rdl2::math::decompose(scene_rdl2::math::xform<scene_rdl2::math::XformT<scene_rdl2::math::Mat3d>>(geomToWorldOpen),
                                    mGeomToWorldOpen);
        scene_rdl2::math::decompose(scene_rdl2::math::xform<scene_rdl2::math::XformT<scene_rdl2::math::Mat3d>>(geomToWorldClose),
                                    mGeomToWorldClose);
        if (scene_rdl2::math::dot(mGeomToWorldOpen.r, mGeomToWorldClose.r) < 0.0) { mGeomToWorldClose.r *= -1.0; }

        // the matrix to go from render space to geometry space
        const scene_rdl2::math::Mat4d worldToGeom = geomToWorldOpen.inverse();
        const scene_rdl2::math::Mat4d renderToWorld = scene_rdl2::math::toDouble(getRender2Camera()) * getCamera2World();
        mRenderToGeom = renderToWorld * worldToGeom;

        // to go from world space to render space (used later when transforming the outgoing rays)
        mWorldToRender = getWorld2Camera() * scene_rdl2::math::toDouble(getCamera2Render());
    }
}

bool
BakeCamera::computeDpdu(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec2f &uv, scene_rdl2::math::Vec3f &dpdu) const
{
    // it might be worth baking this computation into its own map,
    // both for correctness and efficiency reasons
    // this isn't really dpdu, but rather a normalized vector in the
    // dpdu direction.

    const scene_rdl2::math::Vec2f du = scene_rdl2::math::Vec2f(1.f / (mPosMapWidth - 1.f), 0.f);
    // data to the right?

    const scene_rdl2::math::Vec2f rgtUv = uv + du;
    bool rgtHasData = false;
    scene_rdl2::math::Vec3f rgtP;
    if (rgtUv[0] <= 1.f) {
        scene_rdl2::math::Vec3f newN; // unused
        rgtHasData = interpolatePosMap(rgtUv, rgtP, newN);
    }

    // ok, try to the left
    const scene_rdl2::math::Vec2f lftUv = uv - du;
    bool lftHasData = false;
    scene_rdl2::math::Vec3f lftP;
    if (lftUv[0] >= 0.f) {
        scene_rdl2::math::Vec3f newN; // unused
        lftHasData = interpolatePosMap(lftUv, lftP, newN);
    }

    if (rgtHasData && lftHasData) {
        const scene_rdl2::math::Vec3f dp = rgtP - lftP;
        if (!scene_rdl2::math::isZero(dp.lengthSqr())) {
            dpdu = normalize(dp);
            return true;
        }

    }

    // either one side doesn't have data, or the delta
    // between left and right is 0.
    if (rgtHasData) {
        const scene_rdl2::math::Vec3f dp = rgtP - P;
        if (!scene_rdl2::math::isZero(dp.lengthSqr())) {
            dpdu = normalize(dp);
            return true;
        }
    }

    // getting desperate...
    if (lftHasData) {
        const scene_rdl2::math::Vec3f dp = P - lftP;
        if (!scene_rdl2::math::isZero(dp.lengthSqr())) {
            dpdu = normalize(dp);
            return true;
        }
    }
    
    // we failed.... no data
    return false;
}

// return true if uv has a valid position in the uv unwrap map.
// if true, result will contain an interpolated result from the map.
bool
BakeCamera::interpolatePosMap(const scene_rdl2::math::Vec2f &uv, scene_rdl2::math::Vec3f &posResult, scene_rdl2::math::Vec3f &nrmResult) const
{
    const float xf = scene_rdl2::math::clamp(uv[0]) * (mPosMapWidth  - 1);
    const float yf = scene_rdl2::math::clamp(uv[1]) * (mPosMapHeight - 1);
    scene_rdl2::math::BBox2i bbox(scene_rdl2::math::Vec2i(floor(xf), floor(yf)),
                      scene_rdl2::math::Vec2i(ceil(xf), ceil(yf)));

    // bi-linear interpolation.
    //               bbox.upper
    //        +---------+ 
    //        |         |
    //        |  xf,yf  |
    //        |         |
    //        +---------+
    //  bbox.lower

    const scene_rdl2::math::Vec3fa *posLowerLeft  = mPosMap + (bbox.lower.y * mPosMapWidth + bbox.lower.x);
    const scene_rdl2::math::Vec3fa *posUpperLeft  = mPosMap + (bbox.upper.y * mPosMapWidth + bbox.lower.x);
    const scene_rdl2::math::Vec3fa *posLowerRight = mPosMap + (bbox.lower.y * mPosMapWidth + bbox.upper.x);
    const scene_rdl2::math::Vec3fa *posUpperRight = mPosMap + (bbox.upper.y * mPosMapWidth + bbox.upper.x);

    const scene_rdl2::math::Vec3f *nrmLowerLeft  = mNrmMap? mNrmMap + (bbox.lower.y * mPosMapWidth + bbox.lower.x) : nullptr;
    const scene_rdl2::math::Vec3f *nrmUpperLeft  = mNrmMap? mNrmMap + (bbox.upper.y * mPosMapWidth + bbox.lower.x) : nullptr;
    const scene_rdl2::math::Vec3f *nrmLowerRight = mNrmMap? mNrmMap + (bbox.lower.y * mPosMapWidth + bbox.upper.x) : nullptr;
    const scene_rdl2::math::Vec3f *nrmUpperRight = mNrmMap? mNrmMap + (bbox.upper.y * mPosMapWidth + bbox.upper.x) : nullptr;

    // left side average
    scene_rdl2::math::Vec3f posLeftSideAvg(0.f, 0.f, 0.f);
    scene_rdl2::math::Vec3f nrmLeftSideAvg(0.f, 0.f, 0.f);
    bool hasLeftSide(false);
    if (posLowerLeft->w != 0.f) {
        if (posUpperLeft->w != 0.f) {
            // both have data, interpolate the two
            const scene_rdl2::math::Vec3f a(posLowerLeft->asVec3f());
            const scene_rdl2::math::Vec3f b(posUpperLeft->asVec3f());
            posLeftSideAvg = scene_rdl2::math::lerp(a, b, yf - bbox.lower.y);
            if (mNrmMap) {
                const scene_rdl2::math::Vec3f a(*nrmLowerLeft);
                const scene_rdl2::math::Vec3f b(*nrmUpperLeft);
                nrmLeftSideAvg = scene_rdl2::math::lerp(a, b, yf - bbox.lower.y);
            }
            hasLeftSide = true;
        } else {
            // only the lower left has data, if we are closer
            // to the lower side than the upper side, use the data,
            // otherwise, use no data for left side
            if (yf - bbox.lower.y < .5f) {
                posLeftSideAvg = posLowerLeft->asVec3f();
                if (mNrmMap) {
                    nrmLeftSideAvg = *nrmLowerLeft;
                }
                hasLeftSide = true;
            }
        }
    } else if (posUpperLeft->w != 0.f) {
        // only the upper left has data, if we are closer
        // to the upper side than the lower side, use the data,
        // otherwise, use no data for the left side
        if (yf - bbox.lower.y >= .5f) {
            posLeftSideAvg = posUpperLeft->asVec3f();
            if (mNrmMap) {
                nrmLeftSideAvg = *nrmUpperLeft;
            }
            hasLeftSide = true;
        }
    }

    // right side average
    scene_rdl2::math::Vec3f posRightSideAvg(0.f, 0.f, 0.f);
    scene_rdl2::math::Vec3f nrmRightSideAvg(0.f, 0.f, 0.f);
    bool hasRightSide(false);
    if (posLowerRight->w != 0.f) {
        if (posUpperRight->w != 0.f) {
            // both have data, interpolate the two
            const scene_rdl2::math::Vec3f a(posLowerRight->asVec3f());
            const scene_rdl2::math::Vec3f b(posUpperRight->asVec3f());
            posRightSideAvg = scene_rdl2::math::lerp(a, b, yf - bbox.lower.y);
            if (mNrmMap) {
                const scene_rdl2::math::Vec3f a(*nrmLowerRight);
                const scene_rdl2::math::Vec3f b(*nrmUpperRight);
                nrmRightSideAvg = scene_rdl2::math::lerp(a, b, yf - bbox.lower.y);
            }
            hasRightSide = true;
        } else {
            // only the lower right has data, if we are closer
            // to the lower side than the upper side, use the data,
            // otherwise, use no data for right side
            if (yf - bbox.lower.y < .5f) {
                posRightSideAvg = posLowerRight->asVec3f();
                if (mNrmMap) {
                    nrmRightSideAvg = *nrmLowerRight;
                }
                hasRightSide = true;
            }
        }
    } else if (posUpperRight->w != 0.f) {
        // only the upper right has data, if we are closer
        // to the upper side than the lower side, use the data,
        // otherwise, use no data for the right side
        if (yf - bbox.lower.y >= .5f) {
            posRightSideAvg = posUpperRight->asVec3f();
            if (mNrmMap) {
                nrmRightSideAvg = *nrmUpperRight;
            }
            hasRightSide = true;
        }
    }

    // now average the left and right side values
    bool hasData = false;
    if (hasLeftSide) {
        if (hasRightSide) {
            // both have data, interpolate the two
            posResult = scene_rdl2::math::lerp(posLeftSideAvg, posRightSideAvg, xf - bbox.lower.x);
            if (mNrmMap) {
                nrmResult = scene_rdl2::math::lerp(nrmLeftSideAvg, nrmRightSideAvg, xf - bbox.lower.x);
            }
            hasData = true;
        } else {
            // only the left side has data.  if we are closer
            // to the left side than the right, use the data,
            // otherwise, use no data
            if (xf - bbox.lower.x < .5f) {
                posResult = posLeftSideAvg;
                if (mNrmMap) {
                    nrmResult = nrmLeftSideAvg;
                }
                hasData = true;
            }
        }
    } else if (hasRightSide) {
        // only the right side has data.  if we are closer
        // to the right side than the left, use the data,
        // othwerwise, use no data
        if (xf - bbox.lower.x >= .5f) {
            posResult = posRightSideAvg;
            if (mNrmMap) {
                nrmResult = nrmRightSideAvg;
            }
            hasData = true;
        }
    }

    // we need to re-normalize our interpolated normal
    if (hasData && mNrmMap) {
        nrmResult = nrmResult.normalize();
    }

    return hasData;
}

void
BakeCamera::createRayImpl(mcrt_common::RayDifferential *dstRay,
                          float x,
                          float y,
                          float time,
                          float lensU,
                          float lensV) const
{
    // raster location
    const scene_rdl2::math::Vec3f Pr(x, y, -1.0f);

    // uv location
    scene_rdl2::math::Vec2f uv;
    uv[0] = Pr.x / mWidth;
    uv[1] = Pr.y / mHeight;

    const scene_rdl2::math::Vec2f Puv = scene_rdl2::math::Vec2f(uv[0], uv[1]);

    // lookup P and optionally N in render space
    scene_rdl2::math::Vec3f P = scene_rdl2::math::Vec3f(0, 0, 0);
    scene_rdl2::math::Vec3f N = scene_rdl2::math::Vec3f(0, 0, 0);
    bool hasData = interpolatePosMap(Puv, P, N);

    // TODO: extend camera API to more explicitly handle an invalid ray case
    if (!hasData) {
        *dstRay = mcrt_common::RayDifferential(scene_rdl2::math::Vec3f(0, 0, 0), scene_rdl2::math::Vec3f(0, 0, 0),
                                               scene_rdl2::math::Vec3f(0, 0, 0), scene_rdl2::math::Vec3f(0, 0, 0),
                                               scene_rdl2::math::Vec3f(0, 0, 0), scene_rdl2::math::Vec3f(0, 0, 0),
                                               /* near = */ scene_rdl2::math::sMaxValue,
                                               /* far = */ scene_rdl2::math::sMaxValue,
                                               /* time = */ 0.f, /* depth = */ 0);
        return;
    }

    // Adjacent UVs along X and Y
    float uvX = scene_rdl2::math::min(uv[0] + 1.0f / mWidth, 1.0f);
    float uvY = scene_rdl2::math::min(uv[1] + 1.0f / mHeight, 1.0f);

    // Adjacent Points
    const scene_rdl2::math::Vec2f PuvX = scene_rdl2::math::Vec2f(uvX,   uv[1]);
    const scene_rdl2::math::Vec2f PuvY = scene_rdl2::math::Vec2f(uv[0], uvY);
    scene_rdl2::math::Vec3f Px, Py, Nx, Ny;
    bool hasDataX = interpolatePosMap(PuvX, Px, Nx);
    if (!hasDataX) {
        Px = P; Nx = N;
    }
    bool hasDataY = interpolatePosMap(PuvY, Py, Ny);
    if (!hasDataY) {
        Py = P; Ny = N;
    }

    // sample our normal map if we have one
    if (needNormals() && haveNormalMap()) {
        scene_rdl2::math::Vec3f nMap; // sampled value from map
        OIIO::TextureOpt textureOpt;
        textureOpt.swrap = OIIO::TextureOpt::Wrap::WrapClamp;
        textureOpt.twrap = OIIO::TextureOpt::Wrap::WrapClamp;
        textureOpt.interpmode = OIIO::TextureOpt::InterpMode::InterpBilinear;
        bool res = mNormalMap.mTextureSystem->texture(mNormalMap.mTextureHandle,
                                                      mNormalMap.mTextureSystem->get_perthread_info(),
                                                      textureOpt,
                                                      /* s = */ Puv.x, /* t = */ 1.f - Puv.y,
                                                      /* dsdx = */ 0.f, /* dsdy = */ 0.f,
                                                      /* dtdx = */ 0.f, /* dtdy = */ 0.f,
                                                      /* nchans = */ 3, &nMap[0]);

        // not sure what to do if this fails
        MNRY_ASSERT(res);
        if (!res) {
            nMap = N;
        }

        // if our normal map is in render space (likely created
        // via a normal aov) - then just use the normal map value directly
        if (getRdlCamera()->get(mAttrNormalMapSpace) == 0) {
            // render space
            N = nMap;
        } else {
            // the normal map is in tangent space (see
            // evalNormal in shading/EvalAttribute.h for how these
            // normals are expected to be encoded).  Note that the frame
            // input is the state normal (which is stored in our Nrm map
            // and dpds (which we can compute from our posMap).
            scene_rdl2::math::Vec3f dpdu;
            bool hasData = computeDpdu(P, Puv, dpdu);
            if (hasData) {
                MNRY_ASSERT(scene_rdl2::math::isNormalized(N));
                MNRY_ASSERT(scene_rdl2::math::isNormalized(dpdu));
                const scene_rdl2::math::ReferenceFrame frame(N, dpdu);
                // recenter Nmap [0, 1] -> [-1, 1]
                nMap = 2.0f * nMap - scene_rdl2::math::Vec3f(1.f, 1.f, 1.f);
                nMap = frame.localToGlobal(nMap);
                N = normalize(nMap);
            } else {
                // not a lot of good options in this case, we'll construct
                // a reference frame around N and hope for the best
                const scene_rdl2::math::ReferenceFrame frame(N);
                // recenter Nmap [0, 1] -> [-1, 1]
                nMap = 2.0f * nMap - scene_rdl2::math::Vec3f(1.f, 1.f, 1.f);
                nMap = frame.localToGlobal(nMap);
                N = normalize(nMap);
            }
        }
    }
    // Transform the normal and the point on the geometry if the geometry is moving:
    xformPoint(time, P, N);
    xformPoint(time, Px, Nx);
    xformPoint(time, Py, Ny);

    // ray direction and ray origin depend on mode
    scene_rdl2::math::Vec3f rayOrg,  rayDir;
    scene_rdl2::math::Vec3f rayOrgX, rayDirX;
    scene_rdl2::math::Vec3f rayOrgY, rayDirY;
    float near = getNear();
    float far = getFar();
    float bias = getRdlCamera()->get(mAttrBias);
    if (getRdlCamera()->get(mAttrUseRelativeBias)) {
        // scale the bias so that the magnitude of bias compared to 1.0
        // remains constant to the magnitude of the largest component of P
        // this is similar to how we implement relative floating point comparisons
        bias *= scene_rdl2::math::max(scene_rdl2::math::abs(P.x), scene_rdl2::math::abs(P.y),
                                      scene_rdl2::math::abs(P.z), 1.f);
    }

    // Camera position in camera space:
    scene_rdl2::math::Vec3f cameraPos = scene_rdl2::math::Vec3f(0, 0, 0);

    switch (getRdlCamera()->get(mAttrMode)) {
    case MODE_FROM_CAMERA_TO_SURFACE:
        // Transform the camera as appropriate:
        cameraPos = scene_rdl2::math::transformPoint(computeCamera2Render(time), cameraPos);

        fromCameraToSurface(bias, cameraPos, P, N,
                            rayOrg, rayDir);
        fromCameraToSurface(bias, cameraPos, Px, Nx,
                            rayOrgX, rayDirX);
        fromCameraToSurface(bias, cameraPos, Py, Ny,
                            rayOrgY, rayDirY);

        // if the bias is positive,
        // reset the near plane so that it is between
        // the bias location and P
        //   *-----|-----P----------|
        // bias   near              far
        if (bias > 0) {
            near = bias / 2.f;
        }
        break;

    case MODE_FROM_SURFACE_ALONG_NORMAL:
        fromSurfaceAlongNormal(bias, P, N,
                               rayOrg, rayDir);
        fromSurfaceAlongNormal(bias, Px, Nx,
                               rayOrgX, rayDirX);
        fromSurfaceAlongNormal(bias, Py, Ny,
                               rayOrgY, rayDirY);
        break;

    case MODE_FROM_SURFACE_ALONG_REFLECTION_VECTOR:
        // Transform the camera as appropriate:
        cameraPos = scene_rdl2::math::transformPoint(computeCamera2Render(time), cameraPos);
        fromSurfaceAlongReflectionVector(bias, cameraPos,
                                         P, N,
                                         rayOrg, rayDir);
        fromSurfaceAlongReflectionVector(bias, cameraPos,
                                         Px, Nx,
                                         rayOrgX, rayDirX);
        fromSurfaceAlongReflectionVector(bias, cameraPos,
                                         Py, Ny,
                                         rayOrgY, rayDirY);
        break;

    case MODE_ABOVE_SURFACE_REVERSE_NORMAL:

        fromAboveSurfaceNormal(bias, P, N,
                               rayOrg, rayDir);
        fromAboveSurfaceNormal(bias, Px, Nx,
                               rayOrgX, rayDirX);
        fromAboveSurfaceNormal(bias, Py, Ny,
                               rayOrgY, rayDirY);

        // if the bias is positive,
        // reset the near plane so that it is between
        // the bias location and P
        //   *-----|-----P----------|
        // bias   near              far
        if (bias > 0) {
            near = bias / 2.f;
        }

        break;

    default:
        MNRY_ASSERT(0, "unknown bake camera mode\n");
    }

    // fill out our destination ray
    // the RayDifferential now includes a time value to help with object motion blurr.
    *dstRay = mcrt_common::RayDifferential(rayOrg, rayDir,
                                           rayOrgX, rayDirX,
                                           rayOrgY, rayDirY,
                                           near, far,
                                           time, /* depth = */ 0);

}

// Transform the normal and the point on the geometry if the geometry is moving:
void
BakeCamera::xformPoint(float time,
                       scene_rdl2::math::Vec3f& P,
                       scene_rdl2::math::Vec3f& N) const
{
    // first we transform them to geometry space:
    const scene_rdl2::math::Vec3d pGeom = scene_rdl2::math::transformPoint(mRenderToGeom, P);
    const scene_rdl2::math::Vec3d nGeom = scene_rdl2::math::transformVector(mRenderToGeom, N);

    // the transformation to world space given the time provided
    const scene_rdl2::math::Mat4d geomToWorld = scene_rdl2::math::Mat4d(slerp(mGeomToWorldOpen, mGeomToWorldClose, time).combined());
    const scene_rdl2::math::Mat4d geomToRender = geomToWorld * mWorldToRender;

    // transform the point and normal from geometry space to render space:
    P = scene_rdl2::math::transformPoint(geomToRender, pGeom);
    N = scene_rdl2::math::transformVector(geomToRender, nGeom);
}

// Create Rays from surface along reflection vector
void
BakeCamera::fromSurfaceAlongReflectionVector(float bias,
                                             const scene_rdl2::math::Vec3f& cameraPos,
                                             const scene_rdl2::math::Vec3f& P,
                                             const scene_rdl2::math::Vec3f& N,
                                             scene_rdl2::math::Vec3f& rayOrg,
                                             scene_rdl2::math::Vec3f& rayDir) const
{
    // from position along reflection vector (from camera)
    // note that we incorrectly bake back-facing polygons,
    // that should be harmless, but we could improve efficiency
    // by returning invalid rays in that case.
    shading::computeReflectionDirection(N, /* wo = */ normalize(cameraPos - P), rayDir);
    MNRY_ASSERT(isNormalized(rayDir));

    // offset ray position just barely above the surface to avoid
    // self intersection.
    rayOrg = P + bias * rayDir;
}

// Create Rays from Camera to Surface
void
BakeCamera::fromCameraToSurface(float bias,
                                const scene_rdl2::math::Vec3f& cameraPos,
                                const scene_rdl2::math::Vec3f& P,
                                const scene_rdl2::math::Vec3f& N,
                                scene_rdl2::math::Vec3f& rayOrg,
                                scene_rdl2::math::Vec3f& rayDir) const
{
    // camera to position
    rayDir = normalize(P - cameraPos);

    // offset the ray origin just a tad from the surface
    // we want to be close enough that no other geometry will
    // intersect our ray
    // ray direction
    rayOrg = P - bias * rayDir;
}

// Create Rays from Surface Along Normals
void
BakeCamera::fromSurfaceAlongNormal(float bias,
                                   const scene_rdl2::math::Vec3f& P,
                                   const scene_rdl2::math::Vec3f& N,
                                   scene_rdl2::math::Vec3f& rayOrg,
                                   scene_rdl2::math::Vec3f& rayDir) const
{
    // from position along normal
    rayDir = normalize(N);
    // offset ray position just barely above the surface to avoid
    // self intersection.
    rayOrg = P + bias * rayDir;
}

// Create Rays from Above Surface pointing towards reverse normals
void
BakeCamera::fromAboveSurfaceNormal(float bias,
                                   const scene_rdl2::math::Vec3f& P,
                                   const scene_rdl2::math::Vec3f& N,
                                   scene_rdl2::math::Vec3f& rayOrg,
                                   scene_rdl2::math::Vec3f& rayDir) const
{
    // from a position just above the surface in the -N direction
    rayDir = -normalize(N);
    rayOrg = P - bias * rayDir;
}

} // namespace pbr
} // namespace moonray

