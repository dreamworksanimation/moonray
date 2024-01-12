// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#include "Xform.h"

#include <moonray/rendering/bvh/shading/State.h>

using namespace scene_rdl2;

namespace moonray {
namespace shading {

namespace {
void
getR2O(const rdl2::Geometry *geom,
       math::Xform3f *out)
{
    *out = geom->getRender2Object();
}

math::Mat4d
getR2W(const rdl2::SceneContext* context)
{
    MNRY_ASSERT(context->getRender2World());
    return *context->getRender2World();
}

math::Mat4d
getW2C(const rdl2::Camera* camera)
{
    math::Mat4d w2c(math::one);
    if (camera) {
        w2c = camera->get(rdl2::Node::sNodeXformKey).inverse();
    }
    return w2c;
}

math::Mat4f
getC2S(const rdl2::Camera* camera, const std::array<float, 4>& window)
{
    math::Mat4f c2s(math::one);
    if (camera && camera->doesSupportProjectionMatrix()) {
        c2s = camera->computeProjectionMatrix(0.0f, window, 0.0f);
    }
    return c2s;
}

math::Mat4d
getW2O(const rdl2::Node* node)
{
    math::Mat4d w2o(math::one);
    if (node) {
        w2o = node->get(rdl2::Node::sNodeXformKey).inverse();
    }
    return w2o;
}
} // namespace

Xform::Xform(const rdl2::SceneObject *shader) :
        Xform(shader, nullptr, nullptr, nullptr)
{
    // Call specialized constructor with default options
    // to camera, window and geometry (all nullptrs) in
    // the initializer list.
}

Xform::Xform(const rdl2::SceneObject *shader,
             const rdl2::Node *customObject,
             const rdl2::Camera *customCamera,
             const std::array<float, 4>* customWindow)
{
    MNRY_ASSERT(shader);
    const rdl2::SceneContext* ctx = shader->getSceneClass().getSceneContext();

    // Render to object
    scene_rdl2::math::Mat4d objectMatrix(math::one);
    if (customObject != nullptr) { // use custom geometry for object space
        objectMatrix = getR2W(ctx) * getW2O(customObject);
        mIspc.mUseExternalObj = true;
    } else {
        // Use the space of shading point's object
        // Since this is only available during sample, we don't store
        // a transformation here
        mIspc.mGetR2OFn = (intptr_t) getR2O;
        mIspc.mUseExternalObj = false;
    }

    precalculateMatrices(shader,
                         objectMatrix,
                         customCamera,
                         customWindow);
}

Xform::Xform(const rdl2::SceneObject *shader,
             const scene_rdl2::math::Mat4d& objectMatrix)
{
    MNRY_ASSERT(shader);
    const rdl2::SceneContext* ctx = shader->getSceneClass().getSceneContext();

    mIspc.mUseExternalObj = true;

    precalculateMatrices(shader,
                         getR2W(ctx) * objectMatrix.inverse(),
                         nullptr, // no custom camera
                         nullptr);// no custom window
}

void
Xform::precalculateMatrices(const rdl2::SceneObject *shader,
                            const scene_rdl2::math::Mat4d& r2oMatrix,
                            const rdl2::Camera *customCamera,
                            const std::array<float, 4>* customWindow)
{
    MNRY_ASSERT(shader);
    const rdl2::SceneContext* ctx = shader->getSceneClass().getSceneContext();

    std::array<float, 4> window;
    // Build window out of screen's aspect ratio, if custom window is not provided
    if (customWindow == nullptr) {
        const rdl2::SceneVariables &vars = ctx->getSceneVariables();
        const math::HalfOpenViewport& aperture = vars.getRezedApertureWindow();
        const float invAspectRatio = float(aperture.height()) / float(aperture.width());
        window = {-1.0f, -invAspectRatio, 1.0f, invAspectRatio};
    } else {
        window = *customWindow;
    }

    // Pre-calculate various xforms and their inverse xforms
    std::vector<const rdl2::Camera*> cameras = ctx->getActiveCameras();

    // Use current active camera, if custom camera is not provided
    // Render to Camera
    const rdl2::Camera* cam = (customCamera == nullptr) ? cameras[0] : customCamera;
    math::asCpp(mIspc.mR2C) = math::xform<math::Xform3f>(getR2W(ctx) * getW2C(cam));
    math::asCpp(mIspc.mR2CInv) = math::asCpp(mIspc.mR2C).inverse();
    // Render to screen
    math::asCpp(mIspc.mR2S) = toFloat(getR2W(ctx) * getW2C(cam)) * getC2S(cam, window);
    math::asCpp(mIspc.mR2SInv) = math::asCpp(mIspc.mR2S).inverse();

    // Render to world
    math::asCpp(mIspc.mR2W) = math::xform<math::Xform3f>(getR2W(ctx));
    math::asCpp(mIspc.mR2WInv) = math::asCpp(mIspc.mR2W).inverse();

    // Render to object
    math::asCpp(mIspc.mR2O) = math::xform<math::Xform3f>(r2oMatrix);
    math::asCpp(mIspc.mR2OInv) = math::asCpp(mIspc.mR2O).inverse();
}

const ispc::Xform*
Xform::getIspcXform() const
{
    return &mIspc;
}

math::Vec3f
Xform::transformPoint(const int srcSpace,
                      const int dstSpace,
                      const State &state, 
                      const math::Vec3f inPoint) const
{
    math::Vec3f rPoint;
    math::Vec3f result;

    // Transform to render space if necessary
    switch(srcSpace) {
    case ispc::SHADING_SPACE_RENDER:
        rPoint = inPoint;
        break;
    case ispc::SHADING_SPACE_CAMERA:
        rPoint = math::transformPoint(math::asCpp(mIspc.mR2CInv), inPoint);
        break;
    case ispc::SHADING_SPACE_WORLD:
        rPoint = math::transformPoint(math::asCpp(mIspc.mR2WInv), inPoint);
        break;
    case ispc::SHADING_SPACE_SCREEN:
        rPoint = math::transformH(math::asCpp(mIspc.mR2SInv), inPoint);
        break;
    case ispc::SHADING_SPACE_OBJECT:
        {
            if (mIspc.mUseExternalObj) {
                rPoint = math::transformPoint(math::asCpp(mIspc.mR2OInv), inPoint);
            } else {
                math::Xform3f r2o;
                getR2O(state.getGeometryObject(), &r2o);
                rPoint = math::transformPoint(r2o.inverse(), inPoint);
            }
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown src space");
        break;
    }

    // Transform from render space
    switch(dstSpace) {
    case ispc::SHADING_SPACE_RENDER:
        result = rPoint;
        break;
    case ispc::SHADING_SPACE_CAMERA:
        result = math::transformPoint(math::asCpp(mIspc.mR2C), rPoint);
        break;
    case ispc::SHADING_SPACE_WORLD:
        result = math::transformPoint(math::asCpp(mIspc.mR2W), rPoint);
        break;
    case ispc::SHADING_SPACE_SCREEN:
        result = math::transformH(math::asCpp(mIspc.mR2S), rPoint);
        break;
    case ispc::SHADING_SPACE_OBJECT:
        {
            if (mIspc.mUseExternalObj) {
                result = math::transformPoint(math::asCpp(mIspc.mR2O), rPoint);
            } else {
                math::Xform3f r2o;
                getR2O(state.getGeometryObject(), &r2o);
                result = math::transformPoint(r2o, rPoint);
            }
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown dst space");
        break;
    }

    return result;
}

math::Vec3f
Xform::transformNormal(const int srcSpace,
                       const int dstSpace,
                       const State &state, 
                       const math::Vec3f inNormal) const
{
    math::Vec3f rNormal;
    math::Vec3f result;

    // Transform to render space if necessary
    // Normal transform requires inverse transpose of a point xform matrix
    // transformNormal takes care of the transpose internally
    switch(srcSpace) {
    case ispc::SHADING_SPACE_RENDER:
        rNormal = inNormal;
        break;
    case ispc::SHADING_SPACE_CAMERA:
        rNormal = math::transformNormal(math::asCpp(mIspc.mR2C), inNormal);
        break;
    case ispc::SHADING_SPACE_WORLD:
        rNormal = math::transformNormal(math::asCpp(mIspc.mR2W), inNormal);
        break;
    case ispc::SHADING_SPACE_SCREEN:
        rNormal = math::transformNormal(math::asCpp(mIspc.mR2S), inNormal);
        break;
    case ispc::SHADING_SPACE_OBJECT:
        {
            if (mIspc.mUseExternalObj) {
                rNormal = math::transformNormal(math::asCpp(mIspc.mR2O), inNormal);
            } else {
                math::Xform3f r2o;
                getR2O(state.getGeometryObject(), &r2o);
                rNormal = math::transformNormal(r2o, inNormal);
            }
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown src space");
        break;
    }

    // Transform from render space
    switch(dstSpace) {
    case ispc::SHADING_SPACE_RENDER:
        result = rNormal;
        break;
    case ispc::SHADING_SPACE_CAMERA:
        result = math::transformNormal(math::asCpp(mIspc.mR2CInv), rNormal);
        break;
    case ispc::SHADING_SPACE_WORLD:
        result = math::transformNormal(math::asCpp(mIspc.mR2WInv), rNormal);
        break;
    case ispc::SHADING_SPACE_SCREEN:
        result = math::transformNormal(math::asCpp(mIspc.mR2SInv), rNormal);
        break;
    case ispc::SHADING_SPACE_OBJECT:
        {
            if (mIspc.mUseExternalObj) {
                result = math::transformNormal(math::asCpp(mIspc.mR2OInv), rNormal);
            } else {
                math::Xform3f r2o;
                getR2O(state.getGeometryObject(), &r2o);
                result = math::transformNormal(r2o.inverse(), rNormal);
            }
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown dst space");
        break;
    }

    return result;
}

math::Vec3f
Xform::transformVector(const int srcSpace,
                       const int dstSpace,
                       const State &state, 
                       const math::Vec3f inVector) const
{
    math::Vec3f rVector;
    math::Vec3f result;

    // Transform to render space if necessary
    switch(srcSpace) {
    case ispc::SHADING_SPACE_RENDER:
        rVector = inVector;
        break;
    case ispc::SHADING_SPACE_CAMERA:
        rVector = math::transformVector(math::asCpp(mIspc.mR2CInv), inVector);
        break;
    case ispc::SHADING_SPACE_WORLD:
        rVector = math::transformVector(math::asCpp(mIspc.mR2WInv), inVector);
        break;
    case ispc::SHADING_SPACE_SCREEN:
        rVector = math::transformVector(math::asCpp(mIspc.mR2SInv), inVector);
        break;
    case ispc::SHADING_SPACE_OBJECT:
        {
            if (mIspc.mUseExternalObj) {
                rVector = math::transformVector(math::asCpp(mIspc.mR2OInv), inVector);
            } else {
                math::Xform3f r2o;
                getR2O(state.getGeometryObject(), &r2o);
                rVector = math::transformVector(r2o.inverse(), inVector);
            }
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown src space");
        break;
    }

    // Transform from render space
    switch(dstSpace) {
    case ispc::SHADING_SPACE_RENDER:
        result = rVector;
        break;
    case ispc::SHADING_SPACE_CAMERA:
        result = math::transformVector(math::asCpp(mIspc.mR2C), rVector);
        break;
    case ispc::SHADING_SPACE_WORLD:
        result = math::transformVector(math::asCpp(mIspc.mR2W), rVector);
        break;
    case ispc::SHADING_SPACE_SCREEN:
        result = math::transformVector(math::asCpp(mIspc.mR2S), rVector);
        break;
    case ispc::SHADING_SPACE_OBJECT:
        {
            if (mIspc.mUseExternalObj) {
                result = math::transformVector(math::asCpp(mIspc.mR2O), rVector);
            } else {
                math::Xform3f r2o;
                getR2O(state.getGeometryObject(), &r2o);
                result = math::transformVector(r2o, rVector);
            }
        }
        break;
    default:
        MNRY_ASSERT(0 && "unknown dst space");
        break;
    }

    return result;
}

} // namespace shading
} // namespace moonray

