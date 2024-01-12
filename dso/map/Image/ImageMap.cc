// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "ImageMap_ispc_stubs.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/shading/MapApi.h>
#include <moonray/rendering/shading/BasicTexture.h>
#include <moonray/rendering/shading/UdimTexture.h>
#include <moonray/rendering/shading/ColorCorrect.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include <random>

using namespace scene_rdl2;
using namespace scene_rdl2::math;
using namespace moonray::pbr;
using namespace moonray::texture;

static ispc::StaticImageMapData sStaticImageMapData;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(ImageMap, scene_rdl2::rdl2::Map)

public:
    ImageMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~ImageMap() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, math::Color *result);

    static float getRand();

    void rotateTexCoords(float theta, const math::Vec2f& rotationCenter,
                         math::Vec2f& st, float& dsdx, float& dsdy, float& dtdx, float& dtdy) const;

    void applyColorCorrection(Color& result) const;

    ispc::ImageMap mIspc;

    std::unique_ptr<moonray::shading::BasicTexture> mTexture;
    std::unique_ptr<moonray::shading::UdimTexture> mUdimTexture;
    bool mApplyColorCorrection;

RDL2_DSO_CLASS_END(ImageMap)

/// Pixel filtering as described in "Generation of Stratified Samples for
/// B-Spline Pixel Filtering", by Stark, Shirley, and Ashikhmin
finline float
quadraticBSplineWarp(float r)
{
    if (r < 1.0f / 6.0f) {
        return scene_rdl2::math::pow(6.0f * r, 1.0f / 3.0f) - 3.0f / 2.0f;
    } else if (r < 5.0f / 6.0f) {
        float u = (6.0f * r - 3.0f) / 4.0f;
        for (int j = 0; j < 4; ++j) {
            u = (8.0f * u * u * u - 12.0f * r + 6.0f) / (12.0f * u * u - 9.0f);
        }
        return u;
    } else {
        return 3.0f / 2.0f - scene_rdl2::math::pow(6.0f * (1.0f - r), 1.0f / 3.0f);
    }
}

ImageMap::ImageMap(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
        Parent(sceneClass, name)
{
    mSampleFunc = ImageMap::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::ImageMap_getSampleFunc();

    mIspc.mImageMapDataPtr = (ispc::StaticImageMapData*)&sStaticImageMapData;

    mIspc.mRandFn = (intptr_t)getRand;

    // register shade time event messages.  we require and expect
    // these events to have the same value across all instances of the shader.
    // no conditional registration of events is allowed.
    // to allow for the possibility that we may someday create these maps
    // on multiple threads, we'll protect the writes of the class statics
    // with a mutex.
    static tbb::mutex errorMutex;
    tbb::mutex::scoped_lock lock(errorMutex);
    MOONRAY_START_THREADSAFE_STATIC_WRITE
    sStaticImageMapData.sErrorInvalidUdimCoord =
        sLogEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL,
                                      "invalid udim coordinate");
    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
}

ImageMap::~ImageMap()
{
}

void
ImageMap::update()
{
    std::string filename = get(attrTexture);
    std::size_t udimPos = filename.find("<UDIM>");
    IntVector udimValueList = get(attrUDimValues);
    bool areWeAUdim = (udimPos != std::string::npos
                       || (udimValueList.size() > 0));

    const scene_rdl2::rdl2::SceneVariables &sv = getSceneClass().getSceneContext()->getSceneVariables();
    mIspc.mFatalColor = asIspc(sv.get(scene_rdl2::rdl2::SceneVariables::sFatalColor));

    moonray::shading::WrapType wrapS, wrapT;
    if (get(attrWrapAround)) {
        wrapS = moonray::shading::WrapType::Periodic;
        wrapT = moonray::shading::WrapType::Periodic;
    } else {
        wrapS = moonray::shading::WrapType::Clamp;
        wrapT = moonray::shading::WrapType::Clamp;
    }

    if (areWeAUdim) {
        // Udim update, if needed
        mTexture = nullptr;
        mIspc.mTexture = nullptr;
        bool needsUpdate = false;
        if (!mUdimTexture) {
            needsUpdate = true;
            mUdimTexture = fauxstd::make_unique<moonray::shading::UdimTexture>(this);
            mIspc.mUdimTexture = &mUdimTexture->getUdimTextureData();
        }
        if (needsUpdate ||
            hasChanged(attrTexture) ||
            hasChanged(attrGamma) ||
            hasChanged(attrWrapAround) ||
            hasChanged(attrUseDefaultColor) ||
            hasChanged(attrDefaultColor) ||
            hasChanged(attrMaxVdim) ||
            hasChanged(attrUDimValues) ||
            hasChanged(attrUDimFiles)) {
            std::string errorStr;
            if (!mUdimTexture->update(this,
                                      sLogEventRegistry,
                                      get(attrTexture),
                                      static_cast<ispc::TEXTURE_GammaMode>(get(attrGamma)),
                                      wrapS,
                                      wrapT,
                                      get(attrUseDefaultColor),
                                      get(attrDefaultColor),
                                      asCpp(mIspc.mFatalColor),
                                      get(attrMaxVdim),
                                      get(attrUDimValues),
                                      get(attrUDimFiles),
                                      errorStr)) {
                fatal(errorStr);
                mUdimTexture = nullptr;
                mIspc.mUdimTexture = nullptr;
                return;
            }
        }
    } else {
        // Non-udim update, if needed
        mUdimTexture = nullptr;
        mIspc.mUdimTexture = nullptr;
        bool needsUpdate = false;
        if (!mTexture) {
            needsUpdate = true;
            mTexture = fauxstd::make_unique<moonray::shading::BasicTexture>(this, sLogEventRegistry);
            mIspc.mTexture = &mTexture->getBasicTextureData();
        }
        if (needsUpdate ||
            hasChanged(attrTexture) ||
            hasChanged(attrGamma) ||
            hasChanged(attrWrapAround)) {
            std::string errorStr;
            if (!mTexture->update(get(attrTexture),
                                  static_cast<ispc::TEXTURE_GammaMode>(get(attrGamma)),
                                  wrapS,
                                  wrapT,
                                  get(attrUseDefaultColor),
                                  get(attrDefaultColor),
                                  asCpp(mIspc.mFatalColor),
                                  errorStr)) {
                fatal(errorStr);
                mTexture = nullptr;
                mIspc.mTexture = nullptr;
                return;
            }
        }
    }

    // Update required attributes to reflect current texture enum.
    mRequiredAttributes.clear();
    if (get(attrTextureEnum) == ispc::SURFACE_ST) {
        mIspc.mHairSurfaceSTKey = moonray::shading::StandardAttributes::sSurfaceST;
        mRequiredAttributes.push_back(moonray::shading::StandardAttributes::sSurfaceST);
    } else if (get(attrTextureEnum) == ispc::CLOSEST_SURFACE_ST) {
        mIspc.mHairSurfaceSTKey = moonray::shading::StandardAttributes::sClosestSurfaceST;
        mRequiredAttributes.push_back(moonray::shading::StandardAttributes::sClosestSurfaceST);
    }

    mApplyColorCorrection = get(attrSaturationEnabled) ||
                            get(attrContrastEnabled) ||
                            get(attrGammaEnabled) ||
                            get(attrGainOffsetEnabled) ||
                            get(attrTMIControlEnabled);

    mIspc.mApplyColorCorrection = mApplyColorCorrection;
}

float
ImageMap::getRand()
{
    std::random_device rd;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    return distribution(rd);
}

void
ImageMap::sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                 const moonray::shading::State &state, math::Color *result)
{
    ImageMap const *me = static_cast<ImageMap const *>(self);

    Vec2f st;
    float dsdx = state.getdSdx();
    float dsdy = state.getdSdy();
    float dtdx = state.getdTdx();
    float dtdy = state.getdTdy();

    switch ( me->get( attrTextureEnum ) ) {
    case ispc::ST:
        st = state.getSt();
        break;

    case ispc::SURFACE_ST:
        // Hair Surface ST
        st = state.getAttribute(moonray::shading::StandardAttributes::sSurfaceST);
        // reset texture lookup dimensions to (for now) perform point sampling
        // (assume each hair has a "point sample" wide diameter
        dsdx = dsdy = dtdx = dtdy = 0.0;
        break;
    case ispc::CLOSEST_SURFACE_ST:
        // Hair Closest Surface ST
        st = state.getAttribute(moonray::shading::StandardAttributes::sClosestSurfaceST);
        // reset texture lookup dimensions to (for now) perform point sampling
        // (assume each hair has a "point sample" wide diameter
        dsdx = dsdy = dtdx = dtdy = 0.0;
        break;
    case ispc::INPUT_TEXTURE_COORDINATES:
        {
            Vec3f uvw = evalVec3f(me, attrInputTextureCoordinate, tls, state);
            st[0] = uvw[0];
            st[1] = uvw[1];
            // TODO: How do we get input texture coordinates derivatives ?
            dsdx = dsdy = dtdx = dtdy = 0.0;

            // A negative w value signals an "invalid" texture coordinate. This is
            // purely a convention.  An example usage would be where a tex coordinate
            // generator map (eg. camera projection) wants to signal to Image map that
            // the coordinate should be considered out of range, and thus we should
            // simply return black.
            if (uvw[2] < 0.f) {
                *result = math::sBlack;
                return;
            }
        }
        break;
    default:
        st = state.getSt();
        break;
    }

    const float mipBias = 1.0f + evalFloat(me, attrMipBias, tls, state);
    const Vec2f scale  = me->get(attrScale);
    int udim = -1;
    int width, height;
    if (me->mTexture) {
        // rotation and scaling only for non-udim case
        const Vec2f offset = me->get(attrOffset);
        const Vec2f rotationCenter = me->get(attrRotationCenter);

        // Rotate coords and derivatives.
        const float theta = math::deg2rad(me->get(attrRotationAngle));
        if (!isZero(theta)) {
            me->rotateTexCoords(theta,
                                rotationCenter,
                                st,
                                dsdx, dsdy, dtdx, dtdy);
        }

        // Scale and translate coords.
        st.x = scale.x * st.x + offset.x;
        st.y = scale.y * st.y + offset.y;

        // Invert t coord.
        st.y = 1.0 - st.y;

        me->mTexture->getDimensions(width, height);
    } else if (me->mUdimTexture) {
        // compute udim index
        udim = me->mUdimTexture->computeUdim(tls, st.x, st.y);
        if (udim == -1) {
            moonray::shading::logEvent(me, me->mIspc.mImageMapDataPtr->sErrorInvalidUdimCoord);
            *result = asCpp(me->mIspc.mFatalColor);
            return;
        }

        // take fractional parts of st
        st.x = st.x - int(st.x);
        st.y = st.y - int(st.y);

        // Invert t coord.
        st.y = 1.0 - st.y;

        me->mUdimTexture->getDimensions(udim, width, height);
    }

    math::Color4 tx;

    if (me->mTexture) {
        // Set and scale derivatives.
        float derivatives[4] = { dsdx * scale.x * mipBias,
                                -dtdx * scale.x * mipBias,
                                 dsdy * scale.y * mipBias,
                                -dtdy * scale.y * mipBias };


        // sample the texture
        tx = me->mTexture->sample(tls,
                                  state,
                                  st,
                                  derivatives);
    } else if (me->mUdimTexture) {
        // Set derivatives.
        float derivatives[4] = { dsdx * mipBias,
                                -dtdx * mipBias,
                                 dsdy * mipBias,
                                -dtdy * mipBias };

        // sample the texture
        tx = me->mUdimTexture->sample(tls,
                                      state,
                                      udim,
                                      st,
                                      derivatives);
    }

    if (me->get(attrAlphaOnly)) {
        const math::Color alpha(tx.a, tx.a, tx.a);
        *result = alpha;
    } else {
        const math::Color rgb(tx);
        *result = rgb;
    }

    if (me->mApplyColorCorrection) {
        me->applyColorCorrection(*result);
    }
}

void
ImageMap::rotateTexCoords(float theta, const math::Vec2f& rotationCenter,
                          math::Vec2f& st, float& dsdx, float& dsdy, float& dtdx, float& dtdy) const
{
    math::Mat3f R( math::cos(theta), -math::sin(theta),     0,
                   math::sin(theta),  math::cos(theta),     0,
                   0,                 0,                    1);
    math::Vec3f st3(st.x, st.y, 0.f);
    math::Vec3f rotationCenter3(rotationCenter.x, rotationCenter.y, 0.f);
    // Translate rotation center to origin.
    st3 -= rotationCenter3;
    // Rotate.
    st3 = st3 * R;
    // Translate rotation center back.
    st3 += rotationCenter3;
    st.x = st3.x;
    st.y = st3.y;

    // Rotate derivatives.
    math::Vec3f dsdxy3(dsdx, dsdy, 0.f);
    math::Vec3f dtdxy3(dtdx, dtdy, 0.f);
    dsdxy3 = dsdxy3 * R.transposed();
    dtdxy3 = dtdxy3 * R.transposed();
    dsdx = dsdxy3.x;
    dsdy = dsdxy3.y;
    dtdx = dtdxy3.x;
    dtdy = dtdxy3.y;
}

void
ImageMap::applyColorCorrection(Color& result) const
{
    if (get(attrSaturationEnabled)) {
        const Color saturation = get(attrSaturation);
        moonray::shading::applySaturation(saturation, result);
    }

    if (get(attrContrastEnabled)) {
        const Color contrast = get(attrContrast);
        moonray::shading::applyNukeContrast(contrast, result);
    }

    if (get(attrGammaEnabled)) {
        const Color gamma = get(attrGammaAdjust);
        const Color invGamma = Color(1.0f / max(sEpsilon, gamma.r),
                                     1.0f / max(sEpsilon, gamma.g),
                                     1.0f / max(sEpsilon, gamma.b));
        moonray::shading::applyGamma(invGamma, result);
    }

    if (get(attrGainOffsetEnabled)) {
        const Color gain = get(attrGain);
        const Color offset = get(attrOffsetAdjust);
        moonray::shading::applyGainAndOffset(gain, offset, result);
    }

    if (get(attrTMIControlEnabled)) {
        const Vec3f tmi = get(attrTMI);
        const Color tmiColor = Color(tmi.x, tmi.y, tmi.z);
        moonray::shading::applyTMI(tmiColor, result);
    }
}

//---------------------------------------------------------------------------

