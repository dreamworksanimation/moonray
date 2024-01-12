// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdUVTexture.cc

#include "attributes.cc"
#include "UsdUVTexture_ispc_stubs.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/shading/BasicTexture.h>
#include <moonray/rendering/shading/ColorCorrect.h>
#include <moonray/rendering/shading/MapApi.h>
#include <moonray/rendering/shading/UdimTexture.h>
#include <scene_rdl2/render/util/stdmemory.h>

using namespace moonray::shading;
using namespace scene_rdl2::math;

static ispc::StaticUsdUVTextureData sStaticUsdUVTextureData;

//----------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(UsdUVTexture, scene_rdl2::rdl2::Map)

public:
    UsdUVTexture(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name);
    ~UsdUVTexture() override;
    void update() override;

private:
    static void sample(const scene_rdl2::rdl2::Map *self, moonray::shading::TLState *tls,
                       const moonray::shading::State &state, Color *sample);

    ispc::UsdUVTexture mIspc;
    std::unique_ptr<moonray::shading::BasicTexture> mTexture;
    std::unique_ptr<moonray::shading::UdimTexture> mUdimTexture;

RDL2_DSO_CLASS_END(UsdUVTexture)

//----------------------------------------------------------------------------

UsdUVTexture::UsdUVTexture(const scene_rdl2::rdl2::SceneClass &sceneClass, const std::string &name):
    Parent(sceneClass, name)
{
    mSampleFunc = UsdUVTexture::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::UsdUVTexture_getSampleFunc();

    mIspc.mUsdUVTextureDataPtr = (ispc::StaticUsdUVTextureData*)&sStaticUsdUVTextureData;

    // register shade time event messages.  we require and expect
    // these events to have the same value across all instances of the shader.
    // no conditional registration of events is allowed.
    // to allow for the possibility that we may someday create these maps
    // on multiple threads, we'll protect the writes of the class statics
    // with a mutex.
    static tbb::mutex errorMutex;
    tbb::mutex::scoped_lock lock(errorMutex);
    MOONRAY_START_THREADSAFE_STATIC_WRITE
    sStaticUsdUVTextureData.sErrorInvalidUdimCoord =
        sLogEventRegistry.createEvent(scene_rdl2::logging::ERROR_LEVEL,
                                      "invalid udim coordinate");
    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
}

UsdUVTexture::~UsdUVTexture()
{
}

void
UsdUVTexture::update()
{
    const std::string filename = get(attrFile);
    const std::size_t udimPos = filename.find("<UDIM>");
    const bool areWeAUdim = udimPos != std::string::npos;

    const scene_rdl2::rdl2::SceneVariables &sv = getSceneClass().getSceneContext()->getSceneVariables();
    mIspc.mFatalColor = asIspc(sv.get(scene_rdl2::rdl2::SceneVariables::sFatalColor));

    // If wrap types are set to the default which uses
    // the metadata from the file then fall back to "black"
    moonray::shading::WrapType wrapS = static_cast<moonray::shading::WrapType>(get(attrWrapS));
    if (wrapS == moonray::shading::WrapType::Default) {
        wrapS = moonray::shading::WrapType::Periodic;
    }

    moonray::shading::WrapType wrapT = static_cast<moonray::shading::WrapType>(get(attrWrapT));
    if (wrapT == moonray::shading::WrapType::Default) {
        wrapT = moonray::shading::WrapType::Periodic;
    }

    std::string errorStr;
    if (areWeAUdim) {
        // Udim update, if needed
        mTexture.reset();
        mIspc.mTexture = nullptr;
        bool needsUpdate = false;
        if (!mUdimTexture) {
            needsUpdate = true;
            mUdimTexture = fauxstd::make_unique<moonray::shading::UdimTexture>(this);
            mIspc.mUdimTexture = &mUdimTexture->getUdimTextureData();
        }

        if (needsUpdate ||
            hasChanged(attrFile) ||
            hasChanged(attrWrapS) ||
            hasChanged(attrWrapT) ||
            hasChanged(attrFallback)) {

            if (!mUdimTexture->update(this,
                                      sLogEventRegistry,
                                      filename,
                                      static_cast<ispc::TEXTURE_GammaMode>(get(attrSourceColorSpace)),
                                      wrapS,
                                      wrapT,
                                      true, // use default/fallback color
                                      get(attrFallback),
                                      asCpp(mIspc.mFatalColor),
                                      get(attrMaxVdim),
                                      get(attrUDimValues),
                                      get(attrUDimFiles),
                                      errorStr)) {
                fatal(errorStr);
                return;
            }
        }
    } else {
        // Non-udim update, if needed
        mUdimTexture.reset();
        mIspc.mUdimTexture = nullptr;
        bool needsUpdate = false;
        if (!mTexture) {
            needsUpdate = true;
            mTexture = fauxstd::make_unique<moonray::shading::BasicTexture>(this, sLogEventRegistry);
            mIspc.mTexture = &mTexture->getBasicTextureData();
        }
        if (needsUpdate ||
            hasChanged(attrFile) ||
            hasChanged(attrWrapS) ||
            hasChanged(attrWrapT) ||
            hasChanged(attrFallback)) {

            if (!mTexture->update(filename,
                                  static_cast<ispc::TEXTURE_GammaMode>(get(attrSourceColorSpace)),
                                  wrapS,
                                  wrapT,
                                  true, // use default/fallback color
                                  get(attrFallback),
                                  asCpp(mIspc.mFatalColor),
                                  errorStr)) {
                fatal(errorStr);
                return;
            }
        }
    }

    if ((mTexture && !mTexture->isValid()) ||
        (mUdimTexture && !mUdimTexture->isValid())) {
        fatal(errorStr);
    }
}

void
UsdUVTexture::sample(const scene_rdl2::rdl2::Map *self,
                     moonray::shading::TLState *tls,
                     const moonray::shading::State &state,
                     Color *sample)
{
    UsdUVTexture const *me = static_cast<UsdUVTexture const *>(self);

    float dsdx, dsdy, dtdx, dtdy;

    Vec2f st = evalVec2f(me, attrSt, tls, state);

    // TODO: How do we get input texture coordinates derivatives ?
    dsdx = dsdy = dtdx = dtdy = 0.0;

    Color4 tx;
    if (me->mTexture) {
        // Invert t coord.
        st.y = 1.0 - st.y;

        // Set and scale derivatives.
        float derivatives[4] = { dsdx, -dtdx, dsdy, -dtdy };

        // sample the texture
        tx = me->mTexture->sample(tls,
                                  state,
                                  st,
                                  derivatives);
    } else if (me->mUdimTexture) {
        // compute udim index
        int udim = me->mUdimTexture->computeUdim(tls, st.x, st.y);
        if (udim == -1) {
            logEvent(me, me->mIspc.mUsdUVTextureDataPtr->sErrorInvalidUdimCoord);
            *sample = asCpp(me->mIspc.mFatalColor);
            return;
        }

        // take fractional parts of st
        st.x = st.x - int(st.x);
        st.y = st.y - int(st.y);

        // Invert t coord.
        st.y = 1.0 - st.y;

        // Set derivatives.
        float derivatives[4] = { dsdx, -dtdx, dsdy, -dtdy };

        // sample the texture
        tx = me->mUdimTexture->sample(tls,
                                      state,
                                      udim,
                                      st,
                                      derivatives);
    }

    Color rgb;
    switch (me->get(attrOutputMode)) {
    case ispc::OUTPUT_R:
        rgb = Color(tx.r);
        break;
    case ispc::OUTPUT_G:
        rgb = Color(tx.g);
        break;
    case ispc::OUTPUT_B:
        rgb = Color(tx.b);
        break;
    case ispc::OUTPUT_A:
        rgb = Color(tx.a);
        break;
    default:
        rgb = Color(tx);
    }

    rgb = rgb * me->get(attrScale) + me->get(attrBias);
    *sample = rgb;
}

