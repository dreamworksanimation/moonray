// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "DebugMap_ispc_stubs.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/shading/MapApi.h>

#include <string>

using namespace scene_rdl2::math;
using namespace moonray::shading;


//---------------------------------------------------------------------------

namespace {


void
sample2D(const Vec2f& value, bool checkerBoard, Color& sample)
{
    sample.r = value.x;
    sample.g = value.y;
    sample.b = 0.0f;

    if (checkerBoard) {
        if (((((int)(value.x * 16.0f) % 2) +
              ((int)(value.y * 16.0f) % 2)) % 2) == 0) {
            sample *= 0.5f;
        }
    }
}

void
sample3D(const Vec3f& value, bool checkerBoard, Color& sample)
{
    sample.r = value.x;
    sample.g = value.y;
    sample.b = value.z;
    if (checkerBoard) {
        if ((((((int)(value.x * 16.0f) % 2) +
                ((int)(value.y * 16.0f) % 2)) +
                ((int)(value.z * 16.0f) % 2))) % 2 == 0) {
            sample *= 0.5f;
        }
    }
}

} // namespace


//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(DebugMap, scene_rdl2::rdl2::Map)

public:
    DebugMap(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);
    ~DebugMap();
    virtual void update();

private:
    static void sample(const scene_rdl2::rdl2::Map* self, moonray::shading::TLState *tls,
                       const moonray::shading::State& state, Color* sample);
    
    ispc::DebugMap mData;
    
RDL2_DSO_CLASS_END(DebugMap)


//---------------------------------------------------------------------------

DebugMap::DebugMap(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name) :
    Parent(sceneClass, name)
{
    mSampleFunc = DebugMap::sample;
    mSampleFuncv = (scene_rdl2::rdl2::SampleFuncv) ispc::DebugMap_getSampleFunc();
}

DebugMap::~DebugMap()
{
}

void
DebugMap::update()
{
    if (hasChanged(attrDebugMapType) ||
        hasChanged(attrPrimitiveAttributeType) ||
        hasChanged(attrPrimitiveAttributeName)) {
        mRequiredAttributes.clear();
        if (get(attrDebugMapType) == ispc::DebugMapType::PRIMITIVE_ATTRIBUTE) {
            int type = get(attrPrimitiveAttributeType);
            const std::string& name = get(attrPrimitiveAttributeName);
            if (type == ispc::TYPE_FLOAT) {
                mData.mPrimitiveAttributeType = ispc::TYPE_FLOAT;
                moonray::shading::TypedAttributeKey<float> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mRequiredAttributes.push_back(attributeKey);
            } else if (type == ispc::TYPE_VEC2F) {
                mData.mPrimitiveAttributeType = ispc::TYPE_VEC2F;
                moonray::shading::TypedAttributeKey<Vec2f> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mRequiredAttributes.push_back(attributeKey);
            } else if (type == ispc::TYPE_VEC3F) {
                mData.mPrimitiveAttributeType = ispc::TYPE_VEC3F;
                moonray::shading::TypedAttributeKey<Vec3f> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mRequiredAttributes.push_back(attributeKey);
            } else if (type == ispc::TYPE_RGB) {
                mData.mPrimitiveAttributeType = ispc::TYPE_RGB;
                moonray::shading::TypedAttributeKey<Color> attributeKey(name);
                mData.mPrimitiveAttributeIndex = attributeKey;
                mRequiredAttributes.push_back(attributeKey);
            } else {
                MNRY_ASSERT(false, "unsupported primitive attribute type");
            }
        }
    }
}

void
DebugMap::sample(const scene_rdl2::rdl2::Map* self, moonray::shading::TLState *tls,
                 const moonray::shading::State& state, Color* sample)
{
    const DebugMap* me = static_cast<const DebugMap*>(self);

    if (me->get(attrDebugMapType) == ispc::DebugMapType::N) {
        sample3D(moonray::shading::evalNormal(me, attrInputNormal, attrInputNormalDial, attrInputNormalSpace, tls, state),
                                     me->get(attrCheckerboard), *sample);
    } else if (me->get(attrDebugMapType) == ispc::DebugMapType::ST) {
        sample2D(state.getSt(), me->get(attrCheckerboard), *sample);
    } else if (me->get(attrDebugMapType) == ispc::DebugMapType::P) {
        sample3D(state.getP(), me->get(attrCheckerboard), *sample);
    } else if (me->get(attrDebugMapType) == ispc::DebugMapType::NG) {
        sample3D(state.getNg(), me->get(attrCheckerboard), *sample);
    } else if (me->get(attrDebugMapType) == ispc::DebugMapType::DPDS) {
        sample3D(state.getdPds(), me->get(attrCheckerboard), *sample);
    } else if (me->get(attrDebugMapType) == ispc::DebugMapType::DPDT) {
        sample3D(state.getdPdt(), me->get(attrCheckerboard), *sample);
    } else if (me->get(attrDebugMapType) == ispc::DebugMapType::PRIMITIVE_ATTRIBUTE) {
        int key = me->mData.mPrimitiveAttributeIndex;
        int type = me->mData.mPrimitiveAttributeType;
        if (type == ispc::TYPE_FLOAT) {
            float f = state.getAttribute(moonray::shading::TypedAttributeKey<float>(key));
            *sample = Color(f);
        } else if (type == ispc::TYPE_VEC2F) {
            const Vec2f& v2 = state.getAttribute(moonray::shading::TypedAttributeKey<Vec2f>(key));
            sample2D(v2, me->get(attrCheckerboard), *sample);
        } else if (type == ispc::TYPE_VEC3F) {
            const Vec3f& v3 = state.getAttribute(moonray::shading::TypedAttributeKey<Vec3f>(key));
            sample3D(v3, me->get(attrCheckerboard), *sample);
        } else if (type == ispc::TYPE_RGB) {
            const Color& c = state.getAttribute(moonray::shading::TypedAttributeKey<Color>(key));
            sample3D(Vec3f(c.r, c.g, c.b), me->get(attrCheckerboard), *sample);
        }
    } else {
        sample->r = 0.f;
        sample->g = 0.f;
        sample->b = 0.f;
        return;
    }
}


//---------------------------------------------------------------------------

