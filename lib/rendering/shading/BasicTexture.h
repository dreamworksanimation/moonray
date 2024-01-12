// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <moonray/rendering/shading/Shading.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <moonray/rendering/shading/ispc/BasicTexture_ispc_stubs.h>

namespace moonray {
namespace shading {

class BasicTexture {
public:
    BasicTexture(scene_rdl2::rdl2::Shader *shader,
                 scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry);

    ~BasicTexture();

    bool update(const std::string &filename,
                ispc::TEXTURE_GammaMode gammaMode,
                WrapType wrapS,
                WrapType wrapT,
                bool useDefaultColor,
                const scene_rdl2::math::Color& defaultColor,
                const scene_rdl2::math::Color& fatalColor,
                std::string &errorMsg);

    bool isValid() const;

    scene_rdl2::math::Color4 sample(shading::TLState *tls,
                                    const shading::State& state,
                                    const scene_rdl2::math::Vec2f& st,
                                    float *derivatives) const;

    void getDimensions(int &x, int& y) const;

    float getPixelAspectRatio() const;

    const ispc::BASIC_TEXTURE_Data& getBasicTextureData() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

// Function exposed to ISPC:
extern "C"
{
void CPP_oiioTexture(const ispc::BASIC_TEXTURE_Data* tx,
                     shading::TLState *tls,
                     const uint32_t displacement,
                     const int pathType, // todo: shading::Intersection::PathType
                     const float* derivatives,
                     const float* st,
                     float* result);
}

} // namespace shading
} // namespace moonray

