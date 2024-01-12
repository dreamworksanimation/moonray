// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/shading/Shading.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/scene/rdl2/Map.h>
#include <scene_rdl2/common/math/Color.h>

#include <moonray/rendering/shading/ispc/UdimTexture_ispc_stubs.h>

namespace moonray {
namespace shading {

class UdimTexture {
public:
    UdimTexture(scene_rdl2::rdl2::Shader *shader);

    ~UdimTexture();

    bool update(scene_rdl2::rdl2::Shader *shader,
                scene_rdl2::rdl2::ShaderLogEventRegistry& logEventRegistry,
                const std::string &filename,
                ispc::TEXTURE_GammaMode gammaMode,
                WrapType wrapS,
                WrapType wrapT,
                bool useDefaultColor,
                const scene_rdl2::math::Color& defaultColor,
                const scene_rdl2::math::Color& fatalColor,
                int maxVdim,
                const std::vector<int>& udimValues,
                const std::vector<std::string>& udimFiles,
                std::string &errorMsg);


    bool isValid() const;

    scene_rdl2::math::Color4 sample(shading::TLState *tls,
                        const shading::State& state,
                        int udim,
                        const scene_rdl2::math::Vec2f& st,
                        float *derivatives) const;

    void getDimensions(int udim, int &x, int& y) const;

    float getPixelAspectRatio(int udim) const;

    // returns -1 if out of range
    int computeUdim(const shading::TLState *tls, const float u, const float v) const;

    const ispc::UDIM_TEXTURE_Data& getUdimTextureData() const;

    static void setUdimMissingTextureWarningSwitch(bool flag);
    static bool getUdimMissingTextureWarningSwitch();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;

};

// Function exposed to ISPC:
extern "C"
{
void CPP_oiioUdimTexture(const ispc::UDIM_TEXTURE_Data* tx,
                         shading::TLState *tls,
                         const uint32_t displacement,
                         const int pathType, // todo: shading::Intersection::PathType
                         const float* derivatives,
                         const int udim,
                         const float* st,
                         float* result);
}

} // namespace shading
} // namespace moonray

