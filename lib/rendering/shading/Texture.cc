// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Texture.h"

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>

namespace moonray {
namespace shading {

int
getTextureOptionIndex(bool isDisplacement,
                      const shading::State &state)
{
    const shading::Intersection::PathType pathType = state.getIntersection()->getPathType();

    if (isDisplacement) {
        return TrilinearIsotropic;
    }

    switch (pathType) {
    case Intersection::IndirectDiffuse:
        return ClosestMipClosestTexel;

    case Intersection::IndirectGlossy:
        return LinearMipClosestTexel;

    case Intersection::Primary:
        return TrilinearAnisotropic;

    case Intersection::IndirectMirror:
        return LinearMipClosestTexel;

    case Intersection::LightPath:
        return LinearMipClosestTexel;

    default:
        MNRY_ASSERT(0  &&  "Missing switch case");
        return 0;
    }
}

int
getTextureOptionIndex(bool isDisplacement,
                      shading::Intersection::PathType pathType)
{
    if (isDisplacement) {
        return TrilinearIsotropic;
    }

    switch (pathType) {
    case Intersection::IndirectDiffuse:
        return ClosestMipClosestTexel;

    case Intersection::IndirectGlossy:
        return LinearMipClosestTexel;

    case Intersection::Primary:
        return TrilinearAnisotropic;

    case Intersection::IndirectMirror:
        return LinearMipClosestTexel;

    case Intersection::LightPath:
        return LinearMipClosestTexel;

    default:
        MNRY_ASSERT(0  &&  "Missing switch case");
        return 0;
    }
}

bool
checkTextureWindow(texture::TextureSampler* textureSampler,
                   texture::TextureHandle* handle,
                   const std::string &filename,
                   std::string &errorMsg)
{
    int dataWindow[4] = {0};
    textureSampler->getTextureInfo(handle, "datawindow", dataWindow);
    if (dataWindow[0] < 0 || dataWindow[1] < 0 ||
        dataWindow[2] < 0 || dataWindow[3] < 0) {
        std::ostringstream os;
        os << "FATAL: \"" << filename <<
              "\" has negative pixels in dataWindow: (" <<
              dataWindow[0] << " " << dataWindow[1] << " " <<
              dataWindow[2] << " " << dataWindow[3] << ")";
        errorMsg = os.str();
        return false;
    }

    int displayWindow[4] = {0};
    textureSampler->getTextureInfo(handle, "displaywindow", displayWindow);
    if (displayWindow[0] < 0 || displayWindow[1] < 0 ||
        displayWindow[2] < 0 || displayWindow[3] < 0) {
        std::ostringstream os;
        os << "FATAL: \"" << filename <<
              "\" has negative pixels in displayWindow: (" <<
              displayWindow[0] << " " << displayWindow[1] << " " <<
              displayWindow[2] << " " << displayWindow[3] << ")";
        errorMsg = os.str();
        return false;
    }

    return true;
}

} // namespace shading
} // namespace moonray

