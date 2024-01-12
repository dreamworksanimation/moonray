// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file EvalShader.cc

#include "BsdfLabels.h"

namespace moonray {
namespace shading {
namespace internal {

extern "C"
void CPP_setBsdfLabels(const scene_rdl2::rdl2::Material *material,
                       shading::TLState *tls,
                       int numStatev,
                       const shading::Statev *statev,
                       scene_rdl2::rdl2::Bsdfv *bsdfv,
                       const int parentLobeCount)
{
    internal::setBsdfLabels(material, tls, numStatev, statev, bsdfv, parentLobeCount);
}

} // namespace internal
} // namespace shading
} // namespace moonray

