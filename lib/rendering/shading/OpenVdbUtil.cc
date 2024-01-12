// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "OpenVdbUtil.h"

#include <scene_rdl2/common/except/exceptions.h>

using namespace scene_rdl2::rdl2;

namespace moonray {
namespace shading {

bool
isOpenVdbGeometry(const Geometry* geom)
{
    static const std::string sOpenVdbGeometrySceneClassName("OpenVdbGeometry");
    static const std::string sVdbGeometrySceneClassName("VdbGeometry");
    const SceneClass &sc = geom->getSceneClass();
    return (sc.getName() == sOpenVdbGeometrySceneClassName || sc.getName() == sVdbGeometrySceneClassName);
}

AttributeKey<String>
getModelAttributeKey(const Geometry* geom)
{
    // initially !isValid()
    AttributeKey<String> result;

    const SceneClass &sc = geom->getSceneClass();
    try {
        result = sc.getAttributeKey<String>("model");
    } catch (const scene_rdl2::except::TypeError&) {
        // This SceneClass does not have a rdl2::String attr
        // named 'model', just return the invalid AttributeKey
    }

    return result;
}

} // namespace shading
} // namespace moonray


