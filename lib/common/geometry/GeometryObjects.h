// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#ifndef LIB_COMMON_SHARED_GEOMETRYOBJECTS_H_
#define LIB_COMMON_SHARED_GEOMETRYOBJECTS_H_

#include <vector>

namespace moonray {

struct ObjectMesh {
    std::string mMeshName;
    std::vector<float> mMeshPositions;
};
typedef std::vector<ObjectMesh> ObjectMeshes;

struct ObjectData {
    std::string mAssetName;
    std::string mSubAssetName;
    std::string mNodeName;
    ObjectMeshes mObjectMeshes;
};

} // end namespace moonray

#endif /* LIB_COMMON_SHARED_GEOMETRYOBJECTS_H_ */

