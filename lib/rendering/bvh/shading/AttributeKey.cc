// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "AttributeKey.h"

using namespace scene_rdl2;

namespace moonray {
namespace shading {

tbb::mutex AttributeKey::sRegisterMutex;
std::vector<std::string> AttributeKey::sKeyNames;
std::vector<AttributeType> AttributeKey::sKeyTypes;
std::vector<size_t> AttributeKey::sKeySizes;
std::vector<int8_t> AttributeKey::sHasDerivatives;
std::map<std::pair<std::string, AttributeType>, int> AttributeKey::sTable;
int AttributeKey::sNumKeys = 0;

TypedAttributeKey<math::Vec2f> StandardAttributes::sUv;
TypedAttributeKey<math::Vec2f> StandardAttributes::sClosestSurfaceST;
TypedAttributeKey<math::Vec2f> StandardAttributes::sSurfaceST;
TypedAttributeKey<math::Mat4f> StandardAttributes::sInstanceObjectTransform;
TypedAttributeKey<math::Mat4f> StandardAttributes::sInstanceTransformLevel0;
TypedAttributeKey<math::Mat4f> StandardAttributes::sInstanceTransformLevel1;
TypedAttributeKey<math::Mat4f> StandardAttributes::sInstanceTransformLevel2;
TypedAttributeKey<math::Mat4f> StandardAttributes::sInstanceTransformLevel3;
TypedAttributeKey<math::Mat4f> StandardAttributes::sInstanceTransformLevel4;
TypedAttributeKey<float> StandardAttributes::sScatterTag;
TypedAttributeKey<float> StandardAttributes::sShadowRayEpsilon;
TypedAttributeKey<math::Vec3f> StandardAttributes::sNormal;
TypedAttributeKey<math::Vec3f> StandardAttributes::sRefP;
TypedAttributeKey<math::Vec3f> StandardAttributes::sRefN;
TypedAttributeKey<math::Vec3f> StandardAttributes::sdPds;
TypedAttributeKey<math::Vec3f> StandardAttributes::sRefdPds;
TypedAttributeKey<math::Vec3f> StandardAttributes::sdPdt;
TypedAttributeKey<math::Vec3f> StandardAttributes::sVelocity;
TypedAttributeKey<math::Vec3f> StandardAttributes::sP0;
TypedAttributeKey<math::Vec3f> StandardAttributes::sAcceleration;
TypedAttributeKey<math::Vec3f> StandardAttributes::sMotion;
TypedAttributeKey<math::Vec3f> StandardAttributes::sPolyVertices[MAX_NUM_POLYVERTICES];
TypedAttributeKey<int> StandardAttributes::sNumPolyVertices;
TypedAttributeKey<int> StandardAttributes::sPolyVertexType;
TypedAttributeKey<int> StandardAttributes::sId;
TypedAttributeKey<bool> StandardAttributes::sReversedNormals;
TypedAttributeKey<bool> StandardAttributes::sExplicitShading;

} // namespace shading
} // namespace rendering


