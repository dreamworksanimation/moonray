// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file InstanceProceduralLeaf.cc
/// $Id$
///

#include "InstanceProceduralLeaf.h"

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/SharedPrimitive.h>

namespace {

void
validateXformAttrs(const moonray::geom::InstanceMethod instanceMethod,
                   const scene_rdl2::rdl2::Vec4fVector& orientations,
                   const scene_rdl2::rdl2::Vec3fVector& scales,
                   const scene_rdl2::rdl2::Geometry& rdlGeometry,
                   const size_t xformsCount,
                   bool& applyOrientation,
                   bool& applyScale)
{
    if (instanceMethod == moonray::geom::InstanceMethod::XFORM_ATTRIBUTES) {
        // validate orientations
        if (orientations.size() == xformsCount) {
            applyOrientation = true;
        } else if (orientations.empty()) {
            applyOrientation = false;
        } else {
            rdlGeometry.warn("orientations count(", orientations.size(),
                    ") is not equal to positions count(", xformsCount,
                    " ). Skip applying orientations");
            applyOrientation = false;
        }

        // validate scales
        if (scales.size() == xformsCount) {
            applyScale = true;
        } else if (scales.empty()) {
            applyScale = false;
        } else {
            rdlGeometry.warn("scales count(", scales.size(),
                    ") is not equal to positions count(", xformsCount,
                    "). Skip applying scales");
            applyScale = false;
        }
    } else { // InstanceMethod::XFORMS
        applyOrientation = false;
        applyScale = false;
    }
}

bool
isValidXform(const moonray::shading::XformSamples& xforms)
{
    for (const auto& xform : xforms) {
        if (!scene_rdl2::math::isFinite(xform.row0()) ||
            !scene_rdl2::math::isFinite(xform.row1()) ||
            !scene_rdl2::math::isFinite(xform.row2()) ||
            !scene_rdl2::math::isFinite(xform.row3())) {
            return false;
        }
    }
    return true;
}

} // end anonymous namespace


namespace moonray {
namespace geom {

bool
getReferenceData(const scene_rdl2::rdl2::Geometry& rdlGeometry,
                 const scene_rdl2::rdl2::SceneObjectVector& references,
                 std::vector<std::shared_ptr<SharedPrimitive>>& refPrimitiveGroups,
                 std::vector<moonray::shading::XformSamples>& refXforms,
                 std::vector<float>& refShadowRayEpsilons)
{
    if (references.empty()) {
        rdlGeometry.error("Did not find any references geometry. "
            "Please make sure the \"referencess\" field contains "
            "at least one source references geometry");
        return false;
    }

    for (size_t i = 0; i < references.size(); ++i) {
        if (references[i]->isA<scene_rdl2::rdl2::Geometry>()) {
            const scene_rdl2::rdl2::Geometry* g = references[i]->asA<scene_rdl2::rdl2::Geometry>();
            refShadowRayEpsilons.push_back(g->getShadowRayEpsilon());
            refPrimitiveGroups[i] = g->getProcedural()->getReference();

            scene_rdl2::math::Mat4f l2r0 =
                scene_rdl2::math::toFloat(g->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                                 scene_rdl2::rdl2::TIMESTEP_BEGIN));
            refXforms[i].push_back(scene_rdl2::math::Xform3f(l2r0[0][0], l2r0[0][1], l2r0[0][2],
                                                             l2r0[1][0], l2r0[1][1], l2r0[1][2],
                                                             l2r0[2][0], l2r0[2][1], l2r0[2][2],
                                                             l2r0[3][0], l2r0[3][1], l2r0[3][2]));

            scene_rdl2::math::Mat4f l2r1 =
                scene_rdl2::math::toFloat(g->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                                 scene_rdl2::rdl2::TIMESTEP_END));
            refXforms[i].push_back(scene_rdl2::math::Xform3f(l2r1[0][0], l2r1[0][1], l2r1[0][2],
                                                             l2r1[1][0], l2r1[1][1], l2r1[1][2],
                                                             l2r1[2][0], l2r1[2][1], l2r1[2][2],
                                                             l2r1[3][0], l2r1[3][1], l2r1[3][2]));

        }
    }

    return true;
}

InstanceProceduralLeaf::InstanceProceduralLeaf(const State &state) :
        ProceduralLeaf(state)
{
}

InstanceProceduralLeaf::~InstanceProceduralLeaf()
{
}

void
InstanceProceduralLeaf::instanceWithXforms(const GenerateContext& generateContext,
                                           const shading::XformSamples& parent2render,
                                           const scene_rdl2::math::Mat4f& nodeXform,
                                           const std::vector<std::shared_ptr<SharedPrimitive>>& ref,
                                           const std::vector<moonray::shading::XformSamples>& refXforms,
                                           const std::vector<float>& shadowRayEpsilons,
                                           const InstanceMethod instanceMethod,
                                           bool useRefXforms,
                                           bool useRefAttrs,
                                           bool explicitShading,
                                           const int instanceLevel,
                                           const scene_rdl2::rdl2::Vec3fVector& velocities,
                                           const float evaluationFrame,
                                           const scene_rdl2::rdl2::IntVector& indices,
                                           const scene_rdl2::rdl2::IntVector& disableIndices,
                                           const scene_rdl2::rdl2::SceneObjectVector& attributes,
                                           const scene_rdl2::rdl2::Vec3fVector& positions,
                                           const scene_rdl2::rdl2::Vec4fVector& orientations,
                                           const scene_rdl2::rdl2::Vec3fVector& scales,
                                           const scene_rdl2::rdl2::Mat4dVector& xforms)
    {
        const scene_rdl2::rdl2::Geometry* rdlGeometry = generateContext.getRdlGeometry();

        // how many instances are we going to create ?
        size_t xformsCount;
        if (instanceMethod == InstanceMethod::XFORM_ATTRIBUTES) {
            xformsCount = positions.size();
        } else { // InstanceMethod::XFORMS
            xformsCount = xforms.size();
        }

        // motion blur
        bool applyVelocity;
        bool hasValidVelocity = velocities.size() == xformsCount;
        if (generateContext.isMotionBlurOn() && !velocities.empty()) {
            if (!hasValidVelocity) {
                rdlGeometry->warn("velocities count(", velocities.size(),
                    ") is not equal to positions count(", xformsCount,
                    " ). Skip applying velocities");
                applyVelocity = false;
            } else {
                applyVelocity = true;
            }
        } else {
            applyVelocity = false;
        }
        float dt0 = 0.0f;
        float dt1 = 0.0f;
        if (applyVelocity) {
            const scene_rdl2::rdl2::SceneVariables& vars =
                rdlGeometry->getSceneClass().getSceneContext()->getSceneVariables();
            float fps = std::max(vars.get(scene_rdl2::rdl2::SceneVariables::sFpsKey), 1.0f);
            const auto& motionSteps = generateContext.getMotionSteps();
            // TODO only support at most two time samples now for motion blur
            MNRY_ASSERT(motionSteps.size() == 2);

            // BAND-AID fix to evaluate from a different evaluation frame (relative)
            // defaults to 0 (current frame)
            dt0 = (motionSteps[0] - evaluationFrame) / fps;
            dt1 = (motionSteps[1] - evaluationFrame) / fps;
        }

        // indexing
        bool applyIndex;
        bool groupOnly = false;
        if (indices.size() == xformsCount) {
            applyIndex = true;
        } else if (indices.empty()) {
            applyIndex = false;
        } else if (xformsCount == 0) {
            // If there are no xforms then treat as a group node
            // with one instance per reference with no transforms
            applyIndex = true;
            groupOnly = true;
            xformsCount = indices.size();
        } else {
            rdlGeometry->warn("refIndices count(", indices.size(),
                ") is not equal to xforms/positions count(", xformsCount,
                "). Skip applying refIndices");
            applyIndex = false;
        }

        // disabled indices
        std::unordered_set<int> disableIndicesSet;
        for (auto &i : disableIndices) {
            if (i < (int) xformsCount) disableIndicesSet.insert(i);
        }

        // primitive attributes
        std::unordered_map<shading::AttributeKey, const void*, shading::AttributeKeyHash> attrMap;

        static const shading::TypedAttributeKey<scene_rdl2::math::Mat4f> instanceLevelKeys[] = {
            shading::StandardAttributes::sInstanceTransformLevel0,
            shading::StandardAttributes::sInstanceTransformLevel1,
            shading::StandardAttributes::sInstanceTransformLevel2,
            shading::StandardAttributes::sInstanceTransformLevel3,
            shading::StandardAttributes::sInstanceTransformLevel4
        };


        bool addInstanceTransformAttribute = false;
        shading::TypedAttributeKey<scene_rdl2::math::Mat4f> instanceLevelKey =
            instanceLevelKeys[instanceLevel];

        bool addInstanceObjectTransformAttribute = false;

        for (const auto& key : generateContext.getRequestedAttributes()) {
            if (key == instanceLevelKey) {
                addInstanceTransformAttribute = true;
                continue;
            } else if (key == moonray::shading::StandardAttributes::sInstanceObjectTransform) {
                addInstanceObjectTransformAttribute = true;
                continue;
            }

            for (const auto& attr : attributes) {
                if (!attr->isA<scene_rdl2::rdl2::UserData>()) {
                    continue;
                }
                if (attrMap.find(key) != attrMap.end()) {
                    continue;
                }
                const scene_rdl2::rdl2::UserData* data = attr->asA<scene_rdl2::rdl2::UserData>();
                scene_rdl2::rdl2::AttributeType type = key.getType();
                const std::string& name = key.getName();
                if (type == scene_rdl2::rdl2::TYPE_BOOL && data->hasBoolData() &&
                    name == data->getBoolKey()) {
                    if (data->getBoolValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getBoolValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_INT && data->hasIntData() &&
                    name == data->getIntKey()) {
                    if (data->getIntValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getIntValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_FLOAT && data->hasFloatData() &&
                    name == data->getFloatKey()) {
                    if (data->getFloatValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getFloatValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_STRING && data->hasStringData() &&
                    name == data->getStringKey()) {
                    if (data->getStringValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getStringValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_RGB && data->hasColorData() &&
                    name == data->getColorKey()) {
                    if (data->getColorValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getColorValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_VEC2F && data->hasVec2fData() &&
                    name == data->getVec2fKey()) {
                    if (data->getVec2fValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getVec2fValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_VEC3F && data->hasVec3fData() &&
                    name == data->getVec3fKey()) {
                    if (data->getVec3fValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getVec3fValues());
                    }
                } else if (type == scene_rdl2::rdl2::TYPE_MAT4F && data->hasMat4fData() &&
                    name == data->getMat4fKey()) {
                    if (data->getMat4fValues().size() == xformsCount) {
                        attrMap[key] = (const void*)(&data->getMat4fValues());
                    }
                }
            }
            attrMap.insert({key, nullptr}); // Will only succeed if key does not already exist.
        }

        bool addVelocityAttribute = false;
        if (generateContext.requestAttribute(shading::StandardAttributes::sVelocity) &&
            hasValidVelocity) {
            addVelocityAttribute = true;
        }

        bool applyOrientation, applyScale;
        if (!groupOnly) {
            validateXformAttrs(instanceMethod, 
                               orientations,
                               scales,
                               *rdlGeometry,
                               xformsCount,
                               applyOrientation,
                               applyScale);
        }

        size_t badXformCount = 0;
        int maxIndex = ref.size() - 1;

        // disableIndicesSet contains no duplicates or elements >= xformsCount
        reservePrimitive(xformsCount - disableIndicesSet.size());
        // combine above info to XformSamples
        for (size_t i = 0; i < xformsCount; ++i) {
            if (!disableIndicesSet.empty() && disableIndicesSet.find(i) != disableIndicesSet.end()) {
                continue;
            }

            shading::XformSamples xform;

            if (groupOnly) {
                xform.emplace_back(scene_rdl2::math::Mat3f(scene_rdl2::math::one), scene_rdl2::math::Vec3f(scene_rdl2::math::zero));
            } else {
                if (instanceMethod == InstanceMethod::XFORM_ATTRIBUTES) {
                    Vec3f scale = applyScale ? scales[i] : Vec3f(scene_rdl2::math::one);
                    scene_rdl2::math::Quaternion3f orient(scene_rdl2::math::one);
                    if (applyOrientation && orientations[i].lengthSqr() > 0.0f) {
                        orient.i = orientations[i][0];
                        orient.j = orientations[i][1];
                        orient.k = orientations[i][2];
                        orient.r = orientations[i][3];
                        orient = normalize(orient);
                    }
                    scene_rdl2::math::Mat3f scaleRotate(
                            scene_rdl2::math::Mat3f::scale(scale) * scene_rdl2::math::Mat3f(orient));
                    if (applyVelocity) {
                        xform.emplace_back(
                                scaleRotate, positions[i] + velocities[i] * dt0);
                        xform.emplace_back(
                                scaleRotate, positions[i] + velocities[i] * dt1);
                    } else {
                        xform.emplace_back(scaleRotate, positions[i]);
                    }
                } else if (instanceMethod == InstanceMethod::XFORMS) {
                    if (applyVelocity) {
                        const auto& mat4 = xforms[i];

                        scene_rdl2::math::Xform3f xf0 = scene_rdl2::math::xform<scene_rdl2::math::Xform3f>(mat4);
                        scene_rdl2::math::Xform3f xf1 = scene_rdl2::math::xform<scene_rdl2::math::Xform3f>(mat4);

                        xf0.p += velocities[i] * dt0;
                        xf1.p += velocities[i] * dt1;

                        xform.push_back(xf0);
                        xform.push_back(xf1);
                    } else {
                        const auto& mat4 = xforms[i];
                        xform.push_back(scene_rdl2::math::xform<scene_rdl2::math::Xform3f>(mat4));
                    }
                } else {
                   rdlGeometry->error("Unknown instancing method.");
                   return;
                }
            }

            int index = applyIndex ? indices[i] : 0;
            if (index > maxIndex) {
                index = 0;
            }
            if (!ref[index]) {
                continue;
            }

            shading::PrimitiveAttributeTable primitiveAttributeTable;

            // handle a special case that user requests velocity,
            // which is stored in its own separate location "velocities"
            // instead of "primitive attributes" field
            if (addVelocityAttribute) {
                primitiveAttributeTable.addAttribute(
                    shading::StandardAttributes::sVelocity, shading::RATE_CONSTANT,
                    {velocities[i]});
            }
            if (addInstanceTransformAttribute) {
                scene_rdl2::math::Mat4f instanceTransform(xform[0]);
                if (useRefXforms) {
                    instanceTransform = static_cast<scene_rdl2::math::Mat4f>(refXforms[index][0]) * instanceTransform;
                }

                primitiveAttributeTable.addAttribute(
                    instanceLevelKey,
                    shading::RATE_CONSTANT,
                    { instanceTransform });
            }
            if (addInstanceObjectTransformAttribute) {
                primitiveAttributeTable.addAttribute(
                    shading::StandardAttributes::sInstanceObjectTransform,
                    shading::RATE_CONSTANT,
                    { nodeXform });
            }
            if (useRefAttrs) {
                // Only adding shadow_ray_epsilon for now
                // MOONRAY-4313 - Propagate common geometry attributes to instanced primitives
                std::vector<float> shadowRayEpsilon = { shadowRayEpsilons[index] };
                primitiveAttributeTable.addAttribute(
                    shading::StandardAttributes::sShadowRayEpsilon,
                    shading::RATE_CONSTANT,
                    std::move(shadowRayEpsilon));
            }

            // Add explicit shading primitive attribute if explicit shading is enabled
            if (explicitShading &&
                !addExplicitShading(rdlGeometry, primitiveAttributeTable)) {
                return;
            }

            for (const auto& kv : attrMap) {
                const auto& key = kv.first;
                const void* values = kv.second;
                if (values == nullptr) {
                    continue;
                }
                switch (key.getType()) {
                case scene_rdl2::rdl2::TYPE_BOOL:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<bool>(key), shading::RATE_CONSTANT,
                        std::vector<bool>({ (*(const scene_rdl2::rdl2::BoolVector*)values)[i]}) );
                    break;
                case scene_rdl2::rdl2::TYPE_INT:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<int>(key), shading::RATE_CONSTANT,
                        std::vector<int>({ (*(const scene_rdl2::rdl2::IntVector*)values)[i]}) );
                    break;
                case scene_rdl2::rdl2::TYPE_FLOAT:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<float>(key), shading::RATE_CONSTANT,
                        std::vector<float>({ (*(const scene_rdl2::rdl2::FloatVector*)values)[i]}) );
                    break;
                case scene_rdl2::rdl2::TYPE_STRING:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<std::string>(key), shading::RATE_CONSTANT,
                        {(*(const scene_rdl2::rdl2::StringVector*)values)[i]});
                    break;
                case scene_rdl2::rdl2::TYPE_RGB:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<scene_rdl2::math::Color>(key), shading::RATE_CONSTANT,
                        {(*(const scene_rdl2::rdl2::RgbVector*)values)[i]});
                    break;
                case scene_rdl2::rdl2::TYPE_VEC2F:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<Vec2f>(key), shading::RATE_CONSTANT,
                        {(*(const scene_rdl2::rdl2::Vec2fVector*)values)[i]});
                    break;
                case scene_rdl2::rdl2::TYPE_VEC3F:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<Vec3f>(key), shading::RATE_CONSTANT,
                        {(*(const scene_rdl2::rdl2::Vec3fVector*)values)[i]});
                    break;
                case scene_rdl2::rdl2::TYPE_MAT4F:
                    primitiveAttributeTable.addAttribute(
                        shading::TypedAttributeKey<scene_rdl2::math::Mat4f>(key), shading::RATE_CONSTANT,
                        {(*(const scene_rdl2::rdl2::Mat4fVector*)values)[i]});
                    break;
                default:
                    break;
                }
            }
            if (useRefXforms) {
                for (size_t j = 0; j < xform.size(); ++j) {
                    xform[j] = refXforms[index][j] * xform[j];
                }
            }
            if (isValidXform(xform)) {
                auto instance = createInstance(xform,
                                               ref[index],
                                               std::move(primitiveAttributeTable));
                addPrimitive(std::move(instance),
                             generateContext.getMotionBlurParams(),
                             parent2render);
            } else {
                badXformCount++;
            }
        }
        if (badXformCount > 0) {
            rdlGeometry->warn("Skipped ", badXformCount,
                " instances which contained invalid transform matrices");
        }
    }



} // namespace geom
} // namespace moonray

