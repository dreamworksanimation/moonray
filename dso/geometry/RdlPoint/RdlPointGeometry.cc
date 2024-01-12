// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file RdlPointGeometry.cc

/*
 * RdlPointGeometry is a procedural that pretends to be a primitive.
 * This allows geometry data to be stored directly in the RDL
 * scene rather than requiring it be loaded from a container file.
 * It's bulky but useful.
 */

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/PrimitiveUserData.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <string>
#include <vector>

#include "attributes.cc"

using namespace scene_rdl2;

RDL2_DSO_CLASS_BEGIN(RdlPointGeometry, scene_rdl2::rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(RdlPointGeometry)
    moonray::geom::Procedural* createProcedural() const;
    void destroyProcedural() const;

RDL2_DSO_CLASS_END(RdlPointGeometry)

namespace {

const scene_rdl2::rdl2::String sDefaultPartName("");

moonray::geom::LayerAssignmentId
createPerPointAssignmentId(const RdlPointGeometry* rdlPointGeometry,
                           const scene_rdl2::rdl2::Layer* rdlLayer,
                           const size_t pointCount) {

    // part_list: list of part names
    // part_indices: part index for each point in vertex_list
    const std::vector<scene_rdl2::rdl2::Vec3f> &points = rdlPointGeometry->get(attrPos0);
    const std::vector<scene_rdl2::rdl2::Int> &partIndices = rdlPointGeometry->get(attrPartIndices);
    const std::vector<scene_rdl2::rdl2::String> &partList = rdlPointGeometry->get(attrPartList);

    // default assignment id
    const int defaultAssignmentId = rdlLayer->getAssignmentId(rdlPointGeometry,
            sDefaultPartName);

    if (partList.empty()) {
        return moonray::geom::LayerAssignmentId(defaultAssignmentId);
    }

    // input validation, if we have parts
    if (partIndices.size() && points.size() != partIndices.size()) {
        rdlPointGeometry->warn("part_indices size does not match vertex_list size");
    }

    // loop to assign parts
    bool warn = false;
    std::vector<int> pointAssignmentIds(points.size(), defaultAssignmentId);
    for (size_t i = 0; i < partIndices.size() && i < points.size(); ++i) {
        const int partIndex = partIndices[i];
        if (partIndex >= 0 && size_t(partIndex) < partList.size()) {
            const int layerAssignmentId = rdlLayer->getAssignmentId(
                rdlPointGeometry, partList[partIndex]);
            pointAssignmentIds[i] = layerAssignmentId;
        } else if (!warn) {
            rdlPointGeometry->warn("part_indices contains values outside of part_list length");
            warn = true;
        }
    }

    return moonray::geom::LayerAssignmentId(std::move(pointAssignmentIds));
}

}  // anonymous namespace


namespace moonray {
namespace geom {

class RdlPointProcedural : public ProceduralLeaf
{
public:
    RdlPointProcedural(const State &state)
        : ProceduralLeaf(state)
    {}
    
    void generate(const GenerateContext &generateContext,
            const shading::XformSamples &parent2render);
};

void
RdlPointProcedural::generate(
        const GenerateContext &generateContext,
        const shading::XformSamples &parent2render)
{
    const scene_rdl2::rdl2::Geometry *rdlGeometry = generateContext.getRdlGeometry();

    const RdlPointGeometry *rdlPointGeometry =
        static_cast<const RdlPointGeometry*>(rdlGeometry);

    const scene_rdl2::rdl2::Layer *rdlLayer = generateContext.getRdlLayer();

    shading::PrimitiveAttributeTable primitiveAttributeTable;

    auto& procPosList0 = rdlPointGeometry->get(attrPos0);
    auto& procPosList1 = rdlPointGeometry->get(attrPos1);
    auto& procVelList0 = rdlPointGeometry->get(attrVel0);
    auto& procVelList1 = rdlPointGeometry->get(attrVel1);
    auto& procAccList0 = rdlPointGeometry->get(attrAcc);

    const size_t vertCount  = procPosList0.size();

    bool pos1Valid = sizeCheck(rdlPointGeometry, getName(rdlPointGeometry, attrPos1), procPosList1.size(), vertCount);
    bool vel0Valid = sizeCheck(rdlPointGeometry, getName(rdlPointGeometry, attrVel0), procVelList0.size(), vertCount);
    bool vel1Valid = sizeCheck(rdlPointGeometry, getName(rdlPointGeometry, attrVel1), procVelList1.size(), vertCount);
    bool acc0Valid = sizeCheck(rdlPointGeometry, getName(rdlPointGeometry, attrAcc), procAccList0.size(), vertCount);

    // Fall back on static case if we don't have sufficient data for requested mb type
    int numPosSamples = 1;
    int numVelSamples = 0;
    int numAccSamples = 0;

    bool err = false;
    scene_rdl2::rdl2::MotionBlurType motionBlurType =
            static_cast<scene_rdl2::rdl2::MotionBlurType>(rdlPointGeometry->get(attrMotionBlurType));

    // Set motion blur type to static if motion blur is disabled for the scene
    const scene_rdl2::rdl2::SceneVariables &sv = rdlGeometry->getSceneClass().getSceneContext()->getSceneVariables();
    if (!sv.get(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur)) {
        motionBlurType = scene_rdl2::rdl2::MotionBlurType::STATIC;
    }

    switch (motionBlurType) {

    case scene_rdl2::rdl2::MotionBlurType::STATIC:
    {
        break;
    }

    case scene_rdl2::rdl2::MotionBlurType::VELOCITY:
    {
        if (vel0Valid) {
            numVelSamples = 1;
        } else {
            err = true;
        }
        break;
    }

    case scene_rdl2::rdl2::MotionBlurType::FRAME_DELTA:
    {
        if (pos1Valid) {
            numPosSamples = 2;
        } else {
            err = true;
        }
        break;
    }

    case scene_rdl2::rdl2::MotionBlurType::ACCELERATION:
    {
        if (vel0Valid && acc0Valid) {
            numVelSamples = 1;
            numAccSamples = 1;
        } else {
            err = true;
        }
        break;
    }

    case scene_rdl2::rdl2::MotionBlurType::HERMITE:
    {
        if (pos1Valid && vel0Valid && vel1Valid) {
            numPosSamples = 2;
            numVelSamples = 2;
        } else {
            err = true;
        }
        break;
    }

    case scene_rdl2::rdl2::MotionBlurType::BEST:
    {
        if (pos1Valid && vel0Valid && vel1Valid) {
            // use Hermite mb type
            numPosSamples = 2;
            numVelSamples = 2;
        } else if (vel0Valid && acc0Valid) {
            // use acceleration mb type
            numVelSamples = 1;
            numAccSamples = 1;
        } else if (pos1Valid) {
            // use frame delta mb type
            numPosSamples = 2;
        } else if (vel0Valid) {
            // use velocity mb type
            numVelSamples = 1;
        }
        // else just keep static mb type
        break;
    }

    default:
    {
        err = true;
        break;
    }

    } // end of switch statement

    if (err) {
        rdlPointGeometry->warn("Insufficient data for requested motion blur type. "
                               "Falling back to static case.");
    }

    // Copy vertices
    Points::VertexBuffer vertices(vertCount, numPosSamples);
    for (size_t i = 0; i < vertCount; i++) {
        const auto& p = procPosList0[i];
        vertices(i, 0) = Vec3f(p[0], p[1], p[2]);
    }
    if (numPosSamples == 2) {
        for (size_t i = 0; i < vertCount; i++) {
            const auto& p = procPosList1[i];
            vertices(i, 1) = Vec3f(p[0], p[1], p[2]);
        }
    }

    // Add velocity data
    if (numVelSamples > 0) {
        const float velocityScale = rdlPointGeometry->get(attrVelocityScale);
        std::vector<std::vector<Vec3f>> velocities;
        velocities.push_back(std::vector<Vec3f>(procVelList0.begin(), procVelList0.end()));
        for (Vec3f& velocity : velocities.back()) {
            velocity *= velocityScale;
        }
        if (numVelSamples == 2) {
            velocities.push_back(std::vector<Vec3f>(procVelList1.begin(), procVelList1.end()));
            for (Vec3f& velocity : velocities.back()) {
                velocity *= velocityScale;
            }
        }
        primitiveAttributeTable.addAttribute(shading::StandardAttributes::sVelocity,
                                             shading::RATE_VERTEX, std::move(velocities));
    }

    // Add acceleration data
    if (numAccSamples > 0) {
        std::vector<Vec3f> accelerations(procAccList0.begin(), procAccList0.end());
        primitiveAttributeTable.addAttribute(shading::StandardAttributes::sAcceleration,
                                             shading::RATE_VERTEX, std::move(accelerations));
    }

    // radius buffer
    const std::vector<float> &rv = rdlPointGeometry->get(attrRadius);
    Points::RadiusBuffer radius(vertCount);
    const moonray::geom::RateCounts rates{rdlPointGeometry->get(attrPartList).size(), 0, 0, vertCount, 0};
    switch (pickRate(rdlPointGeometry, getName(rdlPointGeometry, attrRadius), rv.size(), rates)) {
    case AttributeRate::RATE_UNKNOWN:
        radius.assign(vertCount, 0.5f);
        break;
    case AttributeRate::RATE_CONSTANT:
        radius.assign(vertCount, rv[0]);
        break;
    case AttributeRate::RATE_PART: {
        const std::vector<scene_rdl2::rdl2::Int> &partIndices = rdlPointGeometry->get(attrPartIndices);
        radius.reserve(vertCount);
        for (size_t i = 0; i < vertCount; ++i)
            radius.emplace_back(rv[partIndices[i]]);
        break;}
    default:
        radius.assign(rv.begin(), rv.begin()+vertCount);
        break;
    }

    // layer assignments
    LayerAssignmentId layerAssignmentId = createPerPointAssignmentId(rdlPointGeometry,
                                                                     rdlLayer,
                                                                     vertCount);

    // primitive attributes
    scene_rdl2::rdl2::PrimitiveAttributeFrame primitiveAttributeFrame =
        static_cast<scene_rdl2::rdl2::PrimitiveAttributeFrame>(rdlPointGeometry->get(attrPrimitiveAttributeFrame));

    bool useFirstFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::SECOND_MOTION_STEP);
    bool useSecondFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::FIRST_MOTION_STEP);

    moonray::geom::processArbitraryData(rdlPointGeometry,
                                        attrPrimitiveAttributes,
                                        primitiveAttributeTable,
                                        rates,
                                        useFirstFrame,
                                        useSecondFrame);

    // Add explicit shading primitive attribute if explicit shading is enabled
    if (rdlGeometry->get(attrExplicitShading) &&
        !addExplicitShading(rdlGeometry, primitiveAttributeTable)) {
        return;
    }
    
    std::unique_ptr<Points> primitive = createPoints(std::move(vertices),
                                                     std::move(radius),
                                                     std::move(layerAssignmentId),
                                                     std::move(primitiveAttributeTable));

    if (primitive) {
        primitive->setCurvedMotionBlurSampleCount(rdlPointGeometry->get(attrCurvedMotionBlurSampleCount));

        // may need to convert the primitive to instance to handle
        // rotation motion blur
        std::unique_ptr<Primitive> p =
            convertForMotionBlur(generateContext,
                                 std::move(primitive),
                                 (rdlPointGeometry->get(attrUseRotationMotionBlur) && parent2render.size() > 1));

        addPrimitive(std::move(p),
                     generateContext.getMotionBlurParams(),
                     parent2render);
    }
}


} // namespace geom
} // namespace moonray

moonray::geom::Procedural*
RdlPointGeometry::createProcedural() const
{
    moonray::geom::State state;
    // Do not call state.setName here since scene_rdl2::rdl2::rdlLayer::assignmentId already
    // use rdlGeometry name. 
        
    return new moonray::geom::RdlPointProcedural(state);
}

void
RdlPointGeometry::destroyProcedural() const
{
    delete mProcedural;
}
