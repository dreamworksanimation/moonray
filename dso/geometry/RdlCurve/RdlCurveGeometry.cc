// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/*
 * RdlCurveGeometry is a procedural that pretends to be a primitive.
 * This allows geometry data to be stored directly in the RDL
 * scene rather than requiring it be loaded from a container file.
 * It's bulky but useful.
 */

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/PrimitiveUserData.h>

#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <string>
#include <vector>

#include "attributes.cc"

using namespace scene_rdl2;

RDL2_DSO_CLASS_BEGIN(RdlCurveGeometry, scene_rdl2::rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(RdlCurveGeometry)
    moonray::geom::Procedural* createProcedural() const;
    void destroyProcedural() const;

RDL2_DSO_CLASS_END(RdlCurveGeometry)

namespace {

const scene_rdl2::rdl2::String sDefaultPartName("");

moonray::geom::LayerAssignmentId
createPerCurveAssignmentId(const RdlCurveGeometry* rdlCurveGeometry,
                           const scene_rdl2::rdl2::Layer* rdlLayer,
                           const size_t curveCount) {
    // part_list: list of part names
    const std::vector<scene_rdl2::rdl2::String> &partList = rdlCurveGeometry->get(attrPartList);
    // part_indices: part index for each curve
    const std::vector<scene_rdl2::rdl2::Int> &partIndices = rdlCurveGeometry->get(attrPartIndices);

    // default assignment id
    const int defaultAssignmentId = rdlLayer->getAssignmentId(rdlCurveGeometry,
            sDefaultPartName);

    if (partList.empty()) {
        return moonray::geom::LayerAssignmentId(defaultAssignmentId);
    }

    // input validation, if we have parts
    if (partIndices.size() && curveCount != partIndices.size()) {
        rdlCurveGeometry->warn("part_indices size does not match curveCount");
    }

    // loop to assign parts
    bool warn = false;
    std::vector<int> curveAssignmentIds(curveCount, defaultAssignmentId);
    for (size_t i = 0; i < partIndices.size() && i < curveCount; ++i) {
        const int partIndex = partIndices[i];
        if (partIndex >= 0 && size_t(partIndex) < partList.size()) {
            const int layerAssignmentId = rdlLayer->getAssignmentId(
                rdlCurveGeometry, partList[partIndex]);
            curveAssignmentIds[i] = layerAssignmentId;
        } else if (!warn) {
            rdlCurveGeometry->warn("part_indices contains values outside of part_list length");
            warn = true;
        }
    }

    return moonray::geom::LayerAssignmentId(std::move(curveAssignmentIds));
}

} // anonymous

namespace moonray {
namespace geom {

class RdlCurveProcedural : public ProceduralLeaf
{
public:
    explicit RdlCurveProcedural(const State &state)
        : ProceduralLeaf(state)
    {}

    void generate(const GenerateContext &generateContext,
                  const shading::XformSamples &parent2render);

private:

};

void
RdlCurveProcedural::generate(
        const GenerateContext &generateContext,
        const shading::XformSamples &parent2render)
{
    clear();

    const scene_rdl2::rdl2::Geometry *rdlGeometry = generateContext.getRdlGeometry();

    const RdlCurveGeometry *rdlCurveGeometry =
            static_cast<const RdlCurveGeometry*>(rdlGeometry);

    const scene_rdl2::rdl2::Layer *rdlLayer = generateContext.getRdlLayer();

    // curve type
    const int procType = rdlGeometry->get(attrCurvesType);

    Curves::Type type = Curves::Type::UNKNOWN;
    switch (procType) {
    case 0:
        type = Curves::Type::LINEAR;
        break;
    case 1:
        type = Curves::Type::BEZIER;
        break;
    case 2:
        type = Curves::Type::BSPLINE;
        break;
    default:
        rdlGeometry->warn("Unknown curve type, defaulting to Bezier.");
        type = Curves::Type::BEZIER;
    }

    Curves::SubType subtype = Curves::SubType::UNKNOWN;
    switch (rdlGeometry->get(attrCurvesSubType)) {
    case 0:
        subtype = Curves::SubType::RAY_FACING;
        break;
    case 1:
        subtype = Curves::SubType::ROUND;
        break;
    case 2:
        subtype = Curves::SubType::NORMAL_ORIENTED;
        break;
    default:
        rdlGeometry->warn("Unknown curve subtype, defaulting to ray facing.");
        subtype = Curves::SubType::RAY_FACING;
    }

    int tessellationRate = rdlGeometry->get(attrTessellationRate);

    shading::PrimitiveAttributeTable primitiveAttributeTable;

    auto& procPosList0 = rdlCurveGeometry->get(attrPos0);
    auto& procPosList1 = rdlCurveGeometry->get(attrPos1);
    auto& procVelList0 = rdlCurveGeometry->get(attrVel0);
    auto& procVelList1 = rdlCurveGeometry->get(attrVel1);
    auto& procAccList0 = rdlCurveGeometry->get(attrAcc);

    const size_t vertCount  = procPosList0.size();

    bool pos1Valid = sizeCheck(rdlCurveGeometry, getName(rdlCurveGeometry, attrPos1), procPosList1.size(), vertCount);
    bool vel0Valid = sizeCheck(rdlCurveGeometry, getName(rdlCurveGeometry, attrVel0), procVelList0.size(), vertCount);
    bool vel1Valid = sizeCheck(rdlCurveGeometry, getName(rdlCurveGeometry, attrVel1), procVelList1.size(), vertCount);
    bool acc0Valid = sizeCheck(rdlCurveGeometry, getName(rdlCurveGeometry, attrAcc), procAccList0.size(), vertCount);

    // Fall back on static case if we don't have sufficient data for requested mb type
    int numPosSamples = 1;
    int numVelSamples = 0;
    int numAccSamples = 0;

    bool err = false;
    scene_rdl2::rdl2::MotionBlurType motionBlurType =
            static_cast<scene_rdl2::rdl2::MotionBlurType>(rdlCurveGeometry->get(attrMotionBlurType));

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
        rdlCurveGeometry->warn("Insufficient data for requested motion blur type. "
                               "Falling back to static case.");
    }

    // number of vertices per curve
    const auto& curvesVertexCount = rdlGeometry->get(attrCurvesVertexCount);
    Curves::CurvesVertexCount vertexCounts(curvesVertexCount.begin(), curvesVertexCount.end());

    // Figure out number of segements for varying.
    // These are only correct if each curve segment has a correct number of vertices and
    // the total number of vertices matches the sum of the vertex counts. However if that is
    // false Curve.cc will produce an error and not render the geometry.
    size_t varyingCount;
    switch (type) {
    case Curves::Type::LINEAR:
        varyingCount = vertCount;
        break;
    case Curves::Type::BEZIER:
        varyingCount = (vertCount - vertexCounts.size()) / 3 + vertexCounts.size();
        break;
    case Curves::Type::BSPLINE:
    default:
        varyingCount = vertCount - 2 * vertexCounts.size();
        break;
    }

    // Handle radius interpolations
    const moonray::geom::RateCounts rates{rdlCurveGeometry->get(attrPartList).size(), vertexCounts.size(), varyingCount, vertCount, 0};
    const std::vector<float> &rv = rdlCurveGeometry->get(attrRadius);
    std::vector<float> radius;
    switch (pickRate(rdlCurveGeometry, getName(rdlCurveGeometry, attrRadius), rv.size(), rates)) {
    case AttributeRate::RATE_UNKNOWN:
    default:
        radius.assign(vertCount, 0.5f);
        break;
    case AttributeRate::RATE_CONSTANT:
        radius.assign(vertCount, rv[0]);
        break;
    case AttributeRate::RATE_UNIFORM: // per-curve
        radius.reserve(vertCount);
        for (size_t i = 0; i < vertexCounts.size(); ++i)
            for (size_t n = vertexCounts[i]; n--;)
                radius.emplace_back(rv[i]);
        break;
    case AttributeRate::RATE_PART: {
        const std::vector<scene_rdl2::rdl2::Int> &partIndices = rdlCurveGeometry->get(attrPartIndices);
        radius.reserve(vertCount);
        for (size_t i = 0; i < vertexCounts.size(); ++i)
            for (size_t n = vertexCounts[i]; n--;)
                radius.emplace_back(rv[partIndices[i]]);
        break;}
    case AttributeRate::RATE_VARYING: // per-segment
        switch (type) {
        case Curves::Type::LINEAR:
            radius.assign(rv.begin(), rv.begin() + vertCount);
            break;
        case Curves::Type::BEZIER: {
            radius.reserve(vertCount);
            size_t i = 0;
            for (size_t count : vertexCounts) {
                radius.emplace_back(rv[i]);
                for (size_t j = 0; j < count; j += 3) {
                    radius.emplace_back((rv[i]*2 + rv[i+1])/3);
                    radius.emplace_back((rv[i] + rv[i+1]*2)/3);
                    radius.emplace_back(rv[i+1]);
                    i++;
                }
            }
            break;}
        case Curves::Type::BSPLINE: {
            radius.reserve(vertCount);
            size_t i = 0;
            for (size_t count : vertexCounts) {
                radius.emplace_back(2*rv[i] - rv[i+1]);
                for (size_t j = 0; j < count-2; j++)
                    radius.emplace_back(rv[i++]);
                radius.emplace_back(2*rv[i-1] - rv[i-2]);
            }
            break;}
        default:
            break;
        }
        break;
    case AttributeRate::RATE_VERTEX:
        radius.assign(rv.begin(), rv.begin() + vertCount);
        break;
    }

    // Copy vertices, radius values are stored in the aligned channel
    Curves::VertexBuffer vertices(vertCount, numPosSamples);
    for (size_t i = 0; i < vertCount; i++) {
        const auto& p = procPosList0[i];
        vertices(i, 0) = Vec3fa(p[0], p[1], p[2], radius[i]);
    }
    if (numPosSamples == 2) {
        for (size_t i = 0; i < vertCount; i++) {
            const auto& p = procPosList1[i];
            vertices(i, 1) = Vec3fa(p[0], p[1], p[2], radius[i]);
        }
    }

    // Add velocity data
    if (numVelSamples > 0) {
        const float velocityScale = rdlCurveGeometry->get(attrVelocityScale);
        std::vector<std::vector<Vec3f>> velocities;
        velocities.push_back(std::vector<Vec3f>(procVelList0.begin(), procVelList0.end()));
        for (size_t i = 0; i < procVelList0.size(); i++) {
            velocities.back()[i] *= velocityScale;
        }
        if (numVelSamples == 2) {
            velocities.push_back(std::vector<Vec3f>(procVelList1.begin(), procVelList1.end()));
            for (size_t i = 0; i < procVelList1.size(); i++) {
                velocities.back()[i] *= velocityScale;
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

    // layer assignments
    LayerAssignmentId layerAssignmentId = createPerCurveAssignmentId(
        rdlCurveGeometry, rdlLayer, vertexCounts.size());

    // primitive attributes
    scene_rdl2::rdl2::PrimitiveAttributeFrame primitiveAttributeFrame =
        static_cast<scene_rdl2::rdl2::PrimitiveAttributeFrame>(rdlCurveGeometry->get(attrPrimitiveAttributeFrame));

    bool useFirstFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::SECOND_MOTION_STEP);
    bool useSecondFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::FIRST_MOTION_STEP);

    moonray::geom::processArbitraryData(rdlCurveGeometry,
                                        attrPrimitiveAttributes,
                                        primitiveAttributeTable,
                                        rates,
                                        useFirstFrame,
                                        useSecondFrame);

    // try to add UVs if we haven't already
    if (!primitiveAttributeTable.hasAttribute(shading::StandardAttributes::sUv)) {
        const rdl2::Vec2fVector& stList = rdlGeometry->get(attrUVs);
        if (!stList.empty()) {
            primitiveAttributeTable.addAttribute(shading::StandardAttributes::sUv,
                                                 pickRate(rdlGeometry,
                                                          getName(rdlGeometry, attrUVs),
                                                          stList.size(),
                                                          rates),
                                                 std::move(static_cast<std::vector<scene_rdl2::math::Vec2f>>(stList)));
        }
    }

    // Add explicit shading primitive attribute if explicit shading is enabled
    if (rdlGeometry->get(attrExplicitShading) &&
        !addExplicitShading(rdlGeometry, primitiveAttributeTable)) {
        return;
    }

    // Check the validity of the curves data and
    // print out any error messages
    std::string errorMessage;
    Primitive::DataValidness dataValid = Curves::checkPrimitiveData(type,
                                                                    subtype,
                                                                    tessellationRate,
                                                                    vertexCounts,
                                                                    vertices,
                                                                    primitiveAttributeTable,
                                                                    &errorMessage);

    if (dataValid != Primitive::DataValidness::VALID) {
        rdlGeometry->error(errorMessage);
        return;
    }

    std::unique_ptr<Curves> primitive = createCurves(type,
                                                     subtype,
                                                     tessellationRate,
                                                     std::move(vertexCounts),
                                                     std::move(vertices),
                                                     LayerAssignmentId(std::move(layerAssignmentId)),
                                                     std::move(primitiveAttributeTable));

    if (primitive) {
        primitive->setCurvedMotionBlurSampleCount(rdlCurveGeometry->get(attrCurvedMotionBlurSampleCount));

        // may need to convert the primitive to instance to handle
        // rotation motion blur
        std::unique_ptr<Primitive> p =
            convertForMotionBlur(generateContext,
                                 std::move(primitive),
                                 (rdlCurveGeometry->get(attrUseRotationMotionBlur) && parent2render.size() > 1));

        addPrimitive(std::move(p),
                     generateContext.getMotionBlurParams(),
                     parent2render);
    }
}

} // namespace geom
} // namespace moonray

moonray::geom::Procedural*
RdlCurveGeometry::createProcedural() const
{
    moonray::geom::State state;
    // Do not call state.setName here since scene_rdl2::rdl2::rdlLayer::assignmentId already
    // use rdlGeometry name.

    return new moonray::geom::RdlCurveProcedural(state);
}

void
RdlCurveGeometry::destroyProcedural() const
{
    delete mProcedural;
}
