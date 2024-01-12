// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/*
 * RdlMeshGeometry is a procedural that pretends to be a primitive.
 * This allows geometry data to be stored directly in the RDL
 * scene rather than requiring it be loaded from a container file.
 * It's bulky but useful.
 */

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/PrimitiveUserData.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <string>
#include <vector>
#include <numeric>

#include "attributes.cc"

RDL2_DSO_CLASS_BEGIN(RdlMeshGeometry, scene_rdl2::rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(RdlMeshGeometry)
    moonray::geom::Procedural* createProcedural() const;
    void destroyProcedural() const;

RDL2_DSO_CLASS_END(RdlMeshGeometry)

namespace {

const scene_rdl2::rdl2::String sDefaultPartName("");

moonray::geom::LayerAssignmentId
createPerFaceAssignmentId(const RdlMeshGeometry* rdlMeshGeometry,
                          const scene_rdl2::rdl2::Layer* rdlLayer,
                          const size_t faceCount) {
    // per face assignment id
    const int meshAssignmentId = rdlLayer->getAssignmentId(rdlMeshGeometry,
            sDefaultPartName);

    auto& partList = rdlMeshGeometry->get(attrPartList);
    if (!partList.empty()) {
        auto& partFaceCountList = rdlMeshGeometry->get(
                attrPartFaceCountList);
        auto& partFaceIndices = rdlMeshGeometry->get(attrPartFaceIndices);

        if (partList.size() == partFaceCountList.size()) {
            // validate size of face indices
            const int expectedFaces = std::accumulate(
                    partFaceCountList.begin(), partFaceCountList.end(), 0);
            if (expectedFaces == partFaceIndices.size()) {
                std::vector<int> faceAssignmentIds(faceCount,
                        meshAssignmentId);

                size_t begin = 0;
                for (size_t i = 0; i < partList.size(); ++i) {
                    const scene_rdl2::rdl2::String& partName = partList[i];
                    const int partFaceCount = partFaceCountList[i];
                    const int partAssignmentId = rdlLayer->getAssignmentId(
                            rdlMeshGeometry, partName);
                    for (size_t pF = begin; pF < begin + partFaceCount;
                            ++pF) {
                        const int faceIndex = partFaceIndices[pF];
                        faceAssignmentIds[faceIndex] = partAssignmentId;
                    }
                    begin += partFaceCount;
                }

                return moonray::geom::LayerAssignmentId(std::move(faceAssignmentIds));
            } else {
                rdlMeshGeometry->warn(
                        "total part face count does not match "
                                "part face indices count, skipping");
            }
        } else {
            rdlMeshGeometry->warn("part list is incorrect size for "
                    "partface count list, skipping");
        }
    }

    return moonray::geom::LayerAssignmentId(meshAssignmentId);
}

}  // anonymous namespace


namespace moonray {
namespace geom {

class RdlMeshProcedural : public ProceduralLeaf
{
public:
    RdlMeshProcedural(const State &state)
        : ProceduralLeaf(state)
        , mSubdMesh(nullptr)
        , mPolygonMesh(nullptr)
    {}
    
    void generate(const GenerateContext &generateContext,
            const shading::XformSamples &parent2render);

    void update(const UpdateContext &updateContext,
            const shading::XformSamples &parent2render);

private:
    VertexBuffer<Vec3fa, InterleavedTraits> getVertexData(
        const scene_rdl2::rdl2::Geometry* rdlGeometry,
        shading::PrimitiveAttributeTable &primitiveAttributeTable,
        const moonray::geom::RateCounts& rates
    );

    std::unique_ptr<SubdivisionMesh> createSubdMesh(
        const scene_rdl2::rdl2::Geometry* rdlGeometry,
        const scene_rdl2::rdl2::Layer* rdlLayer);

    std::unique_ptr<PolygonMesh> createPolyMesh(
        const scene_rdl2::rdl2::Geometry* rdlGeometry,
        const scene_rdl2::rdl2::Layer* rdlLayer);
    
    SubdivisionMesh* mSubdMesh;
    PolygonMesh* mPolygonMesh;

    static const std::string sPrimitiveName;
};

const std::string RdlMeshProcedural::sPrimitiveName("generated_mesh");


void
RdlMeshProcedural::generate(
        const GenerateContext &generateContext,
        const shading::XformSamples &parent2render)
{
    const scene_rdl2::rdl2::Geometry *rdlGeometry = generateContext.getRdlGeometry();

    const RdlMeshGeometry *rdlMeshGeometry =
        static_cast<const RdlMeshGeometry*>(rdlGeometry);

    const bool isSubd = rdlMeshGeometry->get(attrIsSubd);

    const scene_rdl2::rdl2::Layer *rdlLayer = generateContext.getRdlLayer();

    std::unique_ptr<Primitive> primitive;
    if (!isSubd) {
        primitive = createPolyMesh(rdlGeometry, rdlLayer);
    } else {
        primitive = createSubdMesh(rdlGeometry, rdlLayer);
    }

    if (primitive) {
        // may need to convert the primitive to instance to handle
        // rotation motion blur
        std::unique_ptr<Primitive> p = convertForMotionBlur(
            generateContext, std::move(primitive),
            (rdlMeshGeometry->get(attrUseRotationMotionBlur) &&
            parent2render.size() > 1));
        addPrimitive(std::move(p),
            generateContext.getMotionBlurParams(), parent2render);

    }
}


void
RdlMeshProcedural::update(
        const UpdateContext &updateContext,
        const shading::XformSamples &parent2render)
{
    const std::vector<const std::vector<float>*> &vertexDatas =
        updateContext.getMeshVertexDatas();

    shading::XformSamples prim2render = computePrim2Render(getState(), parent2render);
    if (mSubdMesh != nullptr) {
        mSubdMesh->updateVertexData(
            *vertexDatas[0], prim2render);
    } else {
        mPolygonMesh->updateVertexData(
            *vertexDatas[0], prim2render);
    }

    mDeformed = true;
}


VertexBuffer<Vec3fa, InterleavedTraits>
RdlMeshProcedural::getVertexData(
    const scene_rdl2::rdl2::Geometry* rdlGeometry,
    shading::PrimitiveAttributeTable &primitiveAttributeTable,
    const moonray::geom::RateCounts& rates)
{
    const RdlMeshGeometry* rdlMeshGeometry =
        static_cast<const RdlMeshGeometry*>(rdlGeometry);

    auto& procPosList0 = rdlMeshGeometry->get(attrPos0);
    auto& procPosList1 = rdlMeshGeometry->get(attrPos1);
    auto& procVelList0 = rdlMeshGeometry->get(attrVel0);
    auto& procVelList1 = rdlMeshGeometry->get(attrVel1);
    auto& procAccList0 = rdlMeshGeometry->get(attrAcc);

    const size_t vertCount  = procPosList0.size();

    bool pos1Valid = sizeCheck(rdlMeshGeometry, getName(rdlMeshGeometry, attrPos1), procPosList1.size(), vertCount);
    bool vel0Valid = sizeCheck(rdlMeshGeometry, getName(rdlMeshGeometry, attrVel0), procVelList0.size(), vertCount);
    bool vel1Valid = sizeCheck(rdlMeshGeometry, getName(rdlMeshGeometry, attrVel1), procVelList1.size(), vertCount);
    bool acc0Valid = sizeCheck(rdlMeshGeometry, getName(rdlMeshGeometry, attrAcc), procAccList0.size(), vertCount);

    // Fall back on static case if we don't have sufficient data for requested mb type
    int numPosSamples = 1;
    int numVelSamples = 0;
    int numAccSamples = 0;

    bool err = false;
    scene_rdl2::rdl2::MotionBlurType motionBlurType =
        static_cast<scene_rdl2::rdl2::MotionBlurType>(rdlMeshGeometry->get(attrMotionBlurType));

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
            numVelSamples =1;
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
        rdlMeshGeometry->warn("Insufficient data for requested motion blur type. "
                              "Falling back to static case.");
    }

    // Copy vertices
    VertexBuffer<Vec3fa, InterleavedTraits> vertices(vertCount, numPosSamples);
    for (size_t i = 0; i < vertCount; i++) {
        const auto& p = procPosList0[i];
        vertices(i, 0) = Vec3fa(p[0], p[1], p[2], 0.f);
    }
    if (numPosSamples == 2) {
        for (size_t i = 0; i < vertCount; i++) {
            const auto& p = procPosList1[i];
            vertices(i, 1) = Vec3fa(p[0], p[1], p[2], 0.f);
        }
    }

    // Add velocity data
    if (numVelSamples > 0) {
        const float velocityScale = rdlMeshGeometry->get(attrVelocityScale);
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
                                             shading::AttributeRate::RATE_VERTEX, std::move(velocities));
    }

    // Add acceleration data
    if (numAccSamples > 0) {
        std::vector<Vec3f> accelerations(procAccList0.begin(), procAccList0.end());
        primitiveAttributeTable.addAttribute(shading::StandardAttributes::sAcceleration,
                                             shading::AttributeRate::RATE_VERTEX, std::move(accelerations));
    }

    // Add UVs
    auto& procUVList = rdlMeshGeometry->get(attrUVs);
    if (!procUVList.empty()) {
        shading::AttributeRate attrRate(
            pickRate(rdlMeshGeometry, getName(rdlMeshGeometry, attrUVs), procUVList.size(), rates));
        std::vector<Vec2f> textureUV(procUVList.begin(), procUVList.end());
        primitiveAttributeTable.addAttribute(shading::StandardAttributes::sSurfaceST,
                                             attrRate, std::move(textureUV));
    }

    // Add normals
    auto& procNormalList = rdlMeshGeometry->get(attrNormals);
    if (!procNormalList.empty()) {
        shading::AttributeRate attrRate(
            pickRate(rdlMeshGeometry, getName(rdlMeshGeometry, attrNormals), procNormalList.size(), rates));
        std::vector<Vec3f> normals(procNormalList.begin(), procNormalList.end());
        primitiveAttributeTable.addAttribute(shading::StandardAttributes::sNormal,
                                             attrRate, std::move(normals));
    }

    return vertices;
}


std::unique_ptr<SubdivisionMesh>
RdlMeshProcedural::createSubdMesh(
    const scene_rdl2::rdl2::Geometry* rdlGeometry,
    const scene_rdl2::rdl2::Layer* rdlLayer)
{
    const RdlMeshGeometry* rdlMeshGeometry =
        static_cast<const RdlMeshGeometry*>(rdlGeometry);

    shading::PrimitiveAttributeTable primitiveAttributeTable;

    // Set the vert per face count
    auto& procFaceVertexCount = rdlMeshGeometry->get(attrFaceVertexCount);
    SubdivisionMesh::FaceVertexCount faceVertexCount(
        procFaceVertexCount.begin(), procFaceVertexCount.end());

    // Store the vert indices in order of face list to build the mesh
    auto& procIndices = rdlMeshGeometry->get(attrVertexIndex);
    SubdivisionMesh::IndexBuffer indices(procIndices.begin(), procIndices.end());
    if (indices.empty()) {
        return nullptr;
    }

    auto& procVertList = rdlMeshGeometry->get(attrPos0);
    const size_t vertCount = procVertList.size();
    const size_t faceCount = faceVertexCount.size();
    const size_t faceVaryingCount = procIndices.size();

    // Fill in table of faces->parts
    auto& partFaceCountList = rdlMeshGeometry->get(attrPartFaceCountList);
    const size_t partCount = partFaceCountList.size();
    auto& partFaceIndices = rdlMeshGeometry->get(attrPartFaceIndices);
    SubdivisionMesh::FaceToPartBuffer faceToPart(faceCount, 0);
    for (size_t p = 0, j = 0; p < partCount; p++) {
         for (int i = 0; i < partFaceCountList[p]; i++, j++) {
             faceToPart[partFaceIndices[j]] = p;
         }
    }
    if (faceVertexCount.empty()) {
        return nullptr;
    }
    const moonray::geom::RateCounts rates{partCount, faceCount, vertCount, vertCount, faceVaryingCount};

    // Get the vertices, velocities, uvs, normals etc.
    SubdivisionMesh::VertexBuffer vertices = getVertexData(rdlGeometry, primitiveAttributeTable, rates);
    if (vertices.empty()) {
        return nullptr;
    }

    // per face assignment id
    LayerAssignmentId layerAssignmentId =
        createPerFaceAssignmentId(rdlMeshGeometry, rdlLayer, faceCount);

    // get sidedness
    const int sideType = rdlMeshGeometry->getSideType();
    const bool singleSided = sideType == scene_rdl2::rdl2::Geometry::SINGLE_SIDED;

    // get resolution
    const int meshResolution = rdlMeshGeometry->get(attrMeshResolution);

    // get adaptive error (only used when adaptive tessellation got enabled)
    float adaptiveError = rdlMeshGeometry->get(attrAdaptiveError);
    // TODO rotation motion blur involves instancing logic that would
    // break adaptive tessellation right now.
    // Remove this switching logic once we have instancing supports
    // adaptive tessellation
    if (rdlMeshGeometry->get(attrUseRotationMotionBlur)) {
        adaptiveError = 0.0f;
    }

    // get subd scheme
    const int procScheme = rdlMeshGeometry->get(attrSubdScheme);
    SubdivisionMesh::Scheme scheme = SubdivisionMesh::Scheme::CATMULL_CLARK;
    if (SubdivisionMesh::isValidScheme(procScheme)) {
        // Values in the Rdl attribute align with those of the enum, so cast when valid:
        scheme = static_cast<SubdivisionMesh::Scheme>(procScheme);
    }

    // get arbitrary data
    scene_rdl2::rdl2::PrimitiveAttributeFrame primitiveAttributeFrame =
        static_cast<scene_rdl2::rdl2::PrimitiveAttributeFrame>(rdlMeshGeometry->get(attrPrimitiveAttributeFrame));

    bool useFirstFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::SECOND_MOTION_STEP);
    bool useSecondFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::FIRST_MOTION_STEP);

    moonray::geom::processArbitraryData(rdlMeshGeometry,
                                        attrPrimitiveAttributes,
                                        primitiveAttributeTable,
                                        rates,
                                        useFirstFrame,
                                        useSecondFrame);

    // Add explicit shading primitive attribute if explicit shading is enabled
    if (rdlGeometry->get(attrExplicitShading) &&
        !addExplicitShading(rdlGeometry, primitiveAttributeTable)) {
        return nullptr;
    }

    // build the primitive
    std::unique_ptr<SubdivisionMesh> primitive =
        createSubdivisionMesh(scheme,
                              std::move(faceVertexCount),
                              std::move(indices),
                              std::move(vertices),
                              std::move(layerAssignmentId),
                              std::move(primitiveAttributeTable));

    // set optional subdivision interpolation properties
    SubdivisionMesh::BoundaryInterpolation subdBoundary =
        SubdivisionMesh::BoundaryInterpolation::EDGE_AND_CORNER;
    SubdivisionMesh::FVarLinearInterpolation subdFVarLinear =
        SubdivisionMesh::FVarLinearInterpolation::CORNERS_ONLY;

    const bool useStudioDefaultSubdInterpolationOptions = false;
    if (!useStudioDefaultSubdInterpolationOptions) {
        const int procBoundary = rdlMeshGeometry->get(attrSubdBoundary);
        if (SubdivisionMesh::isValidBoundaryInterpolation(procBoundary)) {
            // Values in the Rdl attribute align with those of the enum, so cast when valid:
            subdBoundary = static_cast<SubdivisionMesh::BoundaryInterpolation>(procBoundary);
        } else {
            rdlGeometry->warn("Unknown subd boundary value, defaulting to 'edge and corner'.");
        }

        const int procFVarLinear = rdlMeshGeometry->get(attrSubdFVarLinear);
        if (SubdivisionMesh::isValidFVarLinearInterpolation(procFVarLinear)) {
            // Values in the Rdl attribute align with those of the enum, so cast when valid:
            subdFVarLinear = static_cast<SubdivisionMesh::FVarLinearInterpolation>(procFVarLinear);
        } else {
            rdlGeometry->warn("Unknown subd fvar linear value, defaulting to 'corners only'.");
        }
    }
    primitive->setSubdBoundaryInterpolation(subdBoundary);
    primitive->setSubdFVarLinearInterpolation(subdFVarLinear);

    // set optional subdivision creases, corners and holes:
    auto& procCreaseIndices = rdlMeshGeometry->get(attrSubdCreaseIndices);
    SubdivisionMesh::IndexBuffer creaseIndices(procCreaseIndices.begin(), procCreaseIndices.end());

    auto& procCreaseSharpnesses = rdlMeshGeometry->get(attrSubdCreaseSharpnesses);
    SubdivisionMesh::SharpnessBuffer creaseSharpnesses(procCreaseSharpnesses.begin(), procCreaseSharpnesses.end());

    auto& procCornerIndices = rdlMeshGeometry->get(attrSubdCornerIndices);
    SubdivisionMesh::IndexBuffer cornerIndices(procCornerIndices.begin(), procCornerIndices.end());

    auto& procCornerSharpnesses = rdlMeshGeometry->get(attrSubdCornerSharpnesses);
    SubdivisionMesh::SharpnessBuffer cornerSharpnesses(procCornerSharpnesses.begin(), procCornerSharpnesses.end());

    if (!creaseIndices.empty() && !creaseSharpnesses.empty()) {
        primitive->setSubdCreases(std::move(creaseIndices), std::move(creaseSharpnesses));
    }
    if (!cornerIndices.empty() && !cornerSharpnesses.empty()) {
        primitive->setSubdCorners(std::move(cornerIndices), std::move(cornerSharpnesses));
    }

    // set additional primitive attributes
    primitive->setMeshResolution(meshResolution);
    primitive->setAdaptiveError(adaptiveError);
    primitive->setName(sPrimitiveName);
    primitive->setIsSingleSided(singleSided);
    primitive->setIsNormalReversed(rdlMeshGeometry->getReverseNormals());
    primitive->setIsOrientationReversed(rdlMeshGeometry->get(attrOrientation) == ORIENTATION_LEFT_HANDED);
    primitive->setModifiability(Primitive::Modifiability::DEFORMABLE);
    primitive->setParts(partCount, std::move(faceToPart));
    primitive->setCurvedMotionBlurSampleCount(rdlMeshGeometry->get(attrCurvedMotionBlurSampleCount));

    mSubdMesh = primitive.get();
    return primitive;
}


std::unique_ptr<PolygonMesh>
RdlMeshProcedural::createPolyMesh(
    const scene_rdl2::rdl2::Geometry* rdlGeometry,
    const scene_rdl2::rdl2::Layer* rdlLayer)
{
    const RdlMeshGeometry* rdlMeshGeometry =
        static_cast<const RdlMeshGeometry*>(rdlGeometry);

    shading::PrimitiveAttributeTable primitiveAttributeTable;

    // Set the vert per face count
    auto& procFaceVertexCount = rdlMeshGeometry->get(attrFaceVertexCount);
    PolygonMesh::FaceVertexCount faceVertexCount(
        procFaceVertexCount.begin(), procFaceVertexCount.end());

    // Store the vert indices in order of face list to build the mesh
    auto& procIndices = rdlMeshGeometry->get(attrVertexIndex);
    PolygonMesh::IndexBuffer indices(procIndices.begin(), procIndices.end());
    if (indices.empty()) {
        return nullptr;
    }

    auto& procVertList = rdlMeshGeometry->get(attrPos0);
    const size_t vertCount = procVertList.size();
    const size_t faceCount = faceVertexCount.size();
    const size_t faceVaryingCount = procIndices.size();

    // Fill in table of faces->parts
    auto& partFaceCountList = rdlMeshGeometry->get(attrPartFaceCountList);
    const size_t partCount = partFaceCountList.size();
    auto& partFaceIndices = rdlMeshGeometry->get(attrPartFaceIndices);
    PolygonMesh::FaceToPartBuffer faceToPart(faceCount, 0);
    for (size_t p = 0, j = 0; p < partCount; p++) {
         for (int i = 0; i < partFaceCountList[p]; i++, j++) {
             faceToPart[partFaceIndices[j]] = p;
         }
    }
    if (faceVertexCount.empty()) {
        return nullptr;
    }
    const moonray::geom::RateCounts rates{partCount, faceCount, vertCount, vertCount, faceVaryingCount};

    // Get the vertices, velocities, uvs, normals etc.
    PolygonMesh::VertexBuffer vertices = getVertexData(rdlGeometry, primitiveAttributeTable, rates);
    if (vertices.empty()) {
        return nullptr;
    }

    // per face assignment id
    LayerAssignmentId layerAssignmentId =
            createPerFaceAssignmentId(rdlMeshGeometry, rdlLayer, faceVertexCount.size());

    // get sidedness
    const int sideType = rdlMeshGeometry->getSideType();
    const bool singleSided = sideType == scene_rdl2::rdl2::Geometry::SINGLE_SIDED;

    // get resolution
    const int meshResolution = rdlMeshGeometry->get(attrMeshResolution);

    // get adaptive error (only used when adaptive tessellation got enabled)
    float adaptiveError = rdlMeshGeometry->get(attrAdaptiveError);

    // get arbitrary data
    scene_rdl2::rdl2::PrimitiveAttributeFrame primitiveAttributeFrame =
        static_cast<scene_rdl2::rdl2::PrimitiveAttributeFrame>(rdlMeshGeometry->get(attrPrimitiveAttributeFrame));

    bool useFirstFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::SECOND_MOTION_STEP);
    bool useSecondFrame = (primitiveAttributeFrame != scene_rdl2::rdl2::PrimitiveAttributeFrame::FIRST_MOTION_STEP);

    moonray::geom::processArbitraryData(rdlMeshGeometry,
                                        attrPrimitiveAttributes,
                                        primitiveAttributeTable,
                                        rates,
                                        useFirstFrame,
                                        useSecondFrame);

    // Add explicit shading primitive attribute if explicit shading is enabled
    if (rdlGeometry->get(attrExplicitShading) &&
        !addExplicitShading(rdlGeometry, primitiveAttributeTable)) {
        return nullptr;
    }

    removeUnassignedFaces(rdlLayer,
                          layerAssignmentId,
                          faceToPart,
                          faceVertexCount,
                          indices,
                          &primitiveAttributeTable);
    
    // check if mesh is still valid
    if (faceVertexCount.empty() || indices.empty()) {
        // the mesh doesn't have any assigned materials,
        // skip generating the primitive
        return nullptr;
    }

    // build the primitive
    std::unique_ptr<PolygonMesh> primitive =
        createPolygonMesh(std::move(faceVertexCount),
                                    std::move(indices),
                                    std::move(vertices),
                                    std::move(layerAssignmentId),
                                    std::move(primitiveAttributeTable));

    // set additional primitive attributes
    primitive->setMeshResolution(meshResolution);
    primitive->setAdaptiveError(adaptiveError);
    primitive->setName(sPrimitiveName);
    primitive->setIsSingleSided(singleSided);
    primitive->setIsNormalReversed(rdlMeshGeometry->getReverseNormals());
    primitive->setIsOrientationReversed(rdlMeshGeometry->get(attrOrientation) == ORIENTATION_LEFT_HANDED);
    primitive->setParts(partCount, std::move(faceToPart));
    primitive->setSmoothNormal(rdlMeshGeometry->get(attrSmoothNormal));
    primitive->setCurvedMotionBlurSampleCount(rdlMeshGeometry->get(attrCurvedMotionBlurSampleCount));
    mPolygonMesh = primitive.get();
    return primitive;
}


} // namespace geom
} // namespace moonray

moonray::geom::Procedural*
RdlMeshGeometry::createProcedural() const
{
    moonray::geom::State state;
    // Do not call state.setName here since scene_rdl2::rdl2::rdlLayer::assignmentId already
    // use rdlGeometry name. 
        
    return new moonray::geom::RdlMeshProcedural(state);
}

void
RdlMeshGeometry::destroyProcedural() const
{
    delete mProcedural;
}
