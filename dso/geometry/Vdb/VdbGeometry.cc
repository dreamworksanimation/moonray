// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VdbGeometry.cc
///

#include <moonray/common/file_resource/file_resource.h>
#include <moonray/common/file_resource/FileResource.h>

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/shading/Shading.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <openvdb/io/GridDescriptor.h>

#include "attributes.cc"

RDL2_DSO_CLASS_BEGIN(VdbGeometry, scene_rdl2::rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(VdbGeometry)
    moonray::geom::Procedural* createProcedural() const;
    void destroyProcedural() const;

RDL2_DSO_CLASS_END(VdbGeometry)

//----------------------------------------------------------------------------

namespace moonray {
namespace geom {

std::string
getUniqueNameFromGridName(const std::string& gridName, const std::string& name)
{
    const openvdb::Name uniqueName = openvdb::io::GridDescriptor::stringAsUniqueName(gridName);
    if (uniqueName == gridName) {
        // there is no suffix
        return name;
    }

    // ASCII "record separator" character
    static constexpr char delimiter = 30;
    if (openvdb::io::GridDescriptor::stringAsUniqueName(name).find(delimiter) != std::string::npos) {
        // name already has a suffix
        return name;
    }
    int suffix = std::stoi(uniqueName.substr(uniqueName.find(delimiter) + 1, uniqueName.size()));

    return openvdb::io::GridDescriptor::nameAsString(openvdb::io::GridDescriptor::addSuffix(name, suffix));
}

std::string
getIndexedFile(const std::string& vdbFilePath,
               const scene_rdl2::rdl2::Geometry& rdlGeometry)
{
    std::string resolvedPath;
    const file_resource::FileResource* resource = file_resource::getFileResource(vdbFilePath);
    if (resource->supportsIndexing()) {
        const scene_rdl2::rdl2::SceneContext* sceneContext = rdlGeometry.getSceneClass().getSceneContext();
        int currentFrame = sceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sFrameKey);
        const file_resource::FileResource* subResource = resource->getIndexed(currentFrame);
        resolvedPath = subResource->userLabel();
    } else {
        resolvedPath = resource->userLabel();
    }

    return resolvedPath;
}

class OpenVdbProcedural : public ProceduralLeaf
{
public:
    OpenVdbProcedural(const State& state) : ProceduralLeaf(state) {}

    void generate(const GenerateContext& generateContext,
            const moonray::shading::XformSamples& parent2render)
    {
        const scene_rdl2::rdl2::Geometry* rdlGeometry = generateContext.getRdlGeometry();
        const VdbGeometry* pVdbGeometry =
            static_cast<const VdbGeometry*>(rdlGeometry);

        const std::string& vdbFilePath = pVdbGeometry->get(attrModel);
        if (vdbFilePath.empty()) {
            rdlGeometry->warn("No resource specified for reading");
            return;
        }
        const scene_rdl2::rdl2::Layer* rdlLayer = generateContext.getRdlLayer();
        int32_t layerAssignmentId;
        if (!getAssignmentId(rdlLayer, rdlGeometry, "", layerAssignmentId)) {
            rdlGeometry->error("unassigned primitive.");
            return;
        }

        // If the volume is bound to AmorphousVolumeShader, request VdbVolume
        // to initialize amorphous samplers
        moonray::shading::TypedAttributeKey<bool> amorphousKey("amorphous_meta_data");
        moonray::shading::PrimitiveAttributeTable primitiveAttributeTable;
        if (generateContext.requestAttribute(amorphousKey)) {
            primitiveAttributeTable.addAttribute(amorphousKey,
                moonray::shading::RATE_CONSTANT, std::vector<bool>{true});
        }

        const std::string resolvedVdbFilePath = getIndexedFile(vdbFilePath, *rdlGeometry);
        const std::string densityGridName = pVdbGeometry->get(attrDensityGrid);
        const std::string emissionGridName = pVdbGeometry->get(attrEmissionGrid);
        moonray::geom::VdbVolume::VdbInitData vdbInitData = {
                    resolvedVdbFilePath,
                    densityGridName,
                    emissionGridName,
                    getUniqueNameFromGridName(densityGridName, pVdbGeometry->get(attrVelocityGrid))};

        std::unique_ptr<VdbVolume> primitive = createVdbVolume(
            vdbInitData,
            generateContext.getMotionBlurParams(),
            LayerAssignmentId(layerAssignmentId),
            std::move(primitiveAttributeTable));
        if (primitive) {
            VdbVolume::Interpolation interpolationMode =
                moonray::geom::VdbVolume::Interpolation(
                pVdbGeometry->get(attrInterpolation));
            primitive->setInterpolation(interpolationMode);
            primitive->setVelocityScale(
                pVdbGeometry->get(attrVelocityScale));
            primitive->setVelocitySampleRate(
                pVdbGeometry->get(attrVelocitySampleRate));
            primitive->setEmissionSampleRate(
                pVdbGeometry->get(attrEmissionSampleRate));
            primitive->setName(densityGridName);
            addPrimitive(std::move(primitive),
                generateContext.getMotionBlurParams(), parent2render);
        }
    }
};

//----------------------------------------------------------------------------

} // namespace geom
} // namespace moonray


moonray::geom::Procedural* VdbGeometry::createProcedural() const
{
    moonray::geom::State state;
    return new moonray::geom::OpenVdbProcedural(state);
}


void VdbGeometry::destroyProcedural() const
{
    delete mProcedural;
}
