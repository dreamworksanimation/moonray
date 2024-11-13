// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Api.cc
/// $Id$
///

#include "Api.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <scene_rdl2/render/util/stdmemory.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/Layer.h>

#include <exception>
#include <numeric>

using namespace scene_rdl2;
namespace moonray {
namespace geom {

class PrimitiveTypeChecker : public PrimitiveVisitor
{
public:
    PrimitiveTypeChecker(const Primitive& p): mType(PrimitiveType::UNKNOWN) {
        // we know we are not going to modify the primitive here
        const_cast<Primitive&>(p).accept(*this);
    }

    PrimitiveType getType() const { return mType; }

    virtual void visitPrimitive(Primitive& p) override {
        MNRY_ASSERT_REQUIRE(false);
    }

    virtual void visitCurves(Curves& c) override {
        mType = PrimitiveType::CURVES;
    }

    virtual void visitInstance(Instance& i) override {
        mType = PrimitiveType::INSTANCE;
    }

    virtual void visitPoints(Points& p) override {
        mType = PrimitiveType::POINTS;
    }

    virtual void visitPolygonMesh(PolygonMesh& p) override {
        mType = PrimitiveType::POLYGON_MESH;
    }

    virtual void visitPrimitiveGroup(PrimitiveGroup& pg) override {
        mType = PrimitiveType::PRIMITIVE_GROUP;
    }

    virtual void visitSphere(Sphere& s) override {
        mType = PrimitiveType::SPHERE;
    }

    virtual void visitBox(Box& b) override {
        mType = PrimitiveType::BOX;
    }

    virtual void visitSubdivisionMesh(SubdivisionMesh& s) override {
        mType = PrimitiveType::SUBDIVISION_MESH;
    }

    virtual void visitTransformedPrimitive(TransformedPrimitive& t) override {
        mType = PrimitiveType::TRANSFORMED_PRIMITIVE;
    }

    virtual void visitVdbVolume(VdbVolume& v) override {
        mType = PrimitiveType::VDB_VOLUME;
    }

private:
    PrimitiveType mType;
};

PrimitiveType
getPrimitiveType(const Primitive& p)
{
    return PrimitiveTypeChecker(p).getType();
}

bool
validateVertexBuffer3fa(const VertexBuffer<Vec3fa, InterleavedTraits>& vertices,
                        const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    bool isValid = true;
    for (size_t v = 0; v < vertices.size(); ++v) {
        for (size_t t = 0; t < vertices.get_time_steps(); ++t) {
            if (std::isnan(vertices(v, t).x) ||
                std::isnan(vertices(v, t).y) ||
                std::isnan(vertices(v, t).z) ||
                std::isnan(vertices(v, t).w)) {
                isValid = false;
                rdlGeometry->error("Vertex data has a nan value at index (", v, ", ", t, ")");
            } else if (!std::isfinite(vertices(v, t).x) ||
                       !std::isfinite(vertices(v, t).y) ||
                       !std::isfinite(vertices(v, t).z) ||
                       !std::isfinite(vertices(v, t).w)) {
                isValid = false;
                rdlGeometry->error("Vertex data has a infinite value at index (", v, ", ", t, ")");
            }
        }
    }
    return isValid;
}

bool
validateVertexBuffer3f(const VertexBuffer<Vec3f, InterleavedTraits>& vertices,
                       const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    bool isValid = true;
    for (size_t v = 0; v < vertices.size(); ++v) {
        for (size_t t = 0; t < vertices.get_time_steps(); ++t) {
            if (std::isnan(vertices(v, t).x) ||
                std::isnan(vertices(v, t).y) ||
                std::isnan(vertices(v, t).z)) {
                isValid = false;
                rdlGeometry->error("Vertex data has a nan value at index (", v, ", ", t, ")");
            } else if (!std::isfinite(vertices(v, t).x) ||
                       !std::isfinite(vertices(v, t).y) ||
                       !std::isfinite(vertices(v, t).z)) {
                isValid = false;
                rdlGeometry->error("Vertex data has a infinite value at index (", v, ", ", t, ")");
            }
        }
    }
    return isValid;
}

template <typename T> bool
validateAttr(const T& attr,
             const std::string& attrName,
             const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (std::isnan(attr)) {
        rdlGeometry->error("Primitive attribute ", attrName," has a nan value");
        return false;
    } else if (!std::isfinite(attr)) {
        rdlGeometry->error("Primitive attribute ", attrName," has a infinite value");
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Vec2f>(const scene_rdl2::rdl2::Vec2f& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Float>(attr.x, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.y, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Vec3f>(const scene_rdl2::rdl2::Vec3f& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Float>(attr.x, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.y, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.z, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Vec4f>(const scene_rdl2::rdl2::Vec4f& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Float>(attr.x, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.y, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.z, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.w, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Vec2d>(const scene_rdl2::rdl2::Vec2d& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Double>(attr.x, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Double>(attr.y, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Vec3d>(const scene_rdl2::rdl2::Vec3d& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Double>(attr.x, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Double>(attr.y, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Double>(attr.z, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Vec4d>(const scene_rdl2::rdl2::Vec4d& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Double>(attr.x, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Double>(attr.y, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Double>(attr.z, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Double>(attr.w, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Rgb>(const scene_rdl2::rdl2::Rgb& attr,
                                    const std::string& attrName,
                                    const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Float>(attr.r, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.g, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.b, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Rgba>(const scene_rdl2::rdl2::Rgba& attr,
                                     const std::string& attrName,
                                     const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    if (!validateAttr<scene_rdl2::rdl2::Float>(attr.r, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.g, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.b, attrName, rdlGeometry) ||
        !validateAttr<scene_rdl2::rdl2::Float>(attr.a, attrName, rdlGeometry)) {
        return false;
    } else {
        return true;
    }
}

template <> bool
validateAttr<scene_rdl2::rdl2::Mat4f>(const scene_rdl2::rdl2::Mat4f& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    bool isValid = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (!validateAttr<scene_rdl2::rdl2::Float>(attr[i][j], attrName, rdlGeometry)) {
                isValid = false;
            }
        }
    }
    return isValid;
}

template <> bool
validateAttr<scene_rdl2::rdl2::Mat4d>(const scene_rdl2::rdl2::Mat4d& attr,
                                      const std::string& attrName,
                                      const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    bool isValid = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (!validateAttr<scene_rdl2::rdl2::Double>(attr[i][j], attrName, rdlGeometry)) {
                isValid = false;
            }
        }
    }
    return isValid;
}

template <typename T> bool
validateAttrs(const shading::PrimitiveAttributeTable& pat,
              const shading::AttributeKey& key,
              const float timeSample,
              const scene_rdl2::rdl2::Geometry* rdlGeometry) {
    const shading::PrimitiveAttribute<T> &attr =
        pat.getAttribute(shading::TypedAttributeKey<T>(key), timeSample);
    for (size_t i = 0; i < attr.size(); ++i) {
        if (!validateAttr<T>(attr[i], key.getName(), rdlGeometry)) {
            return false;
        }
    }
    return true;
}

bool
validatePrimitiveAttributeTable(const shading::PrimitiveAttributeTable& pat,
                                const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    bool isValid = true;
    for (const auto& p : pat) {

        const shading::AttributeKey& key = p.first;
        const size_t motionSamplesCount = pat.getTimeSampleCount(key);

        for (size_t t = 0; t < motionSamplesCount; ++t) {

            switch (key.getType()) {
            case scene_rdl2::rdl2::TYPE_BOOL:
                validateAttrs<scene_rdl2::rdl2::Bool>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_INT:
                validateAttrs<scene_rdl2::rdl2::Int>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_LONG:
                validateAttrs<scene_rdl2::rdl2::Long>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_FLOAT:
                validateAttrs<scene_rdl2::rdl2::Float>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_DOUBLE:
                validateAttrs<scene_rdl2::rdl2::Double>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2F:
                validateAttrs<scene_rdl2::rdl2::Vec2f>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3F:
                validateAttrs<scene_rdl2::rdl2::Vec3f>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_VEC4F:
                validateAttrs<scene_rdl2::rdl2::Vec4f>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2D:
                validateAttrs<scene_rdl2::rdl2::Vec2d>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3D:
                validateAttrs<scene_rdl2::rdl2::Vec3d>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_VEC4D:
                validateAttrs<scene_rdl2::rdl2::Vec4d>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_RGB:
                validateAttrs<scene_rdl2::rdl2::Rgb>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_RGBA:
                validateAttrs<scene_rdl2::rdl2::Rgba>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_MAT4F:
                validateAttrs<scene_rdl2::rdl2::Mat4f>(pat, key, t, rdlGeometry);
                break;
            case scene_rdl2::rdl2::TYPE_MAT4D:
                validateAttrs<scene_rdl2::rdl2::Mat4d>(pat, key, t, rdlGeometry);
                break;
            }
        }
    }

    return isValid;
}

std::unique_ptr<Curves>
createCurves(Curves::Type type,
             Curves::SubType subtype,
             int tessellationRate,
             Curves::CurvesVertexCount&& curvesVertexCount,
             Curves::VertexBuffer&& vertices,
             LayerAssignmentId&& layerAssignmentId,
             shading::PrimitiveAttributeTable&& primitiveAttributeTable,
             bool validateGeometry,
             const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    if (type == Curves::Type::UNKNOWN || subtype == Curves::SubType::UNKNOWN) {
        return nullptr;
    }

    if (subtype == geom::Curves::SubType::NORMAL_ORIENTED) {
        if (!primitiveAttributeTable.hasAttribute(shading::StandardAttributes::sNormal)) {
            rdlGeometry->error("The curves subtype is set to \"normal_oriented\" but ",
                               " the curves are missing the required ",
                               shading::StandardAttributes::sNormal,
                               " primitive attribute");
            return nullptr;
        }

        const shading::PrimitiveAttribute<scene_rdl2::math::Vec3f>& normalAttr =
            primitiveAttributeTable.getAttribute<scene_rdl2::math::Vec3f>(shading::StandardAttributes::sNormal);

        shading::AttributeRate rate =  normalAttr.getRate();
        if (rate != shading::AttributeRate::RATE_CONSTANT &&
            rate != shading::AttributeRate::RATE_UNIFORM &&
            rate != shading::AttributeRate::RATE_VARYING &&
            rate != shading::AttributeRate::RATE_VERTEX) {

            rdlGeometry->error("The curves subtype is set to \"normal_oriented\" but ",
                               " the ", shading::StandardAttributes::sNormal,
                               " primitive attribute has an invalid rate: ", rate);
            return nullptr;

        }

        // Linear normal oriented curves are not currently supported in embree.
        // Use ray facing instead.
        if (type == Curves::Type::LINEAR) {
            rdlGeometry->warn("Linear curves do not currently support the normal ",
                              "oriented style, defaulting to ray facing.");
            subtype = Curves::SubType::RAY_FACING;
        }
    }

    if (validateGeometry) {
        if (!validateVertexBuffer3fa(vertices, rdlGeometry)) {
            return nullptr;
        }
        if (!validatePrimitiveAttributeTable(primitiveAttributeTable, rdlGeometry)) {
            return nullptr;
        }
    }

    return fauxstd::make_unique<Curves>(type,
                                        subtype,
                                        tessellationRate,
                                        std::move(curvesVertexCount),
                                        std::move(vertices),
                                        std::move(layerAssignmentId),
                                        std::move(primitiveAttributeTable));
}

std::unique_ptr<Points>
createPoints(Points::VertexBuffer&& position,
             Points::RadiusBuffer&& radius,
             LayerAssignmentId&& layerAssignmentId,
             shading::PrimitiveAttributeTable&& primitiveAttributeTable,
             bool validateGeometry,
             const scene_rdl2::rdl2::Geometry* rdlGeometry)
{ 
    if (validateGeometry) {
        if (!validateVertexBuffer3f(position, rdlGeometry)) {
            return nullptr;
        }
        if (!validatePrimitiveAttributeTable(primitiveAttributeTable, rdlGeometry)) {
            return nullptr;
        }
    }

    return fauxstd::make_unique<Points>(std::move(position),
                                        std::move(radius),
                                        std::move(layerAssignmentId),
                                        std::move(primitiveAttributeTable));
}

std::unique_ptr<PolygonMesh>
createPolygonMesh(PolygonMesh::FaceVertexCount&& faceVertexCount,
                  PolygonMesh::IndexBuffer&& indices,
                  PolygonMesh::VertexBuffer&& vertices,
                  LayerAssignmentId&& layerAssignmentId,
                  shading::PrimitiveAttributeTable&& primitiveAttributeTable,
                  bool validateGeometry,
                  const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    if (validateGeometry) {
        if (!validateVertexBuffer3fa(vertices, rdlGeometry)) {
            return nullptr;
        }
        if (!validatePrimitiveAttributeTable(primitiveAttributeTable, rdlGeometry)) {
            return nullptr;
        }
    }

    return fauxstd::make_unique<PolygonMesh>(std::move(faceVertexCount),
                                             std::move(indices),
                                             std::move(vertices),
                                             std::move(layerAssignmentId),
                                             std::move(primitiveAttributeTable));
}

std::unique_ptr<VdbVolume>
createVdbVolume(const VdbVolume::VdbInitData& vdbInitData,
                const MotionBlurParams& motionBlurParams,
                LayerAssignmentId&& layerAssignmentId,
                shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    return fauxstd::make_unique<VdbVolume>(vdbInitData,
                                           motionBlurParams,
                                           std::move(layerAssignmentId),
                                           std::move(primitiveAttributeTable));
}

std::unique_ptr<PrimitiveGroup>
createPrimitiveGroup()
{
    return fauxstd::make_unique<PrimitiveGroup>();
}

std::unique_ptr<Sphere>
createSphere(float radius,
             LayerAssignmentId&& layerAssignmentId,
             shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    return fauxstd::make_unique<Sphere>(radius,
                                        std::move(layerAssignmentId),
                                        std::move(primitiveAttributeTable));
}

std::unique_ptr<Box>
createBox(float length,
          float width,
          float height,
          LayerAssignmentId&& layerAssignmentId,
          shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    return fauxstd::make_unique<Box>(length,
                                     width,
                                     height,
                                     std::move(layerAssignmentId),
                                     std::move(primitiveAttributeTable));
}

std::unique_ptr<SubdivisionMesh>
createSubdivisionMesh(SubdivisionMesh::Scheme scheme,
                      SubdivisionMesh::FaceVertexCount&& faceVertexCount,
                      SubdivisionMesh::IndexBuffer&& indices,
                      SubdivisionMesh::VertexBuffer&& vertices,
                      LayerAssignmentId&& layerAssignmentId,
                      shading::PrimitiveAttributeTable&& primitiveAttributeTable,
                      bool validateGeometry,
                      const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    if (validateGeometry) {
        if (!validateVertexBuffer3fa(vertices, rdlGeometry)) {
            return nullptr;
        }
        if (!validatePrimitiveAttributeTable(primitiveAttributeTable, rdlGeometry)) {
            return nullptr;
        }
    }

    return fauxstd::make_unique<SubdivisionMesh>(scheme,
                                                 std::move(faceVertexCount),
                                                 std::move(indices),
                                                 std::move(vertices),
                                                 std::move(layerAssignmentId),
                                                 std::move(primitiveAttributeTable));
}

std::unique_ptr<Instance>
createInstance(const Mat43& xform,
               std::shared_ptr<SharedPrimitive> reference,
               shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    return fauxstd::make_unique<Instance>(shading::XformSamples({xform}),
                                          reference,
                                          std::move(primitiveAttributeTable));
}

std::unique_ptr<Instance>
createInstance(const shading::XformSamples& xform,
               std::shared_ptr<SharedPrimitive> reference,
               shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    return fauxstd::make_unique<Instance>(xform,
                                          reference,
                                          std::move(primitiveAttributeTable));
}

std::shared_ptr<SharedPrimitive>
createSharedPrimitive(std::unique_ptr<Primitive> primitive)
{
    return std::make_shared<SharedPrimitive>(std::move(primitive));
}

std::unique_ptr<TransformedPrimitive>
createTransformedPrimitive(const Mat43& xform,
                           std::unique_ptr<Primitive> primitive)
{
    return fauxstd::make_unique<TransformedPrimitive>(shading::XformSamples({xform}),
                                                      std::move(primitive));
}

std::unique_ptr<TransformedPrimitive>
createTransformedPrimitive(const shading::XformSamples& xform,
                           std::unique_ptr<Primitive> primitive)
{
    return fauxstd::make_unique<TransformedPrimitive>(xform,
                                                      std::move(primitive));
}

static void
removeUnassignedFaces(const scene_rdl2::rdl2::Layer* layer,
                      int constAssignmentId,
                      std::vector<Primitive::IndexType>& faceToPart,
                      std::vector<Primitive::size_type>& faceVertexCount,
                      std::vector<Primitive::IndexType>& indices,
                      shading::PrimitiveAttributeTable* primitiveAttributeTable)
{
    bool hasValidAssignment = constAssignmentId != -1 &&
        (layer->lookupMaterial(constAssignmentId) != nullptr ||
        layer->lookupVolumeShader(constAssignmentId) != nullptr);
    // the whole mesh is a part and it's not assigned
    if (!hasValidAssignment) {
        faceToPart.resize(0);
        faceVertexCount.resize(0);
        indices.resize(0);
        if (primitiveAttributeTable) {
            std::vector<shading::AttributeKey> unusedAttributes;
            for (auto& kv : *primitiveAttributeTable) {
                shading::AttributeRate rate =
                    primitiveAttributeTable->getRate(kv.first);
                if (rate == shading::RATE_UNIFORM || rate == shading::RATE_FACE_VARYING) {
                    unusedAttributes.push_back(kv.first);
                }
            }
            for (auto& key : unusedAttributes) {
                primitiveAttributeTable->erase(key);
            }
        }
    }
}

static void
removeUnassignedFaces(const scene_rdl2::rdl2::Layer* layer,
                      std::vector<int>& varyingAssignmentId,
                      std::vector<Primitive::IndexType>& faceToPart,
                      std::vector<Primitive::size_type>& faceVertexCount,
                      std::vector<Primitive::IndexType>& indices,
                      shading::PrimitiveAttributeTable* primitiveAttributeTable)
{
    size_t faceCount = faceVertexCount.size();
    MNRY_ASSERT(varyingAssignmentId.size() == faceCount);
    size_t totalFaceVertices = std::accumulate(
        faceVertexCount.begin(), faceVertexCount.end(), 0);
    MNRY_ASSERT(indices.size() == totalFaceVertices);

    std::vector<int> newFaceIndexes(faceCount, -1);
    std::vector<int> newFaceVertexIndexes(totalFaceVertices, -1);
    // calculate index mapping from input face list to shrinked face list
    // and mapping from input face vertex list to shrinked face vertex list
    size_t currentFaceVertex = 0;
    size_t assignedFaceCount = 0;
    size_t assignedFaceVertices = 0;
    for (size_t i = 0; i < faceCount; ++i) {
        bool hasValidAssignment = varyingAssignmentId[i] != -1 &&
            (layer->lookupMaterial(varyingAssignmentId[i]) != nullptr ||
            layer->lookupVolumeShader(varyingAssignmentId[i]) != nullptr);
        if (hasValidAssignment) {
            newFaceIndexes[i] = assignedFaceCount++;
            for (size_t j = 0; j < faceVertexCount[i]; ++j) {
                newFaceVertexIndexes[currentFaceVertex + j] =
                    assignedFaceVertices++;
            }
        }
        currentFaceVertex += faceVertexCount[i];
    }
    // remove the unassigned face
    for (size_t i = 0; i < faceCount; ++i) {
        if (newFaceIndexes[i] != -1) {
            varyingAssignmentId[newFaceIndexes[i]] = varyingAssignmentId[i];
            if (faceToPart.size() > 0) {
                faceToPart[newFaceIndexes[i]] = faceToPart[i];
            }
            faceVertexCount[newFaceIndexes[i]] = faceVertexCount[i];
        }
    }
    if (faceToPart.size() > 0) {
        faceToPart.resize(assignedFaceCount);
    }
    varyingAssignmentId.resize(assignedFaceCount);
    faceVertexCount.resize(assignedFaceCount);
    // remove the unassigned face vertex
    for (size_t i = 0; i < totalFaceVertices; ++i) {
        if (newFaceVertexIndexes[i] != -1) {
            indices[newFaceVertexIndexes[i]] = indices[i];
        }
    }
    indices.resize(assignedFaceVertices);
    // we are done if there is no primitive attributes to process
    if (!primitiveAttributeTable) {
        return;
    }
    // collect the uniform/facevarying attributes that need to process
    std::vector<shading::AttributeKey> uniformKeys;
    std::vector<shading::AttributeKey> faceVaryingKeys;
    for (auto& kv : *primitiveAttributeTable) {
        shading::AttributeRate rate =
            primitiveAttributeTable->getRate(kv.first);
        if (rate == shading::RATE_UNIFORM) {
            uniformKeys.push_back(kv.first);
        } else if (rate == shading::RATE_FACE_VARYING) {
            faceVaryingKeys.push_back(kv.first);
        }
    }
    // shrink the uniform primitive attributes
    for (const auto& key : uniformKeys) {
        auto& attribute = primitiveAttributeTable->find(key)->second;
        for (size_t t = 0; t < attribute.size(); ++t) {
            // MOONRAY-3803: Same fix as below, the size may be greater than
            // the faceCount (it will be less than facevarying count)
            MNRY_ASSERT(attribute[t]->size() >= faceCount);
            for (size_t f = 0; f < faceCount; ++f) {
                if (newFaceIndexes[f] != -1) {
                    attribute[t]->copyInPlace(f, newFaceIndexes[f]);
                }
            }
            attribute[t]->resize(assignedFaceCount);
        }
    }
    // shrink the facevarying primitive attributes
    for (const auto& key : faceVaryingKeys) {
        auto& attribute = primitiveAttributeTable->find(key)->second;
        for (size_t t = 0; t < attribute.size(); ++t) {
            // MOONRAY-3803: if the primvar size is larger than needed, the
            // face-varying interpolation is selected, but the array still stays the
            // larger size. Do not use the size(), stop at the expected count.
            MNRY_ASSERT(attribute[t]->size() >= totalFaceVertices);
            for (size_t fv = 0; fv < totalFaceVertices; ++fv) {
                if (newFaceVertexIndexes[fv] != -1) {
                    attribute[t]->copyInPlace(fv, newFaceVertexIndexes[fv]);
                }
            }
            attribute[t]->resize(assignedFaceVertices);
        }
    }
}

std::unique_ptr<Primitive>
convertForMotionBlur(const ProceduralContext& context,
                     std::unique_ptr<Primitive> p, bool useRotationMotionBlur)
{
    if (context.isMotionBlurOn() && useRotationMotionBlur&&
        getPrimitiveType(*p) != PrimitiveType::INSTANCE) {
        // only instance primitive can handle rotation motion blur well
        return createInstance(Mat43(scene_rdl2::math::one),
            createSharedPrimitive(std::move(p)));
    } else {
        return p;
    }
}

void
removeUnassignedFaces(const scene_rdl2::rdl2::Layer* layer,
                      LayerAssignmentId& layerAssignmentId,
                      std::vector<Primitive::IndexType>& faceToPart,
                      std::vector<Primitive::size_type>& faceVertexCount,
                      std::vector<Primitive::IndexType>& indices,
                      shading::PrimitiveAttributeTable* primitiveAttributeTable)
{
    if (layerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT) {
        removeUnassignedFaces(layer, layerAssignmentId.getConstId(),
            faceToPart, faceVertexCount, indices, primitiveAttributeTable);
    } else {
        removeUnassignedFaces(layer, layerAssignmentId.getVaryingId(),
            faceToPart, faceVertexCount, indices, primitiveAttributeTable);
    }
}

bool
getAssignmentId(const scene_rdl2::rdl2::Layer* layer, const scene_rdl2::rdl2::Geometry* geometry,
                const std::string& partName, int& assignmentId)
{
    assignmentId = layer->getAssignmentId(geometry, partName);
    return assignmentId != -1 &&
        (layer->lookupMaterial(assignmentId) != nullptr ||
         layer->lookupVolumeShader(assignmentId) != nullptr);
}

bool
validateExplicitShading(const shading::PrimitiveAttributeTable& pat,
                        const scene_rdl2::rdl2::Geometry* rdlGeometry)
{
    bool hasExplicitAttributes = true;

    if (!pat.hasAttribute(shading::StandardAttributes::sNormal)) {
        rdlGeometry->error("Missing normal explicit shading primitive attribute");
        hasExplicitAttributes = false;
    }
    if (!pat.hasAttribute(shading::StandardAttributes::sdPds)) {
        rdlGeometry->error("Missing dPds explicit shading primitive attribute");
        hasExplicitAttributes = false;
    }
    if (!pat.hasAttribute(shading::StandardAttributes::sdPdt)) {
        rdlGeometry->error("Missing dPdt explicit shading primitive attribute ");
        hasExplicitAttributes = false;
    }

    return hasExplicitAttributes;
}

bool
addExplicitShading(const scene_rdl2::rdl2::Geometry* rdlGeometry,
                   shading::PrimitiveAttributeTable& pat)
{
    if (validateExplicitShading(pat,
                                rdlGeometry)) {
        std::vector<bool> data = { true };
        pat.addAttribute(shading::StandardAttributes::sExplicitShading,
                         shading::AttributeRate::RATE_CONSTANT,
                         std::move(data));
        return true;
    } else {
        return false;
    }
}

void
getScreenSpaceRadiusAttr(const shading::TypedAttributeKey<float> key,
                         const int numVerts,
                         const Curves::CurvesVertexCount& vertexCounts,
                         const shading::PrimitiveAttributeTable& pat,
                         std::vector<float>& vals)
{
    vals.resize(numVerts, 1.0f);
    if (pat.hasAttribute(key)) {
        const shading::PrimitiveAttribute<float>& attrVals = pat.getAttribute(key, 0);
        switch (pat.getRate(key)) {
        case shading::AttributeRate::RATE_CONSTANT:
            std::fill(vals.begin(), vals.end(), attrVals[0]);
            break;
        case shading::AttributeRate::RATE_UNIFORM:
        {
            size_t i = 0;
            for (size_t c = 0; c < vertexCounts.size(); ++c) {
                const float val = attrVals[c];
                for (size_t v = 0; v < vertexCounts[c]; ++v) {
                    vals[i] = val;
                    ++i;
                }
            }
            break;
        }
        case shading::AttributeRate::RATE_VARYING:
        case shading::AttributeRate::RATE_VERTEX:
            for (size_t i = 0; i < numVerts; ++i) {
                vals[i] = attrVals[i];
            }
            break;
        }
    }
}

bool
getCameraPositionAndFocal(const rdl2::Geometry* geometry,
                          math::Vec3f& cameraPosition,
                          float& cameraFocal)
{
    // Get camera position and focal
    const rdl2::Camera* primaryCamera = geometry->getSceneClass().getSceneContext()->getPrimaryCamera();
    try {
        cameraFocal = primaryCamera->get<float>("focal");
    } catch (const except::KeyError&) {
        Logger::error("The camera does not have a \"focal\" parameter. Disabling screen space radius.");
        return false;
    }
    const math::Mat4d m = primaryCamera->get(rdl2::Node::sNodeXformKey, rdl2::TIMESTEP_BEGIN);
    cameraPosition = math::Vec3f(m.row3().x, m.row3().y, m.row3().z);
    return true;
}

float
applyScreenSpaceRadius(const math::Vec3f& position,
                       const math::Vec3f& cameraPosition,
                       const float cameraFocal,
                       const float minDist,
                       const float maxDist,
                       const float minDistRadius,
                       const float maxDistRadius,
                       const float radius)
{
    const float distance = math::distance(position, cameraPosition);
    const float distT = math::clamp((distance - minDist) / (maxDist - minDist), 0.0f, 1.0f);
    const float distFactor = (minDistRadius * (1.0f - distT)) + (maxDistRadius * distT);
    return math::max(0.f, radius * (distance / cameraFocal) * distFactor);
}

void
applyScreenSpaceRadius(const rdl2::Geometry* geometry,
                       const Points::VertexBuffer& vertices,
                       const shading::PrimitiveAttributeTable& pat,
                       Points::RadiusBuffer& radiusBuffer,
                       const moonray::shading::XformSamples& parent2root)
{
    float cameraFocal;
    math::Vec3f cameraPosition;
    if (!getCameraPositionAndFocal(geometry, cameraPosition, cameraFocal)) {
        return;
    }

    // Get screen space primitive attributes if they exist
    const size_t numVerts = vertices.size();
    std::vector<float> minDist, maxDist, minDistRadius, maxDistRadius;
    Curves::CurvesVertexCount vertexCounts;
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMinDistance, numVerts, vertexCounts, pat, minDist);
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMaxDistance, numVerts, vertexCounts, pat, maxDist);
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMinDistanceRadius, numVerts, vertexCounts, pat, minDistRadius);
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMaxDistanceRadius, numVerts, vertexCounts, pat, maxDistRadius);

    const math::Mat4f nodeXform = math::toFloat(geometry->get(rdl2::Node::sNodeXformKey,
                                                rdl2::TIMESTEP_BEGIN));

    for (size_t i = 0; i < radiusBuffer.size(); ++i) {
        math::Vec3f position(vertices(i, 0).x, vertices(i, 0).y, vertices(i, 0).z);
        position = math::transformPoint(parent2root[0], position);
        position = math::transformPoint(nodeXform, position);
        radiusBuffer[i] = applyScreenSpaceRadius(position,
                                                 cameraPosition,
                                                 cameraFocal,
                                                 minDist[i], maxDist[i],
                                                 minDistRadius[i], maxDistRadius[i],
                                                 radiusBuffer[i]);
    }
}

void
applyScreenSpaceRadius(const scene_rdl2::rdl2::Geometry* geometry,
                       Curves::VertexBuffer& vertices,
                       const Curves::CurvesVertexCount& vertexCounts,
                       const shading::PrimitiveAttributeTable& pat,
                       const moonray::shading::XformSamples& parent2root)
{
    float cameraFocal;
    math::Vec3f cameraPosition;
    if (!getCameraPositionAndFocal(geometry, cameraPosition, cameraFocal)) {
        return;
    }

    // Get screen space primitive attributes if they exist
    const size_t numVerts = vertices.size();
    std::vector<float> minDist, maxDist, minDistRadius, maxDistRadius;
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMinDistance, numVerts, vertexCounts, pat, minDist);
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMaxDistance, numVerts, vertexCounts, pat, maxDist);
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMinDistanceRadius, numVerts, vertexCounts, pat, minDistRadius);
    getScreenSpaceRadiusAttr(shading::StandardAttributes::sMaxDistanceRadius, numVerts, vertexCounts, pat, maxDistRadius);

    const math::Mat4f nodeXform = math::toFloat(geometry->get(rdl2::Node::sNodeXformKey,
                                                rdl2::TIMESTEP_BEGIN));

    for (size_t t = 0; t < vertices.get_time_steps(); ++t) {
        for (size_t i = 0; i < vertices.size(); ++i) {
            math::Vec3f position(vertices(i, t).x, vertices(i, t).y, vertices(i, t).z);
            position = math::transformPoint(parent2root[parent2root.size() == 1 ? 0 : t], position);
            position = math::transformPoint(nodeXform, position);
            vertices(i, t).w = applyScreenSpaceRadius(position,
                                                      cameraPosition,
                                                      cameraFocal,
                                                      minDist[i], maxDist[i],
                                                      minDistRadius[i], maxDistRadius[i],
                                                      vertices(i, t).w);
        }
    }
}



} // namespace geom
} // namespace moonray

