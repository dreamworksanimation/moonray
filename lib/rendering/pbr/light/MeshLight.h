// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "Light.h"

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <embree4/rtcore.h>

// Forward declaration of the ISPC types
namespace ispc {
    struct MeshLight;
}

namespace scene_rdl2 {
namespace rdl2 {
    class Light;
}
}

namespace moonray {
namespace geom {
namespace internal {
    class Mesh;
    class Primitive;
}
}

namespace shading {
    class AttributeTable;
}

namespace pbr {

//----------------------------------------------------------------------------
// See "Importance Sampling of Many Lights with Adaptive Tree
// Splitting" by Alejandro Conty Estevez and Christopher Kulla (2017)
// for more details on the mesh light implementation.

// A Face is a polygon in the mesh. It behaves as a flat polygonal light.
struct Face;
// A Node is a node in the SAOH Sampling BVH. It contains the necessary
// information to traverse the BVH for the purpose of importance sampling
// the mesh light.
struct Node;

//----------------------------------------------------------------------------

/// @brief Implements light sampling for mesh lights.

class MeshLight : public LocalParamLight
{
    friend class MeshLightTester;

public:
    /// Constructor / Destructor
    explicit MeshLight(const scene_rdl2::rdl2::Light* rdlLight, bool uniformSampling = false);
    virtual ~MeshLight();

    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        MESH_LIGHT_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(MeshLight);

    virtual bool update(const scene_rdl2::math::Mat4d& world2render) override;

    /// Intersection and sampling API
    virtual bool canIlluminate(const scene_rdl2::math::Vec3f p, const scene_rdl2::math::Vec3f *n, float time, float radius,
            const LightFilterList* lightFilterList) const override;
    virtual bool isBounded() const override;
    virtual bool isDistant() const override;
    virtual bool isEnv() const override;
    virtual scene_rdl2::math::BBox3f getBounds() const override;
    virtual bool intersect(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, const scene_rdl2::math::Vec3f &wi, float time,
            float maxDistance, LightIntersection &isect) const override;
    virtual bool sample(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f *n, float time, const scene_rdl2::math::Vec3f& r,
            scene_rdl2::math::Vec3f &wi, LightIntersection &isect, float rayDirFootprint) const override;
    virtual scene_rdl2::math::Color eval(mcrt_common::ThreadLocalState* tls, const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &p, const LightFilterRandomValues& filterR,
            float time, const LightIntersection &isect, bool fromCamera, const LightFilterList *lightFilterList,
            float rayDirFootprint, float *pdf = nullptr) const override;
    virtual scene_rdl2::math::Vec3f getEquiAngularPivot(const scene_rdl2::math::Vec3f& r, float time) const override;

    // TODO: update with correct values once MeshLight is supported with adaptive light sampling
    float getThetaO() const override { return 0.f; }
    float getThetaE() const override { return 0.f; }

    // Get the rdl geometry. Used before MeshLight is finalized during render prep
    scene_rdl2::rdl2::Geometry* getReferenceGeometry() const
    {
        scene_rdl2::rdl2::SceneObject* so = mRdlLight->get(sGeometryKey);
        return so ? so->asA<scene_rdl2::rdl2::Geometry>() : nullptr;
    }

    // Get parts list
    const scene_rdl2::rdl2::StringVector& getPartsList() const
    {
        return mRdlLight->get(sPartsKey);
    }

    // Adds vertex buffer to light
    // There may be multiple mesh primitives per light
    void setMesh(geom::internal::Primitive* prim);

    // Builds BVH
    void finalize();

    // get rdl geometry. Used after MeshLight is finalized during mcrt.
    const scene_rdl2::rdl2::Geometry* getRdlGeometry() const
    {
        return mRdlGeometry;
    }

    // Sets the layer that contains the assignment ids of the reference meshes
    void setLayer(const scene_rdl2::rdl2::Layer* layer)
    {
        mLayer = layer;
    }

    const scene_rdl2::rdl2::Layer* getLayer() const
    {
        return mLayer;
    }

    // Sets the attribute table required by the map shader
    void setAttributeTable(const shading::AttributeTable* table)
    {
        mAttributeTable = table;
    }

    const shading::AttributeTable* getAttributeTable() const
    {
        return mAttributeTable;
    }

    const geom::internal::Mesh* getGeomMesh(size_t geomId) const
    {
        return mGeomMeshes[geomId];
    }

    // Sets up the embree rtcScene and submits the meshes to the embree scene.
    // The Embree scene is for intersecting the light. Embree performs better than
    // a custom intersection method for meshes that have more than ~1000 vertices.
    // This increases the BVH memeory usage, but mesh lights usually do not take a
    // large percentage of the overall scene geometry.
    void setEmbreeAccelerator(RTCDevice rtcDevice);

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass &sc);
    void reset();

    /// Copy is disabled
    MeshLight(const MeshLight &other);
    const MeshLight &operator=(const MeshLight &other);

    MESH_LIGHT_MEMBERS;

    // the reference rdl geometry
    const scene_rdl2::rdl2::Geometry* mRdlGeometry;

    // The layer contains the reference geometry's assignment ids.
    // This is needed for primitive attributes used in the map shader.
    const scene_rdl2::rdl2::Layer* mLayer;
    // The attribute table is a list of all primitive attributes requested by the
    // map shader. TODO: make scene_rdl2::rdl2::Map a RootShader so we don't need to store
    // an attribute table separately.
    const shading::AttributeTable* mAttributeTable;

    // Bounding box in render space needed for LightAccelerator.
    scene_rdl2::math::BBox3f mRenderSpaceBounds;
    // Bounding box in local space needed for building mBVH.
    scene_rdl2::math::BBox3f mLocalSpaceBounds;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // We receive the mesh's vertex buffer in render space. However, we want to
    // apply the light's local to world transform to the mesh light. The transform
    // hierarchy breakdown goes like this:

    // v : a vertex

    // We'll use the following subscripts:
    // W : world space
    // R : render space
    // G : the geometry's local space
    // L : the light's local space

    // So we have:
    // vR   : Input vertex buffer in render space
    // vG   : vertex buffer in geometry's local space
    // G2W : geometry's local to world transform
    // W2R : scene's world to render transform

    // So the vertices we're given have been generated by this equation:

    // vR = vG * G2W * W2R

    // By definition, the light's transform is the parent of the geometry's transform,
    // i.e. we need to apply the light's transform just after the geometry's transform
    // in the sequence above. To do this, first we change the interpretation of the
    // transformations in the above equation, replacing world space with the light's
    // local space. The two spaces are identical at this point, because we haven't
    // applied the light's transform, so this is just a renaming:

    // vR = vG * G2L * L2R

    // So this vR is what we are given. But what we want to compute is vR', the vertex
    // position in render space having also applied the light to world transform L2W:

    // vR' = vG * G2L * L2W * W2R

    // Our task is to convert vR to vR'. To do this, first we invert the expression for
    // vR to obtain an expression for vG:

    // vG = vR * L2R^-1 * G2L^-1
    //    = vR * R2L * L2G

    // and then substitute this into the expression for vR':

    // vR' = (vR * R2L * L2G) * G2L * L2W * W2R
    //     = vR * R2L * L2W * W2R

    // We already have vR, L2W and W2R. Now we just need to account for R2L.
    // But this can be obtained by simply inverting W2R. The justification for
    // this is that in the expression for vR, we've reinterpreted W2R as L2R,
    // because vR had no light transform applied.

    // We need to store the R2L, which is the scene's render to world transform.
    scene_rdl2::math::Mat4d mRender2Light;
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // Vertex buffer. Flat interleaved array. Different motion samples of the same vertex are next to each other in
    // memory. Meshes are ordered one after another.
    // Example: If there are 3 meshes, 2 motion samples, and the meshes have 2, 3, and 2 vertices respectively,
    // the vertex layout looks like:
    //
    //         mesh 0             |                mesh 1                   |          mesh 2
    // m0v0t0 m0v0t1 m0v1t0 m0v1t1 m1v0t0 m1v0t1 m1v1t0 m1v1t1 m1v2t0 m1v2t1 m2v0t0 m2v0t1 m2v1t0 m2v1t1
    //
    std::vector<scene_rdl2::math::Vec3f> mVertices;

    // Geometry buffers. Per mesh.
    std::vector<const geom::internal::Mesh*> mGeomMeshes;
    // Number of vertices are in each mesh. If there are multiple vertex buffers for motion blur,
    // it is the sum total of those buffers.
    std::vector<unsigned> mVertexOffset;
    // Number of faces in each mesh
    std::vector<unsigned> mFaceOffset;
    // How many vertices per face in each mesh. Should be either 3 or 4.
    std::vector<unsigned> mFaceVertexCount;

    // The mesh light is made up of faces. Each face acts as an area light.
    std::vector<Face> mFaces;

    // The MeshLight BVH is structured as a vector of nodes. For a node at index
    // n, its left child is at index n+1 and its right child is at node.mRightIndex.
    std::vector<Node> mBVH;

    // Given the geometry ID and primitive id of a face, map is back to the node
    // index in the BVH.
    std::vector<int> mPrimIDToNodeID;

    //--------------- Accessors ------------------//

    unsigned int getFaceVertexCount(const Face& face) const;
    const scene_rdl2::math::Vec3f getFaceVertex(const Face& face, size_t index, float time) const;

    //------------- Building BVH ----------------//

    // This naive split function is a fall back function for when the splitting
    // heuristic cannot be used (either there are too few primitives in the node
    // or the split failed). It sorts by the center of each primitive and puts
    // the lower half of the primitives in the left node and the upper half in
    // the right node.
    int naiveSplit(std::vector<Face>& faces, int start, int end, unsigned splitAxis,
                   scene_rdl2::math::BBox3f& leftBound, scene_rdl2::math::BBox3f& rightBound) const;

    // Recursively builds the bvh by splitting primitives into left and right nodes
    // Returns the index of the node in the bvh.
    int buildBVHRecurse(const scene_rdl2::math::BBox3f& bbox, std::vector<Face>& faces, int start,
        int end, int parentIndex);

    // Verify that the bvh is built properly
    bool verifyBuild(int index) const;

    //------------- Traversing BVH ----------------//

    // Given a shading point and shading normal, what is the importance value of
    // this node in the bvh?
    float importance(const scene_rdl2::math::Vec3f& shadingPoint, const scene_rdl2::math::Vec3f* shadingNormal, const Node& node) const;

    // Called during sample operation. This traverses the bvh using the random
    // number u. It uses the shadingPoint and shadingNormal to determine the
    // probability of picking the left or right node (PL and PR). If u < PL,
    // we traverse down the left side of the tree, otherwise we traverse down
    // the right side. We terminate at a leaf node, which contains the index of
    // a mesh face.
    // During the traversal we compute the pdf by multiplying the probabilities of
    // traversing each node. For example, the root node of the BVH has probability
    // P = 1. If we traverse the left node, that is P *= PL. If we traverse the
    // right node P *= PR. Once we have reached the leaf node, we have the pdf
    // for that face.
    // Returns index of the leaf node in the BVH.
    int drawSampleRecurse(const scene_rdl2::math::Vec3f& shadingPoint, const scene_rdl2::math::Vec3f *shadingNormal,
        int currentNodeIndex, float u, float& pdf) const;

    // If we know the face that we intersected, what is the pdf of selecting that
    // face?
    // nodeID is the node index in the BVH. It should be the index of a leaf node.
    // p is the shading point and n is the shading normal. These are needed to
    // compute the importance at each node.
    float getPdfOfFace(size_t nodeID, const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f* n) const;

    //------------- Sample Map Shader ----------------//
    // The map shader is a fully functional map shader that includes primitive
    // attribute access of the reference geometry.
    scene_rdl2::math::Color sampleMapShader(mcrt_common::ThreadLocalState* tls, int geomID,
        int primID, const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n, const scene_rdl2::math::Vec2f& uv) const;

    //
    // Cached attribute keys:
    //

    static bool                                                             sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>           sNormalizedKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>           sApplySceneScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject *>  sGeometryKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::StringVector>   sPartsKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject *>  sMapShaderKey;
    DECLARE_ATTR_SKEYS_CLEAR_RADIUS
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

