// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PolyMeshCalcNv.h
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/PolygonMesh.h>

namespace moonray {
namespace geom {

class PolyMeshCalcNv
{
public:
    PolyMeshCalcNv(bool fixInvalid) : mVCount(0), mFCount(0), mMotionSampleCount(0),
        mFixInvalid(fixInvalid) {}
    virtual ~PolyMeshCalcNv() {}

    // set new vertex/face info and compute vn
    void set(const PolygonMesh::VertexBuffer &vertices,
             const PolygonMesh::IndexBuffer &indices);

    // update vtx w/ mtx info and compute vn. You have to call set() first.
    void update(const PolygonMesh::VertexBuffer &vertices);

    size_t getVCount() const { return mVCount; }
    const float *getVn(uint vId, size_t timeIndex = 0) const {
        return (const float *)&mVnv[vId * mMotionSampleCount + timeIndex];
    }

    void showVtx() const;
    void showVnv() const;

private:
    // copy index buffer and resize mFnv based on number of triangle or quad
    virtual void setFaceVid(const PolygonMesh::IndexBuffer &indices,
            size_t motionSampleColunt) = 0;
    // compute weighted face normal based on face area
    virtual void computeFn(size_t fId) = 0;

protected:
    void setVtx(const PolygonMesh::VertexBuffer &vertices);

    void updateVtx(const PolygonMesh::VertexBuffer &vertices);
    // compute vertex normal by averaging adjacent face normal
    void computeVn();

    size_t mVCount; // number of vertices
    size_t mFCount; // number of faces
    size_t mMotionSampleCount; // number of motion blur samples

    std::vector<Vec3f> mVtx; // input vtx position
    std::vector<Vec3f> mVnv; // vertex normals

    std::vector<uint> mFVid; // face vertex id array
    std::vector<Vec3f> mFnv; // work memory for vn computation

    std::vector< std::vector<uint> > mVFlink; // work memory for vn computation : vtx -> face link
    bool mFixInvalid;
};

class TriMeshCalcNv : public PolyMeshCalcNv
{
public:
    TriMeshCalcNv(bool fixInvalid) : PolyMeshCalcNv(fixInvalid) {}
private:
    virtual void setFaceVid(const PolygonMesh::IndexBuffer &indices,
            size_t motionSampleColunt) override;
    virtual void computeFn(size_t fId) override;

    // Setup Vertex-Face lookup table
    void setupVFlink();
};

class QuadMeshCalcNv : public PolyMeshCalcNv
{
public:
    QuadMeshCalcNv(bool fixInvalid) : PolyMeshCalcNv(fixInvalid) {}
private:
    virtual void setFaceVid(const PolygonMesh::IndexBuffer &indices,
            size_t motionSampleColunt) override;
    virtual void computeFn(size_t fId) override;

    // Setup Vertex-Face lookup table
    void setupVFlink();
};
} // namespace geom
} // namespace moonray

