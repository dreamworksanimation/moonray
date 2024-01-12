// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PolyMeshCalcNv.cc
/// $Id$
///

#include "PolyMeshCalcNv.h"

#include <moonray/rendering/geom/prim/QuadMesh.h>
#include <moonray/rendering/geom/prim/TriMesh.h>

#include <tbb/parallel_for.h>

namespace moonray {
namespace geom {

using namespace scene_rdl2::math;

void
PolyMeshCalcNv::set(const PolygonMesh::VertexBuffer &vertices,
                    const PolygonMesh::IndexBuffer &indices)
{
    setVtx(vertices);
    setFaceVid(indices, vertices.get_time_steps());

    computeVn();
}

void    
PolyMeshCalcNv::update(const PolygonMesh::VertexBuffer &vertices)
{
    updateVtx(vertices);

    computeVn();
}

void
PolyMeshCalcNv::updateVtx(const PolygonMesh::VertexBuffer &vertices)
{
    size_t div = 40;

    tbb::blocked_range<size_t> range(0, mVCount, mVCount / div);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t vId = r.begin(); vId < r.end(); ++vId) {
                for (size_t t = 0; t < mMotionSampleCount; ++t) {
                    const float *vPos = (const float *)&vertices(vId, t);
                    mVtx[vId * mMotionSampleCount + t] = Vec3f(vPos[0], vPos[1], vPos[2]);
                }
            }
        });
}

void
PolyMeshCalcNv::computeVn()
{
    // take average of adjacent face normals as vertex normal.
    // smooth normal is less accurate but can be faster.
    size_t div = 40;

    // compute fnv
    tbb::blocked_range<size_t> range1(0, mFCount, mFCount / div);
    tbb::parallel_for(range1, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t fId = r.begin(); fId < r.end(); ++fId) {
                computeFn(fId);
            }
        });

    // gather fn based on mVFlink info and normalize
    tbb::blocked_range<size_t> range2(0, mVCount, mVCount / div);
    tbb::parallel_for(range2, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t vId = r.begin(); vId < r.end(); ++vId) {
                size_t vfTotal = mVFlink[vId].size();
                for (size_t t = 0; t < mMotionSampleCount; ++t) {
                    Vec3f cVn(0.f, 0.f, 0.f);
                    for (size_t vfId = 0; vfId < vfTotal; ++vfId) {
                        size_t fId = (mVFlink[vId])[vfId];
                        Vec3f &fn = mFnv[fId * mMotionSampleCount + t];
                        cVn += fn;
                    }
                    size_t index = mMotionSampleCount * vId + t;
                    mVnv[index] = normalize(cVn);
                }
            }
        });

    if (mFixInvalid) {
        tbb::blocked_range<size_t> range2(0, mVnv.size(), mVnv.size() / div);
        tbb::parallel_for(range2, [&](const tbb::blocked_range<size_t> &r) {
                for (size_t v = r.begin(); v < r.end(); ++v) {
                    if (!scene_rdl2::math::isFinite(mVnv[v])) {
                        // degened surface, assign a valid but meaningless value
                        mVnv[v] = Vec3f(0, 0, 1);
                    }
                }
            });
    }

    // showVnv();
}

void    
PolyMeshCalcNv::showVtx() const
{
    std::cout << "vtx (total:" << mVtx.size() << ") {" << std::endl;

    for (size_t vId = 0; vId < mVtx.size(); ++vId) {
        if (vId % 100 == 0) {
            std::cout << " i:" << vId << " " << mVtx[vId] << std::endl;
        }
    }
    std::cout << "}" << std::endl;
}

void    
PolyMeshCalcNv::showVnv() const
{
    std::cout << "vnv (total:" << mVnv.size() << ") {" << std::endl;

    for (size_t vId = 0; vId < mVnv.size(); ++vId) {
        if (vId % 100 == 0) {
            std::cout << " i:" << vId << " " << mVnv[vId] << std::endl;
        }
    }
    std::cout << "}" << std::endl;
}

void
PolyMeshCalcNv::setVtx(const PolygonMesh::VertexBuffer &vertices)
{
    mVCount = vertices.size();
    mMotionSampleCount = vertices.get_time_steps();

    mVtx.resize(mVCount * mMotionSampleCount);
    for (size_t vId = 0; vId < mVCount; ++vId) {
        for (size_t t = 0; t < mMotionSampleCount; ++t) {
            const float *vPos = (const float *)&vertices(vId, t);
            mVtx[mMotionSampleCount * vId + t] = Vec3f(vPos[0], vPos[1], vPos[2]);
        }
    }
    //showVtx();

    mVnv.assign(mVCount * mMotionSampleCount, Vec3f(0, 0, 0));
}

void
TriMeshCalcNv::setFaceVid(const PolygonMesh::IndexBuffer &indices,
        size_t motionSampleCount)
{
    mFCount = indices.size() / 3;
    mFVid.assign(indices.begin(), indices.end());
    // work memory for vn computation
    mFnv.resize(mFCount * motionSampleCount);

    setupVFlink();
}

void    
TriMeshCalcNv::setupVFlink()
{
    mVFlink.resize(mVCount);
    for (size_t fId = 0; fId < mFCount; ++fId) {
        uint fv0Id = fId * 3;
        uint fv1Id = fv0Id + 1;
        uint fv2Id = fv0Id + 2;

        uint v0Id = mFVid[fv0Id];
        uint v1Id = mFVid[fv1Id];
        uint v2Id = mFVid[fv2Id];

        mVFlink[v0Id].push_back(fId);
        mVFlink[v1Id].push_back(fId);
        mVFlink[v2Id].push_back(fId);
    }
}
void
TriMeshCalcNv::computeFn(size_t fId)
{
    uint baseId = fId * 3;
    uint v0Id = mFVid[baseId    ];
    uint v1Id = mFVid[baseId + 1];
    uint v2Id = mFVid[baseId + 2];

    for (size_t t = 0; t < mMotionSampleCount; ++t) {
        Vec3f &v0 = mVtx[v0Id * mMotionSampleCount + t];
        Vec3f &v1 = mVtx[v1Id * mMotionSampleCount + t];
        Vec3f &v2 = mVtx[v2Id * mMotionSampleCount + t];
        // e1 x e2 = 2*area(triangle)
        // where e1 and e2 are edge vectors
        mFnv[fId * mMotionSampleCount + t] = cross(v1 - v0, v2 - v0);
    }
}

void
QuadMeshCalcNv::setFaceVid(const PolygonMesh::IndexBuffer &indices,
        size_t motionSampleCount)
{
    mFCount = indices.size() / 4;
    mFVid.assign(indices.begin(), indices.end());
    // work memory for vn computation
    mFnv.resize(mFCount * motionSampleCount);

    setupVFlink();
}

void
QuadMeshCalcNv::setupVFlink()
{
    mVFlink.resize(mVCount);
    for (size_t fId = 0; fId < mFCount; ++fId) {
        uint fv0Id = fId * 4;
        uint fv1Id = fv0Id + 1;
        uint fv2Id = fv0Id + 2;
        uint fv3Id = fv0Id + 3;

        uint v0Id = mFVid[fv0Id];
        uint v1Id = mFVid[fv1Id];
        uint v2Id = mFVid[fv2Id];
        uint v3Id = mFVid[fv3Id];

        mVFlink[v0Id].push_back(fId);
        mVFlink[v1Id].push_back(fId);
        mVFlink[v2Id].push_back(fId);
        mVFlink[v3Id].push_back(fId);
    }
}

void
QuadMeshCalcNv::computeFn(size_t fId)
{
    uint baseId = fId * 4;
    uint v0Id = mFVid[baseId    ];
    uint v1Id = mFVid[baseId + 1];
    uint v2Id = mFVid[baseId + 2];
    uint v3Id = mFVid[baseId + 3];

    for (size_t t = 0; t < mMotionSampleCount; ++t) {
        Vec3f &v0 = mVtx[v0Id * mMotionSampleCount + t];
        Vec3f &v1 = mVtx[v1Id * mMotionSampleCount + t];
        Vec3f &v2 = mVtx[v2Id * mMotionSampleCount + t];
        Vec3f &v3 = mVtx[v3Id * mMotionSampleCount + t];
        // e1 x e2 = 2*(area(1st triangle) + area(2nd triangle))
        // where e1 and e2 are diagonal vectors
        mFnv[fId * mMotionSampleCount + t] = cross(v2 - v0, v3 - v1);
    }
}

} // namepsace geom
} // namespace moonray

