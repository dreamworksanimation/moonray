// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file PathGuide.cc

#include "PathGuide.h"

#include <scene_rdl2/render/util/AtomicFloat.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Constants.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/SceneVariables.h>

#include <atomic>
#include <memory>
#include <stack>
#include <stdint.h>
#include <vector>

namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;

//===--------------------------------------------------------------------------
// SD-Tree based path guiding
//   See "Practical Path Guiding for Efficient Light-Transport Simulation"
//     Thomas Muller, Markus Gross, Jan Novak
//     Proceedings of EGSR 2017, vol. 36, no.4
//
// The paper goes to great lengths describing the optimal number of samples
// to choose for each pass and the optimal ratio of learning to
// rendering.  This implementation pretty much ignores those apsects
// of the paper.  The implementation just expects render driver code to
// divide rendering up into a set of passes where the number of samples
// roughly double at every pass.  Learning takes place during every pass and
// sampling is done based on the learning from the previous pass.
//===--------------------------------------------------------------------------
class DirTree
{
public:
    DirTree();
    ~DirTree();

    void setNumSamplesBuild(uint64_t numSamples);
    uint64_t getNumSamplesBuild() const;
    void recordRadiance(const Vec3f &dir, const Color &radiance);

    void reset(int maxDepth, float threshold);
    void build();

    float getPdf(const Vec3f &dir) const;
    Vec3f sampleDirection(float r1, float r2) const;

private:
    struct Tree
    {
        Tree();
        Tree(const Tree &);
        Tree &operator=(const Tree &);
        ~Tree() = default;

        void setNumSamples(uint64_t numSamples);
        uint64_t getNumSamples() const;
        void recordRadiance(const Vec2f &pos, const Color &radiance);
        
        struct Node
        {
            Node();
            Node(const Node &);
            Node &operator=(const Node &);
            
            std::atomic<float> mMean[4];
            uint32_t mChildren[4];
        };
        
        std::vector<Node> mNodes;
        int mMaxDepth;
        std::atomic<uint64_t> mNumSamples;
        std::atomic<float> mSum;
    };

    // The directional trees are double-buffered.  One is used for all
    // record radiance operations (mBuild), the other is used for all
    // sampling (mSample).  At pass reset time, the new information in
    // mBuild is incorporated into mSample.
    Tree mBuild;
    Tree mSample;
};

// uniformly maps a 3D direction to a 2D planar position
// in [0,1]x[0,1].
static Vec2f
dirToPos(const Vec3f &dir)
{
    const float cosTheta = std::min(std::max(dir.z, -1.0f), 1.0f);

    // computing phi using double precision is needed and intentional
    float phi = std::atan2(dir.y, dir.x);
    while (phi < 0) {
        phi += static_cast<double>(scene_rdl2::math::two_pi);
    }

    return Vec2f((cosTheta + 1.f) * 0.5f, phi * scene_rdl2::math::sOneOverTwoPi);
}

// maps a 2D planar position [0, 1]x[0, 1] to a 3D direction
static Vec3f
posToDir(const Vec2f &pos)
{
    const float cosTheta = 2 * pos.x - 1;
    const float phi = scene_rdl2::math::sTwoPi * pos.y;

    const float sinTheta = scene_rdl2::math::sqrt(1 - cosTheta * cosTheta);
    float sinPhi, cosPhi;
    scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

    return Vec3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

// returns quadrant number (0, 1, 2, or 3) of position pos, and remaps
// pos such that it is normalized to that quadrant
static uint32_t
getChildIndexAndRemap(Vec2f &pos)
{
    MNRY_ASSERT(pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f);
    uint32_t childIndex = 0;
    if (pos.x < 0.5f) {
        // map x [0, .5] to [0, 1]
        pos.x *= 2;
    } else {
        // map x [.5, 1] to [0, 1]
        pos.x = (pos.x - 0.5f) * 2;
        childIndex |= 1;
    }

    if (pos.y < 0.5f) {
        // map y [0, .5] to [0, 1]
        pos.y *= 2;
    } else {
        // map y [.5, 1] to [0, 1]
        pos.y = (pos.y - 0.5f) * 2;
        childIndex |= 2;
    }

    return childIndex;
}

DirTree::Tree::Node::Node()
{
    // Relaxed store: we have no dependencies here.
    mMean[0].store(0.0f, std::memory_order_relaxed);
    mMean[1].store(0.0f, std::memory_order_relaxed);
    mMean[2].store(0.0f, std::memory_order_relaxed);
    mMean[3].store(0.0f, std::memory_order_relaxed);
    mChildren[0] = mChildren[1] = mChildren[2] = mChildren[3] = 0;
}

DirTree::Tree::Node::Node(const Node &that)
{
    for (int i = 0; i < 4; ++i) {
        mMean[i].store(that.mMean[i].load(std::memory_order_acquire), std::memory_order_release);
        mChildren[i] = that.mChildren[i];
    }
}

DirTree::Tree::Node &
DirTree::Tree::Node::operator=(const Node &that)
{
    for (int i = 0; i < 4; ++i) {
        // Relaxed loads and stores: we're only concerned about the ordering within this thread.
        mMean[i].store(that.mMean[i].load(std::memory_order_acquire), std::memory_order_release);
        mChildren[i] = that.mChildren[i];
    }
    return *this;
}

DirTree::Tree::Tree()
{
    mNodes.resize(1);  // root node
    mMaxDepth = 0;
    mNumSamples.store(0, std::memory_order_relaxed); // No dependencies
    mSum.store(0.0f, std::memory_order_relaxed);     // No dependencies
}

DirTree::Tree::Tree(const Tree &that):
    mNodes(that.mNodes),
    mMaxDepth(that.mMaxDepth)
{
    mNumSamples.store(that.mNumSamples.load(std::memory_order_acquire), std::memory_order_release);
    mSum.store(that.mSum.load(std::memory_order_acquire), std::memory_order_release);
}

DirTree::Tree &
DirTree::Tree::operator=(const Tree &that)
{
    mNodes = that.mNodes;
    mMaxDepth = that.mMaxDepth;
    mNumSamples.store(that.mNumSamples.load(std::memory_order_acquire), std::memory_order_release);
    mSum.store(that.mSum.load(std::memory_order_acquire), std::memory_order_release);

    return *this;
}

uint64_t
DirTree::Tree::getNumSamples() const
{
    return mNumSamples.load(std::memory_order_acquire);
}

void
DirTree::Tree::setNumSamples(uint64_t numSamples)
{
    mNumSamples.store(numSamples, std::memory_order_release);
}

void
DirTree::Tree::recordRadiance(const Vec2f &pos, const Color &radiance)
{
    MNRY_ASSERT(scene_rdl2::math::isFinite(radiance));
    if (!scene_rdl2::math::isFinite(radiance)) return;
    
    const float luminance = scene_rdl2::math::luminance(radiance);
    mNumSamples.fetch_add(1, std::memory_order_acq_rel);
    mSum.fetch_add(luminance, std::memory_order_acq_rel);

    // update nodes
    uint64_t index = 0;
    Vec2f curPos = pos;
    do {
        Node &node = mNodes[index];

        // which quad does this direction belong to?  update the mean
        // for that quad.
        uint32_t childIndex = getChildIndexAndRemap(curPos);

        // We're make sure that we're consistent with other accesses to mMean, even though Muller's reference
        // implementation uses relaxed atomics everywhere.
        node.mMean[childIndex].fetch_add(luminance, std::memory_order_acq_rel);

        // descend into the quad if it is not a leaf.
        // by construction, if a quad is a leaf, the child index for that quad is 0
        index = node.mChildren[childIndex];
    } while (index != 0);
}


DirTree::DirTree()
{
}

DirTree::~DirTree()
{
}

void
DirTree::setNumSamplesBuild(uint64_t numSamples)
{
    mBuild.setNumSamples(numSamples);
}

uint64_t
DirTree::getNumSamplesBuild() const
{
    return mBuild.getNumSamples();
}

void
DirTree::recordRadiance(const Vec3f &dir, const Color &radiance)
{
    const Vec2f pos = dirToPos(dir);
    mBuild.recordRadiance(pos, radiance);
}

void
DirTree::reset(int maxDepth, float threshold)
{
    // This method is called during a passReset() after a call to build().  It
    // computes the topology of a new mBuild to a refined (or coarser)
    // version of mSample

    // First, initialize mBuild with a single root node
    mBuild.mNodes.clear();
    mBuild.mNodes.resize(1);
    mBuild.mMaxDepth = 0;
    mBuild.mNumSamples.store(0, std::memory_order_relaxed); // No dependencies
    mBuild.mSum.store(0.f, std::memory_order_relaxed);      // No dependencies

    // Build up mBuild as a refined version of mSample.  mSample is traversed
    // in depth first order and is refined if enough total energy exists
    // (based on threshold) in a node and we are within the maxDepth parameter
    struct StackItem {
        uint64_t mBuildIndex;  // index of node we are building (in mBuild)
        uint64_t mSourceIndex; // index of node we may subdivide
        Tree *mSourceTree;     // tree that source node belongs to (could be sample or build)
        int mDepth;            // depth of node
    };

    std::stack<StackItem> todo;
    todo.push({ 0, 0, &mSample, 1 });

    // total energy in mSample tree, we'll subdivide when any node fraction of this
    // value exceeds threshold
    const float nodeThreshold = mSample.mSum.load(std::memory_order_acquire) * threshold;

    while (!todo.empty()) {
        StackItem sItem = todo.top();
        todo.pop();

        mBuild.mMaxDepth = std::max(mBuild.mMaxDepth, sItem.mDepth);
        Tree::Node &sourceNode = sItem.mSourceTree->mNodes[sItem.mSourceIndex];

        for (int i = 0; i < 4; ++i) {
            // We'll acquire here to sync with any previously run threads.
            const float sourceNodeMean = sourceNode.mMean[i].load(std::memory_order_acquire);
            if (sItem.mDepth < maxDepth && sourceNodeMean > nodeThreshold) {
                // if this quadrant is not a leaf in the source node, push the child
                // onto the stack.  note that the only case where the source node
                // cannot be a leaf is if the source node comes from the sample tree.
                if (sourceNode.mChildren[i] != 0) {
                    // not a leaf.
                    MNRY_ASSERT(sItem.mSourceTree == &mSample);
                    todo.push({ mBuild.mNodes.size(), sourceNode.mChildren[i], sItem.mSourceTree, sItem.mDepth + 1 });
                } else {
                    // is a leaf. could be from mBuild or mSample
                    todo.push({ mBuild.mNodes.size(), mBuild.mNodes.size(), &mBuild, sItem.mDepth + 1 });
                }

                mBuild.mNodes[sItem.mBuildIndex].mChildren[i] = static_cast<uint32_t>(mBuild.mNodes.size());
                mBuild.mNodes.emplace_back(); // add node
                // spread the energy evenly among the new node's quadrants
                const float newMean = sourceNodeMean * 0.25f;
                Tree::Node &newNode = mBuild.mNodes.back();
                for (int j = 0; j < 4; ++j) {
                    // We're creating new nodes: we have no dependencies: relaxed ordering
                    newNode.mMean[j].store(newMean, std::memory_order_relaxed);
                }

                if (mBuild.mNodes.size() > std::numeric_limits<uint32_t>::max()) {
                    scene_rdl2::Logger::warn("DirTree hit max children count.");
                    todo = std::stack<StackItem>();
                    break;
                }
                
            }
        }
    }

    // now set all the energy to 0
    for (Tree::Node &node : mBuild.mNodes) {
        // No dependencies: relaxed ordering
        node.mMean[0].store(0.0f, std::memory_order_relaxed);
        node.mMean[1].store(0.0f, std::memory_order_relaxed);
        node.mMean[2].store(0.0f, std::memory_order_relaxed);
        node.mMean[3].store(0.0f, std::memory_order_relaxed);
    }

    // I experimented with mBuild.mNodes.shrink_to_fit().  But this
    // actually caused the renderer to use more memory in my test cases.
}

void
DirTree::build()
{
    // This method is called during a pass reset.
    // mBuild is copied into mSample.  A new topology (refined or coarser)
    // for mBuild will be computed in reset() based on mSample.
    // We need to make a copy, because mBuild will be modified as the spatial
    // tree is refined, but before the new mBuild topology is computed.
    mSample = mBuild;
}

float
DirTree::getPdf(const Vec3f &dir) const
{
    // Use the sample tree

    // First check for an empty (or nearly empty) tree
    // in which case, we'll just use spherical sampling
    const auto numSamples = mSample.mNumSamples.load(std::memory_order_acquire);
    const auto sum = mSample.mSum.load(std::memory_order_acquire);
    if (numSamples == 0 || (sum / (scene_rdl2::math::sFourPi * numSamples)) == 0.0f) {
        return scene_rdl2::math::sOneOverFourPi;
    }

    Vec2f pos = dirToPos(dir);

    // recurse into the nodes
    float pdf = scene_rdl2::math::sOneOverFourPi;
    uint64_t index = 0; // start at the root
    do {
        // which quadrant?
        const uint32_t i = getChildIndexAndRemap(pos); // 0, 1, 2, or 3

        if (mSample.mNodes[index].mMean[i].load(std::memory_order_acquire) <= 0.f) {
            return 0; // invalid pdf
        }

        float total = 0.f;
        for (uint32_t j = 0; j < 4; ++j) {
            total += mSample.mNodes[index].mMean[j].load(std::memory_order_acquire);
        }
        // Why the 4.0f?
        // The domain is the unit square.  Broken into
        // four quadrants - each with area of 1/4.
        // mean[0]/total + mean[1]/total + mean[2]/total + mean[3]/total = 1.
        //        A               B               C               D
        // Total integral value is A/4 + B/4 + C/4 + D/4 = (A + B + C + D)/4.
        pdf *= (4.0f * mSample.mNodes[index].mMean[i].load(std::memory_order_acquire) / total);

        index = mSample.mNodes[index].mChildren[i];
    } while (index != 0); // until we hit a leaf

    return pdf;
}

static inline bool
checkForZero(float val, int index, const Vec2f &rpos, Vec2f &pos)
{
    // It is possible that some denominators in sampleDirection can reach exactly zero, even though they should
    // theoretically be greater than 0.  In this case we need to break from the recursion. If this is our first time
    // through in the recursion (index == 0), then we need to set the quad position to some reasonable default.  (we'll
    // just use some random position).  We return true if the recursion needs to be broken, false otherwise.
    if (val == 0.f) {
        if (index == 0) {
            pos = rpos;
        }
        return true;
    }

    return false;
}

Vec3f
DirTree::sampleDirection(float r1, float r2) const
{
    // Use the sample tree

    // in some cases we may return just a random direction
    const Vec2f rpos(r1, r2);  // random quad position

    // First check for an empty tree
    // in which case, we'll just use spherical sampling
    const auto numSamples = mSample.mNumSamples.load(std::memory_order_acquire);
    const auto sum = mSample.mSum.load(std::memory_order_acquire);
    if (numSamples == 0 || (sum / (scene_rdl2::math::sFourPi * numSamples)) == 0.0f) {
        return posToDir(rpos);
    }

    // recurse into the sample nodes
    // Pick quadtree children based on their mean value, compared
    // to the random sample values.
    uint64_t index = 0; // start at the root
    Vec2f pos = Vec2f(0.f, 0.f);
    Vec2f sample = rpos;
    float scale = 1.0f; // halved with each recursion
    do {
        const Tree::Node &node = mSample.mNodes[index];

        // compute a quadrant, as well as a location in
        // the quadrant
        int quadrant = 0;               // quadrant in node 0, 1, 2, or 3
        Vec2f quadrantOrigin(0.f, 0.f); // [0 or .5, 0 or .5]

        const float topLeft  = node.mMean[0].load(std::memory_order_acquire);
        const float topRight = node.mMean[1].load(std::memory_order_acquire);
        const float botLeft  = node.mMean[2].load(std::memory_order_acquire);
        const float botRight = node.mMean[3].load(std::memory_order_acquire);
        const float total = topLeft + topRight + botLeft + botRight;

        if (checkForZero(total, index, rpos, pos)) break;

        // first the x-axis, re-normalizing sample
        // as needed
        float partial = topLeft + botLeft;
        float boundary = partial / total;

        if (sample.x < boundary) {
            if (checkForZero(boundary, index, rpos, pos)) break;
            sample.x /= boundary;
            if (checkForZero(partial, index, rpos, pos)) break;
            boundary = topLeft / partial;
        } else {
            partial = total - partial;
            quadrantOrigin.x = 0.5f;
            if (checkForZero(1.0f - boundary, index, rpos, pos)) break;
            sample.x = (sample.x - boundary) / (1.0f - boundary);
            if (checkForZero(partial, index, rpos, pos)) break;
            boundary = topRight / partial;
            quadrant |= 1;
        }

        // now split the y axis
        if (sample.y < boundary) {
            if (checkForZero(boundary, index, rpos, pos)) break;
            sample.y /= boundary;
        } else {
            quadrantOrigin.y = 0.5f;
            if (checkForZero(1.0f - boundary, index, rpos, pos)) break;
            sample.y = (sample.y - boundary) / (1.0f - boundary);
            quadrant |= 2;
        }

        if (node.mChildren[quadrant] == 0) {
            // we hit a leaf, we are at the end of the recursion
            pos += scale * (quadrantOrigin + 0.5f * sample);
        } else {
            // add in o and continue the recursion,
            pos += scale * quadrantOrigin;
            scale *= 0.5f;
        }
        
        index = node.mChildren[quadrant];
    } while (index != 0); // until we hit a leaf

    return posToDir(pos);
}

class SpatialTree
{
public:
    explicit SpatialTree(const scene_rdl2::math::BBox3f &bbox);
    SpatialTree(const SpatialTree &) = delete;
    SpatialTree &operator=(const SpatialTree &) = delete;
    ~SpatialTree() = default;

    void refine(uint64_t threshold);

    void resetDirTrees(int maxDepth, float threshold);
    void buildDirTrees();
    DirTree *getDirTree(const Vec3f &p);

private:
    struct Node
    {
        DirTree mDirTree;
        int32_t mAxis;
        uint32_t mChildren[2];
        bool mIsLeaf;

        Node():
            mAxis(0),
            mChildren{0, 0},
            mIsLeaf(true)
        {
        }
    };

    std::vector<Node> mNodes;
    scene_rdl2::math::BBox3f mBbox;
};

SpatialTree::SpatialTree(const scene_rdl2::math::BBox3f &bbox)
{
    // we get better subdivision efficiency if we expand the bounding
    // box so that it is a cube.
    const Vec3f maxDim = Vec3f(scene_rdl2::math::reduce_max(bbox.size()));
    const Vec3f expand = (maxDim - bbox.size()) / 2.0f;
    mBbox.lower = bbox.lower - expand;
    mBbox.upper = bbox.upper + expand;

    // start with a single leaf node
    mNodes.resize(1);
}

void
SpatialTree::refine(uint64_t threshold)
{
    // TODO: add support for max memory parameter which will
    // cause an early return from this method if the predicted
    // max memory will be exceeded.  For now, we work under the
    // assumption of unlimited memory.

    // do a depth first traversal of the tree, split nodes as we go
    // if the number of radiance updates to that node exceeds the threshold
    std::stack<uint64_t> todo;
    todo.push(0); // root node
    while (!todo.empty()) {
        uint64_t index = todo.top();
        todo.pop();

        // if this is a leaf node, we might need to subdivide it
        if (mNodes[index].mIsLeaf) {
            // TODO: we may want to place some sort of limit on the
            // total number of nodes in the tree.  If we exceed that
            // limit we'll disable splitting
            if (mNodes[index].mDirTree.getNumSamplesBuild() > threshold) {
                // add two child nodes - alternate split x/y/z axis
                // each child inherits a copy of the parent's dirtree
                // but is assigned half the samples
                mNodes.resize(mNodes.size() + 2);
                for (size_t i = mNodes.size() - 2; i < mNodes.size(); ++i) {
                    Node &n = mNodes[i];
                    n.mDirTree = mNodes[index].mDirTree;
                    n.mDirTree.setNumSamplesBuild(n.mDirTree.getNumSamplesBuild() / 2);
                    n.mAxis = (mNodes[index].mAxis + 1) % 3;
                }

                mNodes[index].mDirTree = {};
                mNodes[index].mIsLeaf = false;
                mNodes[index].mChildren[0] = mNodes.size() - 2;
                mNodes[index].mChildren[1] = mNodes.size() - 1;
            }
        }

        if (!mNodes[index].mIsLeaf) {
            // push its two children onto the stack
            todo.push(mNodes[index].mChildren[0]);
            todo.push(mNodes[index].mChildren[1]);
        }
    }

    // I experiemented with mNodes.shrink_to_fit().  But this
    // actually caused the renderer to use more memory in my test cases.
}

void
SpatialTree::resetDirTrees(int maxDepth, float threshold)
{
    // TODO: we could spawn new threads and parallelize this operation.
    // it is expected that all other rendering threads are blocked during
    // this operation.
    for (size_t i = 0; i < mNodes.size(); ++i) {
        if (mNodes[i].mIsLeaf) {
            mNodes[i].mDirTree.reset(maxDepth, threshold);
        }
    }
}

void
SpatialTree::buildDirTrees()
{
    // TODO: we could spawn new threads and parallelize this operation.
    // it is expected that all other rendering threads are blocked during
    // this operation.
    for (size_t i = 0; i < mNodes.size(); ++i) {
        if (mNodes[i].mIsLeaf) {
            mNodes[i].mDirTree.build();
        }
    }
}

DirTree *
SpatialTree::getDirTree(const Vec3f &p)
{
    // normalize all position look ups to maximize precision
    Vec3f np = (p - mBbox.lower) / mBbox.size();

    uint64_t index = 0; // start at root
    while (!mNodes[index].mIsLeaf) {
        if (np[mNodes[index].mAxis] < 0.5f) {
            // descend into the 0 child, expand the split
            // axis so [0, 0.5] -> [0, 1]
            np[mNodes[index].mAxis] *= 2.f;
            index = mNodes[index].mChildren[0];
        } else {
            // descend into the 1 child, expand the
            // split axis so [0.5, 1.0] -> [0, 1]
            np[mNodes[index].mAxis] = 2.0f * (np[mNodes[index].mAxis] - 0.5f);
            index = mNodes[index].mChildren[1];
        }
    }

    MNRY_ASSERT(index < mNodes.size());
    return &mNodes[index].mDirTree;
}

//===--------------------------------------------------------------------------
// PathGuide::Impl
//===--------------------------------------------------------------------------

class PathGuide::Impl
{
public:
    Impl();
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    ~Impl() = default;

    void startFrame(const BBox3f &bbox, const scene_rdl2::rdl2::SceneVariables &vars);
    void passReset();
    void recordRadiance(const Vec3f &p, const Vec3f &dir, const Color &radiance) const;
    float getPdf(const Vec3f &p, const Vec3f &dir) const;
    Vec3f sampleDirection(const Vec3f &p, float r1, float r2, float *pdf) const;
    bool isEnabled() const;
    bool canSample() const;
    float getPercentage() const;

private:
    bool mEnable;
    float mPercentage;
    mutable std::unique_ptr<SpatialTree> mSpatialTree;
    int mSpatialTreeThreshold;
    int mDirTreeMaxDepth;
    float mDirTreeThreshold;
    int mMinResetIterations;
    unsigned int mResetIterations;
    
};

PathGuide::Impl::Impl():
    mEnable(false),
    mPercentage(0.0f),
    mSpatialTree(nullptr),
    mSpatialTreeThreshold(0),
    mDirTreeMaxDepth(0),
    mDirTreeThreshold(0.f),
    mMinResetIterations(0),
    mResetIterations(0)
{
}

void
PathGuide::Impl::startFrame(const BBox3f &bbox, const scene_rdl2::rdl2::SceneVariables &vars)
{
    mSpatialTree.reset();
    mEnable = vars.get(scene_rdl2::rdl2::SceneVariables::sPathGuideEnable);
    if (!mEnable) return;

    // these could become user settings via rdl scene variables
    // so far, these defaults seem to work reasonably well
    mPercentage = 0.5f; // half of bsdf samples should use path guiding
    mSpatialTreeThreshold = 12000;
    mDirTreeMaxDepth = 20;
    mDirTreeThreshold = 0.01f;
    mMinResetIterations = 2; // minimum reset iterations before sampling can be used

    mResetIterations = 2;    

    mSpatialTree.reset(new SpatialTree(bbox));
    mSpatialTree->refine(mSpatialTreeThreshold);
    mSpatialTree->resetDirTrees(mDirTreeMaxDepth, mDirTreeThreshold);
}

void
PathGuide::Impl::passReset()
{
    // Radiance can be recorded, directions sampled, and pdfs computed
    // safely on multiple threads based on the use of atomic types.
    // However, this method must only be called by a single thread and
    // furthermore no other methods of the path guide may be accessed.
    // It is the responsibility of the render driver to ensure that
    // this thread safety requirement is met.
    //
    // This method is single threaded.  If it becomes a slow bottleneck
    // we can investigate ways to parallelize it - probably by parallelizing
    // the resetDirTrees method.

    MNRY_ASSERT(mEnable);
    mSpatialTree->buildDirTrees();
    ++mResetIterations;
    // Split a spatial tree node when the number of samples in the node's dirtree
    // exceeds this parameter value.  We want this value to exponentially increase
    // with each pass reset.  This is the main reason why we work best when each
    // pass has roughly twice as many samples as the previous.
    mSpatialTree->refine(std::pow(2, mResetIterations / 2.0f) * mSpatialTreeThreshold);
    mSpatialTree->resetDirTrees(mDirTreeMaxDepth, mDirTreeThreshold);
}

void
PathGuide::Impl::recordRadiance(const Vec3f &p, const Vec3f &dir, const Color &radiance) const
{
    if (!mEnable) return;

    DirTree *dirTree = mSpatialTree->getDirTree(p);
    MNRY_ASSERT(dirTree != nullptr);
    dirTree->recordRadiance(dir, radiance);
}

float
PathGuide::Impl::getPdf(const Vec3f &p, const Vec3f &dir) const
{
    MNRY_ASSERT(mEnable);
    const DirTree *dirTree = mSpatialTree->getDirTree(p);
    return dirTree->getPdf(dir);
}

Vec3f
PathGuide::Impl::sampleDirection(const Vec3f &p, float r1, float r2, float *pdf) const
{
    MNRY_ASSERT(mEnable);
    const DirTree *dirTree = mSpatialTree->getDirTree(p);
    MNRY_ASSERT(dirTree != nullptr);
    const Vec3f dir = dirTree->sampleDirection(r1, r2);
    if (pdf) {
        *pdf = dirTree->getPdf(dir);
    }

    return dir;
}

bool
PathGuide::Impl::isEnabled() const
{
    return mEnable;
}

bool
PathGuide::Impl::canSample() const
{
    // Require at least mMinResetIterations resets before using
    // the path guide for sampling.  Anything less may just
    // lead to results that are worse than bsdf sampling.
    return mEnable && (mResetIterations > mMinResetIterations);
}

float
PathGuide::Impl::getPercentage() const
{
    return mPercentage;
}

//===--------------------------------------------------------------------------
// PathGuide
//===--------------------------------------------------------------------------
PathGuide::PathGuide():
    mImpl(new Impl)
{
}

PathGuide::~PathGuide()
{
}

void
PathGuide::startFrame(const BBox3f &bbox, const scene_rdl2::rdl2::SceneVariables &vars)
{
    mImpl->startFrame(bbox, vars);
}

void
PathGuide::passReset()
{
    mImpl->passReset();
}

void
PathGuide::recordRadiance(const Vec3f &p, const Vec3f &dir, const Color &radiance) const
{
    mImpl->recordRadiance(p, dir, radiance);
}

float
PathGuide::getPdf(const Vec3f &p, const Vec3f &dir) const
{
    return mImpl->getPdf(p, dir);
}

Vec3f
PathGuide::sampleDirection(const Vec3f &p, float r1, float r2, float *pdf) const
{
    return mImpl->sampleDirection(p, r1, r2, pdf);
}

bool
PathGuide::isEnabled() const
{
    return mImpl->isEnabled();
}

bool
PathGuide::canSample() const
{
    return mImpl->canSample();
}

float
PathGuide::getPercentage() const
{
    return mImpl->getPercentage();
}

} // namespace pbr
} // namespace moonray

