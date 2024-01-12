// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DebugRay.h"
#include <scene_rdl2/render/util/BitUtils.h>
#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <scene_rdl2/render/logging/logging.h>

#include <algorithm>
#include <map>

namespace moonray {
namespace pbr {

#define DEBUG_RAYS_VERSION      4
#define NULL_RAY_VERTEX_ID      uint32_t(-1)
#define NULL_THREAD_ID          uint16_t(-1)

#define MAX_VERTS_PER_BLOCK     (1024 * 16)

using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

namespace
{
const unsigned MAX_VERTS_PER_BLOCK_SHIFT = scene_rdl2::util::countTrailingZeros(MAX_VERTS_PER_BLOCK);

// hands out the next unused value in sequence if an id hasn't been seen
// before, otherwise hands out the previous id it was assigned
template<typename FROM, typename TO>
struct IdRemapper : public std::map<FROM, TO>
{
    TO remapId(FROM id)
    {
        auto it = this->find(id);
        if (it != this->end())
            return it->second;
        TO newId = TO(this->size());
        this->emplace(id, newId);
        return newId;
    }

    typedef std::map<FROM, TO>  MapType;
};

finline uint64_t
fullId(uint32_t vertId, uint16_t threadId)
{
    return (uint64_t(threadId) << 32) | uint64_t(vertId);
}

finline uint64_t
fullId(DebugRayVertex const &vert)
{
    return fullId(vert.mId, vert.mThreadId);
}

finline uint64_t
fullParentId(DebugRayVertex const &vert)
{
    return fullId(vert.mParentId, vert.mParentThreadId);
}

//
// IO helpers
//

finline bool 
readRawData(FILE *file, void *buffer, size_t numBytes)
{
    size_t numBytesRead = fread(buffer, 1, numBytes, file);
    return !ferror(file) && numBytesRead == numBytes;
}

template<typename T>
finline bool
read(FILE *file, T *dest)
{
    return readRawData(file, dest, sizeof(dest[0]));
}

finline bool
writeRawData(FILE *file, void const *buffer, size_t numBytes)
{
    size_t numBytesWritten = fwrite(buffer, 1, numBytes, file);
    return !ferror(file) && numBytesWritten == numBytes;
}

template<typename T>
finline bool
write(FILE *file, T value)
{
    return writeRawData(file, &value, sizeof(value));
}

}   // end of anonymous namespace

//----------------------------------------------------------------------------

BBox2i DebugRayRecorder::sActiveViewport = BBox2i(Vec2i(0), Vec2i(-1));
Mat4f DebugRayRecorder::sRender2World = Mat4f(one);
unsigned DebugRayRecorder::sMaxVertsPerThread = 0;
bool DebugRayRecorder::sRecordingEnabled = false;

DebugRayRecorder::DebugRayRecorder(uint32_t threadId) :
    mThreadId(uint16_t(threadId))
{
    MNRY_ASSERT(threadId < 0xffff);

    // we may be trying to record right as this recorder gets initialized...
    // this will essentially be a no-op if enableRecording wasn't already called
    record();
}

void
DebugRayRecorder::record()
{
    cleanUp();

    mNextId = 0;
    mEndId = sMaxVertsPerThread;

    if (sMaxVertsPerThread) {
        VertexBlock *vertBlock = new VertexBlock;
        vertBlock->reserve(MAX_VERTS_PER_BLOCK);
        mVertexBlocks.push_back(vertBlock);
    }
}

void
DebugRayRecorder::stopRecording()
{
    mEndId = mNextId;
}

void
DebugRayRecorder::cleanUp()
{
    mEndId = 0;
    mNextId = 0;

    for (auto it = mVertexBlocks.begin(); it != mVertexBlocks.end(); ++it) {
        delete *it;
    }

    mVertexBlocks.clear();

    memset(&mDummyVertex, 0, sizeof(mDummyVertex));
    mDummyVertex.mScreenX = uint16_t(-1);
}

DebugRayVertex *
DebugRayRecorder::startNewRay(Vec3f const &origin, unsigned screenX, unsigned screenY, uint32_t userTags)
{
    if (int(screenX) >= sActiveViewport.lower.x && int(screenX) <= sActiveViewport.upper.x &&
        int(screenY) >= sActiveViewport.lower.y && int(screenY) <= sActiveViewport.upper.y &&
        mNextId < mEndId) {

        DebugRayVertex *vert = allocRayVertex();

        vert->mHitPoint = origin;
        vert->mUserTags = userTags;
        vert->mParentId = NULL_RAY_VERTEX_ID;
        vert->mDepth = uint8_t(-1);
        vert->mScreenX = uint16_t(screenX);
        vert->mScreenY = uint16_t(screenY);
        vert->mParentThreadId = NULL_THREAD_ID;

        return vert;
    }

    MNRY_ASSERT(mDummyVertex.mScreenX == uint16_t(-1));

    return &mDummyVertex;
}

DebugRayVertex *
DebugRayRecorder::extendRay(DebugRayVertex const *parent)
{
    MNRY_ASSERT(parent);

    if (parent->mScreenX != uint16_t(-1) && mNextId < mEndId) {

        // we store depths in 8-bits, check that we don't overflow
        // the value 0xff is reserved
        MNRY_ASSERT(parent->mDepth != 254);

        DebugRayVertex *vert = allocRayVertex();

        vert->mParentId = parent->mId;
        vert->mDepth = parent->mDepth + 1;
        vert->mScreenX = parent->mScreenX;
        vert->mScreenY = parent->mScreenY;
        vert->mParentThreadId = parent->mThreadId;

        return vert;
    }

    MNRY_ASSERT(mDummyVertex.mScreenX == uint16_t(-1));

    return &mDummyVertex;
}

void
DebugRayRecorder::enableRecording(BBox2i viewport, Mat4f const &render2world,
        uint32_t maxVerticesPerThread)
{
    MOONRAY_START_THREADSAFE_STATIC_WRITE

    sActiveViewport = viewport;
    sRender2World = render2world;
    sMaxVertsPerThread = maxVerticesPerThread;
    sRecordingEnabled = true;

    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
}

void
DebugRayRecorder::disableRecording()
{
    MOONRAY_START_THREADSAFE_STATIC_WRITE
    sActiveViewport = BBox2i(Vec2i(0), Vec2i(-1));
    sRender2World = Mat4f(one);
    sMaxVertsPerThread = 0;
    sRecordingEnabled = false;
    MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
}

DebugRayVertex *
DebugRayRecorder::allocRayVertex()
{
    // we have checked this is true higher up
    MNRY_ASSERT(mNextId < mEndId);

    VertexBlock *vertBlock = mVertexBlocks.back();
    if (vertBlock->size() == MAX_VERTS_PER_BLOCK) {
        // time to allocate a new list
        vertBlock = new VertexBlock;
        vertBlock->reserve(MAX_VERTS_PER_BLOCK);
        mVertexBlocks.push_back(vertBlock);
    }

    vertBlock->push_back(DebugRayVertex());

    DebugRayVertex *vert = &vertBlock->back();
    memset(vert, 0, sizeof(DebugRayVertex));

    // fill in common fields between parent and child vertices
    vert->mContribution = Vec3f(1.0f, 0.41f, 0.71f); // hot pink
    vert->mId = mNextId++;
    vert->mThreadId = mThreadId;

    MNRY_ASSERT(getRay(mNextId - 1) == vert);

    return vert;
}

DebugRayVertex *
DebugRayRecorder::getRay(uint32_t id)
{
    if (id < mEndId) {
        size_t listIdx = id >> MAX_VERTS_PER_BLOCK_SHIFT;
        size_t elemIdx = id & (MAX_VERTS_PER_BLOCK - 1);

        if (listIdx < mVertexBlocks.size() && elemIdx < mVertexBlocks[listIdx]->size()) {
            return &((*mVertexBlocks[listIdx])[elemIdx]);
        }
    }
    return nullptr;
}

//----------------------------------------------------------------------------

bool
DebugRayBuilder::build(unsigned width, unsigned height, std::vector<DebugRayRecorder *> const &recorders)
{
    cleanUp();

    if (recorders.empty() || !width || !height) {
        return false;
    }

    mWidth = uint32_t(width);
    mHeight = uint32_t(height);

    //
    // build up a rich tree structure from the inputs which we can manipulate
    //

    // compute total amount of debug vertices
    size_t numSrcVertices = 0;
    for (size_t i = 0; i < recorders.size(); ++i) {
        numSrcVertices += recorders[i]->getNumRayVertices();
    }

    if (!numSrcVertices) {
        return false;
    }

    //
    // linearize the full 48-bit sparse address space into compact
    // sequential 32-bit ids and assign all rays to corresponding nodes
    //

    // add 1 to account for a master root node which we insert into the node tree
    mNodes.resize(numSrcVertices + 1);
    mNodes[0].mVertex = nullptr;
    mNodes[0].mParent = nullptr;
    mNodes[0].mNumSubnodes = 1;   // start at 1 to account for self

    {
        size_t numNodes = 1;
        IdRemapper<uint64_t, uint32_t> remapper;

        // reserve 0 as the parent id
        remapper.remapId(fullId(NULL_RAY_VERTEX_ID, NULL_THREAD_ID));

        for (size_t i = 0; i < recorders.size(); ++i) {
            DebugRayRecorder *recorder = recorders[i];
            for (size_t iblock = 0; iblock < recorder->mVertexBlocks.size(); ++iblock) {
                DebugRayRecorder::VertexBlock &vertBlock = *recorders[i]->mVertexBlocks[iblock];
                for (size_t ivert = 0; ivert < vertBlock.size(); ++ivert, ++numNodes) {
                    DebugRayVertex *vert = &vertBlock[ivert];
                    MNRY_ASSERT(vert->mScreenX < mWidth && vert->mScreenY < mHeight);

                    vert->mParentId = remapper.remapId(fullParentId(*vert));
                    vert->mId = remapper.remapId(fullId(*vert));

                    // verify the index of the vert matches up with its location in the
                    // mNodes array
                    MNRY_ASSERT(vert->mId == numNodes);

                    Node *node = &mNodes[numNodes];
                    node->mVertex = &vertBlock[ivert];
                    node->mParent = nullptr;    // this gets hookup up later
                    node->mNumSubnodes = 1;     // start at 1 to account for self
                }
            }
        }

        MNRY_ASSERT(numNodes == mNodes.size());

        // all ids are now sequential! we can release the IdRemapper
    }

    // hook up all parents and children
    for (size_t i = 1; i < mNodes.size(); ++i) {
        Node *node = &mNodes[i];
        uint32_t parentId = node->mVertex->mParentId;

        node->mParent = &mNodes[parentId];
        node->mParent->mChildren.insert(node);
    }

    // verify that there are no orphaned nodes, and we have reasonable values
    // for hit points
    MNRY_DURING_ASSERTS
    (
        for (size_t i = 0; i < mNodes.size(); ++i) {
            Node *node = &mNodes[i];
            MNRY_ASSERT(node->mParent || !node->mChildren.empty());
            if (node->mVertex) {
                MNRY_ASSERT(isValidFloat(node->mVertex->mHitPoint.x));
                MNRY_ASSERT(isValidFloat(node->mVertex->mHitPoint.y));
                MNRY_ASSERT(isValidFloat(node->mVertex->mHitPoint.z));
            }
        }
    );

    //
    // At this point, the children in each node are sorted by pixel location.
    // To get the final ordering which we want to write out, we just need to
    // do a depth first preorder traversal and write out vertices in that order.
    //
    std::vector<uint32_t> ordering;
    ordering.reserve(mNodes.size());
    depthFirstPreorderTraversal(&mNodes[0], &mNodes[0], &ordering);

    MNRY_ASSERT(ordering[0] == 0);
    MNRY_ASSERT(ordering.size() <= mNodes.size());
    MNRY_ASSERT(mNodes[0].mNumSubnodes == ordering.size());

    if (ordering.size() == 1) {
        return false;
    }

    //
    // copy DebugRays with the final ordering, whilst getting rid of temporary master root node
    //

    IdRemapper<uint32_t, uint32_t> remapper;

    size_t numSortedVerts = ordering.size() - 1;
    mSortedVertices.reserve(numSortedVerts);

    // note - skip over master root node so we're back to the original set of inputs
    for (size_t i = 1; i < ordering.size(); ++i) {
        Node *node = &mNodes[ordering[i]];
        DebugRayVertex *vert = node->mVertex;

        MNRY_ASSERT(vert->mId != NULL_RAY_VERTEX_ID && vert->mParentId != NULL_RAY_VERTEX_ID);

        if (vert->mParentId == 0) {
            MNRY_ASSERT(node->mParent == &mNodes[0]);
            vert->mParentId = NULL_RAY_VERTEX_ID;
        } else {
            vert->mParentId = remapper.remapId(vert->mParentId);
        }

        vert->mId = remapper.remapId(vert->mId);

        mSortedVertices.push_back(*vert);
        mSortedVertices.back().mNumSubnodes = node->mNumSubnodes;
    }

    MNRY_ASSERT(mSortedVertices.size() == numSortedVerts);
    MNRY_ASSERT(remapper.size() == numSortedVerts);

    //
    // clean up temp memory
    //
    mNodes.clear();

    //
    // build up table of primary vert entry points
    //
    MNRY_DURING_ASSERTS(size_t prevPrimaryRay = 0);

    for (size_t i = 0; i < mSortedVertices.size(); ++i) {
        DebugRayVertex const &curr = mSortedVertices[i];
        if (curr.mParentId == NULL_RAY_VERTEX_ID) {
            mPrimaryRayIndices.push_back(uint32_t(i));

            MNRY_DURING_ASSERTS
            (
                if (mPrimaryRayIndices.size() > 1) {
                    DebugRayVertex const &prev = mSortedVertices[prevPrimaryRay];

                    // debug validation - check everything is sorted by screen location
                    MNRY_ASSERT(prev.mId < curr.mId);
                    MNRY_ASSERT(prev.mScreenY * mWidth + prev.mScreenX <=
                               curr.mScreenY * mWidth + curr.mScreenX);

                    // debug validation - check the numbers of subnodes make sense
                    MNRY_ASSERT(prev.mNumSubnodes == i - prevPrimaryRay);
                }
            );

            MNRY_DURING_ASSERTS(prevPrimaryRay = i);
        }
    }

    //
    // transform everything into world space before saving
    //
    Mat4f render2world = DebugRayRecorder::sRender2World;
    if (render2world != Mat4f(one)) {

        for (size_t i = 0; i < mSortedVertices.size(); ++i) {

            // there should be no scaling in the render to world matrix so we can
            // use it as is to transform normals

            DebugRayVertex &vtx = mSortedVertices[i];
            vtx.mHitPoint  = transformPoint(render2world, vtx.mHitPoint);
            vtx.mNormal    = transform3x3(render2world, vtx.mNormal);
            vtx.mRayDiffX  = transformPoint(render2world, vtx.mRayDiffX);
            vtx.mRayDiffY  = transformPoint(render2world, vtx.mRayDiffY);
        }
    }

    return true;
}

void
DebugRayBuilder::cleanUp()
{
    mWidth = 0;
    mHeight = 0;
    mNodes.clear();
    mSortedVertices.clear();
    mPrimaryRayIndices.clear();
}

bool
DebugRayBuilder::exportDatabase(DebugRayDatabase *dst) const
{
    MNRY_ASSERT(dst);

    if (!mWidth || !mHeight || mSortedVertices.empty() || mPrimaryRayIndices.empty()) {
        return false;
    }

    dst->init(mWidth, mHeight, mSortedVertices, mPrimaryRayIndices);

    return true;
}

void
DebugRayBuilder::depthFirstPreorderTraversal(Node const *rootNode, Node *node, std::vector<uint32_t> *ordering) const
{
    // Remove any nodes which have no parent (i.e. dummy root node is the parent)
    // and no children, these represent rays which startNewRay was called on,
    // but weren't extended any further. Since these are only single points, don't
    // bother storing them. We avoid storing them in the final output by not adding
    // them to the ordering vector.
    if (node->mParent != rootNode || !node->mChildren.empty()) {
        ordering->push_back(uint32_t(node - rootNode));
    } else {
        node->mNumSubnodes = 0;
    }

    for (auto it = node->mChildren.begin(); it != node->mChildren.end(); ++it) {
        Node *child = *it;
        MNRY_ASSERT(child->mParent == node);
        depthFirstPreorderTraversal(rootNode, child, ordering);
        node->mNumSubnodes += child->mNumSubnodes;
    }
}

//----------------------------------------------------------------------------

DebugRayDatabase::Iterator::Iterator(DebugRayDatabase const &db, DebugRayFilter const *filter) :
    mDb(&db),
    mNumPrimariesSeen(0),
    mCurrY(0),
    mEndY(0),
    mCurrPrimaryIdx(0),
    mEndPrimaryIdx(0),
    mCurrVertex(nullptr),
    mEndVertex(nullptr),
    mJumpToVertexLoop(false),
    mIsDone(false)
{
    if (conditionFilter(filter ? *filter : DebugRayFilter(), &mFilter)) {
        mCurrY = mFilter.mViewport.lower.y;
        mEndY = mFilter.mViewport.upper.y;
        mIsDone = !getNextVertex();
    } else {
        mIsDone = true;
    }
}

bool
DebugRayDatabase::Iterator::conditionFilter(DebugRayFilter const &input, DebugRayFilter *output) const
{
    MNRY_ASSERT(output);

    output->mViewport.lower.x = (input.mViewport.lower.x == -1) ? 0 : min(input.mViewport.lower.x, int(mDb->mWidth) - 1);
    output->mViewport.lower.y = (input.mViewport.lower.y == -1) ? 0 : min(input.mViewport.lower.y, int(mDb->mHeight) - 1);
    output->mViewport.upper.x = (input.mViewport.upper.x == -1) ? int(mDb->mWidth) - 1 : min(input.mViewport.upper.x, int(mDb->mWidth) - 1);
    output->mViewport.upper.y = (input.mViewport.upper.y == -1) ? int(mDb->mHeight) -1 : min(input.mViewport.upper.y, int(mDb->mHeight) - 1);

    output->mStartPrimaryRay = (input.mStartPrimaryRay == uint32_t(-1)) ? 0 : input.mStartPrimaryRay;
    output->mEndPrimaryRay   = (input.mEndPrimaryRay == uint32_t(-1)) ? uint32_t(mDb->mPrimaryRayIndices.size() - 1) : min(input.mEndPrimaryRay, uint32_t(mDb->mPrimaryRayIndices.size() - 1));

    output->mTags = input.mTags;
    output->mMatchAll = input.mMatchAll ? 1 : 0;

    output->mMinDepth = (input.mMinDepth == uint32_t(-1)) ? 0 : input.mMinDepth;
    output->mMaxDepth = (input.mMaxDepth == uint32_t(-1)) ? 254 : input.mMaxDepth;

    return true;
}

bool
DebugRayDatabase::Iterator::getNextVertex()
{
    std::vector<uint32_t>::const_iterator startIt, endIt;
    uint32_t startScan, endScan, rayIdx;
    DebugRayVertex const *primaryRay;

    if (mJumpToVertexLoop) {
        mJumpToVertexLoop = false;
        goto NextRayVertex;
    }

    // find first valid vert which passes the filter
    while (mCurrY <= mEndY) {

        // find a vert greater than or equal to start x
        startScan = mCurrY * mDb->mWidth + mFilter.mViewport.lower.x;
        startIt = std::lower_bound(mDb->mScreenPositions.begin(), mDb->mScreenPositions.end(), startScan);
        if (startIt == mDb->mScreenPositions.end()) {
            return false;
        }

        // find a vert less than or equal to end x
        endScan = mCurrY * mDb->mWidth + mFilter.mViewport.upper.x;
        endIt = std::upper_bound(mDb->mScreenPositions.begin(), mDb->mScreenPositions.end(), endScan);

        mCurrPrimaryIdx = uint32_t(startIt - mDb->mScreenPositions.begin());
        mEndPrimaryIdx = uint32_t(endIt - mDb->mScreenPositions.begin());

        while (mCurrPrimaryIdx != mEndPrimaryIdx) {

            // filter by range
            if (mNumPrimariesSeen < mFilter.mStartPrimaryRay) {
                goto NextPrimaryRay;
            }

            if (mNumPrimariesSeen > mFilter.mEndPrimaryRay) {
                return false;
            }

            rayIdx = mDb->mPrimaryRayIndices[mCurrPrimaryIdx];
            primaryRay = &mDb->mVertices[rayIdx];

            mCurrVertex = primaryRay;
            mEndVertex = mCurrVertex + primaryRay->mNumSubnodes;

            while (mCurrVertex != mEndVertex) {

                // filter by depth
                if (mCurrVertex->mDepth != uint8_t(-1)) {
                    if (mCurrVertex->mDepth < mFilter.mMinDepth || mCurrVertex->mDepth > mFilter.mMaxDepth) {
                        goto NextRayVertex;
                    }
                }

                // filter by tags
                if (mFilter.mTags) {
                    uint32_t maskedTags = mCurrVertex->mUserTags & mFilter.mTags;
                    if ((!mFilter.mMatchAll && !maskedTags) || (mFilter.mMatchAll && (maskedTags != mFilter.mTags))) {
                        goto NextRayVertex;
                    }
                }

                // bingo, we are now pointing to a vert which survived all of the filters
                mJumpToVertexLoop = true;
                return true;

            NextRayVertex:
                ++mCurrVertex;
            }

        NextPrimaryRay:
            ++mCurrPrimaryIdx;
            ++mNumPrimariesSeen;
        }

        ++mCurrY;
    }

    return false;
}

void
DebugRayDatabase::init(uint32_t width, uint32_t height, std::vector<DebugRayVertex> const &vertices, std::vector<uint32_t> const &primaryRayIndices)
{
    cleanUp();

    mWidth = width;
    mHeight = height;
    mVertices = vertices;
    mPrimaryRayIndices = primaryRayIndices;

    buildScreenPositions();

    MNRY_ASSERT(isValid());
}

void
DebugRayDatabase::cleanUp()
{
    mWidth = 0;
    mHeight = 0;
    mVertices.clear();
    mPrimaryRayIndices.clear();
    mScreenPositions.clear();
}

DebugRayVertex const *
DebugRayDatabase::getRay(uint32_t idx) const
{
    if (idx < mVertices.size()) {
        return &mVertices[idx];
    }

    return nullptr;
}

#define CHECK_WITH_FAILURE(x)     if (!(x)) goto Failure

bool
DebugRayDatabase::load(char const *fileName)
{
    FILE *file = fopen(fileName, "rb");
    if (!file) {
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] Unable to open file \"" , fileName , "\".");
        return false;
    }

    cleanUp();

    uint32_t version = 0;
    CHECK_WITH_FAILURE( read(file, &version) );
    CHECK_WITH_FAILURE( version == DEBUG_RAYS_VERSION );

    CHECK_WITH_FAILURE( read(file, &mWidth) );
    CHECK_WITH_FAILURE( read(file, &mHeight) );

    uint32_t numVerts;
    CHECK_WITH_FAILURE( read(file, &numVerts) );

    if (numVerts) {
        mVertices.resize(numVerts);
        CHECK_WITH_FAILURE( readRawData(file, &mVertices[0], sizeof(DebugRayVertex) * numVerts) );
    }

    uint32_t numPrimaryRayIndices;
    CHECK_WITH_FAILURE( read(file, &numPrimaryRayIndices) );

    if (numPrimaryRayIndices) {
        mPrimaryRayIndices.resize(numPrimaryRayIndices);
        CHECK_WITH_FAILURE( readRawData(file, &mPrimaryRayIndices[0], sizeof(uint32_t) * numPrimaryRayIndices) );
    }

    buildScreenPositions();

    fclose(file);

    MNRY_ASSERT(isValid());

    return true;

Failure:
    scene_rdl2::logging::Logger::error("[MCRT-RENDER] Failed to load file \"" , fileName , "\".");
    cleanUp();
    fclose(file);
    return false;
}

bool
DebugRayDatabase::save(char const *fileName) const
{
    MNRY_ASSERT(isValid());

    FILE *file = fopen(fileName, "wb");
    if (!file) {
        scene_rdl2::logging::Logger::error("[MCRT-RENDER] Unable to create file \"" , fileName , "\".");
        return false;
    }

    CHECK_WITH_FAILURE( write(file, uint32_t(DEBUG_RAYS_VERSION)) );
    CHECK_WITH_FAILURE( write(file, mWidth) );
    CHECK_WITH_FAILURE( write(file, mHeight) );
    CHECK_WITH_FAILURE( write(file, uint32_t(mVertices.size())) );

    if (!mVertices.empty()) {
        CHECK_WITH_FAILURE( writeRawData(file, &mVertices[0], sizeof(DebugRayVertex) * mVertices.size()) );
    }

    CHECK_WITH_FAILURE( write(file, uint32_t(mPrimaryRayIndices.size())) );
    if (!mPrimaryRayIndices.empty()) {
        CHECK_WITH_FAILURE( writeRawData(file, &mPrimaryRayIndices[0], sizeof(uint32_t) * mPrimaryRayIndices.size()) );
    }

    // to allow future expansion of file format
    CHECK_WITH_FAILURE( write(file, uint32_t(0)) );

    fclose(file);
    return true;

Failure:
    scene_rdl2::logging::Logger::error("[MCRT-RENDER] Failed to save file \"" , fileName , "\".");
    fclose(file);
    return false;
}

#undef CHECK_WITH_FAILURE

bool 
DebugRayDatabase::isValid() const
{
    MNRY_DURING_ASSERTS
    (
        if (mVertices.empty()) {
            MNRY_ASSERT(mPrimaryRayIndices.empty());
            MNRY_ASSERT(mScreenPositions.empty());
            return true;
        }

        for (size_t i = 0; i < mVertices.size(); ++i) {
            DebugRayVertex const &vert = mVertices[i];
            MNRY_ASSERT(isValidFloat(vert.mHitPoint.x));
            MNRY_ASSERT(isValidFloat(vert.mHitPoint.y));
            MNRY_ASSERT(isValidFloat(vert.mHitPoint.z));
        }
    );

    return true;
}

void
DebugRayDatabase::buildScreenPositions()
{
    if (mPrimaryRayIndices.empty()) {
        return;
    }

    size_t numPrimaryRay = mPrimaryRayIndices.size();
    MNRY_ASSERT(mVertices.size() >= numPrimaryRay);

    mScreenPositions.resize(numPrimaryRay);

    for (size_t i = 0; i < numPrimaryRay; ++i) {
        DebugRayVertex const &vert = mVertices[mPrimaryRayIndices[i]];
        MNRY_ASSERT(vert.mParentId == NULL_RAY_VERTEX_ID);

        mScreenPositions[i] = vert.mScreenY * mWidth + vert.mScreenX;
    }
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

