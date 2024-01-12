// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "PbrTLState.h"
#include <moonray/rendering/pbr/light/Light.h>

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/mcrt_common/Ray.h>

#include <set>
#include <vector>

/// Comment this out to remove the debug ray instrumentation from the integrator.
#define ALLOW_DEBUG_RAY_RECORDING

namespace mm {
    class Mesh;
}

namespace moonray {
namespace pbr {

class DebugRayDatabase;

//
// Debug ray related helpers:
//

enum DebugRayTags
{
    // these should be kept sync with the BsdfLobe types

    // mutually exclusive
    TAG_REFLECTION      = 1 << 0,       //   1
    TAG_TRANSMISSION    = 1 << 1,       //   2

    // mutually exclusive
    TAG_DIFFUSE         = 1 << 2,       //   4
    TAG_GLOSSY          = 1 << 3,       //   8 
    TAG_SPECULAR        = 1 << 4,       //  16 

    TAG_ENVLIGHT        = 1 << 5,       //  32 
    TAG_AREALIGHT       = 1 << 6,       //  64 

    TAG_PRIMARY         = 1 << 7,       // 128 

    // ... and so on
};

#ifdef ALLOW_DEBUG_RAY_RECORDING

/// Call this to start recording a brand new ray. px and py are the integer 
/// pixel coordinates which the ray is being sent through.
#define RAYDB_START_NEW_RAY(tls, org, px, py) \
    StartNewDebugRay startNewRay(tls, org, px, py)

/// This extends the most recent ray on the specified tls. It must be called
/// *after* transfer has been called on the ray!
#define RAYDB_EXTEND_RAY(tls, ray, isect) \
    ExtendDebugRay extendRay(tls, ray, isect)

/// This is for recording environment light hits where we don't actually collide
/// with any geometry but still want to record the event as a hit for debugging
/// purposes. dist is a value in world units of how far away the from the ray
/// origin to place the hit event. The input ray is assumed to be untransferred.
#define RAYDB_EXTEND_RAY_NO_HIT(tls, ray, dist) \
    ExtendDebugRay extendRay(tls, ray, dist)

/// This function updates the contribution for the most recent ray vertex on the
/// specified tls.
#define RAYDB_SET_CONTRIBUTION(tls, c) \
    if (moonray::pbr::DebugRayRecorder::isRecordingEnabled()) \
        debugRaysConvertAndSet(&tls->mRayVertexStack.back()->mContribution, (c))

/// This function updates the tags associated with the most recent ray vertex on
/// the specified tls.
#define RAYDB_SET_TAGS(tls, t) \
    if (moonray::pbr::DebugRayRecorder::isRecordingEnabled()) \
        tls->mRayVertexStack.back()->mUserTags = uint32_t(t)

#define RAYDB_ADD_TAGS(tls, t) \
    if (moonray::pbr::DebugRayRecorder::isRecordingEnabled()) \
        tls->mRayVertexStack.back()->mUserTags |= uint32_t(t)

#else   // #ifdef ALLOW_DEBUG_RAY_RECORDING

#define RAYDB_START_NEW_RAY(tls, org, px, py)     
#define RAYDB_EXTEND_RAY(tls, ray, isect)          
#define RAYDB_EXTEND_RAY_NO_HIT(tls, ray, dist) 
#define RAYDB_SET_CONTRIBUTION(tls, c)          
#define RAYDB_SET_TAGS(tls, t)                  
#define RAYDB_ADD_TAGS(tls, t)                  

#endif  // #ifdef ALLOW_DEBUG_RAY_RECORDING

///
/// A RayVertex structure serves a dual responsibilities. During the recording
/// phase, it's the internal structure we use to build up the collections of 
/// hitpoints along each ray. It also serves as the final representation for
/// each hitpoint in the final built DebugRayDatabase.
///

struct DebugRayVertex
{
    //
    // filled in by the app after it calls DebugRayRecorder::extendRay
    // (76 bytes)
    //

    scene_rdl2::math::Vec3f       mHitPoint;          ///< in world space
    scene_rdl2::math::Vec3f       mNormal;            ///< from differential geometry
    scene_rdl2::math::Vec3f       mRayDiffX;          ///< final world space position of ray differential in x
    scene_rdl2::math::Vec3f       mRayDiffY;          ///< final world space position of ray differential in y
    scene_rdl2::math::Vec3f       mFree;              ///< TBD
    scene_rdl2::math::Vec3f       mContribution;      ///< shader eval with lighting
    uint32_t    mUserTags;          ///< bitfield of user specified tags

    //
    // filled in automatically by the system - do not edit
    // (20 bytes)
    //

    uint32_t    mId;                ///< unique id per ray vertex
    uint32_t    mParentId;          ///< 0 means root or no parent
    uint16_t    mThreadId;          ///< thread id we were rendered on
    uint8_t     mDepth;             ///< which bounce are we on
    uint8_t     mPad;               

    uint16_t    mScreenX;           ///< if this is -1 then we are a dummy ray (record and build time only)
    uint16_t    mScreenY;           

    union {
        uint32_t    mNumSubnodes;       ///< how many nodes in this entire branch, including the current node
        uint16_t    mParentThreadId;    ///< only used by the recorder and builder, not needed when in database form
    };

    //
    // these are only valid after rays have been sent though the debug ray builder
    //
    bool        isRoot() const      { return mParentId == uint32_t(-1); }
    bool        isLeaf() const      { return mNumSubnodes == 1; }
};

static_assert(sizeof(DebugRayVertex) == 96, "DebugRayVertex is an unexpected size");

//----------------------------------------------------------------------------

///
/// The steps needed to generate a ray database are:
/// 
///     1. Record rays using the DebugRayRecorder class. This can be done in
///        parallel using a separate DebugRayRecorder instance for each separate 
///        thread.
///     2. Take all the recorded ray data and use the DebugRayBuilder class to 
///        condition it into a queryable ray database.
///     3. Once the data is conditioned, we can store it in a DebugRayDatabase
///        class which gives us the ability to submit queries against it and also
///        serialize the data.
///
/// Said another way, here is how the data flows:
///      DebugRayRecorder -> DebugRayBuilder -> DebugRayDatabase
///
/// The usage of the DebugRayRecorder is hidden behind the RAYDB_* macros, and
/// generally you shouldn't need to interact with it directly, although it is
/// fine to do so if you want. 
///
/// Each thread will have its own instance of a DebugRayRecorder in thread local
/// storage. Data from each instance then gets merged inside of the DebugRayRecorder.
///

class DebugRayRecorder
{
    friend class DebugRayBuilder;

public:
                    DebugRayRecorder(uint32_t threadId);
                    ~DebugRayRecorder() { cleanUp(); }

    /// Recording will continue until stopRecording is called or we run out of space.
    /// Use maxVerticesPerThread if you want to put a cap on the number of rays
    /// which can be recorded. When we run out of entry space, we return the dummy
    /// ray. Client code should be none the wiser.
    void            record();
    void            stopRecording();

    bool            isRecording() const             { return mNextId < mEndId; }

    void            cleanUp();

    size_t          getNumRayVertices() const       { return mNextId; }

    DebugRayVertex *startNewRay(scene_rdl2::math::Vec3f const &origin, unsigned screenX,
                                unsigned screenY, uint32_t userTags = 0);
    DebugRayVertex *extendRay(DebugRayVertex const *parent);

    ///
    /// Master on/off switch for recording. Calling the record function won't 
    /// do anything if these params aren't setup beforehand. Call disableRecording
    /// to prevent future record calls from succeeding.
    /// 
    /// Optionally pass in:
    /// - An active viewport, otherwise the assumption is we record the entire screen.
    ///   Coordinates are minx, miny, maxx, maxy inclusive.
    /// - A cap on the amount of rays we can record on each thread, otherwise
    ///   there no limit, but beware that the machine may have to switch to swap
    ///   for even very modest images.
    ///
    static void     enableRecording(scene_rdl2::math::BBox2i viewport,
                                    scene_rdl2::math::Mat4f const &render2world,
                                    uint32_t maxVerticesPerThread = 0x200000);

    static void     disableRecording();
    static bool     isRecordingEnabled()    { return sRecordingEnabled; }

    static scene_rdl2::math::Mat4f const &getRenderToWorldMatrix()   { return sRender2World; }

    // hide copy constructor and assignment operator
                    DebugRayRecorder(DebugRayRecorder const &other) = delete;
    DebugRayRecorder &operator = (DebugRayRecorder const &other) = delete;

private:
    DebugRayVertex *allocRayVertex();

    DebugRayVertex *getRay(uint32_t id);
    DebugRayVertex const *getRay(uint32_t id) const    { return const_cast<DebugRayRecorder *>(this)->getRay(id); }

    const uint16_t  mThreadId;

    uint32_t        mNextId;
    uint32_t        mEndId;     // one past end

    typedef std::vector<DebugRayVertex> VertexBlock;
    std::vector<VertexBlock *>  mVertexBlocks;

    /// Return this if we are starting a new ray outside of the active area.
    DebugRayVertex  mDummyVertex;

    static scene_rdl2::math::BBox2i sActiveViewport;
    static scene_rdl2::math::Mat4f sRender2World;
    static unsigned sMaxVertsPerThread;
    static bool sRecordingEnabled;
};

//----------------------------------------------------------------------------

///
/// This class takes a number of RayRecorder instances, merges and sorts all
/// the data in such a way that we can submit queries against it.
/// 
/// Each recorder will contain a group of ray hitpoints with connectivity
/// information. There is no limitations on which hitpoints a particular recorder
/// can record. For example, portions of the same primary ray may be propagated on 
/// different threads in the renderer. Since we record in a thread local fashion,
/// a single ray may jump around from recorder instance to recorder instance -
/// this is fine, it's supported behavior.
/// 
/// Internally, data is sorted into a big hierarchical tree. The nodes in this
/// tree are laid out linearly in memory in the mVertices array. Primary rays
/// appear at the root, first hits next, second hits another level underneath the
/// first hits, and so on. All primary rays are sorted amongst themselves based
/// on screen coordinate in a scanline by scanline fashion. We generate 2
/// additional "indexes" which accelerate lookups.
/// 
///     1. We stored a sorted vector of all primary ray indices in the member
///        mPrimaryRayIndices. Each element in this vector gives the offset of
///        each successive primary ray (sorted by screen location) in the mVertices
///        array.
///     2. To accelerate finding primary rays for particular image locations,
///        we have another sorted vector called mScreenPositions. This simply
///        stores the screen locations for each primary ray in mPrimaryRayIndices
///        so that we can locate primary rays at any location without having to 
///        touch the mVertices array at all.
/// 
/// After the build step has been executed, you need to call exportDatabase to
/// initialize an actual DebugRayDatabase object. Once done, the DebugRayBuilder
/// doesn't need to persist in memory any longer.
/// 

class DebugRayBuilder
{
public:
    /// Calling this function mutates the DebugRayRecorders but not in any harmful
    /// way, i.e. calling build twice in a row will still produce the correct results.

    // @@@ TODO, Pass in a matrix to go from shade space to world space before
    // saving data out.
    bool    build(unsigned width, unsigned height, std::vector<DebugRayRecorder *> const &recorders);
    void    cleanUp();

    bool    exportDatabase(DebugRayDatabase *dst) const;

private:
    struct Node;

    // fills in the ordering vector with the final desired locations of each DebugRayVertex
    void    depthFirstPreorderTraversal(Node const *rootNode, Node *node, std::vector<uint32_t> *ordering) const;

    struct SortChildren
    {
        finline bool operator()(Node const *a, Node const *b) const
        {
            DebugRayVertex const *vertA = a->mVertex;
            DebugRayVertex const *vertB = b->mVertex;

            if (vertA->mScreenY != vertB->mScreenY)     return vertA->mScreenY < vertB->mScreenY;
            if (vertA->mScreenX != vertB->mScreenX)     return vertA->mScreenX < vertB->mScreenX;
            if (vertA->mDepth != vertB->mDepth)         return vertA->mDepth < vertB->mDepth;
            if (vertA->mThreadId != vertB->mThreadId)   return vertA->mThreadId < vertB->mThreadId;

            MNRY_ASSERT(vertA->mId != vertB->mId);
            return vertA->mId < vertB->mId;
        }
    };

    struct Node
    {
        DebugRayVertex * mVertex;
        Node *      mParent;
        uint32_t    mNumSubnodes; // node count for all subnodes from this point in the tree include self
        std::set<Node *, SortChildren>  mChildren;
    };

    uint32_t                mWidth;
    uint32_t                mHeight;

    std::vector<Node>       mNodes;
    std::vector<DebugRayVertex>  mSortedVertices;

    // this vector indexes into the mSortedVertices vector
    std::vector<uint32_t>   mPrimaryRayIndices;
};

//----------------------------------------------------------------------------

///
/// DebugRayFilter is the structure a user would use to generate a ray database
/// query. Each member relates to another way to filter down the potentially
/// huge ray set. When iterating through the data using DebugRayDatabase::Iterator,
/// only ray vertices which have passed all filters will be returned.
/// 

struct DebugRayFilter
{
    // initialize everything to uint32_t(-1), meaning don't to any filtering
    DebugRayFilter() { memset(this, 0xff, sizeof(DebugRayFilter)); mTags = mMatchAll = 0; }

    // only display rays which fall inside of this rectangle, rect is inclusive of all edges
    scene_rdl2::math::BBox2i mViewport;

    // for rays which pass the above test, these fields allow us to define a range of which rays
    // we want to view, end is non-inclusive (i.e. one past the end)
    uint32_t mStartPrimaryRay;
    uint32_t mEndPrimaryRay;

    // filter by user tags
    uint32_t mTags;             // tags to check for
    uint32_t mMatchAll;         // set to zero to match any tag, or non-zero to match all tags

    // for rays which pass the above tests, we can limit which bounces we want to view
    uint32_t mMinDepth;
    uint32_t mMaxDepth;
};

///
/// The DebugRayDatabase class is really just a container for the data generated
/// by the DebugRayBuilder. See DebugRayBuilder for further comments about what
/// each member vector contains.
///
/// In addition, it provides filtering and iteration functionality, as well as
/// the ability to serialize/deserialize itself.
///

class DebugRayDatabase
{
public:
    // iteration is thread safe as long as we are using different iterator
    // instances on separate threads
    class Iterator
    {
    public:
        // filter is optional
                        Iterator(DebugRayDatabase const &db, DebugRayFilter const *filter = nullptr);

        bool            isDone() const      { return mIsDone; }

        // pre-increment
        Iterator &      operator++()        { if (!mIsDone) mIsDone = !getNextVertex(); return *this; }
        DebugRayVertex const &operator*() const  { return *mCurrVertex; }

    private:
        bool            conditionFilter(DebugRayFilter const &input, DebugRayFilter *output) const;
        bool            getNextVertex();

        DebugRayDatabase const *mDb;
        DebugRayFilter  mFilter;

        uint32_t        mNumPrimariesSeen;
        uint32_t        mCurrY;                 ///< current scanline y
        uint32_t        mEndY;                  ///< final scanline y
        uint32_t        mCurrPrimaryIdx;        ///< index within current scanline
        uint32_t        mEndPrimaryIdx;         ///< index within current scanline
        DebugRayVertex const *mCurrVertex;           ///< ray segment within current primary ray
        DebugRayVertex const *mEndVertex;            ///< one past the end of the current primary ray branch

        bool            mJumpToVertexLoop;
        bool            mIsDone;
    };

                        DebugRayDatabase()      { cleanUp(); }
                        ~DebugRayDatabase()     { cleanUp(); }

    void                init(uint32_t width, uint32_t height, std::vector<DebugRayVertex> const &vertices, std::vector<uint32_t> const &primaryRayIndices);
    void                cleanUp();

    bool                empty() const           { return mPrimaryRayIndices.empty(); }

    std::vector<DebugRayVertex> const &getRays() const               { return mVertices; }
    std::vector<uint32_t> const &getPrimaryRayIndices() const   { return mPrimaryRayIndices; }

    DebugRayVertex const *getRay(uint32_t id) const;

    bool                load(char const *fileName);
    bool                save(char const *fileName) const;

    bool                isValid() const;

private:
    void                buildScreenPositions();

    uint32_t            mWidth;
    uint32_t            mHeight;

    std::vector<DebugRayVertex>  mVertices;

    /// Each element in this vector is the index of a new primary ray the mVertices vector.
    std::vector<uint32_t>   mPrimaryRayIndices;

    /// This vector is one-to-one with mPrimaryRayIndices, it contains the screen pixel
    /// location where each of these primary vertices start.
    std::vector<uint32_t>   mScreenPositions;
};

//----------------------------------------------------------------------------

finline void  
debugRaysConvertAndSet(scene_rdl2::math::Vec3f *dst, scene_rdl2::math::Vec3f const &src)
{
    *dst = src;
}

finline void 
debugRaysConvertAndSet(scene_rdl2::math::Vec3f *dst, scene_rdl2::math::Color const &src)
{
    dst->x = src.r;
    dst->y = src.g;
    dst->z = src.b;
}

struct StartNewDebugRay
{
    finline StartNewDebugRay(pbr::TLState *pbrTls, scene_rdl2::math::Vec3f o,
            unsigned px, unsigned py) :
        mTls(pbrTls)
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            MNRY_ASSERT(mTls->mRayVertexStack.empty());
            mTls->mRayVertexStack.push_back(mTls->mRayRecorder->startNewRay(o,
                    px, py, TAG_PRIMARY));
        }
    }
          
    finline ~StartNewDebugRay()
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            mTls->mRayVertexStack.pop_back();
            MNRY_ASSERT(mTls->mRayVertexStack.empty());
        }
    }
          
    pbr::TLState *mTls;
};


#ifdef ALLOW_DEBUG_RAY_RECORDING

finline void
addRayVertex(pbr::TLState *tls,
             scene_rdl2::math::Vec3f const &hitPoint,
             scene_rdl2::math::Vec3f const &hitNormal,
             scene_rdl2::math::Vec3f const &rayDiffX,
             scene_rdl2::math::Vec3f const &rayDiffY)
{
    MNRY_ASSERT(!tls->mRayVertexStack.empty());

    tls->mRayVertexStack.push_back(tls->mRayRecorder->extendRay(tls->mRayVertexStack.back()));

    DebugRayVertex *vert = tls->mRayVertexStack.back();
    vert->mHitPoint = hitPoint;
    vert->mNormal = hitNormal;
    vert->mRayDiffX = rayDiffX;
    vert->mRayDiffY = rayDiffY;

#ifdef DEBUG
    // Debug functionality to isolate individual vertices of a ray differential.
    // You can find the coordinate of a debug ray mesh in in mm_view and set
    // vertToFind to that value to retrieve the callstack for how we got there.
    static scene_rdl2::math::Vec3f vertToFind = scene_rdl2::math::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);

    if (vertToFind != scene_rdl2::math::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX)) {

        scene_rdl2::math::Vec3f halfDx = (rayDiffX - hitPoint) * 0.5f;
        scene_rdl2::math::Vec3f halfDy = (rayDiffY - hitPoint) * 0.5f;
        scene_rdl2::math::Vec3f p0 = hitPoint - halfDx - halfDy;
        scene_rdl2::math::Vec3f p1 = hitPoint + halfDx - halfDy;
        scene_rdl2::math::Vec3f p2 = hitPoint - halfDx + halfDy;
        scene_rdl2::math::Vec3f p3 = hitPoint + halfDx + halfDy;

        scene_rdl2::math::Vec3f hit = transformPoint(DebugRayRecorder::getRenderToWorldMatrix(), hitPoint);

        p0  = transformPoint(DebugRayRecorder::getRenderToWorldMatrix(), p0);
        p1  = transformPoint(DebugRayRecorder::getRenderToWorldMatrix(), p1);
        p2  = transformPoint(DebugRayRecorder::getRenderToWorldMatrix(), p2);
        p3  = transformPoint(DebugRayRecorder::getRenderToWorldMatrix(), p3);

        const float eps = 0.01f;

        if (isEqual(vertToFind, hit, eps) ||
            isEqual(vertToFind, p0,  eps) ||
            isEqual(vertToFind, p1,  eps) ||
            isEqual(vertToFind, p2,  eps) ||
            isEqual(vertToFind, p3,  eps)) {
            scene_rdl2::logging::Logger::info("Hit debug ray vertex at " , vertToFind ,
                         ", hit dx = " , (rayDiffX - hitPoint) ,
                         ", hit dy = " , (rayDiffY - hitPoint));
        }
    }
#endif
}


struct ExtendDebugRay
{
    // assumes transferred ray
    finline ExtendDebugRay(pbr::TLState *pbrTls,
            mcrt_common::RayDifferential const &ray, shading::Intersection const &isect) :
        mTls(pbrTls)
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            addRayVertex(mTls,
                         ray.getOrigin(),
                         isect.getN(),
                         ray.hasDifferentials() ? ray.getOriginX() : ray.getOrigin(),
                         ray.hasDifferentials() ? ray.getOriginY() : ray.getOrigin());
        }
    }

    // assumes untransferred ray
    finline ExtendDebugRay(pbr::TLState *pbrTls,
            mcrt_common::RayDifferential const &ray, float dist) :
        mTls(pbrTls)
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            MNRY_ASSERT(dist < sDistantLightDistance);
            mcrt_common::RayDifferential rd = ray;
            rd.transfer(ray.getOrigin() + ray.getDirection() * dist, -ray.getDirection());
            addRayVertex(mTls,
                         rd.getOrigin(),
                         -rd.getDirection(),
                         rd.hasDifferentials() ? rd.getOriginX() : rd.getOrigin(),
                         rd.hasDifferentials() ? rd.getOriginY() : rd.getOrigin());
        }
    }

    // overload version of "with hit" constructor which accepts a mcrt_common::Ray
    // instead of a mcrt_common::RayDifferential
    finline ExtendDebugRay(pbr::TLState *pbrTls,
            mcrt_common::Ray const &ray, shading::Intersection const &isect) :
        mTls(pbrTls)
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            scene_rdl2::math::ReferenceFrame fr(isect.getN());
            float dist = ray.getEnd();
            float scale = dist * 0.1f;
            addRayVertex(mTls,
                         isect.getP(),
                         isect.getN(),
                         isect.getP() + fr.getX() * scale,
                         isect.getP() + fr.getY() * scale);
        }
    }

    // overload version of "NO_HIT" constructor which accepts a mcrt_common::Ray
    // instead of a mcrt_common::RayDifferential
    finline ExtendDebugRay(pbr::TLState *pbrTls,
            mcrt_common::Ray const &ray, float dist) :
        mTls(pbrTls)
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            MNRY_ASSERT(dist < sDistantLightDistance);
            scene_rdl2::math::Vec3f hitPoint = ray.getOrigin() + ray.getDirection() * dist;
            scene_rdl2::math::Vec3f hitNormal = -ray.getDirection();
            scene_rdl2::math::ReferenceFrame fr(hitNormal);
            float scale = dist * 0.1f;
            addRayVertex(mTls,
                         hitPoint,
                         hitNormal,
                         hitPoint + fr.getX() * scale,
                         hitPoint + fr.getY() * scale);
        }
    }


    finline ~ExtendDebugRay()
    {
        if (DebugRayRecorder::isRecordingEnabled()) {
            mTls->mRayVertexStack.pop_back();
            MNRY_ASSERT(!mTls->mRayVertexStack.empty());
        }
    }
          
    pbr::TLState *mTls;
};

#endif  // #ifdef ALLOW_DEBUG_RAY_RECORDING

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

