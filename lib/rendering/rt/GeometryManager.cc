// Copyright 2023-2024 DreamWorks Animation LLC and Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/geom/prim/NamedPrimitive.h>
#include <moonray/rendering/geom/prim/VdbVolume.h>

#include "GeometryManager.h"
#include "GeomContext.h"

#include <moonray/common/time/Ticker.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/BakedAttribute.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/TransformedPrimitive.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/Curves.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/geom/prim/Sphere.h>
#include <moonray/rendering/geom/prim/VolumeAssignmentTable.h>
#include <moonray/rendering/pbr/camera/PerspectiveCamera.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/RootShader.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/stdmemory.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <string.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

#include <malloc.h>    // malloc_trim

namespace moonray {
namespace rt {

using namespace scene_rdl2::math;
using namespace scene_rdl2::util;

using Timer = time::TimerAverageDouble;

class PrimitiveCollector : public geom::PrimitiveVisitor
{
public:
    explicit PrimitiveCollector(geom::SharedPrimitiveSet& sharedPrimitives,
                                geom::InternalPrimitiveList& primitivesToTessellate,
                                geom::InternalPrimitiveList& curvePrimitives):
        mSharedPrimitives(sharedPrimitives),
        mPrimitivesToTessellate(primitivesToTessellate),
        mCurvePrimitives(curvePrimitives)
    {}

    virtual void visitPrimitive(geom::Primitive& p) override
    {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        if (pImpl->getType() != geom::internal::Primitive::QUADRIC &&
            pImpl->getType() != geom::internal::Primitive::CURVES) {
            mPrimitivesToTessellate.push_back(pImpl);
        }

        if (pImpl->getType() == geom::internal::Primitive::CURVES) {
            mCurvePrimitives.push_back(pImpl);
        }
    }

    virtual void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override
    {
        pg.forEachPrimitive(*this);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t) override
    {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitInstance(geom::Instance& i) override
    {
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            ref->getPrimitive()->accept(*this);
        }
    }

private:
    geom::SharedPrimitiveSet& mSharedPrimitives;
    geom::InternalPrimitiveList& mPrimitivesToTessellate;
    geom::InternalPrimitiveList& mCurvePrimitives;
};

class GeometryManagerPrimitiveVisitor : public geom::PrimitiveVisitor
{
public:
    GeometryManagerPrimitiveVisitor()
    {}

    virtual void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override
    {
        pg.forEachPrimitive(*this);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t) override
    {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitInstance(geom::Instance& i) override
    {
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            ref->getPrimitive()->accept(*this);
        }
    }

protected:
    geom::SharedPrimitiveSet mSharedPrimitives;
};

class RdlGeometrySetter : public GeometryManagerPrimitiveVisitor
{
public:
    explicit RdlGeometrySetter(const scene_rdl2::rdl2::Geometry* geometry):
        mGeometry(geometry)
    {}

    virtual void visitPrimitive(geom::Primitive& p) override
    {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);

        // tessellation statistics needs this for printout
        pImpl->setRdlGeometry(mGeometry);
    }

private:
    const scene_rdl2::rdl2::Geometry* mGeometry;
};

class ShadowLinkingSetter : public GeometryManagerPrimitiveVisitor
{
public:
    explicit ShadowLinkingSetter(const scene_rdl2::rdl2::Layer* layer,
                                 const std::vector<ShadowExclusionMapping>& mappings):
        mLayer(layer),
        mShadowExclusionMappings(mappings)
    {}

    virtual void visitPrimitive(geom::Primitive& p) override
    {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);

        if (pImpl->getType() == geom::internal::Primitive::INSTANCE) {
            return;
        }

        geom::internal::NamedPrimitive* np =
            static_cast<geom::internal::NamedPrimitive*>(pImpl);
        np->resetShadowLinking();

        // get caster ids from primitive
        std::unordered_set<int> casterIds;
        np->getUniqueAssignmentIds(casterIds);
        for (int casterId : casterIds) {
            if (casterId < 0) {
                continue;
            }

            // find caster id in mappings
            const ShadowExclusionMapping* mappingPtr = nullptr;
            for (auto& mapping : mShadowExclusionMappings) {
                if (mapping.mCasterParts.empty() ||
                    mapping.mCasterParts.find(casterId) != mapping.mCasterParts.end()) {
                    mappingPtr = &mapping;
                    break;
                }
            }
            // TO-DO: should we check that it doesn't occur a 2nd time?

            const scene_rdl2::rdl2::ShadowSet* shadowSet = mLayer->lookupShadowSet(casterId);
            const scene_rdl2::rdl2::ShadowReceiverSet* shadowReceiverSet = mLayer->lookupShadowReceiverSet(casterId);
            bool complementShadowReceiverSet = (shadowReceiverSet &&
                shadowReceiverSet->get(scene_rdl2::rdl2::ShadowReceiverSet::sComplementKey));

            // create shadow linking if needed
            if (shadowSet || shadowReceiverSet || mappingPtr) {
                np->createShadowLinking(casterId, complementShadowReceiverSet);
            }

            // add lights to shadow set
            if (shadowSet) {
                const scene_rdl2::rdl2::SceneObjectVector &lights = shadowSet->getLights();
                for (auto& light : lights) {
                    np->addShadowLinkedLight(casterId, static_cast<const scene_rdl2::rdl2::Light *>(light));
                }
            }

            // Add mappings between casters and receivers (via rdla ShadowReceiverSet mechanism)
            if (shadowReceiverSet) {
                const scene_rdl2::rdl2::SceneObjectIndexable& receivers = shadowReceiverSet->getGeometries();
                for (auto& sceneObject : receivers) {
                    scene_rdl2::rdl2::Geometry* receiverGeom = sceneObject->asA<scene_rdl2::rdl2::Geometry>();
                    int receiverId = mLayer->getAssignmentId(receiverGeom, "");
                    if (receiverId != -1) {
                        // No parts, just add the geometry
                        np->addShadowLinkedGeom(casterId, receiverId);
                    } else {
                        // Geometry has parts - add them individually
                        for (auto it = mLayer->begin(receiverGeom); it != mLayer->end(receiverGeom); ++it) {
                            receiverId = *it;
                            np->addShadowLinkedGeom(casterId, receiverId);
                        }
                    }
                }
            }

            // Add mappings between casters and receivers (via geometry label attributes mechanism)
            if (mappingPtr) {
                for (auto& labeledGeoSet : mappingPtr->mLabeledGeoSets) {
                    for (auto& receiverGeom : labeledGeoSet->mGeometries) {
                        int receiverId = mLayer->getAssignmentId(receiverGeom, "");
                        if (receiverId != -1) {
                            // No parts, just add the geometry
                            np->addShadowLinkedGeom(casterId, receiverId);
                        } else {
                            // Geometry has parts - add them individually
                            for (auto it = mLayer->begin(receiverGeom); it != mLayer->end(receiverGeom); ++it) {
                                receiverId = *it;
                                np->addShadowLinkedGeom(casterId, receiverId);
                            }
                        }
                    }
                }
            }
        }
    }

private:
    const scene_rdl2::rdl2::Layer* mLayer;
    const std::vector<ShadowExclusionMapping>& mShadowExclusionMappings;
};

// This visitor performs two functions:
//   * Sets the assignment on a primitive: either a volume assignment or a surface assignment.
//   * Initializes the transform of instance primitives.
// Both these tasks must be done before initializing the VolumeAssignmentTable
// and building the BVH.
class AssignmentAndXformSetter : public GeometryManagerPrimitiveVisitor
{
public:
    explicit AssignmentAndXformSetter(const scene_rdl2::rdl2::Layer* layer):
        mLayer(layer),
        mHasSurfaceAssignment(false),
        mHasVolumeAssignment(false)
    {}

    virtual void visitInstance(geom::Instance& i) override
    {
        geom::internal::Instance* pInstance =
            static_cast<geom::internal::Instance*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&i));
        // Initializing the xform decomposes it. This must be done after instances have been
        // generated but before initializing the VolumeAssignmentTable and building the BVH.
        pInstance->initializeXform();
        const std::shared_ptr<geom::SharedPrimitive>& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            AssignmentAndXformSetter as(mLayer);
            ref->getPrimitive()->accept(as);
            // store if the reference contains volumes or surfaces
            ref->setHasSurfaceAssignment(as.getHasSurfaceAssignment());
            ref->setHasVolumeAssignment(as.getHasVolumeAssignment());
        }
    }

    virtual void visitBox(geom::Box& b) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&b));
        getAssignments(*pImpl);
    }

    virtual void visitCurves(geom::Curves& c) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&c));
        getAssignments(*pImpl);
    }

    virtual void visitPoints(geom::Points& p) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p));
        getAssignments(*pImpl);
    }

    virtual void visitPolygonMesh(geom::PolygonMesh& p) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p));
        getAssignments(*pImpl);
    }

    virtual void visitSphere(geom::Sphere& s) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s));
        getAssignments(*pImpl);
    }

    virtual void visitSubdivisionMesh(geom::SubdivisionMesh& s) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s));
        getAssignments(*pImpl);
    }

    virtual void visitVdbVolume(geom::VdbVolume& v) override
    {
        geom::internal::NamedPrimitive* pImpl =
            static_cast<geom::internal::NamedPrimitive*>(
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&v));
        getAssignments(*pImpl);
    }

    void getAssignments(const geom::internal::NamedPrimitive &np)
    {
        if (!mHasVolumeAssignment) {
            mHasVolumeAssignment = np.hasVolumeAssignment(mLayer);
        }
        if (!mHasSurfaceAssignment) {
            mHasSurfaceAssignment = np.hasSurfaceAssignment(mLayer);
        }
    }

    bool getHasVolumeAssignment() { return mHasVolumeAssignment; }
    bool getHasSurfaceAssignment() { return mHasSurfaceAssignment; }

private:
    const scene_rdl2::rdl2::Layer* mLayer;
    bool mHasSurfaceAssignment;
    bool mHasVolumeAssignment;
};

class TransformConcatenator : public geom::PrimitiveVisitor
{
public:
    explicit TransformConcatenator(const geom::MotionBlurParams& motionBlurParams)
    : mMotionBlurParams(motionBlurParams)
    {
    }

    virtual void visitPrimitive(geom::Primitive& p) override
    {
        geom::internal::PrimitivePrivateAccess::transformPrimitive(&p, mMotionBlurParams,
            shading::XformSamples(p.getMotionSamplesCount(), Xform3f(scene_rdl2::math::one)));
    }

private:
    geom::MotionBlurParams mMotionBlurParams;
};

// This visitor performs two functions, both related to instancing.
//   * Sets the "isReference" flag on referenced primitives
//   * Sets the adpative tessellation error on referenced primitives that
//     support adaptive tessellation.
class ReferenceSetter : public geom::PrimitiveVisitor
{
public:
    explicit ReferenceSetter(float val, bool isReference) :
        mVal(val),
        mIsReference(isReference)
    {}

    virtual void visitPrimitiveGroup(geom::PrimitiveGroup& pg)
    {
        pg.forEachPrimitive(*this);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t)
    {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitPolygonMesh(geom::PolygonMesh& p)
    {
        p.setAdaptiveError(mVal);

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        pImpl->setIsReference(mIsReference);
    }

    virtual void visitSubdivisionMesh(geom::SubdivisionMesh& s)
    {
        s.setAdaptiveError(mVal);

        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&s);
        pImpl->setIsReference(mIsReference);
    }

    virtual void visitVdbVolume(geom::VdbVolume& g)
    {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&g);
        pImpl->setIsReference(mIsReference);
    }

private:
    float mVal;
    bool mIsReference;
};

class PrimitiveAttributeChecker : public GeometryManagerPrimitiveVisitor
{
public:
    PrimitiveAttributeChecker(const scene_rdl2::rdl2::Geometry* geometry,
        const scene_rdl2::rdl2::Layer* layer):
        mGeometry(geometry), mLayer(layer)
    {}

    virtual void visitPrimitive(geom::Primitive& p) override
    {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
        if (pImpl->getType() != geom::internal::Primitive::INSTANCE) {
            geom::internal::NamedPrimitive* np =
                static_cast<geom::internal::NamedPrimitive*>(pImpl);
            std::unordered_set<int> assignmentIds;
            np->getUniqueAssignmentIds(assignmentIds);
            for (int assignmentId : assignmentIds) {
                if (assignmentId < 0) {
                    continue;
                }
                const scene_rdl2::rdl2::Material* s = mLayer->lookupMaterial(assignmentId);
                if (s == nullptr) {
                    continue;
                }
                const shading::AttributeTable* table =
                    s->get<shading::RootShader>().getAttributeTable();
                for (const auto& key : table->getRequiredAttributes()) {
                    // intentionally omit the check of wireframe related
                    // required attribute since they are filled in during
                    // postIntersect stage instead of render prep stage
                    if (key == shading::StandardAttributes::sPolyVertexType ||
                        key == shading::StandardAttributes::sNumPolyVertices) {
                        continue;
                    }
                    if (!np->hasAttribute(key)) {
                        // geom part missing attribute key shader require
                        mGeometry->error("part \"",
                            mLayer->lookupGeomAndPart(assignmentId).second,
                            "\" fail to provide primitive attribute \"",
                            key.getName(), "\" required by shading network \"",
                            s->getName(), "\"");
                    }
                }
                const scene_rdl2::rdl2::Displacement* d = mLayer->lookupDisplacement(assignmentId);
                if (d == nullptr) {
                    continue;
                }
                table = d->get<shading::RootShader>().getAttributeTable();
                for (const auto& key : table->getRequiredAttributes()) {
                    if (!np->hasAttribute(key)) {
                        // geom part missing attribute key shader require
                        mGeometry->error("part \"",
                            mLayer->lookupGeomAndPart(assignmentId).second,
                            "\" fail to provide primitive attribute \"",
                            key.getName(), "\" required by displacement network \"",
                            d->getName(), "\"");
                    }
                }
            }
        }
    }

private:
    const scene_rdl2::rdl2::Geometry* mGeometry;
    const scene_rdl2::rdl2::Layer* mLayer;
};

GeometryManager::GeometryManager(scene_rdl2::rdl2::SceneContext* sceneContext,
        const GeometryManagerOptions& options):
    mSceneContext(sceneContext),
    mEmbreeAccelerator(nullptr),
    mGPUAccelerator(nullptr),
    mOptions(options),
    mChangeStatus(ChangeFlag::ALL),
    mVolumeAssignmentTable(new geom::internal::VolumeAssignmentTable())
{
    mEmbreeAccelerator.reset(new EmbreeAccelerator(mOptions.accelOptions));

    mDeformedGeometrySets.clear();

    // We *don't* create the GPUAccelerator here because we need a completely
    // initialized scene.  Instead, it is created at the end of RenderContext::renderPrep()
    // which calls GeometryManager::updateGPUAccelerator().
}

static size_t
fillGenerateList(
        const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap& fullGeometryRootShaders,
        scene_rdl2::rdl2::Geometry* topLevelGeometry,
        scene_rdl2::rdl2::Geometry* currentGeometry,
        std::unordered_set<const scene_rdl2::rdl2::Geometry*>& isReferenced,
        std::map<size_t, std::unordered_set<scene_rdl2::rdl2::Geometry*>>& toGenerate,
        scene_rdl2::rdl2::Layer::GeometryToRootShadersMap& toLoadGeometryRootShaders,
        int level)
{
    if (level > 0) {
        isReferenced.insert(currentGeometry);
    }

    // Get the geometry the current geometry references, going down one more level
    const scene_rdl2::rdl2::SceneObjectVector& references =
        currentGeometry->get(scene_rdl2::rdl2::Geometry::sReferenceGeometries);

    size_t generateOrder = 0;
    for (const auto& ref : references) {
        if (!ref->isA<scene_rdl2::rdl2::Geometry>()) {
            continue;
        }
        scene_rdl2::rdl2::Geometry* referencedGeometry = ref->asA<scene_rdl2::rdl2::Geometry>();
        generateOrder = max(generateOrder,
            fillGenerateList(fullGeometryRootShaders, topLevelGeometry, referencedGeometry,
            isReferenced, toGenerate, toLoadGeometryRootShaders, level + 1) + 1);
        // the top level geometry references other geometries and needs to be aware of
        // the shading network attribute requests from referenced geometries
        const auto& refRootShaders = fullGeometryRootShaders.find(referencedGeometry);
        if (refRootShaders == fullGeometryRootShaders.end()) {
            continue;
        }
        for (scene_rdl2::rdl2::RootShader* s : refRootShaders->second) {
            // collect up all the shaders we find in the referenced geometries and
            // and associate them with the top level geometry
            toLoadGeometryRootShaders[topLevelGeometry].insert(s);
        }
    }

    toGenerate[generateOrder].insert(currentGeometry);

    return generateOrder;
}

GeometryManager::GM_RESULT
GeometryManager::loadGeometries(scene_rdl2::rdl2::Layer* layer,
                                const ChangeFlag flag,
                                const Mat4d& world2render,
                                const int currentFrame,
                                const geom::MotionBlurParams& motionBlurParams,
                                const unsigned threadCount,
                                const shading::PerGeometryAttributeKeySet &perGeometryAttributes)
{
    // Get all geometries in the Layer.
    scene_rdl2::rdl2::Layer::GeometryToRootShadersMap fullGeometryRootShaders;
    layer->getAllGeometryToRootShaders(fullGeometryRootShaders);
    scene_rdl2::rdl2::Layer::GeometryToRootShadersMap toLoadGeometryRootShaders;
    if (flag == ChangeFlag::ALL) {
        toLoadGeometryRootShaders = fullGeometryRootShaders;
    } else if (flag == ChangeFlag::UPDATE) {
        // Get geometries with changed shaders which requires generate.
        layer->getChangedGeometryToRootShaders(toLoadGeometryRootShaders);
    }

    if (!toLoadGeometryRootShaders.empty()) {
        // fill out the list of geometry procedural to generate.
        // the geometry that is referenced by other geometry will be generated
        // first (this is somewhat a dependency graph ordered evaluation)
        std::unordered_set<const scene_rdl2::rdl2::Geometry*> isReferenced;
        std::map<size_t, std::unordered_set<scene_rdl2::rdl2::Geometry*>> toGenerate;
        for (const auto& pair : toLoadGeometryRootShaders) {
            scene_rdl2::rdl2::Geometry* geometry = pair.first;
            // For the first iteration of this recursive function call, we just say
            // that the top level geometry and the current geometry are the same
            // which picks up the top level geometry and its shaders first before chasing any
            // deeper referenced geometry.
            fillGenerateList(fullGeometryRootShaders, geometry, geometry, isReferenced,
                toGenerate, toLoadGeometryRootShaders, 0);
        }

        {
            size_t totalGeometries = 0;
            for (const auto &pair : toGenerate) {
                const auto &geom = pair.second;
                totalGeometries += geom.size();
            }
            if (mOptions.stats.mGeometryManagerExecTracker.startLoadGeometries(totalGeometries) ==
                GeometryManagerExecTracker::RESULT::CANCELED) {
                return GM_RESULT::CANCELED;
            }
        }

        std::atomic<bool> geoLoadItemCancelCondition(false);

        // now fire up the generate calls in dependency order
        for (const auto& pair : toGenerate) {
            const auto& geometries = pair.second;
            tbb::parallel_for_each(geometries.begin(), geometries.end(),
                [&](scene_rdl2::rdl2::Geometry* geometry) {

                if (mOptions.stats.mGeometryManagerExecTracker.startLoadGeometriesItem() ==
                    GeometryManagerExecTracker::RESULT::CANCELED) {
                    geoLoadItemCancelCondition = true;
                    return;
                }

                mOptions.stats.logString("Generating " +
                    geometry->getSceneClass().getName() +
                    "(\"" + geometry->getName() + "\")");

                // Load the procedural.
                if (!geometry->getProcedural()) {
                    geometry->loadProcedural();
                }
                // Prepend geometry --> world.
                shading::XformSamples geometry2render;

                Mat4f l2r0, l2r1;
                if (geometry->getUseLocalMotionBlur() && motionBlurParams.isMotionBlurOn()) {
                    // If use_local_motion_blur is on then the node_xform is
                    // baked into the points so we don't use it here.
                    l2r0 = toFloat(world2render);
                    l2r1 = toFloat(world2render);
                } else {
                    l2r0 = toFloat(geometry->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                                 scene_rdl2::rdl2::TIMESTEP_BEGIN) * world2render);
                    l2r1 = toFloat(geometry->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                                 scene_rdl2::rdl2::TIMESTEP_END) * world2render);
                }

                if (scene_rdl2::math::isEqual(l2r0, l2r1) ||
                    !motionBlurParams.isMotionBlurOn()) {
                    geometry2render = {xform<Xform3f>(l2r0)};
                } else {
                    geometry2render = {xform<Xform3f>(l2r0), xform<Xform3f>(l2r1)};
                }

                // TODO the getRender2Object is used in shading stage that doesn't
                // take time factor into account. We should update this function
                // interface to accept more than one Xform sample when we need to
                // do time varying shading calculation
                geometry->setRender2Object(geometry2render[0].inverse());

                shading::AttributeKeySet requestedAttributes;
                // Find which primitive attributes this procedural needs to load.
                const auto& rootShaders = toLoadGeometryRootShaders[geometry];
                for (const scene_rdl2::rdl2::RootShader* const s : rootShaders) {
                    const auto& table = s->get<shading::RootShader>().getAttributeTable();
                    const auto& reqKeys = table->getRequiredAttributes();
                    requestedAttributes.insert(reqKeys.begin(), reqKeys.end());
                    const auto& optKeys = table->getOptionalAttributes();
                    requestedAttributes.insert(optKeys.begin(), optKeys.end());
                }
                shading::PerGeometryAttributeKeySet::const_iterator itr =
                    perGeometryAttributes.find(geometry);
                if (itr != perGeometryAttributes.end()) {
                    if (!itr->second.empty()) {
                        requestedAttributes.insert(itr->second.begin(),
                            itr->second.end());
                    }
                }

                // Generate geometry for the procedural.
                GeomGenerateContext generateContext(layer,
                    geometry, std::move(requestedAttributes), currentFrame,
                    threadCount, motionBlurParams);
                // TODO: We can't handle nested procedurals yet.
                if (!geometry->getProcedural()->isLeaf()) {
                    geometry->error("Nested procedurals not supported yet.");
                }
                geom::Procedural* procedural = geometry->getProcedural();
                try {
                    procedural->clear();
                    if (isReferenced.find(geometry) != isReferenced.end()) {
                        // generate in local space and  promote the
                        // generated primitives in procedural to be
                        // a shared primitive group. this geometry won't show
                        shading::XformSamples localXform({geom::Mat43(scene_rdl2::math::one)});
                        procedural->generate(generateContext, localXform);
                        geom::internal::PrimitivePrivateAccess::transformToReference(
                            procedural);
                    } else {
                        procedural->generate(generateContext, geometry2render);
                    }
                } catch (const std::exception &e) {
                    geometry->error(e.what());
                }
                RdlGeometrySetter rdlGeometrySetter(geometry);
                procedural->forEachPrimitive(rdlGeometrySetter);

                if (mOptions.stats.mGeometryManagerExecTracker.endLoadGeometriesItem() ==
                    GeometryManagerExecTracker::RESULT::CANCELED) {
                    geoLoadItemCancelCondition = true;
                    return;
                }
            });
        }

        if (geoLoadItemCancelCondition) {
            mOptions.stats.mGeometryManagerExecTracker.finalizeLoadGeometriesItem(true); // cancel = true
            return GM_RESULT::CANCELED;
        } else {
            mOptions.stats.mGeometryManagerExecTracker.finalizeLoadGeometriesItem(false); // cancel = false
        }

        for (const auto& gs : toLoadGeometryRootShaders) {
            scene_rdl2::rdl2::Geometry * const geometry = gs.first;
            mOptions.stats.logGeneratingProcedurals(
                geometry->getSceneClass().getName(), geometry->getName());
        }
    } else {
        if (mOptions.stats.mGeometryManagerExecTracker.startLoadGeometries(0) ==
            GeometryManagerExecTracker::RESULT::CANCELED) {
            return GM_RESULT::CANCELED;
        }
    }

    if (mOptions.stats.mGeometryManagerExecTracker.endLoadGeometries() ==
        GeometryManagerExecTracker::RESULT::CANCELED) {
        return GM_RESULT::CANCELED;
    }

    return GM_RESULT::FINISHED;
}

void
GeometryManager::updateGeometryData(scene_rdl2::rdl2::Layer* layer,
                                    scene_rdl2::rdl2::Geometry* geometry,
                                    const std::vector<std::string>& meshNames,
                                    const VertexBufferArray& vertexBuffers,
                                    const Mat4d& world2render,
                                    const int currentFrame,
                                    const geom::MotionBlurParams& motionBlurParams,
                                    const unsigned threadCount)
{
    geom::Procedural* procedural = geometry->getProcedural();
    if (procedural == nullptr) {
        std::string msg =
           std::string("Incorrectly update uninitialized geometry ") +
           geometry->getName();
        MNRY_ASSERT(false, msg.c_str());
        return;
    }

    // Prepend geometry --> world.
    shading::XformSamples geometry2render;
    Mat4f l2r0 = toFloat(geometry->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                       scene_rdl2::rdl2::TIMESTEP_BEGIN) * world2render);
    Mat4f l2r1 = toFloat(geometry->get(scene_rdl2::rdl2::Node::sNodeXformKey,
                                       scene_rdl2::rdl2::TIMESTEP_END) * world2render);
    if (scene_rdl2::math::isEqual(l2r0, l2r1) ||
        !motionBlurParams.isMotionBlurOn()) {
        geometry2render = {xform<Xform3f>(l2r0)};
    } else {
        geometry2render = {xform<Xform3f>(l2r0), xform<Xform3f>(l2r1)};
    }
    // TODO the getRender2Object is used in shading stage that doesn't
    // take time factor into account. We should update this function
    // interface to accept more than one Xform sample when we need to
    // do time varying shading calculation
    geometry->setRender2Object(geometry2render[0].inverse());

    GeomUpdateContext updateContext(layer,
                                    geometry,
                                    currentFrame,
                                    threadCount,
                                    motionBlurParams);

    updateContext.setMeshNames(meshNames);
    updateContext.setMeshVertexDatas(vertexBuffers);

    // Hand the update off to the geom library to process.
    // Note: this also re-tessellates the primitives.
    procedural->update(updateContext, geometry2render);

    // Make sure the accelerator gets updated accordingly for the
    // GeometrySets that contain this geometry
    std::for_each(mSceneContext->beginGeometrySet(),
                  mSceneContext->endGeometrySet(),
                  [&](scene_rdl2::rdl2::GeometrySet* geometrySet) {
        if (geometrySet->contains(geometry)) {
            mDeformedGeometrySets.insert(geometrySet);
        }
    });
}

// collect all the SharedPrimitives (referenced by instances) and
// leaf Primitives that need to be tessellated from input geometrySets
static void
collectPrimitives(const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
                  const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap& g2s,
                  geom::SharedPrimitiveSet& sharedPrimitives,
                  geom::InternalPrimitiveList& primitivesToTessellate,
                  geom::InternalPrimitiveList& curvePrimitives)
{
    // there is chance that multiple geometrySets contain shared geometries,
    // we need to make sure all the geometries to loop through tessellation
    // are not duplicated to avoid race condition
    std::unordered_set<geom::Procedural*> uniqueProceduralSet;
    PrimitiveCollector primitiveCollector(sharedPrimitives, primitivesToTessellate, curvePrimitives);
    for (auto& geometrySet : geometrySets) {
        // Some GeometrySets (e.g. ShadowReceiverSets) are not intended to be part of the scene, but are just
        // there to represent collections of things and should not be added to the BVH.
        if (!geometrySet->includeInBVH()) {
            continue;
        }
        const scene_rdl2::rdl2::SceneObjectIndexable& geometries = geometrySet->getGeometries();
        for (auto& sceneObject : geometries) {
            scene_rdl2::rdl2::Geometry* geom = sceneObject->asA<scene_rdl2::rdl2::Geometry>();
            if (g2s.find(geom) == g2s.end()) {
                continue;
            }
            geom::Procedural * const procedural = geom->getProcedural();
            if (!procedural) {
                // All parts in a procedural are unassigned in the layer
                continue;
            }
            // TODO: We can't handle nested procedurals yet.
            MNRY_ASSERT_REQUIRE(procedural->isLeaf(),
                "Nested procedurals not supported yet.");
            if (uniqueProceduralSet.insert(procedural).second) {
                geom::ProceduralLeaf* const leaf =
                    static_cast<geom::ProceduralLeaf*>(procedural);
                leaf->forEachPrimitive(primitiveCollector);
            } else {
                sceneObject->warn(
                    "This geometry belongs to more than one geometry set. "
                    "This will make the geometry appear in multiple sub BVHs, "
                    "which can be inefficient and cause rendering artifacts.");
            }
        }
    }
}

void
bakeVolumeShaderDensityMap(const scene_rdl2::rdl2::Layer* layer,
                           geom::internal::NamedPrimitive* prim,
                           const geom::MotionBlurParams& motionBlurParams,
                           const geom::internal::VolumeAssignmentTable* volumeAssignmentTable)
{
    // Bake volume shader. This creates a vdb grid.
    if (prim->hasVolumeAssignment(layer) && (
        prim->getType() == geom::internal::Primitive::POLYMESH ||
        prim->getType() == geom::internal::Primitive::VDB_VOLUME)) {

        std::unordered_set<int> assignmentIds;
        prim->getUniqueAssignmentIds(assignmentIds);
        const scene_rdl2::rdl2::VolumeShader* volumeShader = nullptr;
        auto it = assignmentIds.begin();
        int assignmentId;
        // get first volume shader assignment
        while (volumeShader == nullptr && it != assignmentIds.end()) {
            assignmentId = *it++;
            volumeShader = layer->lookupVolumeShader(assignmentId);
        }

        scene_rdl2::math::Mat4f primToRender(scene_rdl2::math::one);
        switch (prim->getType()) {
        case geom::internal::Primitive::POLYMESH:
            primToRender = static_cast<geom::internal::Mesh *>(prim)->getTransform();
            break;
        case geom::internal::Primitive::VDB_VOLUME:
            primToRender = static_cast<geom::internal::VdbVolume *>(prim)->getTransform();
            break;
        }

        prim->bakeVolumeShaderDensityMap(volumeShader, primToRender, motionBlurParams,
                                         volumeAssignmentTable, assignmentId);
    }
}

void
GeometryManager::bakeGeometry(scene_rdl2::rdl2::Layer* layer,
        const geom::MotionBlurParams& motionBlurParams,
        const std::vector<mcrt_common::Frustum>& frustums,
        const scene_rdl2::math::Mat4d& world2render,
        std::vector<std::unique_ptr<geom::BakedMesh>>& bakedMeshes,
        std::vector<std::unique_ptr<geom::BakedCurves>>& bakedCurves,
        const scene_rdl2::rdl2::Camera* globalDicingCamera)
{
    // update changes for BVH
    scene_rdl2::rdl2::Layer::GeometryToRootShadersMap g2s;

    scene_rdl2::rdl2::SceneContext::GeometrySetVector geometrySets;
    geometrySets = mSceneContext->getAllGeometrySets();
    layer->getAllGeometryToRootShaders(g2s);

    geom::SharedPrimitiveSet sharedPrimitives;
    geom::InternalPrimitiveList primitivesToTessellate;
    geom::InternalPrimitiveList curvePrimitives;
    // if a primitive is not generated, we don't want to
    // tessellate it. Pass in g2s to verify whether a geometry
    // in geometrySets has been generated
    collectPrimitives(geometrySets, g2s,
                      sharedPrimitives,
                      primitivesToTessellate,
                      curvePrimitives);
    // trasform shared primitives to their own local space
    TransformConcatenator transformConcatenator(motionBlurParams);
    ReferenceSetter referenceSetter(0.f, true);
    for (auto& sharedPrimitive : sharedPrimitives) {
        sharedPrimitive->getPrimitive()->accept(transformConcatenator);
        // Set adaptive error to 0 on objects that are instanced to disable
        // adaptive tessellation
        sharedPrimitive->getPrimitive()->accept(referenceSetter);
    }

    // tessellate the primitives
    const bool fastGeomUpdate = mSceneContext->getSceneVariables().get(
        scene_rdl2::rdl2::SceneVariables::sFastGeomUpdate);

    const bool enableDisplacement = mSceneContext->getSceneVariables().get(
        scene_rdl2::rdl2::SceneVariables::sEnableDisplacement);

    tbb::blocked_range<size_t> range(0, primitivesToTessellate.size());
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            geom::internal::NamedPrimitive* prim =
                static_cast<geom::internal::NamedPrimitive*>(primitivesToTessellate[i]);
            try {
                std::vector<mcrt_common::Frustum> dicingFrustums;
                scene_rdl2::math::Mat4d dicingWorld2Render;
                bool dicingCamExists = getDicingCameraFrustums(&dicingFrustums, 
                                                               &dicingWorld2Render,
                                                               globalDicingCamera,
                                                               prim,
                                                               world2render);

                const geom::internal::TessellationParams params(layer,
                                                                dicingCamExists ? dicingFrustums : frustums,
                                                                dicingCamExists ? dicingWorld2Render : world2render,
                                                                enableDisplacement,
                                                                fastGeomUpdate,
                                                                /* isBaking = */ true,
                                                                mVolumeAssignmentTable.get());
                prim->tessellate(params);

                // Bake the density map of a volume shader bound to this primitive. This is more
                // optimal than directly sampling the density map during mcrt. This creates a vdb grid.
                bakeVolumeShaderDensityMap(layer, prim, motionBlurParams, mVolumeAssignmentTable.get());
            } catch (const std::exception &e) {
                prim->getRdlGeometry()->error(e.what());
            }
        }
    });

    const float fps = mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sFpsKey);
    const float currentFrame = mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sFrameKey);
    std::vector<float> motionSteps = motionBlurParams.getMotionSteps();

    // retrieve the tessellated geometry
    for (geom::internal::Primitive* prim : primitivesToTessellate) {
        geom::internal::NamedPrimitive* namedPrim =
            static_cast<geom::internal::NamedPrimitive*>(prim);

        geom::internal::Mesh* mesh = dynamic_cast<geom::internal::Mesh*>(namedPrim);
        if (mesh) {
            std::unique_ptr<geom::BakedMesh> bakedMesh = fauxstd::make_unique<geom::BakedMesh>();
            bakedMesh->mRdlGeometry = mesh->getRdlGeometry();
            for (size_t m = 0; m < motionSteps.size(); ++m) {
                bakedMesh->mMotionFrames.push_back((currentFrame + motionSteps[m]) / fps);
            }
            mesh->getBakedMesh(*bakedMesh);
            bakedMeshes.push_back(std::move(bakedMesh));
        }

        // handle baking other primitive types here
    }

    // retrieve the curves geometry
    for (geom::internal::Primitive* prim : curvePrimitives) {
        geom::internal::NamedPrimitive* namedPrim =
            static_cast<geom::internal::NamedPrimitive*>(prim);

        geom::internal::Curves* curves = dynamic_cast<geom::internal::Curves*>(namedPrim);
        if (curves) {
            std::unique_ptr<geom::BakedCurves> bakedCurve = fauxstd::make_unique<geom::BakedCurves>();
            bakedCurve->mRdlGeometry = curves->getRdlGeometry();
            bakedCurve->mLayer = layer;
            for (size_t m = 0; m < motionSteps.size(); ++m) {
                bakedCurve->mMotionFrames.push_back((currentFrame + motionSteps[m]) / fps);
            }
            curves->getBakedCurves(*bakedCurve);
            bakedCurves.push_back(std::move(bakedCurve));
        }
    }
}

GeometryManager::GM_RESULT
GeometryManager::finalizeChanges(scene_rdl2::rdl2::Layer* layer,
        const geom::MotionBlurParams& motionBlurParams,
        const std::vector<mcrt_common::Frustum>& frustums,
        const scene_rdl2::math::Mat4d& world2render,
        OptimizationTarget accelMode, 
        const scene_rdl2::rdl2::Camera* dicingCamera,
        bool updateSceneBVH)
{
    if (mOptions.stats.mGeometryManagerExecTracker.startFinalizeChange() ==
        GeometryManagerExecTracker::RESULT::CANCELED) {
         return GM_RESULT::CANCELED;
    }

    if (mChangeStatus == ChangeFlag::NONE) {
        mOptions.stats.logString("No need to build, skipping...");
    } else {
        // update changes for BVH
        scene_rdl2::rdl2::Layer::GeometryToRootShadersMap g2s;

        scene_rdl2::rdl2::SceneContext::GeometrySetVector geometrySets;
        if (mChangeStatus == ChangeFlag::ALL) {
            geometrySets = mSceneContext->getAllGeometrySets();
            layer->getAllGeometryToRootShaders(g2s);
        } else if (mChangeStatus == ChangeFlag::UPDATE){
            mSceneContext->getUpdatedOrDeformedGeometrySets(layer, geometrySets);
            layer->getChangedGeometryToRootShaders(g2s);
        }

        geom::SharedPrimitiveSet sharedPrimitives;
        geom::InternalPrimitiveList primitivesToTessellate;
        geom::InternalPrimitiveList curvePrimitives;
        // if a primitive is not (re)generated, we don't want to
        // (re)tessellate it. Pass in g2s to verify whether a geometry
        // in geometrySets has been (re)generated
        collectPrimitives(geometrySets, g2s,
                          sharedPrimitives,
                          primitivesToTessellate,
                          curvePrimitives);
        // trasform shared primitives to their own local space
        TransformConcatenator transformConcatenator(motionBlurParams);
        ReferenceSetter referenceSetter(0.f, true);
        for (auto& sharedPrimitive : sharedPrimitives) {
            sharedPrimitive->getPrimitive()->accept(transformConcatenator);
            // Set adaptive error to 0 on objects that are instanced to disable
            // adaptive tessellation
            sharedPrimitive->getPrimitive()->accept(referenceSetter);
        }

        // The following only needs to be done once before we update the BVH
        // and before we tessellate geometry that is in the BVH.
        // MeshLight geometries are not in the BVH, but do need to be tessellated.
        // They do not have layer assignments so they do not need their assignments
        // set here.
        if (updateSceneBVH) {
            // set assignments
            for (auto& gsPair : g2s) {
                scene_rdl2::rdl2::Geometry* geometry = gsPair.first;
                geom::Procedural* procedural = geometry->getProcedural();
                if (procedural) {
                    AssignmentAndXformSetter assignmentSetter(layer);
                    procedural->forEachPrimitive(assignmentSetter);
                }
            }

            // Build the runtime volume<->assignment lookup table before tessellation
            mVolumeAssignmentTable->initLookupTable(layer);
            geom::internal::forEachTLS([this](geom::internal::TLState* tls) {
                tls->mVolumeRayState.initializeVolumeLookup(
                        mVolumeAssignmentTable.get());
            });
        }

        // (re)tessellate the primitives
        if (tessellate(layer, primitivesToTessellate, frustums, world2render, motionBlurParams, dicingCamera) ==
            GM_RESULT::CANCELED) {
            return GM_RESULT::CANCELED;
        }

        // Verify whether all the required primitive attributes from
        // shading network are provided
        for (auto& gsPair : g2s) {
            scene_rdl2::rdl2::Geometry* geometry = gsPair.first;
            geom::Procedural* procedural = geometry->getProcedural();
            if (procedural) {
                PrimitiveAttributeChecker primitiveAttributeChecker(
                    gsPair.first, layer);
                procedural->forEachPrimitive(primitiveAttributeChecker);
            }
        }

        if (updateSceneBVH) {
            if (mOptions.stats.mGeometryManagerExecTracker.startBVHConstruction() ==
                GeometryManagerExecTracker::RESULT::CANCELED) {
                return GM_RESULT::CANCELED;
            }
            updateAccelerator(layer, geometrySets, g2s, accelMode);
            if (mOptions.stats.mGeometryManagerExecTracker.endBVHConstruction() ==
                GeometryManagerExecTracker::RESULT::CANCELED) {
                return GM_RESULT::CANCELED;
            }
        }
    }

    if (mOptions.stats.mGeometryManagerExecTracker.endFinalizeChange() ==
        GeometryManagerExecTracker::RESULT::CANCELED) {
        return GM_RESULT::CANCELED;
    }

    return GM_RESULT::FINISHED;
}

bool 
GeometryManager::getDicingCameraFrustums(std::vector<mcrt_common::Frustum>* frustums,
                                         scene_rdl2::math::Mat4d* dicingWorld2Render,
                                         const scene_rdl2::rdl2::Camera* globalDicingCamera,
                                         const geom::internal::NamedPrimitive* prim,
                                         const scene_rdl2::math::Mat4d& mainWorld2Render) 
{
    // get global dicing camera if there is one
    const scene_rdl2::rdl2::Camera* dicingRdlCamera = globalDicingCamera;
    if (!dicingRdlCamera) {
        // get dicing camera on the geometry if there is one
        dicingRdlCamera = prim->getRdlGeometry()->getDicingCamera();
    }
    const std::string& className = dicingRdlCamera ? dicingRdlCamera->getSceneClass().getName() : "";

    // dicing camera must be of type PerspectiveCamera
    if (className == "PerspectiveCamera") {
        moonray::pbr::PerspectiveCamera dicingCamera(dicingRdlCamera);
        scene_rdl2::math::Mat4d dicingRender2World = dicingRdlCamera->get(scene_rdl2::rdl2::Node::sNodeXformKey);
        *dicingWorld2Render = moonray::pbr::rtInverse(dicingRender2World);

        // The dicing camera's frustums by default assume it's looking down the negative z-axis in main camera space
        // The transform we need moves the frustums into main camera space, then to the dicing camera's position
        scene_rdl2::math::Mat4d frustumTransform(dicingRender2World * mainWorld2Render);
        dicingCamera.update(*dicingWorld2Render); 

        // compute dicing camera frustums, then transform them to be in the main camera's viewing space
        frustums->push_back(mcrt_common::Frustum());
        dicingCamera.computeFrustum(&frustums->back(), 0, true);  // frustum at shutter open
        frustums->back().transformClipPlanes(frustumTransform);

        frustums->push_back(mcrt_common::Frustum());
        dicingCamera.computeFrustum(&frustums->back(), 1, true);  // frustum at shutter close
        frustums->back().transformClipPlanes(frustumTransform);
        return true;
    } else if (className != "") {
        scene_rdl2::logging::Logger::warn("The only type of dicing camera that is supported is the PerspectiveCamera. "
                                          " The primary camera will be used for tessellation.");
    }
    return false;
}

void
GeometryManager::getEmissiveRegions(const scene_rdl2::rdl2::Layer* layer,
        std::vector<geom::internal::EmissiveRegion>& emissiveRegions) const
{
    // maps assignment id to EmissionDistributions for emissive reference volumes
    using SharedEmissionDistributions =
        std::unordered_map<int, std::shared_ptr<geom::internal::EmissionDistribution>>;

    class SharedEmissionDistributionCollector : public geom::PrimitiveVisitor
    {
    public:
        SharedEmissionDistributionCollector(
                SharedEmissionDistributions &emissionDistributions,
                int assignmentId,
                int visibilityMask,
                const scene_rdl2::rdl2::VolumeShader *volumeShader):
            mEmissionDistributions(emissionDistributions),
            mAssignmentId(assignmentId),
            mVisibilityMask(visibilityMask),
            mVolumeShader(volumeShader)
        {
        }

        void visitPrimitive(geom::Primitive &p) override
        {
            geom::internal::Primitive* pImpl =
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
            constexpr int secondaryVisibilities =
                scene_rdl2::rdl2::DIFFUSE_REFLECTION   |
                scene_rdl2::rdl2::DIFFUSE_TRANSMISSION |
                scene_rdl2::rdl2::GLOSSY_REFLECTION    |
                scene_rdl2::rdl2::GLOSSY_TRANSMISSION  |
                scene_rdl2::rdl2::MIRROR_REFLECTION    |
                scene_rdl2::rdl2::MIRROR_TRANSMISSION  |
                scene_rdl2::rdl2::PHASE_REFLECTION     |
                scene_rdl2::rdl2::PHASE_TRANSMISSION;
            bool canIlluminate =
                (secondaryVisibilities & mVisibilityMask) != 0;
            if (pImpl->getType() == geom::internal::Primitive::VDB_VOLUME) {
                auto pVolume = static_cast<geom::internal::VdbVolume*>(pImpl);
                canIlluminate &= pVolume->hasEmissionField();
            }
            if (canIlluminate) {
                mEmissionDistributions[mAssignmentId] =
                    std::shared_ptr<geom::internal::EmissionDistribution>
                    (pImpl->computeEmissionDistribution(mVolumeShader));
                // the optimization for empty emission grids is
                // handled when we create the emissive regions
            }
        }

        void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override
        {
            pg.forEachPrimitive(*this, /* doParallel = */ false);
        }

        void visitTransformedPrimitive(geom::TransformedPrimitive& t) override
        {
            t.getPrimitive()->accept(*this);
        }

    private:
        SharedEmissionDistributions &mEmissionDistributions;
        int mAssignmentId;
        int mVisibilityMask; // of this geometry
        const scene_rdl2::rdl2::VolumeShader *mVolumeShader;
    };

    // collect the shared emission distributions
    SharedEmissionDistributions sharedEmissionDistributions;
    for (int assignmentId = 0; assignmentId < layer->getAssignmentCount(); ++assignmentId) {
        const scene_rdl2::rdl2::VolumeShader *volumeShader = layer->lookupVolumeShader(assignmentId);
        if (volumeShader && volumeShader->getProperties() & scene_rdl2::rdl2::VolumeShader::IS_EMISSIVE) {
            const scene_rdl2::rdl2::Geometry *geom = layer->lookupGeomAndPart(assignmentId).first;
            geom::Procedural *proc = const_cast<geom::Procedural *>(geom->getProcedural());
            if (proc->isLeaf()) {
                const geom::ProceduralLeaf *pLeaf = static_cast<geom::ProceduralLeaf *>(proc);
                if (pLeaf->isReference()) {
                    const geom::SharedPrimitive *sharedPrimitive = pLeaf->getReference().get();
                    SharedEmissionDistributionCollector collector(sharedEmissionDistributions,
                        assignmentId, geom->getVisibilityMask(), volumeShader);
                    sharedPrimitive->getPrimitive()->accept(collector);
                }
            }
        }
    }

    class EmissiveRegionsCollector : public geom::PrimitiveVisitor
    {
    public:
        explicit EmissiveRegionsCollector(
                std::vector<geom::internal::EmissiveRegion>& emissiveRegions,
                int assignmentId, int volumeId,
                const scene_rdl2::rdl2::VolumeShader* volumeShader,
                int visibilityMask):
            mEmissiveRegions(emissiveRegions), mAssignmentId(assignmentId),
            mVolumeId(volumeId), mVolumeShader(volumeShader),
            mVisibilityMask(visibilityMask)
        {}

        virtual void visitPrimitive(
                geom::Primitive& p) override
        {
            geom::internal::Primitive* pImpl =
                geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);
            if (pImpl->hasAssignment(mAssignmentId)) {
                int secondaryVisibilities =
                    scene_rdl2::rdl2::DIFFUSE_REFLECTION   |
                    scene_rdl2::rdl2::DIFFUSE_TRANSMISSION |
                    scene_rdl2::rdl2::GLOSSY_REFLECTION    |
                    scene_rdl2::rdl2::GLOSSY_TRANSMISSION  |
                    scene_rdl2::rdl2::MIRROR_REFLECTION    |
                    scene_rdl2::rdl2::MIRROR_TRANSMISSION  |
                    scene_rdl2::rdl2::PHASE_REFLECTION     |
                    scene_rdl2::rdl2::PHASE_TRANSMISSION;
                bool canIlluminate =
                    (secondaryVisibilities & mVisibilityMask) != 0;
                if (pImpl->getType() == geom::internal::Primitive::VDB_VOLUME) {
                    auto pVolume = static_cast<geom::internal::VdbVolume*>(pImpl);
                    canIlluminate &= pVolume->hasEmissionField();
                }

                if (canIlluminate) {
                    geom::internal::EmissiveRegion emissiveRegion(pImpl,
                        mVolumeShader, mVolumeId, mVisibilityMask);
                    if (emissiveRegion.size() > 0) {
                        mEmissiveRegions.emplace_back(std::move(emissiveRegion));
                    }
                }
            }
        }

        virtual void visitPrimitiveGroup(
                geom::PrimitiveGroup& pg) override
        {
            pg.forEachPrimitive(*this, false);
        }

        virtual void visitTransformedPrimitive(
                geom::TransformedPrimitive& t) override
        {
            t.getPrimitive()->accept(*this);
        }

        virtual void visitInstance(geom::Instance& i) override
        {
            // Keep this here so we skip over non-emissive volume instances
        }

    private:
        std::vector<geom::internal::EmissiveRegion>& mEmissiveRegions;
        int mAssignmentId;
        int mVolumeId;
        const scene_rdl2::rdl2::VolumeShader* mVolumeShader;
        int mVisibilityMask;
    };

    emissiveRegions.clear();
    for (size_t i = 0; i < mVolumeAssignmentTable->getVolumeCount(); ++i) {
        int volumeId = i;
        int assignmentId = mVolumeAssignmentTable->getAssignmentId(volumeId);

        // do we have a shared emission distribution for this assignment id?
        // if so, then this volume id is an instance
        SharedEmissionDistributions::const_iterator itr = sharedEmissionDistributions.find(assignmentId);
        if (itr != sharedEmissionDistributions.end()) {
            std::shared_ptr<geom::internal::EmissionDistribution> emissionDistribution = itr->second;
            // "count > 0" handles the edge case where the user submits an emission grid where
            // all the values are 0.  In that case, it is not worth creating an emissive region.
            if (emissionDistribution->count() > 0) {
                const int visibilityMask = mVolumeAssignmentTable->getVisibilityMask(volumeId);
                geom::internal::EmissionDistribution::Transform transform = emissionDistribution->getTransform();
                geom::Mat43 primToRender;
                if (mVolumeAssignmentTable->evalInstanceXform(volumeId, /*time = */ 0.f, primToRender)) {
                    const scene_rdl2::math::Mat4f distToRender[2] = {
                        transform.getDistToRender()[0] * scene_rdl2::math::Mat4f(primToRender),
                        transform.getDistToRender()[1] * scene_rdl2::math::Mat4f(primToRender)
                    };
                    // need to divide out any scale in primToRender
                    scene_rdl2::math::XformComponent<scene_rdl2::math::Mat3f> comp;
                    scene_rdl2::math::decompose(primToRender, comp);
                    const Vec3f s(comp.s.row0().x, comp.s.row1().y, comp.s.row2().z);
                    const float invUnitVolume = transform.getInvUnitVolume() / (s.x * s.y * s.z);
                    transform = geom::internal::EmissionDistribution::Transform(distToRender, invUnitVolume);
                }
                emissiveRegions.emplace_back(volumeId, visibilityMask, emissionDistribution, transform);
            }
            continue;
        }

        // otherwise, this volume is not an instance, create a distribution
        const scene_rdl2::rdl2::VolumeShader* volumeShader =
            mVolumeAssignmentTable->lookupWithVolumeId(volumeId);
        if (volumeShader->getProperties() & scene_rdl2::rdl2::VolumeShader::IS_EMISSIVE) {
            // unfortunately right now procedural doesn't offer const
            // forEachPrimitive interface so we need to do a const_cast here
            scene_rdl2::rdl2::Geometry* geom =
                const_cast<scene_rdl2::rdl2::Geometry*>(layer->lookupGeomAndPart(assignmentId).first);
            EmissiveRegionsCollector collector(emissiveRegions,
                assignmentId, volumeId, volumeShader,
                geom->getVisibilityMask());
            geom::Procedural* procedural = geom->getProcedural();
            // the EmissiveRegion container is not thread safe,
            // and the number of emissive regions in the scene probably not
            // worth the effort to do parallel run
            procedural->forEachPrimitive(collector, false);
        }
    }
}

// Counter to provide unique thread ids
tbb::atomic<unsigned> gThreadIdCounter;

GeometryManager::GM_RESULT
GeometryManager::tessellate(scene_rdl2::rdl2::Layer* layer,
        geom::InternalPrimitiveList& primitivesToTessellate,
        const std::vector<mcrt_common::Frustum>& frustums,
        const scene_rdl2::math::Mat4d& world2render,
        const geom::MotionBlurParams& motionBlurParams,
        const scene_rdl2::rdl2::Camera* globalDicingCamera)
{
    mOptions.stats.logString(
        "---------- Tessellating Geometry -------------------------");
    size_t statsSize = mOptions.stats.mPerPrimitiveTessellationTime.size();
    mOptions.stats.mPerPrimitiveTessellationTime.resize(statsSize +
        primitivesToTessellate.size());
    // tessellation timer for all primitives
    util::AverageDouble previousTessellationTime(
        mOptions.stats.mTessellationTime.getCount(),
        mOptions.stats.mTessellationTime.getSum());
    Timer tessellationTimer(mOptions.stats.mTessellationTime);
    tessellationTimer.start();

    const bool fastGeomUpdate = mSceneContext->getSceneVariables().get(
        scene_rdl2::rdl2::SceneVariables::sFastGeomUpdate);

    const bool enableDisplacement = mSceneContext->getSceneVariables().get(
        scene_rdl2::rdl2::SceneVariables::sEnableDisplacement);

    // This thread id is for the debug log. We tessellate the primitives
    // in parallel, so we want to know which primitive is assigned to
    // which thread.
    gThreadIdCounter = 0;
    struct ThreadID {
        // When we create a ThreadID, the counter increments and so
        // each thread gets a unique human readable id.
        ThreadID() : mId(gThreadIdCounter.fetch_and_increment()){}
        unsigned mId;
    };
    typedef tbb::enumerable_thread_specific< ThreadID > EnumerableThreadID;
    EnumerableThreadID enumerableThreadID;

    if (mOptions.stats.mGeometryManagerExecTracker.startTessellation(primitivesToTessellate.size()) ==
        GeometryManagerExecTracker::RESULT::CANCELED) {
        return GM_RESULT::CANCELED;
    }

    std::atomic<bool> tessellationCancelCondition(false);

    tbb::blocked_range<size_t> range(0, primitivesToTessellate.size());
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        // When we create this localThreadID, the global atomic gThreadIdCounter is
        // incremented. This way, each thread gets a unique id.
        EnumerableThreadID::reference localThreadID = enumerableThreadID.local();

        for (size_t i = r.begin(); i < r.end(); ++i) {

            if (mOptions.stats.mGeometryManagerExecTracker.startTessellationItem() ==
                GeometryManagerExecTracker::RESULT::CANCELED) {
                tessellationCancelCondition = true;
                return;
            }

            // tessellation timer for each primitive
            util::AverageDouble primTessTime;
            Timer primTessTimer(primTessTime);
            primTessTime.reset();
            geom::internal::NamedPrimitive* prim =
                static_cast<geom::internal::NamedPrimitive*>(
                primitivesToTessellate[i]);

            std::stringstream startMsg;
            startMsg << "Thread " << localThreadID.mId << "\t: START tessellating "
                    << prim->getRdlGeometry()->getName() << " " << prim->getName();
            mOptions.stats.logDebugString(startMsg.str());
            primTessTimer.start();
            try {
                std::vector<mcrt_common::Frustum> dicingFrustums;
                scene_rdl2::math::Mat4d dicingWorld2Render;
                bool dicingCamExists = getDicingCameraFrustums(&dicingFrustums, 
                                                               &dicingWorld2Render, 
                                                               globalDicingCamera, 
                                                               prim, 
                                                               world2render);

                const geom::internal::TessellationParams tessParams(layer,
                                                                    dicingCamExists ? dicingFrustums : frustums,
                                                                    dicingCamExists ? dicingWorld2Render : world2render,
                                                                    enableDisplacement,
                                                                    fastGeomUpdate,
                                                                    /* isBaking = */ false,
                                                                    mVolumeAssignmentTable.get());
                prim->tessellate(tessParams);

                // Bake the density map of a volume shader bound to this primitive. This is more
                // optimal than directly sampling the density map during mcrt. This creates a vdb grid.
                bakeVolumeShaderDensityMap(layer, prim, motionBlurParams, mVolumeAssignmentTable.get());

            } catch (const std::exception &e) {
                prim->getRdlGeometry()->error(e.what());
            }
            primTessTimer.stop();
            // append tessellation time
            mOptions.stats.mPerPrimitiveTessellationTime[statsSize + i] =
                std::make_pair(prim, primTessTime.getSum());

            std::stringstream finishedMsg;
            finishedMsg << "Thread " << localThreadID.mId << "\t: FINISHED tessellating "
                    << prim->getRdlGeometry()->getName() << " " << prim->getName();
            mOptions.stats.logDebugString(finishedMsg.str());

            if (mOptions.stats.mGeometryManagerExecTracker.endTessellationItem() ==
                GeometryManagerExecTracker::RESULT::CANCELED) {
                tessellationCancelCondition = true;
                return;
            }
        }
    });
    tessellationTimer.stop();
    mOptions.stats.mTessellationTime += previousTessellationTime;

    // return unused memory from malloc() arena to OS so process memory usage
    // stats are accurate
    malloc_trim(0);

    mOptions.stats.logString("Tessellation finished.");

    if (tessellationCancelCondition) {
        mOptions.stats.mGeometryManagerExecTracker.finalizeTessellationItem(true); // cancel = true
        return GM_RESULT::CANCELED;
    } else {
        mOptions.stats.mGeometryManagerExecTracker.finalizeTessellationItem(false); // cancel = false
    }

    if (mOptions.stats.mGeometryManagerExecTracker.endTessellation() ==
        GeometryManagerExecTracker::RESULT::CANCELED) {
        return GM_RESULT::CANCELED;
    }
    
    return GM_RESULT::FINISHED;
}

void GeometryManager::updateAccelerator(const scene_rdl2::rdl2::Layer* layer,
        const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
        const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap& g2s,
        OptimizationTarget accelMode)
{
    mOptions.stats.logString(
        "---------- Building BVH ----------------------------------");
    Timer buildBVHTimer(mOptions.stats.mBuildAcceleratorTime);
    buildBVHTimer.start();
    // geometry updates for real time rendering do not have valid
    // GeometryToRootShadersMaps. So we pass in the nullptr instead.
    if (mSceneContext->getSceneVariables().get(
        scene_rdl2::rdl2::SceneVariables::sFastGeomUpdate)) {
        mEmbreeAccelerator->build(accelMode, mChangeStatus, layer,
            geometrySets, nullptr);
    } else {
        mEmbreeAccelerator->build(accelMode, mChangeStatus, layer,
            geometrySets, &g2s);
    }

    // return unused memory from malloc() arena to OS so process memory usage
    // stats are accurate
    malloc_trim(0);

    mOptions.stats.logString("BVH build finished.");

    buildBVHTimer.stop();

    const BBox3f bounds = mEmbreeAccelerator->getBounds();
    const Vec3f& lower = bounds.lower;
    const Vec3f& upper = bounds.upper;
    mOptions.stats.logString("lower: " +
            std::to_string(lower.x) + " " +
            std::to_string(lower.y) + " " +
            std::to_string(lower.z));
    mOptions.stats.logString("upper: " +
            std::to_string(upper.x) + " " +
            std::to_string(upper.y) + " " +
            std::to_string(upper.z));
}

void GeometryManager::updateGPUAccelerator(bool allowUnsupportedFeatures,
                                           const scene_rdl2::rdl2::Layer* layer)
{
    mGPUAccelerator.reset();

    scene_rdl2::rdl2::SceneContext::GeometrySetVector geometrySets;
    geometrySets = mSceneContext->getAllGeometrySets();

    scene_rdl2::rdl2::Layer::GeometryToRootShadersMap g2s;
    const_cast<scene_rdl2::rdl2::Layer*>(layer)->getAllGeometryToRootShaders(g2s);

    mOptions.stats.logString("---------- Setting up GPU ----------------------------------");

    Timer buildGPUBVHTimer(mOptions.stats.mBuildGPUAcceleratorTime);
    buildGPUBVHTimer.start();

    std::vector<std::string> warningMsgs;
    std::string errorMsg;
    mGPUAccelerator.reset(new GPUAccelerator(allowUnsupportedFeatures, layer, geometrySets, &g2s, warningMsgs, &errorMsg));

    buildGPUBVHTimer.stop();

    if (!errorMsg.empty()) {
        mOptions.stats.logString("GPU setup aborted due to error:");
        mOptions.stats.logString(errorMsg);
        mOptions.stats.logString("Falling back to CPU vector mode");
        mGPUAccelerator.reset();
        return;
    }

    for (auto& msg : warningMsgs) {
        mOptions.stats.logString("GPU warning: " + msg);
    }

    mOptions.stats.logString("GPU setup finished.  Using GPU: " +
        mGPUAccelerator->getGPUDeviceName());
}

static void parsePartNames(char* partsString, ShadowExclusionMapping& mapping,
                           const scene_rdl2::rdl2::Geometry* geometry, const scene_rdl2::rdl2::Layer* layer)
{
    // If string is "*", leave mapping empty (which will cause entire geometry to be mapped)
    if (strcmp(partsString, "*") == 0) return;

    // Otherwise, tokenise parts string into part names and map each part we find
    char* context;
    char* partName = strtok_r(partsString, ",", &context);
    while (partName) {
        // Look up part assignment id from part name
        int partId = layer->getAssignmentId(geometry, std::string(partName));
        if (partId != -1) {
            mapping.mCasterParts.insert(partId);
        }
        partName = strtok_r(NULL, ",", &context);
    }
}

static LabeledGeometrySet* findLabeledGeometrySet(const char* label,
                                                  std::vector<LabeledGeometrySet> &labeledGeometrySets)
{
    for (auto& labeledGeoSet : labeledGeometrySets) {
        if (labeledGeoSet.mLabel == label) {
            return &labeledGeoSet;
        }
    }
    return nullptr;
}

static bool parseSetLabels(char* labelsString, ShadowExclusionMapping& mapping,
                           std::vector<LabeledGeometrySet> &geoSets)
{
    // If string is "*", add all labeled geometry sets to mapping
    if (strcmp(labelsString, "*") == 0) {
        for (auto& geoSet : geoSets) {
            mapping.mLabeledGeoSets.insert(&geoSet);
        }
        return true;
    }

    // Otherwise, tokenise labels string into labels and add the corresponding labeled geometry sets to the mapping
    char* context;
    char* label = strtok_r(labelsString, ",", &context);
    while (label) {
        // Look up labeled geometry set from label
        LabeledGeometrySet* geoSet = findLabeledGeometrySet(label, geoSets);
        if (geoSet != nullptr) {
            mapping.mLabeledGeoSets.insert(geoSet);
        }
        label = strtok_r(NULL, ",", &context);
    }
    return (mapping.mLabeledGeoSets.empty() == false);
}

void GeometryManager::updateShadowLinkings(const scene_rdl2::rdl2::Layer* layer)
{
    // Get the geometries from this layer
    scene_rdl2::rdl2::Layer::GeometrySet geometries;
    layer->getAllGeometries(geometries);

    // Generate labeled geometry sets by looking at the shadow receiver label attribute on the geometry objects
    std::vector<LabeledGeometrySet> geoSets;
    for (const scene_rdl2::rdl2::Geometry* geometry : geometries) {
        // Check if geometry has a shadow receiver label
        const std::string label = geometry->getShadowReceiverLabel();
        if (label != "") {
            // It has a label. See if there's already a labeled geometry set with this label
            LabeledGeometrySet* pSet = findLabeledGeometrySet(label.c_str(), geoSets);
            if (pSet != nullptr) {
                // Found the label; add the geom to the corresponding set
                pSet->mGeometries.push_back(geometry);
            } else {
                // No existing labeled geometry set has this label. Create one
                geoSets.push_back( {{geometry}, label} );
            }
        }
    }

    // Now map shadow casters to receiver sets
    for (const scene_rdl2::rdl2::Geometry* geometry : geometries) {

        // Construct the mappings described by the shadow_exclusion_mappings geometry string attribute.
        std::vector<ShadowExclusionMapping> mappings;
        std::string mappingsString = const_cast<std::string&>(geometry->getShadowExclusionMappings());

        // Tokenise mappings string into entries each of the form partsString:labelsString where:
        //   partsString  is a comma-separated list of part names, or "*" to map the entire geometry
        //   labelsString is a comma-separated list of receiver geometry set labels, or "*" for all set labels
        char* context;
        char* entryString = strtok_r(const_cast<char*>(mappingsString.c_str()), " \t\n", &context);
        while (entryString) {
            // Split the entry string into part names and receiver set labels
            int pos = std::string(entryString).find(':');
            if (pos != std::string::npos) {
                entryString[pos] = '\0';
                char* partsString = entryString;
                char* labelsString = &entryString[pos+1];

                // Map listed parts to listed receiver sets
                ShadowExclusionMapping mapping;
                parsePartNames(partsString, mapping, geometry, layer);
                if (parseSetLabels(labelsString, mapping, geoSets)) {
                    mappings.push_back(mapping);
                }
            } else {
                geometry->warn(
                    "Missing colon in shadow_exclusion_mappings entry.");
            } 
            entryString = strtok_r(NULL, " \t\n", &context);
        }

        // Set up the shadow linking info
        const geom::Procedural* procedural = geometry->getProcedural();
        if (procedural) {
            ShadowLinkingSetter shadowLinkingSetter(layer, mappings);
            const_cast<geom::Procedural *>(procedural)->forEachPrimitive(shadowLinkingSetter);
        }
    }
}

} // namespace rt
} // namespace moonray

