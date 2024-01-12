// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "OpenVdbMap_v2_ispc_stubs.h"

#include <moonray/rendering/shading/OpenVdbSampler.h>
#include <moonray/rendering/shading/OpenVdbUtil.h>
#include <moonray/rendering/shading/MapApi.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

//---------------------------------------------------------------------------
RDL2_DSO_CLASS_BEGIN(OpenVdbMap_v2, Map)

public:
    OpenVdbMap_v2(const SceneClass& sceneClass, const std::string& name);
    ~OpenVdbMap_v2() override;
    void update() override;

private:
    enum vdbSource : int {
        VDB_SOURCE_TEXTURE          = 0,
        VDB_SOURCE_OPENVDB_GEOMETRY = 1,
    };

    static void sampleVdb(const Map* map,
                          moonray::shading::TLState *tls,
                          const float px, const float py, const float pz,
                          const Geometry* geom,
                          scene_rdl2::math::Color *out);

    static void sample(const Map* map, moonray::shading::TLState *tls,
                       const moonray::shading::State& state, scene_rdl2::math::Color* sample);

    ispc::OpenVdbMap_v2 mIspc;
    std::unique_ptr<moonray::shading::Xform> mXform;
    std::unique_ptr<moonray::shading::OpenVdbSampler> mSampler;
    std::unordered_map<const Geometry*, std::unique_ptr<moonray::shading::OpenVdbSampler> > mSamplers;

RDL2_DSO_CLASS_END(OpenVdbMap_v2)
//---------------------------------------------------------------------------

OpenVdbMap_v2::OpenVdbMap_v2(const SceneClass& sceneClass, const std::string& name) :
    Parent(sceneClass, name)
{
    mSampleFunc = OpenVdbMap_v2::sample;
    mSampleFuncv = (SampleFuncv) ispc::OpenVdbMap_v2_getSampleFunc();

    mIspc.mRefPKey = moonray::shading::StandardAttributes::sRefP;
    mIspc.mSampleFn = (intptr_t) sampleVdb;
}

OpenVdbMap_v2::~OpenVdbMap_v2()
{
}

void
OpenVdbMap_v2::update()
{
    if (hasChanged(attrTextureCoordSource)) {
        mRequiredAttributes.clear();
        mOptionalAttributes.clear();
        if (get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_REFP) {
            mRequiredAttributes.push_back(mIspc.mRefPKey);
        }
    }

    mXform = fauxstd::make_unique<moonray::shading::Xform>(this, nullptr, nullptr, nullptr);
    mIspc.mXform = mXform->getIspcXform();

    if (hasChanged(attrVdbSource) ||
        hasChanged(attrTexture) ||
        hasChanged(attrOpenVdbGeometry)) {

        mSampler.reset();

        // If using OpenVdbGeometry, we'll append its transform, otherwise not.
        const Mat4d* grid2world = nullptr;

        // Resolve the vdb filename and potentially a grid2world transform
        std::string vdbFilename;
        if (get(attrVdbSource) == VDB_SOURCE_OPENVDB_GEOMETRY) {
            for (SceneObject* sceneObject : get(attrOpenVdbGeometry)) {
                if (!sceneObject) {
                    error("You must specify an OpenVdbGeometry object");
                    return;
                }

                Geometry* geom = sceneObject->asA<Geometry>();
                if (!moonray::shading::isOpenVdbGeometry(geom)) {
                    error("The object specified for 'open_vdb_geometry' is not an OpenVdbGeometry object");
                    return;
                }

                AttributeKey<String> modelAttrKey = moonray::shading::getModelAttributeKey(geom);
                vdbFilename = geom->get<std::string>(modelAttrKey);

                // make sure we successfully resolved a filename
                if (vdbFilename.empty()) {
                    if (get(attrShowWarnings)) {
                        warn("No .vdb filename specified for OpenVdbGeometry ", geom->getName());
                    }
                    mSamplers[geom] = nullptr; // add a null entry for this geometry in the map
                    continue; // skip to next OpenVdbGeometry in the list
                }

                // We only need the render2world transform if using P
                const Mat4d* render2world = nullptr;
                if (get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_P) {
                    const SceneContext* ctx = getSceneClass().getSceneContext();
                    render2world = ctx->getRender2World();
                }

                // Construct an uninitialized OpenVdbSampler, then try to initialize it.
                // If initialization fails for any reason, destroy this useless object
                // immediately. Two-phase construct-and-init design pattern
                mSamplers[geom] = fauxstd::make_unique<moonray::shading::OpenVdbSampler>();

                std::string errorMsg;
                grid2world = &geom->get(Node::sNodeXformKey);
                if (!mSamplers[geom]->initialize(vdbFilename,
                                                get(attrGridName),
                                                render2world,
                                                grid2world,
                                                errorMsg)) {
                    mSamplers[geom].reset();
                    if (get(attrShowWarnings)) {
                        warn("Unable to load grid from file ", vdbFilename);
                    }
                }
            }
        } else { // VDB_SOURCE_TEXTURE
            vdbFilename = get(attrTexture);
            if (vdbFilename.empty()) {
                error("You must specify a valid .vdb filename");
                return;
            }

            // We only need the render2world transform if using P
            const Mat4d* render2world = nullptr;
            if (get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_P) {
                const SceneContext* ctx = getSceneClass().getSceneContext();
                render2world = ctx->getRender2World();
            }

            // Construct an uninitialized OpenVdbSampler, then try to initialize it.
            // If initialization fails for any reason, destroy this useless object
            // immediately. Two-phase construct-and-init design pattern
            mSampler = fauxstd::make_unique<moonray::shading::OpenVdbSampler>();

            std::string errorMsg;
            if (!mSampler->initialize(vdbFilename,
                                     get(attrGridName),
                                     render2world,
                                     grid2world,
                                     errorMsg)) {
                mSampler.reset();
                if (get(attrShowWarnings)) {
                    warn("Unable to load grid from file ", vdbFilename);
                }
            }
        }
    }
}

// called by scalar and vector sample funcs
void
OpenVdbMap_v2::sampleVdb(const Map* map,
                         moonray::shading::TLState *tls,
                         const float px, const float py, const float pz,
                         const Geometry* geom,
                         scene_rdl2::math::Color *out)
{
    const OpenVdbMap_v2* me = static_cast<const OpenVdbMap_v2*>(map);

    const Vec3f P(px, py, pz);
    const moonray::shading::OpenVdbSampler::Interpolation interp =
        static_cast<moonray::shading::OpenVdbSampler::Interpolation>(me->get(attrInterpolation));

    if (me->get(attrVdbSource) == VDB_SOURCE_OPENVDB_GEOMETRY) {
        if (me->mSamplers.find(geom) == me->mSamplers.end() || !me->mSamplers.at(geom)) {
            *out = me->get(attrDefaultValue);
            return;
        }

        if (me->get(attrShowActiveField)) {
            *out = me->mSamplers.at(geom)->getIsActive(tls, P) ?
                        scene_rdl2::math::Color(1.f, 1.f, 1.f) :
                        scene_rdl2::math::Color(0.f, 0.f, 0.f);
        } else {
            *out = me->mSamplers.at(geom)->sample(tls, P, interp);
        }

    } else {
        if (!me->mSampler) {
            *out = me->get(attrDefaultValue);
            return;
        }

        if (me->get(attrShowActiveField)) {
            *out = me->mSampler->getIsActive(tls, P) ?
                        scene_rdl2::math::Color(1.f, 1.f, 1.f) :
                        scene_rdl2::math::Color(0.f, 0.f, 0.f);
        } else {
            *out = me->mSampler->sample(tls, P, interp);
        }
    }
}

void
OpenVdbMap_v2::sample(const Map* map,
                      moonray::shading::TLState *tls,
                      const moonray::shading::State& state,
                      scene_rdl2::math::Color* sample)
{
    const OpenVdbMap_v2* me = static_cast<const OpenVdbMap_v2*>(map);

    Vec3f P = state.getP();

    if (me->get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_REFP) {
        state.getRefP(P);
    } else if (me->get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_INPUT) {
        P = evalVec3f(me, attrInputTextureCoordinate, tls, state);
    }

    const Geometry* geom = state.getGeometryObject();

    sampleVdb(map, tls, P.x, P.y, P.z, geom, sample);
}


//---------------------------------------------------------------------------

