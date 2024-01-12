// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "OpenVdbMap_ispc_stubs.h"

#include <moonray/rendering/shading/OpenVdbSampler.h>
#include <moonray/rendering/shading/OpenVdbUtil.h>
#include <moonray/rendering/shading/MapApi.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include <memory>
#include <sstream>

//---------------------------------------------------------------------------
RDL2_DSO_CLASS_BEGIN(OpenVdbMap, Map)

public:
    OpenVdbMap(const SceneClass& sceneClass, const std::string& name);
    ~OpenVdbMap() override;
    void update() override;

private:
    enum vdbSource : int {
        VDB_SOURCE_TEXTURE          = 0,
        VDB_SOURCE_OPENVDB_GEOMETRY = 1,
    };

    static void sampleVdb(const Map* map,
                          moonray::shading::TLState *tls,
                          const float px, const float py, const float pz,
                          scene_rdl2::math::Color *out);

    static void sample(const Map* map, moonray::shading::TLState *tls,
                       const moonray::shading::State& state, scene_rdl2::math::Color* sample);

    ispc::OpenVdbMap mIspc;
    std::unique_ptr<moonray::shading::Xform> mXform;
    std::unique_ptr<moonray::shading::OpenVdbSampler> mSampler;

RDL2_DSO_CLASS_END(OpenVdbMap)
//---------------------------------------------------------------------------

OpenVdbMap::OpenVdbMap(const SceneClass& sceneClass, const std::string& name) :
    Parent(sceneClass, name)
{
    mSampleFunc = OpenVdbMap::sample;
    mSampleFuncv = (SampleFuncv) ispc::OpenVdbMap_getSampleFunc();

    mIspc.mRefPKey = moonray::shading::StandardAttributes::sRefP;
    mIspc.mSampleFn = (intptr_t) sampleVdb;
}

OpenVdbMap::~OpenVdbMap()
{
}

void
OpenVdbMap::update()
{
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
            const Geometry* geom = get(attrOpenVdbGeometry) ?
                                         get(attrOpenVdbGeometry)->asA<Geometry>() :
                                         nullptr;
            if (!geom) {
                error("You must specify an OpenVdbGeometry object");
            } else if (!moonray::shading::isOpenVdbGeometry(geom)) {
                error("The object specified for 'open_vdb_geometry' is not an OpenVdbGeometry object");
            } else {
                AttributeKey<String> modelAttrKey = moonray::shading::getModelAttributeKey(geom);
                vdbFilename = geom->get<std::string>(modelAttrKey);
                grid2world = &geom->get(Node::sNodeXformKey);
            }
        } else { // VDB_SOURCE_TEXTURE
            vdbFilename = get(attrTexture);
            if (vdbFilename.empty()) {
                error("You must specify a valid .vdb filename");
            }
        }

        // make sure we successfully resolved a filename
        if (!vdbFilename.empty()) {

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
            if (mSampler->initialize(vdbFilename,
                                     get(attrGridName),
                                     render2world,
                                     grid2world,
                                     errorMsg)) {
            } else {
                mSampler.reset();
                if (get(attrShowWarnings)) {
                    warn("Unable to load grid from file ", vdbFilename);
                }
            }
        }
    }

    if (hasChanged(attrTextureCoordSource)) {
        mRequiredAttributes.clear();
        mOptionalAttributes.clear();
        if (get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_REFP) {
            mRequiredAttributes.push_back(mIspc.mRefPKey);
        }
    }
}

// called by scalar and vector sample funcs
void
OpenVdbMap::sampleVdb(const Map* map,
                      moonray::shading::TLState *tls,
                      const float px, const float py, const float pz,
                      scene_rdl2::math::Color *out)
{
    const OpenVdbMap* me = static_cast<const OpenVdbMap*>(map);

    if (!me->mSampler) {
        *out = me->get(attrDefaultValue);
        return;
    }

    const Vec3f P(px, py, pz);

    if (me->get(attrShowActiveField)) {
        *out = me->mSampler->getIsActive(tls, P) ?
                    scene_rdl2::math::Color(1.f, 1.f, 1.f) :
                    scene_rdl2::math::Color(0.f, 0.f, 0.f);
    } else {
        *out = me->mSampler->sample(tls, P,
                static_cast<moonray::shading::OpenVdbSampler::Interpolation>(me->get(attrInterpolation)));
    }
}

void
OpenVdbMap::sample(const Map* map, moonray::shading::TLState *tls,
                 const moonray::shading::State& state, scene_rdl2::math::Color* sample)
{
    const OpenVdbMap* me = static_cast<const OpenVdbMap*>(map);

    Vec3f P = state.getP();

    if (me->get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_REFP) {
        state.getRefP(P);
    } else if (me->get(attrTextureCoordSource) == ispc::OPENVDBMAP_COORD_INPUT) {
        P = evalVec3f(me, attrInputTextureCoordinate, tls, state);
    }

    sampleVdb(map, tls, P.x, P.y, P.z, sample);
}


//---------------------------------------------------------------------------

