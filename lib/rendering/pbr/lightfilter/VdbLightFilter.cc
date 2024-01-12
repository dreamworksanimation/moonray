// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "VdbLightFilter.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/lightfilter/VdbLightFilter_ispc_stubs.h>
#include <moonray/rendering/shading/OpenVdbSampler.h>
#include <moonray/rendering/shading/Util.h>

namespace moonray {
namespace pbr {

using namespace scene_rdl2;
using namespace scene_rdl2::math;

bool VdbLightFilter::sAttributeKeyInitialized;
rdl2::AttributeKey<rdl2::Mat4d> VdbLightFilter::sVdbXformKey;
rdl2::AttributeKey<rdl2::Int> VdbLightFilter::sVdbInterpolation;
rdl2::AttributeKey<rdl2::Float> VdbLightFilter::sDensityRemapInputMinKey;
rdl2::AttributeKey<rdl2::Float> VdbLightFilter::sDensityRemapInputMaxKey;
rdl2::AttributeKey<rdl2::Float> VdbLightFilter::sDensityRemapOutputMinKey;
rdl2::AttributeKey<rdl2::Float> VdbLightFilter::sDensityRemapOutputMaxKey;
rdl2::AttributeKey<rdl2::FloatVector> VdbLightFilter::sDensityRemapOutputsKey;
rdl2::AttributeKey<rdl2::FloatVector> VdbLightFilter::sDensityRemapInputsKey;
rdl2::AttributeKey<rdl2::Bool> VdbLightFilter::sDensityRemapRescaleEnableKey;
rdl2::AttributeKey<rdl2::IntVector> VdbLightFilter::sDensityRemapInterpTypesKey;
rdl2::AttributeKey<rdl2::String> VdbLightFilter::sVdbMapKey;
rdl2::AttributeKey<rdl2::String> VdbLightFilter::sDensityGridNameKey;
rdl2::AttributeKey<rdl2::Rgb> VdbLightFilter::sColorTintNameKey;
rdl2::AttributeKey<rdl2::Float> VdbLightFilter::sBlurValueKey;
rdl2::AttributeKey<rdl2::Int> VdbLightFilter::sBlurType;
rdl2::AttributeKey<rdl2::Bool> VdbLightFilter::sInvertDensityKey;

HUD_VALIDATOR(VdbLightFilter);

VdbLightFilter::VdbLightFilter(const rdl2::LightFilter* rdlLightFilter) 
    : LightFilter(rdlLightFilter)
    , mDensitySampler(nullptr) {
    if (mRdlLightFilter) {
        initAttributeKeys(mRdlLightFilter->getSceneClass());
    }

    ispc::VdbLightFilter_init((ispc::VdbLightFilter *)this->asIspc());
    mSampleFn = (intptr_t) sampleVdb;
}

VdbLightFilter::~VdbLightFilter() { 
    if (mDensitySampler)
        delete mDensitySampler;
}
void
VdbLightFilter::initAttributeKeys(const rdl2::SceneClass &sc) {
    if (sAttributeKeyInitialized) {
        return;
    }

    MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

    sAttributeKeyInitialized = true;

    sVdbXformKey = sc.getAttributeKey<rdl2::Mat4d>("node_xform");
    sVdbMapKey = sc.getAttributeKey<rdl2::String>("vdb_map");
    sDensityGridNameKey = sc.getAttributeKey<rdl2::String>("density_grid_name");
    sVdbInterpolation = sc.getAttributeKey<rdl2::Int>("vdb_interpolation_type");
    sColorTintNameKey = sc.getAttributeKey<rdl2::Rgb>("color_tint");
    sBlurValueKey = sc.getAttributeKey<rdl2::Float>("blur_value");
    sBlurType = sc.getAttributeKey<rdl2::Int>("blur_type");    
    sDensityRemapInputMinKey = sc.getAttributeKey<rdl2::Float>("density_remap_input_min");
    sDensityRemapInputMaxKey = sc.getAttributeKey<rdl2::Float>("density_remap_input_max");
    sDensityRemapOutputMinKey = sc.getAttributeKey<rdl2::Float>("density_remap_output_min");
    sDensityRemapOutputMaxKey = sc.getAttributeKey<rdl2::Float>("density_remap_output_max");
    sDensityRemapRescaleEnableKey = sc.getAttributeKey<rdl2::Bool>("density_rescale_enable");
    sDensityRemapOutputsKey = sc.getAttributeKey<rdl2::FloatVector>("density_remap_outputs");
    sDensityRemapInputsKey = sc.getAttributeKey<rdl2::FloatVector>("density_remap_inputs");
    sDensityRemapInterpTypesKey = sc.getAttributeKey<rdl2::IntVector>("density_remap_interpolation_types");
    sInvertDensityKey = sc.getAttributeKey<rdl2::Bool>("invert_density");

    MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
}

bool
VdbLightFilter::isValidXform(const Mat4d& xf) {
    // Check for zero scale
    return !(isZero(xf.vx[0]) && isZero(xf.vx[1]) && isZero(xf.vx[2]) &&
             isZero(xf.vy[0]) && isZero(xf.vy[1]) && isZero(xf.vy[2]) &&
             isZero(xf.vz[0]) && isZero(xf.vz[1]) && isZero(xf.vz[2]));
}

void
VdbLightFilter::update(const LightFilterMap& /*lightFilters*/,
                       const Mat4d& world2Render) {
    if (!mRdlLightFilter) {
        return;
    }

    Mat4d render2World = world2Render.inverse();
    Mat4d vdb2World0 = mRdlLightFilter->get(sVdbXformKey, 0.f);
    Mat4d vdb2World1 = mRdlLightFilter->get(sVdbXformKey, 1.f);

    if (!isValidXform(vdb2World0) || !isValidXform(vdb2World1)) {
        mRdlLightFilter->error("node_xform is invalid");
    }

    Mat4d world2Vdb0 = vdb2World0.inverse();
    Mat4d world2Vdb1 = vdb2World1.inverse();

    mVdbR2V[0] = Mat4f(render2World * world2Vdb0);
    mVdbR2V[1] = Mat4f(render2World * world2Vdb1);

    Mat4d vdb2Render0 = vdb2World0 * world2Render;
    Mat4d vdb2Render1 = vdb2World1 * world2Render;
    mVdbPos[0] = vdb2Render0.row3().toVec3();
    mVdbPos[1] = vdb2Render1.row3().toVec3();

    // update vdb
    if (mDensitySampler) {
        delete mDensitySampler;
    }
    std::string vdbFilename = mRdlLightFilter->get(sVdbMapKey);
    std::string densityGridName = mRdlLightFilter->get(sDensityGridNameKey);
    if(densityGridName.empty()) {
        densityGridName = "density";
    }

    if (vdbFilename.empty()) {
        mRdlLightFilter->error("You must specify a valid .vdb filename in 'vdb_map'");
    } else {
        // Construct an uninitialized OpenVdbSampler, then try to initialize it.
        // If initialization fails for any reason, destroy this useless object
        // immediately. Two-phase construct-and-init design pattern
        mDensitySampler = new moonray::shading::OpenVdbSampler();

        std::string errorMsg;
        if (mDensitySampler->initialize(vdbFilename,
                                        densityGridName,
                                        nullptr,
                                        nullptr,
                                        errorMsg)) {
        } else {
            delete mDensitySampler;
            mDensitySampler = nullptr;
            //this is an info and not warning as it's not necessary to supply a color grid
            mRdlLightFilter->warn("Unable to load grid ", densityGridName," from file ", vdbFilename, ". Using default Density of 1");
        }      
    }
        
    std::vector<float> inDensityRampVec = mRdlLightFilter->get<rdl2::FloatVector>(sDensityRemapInputsKey);
    std::vector<float> outDensityRampVec = mRdlLightFilter->get<rdl2::FloatVector>(sDensityRemapOutputsKey);
    std::vector<int> interpolationTypesVec = mRdlLightFilter->get<rdl2::IntVector>(sDensityRemapInterpTypesKey);

    if (inDensityRampVec.size() != outDensityRampVec.size() || inDensityRampVec.size() != interpolationTypesVec.size()) {
        mRdlLightFilter->error(
            "Vdb light filter density inputs, remaps and interpolation types are different sizes, using defaults");
        inDensityRampVec = {0.f, 1.f};
        outDensityRampVec = {0.f, 1.f};
        interpolationTypesVec = {ispc::RAMP_INTERPOLATOR_MODE_LINEAR, ispc::RAMP_INTERPOLATOR_MODE_LINEAR};
    }
    
    bool rescaleDensity = mRdlLightFilter->get<rdl2::Bool>(sDensityRemapRescaleEnableKey);
    if(rescaleDensity) {
        float densityRemapInputMin = mRdlLightFilter->get<rdl2::Float>(sDensityRemapInputMinKey);
        float densityRemapInputMax = mRdlLightFilter->get<rdl2::Float>(sDensityRemapInputMaxKey);
        float densityRemapOutputMin = mRdlLightFilter->get<rdl2::Float>(sDensityRemapOutputMinKey);
        float densityRemapOutputMax = mRdlLightFilter->get<rdl2::Float>(sDensityRemapOutputMaxKey);
        
        if (densityRemapInputMin >= densityRemapInputMax) {
            mRdlLightFilter->error(
                "Vdb light filter density_remap_input_range_min is >= density_remap_input_range_max, using defaults");
            densityRemapInputMin = 0.f;
            densityRemapInputMax = 1.f;
        }
        
        if (densityRemapOutputMin >= densityRemapOutputMax) {
            mRdlLightFilter->error(
                "Vdb light filter density_remap_output_range_min is >= density_remap_output_range_max, using defaults");
            densityRemapOutputMin = 0.f;
            densityRemapOutputMax = 1.f;
        }

        //first find min/max of the density remap inputs
        float detectedInputMin = inDensityRampVec[0];
        float detectedInputMax = inDensityRampVec[0];
        for(int i = 1; i < inDensityRampVec.size(); i++) {
            detectedInputMin = min(detectedInputMin, inDensityRampVec[i]);
            detectedInputMax = max(detectedInputMax, inDensityRampVec[i]);
        }
        
        //first find min/max of the density remap outputs
        float detectedRemapMin = outDensityRampVec[0];
        float detectedRemapMax = outDensityRampVec[0];
        for(int i = 1; i < outDensityRampVec.size(); i++) {
            detectedRemapMin = min(detectedRemapMin, outDensityRampVec[i]);
            detectedRemapMax = max(detectedRemapMax, outDensityRampVec[i]);
        }

        if(detectedInputMin < 0.f || detectedInputMin > 1.f || detectedInputMax < 0.f || detectedInputMax > 1.f) {
            mRdlLightFilter->warn(
                "Vdb light filter: Ramp has inputs outside of [0 1] range. the rescale for those inputs will go "
                "outside the range of density_remap_input_range_min and density_remap_input_range_max");
        }

        if(detectedRemapMin < 0.f || detectedRemapMin > 1.f || detectedRemapMax < 0.f || detectedRemapMax > 1.f) {
            mRdlLightFilter->warn(
                "Vdb light filter: Ramp has outputs outside of [0 1] range. the rescale for those inputs will go "
                "outside the range of density_remap_output_range_min and density_remap_output_range_max");
        }

        auto remapHelper = [](std::vector<float>& values, float min, float max) {
            //remap them to user provided min/max, any values outside [0-1] will exceed the min/max range
            for(int i = 0; i < values.size(); i++) {
                values[i] = values[i] * (max - min) + min;
            }
        };
        remapHelper(inDensityRampVec, densityRemapInputMin, densityRemapInputMax);
        remapHelper(outDensityRampVec, densityRemapOutputMin, densityRemapOutputMax);
    }

    mDensityRamp.init(
            inDensityRampVec.size(),
            inDensityRampVec.data(),
            outDensityRampVec.data(),
            reinterpret_cast<const ispc::RampInterpolatorMode*>(interpolationTypesVec.data()));

    mBlurValue = mRdlLightFilter->get<rdl2::Float>(sBlurValueKey);
    mBlurType = mRdlLightFilter->get<rdl2::Int>(sBlurType);

    mInvertDensity = mRdlLightFilter->get<rdl2::Bool>(sInvertDensityKey);
    mVdbInterpolation = mRdlLightFilter->get(sVdbInterpolation);
    mColorTint = mRdlLightFilter->get<rdl2::Rgb>(sColorTintNameKey);
}

bool
VdbLightFilter::canIlluminate(const CanIlluminateData& data) const {
    //assume always true as I couldnt find the method to query bounds of the
    //vdb map
    return true;
}

bool
VdbLightFilter::needsSamples() const {
    return mBlurValue > 0.f;
}

Color
VdbLightFilter::eval(const EvalData& data) const {
    // We need the vdb space position of the render space point.
    Vec3f vdbSpaceP0 = transformPoint(mVdbR2V[0], data.shadingPointPosition);
    Vec3f vdbSpaceP1 = transformPoint(mVdbR2V[1], data.shadingPointPosition);
    Vec3f vdbSpaceP = lerp(vdbSpaceP0, vdbSpaceP1, data.time);

    Vec3f offset = Vec3f(0.f, 0.f, 0.f);
    if (mBlurValue > 0.f) {
        switch (mBlurType) {
        case VDBLIGHTFILTERBLURTYPE_GAUSSIAN:
            // Quadratic bspline "gaussian" filter, which is the same filter the regular camera uses.
            offset.x = quadraticBSplineWarp(data.randVar.r3.x) * mBlurValue;
            offset.y = quadraticBSplineWarp(data.randVar.r3.y) * mBlurValue;
            offset.z = quadraticBSplineWarp(data.randVar.r3.z) * mBlurValue;
            break;
        case VDBLIGHTFILTERBLURTYPE_SPHERICAL:    
            //get random point on sphere
            offset = shading::sampleSphereUniform(data.randVar.r3.x, data.randVar.r3.y);
            //scale it by cuberoot(random) * radius for uniform distribution within sphere
            offset *= mBlurValue * scene_rdl2::math::pow(data.randVar.r3.z, 1.f/3.f);
            break;
        default:
            MNRY_ASSERT(0 && "unhandled blur Type");
        }
    }

    const Vec3f P(vdbSpaceP.x + offset.x, vdbSpaceP.y + offset.y, vdbSpaceP.z + offset.z);

    float densitySample;
    //sample color and density from vdb
    VdbLightFilter::sampleVdb(reinterpret_cast<const rdl2::LightFilter*>(this),
                              data.tls->mShadingTls.get(),
                              P.x, P.y, P.z,
                              &densitySample);
    // Invert the density sample before the remap
    if (mInvertDensity) {
        densitySample = 1.f - densitySample;
    }
    //remap the density
    densitySample = mDensityRamp.eval1D(densitySample);
        
    //clamp density/color to 0 separately in case both are negative, multiply below would make it positive
    densitySample = scene_rdl2::math::max(densitySample, 0.f);

    // density sample controls how much light should be unaffected. less density means
    // we shift towards the tinted color     
    Color result = lerp(mColorTint, Color(1.0f), densitySample);
    return result;
}

void 
VdbLightFilter::sampleVdb(const rdl2::LightFilter* lightFilter,
                          moonray::shading::TLState *tls,
                          const float px, 
                          const float py, 
                          const float pz,
                          float *outDensity) {
    const VdbLightFilter* me = reinterpret_cast<const VdbLightFilter*>(lightFilter);
    const Vec3f P(px, py, pz);
    if (me->mDensitySampler) {
        *outDensity = me->mDensitySampler->sample(tls, P,
                    static_cast<moonray::shading::OpenVdbSampler::Interpolation>(me->mVdbInterpolation)).r;
    } else {
        *outDensity = 1.f;
    }
}

} //namespace pbr
} //namespace moonray

