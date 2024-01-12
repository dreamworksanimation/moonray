// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <vector>

namespace moonray {
namespace rt {

class GeomGenerateContext : public geom::GenerateContext {
public:
    GeomGenerateContext(scene_rdl2::rdl2::Layer* pRdlLayer,scene_rdl2::rdl2::Geometry* pRdlGeometry,
            shading::AttributeKeySet&& requestedAttributes,
            int currentFrame, int threads,
            const geom::MotionBlurParams& motionBlurParams):
            mRdlLayer(pRdlLayer), mRdlGeometry(pRdlGeometry),
            mRequestedAttributes(std::move(requestedAttributes)),
            mCurrentFrame(currentFrame),
            mThreads(threads), mMotionBlurParams(motionBlurParams)
    {
    }

    finline virtual const scene_rdl2::rdl2::Layer *getRdlLayer() const override {
        return mRdlLayer;
    }

    finline virtual const scene_rdl2::rdl2::Geometry *getRdlGeometry() const override {
        return mRdlGeometry;
    }

    finline virtual const shading::AttributeKeySet&
    getRequestedAttributes() const override {
        return mRequestedAttributes;
    }

    finline virtual bool
    requestAttribute(const shading::AttributeKey& attributeKey) const override {
        return mRequestedAttributes.find(attributeKey) !=
            mRequestedAttributes.end();
    }
     
    virtual int getCurrentFrame() const override;

    finline virtual int getThreads() const override {
        return mThreads;
    }

    virtual const geom::MotionBlurParams &getMotionBlurParams() const override;
    
    virtual const std::vector<float> &getMotionSteps() const override;
    
    virtual float getShutterOpen() const override;

    virtual float getShutterClose() const override;

    virtual bool isMotionBlurOn() const override;
    
    virtual void getMotionBlurDelta(
            float& shutterOpenDelta, float& shutterCloseDelta) const override;

private:

    scene_rdl2::rdl2::Layer *mRdlLayer;
    scene_rdl2::rdl2::Geometry *mRdlGeometry;
    shading::AttributeKeySet mRequestedAttributes;
    int mCurrentFrame;
    int mThreads;
    const geom::MotionBlurParams& mMotionBlurParams;
};


class GeomUpdateContext : public geom::UpdateContext {
public:

    GeomUpdateContext(scene_rdl2::rdl2::Layer *pRdlLayer, scene_rdl2::rdl2::Geometry *pRdlGeometry,
            int currentFrame, int threads,
            const geom::MotionBlurParams& motionBlurParams):
            mRdlLayer(pRdlLayer), mRdlGeometry(pRdlGeometry),
            mCurrentFrame(currentFrame), mThreads(threads),
            mMotionBlurParams(motionBlurParams)
    {
    }

    finline virtual const scene_rdl2::rdl2::Layer *getRdlLayer() const override {
        return mRdlLayer;
    }

    finline virtual const scene_rdl2::rdl2::Geometry *getRdlGeometry() const override {
        return mRdlGeometry;
    }

    void setMeshNames(const std::vector<std::string> &meshNames);

    void setMeshVertexDatas(const std::vector<const std::vector<float>* > &meshVertexDatas);

    virtual const std::vector<std::string> &getMeshNames () const override;

    virtual const std::vector< const std::vector<float>* > &getMeshVertexDatas () const override;

    virtual int getCurrentFrame() const override;

    finline virtual int getThreads() const override {
        return mThreads;
    }

    virtual const geom::MotionBlurParams &getMotionBlurParams() const override;

    virtual const std::vector<float> &getMotionSteps() const override;

    virtual float getShutterOpen() const override;

    virtual float getShutterClose() const override;

    virtual bool isMotionBlurOn() const override;

    virtual void getMotionBlurDelta(
            float& shutterOpenDelta, float& shutterCloseDelta) const override;

private:

    scene_rdl2::rdl2::Layer *mRdlLayer;
    scene_rdl2::rdl2::Geometry *mRdlGeometry;
    std::vector<std::string> mMeshNames;
    std::vector<const std::vector<float>* > mMeshVertexDatas;
    int mCurrentFrame;
    int mThreads;
    const geom::MotionBlurParams& mMotionBlurParams;
};

} // namespace rt
} // namespace moonray

