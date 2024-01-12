// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "GeomContext.h"

namespace moonray {
namespace rt {

int
GeomGenerateContext::getCurrentFrame() const
{
    return mCurrentFrame;
}

const geom::MotionBlurParams&
GeomGenerateContext::getMotionBlurParams() const
{
    return mMotionBlurParams;
}

const std::vector<float> &
GeomGenerateContext::getMotionSteps() const
{
    return mMotionBlurParams.getMotionSteps();
}

float
GeomGenerateContext::getShutterOpen() const
{
    return mMotionBlurParams.getShutterOpen();
}

float
GeomGenerateContext::getShutterClose() const
{
    return mMotionBlurParams.getShutterClose();
}

void
GeomGenerateContext::getMotionBlurDelta(float& shutterOpenDelta,
        float& shutterCloseDelta) const {
    mMotionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
}

bool
GeomGenerateContext::isMotionBlurOn() const
{
    return mMotionBlurParams.isMotionBlurOn();
}

void
GeomUpdateContext::setMeshNames(const std::vector<std::string> &meshNames)
{
    this->mMeshNames = meshNames;
}
    
const std::vector<std::string> &
GeomUpdateContext::getMeshNames() const
{
    return mMeshNames;
}

void
GeomUpdateContext::setMeshVertexDatas(const std::vector<const std::vector<float>* > &meshVertexDatas)
{
    this->mMeshVertexDatas = meshVertexDatas;
}
        
const std::vector<const std::vector<float>* > &
GeomUpdateContext::getMeshVertexDatas() const
{
    return mMeshVertexDatas;
}

int
GeomUpdateContext::getCurrentFrame() const
{
    return mCurrentFrame;
}

const geom::MotionBlurParams&
GeomUpdateContext::getMotionBlurParams() const
{
    return mMotionBlurParams;
}

const std::vector<float>&
GeomUpdateContext::getMotionSteps() const
{
    return mMotionBlurParams.getMotionSteps();
}

float
GeomUpdateContext::getShutterOpen() const
{
    return mMotionBlurParams.getShutterOpen();
}

float
GeomUpdateContext::getShutterClose() const
{
    return mMotionBlurParams.getShutterClose();
}

void
GeomUpdateContext::getMotionBlurDelta(float& shutterOpenDelta,
        float& shutterCloseDelta) const {
    mMotionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
}

bool
GeomUpdateContext::isMotionBlurOn() const
{
    return mMotionBlurParams.getMotionSteps().size() > 1;
}

} // namespace rt
} // namespace moonray

