// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ProceduralContext.h
/// $Id$
///

#pragma once

#ifndef GEOM_PROCEDURAL_CONTEXT_HAS_BEEN_INCLUDED
#define GEOM_PROCEDURAL_CONTEXT_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/Types.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/mcrt_common/Frustum.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>

#include <string>
#include <vector>

namespace moonray {
namespace geom  {

///
/// @class ProceduralContext
/// @brief defines the interface to query the data required for
///     procedural to generation and update primitives
///
class ProceduralContext
{

public:
    virtual ~ProceduralContext() {}
   
    /// query scene_rdl2::rdl2::Layer the procedural belong
    /// @return scene_rdl2::rdl2::Layer the pointer to scene_rdl2::rdl2::Layer
    virtual const scene_rdl2::rdl2::Layer *getRdlLayer() const = 0;

    /// query scene_rdl2::rdl2::Geometry the procedural belong
    /// @return scene_rdl2::rdl2::Geometry the pointer to scene_rdl2::rdl2::Geometry
    virtual const scene_rdl2::rdl2::Geometry *getRdlGeometry() const = 0;

    /// Get the current frame number
    /// @return the current frame no.
    virtual int  getCurrentFrame() const = 0;
    
    /// Get the number of worker threads in use.
    /// @return the number of worker threads in use
    virtual int getThreads() const = 0;

    /// Get the parameters related to motion blur
    /// @return the MotionBlurParams that wraps motion blur related parameters
    virtual const MotionBlurParams& getMotionBlurParams() const = 0;

    /// Get the motion blur steps in frame relative value
    /// (For example: -1 stands for previous frame, 0 stands for current frame)
    /// @return the vector contains motion steps
    virtual const std::vector<float>& getMotionSteps() const = 0;

    /// Get the shutter open time in frame relative value
    /// (For example: -0.5 stands for the middle of previous frame and current frame)
    /// @return the shutter open time in frame relative value
    virtual float getShutterOpen() const = 0;
    
    /// Get the shutter close time in frame relative value
    /// (For example: 0.5 stands for the middle of current frame and next frame)
    /// @return the shutter close time in frame relative value
    virtual float getShutterClose() const = 0;

    /// query whether camera motion blur is turned on
    /// @return whether camera motino blur is turned on
    virtual bool isMotionBlurOn() const = 0;

    /// Return the delta fraction of shutter open/close time in
    /// motionSteps duration
    virtual void getMotionBlurDelta(
            float& shutterOpenDelta, float& shutterCloseDelta) const = 0;

};

///
/// @class GenerateContext
/// @brief defines the interface to query the data required for
///     procedural to generate primitives
///
class GenerateContext : public ProceduralContext {

public:
    /// The set of primitive attributes that should be provided by this procedural.
    /// This set is computed from the primitive attributes requested by Shaders
    /// bound to this procedural.
    virtual const shading::AttributeKeySet &getRequestedAttributes() const = 0;

    virtual bool requestAttribute(const shading::AttributeKey& attributeKey) const = 0;
};

///
/// @class UpdateContext
/// @remark For realtime rendering use. Feature film shader development
///     does not require this functionality.
/// @brief defines the interface to query the data required for
///     procedural to update primitives
///
class UpdateContext : public ProceduralContext {

public:
    /// Get the mesh names that need be updated.
    /// @return the mesh names that need be updated.
    virtual const std::vector<std::string> &getMeshNames() const = 0;
    
    /// Get the mesh update vertex data from animation
    /// @return the mesh update vertex data from animation
    virtual const std::vector<const std::vector<float>* > &getMeshVertexDatas() const = 0;
};

} // end of name space geom
} // end of name space moonray

#endif /* GEOM_PROCEDURAL_CONTEXT_HAS_BEEN_INCLUDED */

