// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/pbr/core/Distribution.h>

#include <scene_rdl2/scene/rdl2/AttributeKey.h>

namespace moonray {
namespace pbr {

class LensDistribution
{
public:
    explicit LensDistribution(const scene_rdl2::rdl2::Camera* rdlCamera);
    static LensDistribution createUnitTestDistribution();

    void update();

    float sampleLens(float &u, float &v) const;
protected:

    const scene_rdl2::rdl2::Camera* mCamera;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>   mBokehKey;

    // Shape
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>    mBokehSidesKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> mBokehImageKey;

    // Control
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  mBokehAngleKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  mBokehWeightLocationKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>  mBokehWeightStrengthKey;

private:
    LensDistribution();

    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    void bokehPolygonBuilder();

    bool mBokeh;

    // Bokeh Shape
    int         mBokehSides;
    std::string mBokehImage;

    // Bokeh Controls
    float       mBokehAngle;
    float       mBokehWeightLocation;
    float       mBokehWeightStrength;

    // Bokeh Vectors
    std::unique_ptr<scene_rdl2::math::Vec2f[]> mBokehVertices;
    std::unique_ptr<ImageDistribution> mBokehImageDist;
};

} // namespace pbr
} // namespace moonray


