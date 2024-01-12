// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LensDistribution.h"

#include "Camera.h"
#include "ProjectiveCamera.h"

#include <moonray/rendering/pbr/core/Distribution.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/except/exceptions.h>

namespace moonray {
namespace pbr {
LensDistribution LensDistribution::createUnitTestDistribution()
{
    LensDistribution ret;
    return ret;
}

LensDistribution::LensDistribution() :
    mCamera(nullptr),
    mBokeh(false),
    mBokehSides(0),
    mBokehImage(),
    mBokehAngle(0.0f),
    mBokehWeightLocation(0.0f),
    mBokehWeightStrength(0.0f)
{
}

LensDistribution::LensDistribution(const scene_rdl2::rdl2::Camera* rdlCamera) :
    mCamera(rdlCamera),
    mBokeh(false),
    mBokehSides(0),
    mBokehImage(),
    mBokehAngle(0.0f),
    mBokehWeightLocation(0.0f),
    mBokehWeightStrength(0.0f)
{
    MNRY_ASSERT(mCamera != nullptr);
    initAttributeKeys(rdlCamera->getSceneClass());
}

void
LensDistribution::initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass)
{
    try {
        mBokehKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool>("bokeh");

        // Bokeh Shape
        mBokehSidesKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Int>("bokeh_sides");
        mBokehImageKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::String>("bokeh_image");

        // Bokeh Controls
        mBokehAngleKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("bokeh_angle");
        mBokehWeightLocationKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("bokeh_weight_location");
        mBokehWeightStrengthKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("bokeh_weight_strength");
    } catch (const scene_rdl2::except::KeyError&) {
        scene_rdl2::Logger::error("This SceneClass does not support Bokeh. Disabling Bokeh.");
    }
}

void
LensDistribution::update()
{
    try {
        mBokeh = mCamera->get(mBokehKey);

        if (mBokeh) {
            // Shape
            mBokehSides = mCamera->get(mBokehSidesKey);
            mBokehImage = mCamera->get(mBokehImageKey);

            // Controls
            mBokehAngle          = scene_rdl2::math::deg2rad(mCamera->get(mBokehAngleKey));
            mBokehWeightLocation = mCamera->get(mBokehWeightLocationKey);
            mBokehWeightStrength = mCamera->get(mBokehWeightStrengthKey);

            if (!mBokehImage.empty()) {
                try {
                    mBokehImageDist.reset(new ImageDistribution(mBokehImage, Distribution2D::PLANAR));
                } catch (const scene_rdl2::except::IoError&) {
                    scene_rdl2::Logger::error("Failed to open: ", mBokehImage, ". Switching to Disk mode.");
                    mBokehImage = "";
                }
            } else if (mBokehSides >= 3) {
                bokehPolygonBuilder();
            } else if (mBokehSides > 0) {
                scene_rdl2::Logger::error("Unable to create a shape with less than 3 vertices. Switching to Disk mode.");
            }
        }
    } catch (const scene_rdl2::except::KeyError&) {
        mBokeh = false;
    }
}

float
LensDistribution::sampleLens(float &u, float &v) const
{
    MNRY_ASSERT(0.0f <= u && u < 1.0f);
    MNRY_ASSERT(0.0f <= v && v < 1.0f);

    if (mBokeh) {
        // Input Coordinates - [0, 1)^2
        float localU = u;
        float localV = v;

        if (!mBokehImage.empty()) {
            // Image Mode
            scene_rdl2::math::Vec2f uv;

            mBokehImageDist->sample(localU, localV, 0, &uv, nullptr, TEXTURE_FILTER_NEAREST);

            // Output - [-1, 1]^2
            localU = (2.0f * uv.x) - 1.0f;
            localV = (2.0f * uv.y) - 1.0f;
        } else if (mBokehSides >= 3) {
            // Polygon Mode
            // Graphics Gems I p. 650
            // Find the closest index and the next counter-clockwise index
            int curIndex = std::min(static_cast<int>(localU * mBokehSides), mBokehSides);
            int nextIndex = (curIndex + 1) % mBokehSides;

            // Map u to a point in the sub-triangle
            localU = (localU * mBokehSides) - static_cast<float>(curIndex);
            localV = scene_rdl2::math::sqrt(localV); // Necessary to weight all portions of triangle equally

            // Barycentric coordinates of the triangle
            float a = 1.0f - localV;
            float b = (1.0f - localU) * localV;
            float c = (localU * localV);

            // Vertices of the triangle
            scene_rdl2::math::Vec2f A = scene_rdl2::math::Vec2f(0.0f, 0.0f);
            scene_rdl2::math::Vec2f B = mBokehVertices[curIndex];
            scene_rdl2::math::Vec2f C = mBokehVertices[nextIndex];

            // Mapping the point to a new location within the triangle
            // Output - [-1, 1]^2
            localU = a * A.x + b * B.x + c * C.x;
            localV = a * A.y + b * B.y + c * C.y;
        } else {
            // Disk Mode - Default. Also used if Image cannot be found or Shape cannot be generated.
            // Output - [-1, 1]^2
            toUnitDisk(localU, localV);
        }

        u = localU;
        v = localV;

        // For the pronounced edge effect, the sample's weight needs to be altered depending on position
        // Find the distance of (u, v) from origin
        float localDistance = scene_rdl2::math::sqrt(scene_rdl2::math::pow(scene_rdl2::math::abs(localU), 2) + scene_rdl2::math::pow(scene_rdl2::math::abs(localV), 2));

        float weightLocation = mBokehWeightLocation; // mean
        float weightStrength = mBokehWeightStrength; // stddev

        // Weighting Calculation
        // PDF of a normal function
        // TODO: Edge detection for image and polygon modes
        // TODO: Integrate to 1 so weight values do not need to be manipulated by user
        float weight = (1.0f / (weightStrength * scene_rdl2::math::sqrt(scene_rdl2::math::sTwoPi))) 
                 * scene_rdl2::math::exp((-1.0f * scene_rdl2::math::pow(localDistance - weightLocation, 2)) / 
                             (2.0f * scene_rdl2::math::pow(weightStrength, 2)))
                 + 0.5f;

        return weight;
    } else {
        // TODO: Keith, evaluate Cranley-Patterson rotations or rotational rotations
        // If bokeh is off, just map to a unit disk
        toUnitDisk(u, v);

        return 1.0f;
    }
}

void
LensDistribution::bokehPolygonBuilder()
{
    float mBokehStep = scene_rdl2::math::sTwoPi / mBokehSides; // Figure out the step size
    mBokehVertices.reset(new scene_rdl2::math::Vec2f[mBokehSides]);

    // Calculate the vertices for the polygon
    float curStep = mBokehAngle;
    for (int i = 0; i < mBokehSides; ++i) {
        scene_rdl2::math::sincos(curStep, &mBokehVertices[i].y, &mBokehVertices[i].x);
        curStep += mBokehStep;
    }
}

} // namespace pbr
} // namespace moonray


