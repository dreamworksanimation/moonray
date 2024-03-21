// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestSceneContext.h"

#include <scene_rdl2/common/math/Color.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <scene_rdl2/scene/rdl2/AttributeKey.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>
#include <scene_rdl2/scene/rdl2/SceneClass.h>
#include <scene_rdl2/scene/rdl2/SceneObject.h>
#include <scene_rdl2/scene/rdl2/SceneVariables.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#include <string>

namespace moonray {
namespace unittest {

void
TestSceneContext::setUp()
{
}

void
TestSceneContext::tearDown()
{
}

void
TestSceneContext::testUpdate()
{
    scene_rdl2::rdl2::SceneContext context;
    const scene_rdl2::rdl2::SceneClass* updateTracker = context.createSceneClass("UpdateTracker");
    scene_rdl2::rdl2::Map* tracker = context.createSceneObject(
        "UpdateTracker", "/seq/shot/tracker")->asA<scene_rdl2::rdl2::Map>();
    scene_rdl2::rdl2::Map* other = context.createSceneObject(
        "UpdateTracker", "/seq/shot/other")->asA<scene_rdl2::rdl2::Map>();
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> pizzaKey =
        updateTracker->getAttributeKey<scene_rdl2::rdl2::Float>("pizza");
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> cookieKey =
        updateTracker->getAttributeKey<scene_rdl2::rdl2::Int>("cookie");

    // Make some updates and commit them.
    tracker->beginUpdate();
    tracker->set(pizzaKey, 2.34f);
    tracker->setBinding(pizzaKey, other);
    tracker->set(cookieKey, 13);
    tracker->endUpdate();
    tracker->commitChanges();

    tracker->beginUpdate();
    tracker->set(pizzaKey, 3.45f);
    tracker->set(cookieKey, 14);
    tracker->endUpdate();
    tracker->commitChanges();

    tracker->beginUpdate();
    tracker->set(pizzaKey, 4.56f);
    tracker->endUpdate();
    tracker->commitChanges();

    // Verify the update counts.
    moonray::shading::Intersection intersection;
    moonray::shading::State state(&intersection);
    scene_rdl2::math::Color result;
    tracker->sample(nullptr, state, &result);
    int timesPizzaUpdated = result.r;
    int timesPizzaBindingUpdated = result.g;
    int timesCookieUpdated = result.b;
    CPPUNIT_ASSERT_EQUAL(3, timesPizzaUpdated);
    CPPUNIT_ASSERT_EQUAL(1, timesPizzaBindingUpdated);
    CPPUNIT_ASSERT_EQUAL(2, timesCookieUpdated);
}

} // namespace unittest
} // namespace moonray

