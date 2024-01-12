// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestStateMachine.cc

#include "TestStateMachine.h"

#include <moonray/rendering/lpe/StateMachine.h>

#include <cppunit/extensions/HelperMacros.h>

namespace moonray {
namespace lpe {
namespace unittest {

void
TestStateMachine::testLpe()
{
    StateMachine m;
    CPPUNIT_ASSERT(m.addExpression("CD*L", 1) == 0);
    CPPUNIT_ASSERT(m.addExpression("CSL", 2) == 0);
    CPPUNIT_ASSERT(m.addExpression("NOT AN LPE", 3) != 0);
    CPPUNIT_ASSERT(m.addExpression("C<.D'diffuse'>L", 4) == 0);
    CPPUNIT_ASSERT(m.addExpression("C[<.D'diffuse'><.D'base'>]L", 5) == 0);

    int diffuseLabel = m.getLabelId("diffuse");
    int baseLabel = m.getLabelId("base");
    CPPUNIT_ASSERT(diffuseLabel >= 0);
    CPPUNIT_ASSERT(baseLabel >= 0);

    m.build();

    // test some paths
    int stateId = 0; // initial state
    {
        // C_ RD L_
        stateId = m.transition(stateId, EVENT_TYPE_CAMERA, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        stateId = m.transition(stateId, EVENT_TYPE_REFLECTION, EVENT_SCATTERING_TYPE_DIFFUSE, sNoLabel);
        stateId = m.transition(stateId, EVENT_TYPE_LIGHT, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        CPPUNIT_ASSERT(m.isValid(stateId, 1));
        CPPUNIT_ASSERT(!m.isValid(stateId, 2));
        CPPUNIT_ASSERT(!m.isValid(stateId, 3));
        CPPUNIT_ASSERT(!m.isValid(stateId, 4));
        CPPUNIT_ASSERT(!m.isValid(stateId, 5));
    }

    stateId = 0; // initial state
    {
        // C_ Ts L_
        stateId = m.transition(stateId, EVENT_TYPE_CAMERA, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        stateId = m.transition(stateId, EVENT_TYPE_TRANSMISSION, EVENT_SCATTERING_TYPE_STRAIGHT, sNoLabel);
        stateId = m.transition(stateId, EVENT_TYPE_LIGHT, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        CPPUNIT_ASSERT(!m.isValid(stateId, 1));
        CPPUNIT_ASSERT(!m.isValid(stateId, 2));
        CPPUNIT_ASSERT(!m.isValid(stateId, 3));
        CPPUNIT_ASSERT(!m.isValid(stateId, 4));
        CPPUNIT_ASSERT(!m.isValid(stateId, 5));
    }

    stateId = 0; // initial state
    {
        // C_ RD'diffuse' L_
        stateId = m.transition(stateId, EVENT_TYPE_CAMERA, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        stateId = m.transition(stateId, EVENT_TYPE_REFLECTION, EVENT_SCATTERING_TYPE_DIFFUSE, diffuseLabel);
        stateId = m.transition(stateId, EVENT_TYPE_LIGHT, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        CPPUNIT_ASSERT(m.isValid(stateId, 1));
        CPPUNIT_ASSERT(!m.isValid(stateId, 2));
        CPPUNIT_ASSERT(!m.isValid(stateId, 3));
        CPPUNIT_ASSERT(m.isValid(stateId, 4));
        CPPUNIT_ASSERT(m.isValid(stateId, 5));
    }

    stateId = 0; // initial state
    {
        // C_ RD'base' L_
        stateId = m.transition(stateId, EVENT_TYPE_CAMERA, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        stateId = m.transition(stateId, EVENT_TYPE_REFLECTION, EVENT_SCATTERING_TYPE_DIFFUSE, baseLabel);
        stateId = m.transition(stateId, EVENT_TYPE_LIGHT, EVENT_SCATTERING_TYPE_NONE, sNoLabel);
        CPPUNIT_ASSERT(m.isValid(stateId, 1));
        CPPUNIT_ASSERT(!m.isValid(stateId, 2));
        CPPUNIT_ASSERT(!m.isValid(stateId, 3));
        CPPUNIT_ASSERT(!m.isValid(stateId, 4));
        CPPUNIT_ASSERT(m.isValid(stateId, 5));
    }
}

} // namespace unittest
} // namespace lpe
} // namespace moonray

