// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file StateMachine.ispc

#include "StateMachine.isph"

// must match StateMachine.cc
extern "C" uniform int CPP_LpeStateMachine_transition(const uniform LpeStateMachine * uniform stateMachine,
                                                      uniform int stateId,
                                                      uniform LpeEventType ev,
                                                      uniform LpeEventScatteringType evs,
                                                      uniform int labelId);

varying int
LpeStateMachine_transition(const uniform LpeStateMachine * uniform stateMachine,
                           varying int stateId,
                           uniform LpeEventType ev,
                           uniform LpeEventScatteringType evs,
                           varying int labelId)
{
    varying int result = -1;
    foreach_active(lane) {
        uniform int uStateId = extract(stateId, lane);
        uniform int uLabelId = extract(labelId, lane);
        const uniform int r = CPP_LpeStateMachine_transition(stateMachine, uStateId, ev, evs, uLabelId);
        result = insert(result, lane, r);
    }

    return result;
}

