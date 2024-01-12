// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file StateMachine.h

#pragma once

#include "Event.h"

#include <memory>
#include <string>
#include <vector>

namespace moonray {
namespace lpe {

/// @class StateMachine StateMachine.h <lpe/StateMachine.h>
/// @brief Implements a finite state machine for light path expressions
class StateMachine
{
public:
    /// This is the initial state id to use before beginning a
    /// series of transitions.
    static const int sInitialStateId = 0;

    StateMachine();
    ~StateMachine();

    /// @param lp a light path expression
    /// @param id identifier for this lpe
    /// @return -1 on error, 0 otherwise
    int addExpression(std::string const &lp, int id);

    /// @param label a label that might have been used in an expression
    /// @return labelId for label, if it appeared in an expression, -1 otherwise
    int getLabelId(std::string const &label) const;

    /// After all expressions have been added, the state machine
    /// can be built.
    void build();

    /// Transition the state machine based on an event and
    /// event scattering type.
    /// @param stateId current state to transition from
    /// @param ev event type
    /// @param evs scattering type
    /// @param labelId optional labelId appending after evs, -1 if skipped
    /// @return new stateId
    int transition(int stateId, EventType ev, EventScatteringType evs, int labelId) const;

    /// @param stateId current state of machine
    /// @param id the id of the expression to test
    /// @return true if id is valid at this stateId, false otherwise
    bool isValid(int stateId, int id) const;

private:
    class Impl;

    StateMachine(StateMachine const &) = delete;
    StateMachine &operator=(StateMachine const &) = delete;
    
    std::unique_ptr<Impl> mImpl;
};


} // namespace lpe
} // namespace moonray

