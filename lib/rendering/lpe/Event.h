// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Event.h

#pragma once

#include "Event.hh"

namespace moonray {
namespace lpe {

enum EventType {
    LPE_EVENT_TYPE_ENUMS()
};

enum EventScatteringType {
    LPE_EVENT_SCATTERING_TYPE_ENUMS()
};

const int sNoLabel = LPE_NO_LABEL;


} // namespace lpe
} // namespace moonray

