// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Event.hh

#pragma once

#define LPE_EVENT_TYPE_ENUMS(NS)            \
    NS##EVENT_TYPE_NONE,                    \
    NS##EVENT_TYPE_CAMERA,                  \
    NS##EVENT_TYPE_REFLECTION,              \
    NS##EVENT_TYPE_TRANSMISSION,            \
    NS##EVENT_TYPE_VOLUME,                  \
    NS##EVENT_TYPE_LIGHT,                   \
    NS##EVENT_TYPE_EMISSION,                \
    NS##EVENT_TYPE_BACKGROUND,              \
    NS##EVENT_TYPE_EXTRA,                   \
    NS##EVENT_TYPE_MATERIAL

#define LPE_EVENT_SCATTERING_TYPE_ENUMS(NS) \
    NS##EVENT_SCATTERING_TYPE_NONE,         \
    NS##EVENT_SCATTERING_TYPE_DIFFUSE,      \
    NS##EVENT_SCATTERING_TYPE_GLOSSY,       \
    NS##EVENT_SCATTERING_TYPE_MIRROR,       \
    NS##EVENT_SCATTERING_TYPE_STRAIGHT

#define LPE_NO_LABEL -1

