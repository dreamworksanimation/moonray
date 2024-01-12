// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/// Stereo view attribute of the camera
enum class StereoView
{
    CENTER = 0,    ///< Render from center eye (stereo is off).
    LEFT,          ///< Render from left eye.
    RIGHT,         ///< Render from right eye.
};

