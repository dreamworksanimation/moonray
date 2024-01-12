// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ImageColorCorrect.ispc
/// $Id$
///

#include <moonray/rendering/shading/ispc/ColorCorrect.isph>

export 
void 
PBR_colorCorrectFormatArray(uniform unsigned int size,
                            uniform bool gamma_on,
                            uniform float gamma_x,
                            uniform float gamma_y,
                            uniform float gamma_z,

                            uniform bool gain_offset_on,
                            uniform float gain_x,
                            uniform float gain_y,
                            uniform float gain_z,
                            uniform float offset_x,
                            uniform float offset_y,
                            uniform float offset_z,

                            uniform bool contrast_on,
                            uniform float contrast_x,
                            uniform float contrast_y,
                            uniform float contrast_z,

                            uniform bool saturation_on,
                            uniform float sat_r,
                            uniform float sat_g,
                            uniform float sat_b,

                            uniform bool temperature_control_on,
                            uniform float t,
                            uniform float m,
                            uniform float e,

                            uniform float pixelBuffer[])
{


    // It seems to me like a big waste to run through all the color correction
    // operations in most cases. I would expect most of the time only a couple
    // of the following color correction ops to be truly needed. Shouldn't we
    // have one ISPC kernel that is decomposed into inlined functions that are
    // only invoked if the color correction operation is needed ? That conditional
    // statement is going to be uniform (only depends on the input settings) so
    // it should not cause SIMD code divergence.

    static const uniform float epsilon = 0.000001;

    const uniform float gx = 1.0f / max(epsilon, gamma_x);
    const uniform float gy = 1.0f / max(epsilon, gamma_y);
    const uniform float gz = 1.0f / max(epsilon, gamma_z);

    float in_r, in_g, in_b;
    float r, g, b;

    const uniform int step = programCount * 3;
    const uniform int resize = size * step;
    for (uniform int i = 0; i < resize; i += step) {
        aos_to_soa3( &pixelBuffer[i], &r, &g, &b);

        if (saturation_on) {
            applySaturation(sat_r, sat_g, sat_b, r, g, b);
        }

        if (contrast_on) {
            applyNukeContrast(  contrast_x, contrast_y, contrast_z,
                                r, g, b);
        }

        if (gamma_on) {
            applyGamma(gx, gy, gz, r, g, b);
        }

        if (gain_offset_on) {
            applyGainAndOffset(  gain_x, gain_y, gain_z,
                            offset_x, offset_y, offset_z,
                            r, g, b);
        }

        // TME(I) 
        if (temperature_control_on) {
            applyTMI(t, m, e, r, g, b);
        }

        r = max(r, 0.0f);
        g = max(g, 0.0f);
        b = max(b, 0.0f);

        soa_to_aos3(r, g, b, &pixelBuffer[i]);
    }
}
