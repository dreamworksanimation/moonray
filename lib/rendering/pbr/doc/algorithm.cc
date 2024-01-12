// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file algorithm.cc
/// $Id$
///



namespace moonray {
namespace pbr {


//-----------------------------------------------------------------------------

/*

BSDF SAMPLES:
-------------

draw_bsdf_samples()
{
    Foreach sample
        // We want to select the culled column in the table for the sampled lobe
        // based on the light intersected.
        bsmp = sample_bsdf()
        lphi = eval_lightset(bsmp.wi)
        // Compute one contribution per bsdf sampling strategy in that column
        Foreach contribution i
            strategy = sampledLobeStrategy(i)
            evaluate estimator for that strategy
}

sample_bsdf()
{
    Draw a sample from current lobe
    Foreach contribution i
        init
    Foreach lobe that belongs to the same subsets as the sampled lobe
        Eval() / pdf()
        Foreach contribution i
            Accumulate f and pdf * sampleCount(lobe)
    Foreach contribution i
        pdf /= sampleCout(lobeSubset(i))
        check if contribution is valid
    Set BIS2 non-combined contribution and pdf
}

eval_lightset(wi)
{
    {light, distance, N, uv} = intersect(wi)
    {Li, pdf} = light->eval(wi)
    pdf *= sampleCount(light) / sampleCount(lightSet)
    check if result is valid
}


LIGHT SAMPLES:
--------------

draw_light_samples()
{
    Foreach sample
        // TODO: Select row in the table for the sampled light
        // Compute one contribution per light sampling strategy in that row
        // (LIS? and MIS? only)
        // TODO: Cull based on lobes in the bsdf
        lsmp = sample_lightset()
        fLisMis = eval_bsdf(lsmp.wi)
        Foreach contribution i
            strategy = sampledLightStrategy(i)
            evaluate estimator for that strategy
}

sample_lightset()
{
    Foreach contribution i
        init
    {wi, distance, N, uv} = currentLight->sample()
    lightset->intersect(wi) to check no other light is closer
    {Li, pdf} = currentLight->eval(uv, N);
    pdf *= sampleCount(light) / sampleCount(lightSet)
    check if result is valid
}

eval_bsdf(wi)
{
    Foreach contribution i
        init
    Foreach lobe that belongs to the light sampling strategies in the selected row
        {f, pdf} = lobe->eval(wi)
        Foreach contribution i in the selected row
            Accumulate f and pdf * sampleCount(lobe)
    Foreach contribution i
        pdf /= sampleCout(lobeSubset(i))
        check if contribution is valid
}


 */


} // namespace pbr
} // namespace moonray


