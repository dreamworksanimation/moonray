// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file AovLabels.hh

#pragma once

#ifdef ISPC
#include <scene_rdl2/common/platform/Platform.isph>
#define ISPC_UNIFORM uniform
#else
#include <scene_rdl2/common/platform/Platform.h>
#define ISPC_UNIFORM
#endif

// transformed      (bit 31),
// lpeLabelId       (bits [30, 16]),
// unused           (bit 15),
//  materialLabelId (bits [14, 0])
// If transformed bit is not set, then the label is just
// the raw value produced by the shader (or the default).
static const ISPC_UNIFORM int AOV_LABEL_TRANSFORMED_BIT = 0x80000000;
static const ISPC_UNIFORM int AOV_LABEL_ACTIVE_BITS     = 0x7fff;
static const ISPC_UNIFORM int AOV_LABEL_LPE_SHIFT       = 16;

// Label encoding
//
/// The basic problem is to transform the labels assigned by a material,
/// which are local to that material shader, into global indices in
/// the material and lpe aov systems.
//
/// @param label              The local label id assigned by a material shader
/// @param materialLabelId    The global id used in material aov expressions
///                           representing the material instance
/// @param lpeMaterialLabeLId The global id used in lpe light aov expressions
///                           representing the material instance
/// @param lobeLabelIds       Maps the local lobe label to global label used in
///                           material aov expressions
/// @param lpeLobeLabelIds    Maps the local lobe label to global label used in
///                           lpe light aov expressions
/// @return the global lpe and material id encoding
inline ISPC_UNIFORM int
SHADING_aovEncodeLabels(ISPC_UNIFORM int label,
                        const ISPC_UNIFORM int materialLabelId,
                        const ISPC_UNIFORM int lpeMaterialLabelId,
                        const ISPC_UNIFORM int * ISPC_UNIFORM lobeLabelIds,
                        const ISPC_UNIFORM int * ISPC_UNIFORM lpeLobeLabelIds)
{
    // if the label is already encoded, just return it
    if (label & AOV_LABEL_TRANSFORMED_BIT) return label;

    ISPC_UNIFORM int newLabelId = AOV_LABEL_TRANSFORMED_BIT;

    // set the material aov id in bits [0, 14]
    ISPC_UNIFORM int materialAovId = (label == 0 || !lobeLabelIds)? materialLabelId : lobeLabelIds[label];
    MNRY_ASSERT(materialAovId == -1 || materialAovId < AOV_LABEL_ACTIVE_BITS);
    newLabelId |= (materialAovId & AOV_LABEL_ACTIVE_BITS);

    // set the lpe aov id in bits [16, 30]
    ISPC_UNIFORM int lpeAovId = (label == 0 || !lpeLobeLabelIds)? lpeMaterialLabelId : lpeLobeLabelIds[label];
    MNRY_ASSERT(lpeAovId == -1 || lpeAovId < AOV_LABEL_ACTIVE_BITS);
    newLabelId |= (lpeAovId << AOV_LABEL_LPE_SHIFT);

    return newLabelId;
}

inline ISPC_UNIFORM bool
SHADING_aovLabelIsTransformed(ISPC_UNIFORM int label)
{
    return label & AOV_LABEL_TRANSFORMED_BIT? true : false;
}

inline ISPC_UNIFORM int
SHADING_aovDecodeLabel(ISPC_UNIFORM int encodedLabel)
{
    // all labels used for material aovs are expected to
    // be transformed
    MNRY_ASSERT(SHADING_aovLabelIsTransformed(encodedLabel));
    if (!SHADING_aovLabelIsTransformed(encodedLabel)) {
        // not much we can do, return "no label"
        return -1;
    }

    // material aov label is encoded in bits [0, 14]
    const ISPC_UNIFORM int result = encodedLabel & AOV_LABEL_ACTIVE_BITS;
    return result == AOV_LABEL_ACTIVE_BITS? -1 : result;
}

inline ISPC_UNIFORM int
SHADING_aovDecodeLpeLabel(ISPC_UNIFORM int encodedLabel)
{
    // decoding a non-encoded label for an lpe is an error
    // however, this does happen with the 'LambertBsdfLobe lobeLocal'
    // used by the subsurface integrator.  in this case, a lobe
    // which was not created via a shading operation works its
    // way into the light path.  before a client can call this
    // function, they must ensure that the label has been encoded
    // and handle the case where it is not appropriately.
    // (See LightAovs::computeScatterEventLabel for an example).
    MNRY_ASSERT(SHADING_aovLabelIsTransformed(encodedLabel));
    if (!SHADING_aovLabelIsTransformed(encodedLabel)) {
        // not much we can do, return "no label"
        return -1;
    }

    // lpe aov label is encoded in bits [16, 30]
    const ISPC_UNIFORM int result = (encodedLabel >> 16) & AOV_LABEL_ACTIVE_BITS;
    return result == AOV_LABEL_ACTIVE_BITS? -1 : result;
}


