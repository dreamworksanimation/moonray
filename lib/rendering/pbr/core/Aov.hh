// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Aov.hh
#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

// Sizeof members of STL types for ispc
#if STL_VERSION <= 4
#define SIZEOF_BGEXTRAAOVS          24
#define SIZEOF_LPE_STATEMACHINE     8
#define SIZEOF_LABELSUBSTITUTIONS   48
#define SIZEOF_STD_VECTOR           24
#define AOV_SCHEMA_MEMBERS_PADDING  6
#else
#define SIZEOF_BGEXTRAAOVS          24
#define SIZEOF_LPE_STATEMACHINE     8
#define SIZEOF_LABELSUBSTITUTIONS   56
#define SIZEOF_STD_VECTOR           24
#define AOV_SCHEMA_MEMBERS_PADDING  6
#endif

// The ispc code does not have access to the AovSchemaID enumeration.
// Its getStateVar function is written using this enum
enum {
    AOV_STATE_VAR_P = 1,
    AOV_STATE_VAR_NG,
    AOV_STATE_VAR_N,
    AOV_STATE_VAR_ST,
    AOV_STATE_VAR_DPDS,
    AOV_STATE_VAR_DPDT,
    AOV_STATE_VAR_DSDX,
    AOV_STATE_VAR_DSDY,
    AOV_STATE_VAR_DTDX,
    AOV_STATE_VAR_DTDY,
    AOV_STATE_VAR_WP,
    AOV_STATE_VAR_DEPTH,
    AOV_STATE_VAR_MOTION
};

#define AOV_SCHEMA_MEMBERS                                              \
    HUD_CPP_MEMBER(std::vector<Entry>, mEntries, SIZEOF_STD_VECTOR);    \
    HUD_CPP_MEMBER(int, mAllLpePrefixFlags, 4);                         \
    HUD_MEMBER(unsigned int, mNumChannels);                             \
    HUD_MEMBER(bool, mHasAovFilter);                                    \
    HUD_MEMBER(bool, mHasClosestFilter);                                \
    HUD_ISPC_PAD(mPad, AOV_SCHEMA_MEMBERS_PADDING)

#define AOV_SCHEMA_VALIDATION                    \
    HUD_BEGIN_VALIDATION(AovSchema);             \
    HUD_VALIDATE(AovSchema, mEntries);           \
    HUD_VALIDATE(AovSchema, mAllLpePrefixFlags); \
    HUD_VALIDATE(AovSchema, mNumChannels);       \
    HUD_VALIDATE(AovSchema, mHasAovFilter);      \
    HUD_VALIDATE(AovSchema, mHasClosestFilter);  \
    HUD_END_VALIDATION


#define MATERIAL_AOVS_MEMBERS                                           \
    HUD_CPP_MEMBER(std::vector<Entry>, mEntries, SIZEOF_STD_VECTOR);    \
    HUD_CPP_MEMBER(std::vector<std::string>, mLabels,                   \
                   SIZEOF_STD_VECTOR);                                  \
    HUD_CPP_MEMBER(std::vector<std::string>, mMaterialLabels,           \
                   SIZEOF_STD_VECTOR);                                  \
    HUD_CPP_MEMBER(std::vector<std::string>, mGeomLabels, SIZEOF_STD_VECTOR)

#define MATERIAL_AOVS_VALIDATION                        \
    HUD_BEGIN_VALIDATION(MaterialAovs);                 \
    HUD_VALIDATE(MaterialAovs, mEntries);               \
    HUD_VALIDATE(MaterialAovs, mLabels);                \
    HUD_VALIDATE(MaterialAovs, mMaterialLabels);        \
    HUD_END_VALIDATION
    

#define LIGHT_AOVS_MEMBERS                                             \
    HUD_CPP_MEMBER(lpe::StateMachine, mLpeStateMachine,                \
                   SIZEOF_LPE_STATEMACHINE);                           \
    HUD_CPP_MEMBER(LabelSubstitutions, mLabelSubstitutions,            \
                   SIZEOF_LABELSUBSTITUTIONS);                         \
    HUD_CPP_MEMBER(BackgroundExtraAovs, mBackgroundExtraAovs, SIZEOF_BGEXTRAAOVS); \
    HUD_MEMBER(int, mNextLightAovSchemaId);                            \
    HUD_MEMBER(int, mNextVisibilityAovSchemaId);                       \
    HUD_MEMBER(bool, mFinalized);                                      \
    HUD_ISPC_PAD(mPad, 4)
    
#define LIGHT_AOVS_VALIDATION                                   \
    HUD_BEGIN_VALIDATION(LightAovs);                            \
    HUD_VALIDATE(LightAovs, mLpeStateMachine);                  \
    HUD_VALIDATE(LightAovs, mLabelSubstitutions);               \
    HUD_VALIDATE(LightAovs, mBackgroundExtraAovs);              \
    HUD_VALIDATE(LightAovs, mNextLightAovSchemaId);             \
    HUD_VALIDATE(LightAovs, mNextVisibilityAovSchemaId);        \
    HUD_VALIDATE(LightAovs, mFinalized);                        \
    HUD_END_VALIDATION

