// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Bsdf.ispc

#include "Bsdf.isph"
#include "BsdfSlice.isph"
#include "BsdfUtil.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>


#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(Bsdf);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(BsdfLobe);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(BsdfLobeName);

//----------------------------------------------------------------------------
uniform bool
BsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                     uniform BsdfLobeProperty property,
                     varying float * uniform dest)
{
    uniform bool handled = true;

    BsdfLobeGetPropertyFn getPropertyFn = (BsdfLobeGetPropertyFn) lobe->mGetPropertyFn;
    if (!getPropertyFn(lobe, property, dest)) {
        // handle what we can
        switch (property) {
        case BSDF_LOBE_PROPERTY_COLOR:
            {
                varying Color c = BsdfLobe_computeScaleAndFresnel(lobe, 1.f);
                *dest       = c.r;
                *(dest + 1) = c.g;
                *(dest + 2) = c.b;
            }
            break;
        default:
            handled = false;
        }
    }

    return handled;
}

//----------------------------------------------------------------------------

varying Color
Bsdf_eval(const varying Bsdf * uniform bsdf, const varying BsdfSlice &slice,
          const varying Vec3f &wi, varying float &pdf)
{
    varying BsdfLobeType flags = BsdfSlice_getSurfaceFlags(&slice, *bsdf, wi);

    // Add up all matching lobes' eval()
    varying Color f = Color_ctor(0.f);
    pdf = 0.f;
    uniform int size = bsdf->mNumLobes;
    for (uniform int l = 0; l < size; ++l) {
        varying BsdfLobe * uniform lobe = bsdf->mLobes[l];

        if (isActive(lobe)) {
            // We need to account for lobe pdf, even if the surface flag doesn't
            // match
            varying float tmpPdf = 0.f;
            varying Color color = BsdfLobe_eval(lobe, slice, wi, &tmpPdf);
            if (BsdfLobe_matchesFlags(lobe, flags)) {
                f = f + color;
            }
            pdf += tmpPdf;
        }
    }

    // Assumes all lobes are equi-probabe
    if (size > 1) {
        pdf /= size;
    }

    return f;
}

varying Color
Bsdf_albedo(const varying Bsdf &bsdf,
            const varying BsdfSlice &slice)
{
    varying Color result = Color_ctor(0.f);
    uniform int size = bsdf.mNumLobes;
    for (uniform int l = 0; l < size; ++l) {
        varying BsdfLobe * uniform lobe = bsdf.mLobes[l];
        if (BsdfLobe_matchesFlags(lobe, slice.mFlags)) {
            result = result + BsdfLobe_albedo(lobe, slice);
        }
    }

    return result;
}

void
Bsdf_show(const varying Bsdf &bsdf,
          const uniform Material * const uniform material)
{
    CPP_printBsdfHeader(material, &bsdf);

    for (uniform int i = 0; i < bsdf.mNumLobes; ++i) {
        varying BsdfLobe * uniform lobe = bsdf.mLobes[i];
        BsdfLobe_show(lobe, 0);
        print("\n");
    }

    if (bsdf.mBssrdf) {
        Bssrdf_show(bsdf.mBssrdf, 0);
        print("\n");
    }

    print("[emission] = % % %\n", BSDF_UTIL_EXTRACT(bsdf.mSelfEmission.r),
                                  BSDF_UTIL_EXTRACT(bsdf.mSelfEmission.g),
                                  BSDF_UTIL_EXTRACT(bsdf.mSelfEmission.b));

    print("\n==========================================================\n");
}

void
Bsdf_setPostScatterExtraAovs(varying Bsdf * uniform bsdf,
                             uniform int numExtraAovs,
                             const uniform int * uniform labelIds,
                             const varying Color * uniform colors)
{
    bsdf->mPostScatterExtraAovs.mNum = numExtraAovs;
    bsdf->mPostScatterExtraAovs.mLabelIds = labelIds;
    bsdf->mPostScatterExtraAovs.mColors = colors;
}

//----------------------------------------------------------------------------

