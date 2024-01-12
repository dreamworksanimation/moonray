// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include "attributes.cc"

using namespace scene_rdl2;
using namespace scene_rdl2::math;

RDL2_DSO_CLASS_BEGIN(OrthographicCamera, rdl2::Camera)

public:
    RDL2_DSO_DEFAULT_CTOR(OrthographicCamera)

    bool doesSupportProjectionMatrix() const override
    {
        return true;
    }

    math::Mat4f computeProjectionMatrix(float t, const std::array<float, 4>& window, float /*interocularOffset*/) const override
    {
        // Get film-back width (mm)
        float filmWidth = get(attrFilmWidthApertureKey);
        filmWidth = max(filmWidth, 0.01f);
        float halfFilmWidth = filmWidth * 0.5f;

        // Get near plane
        float near = get(rdl2::Camera::sNearKey);
        near = max(near, 0.01f);

        // Get far plane
        float far = get(rdl2::Camera::sFarKey);
        far = max(far, 0.01f);

        // Get film back offset
        float filmOffsetX = get(attrHorizontalFilmOffset);
        float filmOffsetY = get(attrVerticalFilmOffset);

        // Get pixel aspect ratio
        float par = get(attrPixelAspectRatio);
        if (isEqual(par, 0.0f)) {
            par = 1.0f;
        }

        // Compute window corners on the near plane
        float windowNear[4];
        windowNear[0] = (window[0] * halfFilmWidth + filmOffsetX);
        windowNear[1] = (window[1] * halfFilmWidth + filmOffsetY) / par;
        windowNear[2] = (window[2] * halfFilmWidth + filmOffsetX);
        windowNear[3] = (window[3] * halfFilmWidth + filmOffsetY) / par;

        // See Appendix C.5 of the SGI Graphics Library Programming Guide
        // for format of projection matrix.
        Mat4f c2s;
        c2s[0][0] = 2.0f / (windowNear[2] - windowNear[0]);
        c2s[0][1] = 0.0f;
        c2s[0][2] = 0.0f;
        c2s[0][3] = 0.0f;

        c2s[1][0] = 0.0f;
        c2s[1][1] = 2.0f / (windowNear[3] - windowNear[1]);
        c2s[1][2] = 0.0f;
        c2s[1][3] = 0.0f;

        c2s[2][0] = 0.0f;
        c2s[2][1] = 0.0f;
        c2s[2][2] = -2.0f / (far - near);
        c2s[2][3] = 0.0f;

        c2s[3][0] = -(windowNear[2] + windowNear[0])/(windowNear[2] - windowNear[0]);
        c2s[3][1] = -(windowNear[3] + windowNear[1])/(windowNear[3] - windowNear[1]);
        c2s[3][2] = -(far + near) / (far - near);
        c2s[3][3] = 1.0f;

        return c2s;
    }

RDL2_DSO_CLASS_END(OrthographicCamera)

