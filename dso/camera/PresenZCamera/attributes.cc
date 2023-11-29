// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    // General
    rdl2::AttributeKey<rdl2::Bool>   attrPresenZEnabled;
    rdl2::AttributeKey<rdl2::Int>    attrPhase;
    rdl2::AttributeKey<rdl2::String> attrDetectFile;
    rdl2::AttributeKey<rdl2::String> attrRenderFile;
    rdl2::AttributeKey<rdl2::Float>  attrRenderScale;
    rdl2::AttributeKey<rdl2::Vec3f>  attrZOVScale;
    rdl2::AttributeKey<rdl2::Float>  attrDistanceToGround;

    // Rendering
    rdl2::AttributeKey<rdl2::Bool>  attrDraftRendering;
    rdl2::AttributeKey<rdl2::Bool>  attrFroxtrumRendering;
    rdl2::AttributeKey<rdl2::Int>   attrFroxtrumDepth;
    rdl2::AttributeKey<rdl2::Int>   attrFroxtrumResolution;
    rdl2::AttributeKey<rdl2::Bool>  attrRenderInsideZOV;
    rdl2::AttributeKey<rdl2::Bool>  attrEnableDeepReflections;
    rdl2::AttributeKey<rdl2::Float> attrInterPupillaryDistance;

    // Multiboxing
    rdl2::AttributeKey<rdl2::Int> attrZOVOffsetX;
    rdl2::AttributeKey<rdl2::Int> attrZOVOffsetY;
    rdl2::AttributeKey<rdl2::Int> attrZOVOffsetZ;
    rdl2::AttributeKey<rdl2::Vec3f> attrSpecularPointOffset;

    // Clipping
    rdl2::AttributeKey<rdl2::Bool>  attrEnableClippingSphere;
    rdl2::AttributeKey<rdl2::Float> attrClippingSphereRadius;
    rdl2::AttributeKey<rdl2::Vec3f> attrClippingSphereCenter;
    rdl2::AttributeKey<rdl2::Bool>  attrClippingSphereRenderInside;

RDL2_DSO_ATTR_DEFINE(rdl2::Camera)

    // General
    attrPresenZEnabled =
        sceneClass.declareAttribute<rdl2::Bool>("presenz_enabled", true);
    sceneClass.setMetadata(attrPresenZEnabled, "label", "presenz enabled");
    sceneClass.setMetadata(attrPresenZEnabled, "comment",
        "If enabled use the PresenZ camera.   If disabled, uses a spherical camera.\n");

    attrPhase =
        sceneClass.declareAttribute<rdl2::Int>("phase", 0, rdl2::FLAGS_ENUMERABLE);
    sceneClass.setMetadata(attrPhase, "label", "phase");
    sceneClass.setMetadata(attrPhase, "comment",
        "PresenZ renders with a two phase system, consisting of the detection phase and the render phase.\n");
    sceneClass.setEnumValue(attrPhase, 0, "detect");
    sceneClass.setEnumValue(attrPhase, 1, "render");

    attrDetectFile =
        sceneClass.declareAttribute<rdl2::String>("detect_file", "render.przDetect", rdl2::FLAGS_FILENAME);
    sceneClass.setMetadata(attrDetectFile, "label", "detect file");
    sceneClass.setMetadata(attrDetectFile, "comment",
            "File output from the PresenZ detect phase and read during the render phase.\n");

    attrRenderFile =
        sceneClass.declareAttribute<rdl2::String>("render_file", "render.przRender", rdl2::FLAGS_FILENAME);
    sceneClass.setMetadata(attrRenderFile, "label", "render file");
    sceneClass.setMetadata(attrRenderFile, "comment",
            "File output from the PresenZ render phase.\n");

    attrRenderScale =
        sceneClass.declareAttribute<rdl2::Float>("render_scale", 1.0f);
    sceneClass.setMetadata(attrRenderScale, "label", "render scale");
    sceneClass.setMetadata(attrRenderScale, "comment",
            "Changes the percieved scale of the VR scene.\n");

    attrZOVScale =
        sceneClass.declareAttribute<rdl2::Vec3f>("zov_scale", rdl2::Vec3f(1.0f, 0.5f, 1.0f));
    sceneClass.setMetadata(attrZOVScale, "label", "zov scale");
    sceneClass.setMetadata(attrZOVScale, "comment",
            "Scale of the Zone of View box that the viewer can move around in.\n");

    attrDistanceToGround =
        sceneClass.declareAttribute<rdl2::Float>("distance_to_ground", 1.6f);
    sceneClass.setMetadata(attrDistanceToGround, "label", "distance to ground");
    sceneClass.setMetadata(attrDistanceToGround, "comment",
            "Height (in meters) of the VR view from the ground.\n");

    // Rendering
    attrDraftRendering =
        sceneClass.declareAttribute<rdl2::Bool>("draft_rendering", false);
    sceneClass.setMetadata(attrDraftRendering, "label", "draft rendering");
    sceneClass.setMetadata(attrDraftRendering, "comment",
        "This feature allows you to make a quick VR test render.\n");

    attrFroxtrumRendering =
        sceneClass.declareAttribute<rdl2::Bool>("froxtrum_rendering", false);
    sceneClass.setMetadata(attrFroxtrumRendering, "label", "froxtrum rendering");
    sceneClass.setMetadata(attrFroxtrumRendering, "comment",
        "Enables froxtrum rendering optimization for detect phase.\n");

    attrFroxtrumDepth =
        sceneClass.declareAttribute<rdl2::Int>("froxtrum_depth", 6);
    sceneClass.setMetadata(attrFroxtrumDepth, "label", "froxtrum depth");
    sceneClass.setMetadata(attrFroxtrumDepth, "comment",
        "Set the depth density of frustrum. By default, it's set to 6. Lower will increase froxtrum density, higher will lower it. Another good value is 7.\n");

    attrFroxtrumResolution =
        sceneClass.declareAttribute<rdl2::Int>("froxtrum_resolution", 8);
    sceneClass.setMetadata(attrFroxtrumResolution, "label", "froxtrum resolution");
    sceneClass.setMetadata(attrFroxtrumResolution, "comment",
        "Set the resolution of frustrum. By default, it's set to 8x8 square.\n");

    attrRenderInsideZOV =
        sceneClass.declareAttribute<rdl2::Bool>("render_inside_zov", false);
    sceneClass.setMetadata(attrRenderInsideZOV, "label", "render inside zov");
    sceneClass.setMetadata(attrRenderInsideZOV, "comment",
        "Renders objects inside the zone of view.\n");

    attrEnableDeepReflections =
        sceneClass.declareAttribute<rdl2::Bool>("enable_deep_reflections", true);
    sceneClass.setMetadata(attrEnableDeepReflections, "label", "enable deep reflections");
    sceneClass.setMetadata(attrEnableDeepReflections, "comment",
        "Rendering without this will make any reflection feel like it is baked on the surface of the reflective object.\n");

    attrInterPupillaryDistance =
        sceneClass.declareAttribute<rdl2::Float>("inter_pupillary_distance", 63.5f);
    sceneClass.setMetadata(attrInterPupillaryDistance, "label", "inter pupillary distance");
    sceneClass.setMetadata(attrInterPupillaryDistance, "comment",
        "Distance between the eyes (in millimeters).  Only affects the deep reflections\n");
    sceneClass.setMetadata(attrInterPupillaryDistance, "enable if",
        "OrderedDict([(u'deep_reflections', u'true')])");

    // Multiboxing
    attrZOVOffsetX =
        sceneClass.declareAttribute<rdl2::Int>("zov_offset_x", 0);
    sceneClass.setMetadata(attrZOVOffsetX, "label", "zov offset x");
    sceneClass.setMetadata(attrZOVOffsetX, "comment",
            "Offsets a new zone of view from the original in the x direction.\n");

    attrZOVOffsetY =
        sceneClass.declareAttribute<rdl2::Int>("zov_offset_y", 0);
    sceneClass.setMetadata(attrZOVOffsetY, "label", "zov offset y");
    sceneClass.setMetadata(attrZOVOffsetY, "comment",
            "Offsets a new zone of view from the original in the y direction.\n");

    attrZOVOffsetZ =
        sceneClass.declareAttribute<rdl2::Int>("zov_offset_z", 0);
    sceneClass.setMetadata(attrZOVOffsetZ, "label", "zov offset z");
    sceneClass.setMetadata(attrZOVOffsetZ, "comment",
            "Offsets a new zone of view from the original in the z direction.\n");

    attrSpecularPointOffset =
        sceneClass.declareAttribute<rdl2::Vec3f>("specular_point_offset", rdl2::Vec3f(0.0f, 0.0f, 0.0f));
    sceneClass.setMetadata(attrSpecularPointOffset, "label", "specular point offset");
    sceneClass.setMetadata(attrSpecularPointOffset, "comment",
            "Used to move the specular point (located between the eyes) from which reflections are rendered.\n");

    // Clipping
    attrEnableClippingSphere=
        sceneClass.declareAttribute<rdl2::Bool>("enable_clipping_sphere", false);
    sceneClass.setMetadata(attrEnableClippingSphere, "label", "enable clipping sphere");
    sceneClass.setMetadata(attrEnableClippingSphere, "comment",
        "Enables a clipping sphere.\n");

    attrClippingSphereRadius =
        sceneClass.declareAttribute<rdl2::Float>("clipping_sphere_radius", 100.0f);
    sceneClass.setMetadata(attrClippingSphereRadius, "label", "clipping sphere radius");
    sceneClass.setMetadata(attrClippingSphereRadius, "comment",
            "Radius of the clipping sphere.\n");
    sceneClass.setMetadata(attrClippingSphereRadius, "enable if",
        "OrderedDict([(u'enable_clipping_sphere', u'true')])");

    attrClippingSphereCenter =
        sceneClass.declareAttribute<rdl2::Vec3f>("clipping_sphere_center", rdl2::Vec3f(0.0f, 0.0f, 0.0f));
    sceneClass.setMetadata(attrClippingSphereCenter, "label", "clipping sphere center");
    sceneClass.setMetadata(attrClippingSphereCenter, "comment",
            "Position of the clipping sphere.\n");
    sceneClass.setMetadata(attrClippingSphereCenter, "enable if",
        "OrderedDict([(u'enable_clipping_sphere', u'true')])");

    attrClippingSphereRenderInside =
        sceneClass.declareAttribute<rdl2::Bool>("clipping_sphere_render_inside", true);
    sceneClass.setMetadata(attrClippingSphereRenderInside, "label", "clipping sphere render inside");
    sceneClass.setMetadata(attrClippingSphereRenderInside, "comment",
        "The default is render inside, only what is inside of the clipping sphere will be rendered. By setting this value to false you will render your scene excluding the part that falls within the sphere.\n");
    sceneClass.setMetadata(attrClippingSphereRenderInside, "enable if",
        "OrderedDict([(u'enable_clipping_sphere', u'true')])");

    // Groups

    // General
    sceneClass.setGroup("General", attrPresenZEnabled);
    sceneClass.setGroup("General", attrPhase);
    sceneClass.setGroup("General", attrDetectFile);
    sceneClass.setGroup("General", attrRenderFile);
    sceneClass.setGroup("General", attrRenderScale);
    sceneClass.setGroup("General", attrZOVScale);
    sceneClass.setGroup("General", attrDistanceToGround);

    // Rendering
    sceneClass.setGroup("Rendering", attrDraftRendering);
    sceneClass.setGroup("Rendering", attrRenderInsideZOV);
    sceneClass.setGroup("Rendering", attrEnableDeepReflections);
    sceneClass.setGroup("Rendering", attrInterPupillaryDistance);

    // Multiboxing
    sceneClass.setGroup("Multiboxing", attrZOVOffsetX);
    sceneClass.setGroup("Multiboxing", attrZOVOffsetY);
    sceneClass.setGroup("Multiboxing", attrZOVOffsetZ);
    sceneClass.setGroup("Multiboxing", attrSpecularPointOffset);

    // Clipping
    sceneClass.setGroup("Clipping", attrEnableClippingSphere);
    sceneClass.setGroup("Clipping", attrClippingSphereRadius);
    sceneClass.setGroup("Clipping", attrClippingSphereCenter);
    sceneClass.setGroup("Clipping", attrClippingSphereRenderInside);

RDL2_DSO_ATTR_END
