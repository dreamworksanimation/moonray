// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Bool>      attrNormalized;
    rdl2::AttributeKey<rdl2::Bool>      attrApplySceneScale;
    rdl2::AttributeKey<rdl2::Float>     attrLensRadius;
    rdl2::AttributeKey<rdl2::Float>     attrAspectRatio;
    rdl2::AttributeKey<rdl2::Float>     attrFocalPlaneDistance;
    rdl2::AttributeKey<rdl2::Float>     attrOuterConeAngle;
    rdl2::AttributeKey<rdl2::Float>     attrInnerConeAngle;
    rdl2::AttributeKey<rdl2::Int>       attrAngleFalloffType;
    rdl2::AttributeKey<rdl2::Float>     attrBlackLevel;
    DECLARE_ATTR_KEYS_CLEAR_RADIUS

RDL2_DSO_ATTR_DEFINE(rdl2::Light)

    attrNormalized = sceneClass.declareAttribute<rdl2::Bool >("normalized", true);
    sceneClass.setMetadata(attrNormalized, rdl2::SceneClass::sComment,
        "When set to true, the size of the light can be changed without altering the amount of total energy "
        "cast into the scene. This is achieved via scaling the light's radiance by the reciprocal of its "
        "surface area. When set to false, the radiance is used as-is, regardless of surface area.");

    attrApplySceneScale = sceneClass.declareAttribute<rdl2::Bool>("apply_scene_scale", true);
    sceneClass.setMetadata(attrApplySceneScale, "comment", "apply scene scale variable when normalized");
    sceneClass.setMetadata(attrApplySceneScale, rdl2::SceneClass::sComment,
        "Whether to apply scene scale variable when normalized.");

    attrLensRadius = sceneClass.declareAttribute<rdl2::Float>("lens_radius", 1.0f, { "lens radius" });
    sceneClass.setMetadata(attrLensRadius, "label", "lens radius");
    sceneClass.setMetadata(attrLensRadius, rdl2::SceneClass::sComment,
        "The radius of the SpotLight's lens (when the aspect ratio is 1.0, so that the lens is circular).");

    attrAspectRatio = sceneClass.declareAttribute<rdl2::Float>("aspect_ratio", 1.0f, { "aspect ratio" });
    sceneClass.setMetadata(attrAspectRatio, "label", "aspect ratio");
    sceneClass.setMetadata(attrAspectRatio, rdl2::SceneClass::sComment,
        "The aspect ratio of the lens - its local y dimension divided by its local x dimension. "
        "Values other than 1.0 will give the lens a non-circular elliptical shape.");

    attrFocalPlaneDistance = sceneClass.declareAttribute<rdl2::Float>("focal_plane_distance", 1.0e10f, { "focal plane distance" });
    sceneClass.setMetadata(attrFocalPlaneDistance, "label", "focal plane distance");
    sceneClass.setMetadata(attrFocalPlaneDistance, rdl2::SceneClass::sComment,
        "The distance from the SpotLight's position, measured in the direction the light is pointing, at which "
        "the projected image will be in focus.");

    attrOuterConeAngle = sceneClass.declareAttribute<rdl2::Float>("outer_cone_angle", 60.0f, { "outer cone angle" });
    sceneClass.setMetadata(attrOuterConeAngle, "label", "outer cone angle");
    sceneClass.setMetadata(attrOuterConeAngle, rdl2::SceneClass::sComment,
        "The apex angle of the bounding cone of the light emitted by the SpotLight. No illumination takes place"
        "outside this angle. This is a full angle, measured from one side to the other. "
        "There is a falloff function applied between the outer and inner cones - "
        "see the angle_falloff_type attribute.");

    attrInnerConeAngle = sceneClass.declareAttribute<rdl2::Float>("inner_cone_angle", 30.0f, { "inner cone angle" });
    sceneClass.setMetadata(attrInnerConeAngle, "label", "inner cone angle");
    sceneClass.setMetadata(attrInnerConeAngle, rdl2::SceneClass::sComment,
        "The apex angle of the bright inner cone of the light emitted by the SpotLight. Full illumination takes "
        "place inside this region. This is a full angle, measured from one side to the other. "
        "There is a falloff function applied between the outer and inner cones - "
        "see the angle_falloff_type attribute.");

    attrAngleFalloffType = sceneClass.declareAttribute<rdl2::Int  >("angle_falloff_type", 4, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "angle falloff type" });
    sceneClass.setMetadata(attrAngleFalloffType, "label", "angle falloff type");
    sceneClass.setEnumValue(attrAngleFalloffType, 0, "off");
    sceneClass.setEnumValue(attrAngleFalloffType, 1, "linear");
    sceneClass.setEnumValue(attrAngleFalloffType, 2, "ease in");
    sceneClass.setEnumValue(attrAngleFalloffType, 3, "ease out");
    sceneClass.setEnumValue(attrAngleFalloffType, 4, "ease in/out");
    sceneClass.setMetadata(attrAngleFalloffType, rdl2::SceneClass::sComment,
        "The falloff function applied between the outer and inner cones. To calculate this, the angle from "
        "the cone's axis to the the point being illuminated is measured as seen from the SpotLight's position. "
        "This angle is converted into a fractional value representing the fraction from the outer cone angle "
        "to the inner cone angle, clamped to the range [0,1]. The resulting value is then fed into one of the "
        "following user-selectable functions to determine the final 0-1 scaling value to be applied to the"
        "light's radiance: \n"
        "  0 (off)         - no fallof, a step function at the outer cone boundary is applied\n"
        "  1 (linear)      - a linear ramp, i.e. the fractional parameter is applied as-is\n"
        "  2 (ease in)     - a quadratic ramp with zero gradient at the start point (outer cone)\n"
        "  3 (ease out)    - a quadratic ramp with zero gradient at the end point (inner cone)\n"
        "  4 (ease in/out) - a cubic ramp with zero gradient at both ends (outer and inner cone)\n");

    attrBlackLevel         = sceneClass.declareAttribute<rdl2::Float>("black_level", 0.001f, { "black level" });
    sceneClass.setMetadata(attrBlackLevel, "label", "black level");
    sceneClass.setMetadata(attrBlackLevel, rdl2::SceneClass::sComment,
        "The radiance used for rendering the SpotLight lens as seen through the camera via a primary ray, when "
        "the true computed radiance would otherwise be black. This is simply a convenience feature to make the "
        "SpotLight lens visible in the camera view.");

    DECLARE_ATTRS_CLEAR_RADIUS

    // Group the attributes for Torch's attribute editor
    // DO NOT edit the order the attributes are defined
    // we want them in that order
    sceneClass.setGroup("Properties", attrNormalized);
    sceneClass.setGroup("Properties", attrApplySceneScale);

    sceneClass.setGroup("Cone", attrLensRadius);
    sceneClass.setGroup("Cone", attrAspectRatio);
    sceneClass.setGroup("Cone", attrFocalPlaneDistance);
    sceneClass.setGroup("Cone", attrOuterConeAngle);
    sceneClass.setGroup("Cone", attrInnerConeAngle);

    sceneClass.setGroup("Falloff", attrAngleFalloffType);
    sceneClass.setGroup("Falloff", attrBlackLevel);

    SET_ATTR_GRP_CLEAR_RADIUS
    
    
RDL2_DSO_ATTR_END

