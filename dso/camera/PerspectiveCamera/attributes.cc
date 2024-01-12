// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Float> attrFocalKey;

    rdl2::AttributeKey<rdl2::Int>   attrStereoView;
    rdl2::AttributeKey<rdl2::Float> attrStereoInterocularDistance;
    rdl2::AttributeKey<rdl2::Float> attrStereoConvergenceDistance;

    rdl2::AttributeKey<rdl2::Float> attrHorizontalFilmOffset;
    rdl2::AttributeKey<rdl2::Float> attrVerticalFilmOffset;
    rdl2::AttributeKey<rdl2::Float> attrFilmWidthApertureKey;
    rdl2::AttributeKey<rdl2::Float> attrPixelAspectRatio;

    rdl2::AttributeKey<rdl2::Bool>  attrDofKey;
    rdl2::AttributeKey<rdl2::Float> attrDofApertureKey;
    rdl2::AttributeKey<rdl2::Float> attrDofFocusDistance;

    rdl2::AttributeKey<rdl2::Bool>   attrBokehKey;

    // Shape
    rdl2::AttributeKey<rdl2::Int>    attrBokehSides;
    rdl2::AttributeKey<rdl2::String> attrBokehImage;

    // Control
    rdl2::AttributeKey<rdl2::Float>  attrBokehAngle;
    rdl2::AttributeKey<rdl2::Float>  attrBokehWeightLocation;
    rdl2::AttributeKey<rdl2::Float>  attrBokehWeightStrength;

RDL2_DSO_ATTR_DEFINE(rdl2::Camera)

    attrFocalKey = sceneClass.declareAttribute<rdl2::Float>("focal", 30.0f, rdl2::FLAGS_BLURRABLE);
    sceneClass.setMetadata(attrFocalKey, rdl2::SceneClass::sComment, "Focal length");

    attrStereoView = sceneClass.declareAttribute<rdl2::Int>("stereo_view", 0, rdl2::FLAGS_ENUMERABLE, rdl2::INTERFACE_GENERIC, { "stereo view" });
    sceneClass.setMetadata(attrStereoView, "label", "stereo view");
    sceneClass.setEnumValue(attrStereoView, 0, "center view");
    sceneClass.setEnumValue(attrStereoView, 1, "left view");
    sceneClass.setEnumValue(attrStereoView, 2, "right view");
    sceneClass.setMetadata(attrStereoView, rdl2::SceneClass::sComment, "Render from the center, left, or right stereo view.");
    attrStereoInterocularDistance = sceneClass.declareAttribute<rdl2::Float>("stereo_interocular_distance", 6.3f, { "stereo interocular distance" });
    sceneClass.setMetadata(attrStereoInterocularDistance, "label", "stereo interocular distance");
    sceneClass.setMetadata(attrStereoInterocularDistance, rdl2::SceneClass::sComment, "Distance between the left and right "
                                                                                "'eyes'");
    attrStereoConvergenceDistance = sceneClass.declareAttribute<rdl2::Float>("stereo_convergence_distance", 100.0f, { "stereo convergence distance" });
    sceneClass.setMetadata(attrStereoConvergenceDistance, "label", "stereo convergence distance");
    sceneClass.setMetadata(attrStereoConvergenceDistance, rdl2::SceneClass::sComment, "Distance at which all the stereo "
                                                                                "views converge.");

    attrHorizontalFilmOffset = sceneClass.declareAttribute<rdl2::Float>("horizontal_film_offset", 0.0f, { "horizontal film offset" });
    sceneClass.setMetadata(attrHorizontalFilmOffset, "label", "horizontal film offset");
    sceneClass.setMetadata(attrHorizontalFilmOffset, rdl2::SceneClass::sComment, "Horizontal offset of the frustum.");
    attrVerticalFilmOffset = sceneClass.declareAttribute<rdl2::Float>("vertical_film_offset", 0.0f, { "vertical film offset" });
    sceneClass.setMetadata(attrVerticalFilmOffset, "label", "vertical film offset");
    sceneClass.setMetadata(attrVerticalFilmOffset, rdl2::SceneClass::sComment, "Vertical offset of the frustum.");
    attrFilmWidthApertureKey = sceneClass.declareAttribute<rdl2::Float>("film_width_aperture", 24.0f, { "film width aperture" });
    sceneClass.setMetadata(attrFilmWidthApertureKey, "label", "film width aperture");
    sceneClass.setMetadata(attrFilmWidthApertureKey, rdl2::SceneClass::sComment, "Scale the aperture of the camera "
                                                                                 "(i.e., the frustum) by this value.");
    attrPixelAspectRatio = sceneClass.declareAttribute<rdl2::Float>("pixel_aspect_ratio", 1.0, { "pixel aspect ratio" });
    sceneClass.setMetadata(attrPixelAspectRatio, "label", "pixel aspect ratio");
    sceneClass.setMetadata(attrPixelAspectRatio, rdl2::SceneClass::sComment, "ratio of pixel size y / x");

    attrDofKey = sceneClass.declareAttribute<rdl2::Bool>("dof", false);
    sceneClass.setMetadata(attrDofKey, rdl2::SceneClass::sComment, "Whether to enable depth of field");
    attrDofApertureKey = sceneClass.declareAttribute<rdl2::Float>("dof_aperture", 8.0f, { "dof aperture" });
    sceneClass.setMetadata(attrDofApertureKey, "label", "dof aperture");
    sceneClass.setMetadata(attrDofApertureKey, rdl2::SceneClass::sComment, "Depth of field aperture width");
    attrDofFocusDistance = sceneClass.declareAttribute<rdl2::Float>("dof_focus_distance", 0.0f, { "dof focus distance" });
    sceneClass.setMetadata(attrDofFocusDistance, "label", "dof focus distance");
    sceneClass.setMetadata(attrDofApertureKey, rdl2::SceneClass::sComment, "Depth of field focus distance");

    attrBokehKey = sceneClass.declareAttribute<rdl2::Bool>("bokeh", false);
    sceneClass.setMetadata(attrBokehKey, "label", "bokeh enable");
    sceneClass.setMetadata(attrBokehKey, rdl2::SceneClass::sComment, "Enable Bokeh. Requires DOF to be enabled.");

    // Shape
    attrBokehSides = sceneClass.declareAttribute<rdl2::Int>("bokeh_sides", 0, { "bokeh sides" });
    sceneClass.setMetadata(attrBokehSides, "label", "bokeh sides");
    sceneClass.setMetadata(attrBokehSides, 
                           rdl2::SceneClass::sComment,
                           "Number of sides of the iris. Specifying less than 3 sides will default to a disk.");

    attrBokehImage = sceneClass.declareAttribute<rdl2::String>("bokeh_image", "", { "bokeh image" });
    sceneClass.setMetadata(attrBokehImage, "label", "bokeh image");
    sceneClass.setMetadata(attrBokehImage, rdl2::SceneClass::sComment, "Path to image file to be used for the iris");

    // Control
    attrBokehAngle = sceneClass.declareAttribute<rdl2::Float>("bokeh_angle", 0.0f, { "bokeh angle" });
    sceneClass.setMetadata(attrBokehAngle, "label", "bokeh angle");
    sceneClass.setMetadata(attrBokehAngle, rdl2::SceneClass::sComment, "Angle of iris rotation");

    attrBokehWeightLocation = sceneClass.declareAttribute<rdl2::Float>("bokeh_weight_location",
                                                                  0.0f,
                                                                  { "bokeh weight location" });
    sceneClass.setMetadata(attrBokehWeightLocation, "label", "bokeh weight location");
    sceneClass.setMetadata(attrBokehWeightLocation,
                           rdl2::SceneClass::sComment,
                           "Distance from the origin of Bokeh shape");

    attrBokehWeightStrength = sceneClass.declareAttribute<rdl2::Float>("bokeh_weight_strength",
                                                                        0.0f,
                                                                        { "bokeh weight strength" });
    sceneClass.setMetadata(attrBokehWeightStrength, "label", "bokeh weight location");
    sceneClass.setMetadata(attrBokehWeightStrength,
                           rdl2::SceneClass::sComment,
                           "Controls the strength of weights as samples approach the weight location");


    // Grouping the attributes for Torch - the order of
    // the attributes should be the same as how they are defined.
    sceneClass.setGroup("Frustum", attrFocalKey);

    sceneClass.setGroup("Stereo", attrStereoView);
    sceneClass.setGroup("Stereo", attrStereoInterocularDistance);
    sceneClass.setGroup("Stereo", attrStereoConvergenceDistance);

    sceneClass.setGroup("Frustum", attrFilmWidthApertureKey);
    sceneClass.setGroup("Frustum", attrHorizontalFilmOffset);
    sceneClass.setGroup("Frustum", attrVerticalFilmOffset);
    sceneClass.setGroup("Frustum", attrPixelAspectRatio);

    sceneClass.setGroup("Depth of Field", attrDofKey);
    sceneClass.setGroup("Depth of Field", attrDofApertureKey);
    sceneClass.setGroup("Depth of Field", attrDofFocusDistance);

    sceneClass.setGroup("Depth of Field", attrBokehKey);
    sceneClass.setGroup("Depth of Field", attrBokehSides);
    sceneClass.setGroup("Depth of Field", attrBokehImage);

    sceneClass.setGroup("Depth of Field", attrBokehAngle);
    sceneClass.setGroup("Depth of Field", attrBokehWeightLocation);
    sceneClass.setGroup("Depth of Field", attrBokehWeightStrength);

RDL2_DSO_ATTR_END

