// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;

RDL2_DSO_ATTR_DECLARE

    rdl2::AttributeKey<rdl2::Float> attrMinLatitude;
    rdl2::AttributeKey<rdl2::Float> attrMaxLatitude;
    rdl2::AttributeKey<rdl2::Float> attrLatitudeZoomOffset;
    rdl2::AttributeKey<rdl2::Float> attrMinLongitude;
    rdl2::AttributeKey<rdl2::Float> attrMaxLongitude;
    rdl2::AttributeKey<rdl2::Float> attrLongitudeZoomOffset;
    rdl2::AttributeKey<rdl2::Float> attrFocalKey;
    rdl2::AttributeKey<rdl2::Bool>  attrInsideOut;
    rdl2::AttributeKey<rdl2::Float> attrOffsetRadius;

    RDL2_DSO_ATTR_DEFINE(rdl2::Camera)

    attrMinLatitude = sceneClass.declareAttribute<rdl2::Float>("min_latitude", -90.0f);
    sceneClass.setMetadata(attrMinLatitude, "label", "min latitude");
    sceneClass.setMetadata(attrMinLatitude, rdl2::SceneClass::sComment,
                           "Latitude corresponding to the bottom of the image");

    attrMaxLatitude = sceneClass.declareAttribute<rdl2::Float>("max_latitude", 90.0f);
    sceneClass.setMetadata(attrMaxLatitude, "label", "max latitude");
    sceneClass.setMetadata(attrMaxLatitude, rdl2::SceneClass::sComment,
                           "Latitude corresponding to the top of the image");

    attrLatitudeZoomOffset = sceneClass.declareAttribute<rdl2::Float>("latitude_zoom_offset", 0.0f);
    sceneClass.setMetadata(attrLatitudeZoomOffset, "label", "latitude zoom offset");
    sceneClass.setMetadata(attrLatitudeZoomOffset, rdl2::SceneClass::sComment,
                           "Attribute for controlling the latitude of the center of zoom when the focal length is "
                           "changed. By default, zooming will center around the mean of the min and max latitudes, "
                           "but this behavior can be modified by supplying a non-zero offset which will be added to "
                           "the mean and will reposition the zoom center.");

    attrMinLongitude = sceneClass.declareAttribute<rdl2::Float>("min_longitude", -180.0f);
    sceneClass.setMetadata(attrMinLongitude, "label", "min longitude");
    sceneClass.setMetadata(attrMinLongitude, rdl2::SceneClass::sComment,
                           "Longitude corresponding to the left of the image");

    attrMaxLongitude = sceneClass.declareAttribute<rdl2::Float>("max_longitude", 180.0f);
    sceneClass.setMetadata(attrMaxLongitude, "label", "max longitude");
    sceneClass.setMetadata(attrMaxLongitude, rdl2::SceneClass::sComment,
                           "Longitude corresponding to the right of the image");

    attrLongitudeZoomOffset = sceneClass.declareAttribute<rdl2::Float>("longitude_zoom_offset", 0.0f);
    sceneClass.setMetadata(attrLongitudeZoomOffset, "label", "longitude zoom offset");
    sceneClass.setMetadata(attrLongitudeZoomOffset, rdl2::SceneClass::sComment,
                           "Attribute for controlling the longitude of the center of zoom when the focal length is "
                           "changed. By default, zooming will center around the mean of the min and max longitudes, "
                           "but this behavior can be modified by supplying a non-zero offset which will be added to "
                           "the mean and will reposition the zoom center.");

    attrFocalKey = sceneClass.declareAttribute<rdl2::Float>("focal", 30.0f, rdl2::FLAGS_BLURRABLE);
    sceneClass.setMetadata(attrFocalKey, rdl2::SceneClass::sComment, "Focal length");

    attrInsideOut = sceneClass.declareAttribute<rdl2::Bool>("inside_out", false);
    sceneClass.setMetadata(attrInsideOut, "label", "inside out");
    sceneClass.setMetadata(attrInsideOut, rdl2::SceneClass::sComment,
                           "Set to true if the rendered image is to be mapped onto the outside of a sphere, "
                           "e.g. the Las Vegas Sphere.");

    attrOffsetRadius = sceneClass.declareAttribute<rdl2::Float>("offset_radius", 0.0f);
    sceneClass.setMetadata(attrOffsetRadius, "label", "offset radius");
    sceneClass.setMetadata(attrOffsetRadius, rdl2::SceneClass::sComment,
                           "If using the [\"inside_out\"] attribute, set this value to a radius large enough to "
                           "encompass all the geometry you wish to capture.");

RDL2_DSO_ATTR_END

