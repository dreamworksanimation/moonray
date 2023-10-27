---
title: PresenZ
---

# PresenZ

[PresenZ]({{ "https://www.presenzvr.com" | absolute_url }}) is a third party technology that allows
for rendering of full fidelity scenes for playback in virtual reality.

The [presenz]({{"https://github.com/dreamworksanimation/openmoonray/tree/release/moonray" | absolute_url}})
branch of moonray integrates PresenZ using the SDK which can be found here:
[PresenZ SDK]({{ "https://github.com/Parallaxter-team/PresenZ-SDK" | absolute_url }})

## Components
The main Moonray component used to render PresenZ frames is the new *PresenZCamera*.
Other impacted nodes are *SceneVariables*, *RenderOutput*, and *UserData*.

## General Workflow
This section describes the steps required to render and playback a PresenZ frame.

### Scene Variables
The resolution of the frame is set with the image width and height.  The aspect ratio should be approximately 3:2.
If the set resolution is invalid, the detect phase will throw and error and recommend a valid resolution.  The resolution
will also be a factor during playback depending on the hardware used.   Motion blur should be enabled if the scene
contains animation.   This, combined with the motion vector output described below will result in smoothly
interpolated playback.
```lua
SceneVariables {
    ["image width"] = 2232,
    ["image height"] = 1488,
    ["enable_motion_blur"] = true,
}
```

### Camera 
The `PresenZCamera` is used for several aspects of the workflow.
* Sets the position and extent of the ZOV (zone of view) box.
* Sets the render scale and distance to ground which control how the user percieves the vr world
    * Note that these setting can be modified in the final prztoc file to are not permant in the render
* Sets the rendering phase and output file locations.
* Optionally turns on draft rendering for fast preview
* Optionally defines a clipping sphere

```lua
PresenZCamera("shot_cam_pz") {
    ["node xform"] = translate(-9435, 1050, 1420),
    ["zov_scale"] = Vec3(1.0, 0.5, 1.0),

    ["render_scale"] = 1.0,
    ["distance_to_ground"] = 1.5,

    ["phase"] = "detect",
    ["detect_file"] = "./scene." .. frame .. ".przDetect",
    ["render_file"] = "./scene." .. frame .. ".przRender",

    ["draft_rendering"] = false,

    ["enable_clipping_sphere"] = false,
    ["clipping_sphere_radius"] = 100,
    ["clipping_sphere_center"] = Vec3(-9430, 1000, 1350),
    ["clipping_sphere_render_inside"] = true,
}
```

### Render Output
When rendering a sequence of frames for animated playback, smooth interpolation can be used.  This
means that even if a sequence is rendered at 24 frames per second, when playing it back, the frames
will be interpolated continuously depending on the hardware used.  To enable this feature, motion blur
must be enabled in the `SceneVariables` as described above.  Also, a `RenderOutput` with the following
settings should be added to the scene.  
```lua
RenderOutput("/output/motion_vectors") {
   ["result"] = "material aov",
   ["material_aov"] = "motionvec",
   ["channel_suffix_mode"] = "rgb"
}
```

### Rendering
The rendering of a PresenZ frame is divided in two between a *detect* phase and a *render* phase.
Next, for each frame,  render the detect phase, the render phase, and merge the resulting przDetect
and przRender files into a playable .prz file.

Run the “detect phase” with the following command:
```bash
moonray -exec_mode scalar -in scene.rdla -attr_set "shot_cam_pz" "phase" 0
```

Run the “render phase” with the following command:
```bash
moonray -exec_mode scalar -in scene.rdla -attr_set "shot_cam_pz" "phase" 1
```


### Merging
Merge the resulting detect and render files into a .prz file, which is playable with the viewer on Windows, with the following command:
```bash
presenz_merger . -x --output test.####.prz
```

### Playback
To play the resulting prz file(s), drop the first one onto the player’s icon in Windows.
This will load the sequence and generate a przToc file.  This file can be edited in notepad.
You can add a background image by adding this line to the “scenes” section typically after the “framerate” line.

```
		"backgroundFile" : "sq2601_sky_paint_fix3.jpg",
```

You will also most likely want to set the framerate to 24.0:

```
		"framerate": 24.00000,
```

You may want to change the starting height of the viewer (default is 1.6 and can be changed during
playback with the ‘t’ key):

```
		"cameraHeight": 1.5,
```

## Scene Optimization
The PresenZ player has a limit of 128MB per frame.   To help reduce the size of complex scenes, fur and other complex
geometry can be marked as chaotic.  To do this, add a UserData object to the rdla file:

```
chaotic = UserData("chaotic") {
   ["bool_key"] = "chaotic",
   ["bool_values"] = { true },
}
```

Next, add this to the chaotic objects’ primitive_attributes:

```
RdlCurveGeometry("curves") {
    ...
    ["primitive_attributes"] = {chaotic},
    ...
}
```

Finally turn on “froxtrum_rendering” on the PresenZCamera:

```
PresenZCamera("shot_cam_pz") {
    ["froxtrum_rendering"] = true,
    ["froxtrum_depth"] = 6,
    ["froxtrum_resolution"] = 8,
}
```

