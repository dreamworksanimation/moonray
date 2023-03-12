Navigation in moonray_gui
===

moonray_gui will pop up a graphical window which will allow you to navigate scene in real-time.
There are two supported modes of navigation, which are described below.
The default mode of navigation is orbit, but that may be overridden on the command line
by specifying -freecam, or by using the O key to toggle between modes at run-time.

Controls common to both modes
---

| Action          | Result                                           |
|-----------------|--------------------------------------------------|
| Alt + LMB + RMB | Roll                                             |
| R               | Reset camera to original start-up world location |
| T               | Print current camera matrix to console           |
| O               | Toggle between free camera and orbit camera      |
| U               | Upright camera (remove roll)                     |
| P               | Toggle bucket progress on/off                    |
| W               | Translate forward                                |
| S               | Translate backward                               |
| A               | Translate left                                   |
| D               | Translate right                                  |
| Space           | Translate upward                                 |
| C               | Translate downward                               |
| Q               | Slow down movement                               |
| E               | Speed up movement                                |
| `               | Display RGB channels                             |
| 1               | Display red channel                              |
| 2               | Display green channel                            |
| 3               | Display blue channel                             |
| 4               | Display alpha channel                            |
| 5               | Display luminance                                |
| 6               | Display saturation (not implemented yet)         |
| 7               | Display normalized RGB channels (0-1)            |
| <               | Display previous render output                   |
| >               | Display next render output                       |

Orbit mode
---

This mode is modeled on the current camera behavior in Torch.
The mouse does nothing by default unless you hold down the ALT key.

| Action     | Result                                              |
|------------|-----------------------------------------------------|
| Alt + LMB  | Orbit around current pivot point                    |
| Alt + MMB  | Pan                                                 |
| Alt + RMB  | Zoom (dolly)                                        |
| Ctrl + LMB | Refocus orbit pivot to point under the mouse cursor |

Freecam mode
---
This mode is modeled around the WASD scheme common in first person shooters.
The mouse control differs from the orbit mode in that LMB rotates the camera around the current camera position,
rather than that of some world space location.
The result is that it feels like how you would look around in a typical FPS game.
Used in conjunction with the translation keys, this allow you to "fly" around the scene.

Command line options for moonray/moonray_gui
===
you can use -h flag to display the command line options
```
$> moonray_gui -h
$> moonray -h
```

```
    -in scene.rdl{a|b}
        Input RDL scene data. May appear more than once. Processes multiple
        files in order.

    -deltas file.rdl{a|b}
        Updates to apply to RDL scene data. May appear more than once.
        First renders without deltas and outputs the image. Then applies each
        delta file in order, outputting an image between each one.

    -out scene.exr
        Output image name and type.

    -threads n
        Number of threads to use (all by default).

    -size 1920 1080
        Canonical frame width and height (in pixels).

    -res 1.0
        Resolution divisor for frame dimensions.

    -sub_viewport l b r t
    -sub_vp       l b r t
        Clamp viewport render region.

    -debug_pixel x y
        Only render this one pixel for debugging. Overrides viewport.

    -dso_path dso/path
        Prepend to search path for RDL DSOs.

    -texturesystem texsys
        Choose a specific texture system. Valid options are
        sony or dwaproduction. Default is dwaproduction.

    -camera camera
        Camera to render from.

    -layer layer
        Layer to render from.

    -fast_geometry_update
        Turn on supporting fast geometry update for animation.

    -record_rays .raydb/.mm
        Save ray database or mm for later debugging.

    -primary_range 0 [0]
        Start and end range of primary ray(s) to debug. Only active with
        -record_rays.

    -depth_range 0 [0]
        Start and end range of ray depths to debug. Only active with
        -record_rays.

    -rdla_set "var name" "expression"
        Sets a global variable in the Lua interpreter before any RDLA is
        executed.

    -scene_var "name" "value"
        Override a specific scene variable.

    -attr_set "object" "attribute name" "value"
        Override the value of an attribute on a specific SceneObject.

    -attr_bind "object" "attribute name" "bound object name"
        Override the binding on an attribute of a specific SceneObject.

    -info
        enable verbose progress and statistics logging on stdout.

    -[not]bundled
        Enable [disable] bundled rendering (disabled by default).

    -[not]vectorized
        Enable [disable] vectorized rendering (disabled by default).
        Note: bundled mode must be enabled to allow vectorized mode to work.

    -ray_streaming
        Use the ray streaming approach to ray intersections (only applies to bundled mode).

    -stats filename.csv
        enable logging of statistics to a formatted file.

    -pam_config ("path to config file"|"json text") "config name" "config site"
        Sets the config file (or explicit json text) and name to use when
        Ex:  -pam_config /studio/igo/mstr/general/config/show_services_prod.config igo RWC

    -pam_sandbox "sandbox URI"
        Sets the sandbox to use when reading PAM project files. If omitted, no sandbox is used

```
