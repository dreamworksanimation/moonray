RdlMeshGeometry Procedural
==========================

RdlMeshGeometry is a procedural that pretends to be a primitive in order to 
allow Polymesh and Subd geometry to be created and stored directly in the 
RDL2 scene.

Attributes
----------

The procedural has the following attributes:

- **vertex list** (Vec3fVector): Stores all vertices used by the mesh
- **vertex list mb** (Vec3fVector - optional): If the mesh is in motion, the second motion step is stored in this attribute (necessary because Vec3fVector attributes are not blurrable)
- **vertices by index** (IntVector): Ordered list of vertex indices used to construct the mesh using the vertex list
- **vertex face count** (IntVector): Ordered list of vertices per face, used in conjection with vertices by index to construct the mesh
- **use velocity** (Bool): If true, calculate motion blur using velocities from "velocity list" instead of using a second motion step from "vertex list mb"
- **velocity list** (Vec3fVector - optional): Optionally provide a per-vertex velocity (in units/sec)
- **velocity scale** (Float - optional): Scale the magnitude of the velocity (when using velocity)
- **uv list** (Vec2fVector - optional): If the mesh is using UVs, store them per-face-vertex in this list
- **normal list** (Vec3fVector - optional): If the mesh is using normals, store them per-face-vertex in this list
- **is subd** (Bool): If true, a SubdivisionMesh primitive will be created - PolygonMesh otherwise
- **subd resolution** (Int): Resolution to use for subdividing the subdivision mesh
- **subd scheme** (Enum): CatClark or Bilinear
- **subd boundary** (Enum): Choice of boundary interpolation options
- **subd fvar linear** (Enum): Choice of face-varying linear interpolation options
- **subd crease indices** (IntVector): Pairs of vertex indices corresponding to edges with assigned sharpness
- **subd crease sharpnesses** (FloatVector): Sharpness values for crease edges
- **subd corner indices** (IntVector): Indices corresponding to vertices with assigned sharpness
- **subd corner sharpnesses** (FloatVector): Sharpness values for corner vertices

Usage
-----

### Triangle

![triangle](http://mydw.anim.dreamworks.com/download/attachments/375912387/triangle.jpg)

A simple triangle looks like this:

```
RdlMeshGeometry("triangle") {
    ["vertex list"] = { Vec3(-1, 0, 0), Vec3(0, 2, 0), Vec3(1, 0, 0)},
    ["vertices by index"] = {0, 1, 2},
    ["face vertex count"] = {3},
    ["is subd"] = false,
}
```

All RdlMeshGeometry only has a single part called "mesh_part". Layer objects can be 
constructed using this part:

```
Layer("rndr") {
    {RdlMeshGeometry("triangle"), "mesh_part", BaseMaterial("base"), LightSet("all")},
}
```

### Triangle in Motion

![triangle_mb](http://mydw.anim.dreamworks.com/download/attachments/375912387/triangle_mb.jpg)

Motion blur can be created using the **vertex list mb** attribute:

```
RdlMeshGeometry("triangle") {
    ["vertex list"] = { Vec3(-1, 0, 0), Vec3(0, 2, 0), Vec3(1, 0, 0)},
    ["vertex list mb"] = { Vec3(-0.5, 0, 0), Vec3(0.5, 2, 0), Vec3(1.5, 0, 0)},
    ["vertices by index"] = {0, 1, 2},
    ["face vertex count"] = {3},
    ["is subd"] = false,
    ["static"] = false,
}

```

Motion blur can also be created by providing velocity data using the **velocity list** attribute:

```
RdlMeshGeometry("triangle") {
    ["vertex list"] = { Vec3(-1, 0, 0), Vec3(0, 2, 0), Vec3(1, 0, 0)},
    ["velocity list"] = { Vec3(12, 0, 0), Vec3(12, 0, 0), Vec3(12, 0, 0)},
    ["vertices by index"] = {0, 1, 2},
    ["face vertex count"] = {3},
    ["use velocity"] = true,
    ["is subd"] = false,
    ["static"] = false,
}
```

### Square

![square](http://mydw.anim.dreamworks.com/download/attachments/375912387/square.jpg)

More complex meshes can be created by adding additional vertices to the vertex list, and
extending the **vertices by index** and **face vertex count** lists to describe the mesh 
using the additional vertices. Here I've built a square out of three triangles:

```
RdlMeshGeometry("square") {
    ["vertex list"] = { Vec3(-1, 0, 0), Vec3(0, 2, 0), Vec3(1, 0, 0), Vec3(1, 2, 0), Vec3(-1, 2, 0)},
    ["vertices by index"] = {0, 1, 2, 1, 2, 3, 4, 1, 0},
    ["face vertex count"] = {3, 3, 3},
    ["is subd"] = false,
}
```

Examples
--------

In addition to the usage samples above, there's several more example rdla files here:

`/work/rd/raas/rdlmesh`

Even complex meshes can be represented using RdlMeshGeometry.

![yeti](http://mydw.anim.dreamworks.com/download/attachments/375912387/yeti.jpg)

![baby](http://mydw.anim.dreamworks.com/download/attachments/375912387/baby.jpg)

![wolfcrab](http://mydw.anim.dreamworks.com/download/attachments/375912387/wolfcrab.jpg)

 
