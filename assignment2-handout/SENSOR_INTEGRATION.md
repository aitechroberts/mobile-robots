# Sensor Simulator -- How It Works

This document explains the three core functions implemented in the depth camera
sensor simulator and the math behind each one.

## The Big Picture

The sensor simulator answers a simple question: **"If I put a depth camera at
this pose in a 3D environment, what would it see?"**

The pipeline has three stages that mirror how a real depth camera works, just in
reverse-engineered simulation form:

1. **Ray-Triangle Intersection** -- Cast rays from the camera into the mesh
   world and find where they hit surfaces.
2. **World-to-Camera Transform** -- Take those 3D hit points (in world
   coordinates) and re-express them from the camera's own point of view.
3. **Image Plane Projection** -- Flatten the camera-frame 3D points down into a
   2D depth image, the same format a real depth sensor would output.

---

## 1. `ray_triangle_intersect` -- Moller-Trumbore Algorithm

### What it does

Given a single ray (origin + direction) and a single triangle (three vertices),
determine whether the ray hits the triangle and, if so, how far along the ray
the hit occurs.

### Why this algorithm

The environment mesh is made entirely of triangles (loaded from a PLY file), so
we need a fast ray-triangle test. The Moller-Trumbore algorithm is the standard
choice -- it avoids computing the plane equation explicitly and instead solves
for the intersection directly using the barycentric coordinate system of the
triangle.

### The math

A point on the ray is parameterized as:

\
P(t) = O + t * D
\

where O is the ray origin, D is the ray direction, and t is the scalar distance.

A point inside the triangle defined by vertices v0, v1, v2 can be written using
barycentric coordinates (u, v):

\
P = (1 - u - v) * v0 + u * v1 + v * v2
\

Setting these equal gives us three unknowns (t, u, v) and three equations (one
per x/y/z component). The algorithm solves this system using Cramer's rule with
cross products, which avoids explicitly forming a matrix.

The steps:

1. Compute the two edge vectors from v0:
   - `edge1 = v1 - v0`
   - `edge2 = v2 - v0`

2. Compute `h = D x edge2` (cross product). This is a vector perpendicular to
   both the ray direction and the second edge.

3. Compute `a = edge1 . h` (dot product). This is the determinant of the
   implicit 3x3 system. If a is near zero, the ray is parallel to the triangle
   plane and there is no intersection.

4. Compute `f = 1 / a`, then `s = O - v0`.

5. First barycentric coordinate: `u = f * (s . h)`. If u falls outside [0, 1],
   the intersection point is outside the triangle.

6. Compute `q = s x edge1`.

7. Second barycentric coordinate: `v = f * (D . q)`. If v < 0 or u + v > 1,
   the point is again outside the triangle.

8. Finally, the ray parameter: `t = f * (edge2 . q)`. If t > 0 (hit is in
   front of the camera, not behind it), return t. Otherwise return None.

### How it fits into the pipeline

The caller (`ray_mesh_intersect`) loops over every pre-computed camera ray and
every triangle in the mesh, calling this function each time. It keeps track of
the nearest hit (smallest t) for each ray. If the nearest hit is within
`range_max`, the corresponding world-frame point is recorded; otherwise the
point defaults to `range_max` along the ray direction (representing a miss /
max-range reading).

---

## 2. `transform_to_camera_frame` -- Rigid Body Transform

### What it does

Takes the 3D world-frame hit points from stage 1 and re-expresses them in the
camera's own coordinate frame. This is what a real sensor would actually
"perceive" -- distances relative to itself, not to some global origin.

### The math

We have two transforms available:

- **Twb** -- the body (quadrotor) pose in the world frame.
- **Tbc** -- the camera pose in the body frame (a fixed offset loaded from the
  YAML config: translation + rotation representing how the camera is mounted on
  the robot).

Composing them gives the camera pose in the world frame:

\
Twc = Twb * Tbc
\

This is a 4x4 homogeneous transformation matrix that says "here is where the
camera is, and which way it's pointing, in world coordinates."

To go the other direction -- from world coordinates into the camera's frame --
we need the inverse:

\
Tcw = Twc^(-1)
\

For a homogeneous transform with rotation R and translation t, the inverse is:

\
R_inv = R^T
t_inv = -R^T * t
\

This avoids a general matrix inverse and exploits the orthogonality of rotation
matrices.

Once we have Tcw, transforming all N world-frame points is a single vectorized
operation:

\
camera_points = Rcw * world_points^T + tcw
\

where Rcw is the 3x3 rotation block and tcw is the 3x1 translation of Tcw.

### Intuition

Think of it this way: the world-frame points are like GPS coordinates on a map.
The camera transform says "I'm standing at position tcw on the map, and I'm
rotated by Rcw relative to the map axes." To figure out where each point is
*relative to me*, I subtract my position and undo my rotation. That's exactly
what the inverse transform does.

---

## 3. `project_to_image_plane` -- Pinhole Camera Projection

### What it does

Takes the 3D camera-frame points and produces a 2D depth image -- a grid of
pixels where each pixel's value is the depth (distance along the camera's
Z-axis) of the surface that the corresponding ray hit.

### The math

This uses the standard pinhole camera model (Lecture 03, Slide 32). Given a 3D
point (X, Y, Z) in the camera frame, the projection to integer pixel
coordinates (u, v) is:

\
u = int( (X * fx) / Z + cx + 0.5 )
v = int( (Y * fy) / Z + cy + 0.5 )
\

The `+ 0.5` inside the `int()` truncation is the standard rounding idiom: it
shifts the value so that truncation toward zero acts as round-half-up.

where:
- **fx, fy** are the focal lengths in pixels (computed from the field of view
  and image dimensions at init time).
- **cx, cy** are the principal point, i.e., the pixel coordinates of the
  optical center (the middle of the image).

The depth value written to the image at pixel (u, v) is **Z / to_meters**,
converting from metric depth into the sensor's native units (e.g., millimeters
when to_meters = 0.001). This matches what a real depth camera would produce:
integer-scale depth values that you multiply by `to_meters` to recover meters.

### Where the intrinsics come from

During initialization, the simulator computes the focal lengths from the
configured field of view:

\
fx = cx / tan(fov_x / 2)
fy = cy / tan(fov_y / 2)
\

with cx = size_x / 2 and cy = size_y / 2. This is the standard relationship
between field of view and focal length for a symmetric pinhole camera.

### Consistency with ray generation

The ray generation in `get_normalized_depth_points` does exactly the inverse
of this projection. For each pixel (x, y), it computes the normalized ray
direction:

\
d = normalize( [(x - cx)/fx, (y - cy)/fy, 1] )
\

So projecting a camera-frame point back with `u = fx * X/Z + cx` recovers the
original pixel coordinate. The two operations are inverses of each other by
construction.

### Edge cases handled

- **Z <= 0**: The point is behind the camera. Skip it.
- **Pixel out of bounds**: After projection, if u or v falls outside the image
  dimensions, the point is outside the field of view. Skip it.

---

## Data Flow Summary

```
PLY Mesh (triangles)
        |
        v
[ray_triangle_intersect]   <-- for each ray x each triangle
        |
        v
World-Frame 3D Points (Nx3)
        |
        v
[transform_to_camera_frame] <-- Tcw = (Twb * Tbc)^(-1)
        |
        v
Camera-Frame 3D Points (Nx3)
        |
        v
[project_to_image_plane]    <-- pinhole model: fx, fy, cx, cy
        |
        v
Depth Image (size_y x size_x)
```

## Frame Conventions

| Symbol | Meaning |
|--------|---------|
| Twb    | Body frame expressed in the world frame |
| Tbc    | Camera frame expressed in the body frame (fixed mounting offset) |
| Twc    | Camera frame expressed in the world frame (Twb * Tbc) |
| Tcw    | World frame expressed in the camera frame (Twc inverse) |

The naming convention reads right-to-left: **Twc** = "T that takes a point from
**c**amera frame to **w**orld frame."
