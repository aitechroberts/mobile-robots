# Occupancy Grid Mapping -- How It Works

This document explains the occupancy grid mapping functions, the coordinate
spaces they operate in, and the Bayesian update math that drives them. All
equations reference the 16-761 lecture slides (Mod02 Lec02: Bayes Filter,
Mod02 Lec03: Occupancy Grid Maps).

## The Big Picture

Occupancy grid mapping answers: **"Given known robot poses and depth sensor
readings, which parts of the 3D world are occupied, free, or unknown?"**

The idea (OMs Slide 17) is to decompose the continuous world into a grid of
discrete cells. Each cell is an independent binary random variable -- it is
either OCCUPIED or FREE. We track the probability of occupancy for every cell
and update it each time a sensor ray passes through or terminates in that cell.

The full pipeline per sensor reading:
1. From the robot's known pose, cast rays into the mesh to get world-frame
   hit points (handled by the sensor simulator).
2. For each ray, walk through the 3D grid to find which cells the ray
   intersects (voxel traversal via `get_raycells`).
3. Update each traversed cell: cells the ray passed through become more
   likely to be FREE; the cell where the ray terminated becomes more likely
   to be OCCUPIED.

---

## Three Coordinate Spaces (OMs Slides 8-10)

The occupancy grid operates in three interchangeable spaces:

### Point Space (world frame)

A continuous 3D position in the world, represented as `Point(x, y, z)`.

\
p = [x, y, z] in R^3
\

The grid covers a finite region of the world, starting at the **origin** and
extending `width * resolution` in X, `height * resolution` in Y, and
`depth * resolution` in Z.

### Cell Space (grid coordinates)

A discrete 3D location in the grid, represented as `Cell(row, col, slice)`.

\
c = (row, col, slice) where row, col, slice in N_0
\

The axis mapping is:
- **row** corresponds to the **Y-axis** in 3D
- **col** corresponds to the **X-axis** in 3D
- **slice** corresponds to the **Z-axis** in 3D

### Index Space (flat array)

A single integer indexing into the flat `data` array that stores the log-odds
values. The grid has `width * height * depth` total cells.

\
index in {0, 1, ..., width * height * depth - 1}
\

---

## Conversion Functions

### `cell2index` / `index2cell` (OMs Slide 9)

The 2D hash function from Slide 9 is:

\
index = row * W + col
row = floor(index / W)
col = index % W
\

Extended to 3D with a depth dimension D:

\
index = row * W * D + col * D + slice
\

And the inverse:

\
slice = index % D
col = (index // D) % W
row = index // (D * W)
\

where W = width (number of columns) and D = depth (number of slices).

`cell2index` returns `np.uint64`. `index2cell` returns a `Cell` whose row, col,
and slice are all `np.uint64`.

**Verification against test data:**
- `Cell(0, 14, 44)` --> `0*200*100 + 14*100 + 44 = 1444`
- `Cell(75, 123, 45)` --> `75*200*100 + 123*100 + 45 = 1,512,345`

### `cell2point` / `point2cell` (OMs Slide 10)

Slide 10 defines the lower-left corner convention:

**Cell to Point (lower-left corner of the cell):**

\
x = col * alpha + x_m
y = row * alpha + y_m
z = slice * alpha + z_m
\

where alpha is the resolution and (x_m, y_m, z_m) is the grid origin. The cell
coordinates are cast to `np.float64` before the arithmetic so the resulting
`Point` holds continuous world-frame values with no integer truncation artifacts.

**Point to Cell (floor to get the containing cell):**

\
col = floor( (x - x_m) / alpha )
row = floor( (y - y_m) / alpha )
slice = floor( (z - z_m) / alpha )
\

The floor function is essential here -- it ensures that a point anywhere inside
a cell maps to that cell, not to a neighbor. After flooring, the values are
cast to `np.uint64` so the returned `Cell` holds unsigned integer grid
coordinates.

Note: `cell2point` returns the lower-left corner. When the center of the cell
is needed (e.g., for visualization), `resolution/2` is added separately by
the calling code.

### Type discipline across conversions

Cell coordinates (row, col, slice) are discrete grid positions -- there is no
such thing as a fractional row. By enforcing `np.uint64` in `point2cell`,
`index2cell`, and `cell2index`, and `np.float64` in `cell2point`, the type
contract is handled once at the conversion boundary. Downstream code like
`get_raycells` already operates in `np.uint64` arithmetic for its stepping
logic, so when the helpers produce the right types, everything just works
without ad-hoc casts scattered through the codebase.

### numpy version constraint

The provided `get_raycells` voxel traversal relies on `np.uint64` wrapping
behavior when a ray steps backward past the grid boundary. For example, when
`c.row` is `np.uint64(0)` and `step_row` is the Python int `-1`:

```
c.row = np.uint64(c.row) + step_row
```

In **numpy <= 1.24**, this silently wraps to `np.uint64(18446744073709551615)`,
which `cell_in_grid` catches on the next loop iteration to terminate the
traversal cleanly.

In **numpy >= 2.0**, this raises `OverflowError: Python integer -1 out of
bounds for uint64`. The newer numpy no longer permits adding a negative Python
`int` to a `uint64`.

Note that the inconsistency exists in the provided code: `step_row` is
explicitly cast to `np.float64` when computing `tdelta` (e.g.,
`self.resolution * np.float64(step_row) * ...`) but is left as a bare Python
`int` in the stepping line. This only manifests as an error on numpy 2.x.

The assignment was written for Python 3.8, which is compatible with numpy up to
**1.24.4** (Python 3.8 support was dropped in numpy 1.25). Using numpy 1.24.x
ensures the wrapping behavior works as intended.

**Verification against test data:**
- `Cell(0, 14, 44)` --> `Point(-35 + 14*0.5, -35 + 0*0.5, -2 + 44*0.5)`
  = `Point(-28.0, -35.0, 20.0)`
- `Cell(75, 123, 45)` --> `Point(-35 + 123*0.5, -35 + 75*0.5, -2 + 45*0.5)`
  = `Point(26.5, 2.5, 20.5)`

---

## The Binary Bayes Filter in Log-Odds Form

### Why log-odds? (Bayes Slides 15-17, OMs Slides 19-23)

Each cell holds a probability p(o_i) representing the belief that it is
occupied. Bayes' rule lets us update this belief with each new measurement,
but naively multiplying probabilities leads to numerical underflow and requires
tracking a normalizer.

The log-odds representation eliminates both problems. The log-odds of a
probability p is defined as:

\
l = log( p / (1 - p) )
\

This maps (0, 1) to (-inf, +inf). The key property: the Bayesian update
becomes a simple **addition** in log-odds space.

### `logodds(probability)` (OMs Slide 23)

The logit transform:

\
l = ln( p / (1 - p) )
\

- p = 0.5 (unknown) maps to l = 0
- p near 1 (occupied) maps to large positive l
- p near 0 (free) maps to large negative l

### `probability(logodds)` (OMs Slide 23, inverse)

The sigmoid (inverse logit):

\
p = 1 / (1 + exp(-l))
\

### The recursive update (OMs Slide 23)

From the Bayes filter derivation on Slide 23, the log-odds update rule is:

\
l_t = l(o_i | z_t, x_t) - l_0 + l_{t-1}
\

where:
- l_t is the new log-odds after incorporating measurement z_t at pose x_t
- l(o_i | z_t, x_t) is the inverse sensor model log-odds (either HIT or MISS)
- l_0 is the prior log-odds = log(0.5 / 0.5) = 0
- l_{t-1} is the previous log-odds

Since l_0 = 0 (the prior is p = 0.5 for all cells, which in log-odds is zero),
the update simplifies to:

\
l_t = l_update + l_{t-1}
\

In other words, **just add the update value to the existing cell value**. This
is why log-odds is so powerful: what would be a multiplication of likelihood
ratios in probability space is just an addition in log-odds space.

### `update_logodds(cell, update)` (OMs Slide 24)

Following the pseudocode on Slide 24:

1. Look up the cell's index in the flat data array.
2. Add `update` to `data[index]`.
3. Clamp the result to `[min_clamp, max_clamp]` for numerical stability.

The `update` value is either `logodds_hit` (from hit_probability = 0.99,
giving a large positive ~4.6) or `logodds_miss` (from miss_probability = 0.1,
giving a negative ~-2.2). The clamping prevents any cell from reaching
probability 0 or 1, which would make it impossible to change its state with
future observations.

---

## Ray Update Logic

### `add_ray(start, end, max_range)` (OMs Slide 24 pseudocode)

This is the top-level function called for each sensor ray. It implements the
pseudocode from Slide 24:

```
For every ray (o, d) in z_t:
  For every cell i along the ray:
    If the ray did not terminate in i (MISS):
      l_i = l(MISS) - l_0 + l_i
    Else (HIT):
      l_i = l(HIT) - l_0 + l_i
```

The implementation:

1. Compute the Euclidean length of the ray: `|end - start|`.
2. Call `get_raycells(start, end)` to get the ordered list of cells the ray
   passes through (3D voxel traversal, already provided).
3. Decide if the ray hit a surface or reached max range:
   - **ray_length >= max_range**: The ray did not hit anything within sensor
     range. ALL cells along the ray are updated with MISS (they are free
     space).
   - **ray_length < max_range**: The ray hit a surface. All cells except the
     last are updated with MISS, and the **last cell** is updated with HIT
     (it contains the obstacle).

### Intuition

Think of a laser beam shot from the sensor. Everything the beam passes through
without hitting anything must be free -- the beam went right through it. If
the beam stops at a surface, the cell where it stopped is occupied. If the
beam travels its full max range without hitting anything, even the endpoint
is free.

---

## Data Flow Summary

```
Known Pose (Twb)  +  Mesh Environment
         |
         v
   ray_mesh_intersect (sensor simulator)
         |
         v
   World-Frame Hit Points (Nx3)
         |
         v
   For each ray: add_ray(camera_origin, hit_point, max_range)
         |
         v
   get_raycells (3D voxel traversal)
         |
         v
   Ordered list of cells along the ray
         |
         v
   update_miss / update_hit per cell
         |
         v
   data[] array updated in log-odds
         |
         v
   Classify cells: occupied / free / unknown
```

---

## Config Parameter Reference

These are loaded from `config/map.yaml` and control the grid and update
behavior:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| width | 200 | Number of columns (X-axis cells) |
| height | 200 | Number of rows (Y-axis cells) |
| depth | 100 | Number of slices (Z-axis cells) |
| origin | [-35, -35, -2] | World-frame position of the grid corner |
| resolution | 0.5 | Cell size in meters |
| hit_probability | 0.99 | Inverse sensor model p for a HIT cell |
| miss_probability | 0.1 | Inverse sensor model p for a MISS cell |
| free_threshold | 0.3 | Probability below which a cell is considered FREE |
| occupied_threshold | 0.7 | Probability above which a cell is considered OCCUPIED |
| min_clamp | 0.01 | Minimum allowed probability (prevents log-odds = -inf) |
| max_clamp | 0.9999 | Maximum allowed probability (prevents log-odds = +inf) |

All probability parameters are converted to log-odds at initialization and
stored internally in that form. The `data[]` array stores log-odds values,
never raw probabilities.

---

## Threshold Classification

After updates, each cell is classified based on its log-odds value:

- **OCCUPIED**: `logodds >= logodds(occupied_threshold)` -- the cell has been
  hit enough times to be confident there is an obstacle.
- **FREE**: `logodds <= logodds(free_threshold)` -- enough rays have passed
  through the cell to be confident it is empty.
- **UNKNOWN**: between the two thresholds -- not enough evidence either way.
  All cells start here since `data[]` is initialized to 0 (log-odds of p=0.5).
