# Point Cloud Shapes Dataset (v1)

This dataset contains point clouds sampled **on the surfaces (or filled 2D regions)** of common shapes.
Each shape includes approximately **1500 points**. Coordinates are 3D (x, y, z).

## Shapes

- `square_plane`: filled unit square in the plane z=0 (side = 1.0)
- `disk`: filled circle (disk) in the plane z=0 (radius = 0.5)
- `triangle_eq`: filled equilateral triangle in the plane z=0 (side = 1.0)
- `cube_surface`: surface of a unit cube centered at the origin
- `sphere_surface`: surface of a sphere (radius = 0.5)
- `cylinder_surface`: surface of a cylinder with radius = 0.5 and height = 1.0 (lateral + caps, area-weighted)
- `cone_surface`: surface of a cone with base radius = 0.5 and height = 1.0 (lateral + base, area-weighted)
- `pyramid_square_surface`: surface of a square pyramid (base side = 1.0, height = 1.0)
- `torus_surface`: surface of a torus (major R = 1.0, minor r = 0.3), approximately uniform via rejection sampling

## File Formats

For each shape there are two files:
- `<shape>.csv` — CSV with header `x,y,z`
- `<shape>.npy` — NumPy array of shape (N, 3)

All shapes are centered near the origin with intuitive scales as noted above.

## Notes

- Random seed is fixed (42) for reproducibility.
- Surface sampling aims to be uniform with area weighting where applicable.
- The torus uses rejection sampling for the minor angle to correct for area distortion.
