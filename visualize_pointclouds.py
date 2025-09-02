"""Point cloud visualization utility.

Features:
  - Lists available .npy point cloud files in a directory
  - Visualizes selected clouds (3D scatter) using matplotlib
  - Optionally shows all in a grid or animates rotation
  - Optionally saves figures to an output directory

Examples:
  List files only:
    python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --list

  Show a single shape:
    python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --names sphere_surface

  Show all shapes in a grid and save PNGs:
    python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --all --save-dir renders

  Animated rotation (interactive window):
    python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --names torus_surface --animate
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import numpy as np


def find_pointcloud_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    files = [f for f in os.listdir(root) if f.endswith('.npy')]
    files.sort()
    return files


def load_pointcloud(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:  # (S,N,D) -> treat each sample separately
        # For visualization pick the first sample; user can extend later.
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Point cloud must be (N,>=3). Got {arr.shape}")
    return arr.astype(np.float32)


@dataclass
class VizConfig:
    dir: str
    names: Tuple[str, ...] | None
    show_all: bool
    list_only: bool
    sample: int | None
    point_size: float
    alpha: float
    elevation: int
    azimuth: int
    animate: bool
    frames: int
    interval: float
    save_dir: str | None
    figsize: Tuple[float, float]
    grid_cols: int
    background: str


def parse_args() -> VizConfig:
    p = argparse.ArgumentParser(description="Visualize point cloud .npy files")
    p.add_argument('--dir', required=True, help='Directory containing .npy point cloud files')
    p.add_argument('--names', nargs='*', help='Specific base names (without .npy) to visualize')
    p.add_argument('--all', action='store_true', help='Visualize all .npy files in directory')
    p.add_argument('--list', action='store_true', dest='list_only', help='List available files and exit')
    p.add_argument('--sample', type=int, default=None, help='Random subsample size (per cloud)')
    p.add_argument('--point-size', type=float, default=6.0, help='Matplotlib scatter point size')
    p.add_argument('--alpha', type=float, default=0.85, help='Point alpha transparency')
    p.add_argument('--elevation', type=int, default=20, help='Initial elevation angle')
    p.add_argument('--azimuth', type=int, default=45, help='Initial azimuth angle')
    p.add_argument('--animate', action='store_true', help='Rotate the view continuously')
    p.add_argument('--frames', type=int, default=180, help='Number of frames for animation')
    p.add_argument('--interval', type=float, default=0.05, help='Seconds between frames when animating')
    p.add_argument('--save-dir', type=str, default=None, help='If set, save static figure(s) to this directory')
    p.add_argument('--figsize', type=float, nargs=2, default=(4.0, 4.0), help='Figure size for each subplot')
    p.add_argument('--grid-cols', type=int, default=3, help='Columns for grid when showing multiple clouds')
    p.add_argument('--background', type=str, default='white', help='Figure background color')
    a = p.parse_args()
    return VizConfig(
        dir=a.dir,
        names=tuple(a.names) if a.names else None,
        show_all=a.all,
        list_only=a.list_only,
        sample=a.sample,
        point_size=a.point_size,
        alpha=a.alpha,
        elevation=a.elevation,
        azimuth=a.azimuth,
        animate=a.animate,
        frames=a.frames,
        interval=a.interval,
        save_dir=a.save_dir,
        figsize=(a.figsize[0], a.figsize[1]),
        grid_cols=a.grid_cols,
        background=a.background,
    )


def select_files(cfg: VizConfig, all_files: Sequence[str]) -> List[str]:
    if cfg.list_only:
        return []
    if cfg.show_all:
        return list(all_files)
    if cfg.names:
        chosen = []
        base_map = {f[:-4]: f for f in all_files}
        for name in cfg.names:
            if name in base_map:
                chosen.append(base_map[name])
            else:
                raise FileNotFoundError(f"Requested name not found: {name}")
        return chosen
    # default: first file if exists
    return list(all_files[:1])


def prepare_cloud(cloud: np.ndarray, sample: Optional[int]) -> np.ndarray:
    if sample is not None and cloud.shape[0] > sample:
        idx = np.random.choice(cloud.shape[0], sample, replace=False)
        cloud = cloud[idx]
    return cloud


def axis_equal_3d(ax):
    # Achieve equal aspect ratio in 3D.
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def visualize(cfg: VizConfig, files: Sequence[str]):
    import matplotlib.pyplot as plt  # local import to avoid dependency unless used

    if not files:
        print("No files selected for visualization.")
        return
    n = len(files)
    if n == 1:
        fig = plt.figure(figsize=cfg.figsize)
        ax = fig.add_subplot(111, projection='3d')
        name = files[0]
        cloud = load_pointcloud(os.path.join(cfg.dir, name))
        cloud = prepare_cloud(cloud, cfg.sample)
        sc = ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=cfg.point_size, c=cloud[:, 2], cmap='viridis', alpha=cfg.alpha, linewidths=0)
        ax.set_title(name[:-4])
        ax.view_init(cfg.elevation, cfg.azimuth)
        axis_equal_3d(ax)
        fig.patch.set_facecolor(cfg.background)
        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            out_path = os.path.join(cfg.save_dir, f"{name[:-4]}.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Saved: {out_path}")

        if cfg.animate:
            for i in range(cfg.frames):
                ax.view_init(cfg.elevation, (cfg.azimuth + i * 360 / cfg.frames) % 360)
                plt.draw()
                plt.pause(cfg.interval)
        else:
            plt.show()
        return

    # Multiple files -> grid
    import math
    cols = cfg.grid_cols
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cfg.figsize[0] * cols, cfg.figsize[1] * rows))
    for i, fname in enumerate(files):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        cloud = load_pointcloud(os.path.join(cfg.dir, fname))
        cloud = prepare_cloud(cloud, cfg.sample)
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=cfg.point_size, c=cloud[:, 2], cmap='viridis', alpha=cfg.alpha, linewidths=0)
        ax.set_title(fname[:-4], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(cfg.elevation, cfg.azimuth)
        axis_equal_3d(ax)
    fig.tight_layout()
    fig.patch.set_facecolor(cfg.background)
    if cfg.save_dir:
        os.makedirs(cfg.save_dir, exist_ok=True)
        out_path = os.path.join(cfg.save_dir, 'grid.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
    if not cfg.animate:
        import matplotlib.pyplot as plt  # type: ignore
        plt.show()
    else:
        import matplotlib.pyplot as plt  # type: ignore
        for frame in range(cfg.frames):
            for ax in fig.axes:
                ax.view_init(cfg.elevation, (cfg.azimuth + frame * 360 / cfg.frames) % 360)
            plt.draw(); plt.pause(cfg.interval)


def main():  # pragma: no cover
    cfg = parse_args()
    files = find_pointcloud_files(cfg.dir)
    if cfg.list_only:
        print("Available point clouds (.npy):")
        for f in files:
            print(" -", f)
        return
    selected = select_files(cfg, files)
    if not selected:
        print("No files selected (maybe directory empty?)")
        return
    visualize(cfg, selected)


if __name__ == '__main__':  # pragma: no cover
    main()
