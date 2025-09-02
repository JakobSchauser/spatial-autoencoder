# Encoding spatial information in latent space

 Quantifying the similarity between two point clouds is still an open problem. Using specialized autoencoders, the latent-space correlations can grants us useful and symmetry-respecting information.

## GAT Point Cloud Autoencoder

This repository now contains a minimal Graph Attention Network (GAT) based autoencoder for point clouds using PyTorch Geometric.

### Files

- `models/gat_autoencoder.py` – Defines the `GATPointCloudAutoencoder` and a simple Chamfer distance helper.
- `train.py` – Training script that loads point cloud data from a `.npy` file and trains the autoencoder.
- `visualize_pointclouds.py` – Utility to list and visualize `.npy` point cloud files (3D scatter with matplotlib).
- `requirements.txt` – Core Python dependencies (install PyTorch Geometric per its official instructions if wheels are unavailable for your platform/CUDA combo).

### Installation

1. Create environment (example with `venv`):
```
python -m venv .venv
.\.venv\Scripts\activate
```
2. Install PyTorch matching your CUDA (see https://pytorch.org/ ) then:
```
pip install -r requirements.txt
```
If `torch_geometric` fails via requirements, follow the official instructions: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

### Data Format

Provide a `.npy` file:
- Shape `(N, D)` for a single point cloud, or
- Shape `(S, N, D)` for a dataset of S point clouds (all with N points, D features; typically D=3 for XYZ).

### Train

Basic example:
```
python train.py --data data/pointclouds.npy --epochs 200 --k 16 --batch-size 4
```

Optional flags:
- `--data-dir point_cloud_data/pointcloud_shapes_v1` train on all .npy clouds in a directory (auto-subsamples to common point count).
- `--use-chamfer` add Chamfer distance (order invariance) to reconstruction.
- Augmentations: `--augment --rotational-augment --rotation-mode so3 --scale-augment --scale-min 0.7 --scale-max 1.3 --jitter-std 0.005`.
- Adjust model: `--hidden-dims 64 128 256 --latent-dim 256 --decode-hidden 256 128`.

Default latent size is now 64 for a stronger compression baseline. Increase with `--latent-dim 128` or `--latent-dim 256` if you observe underfitting (blurry / collapsed reconstructions) or want richer embeddings.

Artifacts (checkpoints + config) are stored under `runs/<tag>/`.

### Visualize Point Clouds

List available shapes:
```
python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --list
```

Display a single shape:
```
python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --names sphere_surface
```

Show all shapes in a grid and save a PNG:
```
python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --all --save-dir renders
```

Animated rotation of one shape:
```
python visualize_pointclouds.py --dir point_cloud_data/pointcloud_shapes_v1 --names torus_surface --animate --frames 240 --interval 0.03
```

### Extending

- Replace k-NN graph with dynamic graph recomputation each epoch or radius graphs.
- Swap reconstruction loss with a more efficient Chamfer (e.g. PyTorch3D) for large point counts.
- Add global pooling and a latent bottleneck if you want a single vector per cloud.

### License / Disclaimer

Prototype-quality code generated automatically; validate suitability for production/research before heavy use.
