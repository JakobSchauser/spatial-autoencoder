"""Training script for GAT point cloud autoencoder.

Usage (basic):
    python train.py --data path/to/pointcloud.npy --epochs 200

The data file should be either:
  - A (N, D) single point cloud array (will be treated as one sample, optionally augmented), or
  - A (S, N, D) batch of S point clouds.

Graph construction: k-NN (default k=16) on XYZ coordinates each sample.

Autoencoder reconstructs per-point coordinates; choose --use-chamfer for
order-invariant reconstruction (Chamfer distance) or default MSE for ordered.
You may also combine them with a weight.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from models.gat_autoencoder import GATPointCloudAutoencoder, chamfer_distance

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as e:  # pragma: no cover
    raise ImportError("torch_geometric is required for training. Install it first.") from e


def build_knn_graph(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return edge_index (2, E) for k-NN graph (undirected, with self loops removed)."""
    # Attempt fast PyG knn_graph if available
    try:
        from torch_geometric.nn import knn_graph  # type: ignore

        edge_index = knn_graph(points, k=k, loop=False)
        return edge_index
    except Exception:  # fallback (O(N^2))
        # (N, N)
        dist2 = torch.cdist(points, points, p=2)
        knn = dist2.topk(k=k + 1, largest=False).indices  # include self
        N = points.size(0)
        src_list = []
        dst_list = []
        for i in range(N):
            for j in knn[i].tolist():
                if j == i:
                    continue
                src_list.append(i)
                dst_list.append(j)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        # make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        # remove potential duplicates
        unique, idx = torch.unique(edge_index, dim=1, return_inverse=False, return_counts=False), None
        edge_index = unique
        return edge_index


def random_so3_matrix(device: torch.device) -> torch.Tensor:
    """Generate a random 3x3 rotation matrix using QR decomposition (Haar)."""
    # Draw random matrix with N(0,1)
    mat = torch.randn(3, 3, device=device)
    # QR decomposition
    q, r = torch.linalg.qr(mat)
    # Make Q uniform by adjusting sign of columns using diag of R
    d = torch.diag(r)
    ph = d.sign()
    q = q * ph
    # Ensure right-handed (det=+1)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class PointCloudDataset(Dataset):
    def __init__(
        self,
        array: np.ndarray,
        k: int = 16,
        augment: bool = False,
        jitter_std: float = 0.0,
        rotational_augment: bool = False,
        rotation_mode: str = "z",  # "z" or "so3"
        scale_augment: bool = False,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if array.ndim == 2:  # single cloud
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError("Input array must have shape (N,D) or (S,N,D)")
        self.clouds = array.astype(np.float32)
        self.S, self.N, self.D = self.clouds.shape
        self.k = k
        self.augment = augment
        self.jitter_std = jitter_std
        # Augmentation settings
        self.rotational_augment = rotational_augment
        self.rotation_mode = rotation_mode
        self.scale_augment = scale_augment
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.device = device

    def __len__(self) -> int:
        return self.S

    def _augment(self, pts: torch.Tensor) -> torch.Tensor:
        # Center for rotation/scaling
        center = pts[:, :3].mean(0, keepdim=True)
        if self.rotational_augment and pts.size(1) >= 3:
            if self.rotation_mode == "so3":
                R = random_so3_matrix(pts.device)
            else:  # z-axis
                theta = torch.rand((), device=pts.device) * (2 * math.pi)
                c = torch.cos(theta)
                s = torch.sin(theta)
                R = torch.zeros(3, 3, device=pts.device)
                R[0, 0] = c; R[0, 1] = -s; R[0, 2] = 0.0
                R[1, 0] = s; R[1, 1] = c;  R[1, 2] = 0.0
                R[2, 0] = 0.0; R[2, 1] = 0.0; R[2, 2] = 1.0
            pts_xyz = pts[:, :3] - center
            pts[:, :3] = (R @ pts_xyz.T).T + center
        if self.scale_augment:
            scale = torch.empty(1, device=pts.device).uniform_(self.scale_min, self.scale_max)
            pts[:, :3] = (pts[:, :3] - center) * scale + center
        if self.jitter_std > 0:
            pts = pts + torch.randn_like(pts) * self.jitter_std
        return pts

    def __getitem__(self, idx: int):
        pts = torch.from_numpy(self.clouds[idx])  # (N, D)
        if self.augment:
            pts = self._augment(pts)
        edge_index = build_knn_graph(pts[:, :3], k=self.k)
        data = Data(x=pts, pos=pts[:, :3], edge_index=edge_index)
        return data


@dataclass
class TrainConfig:
    data: Optional[str] = None
    data_dir: Optional[str] = None
    epochs: int = 200
    batch_size: int = 4
    k: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels: int = 3
    hidden_dims: Tuple[int, ...] = (64, 128)
    # Reduced default latent size for a tighter bottleneck (adjust via --latent-dim)
    latent_dim: int = 8
    heads: int = 4
    dropout: float = 0.1
    decode_hidden: Tuple[int, ...] = (128,)
    use_chamfer: bool = False
    chamfer_weight: float = 1.0
    mse_weight: float = 1.0
    augment: bool = False
    jitter_std: float = 0.0
    rotational_augment: bool = False
    rotation_mode: str = "z"  # z or so3
    scale_augment: bool = False
    scale_min: float = 0.8
    scale_max: float = 1.2
    save_dir: str = "runs"
    tag: str = "default"
    save_every: int = 50


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train GAT point cloud autoencoder")
    p.add_argument("--data", type=str, help="Path to a .npy file containing point cloud(s)")
    p.add_argument("--data-dir", type=str, help="Directory containing multiple .npy point cloud files")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 128])
    p.add_argument("--latent-dim", type=int, default=8, help="Latent embedding size (default: 8)")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--decode-hidden", type=int, nargs="*", default=[128])
    p.add_argument("--use-chamfer", action="store_true")
    p.add_argument("--chamfer-weight", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--jitter-std", type=float, default=0.0)
    p.add_argument("--rotational-augment", action="store_true")
    p.add_argument("--rotation-mode", choices=["z", "so3"], default="z")
    p.add_argument("--scale-augment", action="store_true")
    p.add_argument("--scale-min", type=float, default=0.8)
    p.add_argument("--scale-max", type=float, default=1.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()
    if not args.data and not args.data_dir:
        p.error("You must provide either --data or --data-dir")
    return TrainConfig(
        data=args.data,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k=args.k,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        in_channels=args.in_channels,
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
        heads=args.heads,
        dropout=args.dropout,
        decode_hidden=tuple(args.decode_hidden),
        use_chamfer=args.use_chamfer,
        chamfer_weight=args.chamfer_weight,
        mse_weight=args.mse_weight,
        augment=args.augment,
        jitter_std=args.jitter_std,
        rotational_augment=args.rotational_augment,
    rotation_mode=args.rotation_mode,
    scale_augment=args.scale_augment,
    scale_min=args.scale_min,
    scale_max=args.scale_max,
        save_dir=args.save_dir,
        tag=args.tag,
        save_every=args.save_every,
    )


def load_data(path: str) -> np.ndarray:
    return np.load(path)


def load_data_dir(path: str) -> np.ndarray:
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError(f"No .npy files found in directory {path}")
    clouds: List[np.ndarray] = []
    min_pts = None
    feat_dim = None
    for fname in sorted(files):
        arr = np.load(os.path.join(path, fname))
        if arr.ndim == 3:
            arr = arr[0]  # take first if batch present
        if arr.ndim != 2:
            raise ValueError(f"File {fname} must have shape (N,D) or (S,N,D)")
        if feat_dim is None:
            feat_dim = arr.shape[1]
        elif arr.shape[1] != feat_dim:
            raise ValueError("All point clouds must have same feature dimension")
        min_pts = arr.shape[0] if min_pts is None else min(min_pts, arr.shape[0])
        clouds.append(arr)
    assert min_pts is not None
    # Subsample each to min_pts for uniform tensor shape
    proc = []
    for arr in clouds:
        if arr.shape[0] > min_pts:
            idx = np.random.choice(arr.shape[0], min_pts, replace=False)
            arr = arr[idx]
        proc.append(arr)
    stacked = np.stack(proc, axis=0)  # (S, N, D)
    return stacked


def main(cfg: TrainConfig):  # noqa: D401
    os.makedirs(cfg.save_dir, exist_ok=True)
    run_dir = os.path.join(cfg.save_dir, cfg.tag)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")

    if cfg.data_dir:
        array = load_data_dir(cfg.data_dir)
    elif cfg.data:
        array = load_data(cfg.data)
    else:
        raise RuntimeError("No data source provided")
    if array.ndim == 2 and array.shape[1] != cfg.in_channels:
        print(f"Warning: overriding in_channels {cfg.in_channels} -> {array.shape[1]}")
        cfg.in_channels = array.shape[1]
    elif array.ndim == 3 and array.shape[2] != cfg.in_channels:
        print(f"Warning: overriding in_channels {cfg.in_channels} -> {array.shape[2]}")
        cfg.in_channels = array.shape[2]

    dataset = PointCloudDataset(
        array=array,
        k=cfg.k,
        augment=cfg.augment,
        jitter_std=cfg.jitter_std,
        rotational_augment=cfg.rotational_augment,
        rotation_mode=cfg.rotation_mode,
        scale_augment=cfg.scale_augment,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device(cfg.device)
    model = GATPointCloudAutoencoder(
        in_channels=cfg.in_channels,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
        heads=cfg.heads,
        dropout=cfg.dropout,
        decode_hidden_dims=cfg.decode_hidden,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, batch_vec: Optional[torch.Tensor]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        if cfg.mse_weight > 0:
            losses.append(cfg.mse_weight * nn.functional.mse_loss(recon, target))
        if cfg.use_chamfer and cfg.chamfer_weight > 0:
            if batch_vec is None:
                cd = chamfer_distance(recon[:, :3], target[:, :3])
            else:
                cd_vals = []
                for b in batch_vec.unique():
                    mask = batch_vec == b
                    cd_vals.append(chamfer_distance(recon[mask, :3], target[mask, :3]))
                cd = torch.stack(cd_vals).mean()
            losses.append(cfg.chamfer_weight * cd)
        if not losses:
            # Ensure a differentiable zero if both weights disabled.
            return torch.zeros((), device=recon.device, requires_grad=True)
        return torch.stack(losses).sum()

    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            recon, z = model(batch.x, batch.edge_index)
            batch_vec = getattr(batch, 'batch', None)
            loss = reconstruction_loss(recon, batch.x, batch_vec)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch:04d} | loss {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "loss": avg_loss,
            }, os.path.join(run_dir, "best.pt"))

        if epoch % cfg.save_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "loss": avg_loss,
            }, os.path.join(run_dir, f"epoch_{epoch}.pt"))

    print(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":  # pragma: no cover
    cfg = parse_args()
    main(cfg)
