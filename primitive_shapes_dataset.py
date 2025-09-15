
import os
import numpy as np
import torch
from torch.utils.data import Dataset
EXCLUDE_SHAPE = 'torus'  # Set to shape name to exclude from training

class PrimitiveShapesDataset(Dataset):
    def __init__(self, folder, augment=True, exclude_shape=EXCLUDE_SHAPE):
        with open(os.path.join(folder, 'shape_names.txt')) as f:
            self.names = [line.strip() for line in f]
        if exclude_shape is not None:
            self.names = [n for n in self.names if n != exclude_shape]
        self.files = [os.path.join(folder, n + '.npy') for n in self.names]
        self.augment = augment
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        pts = np.load(self.files[idx])
        if self.augment:
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta),  np.cos(theta), 0],
                            [0, 0, 1]])
            pts = pts @ rot.T
            scale = np.random.uniform(0.8, 1.2)
            pts = pts * scale
        x = torch.tensor(pts, dtype=torch.float)
        y = torch.tensor(idx, dtype=torch.long)
        edge_index = self.knn_graph(x, k=16)
        return x, edge_index, y
    def get_samples(self, shape_name, n=10):
        """Return n samples (with augmentation) for the given shape name."""
        idx = self.names.index(shape_name)
        samples = []
        for _ in range(n):
            pts = np.load(self.files[idx])
            if self.augment:
                theta = np.random.uniform(0, 2*np.pi)
                rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0, 0, 1]])
                pts = pts @ rot.T
                scale = np.random.uniform(0.8, 1.2)
                pts = pts * scale
            x = torch.tensor(pts, dtype=torch.float)
            y = torch.tensor(idx, dtype=torch.long)
            edge_index = self.knn_graph(x, k=16)
            samples.append((x, edge_index, y))
        return samples
    def knn_graph(self, x, k=16):
        dists = torch.cdist(x, x)
        idx = dists.topk(k+1, largest=False).indices[:,1:]
        src = torch.arange(x.size(0)).unsqueeze(1).repeat(1, k).flatten()
        dst = idx.flatten()
        return torch.stack([src, dst], dim=0)
