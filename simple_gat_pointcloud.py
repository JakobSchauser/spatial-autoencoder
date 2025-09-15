import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch_geometric.nn import TransformerConv, global_max_pool

# --- Data Loading ---
def load_h5_files(file_list):
    data, label = [], []
    for fname in file_list:
        with h5py.File(fname, 'r') as f:
            data.append(f['data'][:])
            label.append(f['label'][:])
    return np.concatenate(data), np.concatenate(label)

class ModelNet40Dataset(Dataset):
    def __init__(self, files, augment=True):
        self.data, self.labels = load_h5_files(files)
        self.augment = augment
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        pts = self.data[idx]
        if self.augment:
            # Random rotation
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta),  np.cos(theta), 0],
                            [0, 0, 1]])
            pts = pts @ rot.T
            # Random scale
            scale = np.random.uniform(0.8, 1.2)
            pts = pts * scale
        x = torch.tensor(pts, dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        edge_index = self.knn_graph(x, k=16)
        return x, edge_index, y
    def knn_graph(self, x, k=16):
        dists = torch.cdist(x, x)
        idx = dists.topk(k+1, largest=False).indices[:,1:]
        src = torch.arange(x.size(0)).unsqueeze(1).repeat(1, k).flatten()
        dst = idx.flatten()
        return torch.stack([src, dst], dim=0)

def collate_fn(batch):
    xs, edge_indices, ys = zip(*batch)
    x = torch.cat(xs, dim=0)
    # Ensure all ys are at least 1D for torch.cat
    ys = [y if y.ndim > 0 else y.unsqueeze(0) for y in ys]
    ys = torch.cat(ys, dim=0)
    batch_vec = []
    edge_index_list = []
    n = 0
    for i, (pts, ei) in enumerate(zip(xs, edge_indices)):
        edge_index_list.append(ei + n)
        batch_vec.append(torch.full((pts.size(0),), i, dtype=torch.long))
        n += pts.size(0)
    edge_index = torch.cat(edge_index_list, dim=1)
    batch_vec = torch.cat(batch_vec, dim=0)
    return x, edge_index, batch_vec, ys

# --- Model ---
class SimpleNN(nn.Module):
    def __init__(self, in_dim=3, num_classes=40):
        super().__init__()
        self.gat1 = TransformerConv(in_dim, 32, heads=2, concat=True)
        self.gat2 = TransformerConv(64, 64, heads=2, concat=True)
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(32, 128)
        self.lin3 = nn.Linear(128, num_classes)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = global_max_pool(x, batch)
        x_latent = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x_latent))
        return self.lin3(x), x_latent

# --- Primitive Shapes Dataset Import ---
from primitive_shapes_dataset import PrimitiveShapesDataset, EXCLUDE_SHAPE

# --- Training ---
def train(model, loader, opt, device):
    model.train()
    for i, (x, edge_index, batch_vec, y) in enumerate(loader):
        x, edge_index, batch_vec, y = x.to(device), edge_index.to(device), batch_vec.to(device), y.to(device)
        out, x_latent = model(x, edge_index, batch_vec)
        loss = F.cross_entropy(out, y.squeeze())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"Batch {i}: loss = {loss.item():.4f}")

# --- Main ---
if __name__ == "__main__":
    train_files = [f.strip() for f in open("modelnet40_ply_hdf5_2048/train_files.txt")]
    test_files = [f.strip() for f in open("modelnet40_ply_hdf5_2048/test_files.txt")]
    train_files = [os.path.join("modelnet40_ply_hdf5_2048", os.path.basename(f)) for f in train_files]
    test_files = [os.path.join("modelnet40_ply_hdf5_2048", os.path.basename(f)) for f in test_files]
    train_ds1 = ModelNet40Dataset(train_files, augment=True)
    # Primitive shapes (exclude torus)
    primitive_folder = "pointcloud_shapes_v1"
    train_ds2 = PrimitiveShapesDataset(primitive_folder, augment=True, exclude_shape=EXCLUDE_SHAPE)
    # Concatenate datasets
    train_ds = ConcatDataset([train_ds1, train_ds2])
    # Get total number of classes (ModelNet40 + primitives minus excluded)
    with open(os.path.join(primitive_folder, 'shape_names.txt')) as f:
        primitive_names = [line.strip() for line in f]
    if EXCLUDE_SHAPE in primitive_names:
        primitive_names.remove(EXCLUDE_SHAPE)
    num_classes = 40 + len(primitive_names)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.5*1e-3)
    for epoch in range(6):
        train(model, train_loader, opt, device)
        print(f"Epoch {epoch+1} done.")
    torch.save(model.state_dict(), "simple_gat_model.pth")
    print("Model saved to simple_gat_model.pth")
