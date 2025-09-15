from primitive_shapes_dataset import PrimitiveShapesDataset
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import global_max_pool, TransformerConv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if you only want to save images
from mpl_toolkits.mplot3d import Axes3D
import h5py



def load_h5_files(file_list):
    data, label = [], []
    for fname in file_list:
        with h5py.File(fname, 'r') as f:
            data.append(f['data'][:])
            label.append(f['label'][:])
    return np.concatenate(data), np.concatenate(label)

class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, files, augment=True):
        self.data, self.labels = load_h5_files(files)
        self.augment = augment
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        pts = self.data[idx]
        if self.augment:
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta),  np.cos(theta), 0],
                            [0, 0, 1]])
            pts = pts @ rot.T
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


# --- Model (copied from training script) ---
class SimpleNN(torch.nn.Module):
    def __init__(self, in_dim=3, num_classes=40):
        super().__init__()
        self.gat1 = TransformerConv(in_dim, 32, heads=2, concat=True)
        self.gat2 = TransformerConv(64, 64, heads=2, concat=True)
        self.lin1 = torch.nn.Linear(128, 32)
        self.lin2 = torch.nn.Linear(32, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = global_max_pool(x, batch)
        x_latent = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x_latent))
        return self.lin3(x), x_latent


if __name__ == "__main__":
    from torch.utils.data import ConcatDataset, Dataset
    # Load class names for ModelNet40 and primitives
    with open("modelnet40_ply_hdf5_2048/shape_names.txt") as f:
        modelnet_names = [line.strip() for line in f]
    primitive_folder = "pointcloud_shapes_v1"
    with open(os.path.join(primitive_folder, 'shape_names.txt')) as f:
        primitive_names = [line.strip() for line in f]
    from primitive_shapes_dataset import EXCLUDE_SHAPE, PrimitiveShapesDataset
    primitive_names_for_model = [n for n in primitive_names if n != EXCLUDE_SHAPE]
    num_classes = len(modelnet_names) + len(primitive_names_for_model)
    class_names = modelnet_names + primitive_names

    # ModelNet40 test set
    test_files = [f.strip() for f in open("modelnet40_ply_hdf5_2048/test_files.txt")]
    test_files = [os.path.join("modelnet40_ply_hdf5_2048", os.path.basename(f)) for f in test_files]
    test_ds1 = ModelNet40Dataset(test_files, augment=True)

    # Primitive shapes test set (all, including torus), with label offset
    prim_ds = PrimitiveShapesDataset(primitive_folder, augment=True, exclude_shape='')
    class PrimitiveSamplesDataset(Dataset):
        def __init__(self, base_ds, shape_names, offset, n_per_class=10):
            self.samples = []
            for i, name in enumerate(shape_names):
                for x, edge_index, y in base_ds.get_samples(name, n=n_per_class):
                    y = torch.tensor(i + offset, dtype=torch.long)
                    self.samples.append((x, edge_index, y))
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]
    test_ds2 = PrimitiveSamplesDataset(prim_ds, primitive_names, offset=len(modelnet_names), n_per_class=10)

    # --- Generate 10 augmented torus samples at runtime ---
    class TorusAugRuntimeDataset(Dataset):
        def __init__(self, torus_path, label, n_aug=10):
            self.torus_pts = np.load(torus_path)
            self.label = label
            self.n_aug = n_aug
        def __len__(self):
            return self.n_aug
        def __getitem__(self, idx):
            pts = self.torus_pts.copy()
            # Apply random rotation and scaling (same as in PrimitiveShapesDataset)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            psi = np.random.uniform(0, 2*np.pi)
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta),  np.cos(theta), 0],
                          [0, 0, 1]])
            Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(psi), -np.sin(psi)],
                          [0, np.sin(psi),  np.cos(psi)]])
            rot = Rz @ Ry @ Rx
            pts = pts @ rot.T
            scale = np.random.uniform(0.8, 1.2)
            pts = pts * scale
            x = torch.tensor(pts, dtype=torch.float)
            # k-NN graph
            dists = torch.cdist(x, x)
            k = 16
            idxs = dists.topk(k+1, largest=False).indices[:,1:]
            src = torch.arange(x.size(0)).unsqueeze(1).repeat(1, k).flatten()
            dst = idxs.flatten()
            edge_index = torch.stack([src, dst], dim=0)
            y = torch.tensor(self.label, dtype=torch.long)
            return x, edge_index, y

    # Find torus index and path
    torus_idx = None
    torus_path = None
    for i, name in enumerate(primitive_names):
        if name == 'torus':
            torus_idx = i + len(modelnet_names)
            torus_path = os.path.join(primitive_folder, 'torus.npy')
            break

    datasets = [test_ds1, test_ds2]
    if torus_idx is not None and torus_path is not None and os.path.exists(torus_path):
        torus_aug_ds = TorusAugRuntimeDataset(torus_path, torus_idx, n_aug=10)
        datasets.append(torus_aug_ds)
    test_ds = ConcatDataset(datasets)

    # --- Collect all test samples and augment to ensure 10 per class ---
    print("\nBalancing test set: ensuring 10 samples per class (with augmentation if needed)...")
    def flatten_to_triplet(sample):
        # Recursively flatten until a tuple of length 3 is found
        while isinstance(sample, tuple) and len(sample) == 1:
            sample = sample[0]
        if not (isinstance(sample, tuple) and len(sample) == 3):
            raise ValueError(f"Sample could not be flattened to a triplet: {sample}")
        return sample

    all_samples = []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        x, edge_index, y = flatten_to_triplet(sample)
        all_samples.append((x, edge_index, y.item()))

    # Group by class
    from collections import defaultdict
    class_to_samples = defaultdict(list)
    for x, edge_index, y in all_samples:
        class_to_samples[y].append((x, edge_index, y))

    # Helper: augment a sample (random rotation/scale, same as in dataset)
    def augment_sample(x):
        pts = x.numpy().copy()
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        psi = np.random.uniform(0, 2*np.pi)
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                      [0, 1, 0],
                      [-np.sin(phi), 0, np.cos(phi)]])
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(psi), -np.sin(psi)],
                      [0, np.sin(psi),  np.cos(psi)]])
        rot = Rz @ Ry @ Rx
        pts = pts @ rot.T
        scale = np.random.uniform(0.8, 1.2)
        pts = pts * scale
        x_aug = torch.tensor(pts, dtype=torch.float)
        return x_aug

    # For each class, if <10, augment existing samples to reach 10
    balanced_samples = []
    for label in class_to_samples:
        samples = class_to_samples[label]
        n = len(samples)
        # Always ensure y is a torch.Tensor
        def ensure_tensor_label(sample):
            x, edge_index, y = sample
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            return (x, edge_index, y)
        if n >= 10:
            balanced_samples.extend([ensure_tensor_label(s) for s in samples[:10]])
        else:
            # Use all originals, then augment as needed
            balanced_samples.extend([ensure_tensor_label(s) for s in samples])
            for i in range(10 - n):
                x, edge_index, y = samples[i % n]
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.long)
                x_aug = augment_sample(x)
                # Recompute edge_index for augmented sample
                dists = torch.cdist(x_aug, x_aug)
                k = 16
                idxs = dists.topk(k+1, largest=False).indices[:,1:]
                src = torch.arange(x_aug.size(0)).unsqueeze(1).repeat(1, k).flatten()
                dst = idxs.flatten()
                edge_index_aug = torch.stack([src, dst], dim=0)
                balanced_samples.append((x_aug, edge_index_aug, y))

    # Now, balanced_samples has exactly 10 per class (for all classes present)
    print(f"Balanced test set: {len(balanced_samples)} samples, {len(set(y for _,_,y in balanced_samples))} classes.")

    # Create a DataLoader from balanced_samples
    class BalancedTestDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    balanced_test_ds = BalancedTestDataset(balanced_samples)
    test_loader = DataLoader(balanced_test_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Load class names for ModelNet40 and primitives (minus excluded)
    with open("modelnet40_ply_hdf5_2048/shape_names.txt") as f:
        modelnet_names = [line.strip() for line in f]
    import os
    with open(os.path.join(primitive_folder, 'shape_names.txt')) as f:
        primitive_names = [line.strip() for line in f]
    # For model: exclude EXCLUDE_SHAPE (e.g. 'torus')
    from primitive_shapes_dataset import EXCLUDE_SHAPE
    primitive_names_for_model = [n for n in primitive_names if n != EXCLUDE_SHAPE]
    num_classes = len(modelnet_names) + len(primitive_names_for_model)
    # For analysis: include all (including torus)
    class_names = modelnet_names + primitive_names

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("simple_gat_model.pth", map_location=device))
    model.eval()

    # Visualize a batch and compute cosine similarities
    # --- Always include a torus sample in the first display batch ---
    # Find a torus sample index in the balanced test set
    torus_idx = None
    for i, name in enumerate(class_names):
        if name == 'torus':
            torus_idx = i
            break
    torus_sample_idx = None
    for i, sample in enumerate(balanced_samples):
        x, edge_index, y = flatten_to_triplet(sample)
        if y.item() == torus_idx:
            torus_sample_idx = i
            break
    import random
    non_torus_indices = [i for i, sample in enumerate(balanced_samples) if flatten_to_triplet(sample)[2].item() != torus_idx]
    # Fallback: if no torus sample found, just pick 4 random samples
    if torus_sample_idx is not None and len(non_torus_indices) >= 3:
        chosen_indices = random.sample(non_torus_indices, 3)
        display_indices = [torus_sample_idx] + chosen_indices
    else:
        display_indices = random.sample(range(len(balanced_samples)), 4)

    # Build a batch for display
    display_xs, display_eis, display_ys = [], [], []
    for idx in display_indices:
        x, edge_index, y = flatten_to_triplet(balanced_samples[idx])
        display_xs.append(x)
        display_eis.append(edge_index)
        display_ys.append(y)
    # Collate
    x, edge_index, batch_vec, y = collate_fn(list(zip(display_xs, display_eis, display_ys)))
    x, edge_index, batch_vec = x.to(device), edge_index.to(device), batch_vec.to(device)
    with torch.no_grad():
        logits, latents = model(x, edge_index, batch_vec)
        pred = logits.argmax(dim=1).cpu().numpy()
        latents = latents.cpu().numpy()
    y = y.squeeze().cpu().numpy()

    # Plot each shape in the batch
    fig = plt.figure(figsize=(12, 3))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        pts = x[batch_vec==i].cpu().numpy()
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=2)
        ax.set_title(f"GT: {class_names[y[i]]}\nPred: {class_names[pred[i]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()



    # Cosine similarity and scatter plot: use the balanced set, always include torus
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    import random
    n_classes = 10
    n_per_class = 10
    # Collect latent vectors for all classes from balanced_samples
    all_class_latents = {}
    for sample in balanced_samples:
        x, edge_index, y = flatten_to_triplet(sample)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            _, latent = model(x, edge_index, batch_vec)
        latent = latent.cpu().numpy().flatten()  # Ensure 1D
        label = y.item() if isinstance(y, torch.Tensor) else int(y)
        if label not in all_class_latents:
            all_class_latents[label] = []
        if len(all_class_latents[label]) < n_per_class:
            all_class_latents[label].append(latent)

    # Always include torus in the selected classes
    torus_idx = None
    for i, name in enumerate(class_names):
        if name == 'torus':
            torus_idx = i
            break
    eligible = [k for k, v in all_class_latents.items() if len(v) == n_per_class]
    # Always include torus and 9 others
    selected_classes = [torus_idx] if torus_idx is not None and torus_idx in eligible else []
    others = [k for k in eligible if k != torus_idx]
    selected_classes += random.sample(others, n_classes - len(selected_classes))
    # Remove any None values (shouldn't happen, but for safety)
    selected_classes = [cls for cls in selected_classes if cls is not None]
    # Debug: print selected class indices and names
    print("Selected class indices:", selected_classes)
    print("Selected class names:", [class_names[cls] for cls in selected_classes])
    if torus_idx is not None:
        print(f"Torus index: {torus_idx}, In selected: {torus_idx in selected_classes}")

    # Prepare data for cosine similarity and scatter plot
    class_means = np.stack([np.mean(all_class_latents[cls], axis=0) for cls in selected_classes])
    sim = cosine_similarity(class_means)
    print("Cosine similarity matrix between class means (10 random classes, always including torus):")
    print(sim)
    # Visualize similarity matrix
    plt.figure(figsize=(6,6))
    plt.imshow(sim, cmap='viridis')
    plt.colorbar()
    plt.title('Cosine Similarity of Class Means (10 random classes, torus included)')
    plt.xticks(range(len(selected_classes)), [class_names[cls] for cls in selected_classes], rotation=90)
    plt.yticks(range(len(selected_classes)), [class_names[cls] for cls in selected_classes])
    plt.tight_layout()
    plt.show()

    # --- Scatter plot of latent representations (PCA to 2D, same 10 random classes, 10 examples each) ---
    all_latents, all_labels = [], []
    for cls in selected_classes:
        all_latents.extend(all_class_latents[cls])
        all_labels.extend([cls]*n_per_class)
    all_latents = np.stack(all_latents)
    all_labels = np.array(all_labels)
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)
    plt.figure(figsize=(6,6))
    for cls in selected_classes:
        idx = all_labels == cls
        plt.scatter(latents_2d[idx,0], latents_2d[idx,1], label=class_names[cls], s=10)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('PCA of Latent Representations (10 random classes, torus included)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()
