
import torch
import numpy as np
from simple_gat_pointcloud import SimpleNN


class CosineSimilarity:
    def __init__(self, device=None, shape = 'pointcloud_shapes_v1/torus.npy'):
        state_dict = torch.load("simple_gat_model.pth", map_location=device)
        n_classes = len(state_dict[list(state_dict.keys())[-1]])
        model = SimpleNN(num_classes=n_classes).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        shape_pts = np.load(shape) 
        self.shape_x = torch.tensor(shape_pts, dtype=torch.float, device=device)
        self.model = model
        self.device = device

    def cosine_similarity_to_shape(self, x, edge_index,):
        """
        Given a model, a single pointcloud (x, edge_index), computes the cosine similarity
        between the input's latent and the latent of the canonical shape from shape.npy.
        Returns a torch scalar suitable for .backward().
        - x: torch.Tensor [N, 3] (pointcloud)
        - edge_index: torch.LongTensor [2, E] (graph edges)
        - shape: path to shape.npy
        """
        # Load shape pointcloud
        # Build k-NN graph for shape
        dists = torch.cdist(self.shape_x, self.shape_x)
        k = 16
        idxs = dists.topk(k+1, largest=False).indices[:,1:]
        src = torch.arange(self.shape_x.size(0), device=self.device).unsqueeze(1).repeat(1, k).flatten()
        dst = idxs.flatten()
        shape_edge_index = torch.stack([src, dst], dim=0)
        shape_batch = torch.zeros(self.shape_x.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, shape_latent = self.model(self.shape_x, shape_edge_index, shape_batch)
        shape_latent = shape_latent.squeeze()
        # Compute latent for input pointcloud (requires grad)
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        _, latent = self.model(x, edge_index, batch)
        latent = latent.squeeze()
        # Cosine similarity (differentiable)
        sim = torch.dot(latent, shape_latent) / (torch.linalg.norm(latent) * torch.linalg.norm(shape_latent) + 1e-8)
        return sim

