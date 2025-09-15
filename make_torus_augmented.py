import numpy as np

# Load original torus
pts = np.load('pointcloud_shapes_v1/torus.npy')

# Generate 10 augmented torus samples
for i in range(10):
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0, 0, 1]])
    scale = np.random.uniform(0.8, 1.2)
    pts_aug = pts @ rot.T * scale
    np.save(f'pointcloud_shapes_v1/torus_aug_{i}.npy', pts_aug)
print('10 augmented torus samples saved as torus_aug_*.npy')
