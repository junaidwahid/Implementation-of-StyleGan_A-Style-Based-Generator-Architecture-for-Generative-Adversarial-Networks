import numpy as np
import torch
import matplotlib.pyplot as plt

from MicroStyleGan import MicroStyleGANGenerator
from helper_functions import get_truncated_noise, show_tensor_images

plt.rcParams['figure.figsize'] = [15, 15]
z_dim = 128
out_chan = 3
truncation = 0.7
viz_samples = 10

viz_noise = get_truncated_noise(viz_samples, z_dim, truncation) * 10
mu_stylegan = MicroStyleGANGenerator(
    z_dim=z_dim,
    map_hidden_dim=1024,
    w_dim=496,
    in_chan=512,
    out_chan=out_chan,
    kernel_size=3,
    hidden_chan=256
)

mu_stylegan.eval()
images = []
for alpha in np.linspace(0, 1, num=5):
    mu_stylegan.alpha = alpha
    viz_result, _, _ =  mu_stylegan(
        viz_noise,
        return_intermediate=True)
    images += [tensor for tensor in viz_result]
show_tensor_images(torch.stack(images), nrow=viz_samples, num_images=len(images))
mu_stylegan = mu_stylegan.train()
