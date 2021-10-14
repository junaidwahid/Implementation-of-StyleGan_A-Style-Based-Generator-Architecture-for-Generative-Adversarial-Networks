import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()


def get_truncated_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-truncation,truncation, size=(n_samples, z_dim))

    return torch.Tensor(truncated_noise)