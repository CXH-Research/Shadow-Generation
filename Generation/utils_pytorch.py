import torch
import torchvision.transforms as TF
import numpy as np

def get_brightness_mask(size, min_val=0.5):
    """Render per-pixel intensity variation mask within [min_val, 1.].

    Args:
      size: A 2D tensor of target mask size.

    Returns:
      A Tensor of shape [H, W, 1] that is generated with perlin noise pattern.
    """
    perlin_map = perlin_collection((size[0], size[1]), [2, 2], 2,
                                   np.random.uniform(0.05, 0.25))
    perlin_map = perlin_map / (1. / (min_val + 1e-6)) + min_val
    perlin_map = torch.minimum(perlin_map, 1.)
    return perlin_map

def perlin_collection(size, reso, octaves, persistence):
    """Generate perlin patterns of varying frequencies.

    Args:
      size: a tuple of the target noise pattern size.
      reso: a tuple that specifies the resolution along lateral and longitudinal.
      octaves: int, number of octaves to use in the perlin model.
      persistence: int, persistence applied to every iteration of the generation.

    Returns:
      a 2D tensor of the perlin noise pattern.
    """
    noise = torch.zeros(size)
    amplitude = 1.0

    for _ in range(octaves):
        noise += amplitude * perlin(size, reso)
        amplitude *= persistence
        reso[0] *= 2
        reso[1] *= 2

    return noise

def perlin(size, reso):
    """Generate a perlin noise pattern, with specified frequency along x and y.

    Theory: https://flafla2.github.io/2014/08/09/perlinnoise.html

    Args:
      size: a tuple of integers of the target shape of the noise pattern.
      reso: reso: a tuple that specifies the resolution along lateral and longitudinal (x and y).

    Returns:
      a 2D tensor of the target size.
    """
    ysample = torch.linspace(0.0, reso[0], size[0])
    xsample = torch.linspace(0.0, reso[1], size[1])
    xygrid = torch.stack(torch.meshgrid(ysample, xsample), 2)
    xygrid = torch.remainder(torch.transpose(xygrid, [1, 0, 2]), 1.0)

    xyfade = (6.0 * xygrid**5) - (15.0 * xygrid**4) + (10.0 * xygrid**3)
    angles = 2.0 * np.pi * np.random.uniform(reso[0] + 1, reso[1] + 1)
    grads = torch.stack([torch.cos(angles), torch.sin(angles)], 2)

    transform = TF.Resize((size[0], size[1]))

    gradone = transform(grads[0:-1, 0:-1])
    gradtwo = transform(grads[1:, 0:-1])
    gradthr = transform(grads[0:-1, 1:])
    gradfou = transform(grads[1:, 1:])

    gradone = torch.sum(
        gradone * torch.stack([xygrid[:, :, 0], xygrid[:, :, 1]], 2), 2)
    gradtwo = torch.sum(
        gradtwo * torch.stack([xygrid[:, :, 0] - 1, xygrid[:, :, 1]], 2), 2)
    gradthr = torch.sum(
        gradthr * torch.stack([xygrid[:, :, 0], xygrid[:, :, 1] - 1], 2), 2)
    gradfou = torch.sum(
        gradfou * torch.stack([xygrid[:, :, 0] - 1, xygrid[:, :, 1] - 1], 2), 2)

    inteone = (gradone * (1.0 - xyfade[:, :, 0])) + (gradtwo * xyfade[:, :, 0])
    intetwo = (gradthr * (1.0 - xyfade[:, :, 0])) + (gradfou * xyfade[:, :, 0])
    intethr = (inteone * (1.0 - xyfade[:, :, 1])) + (intetwo * xyfade[:, :, 1])

    return torch.sqrt(2.0) * intethr
