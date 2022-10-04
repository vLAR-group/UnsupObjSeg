import os
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from utils.misc import named_leaf_modules

img_folder = None


def _unique_filename(fname, extension):
    def exists(fname):
        return os.path.exists(fname + '.' + extension)

    if not exists(fname):
        return fname
    for i in range(10000):
        t = fname + '_' + str(i)
        if not exists(t):
            return t
    raise RuntimeError("too many files ({})".format(i))


def plot_imgs(imgs, name=None, extension='png', colorbar=True, overwrite=False):
    """
    Plots collection of 1-channel images (as 3D tensor, or 4D tensor with size
    1 on the 2nd dimension) and saves it as png. If any image extends beyond
    [0, 1], all are normalized such that the minimum and maximum are 0 and 1.

    If overwrite is False, it automatically appends an integer to the filename
    to make it unique.
    """

    if imgs.dim() == 4 and imgs.size(1) == 1:
        imgs = imgs.squeeze(1)
    if imgs.dim() != 3:
        msg = ("input tensor must be 3D, or 4D with size 1 on the 2nd "
               "dimension, but has shape {}".format(imgs.shape))
        raise RuntimeError(msg)
    if img_folder is None:
        raise RuntimeError("Image folder not set")
    fname = name.replace(' ', '_')
    fname = os.path.join(img_folder, fname)
    if not overwrite:
        fname = _unique_filename(fname, extension)
    fname = fname + '.' + extension
    imgs = imgs.detach().cpu().unsqueeze(1)   # (N, 1, H, W)
    n_imgs = imgs.size(0)
    _, c = balanced_approx_factorization(n_imgs)  # grid arrangement

    # Get minimum and maximum
    low = imgs.min().item()
    high = imgs.max().item()

    # Normalize if images extend beyond the range [0, 1]
    normalize = low < 0. or high > 1.
    if normalize:
        imgs = (imgs - low) / (high - low)

    # Compute pad value, either 0 or 1
    pad_value = img_grid_pad_value(imgs)

    # Images are now in [0, 1], arrange them into a grid as they are
    # Grid has shape (3, grid_h, grid_w) and has values in [0, 1]
    grid = make_grid(imgs, nrow=c, pad_value=pad_value, normalize=False)

    if colorbar:
        # make_grid made it RGB, now take one channel only
        grid = grid[0]

        # Rescale images to original interval (now including the padding)
        if normalize:
            grid = grid * (high - low) + low

        # Save grid of images with colorbar
        vmin = low if normalize else 0.
        vmax = high if normalize else 1.
        plt.imshow(grid, vmin=vmin, vmax=vmax, cmap='gray')
        plt.colorbar()
        plt.title(name)
        plt.savefig(fname)
        plt.close()

    else:
        from PIL import Image

        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        grid = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
        grid = grid.permute(1, 2, 0).numpy()

        # Save image with PIL
        im = Image.fromarray(grid)
        im.save(fname, format=None)


def img_grid_pad_value(imgs, thresh=.2):
    """
    Hack to visualize boundaries between images with torchvision's save_image().
    If the median border value of all images is below the threshold, use white,
    otherwise black (which is the default)
    :param imgs: 4d tensor
    :param thresh: threshold in (0, 1)
    :return: padding value
    """

    assert imgs.dim() == 4
    imgs = imgs.clamp(min=0., max=1.)
    assert 0. < thresh < 1.

    imgs = imgs.mean(1)  # reduce to 1 channel
    h = imgs.size(1)
    w = imgs.size(2)
    borders = list()
    borders.append(imgs[:, 0].flatten())
    borders.append(imgs[:, h - 1].flatten())
    borders.append(imgs[:, 1:h - 1, 0].flatten())
    borders.append(imgs[:, 1:h - 1, w - 1].flatten())
    borders = torch.cat(borders)
    if torch.median(borders) < thresh:
        return 1.0
    return 0.0


def balanced_approx_factorization(x, ratio=1):
    """
    Util to plot images in a grid.

    :param x: number to be approximately factorized
    :param ratio: ratio columns/rows
    :return: rows, columns
    """

    # We want c/r to be approx equal to ratio, and r*c to be approx equal to x
    # ==> r = x/c = x/(ratio*r)
    # ==> r = sqrt(x/ratio) and c = sqrt(x*ratio)
    assert type(x) == int or x.dtype == int
    c = int(np.ceil(np.sqrt(x * ratio)))
    r = int(np.ceil(x / c))
    return r, c


def balanced_factorization(x):
    """
    Util to plot images in a grid.

    :param x: number to be factorized
    """

    assert type(x) == int or x.dtype == int
    a = int(np.floor(np.sqrt(x)))
    while True:
        b = x / a
        if b.is_integer():
            return int(b), a
        a += 1


def clean_axes():
    """
    Clean current axes.
    """
    plt.gca().tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks
        left=False,  # ticks along the left edge are off
        right=False,
        bottom=False,
        top=False,
        labelleft=False,  # labels along the left edge are off
        labelbottom=False)


def _save_activation_hook(ord_dict, name):
    def hook(model, inp, ret):
        if isinstance(ret, tuple):
            try:
                ret = torch.cat(ret, dim=1)
            except TypeError:
                ret = ret[0]
            except RuntimeError as e:
                print("WARNING:", e)
                return
        ord_dict[name] = ret.detach()
    return hook


def set_up_saving_all_activations(model):
    all_activations = OrderedDict()
    for module_name, module in named_leaf_modules(model):
        module.register_forward_hook(_save_activation_hook(all_activations, module_name))
    return all_activations


#########################
# Test

if __name__ == '__main__':

    ### Test plot imgs

    img_folder = ''

    low = -20.
    high = 30.
    imgs = torch.rand(4, 1, 8, 8) * (high - low) + low
    plot_imgs(imgs, name='test_colorbar_false', colorbar=False)
    plot_imgs(imgs, name='test_colorbar_true', colorbar=True)

    low = .7
    high = 2.
    imgs = torch.rand(4, 1, 8, 8) * (high - low) + low
    plot_imgs(imgs, name='test_colorbar_false', colorbar=False)
    plot_imgs(imgs, name='test_colorbar_true', colorbar=True)

    low = .5
    high = .9
    imgs = torch.rand(4, 1, 8, 8) * (high - low) + low
    plot_imgs(imgs, name='test_colorbar_false', colorbar=False)
    plot_imgs(imgs, name='test_colorbar_true', colorbar=True)


    ### Test img grid pad value
    img_grid_pad_value(torch.rand(6, 3, 32, 32), thresh=.3)
