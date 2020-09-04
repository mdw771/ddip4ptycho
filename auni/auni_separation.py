from net import skip
from net.losses import ExclusionLoss, plot_image_grid, StdLoss
from net.noise import get_noise
from utils.image_io import *
from skimage.measure import compare_psnr
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
import dxchange
import matplotlib
matplotlib.use('TkAgg')
import os
from crosstalk_separation import *


if __name__ == "__main__":

    img = dxchange.read_tiff('data/au_ni.tiff')
    img = img[114:114+272, 114:-114, :]
    img = np.transpose(img, (2, 0, 1))
    img = normalize(img)

    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(img[0])
    # axes[1].imshow(img[1])
    # plt.show()

    input1 = img[0].reshape([1, *img[0].shape])
    input2 = img[1].reshape([1, *img[1].shape])

    # Separation from two images
    t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=7000)
    t.optimize()
    t.finalize()

    # # Separation of textures
    # s = Separation('textures', input2)
    # s.optimize()
    # s.finalize()
