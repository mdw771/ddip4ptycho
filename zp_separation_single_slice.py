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
from single_slice_separation import *


if __name__ == "__main__":

    input = np.squeeze(dxchange.read_tiff('data/zp_single.tiff'))
    img = input[169:617, 169:617]
    img = normalize(img)

    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(img[0])
    # axes[1].imshow(img[1])
    # plt.show()

    input = img[None, :, :]

    device = torch.device('cuda:1')
    torch.cuda.set_device(device)

    # device = None
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Separation from two images
    # Optimial paramas: (blur) gamma = 0.4
    #                   (net) gamma = 0.04 - 0.1
    #                   lr = 1e-3
    #                   gamma_reg = 0.5
    #                   reg_threshold = 100 iters
    for i_repeat in range(0, 1):
        t = Separation('input1', input, num_iter=10000,
                       output_folder='zp_single/output_multislice_1e-1_rep{}/'.format(i_repeat), learning_rate=1e-4,
                       input_type='noise', gamma_excl=1e-1)
        t.optimize()
        t.finalize()
