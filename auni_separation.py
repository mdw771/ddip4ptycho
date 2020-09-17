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

    device = torch.device('cuda:0')
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
        t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=10000,
                                output_folder='auni/_output_multislice_constant_alpha_blur_1e-1_rep{}/'.format(i_repeat), learning_rate=1e-3,
                                input_type='noise', gamma_excl=1e-1, gamma_reg=0.5, blur=True, blur_type='filter', device=device)
        t.optimize()
        t.finalize()

    # # Separation from two images
    # init1 = np.squeeze(dxchange.read_tiff('/data/programs/DoubleDIP/output_multislice_constant_alpha_blur_1_rep0/reflection_2499.tiff'))
    # init2 = np.squeeze(dxchange.read_tiff('/data/programs/DoubleDIP/output_multislice_constant_alpha_blur_1_rep0/transmission_2499.tiff'))
    # init1 = init1.reshape([1, *init1.shape])
    # init2 = init2.reshape([1, *init2.shape])
    # for i_repeat in range(10):
    #     t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=30000, init=[init1, init2],
    #                             output_folder='output_multislice_constant_alpha_blur_1_supp_rep{}/'.format(i_repeat),
    #                             input_type='supplied', gamma_excl=1, blur=True, device=device, learning_rate=1e-4)
    #     t.optimize()
    #     t.finalize()

    # t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=10000,
    #                         output_folder='output_multislice_constant_alpha_blur_10_ramp/', input_type='meshgrid',
    #                         gamma_excl=10, blur=True, device=device)
    # t.optimize()
    # t.finalize()
