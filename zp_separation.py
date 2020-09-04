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

    input1 = dxchange.read_tiff('data/zp_0.tiff')
    input2 = dxchange.read_tiff('data/zp_1.tiff')
    img = np.stack([input1, input2], axis=2)
    img = img[169:617, 169:617, :]
    img = np.transpose(img, (2, 0, 1))
    img = normalize(img)

    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(img[0])
    # axes[1].imshow(img[1])
    # plt.show()

    input1 = img[0].reshape([1, *img[0].shape])
    input2 = img[1].reshape([1, *img[1].shape])

    device = torch.device('cuda:1')
    torch.cuda.set_device(device)

    # device = None
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Separation from two images
    for i_repeat in range(0, 2):
        t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=14000,
                                output_folder='output_multislice_constant_alpha_blur_4e-1_rep{}_dbl/'.format(i_repeat), learning_rate=1e-3,
                                input_type='noise', gamma_excl=4e-1, gamma_reg=0.5, blur=True, blur_type='filter', device=device)
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
