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
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

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

    device = torch.device('cuda:{}'.format(rank % 2))
    torch.cuda.set_device(device)

    # Separation from two images
    # Optimial paramas: (blur) gamma = 0.4
    #                   (net) gamma = 0.04 - 0.1
    #                   lr = 1e-3
    #                   gamma_reg = 0.5
    #                   reg_threshold = 100 iters

    gamma_ls = ['4e-2', '1e-1', '2e-1', '4e-1']

    for gamma in gamma_ls[rank:len(gamma_ls):n_ranks]:
        for i_repeat in range(0, 20):
            t = TwoImagesSeparation('input1', 'input2', input1, input2, num_iter=10000,
                                    output_folder='auni_dist/output_multislice_constant_alpha_net_{}_rep{}/'.format(gamma, i_repeat), learning_rate=1e-3,
                                    input_type='noise', gamma_excl=float(gamma), gamma_reg=0.5, blur=True, blur_type='net', device=device)
            t.optimize()
            t.finalize()
