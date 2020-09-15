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

class Separation(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter=8000, learning_rate=1e-3,
                 original_reflection=None, original_transmission=None, gamma_excl=10, input_type='noise',
                 output_folder='output/'):
        self.image = image
        self.plot_during_training = plot_during_training
        # self.ratio = ratio
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        # self.ratio_net = None
        self.parameters = None
        self.learning_rate = learning_rate
        self.reflection_net_inputs = None
        self.transmission_net_inputs = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.reflection_out = None
        self.transmission_out = None
        self.input_depth = 1
        self.current_result = None
        self.best_result = None
        self.gamma_excl = gamma_excl
        self.output_folder = output_folder
        self.input_type = input_type
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images = create_augmentations(self.image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_inputs(self):
        data_type = torch.cuda.FloatTensor
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                                  self.input_type,
                                                  (self.images_torch[0].shape[2],
                                                   self.images_torch[0].shape[3])).type(data_type).detach())
        self.reflection_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in create_augmentations(origin_noise)]
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             self.input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.transmission_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                      create_augmentations(origin_noise)]

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.mse_loss = nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        # return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(step)
        if step == self.num_iter - 1:
            aug = 0
        reflection_net_input = self.reflection_net_inputs[aug] + (self.reflection_net_inputs[aug].clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_inputs[aug] + (self.transmission_net_inputs[aug].clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)

        self.transmission_out = self.transmission_net(transmission_net_input)

        self.total_loss = self.mse_loss(self.reflection_out + self.transmission_out, self.images_torch[aug])
        self.recon_loss = self.total_loss.item()
        self.exclusion = self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss += self.gamma_excl * self.exclusion
        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.num_iter - 1 or step % 8 == 0:
            reflection_out_np = torch_to_np(self.reflection_out)
            transmission_out_np = torch_to_np(self.transmission_out)
            sum = reflection_out_np + transmission_out_np
            psnr = compare_psnr(self.images[0],  reflection_out_np + transmission_out_np)
            self.psnrs.append(psnr)
            self.current_result = SeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                   psnr=psnr, sum=sum)
            if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Fidelity {:5f} Exclusion {:5f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                                    self.recon_loss,
                                                                                    self.exclusion.item()))
        if step % self.show_every == self.show_every - 1:
            plot_image_grid("left_right_{}".format(step), [self.current_result.reflection, self.current_result.transmission], output_path=self.output_folder)
            save_tiff('reflection_{}.tiff'.format(step), np.squeeze(self.current_result.reflection), output_path=self.output_folder)
            save_tiff('transmission_{}.tiff'.format(step), np.squeeze(self.current_result.transmission), output_path=self.output_folder)
            save_tiff('sum_{}.tiff'.format(step), np.squeeze(self.current_result.sum), output_path=self.output_folder)

    def _plot_distance_map(self):
        calculated_left = self.best_result.reflection
        calculated_right = self.best_result.transmission
        # this is to left for reason
        # print(distance_to_left.shape)
        pass

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs, output_path=self.output_folder)
        save_tiff(self.image_name + "_reflection", self.best_result.reflection, output_path=self.output_folder)
        save_tiff(self.image_name + "_transmission", self.best_result.transmission, output_path=self.output_folder)
        save_tiff(self.image_name + "_original", self.images[0], output_path=self.output_folder)

def normalize(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - img.min()) / (img.max() - img.min())
    return img

SeparationResult = namedtuple("SeparationResult", ['reflection', 'transmission', 'psnr', 'sum'])