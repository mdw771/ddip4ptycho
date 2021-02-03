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
import time
from torchviz import make_dot

TwoImagesSeparationResult = namedtuple("TwoImagesSeparationResult",
                                       ["reflection", "transmission", "psnr", "alpha1", "alpha2", "sum1", "sum2", "kernel1", "kernel2"])

class TwoImagesSeparation(object):
    def __init__(self, image1_name, image2_name, image1, image2, init=None, plot_during_training=True, show_every=500, num_iter=4000,
                 original_reflection=None, learning_rate=1e-3, original_transmission=None, blur=True, blur_type='filter', constant_alpha=True, gamma_excl=10, gamma_reg=0.5,
                 output_folder='output/', input_type='noise', device=torch.device('cuda:0')):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.learning_rate = learning_rate
        self.input_depth = 1
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.recon_loss = None
        self.reg_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self.constant_alpha = constant_alpha
        self.gamma_excl = gamma_excl
        self.gamma_reg = gamma_reg
        self.output_folder = output_folder
        self.pred1 = None
        self.pred2 = None
        self.device = device
        self.input_type = input_type
        self.low_pass_kernel_1 = None
        self.low_pass_kernel_2 = None
        self.filter_net1 = None
        self.filter_net2 = None
        self.dtype = torch.cuda.FloatTensor if (self.device is not None) else torch.FloatTensor
        self.dtype_str = 'float64' if self.dtype == torch.cuda.DoubleTensor else 'float32'
        if init is not None:
            self.init1 = init[0]
            self.init2 = init[1]
            input_type = 'supplied'
        else:
            self.init1 = None
            self.init2 = None
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if blur:
            if blur_type == 'filter':
                k_delta = np.zeros([1, 1, 7, 7], dtype=self.dtype_str)
                k_delta[0, 0, k_delta.shape[2] // 2, k_delta.shape[3] // 2] = 1.
                k_hp = np.array([[0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 1, -4, 1, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0]], dtype=self.dtype_str)
                k_hp = k_hp.reshape([1, 1, *k_hp.shape])
                k_lp = np.full([1, 1, 7, 7], 1. / 49, dtype=self.dtype_str)
                self.low_pass_kernel_1 = torch.tensor(k_lp, requires_grad=True, device=self.device)
                self.low_pass_kernel_2 = torch.tensor(k_hp, requires_grad=True, device=self.device)
            elif blur_type == 'net':
                self.filter_net1 = skip(self.input_depth, self.input_depth, [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                        upsample_mode='bilinear',
                                        filter_size_down=5,
                                        filter_size_up=5,
                                        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU'
                                        ).type(self.dtype)
                self.filter_net2 = skip(self.input_depth, self.input_depth, [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                        upsample_mode='bilinear',
                                        filter_size_down=5,
                                        filter_size_up=5,
                                        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU'
                                        ).type(self.dtype)
                # print(self.filter_net1)
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.image1_torch = np_to_torch(self.image1).type(self.dtype)
        self.image2_torch = np_to_torch(self.image2).type(self.dtype)

    def _init_inputs(self):
        data_type = self.dtype
        if self.input_type == 'supplied':
            self.reflection_net_input = np_to_torch(self.init1).type(self.dtype)
            self.transmission_net_input = np_to_torch(self.init2).type(self.dtype)
            self.alpha_net_input = get_noise(1, 'noise',
                                             (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        else:
            self.reflection_net_input = get_noise(self.input_depth, self.input_type,
                                                  (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
            self.alpha_net_input = get_noise(2, self.input_type,
                                             (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
            self.transmission_net_input = get_noise(self.input_depth, self.input_type,
                                                  (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]
        self.parameters += [p for p in self.alpha.parameters()]
        if self.low_pass_kernel_1 is not None:
            self.parameters += [self.low_pass_kernel_1, self.low_pass_kernel_2]
        elif self.filter_net1 is not None:
            self.parameters += [p for p in self.filter_net1.parameters()]
            self.parameters += [p for p in self.filter_net2.parameters()]

    def _init_nets(self):
        data_type = self.dtype
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        # print(reflection_net)

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)
        alpha_net = skip(
            2, 2,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha = alpha_net.type(data_type)

    def _init_losses(self):
        data_type = self.dtype
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss(dtype=data_type).type(data_type)
        self.blur_loss = StdLoss(dtype=data_type).type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.9)
        for j in range(self.num_iter):
            # if j == 1: t0 = time.time()
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()
            scheduler.step()
            # if j > 1: print((time.time() - t0) / j)
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])

    def _optimization_closure(self, step):
        reg_noise_std = 0
        reflection_net_input = self.reflection_net_input + (self.reflection_net_input.clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_input + \
                                 (self.transmission_net_input.clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)
        self.transmission_out = self.transmission_net(transmission_net_input)
        # make_dot(self.transmission_out).render('model')
        # raise
        alpha_net_input = self.alpha_net_input + (self.alpha_net_input.clone().normal_() * reg_noise_std)
        if self.constant_alpha:
            self.current_alpha = self.alpha(alpha_net_input)
            self.current_alpha1 = self.current_alpha[:, 0:1,
                                 self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                                 self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05
            self.current_alpha2 = self.current_alpha[:, 1:2,
                                 self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                                 self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05
        else:
            self.current_alpha = self.alpha(alpha_net_input)
            self.current_alpha1 = self.current_alpha[:, 0:1, :, :]
            self.current_alpha2 = self.current_alpha[:, 1:2, :, :]

        # print(reflection_net_input.shape, self.reflection_out.shape)
        # print(transmission_net_input.shape, self.transmission_out.shape)
        # print(self.current_alpha1.shape, alpha_net_input.shape)

        if self.low_pass_kernel_1 is not None:
            self.pred1 = self.current_alpha1 * self.reflection_out + \
                         (1 - self.current_alpha1) * torch.nn.functional.conv2d(self.transmission_out, self.low_pass_kernel_1, padding=self.low_pass_kernel_1.shape[-1] // 2)
            self.pred2 = self.current_alpha2 * torch.nn.functional.conv2d(self.reflection_out, self.low_pass_kernel_2, padding=self.low_pass_kernel_2.shape[-1] // 2) + \
                         (1 - self.current_alpha2) * self.transmission_out
        elif self.filter_net1 is not None:
            self.pred1 = self.current_alpha1 * self.reflection_out + (1 - self.current_alpha1) * self.filter_net1(self.transmission_out)
            self.pred2 = self.current_alpha2 * self.filter_net2(self.reflection_out) + (1 - self.current_alpha2) * self.transmission_out
        else:
            self.pred1 = self.current_alpha1 * self.reflection_out + (1 - self.current_alpha1) * self.transmission_out
            self.pred2 = self.current_alpha2 * self.reflection_out + (1 - self.current_alpha2) * self.transmission_out
        self.total_loss = self.mse_loss(self.pred1, self.image1_torch)
        self.total_loss += self.mse_loss(self.pred2, self.image2_torch)
        self.recon_loss = self.total_loss.item()
        self.exclusion = self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss += self.gamma_excl * self.exclusion
        # self.total_loss += self.blur_loss(self.current_alpha2) + self.blur_loss(self.current_alpha1)

        if step < 100:
            self.reg_loss = self.gamma_reg * self.mse_loss(self.current_alpha1,
                                            torch.tensor([[[[0.5]]]]).type(self.dtype))
            self.reg_loss += self.gamma_reg * self.mse_loss(self.current_alpha2,
                                                  torch.tensor([[[[0.5]]]]).type(self.dtype))
            self.total_loss += self.reg_loss
        # self.reg_loss = self.gamma_reg * (1. / torch.abs(self.current_alpha1 - 0)) * (1. / torch.abs(self.current_alpha1 - 1))
        # self.reg_loss += self.gamma_reg * (1. / torch.abs(self.current_alpha2 - 0)) * (1. / torch.abs(self.current_alpha2 - 1))
        # self.reg_loss = torch.mean(self.reg_loss)
        # self.total_loss += self.reg_loss

        self.total_loss.backward()

    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
        transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
        # print(reflection_out_np.shape)
        alpha1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        alpha2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)
        v = alpha1 * reflection_out_np + (1 - alpha1) * transmission_out_np
        # print(v.shape, self.image2.shape)
        psnr1 = compare_psnr(self.image1, v)
        psnr2 = compare_psnr(self.image2, alpha2 * reflection_out_np + (1 - alpha2) * transmission_out_np)
        sum1 = torch_to_np(self.pred1)
        sum2 = torch_to_np(self.pred2)
        if self.low_pass_kernel_1 is not None:
            kernel1 = torch_to_np(self.low_pass_kernel_1)
            kernel2 = torch_to_np(self.low_pass_kernel_2)
        else:
            kernel1 = None
            kernel2 = None
        self.psnrs.append(psnr1+psnr2)
        self.current_result = TwoImagesSeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                        psnr=psnr1, alpha1=alpha1, alpha2=alpha2, sum1=sum1, sum2=sum2,
                                                        kernel1=kernel1, kernel2=kernel2)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Fidelity {:5f} Exclusion {:5f} Reg {:5f} alpha: {:f}|{:f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                                    self.recon_loss,
                                                                                    self.exclusion.item(),
                                                                                    self.reg_loss.item(),
                                                                                    self.current_result.alpha1.item(),
                                                                                    self.current_result.alpha2.item()))

        # print(torch.cuda.memory_allocated()/1024**2, torch.cuda.memory_cached()/1024**2, torch.cuda.max_memory_allocated()/1024**2, torch.cuda.max_memory_cached()/1024**2)
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(step),
                            [self.current_result.reflection, self.current_result.transmission], output_path=self.output_folder)
            # plot_image_grid("learned_mask_{}".format(step),
            #                 [self.current_result.alpha1, self.current_result.alpha2])
            save_tiff("sum1_{}".format(step), np.squeeze(self.current_result.sum1), output_path=self.output_folder)
            save_tiff("sum2_{}".format(step), np.squeeze(self.current_result.sum2), output_path=self.output_folder)
            save_tiff("reflection_{}".format(step), np.squeeze(self.current_result.reflection), output_path=self.output_folder)
            save_tiff("transmission_{}".format(step), np.squeeze(self.current_result.transmission), output_path=self.output_folder)
            if not self.constant_alpha:
                save_tiff("alpha1_{}".format(step), np.squeeze(self.current_result.alpha1),
                          output_path=self.output_folder)
                save_tiff("alpha2_{}".format(step), np.squeeze(self.current_result.alpha2),
                          output_path=self.output_folder)
            if self.low_pass_kernel_1 is not None:
                save_tiff("kernel1_{}".format(step), np.squeeze(self.current_result.kernel1),
                          output_path=self.output_folder)
                save_tiff("kernel2_{}".format(step), np.squeeze(self.current_result.kernel2),
                          output_path=self.output_folder)
            np.savetxt(os.path.join(self.output_folder, 'alpha_{}'.format(step)),
                       [np.squeeze(self.current_result.alpha1), np.squeeze(self.current_result.alpha2)], fmt='%.3f')


    def finalize(self):
        save_graph("fin_psnr", self.psnrs, output_path=self.output_folder)
        save_tiff("fin_reflection", self.best_result.reflection, output_path=self.output_folder)
        save_tiff("fin_transmission", self.best_result.transmission, output_path=self.output_folder)
        save_tiff(self.image1_name + "_original", self.image1, output_path=self.output_folder)
        save_tiff(self.image2_name + "_original", self.image2, output_path=self.output_folder)
        if not self.constant_alpha:
            save_tiff('fin_alpha1', self.best_result.alpha1, output_path=self.output_folder)
            save_tiff('fin_alpha2', self.best_result.alpha2, output_path=self.output_folder)


class ImageSeriesSeparation(object):
    def __init__(self, image1_name, image2_name, image1, image2, init=None, plot_during_training=True, show_every=500,
                 num_iter=4000,
                 original_reflection=None, learning_rate=1e-3, original_transmission=None,
                 constant_alpha=True, gamma_excl=10, gamma_reg=0.5,
                 output_folder='output/', input_type='noise', device=torch.device('cuda:0'),
                 n_images=2):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.num_iter = num_iter
        self.loss_function = None
        self.parameters = None
        self.learning_rate = learning_rate
        self.input_depth = 1
        self.n_images = n_images
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.recon_loss = None
        self.reg_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self.constant_alpha = constant_alpha
        self.gamma_excl = gamma_excl
        self.gamma_reg = gamma_reg
        self.output_folder = output_folder
        self.pred1 = None
        self.pred2 = None
        self.device = device
        self.input_type = input_type
        self.dtype = torch.cuda.FloatTensor if (self.device is not None) else torch.FloatTensor
        self.dtype_str = 'float64' if self.dtype == torch.cuda.DoubleTensor else 'float32'
        if init is not None:
            self.init1 = init[0]
            self.init2 = init[1]
            input_type = 'supplied'
        else:
            self.init1 = None
            self.init2 = None
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
        self.image1_torch = np_to_torch(self.image1).type(self.dtype)
        self.image2_torch = np_to_torch(self.image2).type(self.dtype)

    def _init_inputs(self):
        data_type = self.dtype
        if self.input_type == 'supplied':
            self.reflection_net_input = np_to_torch(self.init1).type(self.dtype)
            self.transmission_net_input = np_to_torch(self.init2).type(self.dtype)
            self.alpha_net1_input = get_noise(self.input_depth, 'noise',
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
            self.alpha_net2_input = get_noise(self.input_depth, 'noise',
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        else:
            self.reflection_net_input = get_noise(self.input_depth, self.input_type,
                                                 (self.image1.shape[1], self.image1.shape[2]), n_images=self.n_images).type(data_type).detach()
            self.alpha_net1_input = get_noise(self.input_depth, self.input_type,
                                             (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
            self.alpha_net2_input = get_noise(self.input_depth, self.input_type,
                                             (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
            self.transmission_net_input = get_noise(self.input_depth, self.input_type,
                                                   (self.image1.shape[1], self.image1.shape[2]), n_images=self.n_images).type(
                data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]
        self.parameters += [p for p in self.alpha1.parameters()]
        self.parameters += [p for p in self.alpha2.parameters()]

    def _init_nets(self):
        data_type = self.dtype
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)
        alpha_net1 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha1 = alpha_net1.type(data_type)

        alpha_net2 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha2 = alpha_net2.type(data_type)

    def _init_losses(self):
        data_type = self.dtype
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss(dtype=data_type).type(data_type)
        self.blur_loss = StdLoss(dtype=data_type).type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.9)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()
            scheduler.step()
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])

    def _optimization_closure(self, step):
        reg_noise_std = 0
        reflection_net_input = self.reflection_net_input + (self.reflection_net_input.clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_input + \
                                 (self.transmission_net_input.clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)
        self.transmission_out = self.transmission_net(transmission_net_input)
        alpha_net_input = self.alpha_net1_input + (self.alpha_net1_input.clone().normal_() * reg_noise_std)
        if self.constant_alpha:
            self.current_alpha1 = self.alpha1(alpha_net_input)[:, :,
                                  self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                                  self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05
        else:
            self.current_alpha1 = self.alpha1(alpha_net_input)

        # print(reflection_net_input.shape, self.reflection_out.shape)
        # print(transmission_net_input.shape, self.transmission_out.shape)
        # print(self.current_alpha1.shape, alpha_net_input.shape)

        alpha_net_input = self.alpha_net2_input + (self.alpha_net2_input.clone().normal_() * reg_noise_std)
        if self.constant_alpha:
            self.current_alpha2 = self.alpha2(alpha_net_input)[:, :,
                                  self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                                  self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05
        else:
            self.current_alpha2 = self.alpha2(alpha_net_input)

        self.pred1 = self.current_alpha1 * self.reflection_out[0:1] + (1 - self.current_alpha1) * self.transmission_out[1:2]
        self.pred2 = self.current_alpha2 * self.reflection_out[1:2] + (1 - self.current_alpha2) * self.transmission_out[0:1]
        self.total_loss = self.mse_loss(self.pred1, self.image1_torch)
        self.total_loss += self.mse_loss(self.pred2, self.image2_torch)
        self.recon_loss = self.total_loss.item()
        self.exclusion = self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss += self.gamma_excl * self.exclusion
        # self.total_loss += self.blur_loss(self.current_alpha2) + self.blur_loss(self.current_alpha1)

        if step < 100:
            self.reg_loss = self.gamma_reg * self.mse_loss(self.current_alpha1,
                                                           torch.tensor([[[[0.5]]]]).type(self.dtype))
            self.reg_loss += self.gamma_reg * self.mse_loss(self.current_alpha2,
                                                            torch.tensor([[[[0.5]]]]).type(self.dtype))
            self.total_loss += self.reg_loss
        # self.reg_loss = self.gamma_reg * (1. / torch.abs(self.current_alpha1 - 0)) * (1. / torch.abs(self.current_alpha1 - 1))
        # self.reg_loss += self.gamma_reg * (1. / torch.abs(self.current_alpha2 - 0)) * (1. / torch.abs(self.current_alpha2 - 1))
        # self.reg_loss = torch.mean(self.reg_loss)
        # self.total_loss += self.reg_loss

        self.total_loss.backward()

    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        reflection_out_np = np.clip(torch_to_np(self.reflection_out, first_only=False), 0, 1)
        transmission_out_np = np.clip(torch_to_np(self.transmission_out, first_only=False), 0, 1)
        # print(reflection_out_np.shape)
        alpha1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        alpha2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)
        v = alpha1 * reflection_out_np[0] + (1 - alpha1) * transmission_out_np[1]
        # print(v.shape, self.image2.shape)
        psnr1 = compare_psnr(self.image1, v)
        psnr2 = compare_psnr(self.image2, alpha2 * reflection_out_np[1] + (1 - alpha2) * transmission_out_np[0])
        sum1 = torch_to_np(self.pred1)
        sum2 = torch_to_np(self.pred2)
        kernel1 = None
        kernel2 = None
        self.psnrs.append(psnr1 + psnr2)
        self.current_result = TwoImagesSeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                        psnr=psnr1, alpha1=alpha1, alpha2=alpha2, sum1=sum1, sum2=sum2,
                                                        kernel1=kernel1, kernel2=kernel2)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Fidelity {:5f} Exclusion {:5f} Reg {:5f} alpha: {:f}|{:f}'.format(step,
                                                                                                               self.total_loss.item(),
                                                                                                               self.recon_loss,
                                                                                                               self.exclusion.item(),
                                                                                                               self.reg_loss.item(),
                                                                                                               self.current_result.alpha1.item(),
                                                                                                               self.current_result.alpha2.item()))
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            # plot_image_grid("reflection_transmission_{}".format(step),
            #                 [self.current_result.reflection, self.current_result.transmission],
            #                 output_path=self.output_folder)
            # plot_image_grid("learned_mask_{}".format(step),
            #                 [self.current_result.alpha1, self.current_result.alpha2])
            save_tiff("sum1_{}".format(step), np.squeeze(self.current_result.sum1), output_path=self.output_folder)
            save_tiff("sum2_{}".format(step), np.squeeze(self.current_result.sum2), output_path=self.output_folder)
            save_tiff("reflection_{}".format(step), np.squeeze(self.current_result.reflection),
                      output_path=self.output_folder)
            save_tiff("transmission_{}".format(step), np.squeeze(self.current_result.transmission),
                      output_path=self.output_folder)
            if not self.constant_alpha:
                save_tiff("alpha1_{}".format(step), np.squeeze(self.current_result.alpha1),
                          output_path=self.output_folder)
                save_tiff("alpha2_{}".format(step), np.squeeze(self.current_result.alpha2),
                          output_path=self.output_folder)


    def finalize(self):
        save_graph("fin_psnr", self.psnrs, output_path=self.output_folder)
        save_tiff("fin_reflection", self.best_result.reflection, output_path=self.output_folder)
        save_tiff("fin_transmission", self.best_result.transmission, output_path=self.output_folder)
        save_tiff(self.image1_name + "_original", self.image1, output_path=self.output_folder)
        save_tiff(self.image2_name + "_original", self.image2, output_path=self.output_folder)
        if not self.constant_alpha:
            save_tiff('fin_alpha1', self.best_result.alpha1, output_path=self.output_folder)
            save_tiff('fin_alpha2', self.best_result.alpha2, output_path=self.output_folder)


def normalize(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - img.min()) / (img.max() - img.min())
    return img
