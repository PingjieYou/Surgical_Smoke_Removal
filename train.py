import os
import time
import queue
import torch
import random
import numpy as np
from tqdm import tqdm
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict

if __name__ == '__main__':
    opt = TrainOptions().parse()  # 解析参数

    # TODO seed
    if not opt.random_seed:
        torch.manual_seed(2024)
        opt.seed = 2024
    else:
        seed = random.randint(0, 2024)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.seed = seed

    # TODO 创建数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # TODO 创建模型
    model = create_model(opt)
    model.setup(opt)  # 打印网络结构，且加载网络

    # net = [model.netG]
    # print("params of all net: ", sum([sum(p.numel() for p in net_.parameters()) for net_ in net]))
    # TODO 可视化
    visualizer = Visualizer(opt)
    total_iters = 0

    best_ssim = 0
    best_mse = 0
    best_psnr = 0
    avg_D_A_loss = 0
    avg_G_A_loss = 0
    avg_cycle_A_loss = 0
    avg_idt_A_loss = 0
    avg_D_B_loss = 0
    avg_G_B_loss = 0
    avg_cycle_B_loss = 0
    avg_idt_B_loss = 0
    val_best_queue = queue.PriorityQueue(maxsize=5)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        for i, data in tqdm(enumerate(dataset.dataloader), total=len(dataset.dataloader), desc="Train Epoch {}".format(epoch)):
            iter_start_time = time.time()

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # 加载数据
            model.optimize_parameters()  # 更新网络参数
            loss = model.get_current_losses()
            avg_D_A_loss += loss['D_A'] if 'D_A' in loss else 0
            avg_G_A_loss += loss['G_A'] if 'G_A' in loss else 0
            avg_cycle_A_loss += loss['cycle_A'] if 'cycle_A' in loss else 0
            avg_idt_A_loss += loss['idt_A'] if 'idt_A' in loss else 0
            avg_D_B_loss += loss['D_B'] if 'D_B' in loss else 0
            avg_G_B_loss += loss['G_B'] if 'G_B' in loss else 0
            avg_cycle_B_loss += loss['cycle_B'] if 'cycle_B' in loss else 0
            avg_idt_B_loss += loss['idt_B'] if 'idt_B' in loss else 0
            iter_data_time = time.time()

        avg_D_A_loss /= dataset_size
        avg_G_A_loss /= dataset_size
        avg_cycle_A_loss /= dataset_size
        avg_idt_A_loss /= dataset_size
        avg_D_B_loss /= dataset_size
        avg_G_B_loss /= dataset_size
        avg_cycle_B_loss /= dataset_size
        avg_idt_B_loss /= dataset_size

        losses = {'D_A': avg_D_A_loss, 'G_A': avg_G_A_loss, 'cycle_A': avg_cycle_A_loss, 'idt_A': avg_idt_A_loss, }
        model.save_networks('latest')

        # TODO 验证
        val_opt = TestOptions().parse()
        val_opt.phase = 'test'
        val_opt.batch_size = 1

        data_val = create_dataset(val_opt)
        val_dataset_size = len(data_val)

        model_val = create_model(val_opt)  # create a model given opt.model and other options
        model_val.setup(val_opt)  # regular setup: load and print networks; create schedulers

        model_val.eval()
        mtpfm_val_list = []
        mse_total = 0
        ssim_total = 0
        psnr_total = 0
        count = 0
        for i, data in tqdm(enumerate(data_val.dataloader), total=len(data_val.dataloader), desc="Validation Epoch {}".format(epoch)):
            # for i, data in enumerate(data_val):
            count += 1
            model_val.set_input(data)  # unpack data from data loader
            model_val.test()  # run inference
            val_visuals = model_val.get_current_visuals()
            mtpfm_val_od = visualizer.cal_current_pfm(epoch, val_visuals)
            mse_total += mtpfm_val_od['MSE']
            psnr_total += mtpfm_val_od['PSNR']
            ssim_total += mtpfm_val_od['SSIM']

        mse_avg = mse_total / count
        ssim_avg = ssim_total / count
        psnr_avg = psnr_total / count

        mtpfm_val = OrderedDict()
        mtpfm_val['MSE'] = mse_avg
        mtpfm_val['SSIM'] = ssim_avg
        mtpfm_val['PSNR'] = psnr_avg

        visualizer.plot_current_ssim_val(epoch, float(epoch_iter) / dataset_size, mtpfm_val)
        t_comp = (time.time() - epoch_start_time) / opt.batch_size
        t_data = time.time() - epoch_start_time
        visualizer.print_current_val_mtx(epoch, losses, t_comp, t_data, mtpfm_val)

        if mtpfm_val['SSIM'] > best_ssim:
            best_ssim = mtpfm_val['SSIM']
            best_mse = mtpfm_val['MSE']
            best_psnr = mtpfm_val['PSNR']
            model.save_networks('best_ssim')
