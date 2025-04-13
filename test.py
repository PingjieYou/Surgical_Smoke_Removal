import os
import torch
import numpy as np
from collections import OrderedDict

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, Visualizer
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # model.netG_A.load_state_dict(torch.load('./checkpoints/' + opt.name + '/best_val/netG_A_0.pth'))

    visualizer = Visualizer(opt)

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    mtpfm_val_list = []
    mse_total = 0
    ssim_total = 0
    psnr_total = 0
    count = 0
    mse_list = []
    ssim_list = []
    psnr_list = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        count += 1

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        val_visuals = model.get_current_visuals()
        mtpfm_val_od = visualizer.cal_current_pfm('1', val_visuals)
        mse_total += mtpfm_val_od['MSE']
        psnr_total += mtpfm_val_od['PSNR']
        ssim_total += mtpfm_val_od['SSIM']

        mse_list.append(mtpfm_val_od['MSE'])
        ssim_list.append(mtpfm_val_od['SSIM'])
        psnr_list.append(mtpfm_val_od['PSNR'])
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    mse_avg = mse_total / count
    ssim_avg = ssim_total / count
    psnr_avg = psnr_total / count
    #
    # mtpfm_val = OrderedDict()
    # mtpfm_val['MSE'] = mse_avg
    # mtpfm_val['SSIM'] = ssim_avg
    # mtpfm_val['PSNR'] = psnr_avg

    # mse_avg = np.mean(mse_list)
    # ssim_avg = np.mean(ssim_list)
    # psnr_avg = np.mean(psnr_list)

    mse_std = np.std(mse_list)
    ssim_std = np.std(ssim_list)
    psnr_std = np.std(psnr_list)

    print("MSE: ", mse_avg, "+-", mse_std)
    print("SSIM: ", ssim_avg, "+-", ssim_std)
    print("PSNR: ", psnr_avg, "+-", psnr_std)

    webpage.save()  # save the HTML

    with open("test_log.txt", "a", encoding="utf-8") as file:
        file.write(opt.name+", "+"MSE: "+str(mse_avg)+"+-"+str(mse_std)+", SSIM: "+str(ssim_avg)+"+-"+str(ssim_std)+", PSNR: "+str(psnr_avg)+"+-"+str(psnr_std)+"\n")