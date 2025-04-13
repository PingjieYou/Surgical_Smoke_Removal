import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        """定义训练和测试中的选项"""
        # TODO 数据集参数和gpu
        parser.add_argument('--dataset', default='cholec80')
        parser.add_argument('--random_seed', type=bool, default=True, help='random seed')
        parser.add_argument('--dataroot', default='/home/ypj/dataset/cholec80_desmoking/psv2rs/light', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='CycleGANResNet9Light', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # TODO 模型参数
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--patch_size', type=int, default=4, help='patch size of self-attention')
        parser.add_argument('--window_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # TODO 数据集参数
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # TODO 额外parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # TODO wandb参数
        parser.add_argument('--use_wandb', type=bool, default=False, help='if specified, then init wandb logging')
        parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')

        parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
        parser.add_argument('--use_norm', type=float, default=1, help='L1 loss weight is 10.0')
        parser.add_argument('--syn_norm', action='store_true', help='use synchronize batch normalization')
        parser.add_argument('--embed_dim', type=int, default=64, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        # parser.add_argument('--depths', type=list,default=[2, 2, 2, 2, 2, 2, 2, 2, 2], help='linear/conv token projection')
        # TODO 日志
        parser.add_argument('--output_folder', type=str, default='./output/checkpoints/', help='output folder')
        self.initialized = True
        return parser

    def gather_options(self):
        """
            初始化解析器，添加模型和数据集的参数

        Returns:
            解析器
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 初始化解析器
            parser = self.initialize(parser)  # 添加模型和数据集的参数

        opt, _ = parser.parse_known_args()  # 解析参数
        # TODO 模型参数设置
        model_name = opt.model  # 获取模型的名称
        model_option_setter = models.get_option_setter(model_name)  # 获取模型的参数设置函数
        parser = model_option_setter(parser, self.isTrain)  # 通过模型参数设置函数设置模型参数
        opt, _ = parser.parse_known_args()  # 再次模型参数设置

        # TODO 数据集参数设置
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)  # 获取数据集参数设置函数
        parser = dataset_option_setter(parser, self.isTrain)  # 通过数据集参数设置函数设置数据集参数

        # TODO 保存解析器并返回
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """
        打印参数
        Args:
            opt: 解析器选项

        Returns:

        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # TODO 保存参数到checkpoints文件夹
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        # expr_dir = os.path.join(str(expr_dir), opt.model + '_' + opt.netG + '_' + opt.netD)
        # util.mkdirs(expr_dir)

        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """解析参数，设置gpu设备
        Returns:
            解析器
        """
        opt = self.gather_options()  # 获取解析器
        opt.isTrain = self.isTrain  # 设置训练标志

        # TODO 处理后缀
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # TODO 设置gpu设备
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
