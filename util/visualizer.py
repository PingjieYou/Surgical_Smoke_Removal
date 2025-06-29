import datetime
import numpy as np
import os
import sys
import ntpath
import time
import torch

from . import util, html
from subprocess import Popen, PIPE
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import io, color, filters
import math
from collections import OrderedDict
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def rgb_to_lab(rgb):
    """
    将RGB颜色值转换为Lab颜色值。

    参数:
    rgb (tuple): RGB颜色值 (R, G, B)，范围0-255。

    返回:
    LabColor: 转换后的Lab颜色对象。
    """
    rgb_normalized = np.array(rgb) / 255.0
    rgb_color = sRGBColor(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2], is_upscaled=False)
    return convert_color(rgb_color, LabColor)


def calculate_ciede2000(a, b):
    """
    计算两张图片的 CIEDE2000 色差。

    参数:
    a (numpy.ndarray): 第一张图像的 RGB 数组。
    b (numpy.ndarray): 第二张图像的 RGB 数组。

    返回:
    (float, float): (CIEDE2000 色差均值, CIEDE2000 色差标准差)
    """
    if a.shape != b.shape:
        raise ValueError("输入的两张图像尺寸必须相同！")

    height, width, _ = a.shape
    delta_e_values = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            lab_a = rgb_to_lab(tuple(a[y, x]))
            lab_b = rgb_to_lab(tuple(b[y, x]))
            delta_e_values[y, x] = delta_e_cie2000(lab_a, lab_b)

    return np.mean(delta_e_values), np.std(delta_e_values)


def rmetrics(a, b):
    """
    计算 MSE、PSNR、SSIM 和 CIEDE2000 色差。

    参数:
    a (numpy.ndarray): 第一张图像的 RGB 数组。
    b (numpy.ndarray): 第二张图像的 RGB 数组。

    返回:
    (float, float, float, float, float): (MSE, PSNR, SSIM, CIEDE2000 均值, CIEDE2000 标准差)
    """
    # 计算 MSE
    mse = np.mean((a - b) ** 2)

    # 计算 PSNR
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(255 / math.sqrt(mse))

    # 计算 SSIM
    ssim = compare_ssim(a, b, channel_axis=2)


    return mse, psnr, ssim


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)


class Visualizer():
    def __init__(self, opt):
        """
        初始化Visualizer类

        Step 1: 缓存训练/测试选项
        Step 2: 连接到visdom服务器
        Step 3: 创建一个HTML对象以保存HTML过滤器
        Step 4: 创建一个日志文件以存储训练损失

        Args:
            opt: 命令行参数
        """
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        # self.use_html = False
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        self.mse = []
        self.psnr = []  # 每个显示迭代
        self.ssim = []

        if self.display_id > 0:  # 连接到visdom服务器
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # 创建一个HTML文件夹以保存网页过滤器
            if not os.path.exists(opt.checkpoints_dir + '/' + opt.name):
                os.mkdir(opt.checkpoints_dir + '/' + opt.name)

            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        # # 创建一个日志文件以存储训练损失
        # self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

        # 创建一个日志文件以存储性能
        # self.log_pfm_name = os.path.join(opt.checkpoints_dir, opt.name, 'Performance_log.txt')
        # with open(self.log_pfm_name, "a") as log_pfm_file:
        #     now = time.strftime("%c")
        #     log_pfm_file.write('================ Performance (%s) ================\n' % now)
        self.log_pfm_val_name = os.path.join(opt.checkpoints_dir, opt.name, 'log.txt')

        if opt.phase == 'train':
            with open(self.log_pfm_val_name, "w") as log_pfm_val_file:
                now = time.strftime("%c")
                # log_pfm_val_file.write('================ Val_Performance (%s) ================\n' % now)
                log_pfm_val_file.write('Initial Settings:\n')
                log_pfm_val_file.write('Dataset: %s\n' % opt.dataset)
                log_pfm_val_file.write('Model: %s\n' % opt.name)
                log_pfm_val_file.write('Learning Rate: %s\n' % opt.lr)
                log_pfm_val_file.write('Beta1: %s\n' % opt.beta1)
                log_pfm_val_file.write('Seed: %s\n' % opt.seed)
                log_pfm_val_file.write('Dataset mode: %s\n' % opt.dataset_mode)
                log_pfm_val_file.write('Generator: %s\n' % opt.netG)
                log_pfm_val_file.write('Discriminator: %s\n' % opt.netD)
                log_pfm_val_file.write('-' * 50 + '\n \n')

    def reset(self):
        """
        重置Visualizer类

        Returns:

        """
        self.saved = False

    def create_visdom_connections(self):
        """
        创建visdom连接

        Returns:

        """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        mse = []
        psnr = []  # every display iter
        ssim = []
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:  # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

                im_out = images[1].transpose([1, 2, 0])
                im_GT = images[2].transpose([1, 2, 0])

                mse, psnr, ssim = rmetrics(im_out, im_GT)



            else:  # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    # message_pfm = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    # for k, v in mtpfm.items():
    #     message_pfm += '%s: %.3f ' % (k, v)
    # print(message_pfm)  # print the message
    #  with open(self.log_pfm_name, "a") as log_pfm_file:
    #      log_pfm_file.write('%s\n' % message_pfm)  # save the message

    def print_current_val_mtx(self, epoch, losses, t_comp, t_data, mtpfm):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            f'[{current_time}] '
            f'Epoch {epoch}: '
            f'Train ')

        for k, v in losses.items():
            message += f'({k}: {v:1.3f}) '

        # message_pfm = '(epoch: %d, time: %.3f, data: %.3f) ' % (epoch, t_comp, t_data)
        message += 'Val '
        for k, v in mtpfm.items():
            message += '(%s: %.3f) ' % (k, v)
        # print(message_pfm)  # print the message

        with open(self.log_pfm_val_name, "a") as log_pfm_val_file:
            log_pfm_val_file.write('%s\n' % message)  # save the message

    def cal_current_pfm(self, epoch, visuals):
        images = []
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            images.append(image_numpy.transpose([2, 0, 1]))

        im_out = images[1].transpose([1, 2, 0])
        im_GT = images[2].transpose([1, 2, 0]) # pix2pix
        # im_GT = images[3].transpose([1, 2, 0]) # cyclegan

        import matplotlib.pyplot as plt

        # plt.imshow(im_out)
        # plt.show()
        #
        # plt.imshow(im_GT)
        # plt.show()

        mse, psnr, ssim = rmetrics(im_out, im_GT)

        mt_pfm = OrderedDict()
        mt_pfm['MSE'] = mse
        mt_pfm['PSNR'] = psnr
        mt_pfm['SSIM'] = ssim

        return mt_pfm

    def plot_current_ssim(self, epoch, counter_ratio, mt_pfm):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """

        if not hasattr(self, 'plot_performance'):
            self.plot_performance = {'X': [], 'Y': [], 'legend': ['MSE']}
        self.plot_performance['X'].append(epoch + counter_ratio)
        self.plot_performance['Y'].append([mt_pfm[k] for k in self.plot_performance['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_performance['X'])] * len(self.plot_performance['legend']), 1),
                Y=np.array(self.plot_performance['Y']),
                opts={
                    'title': self.name + ' MSE over time',
                    'legend': self.plot_performance['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'Performance'},
                win=self.display_id + 3)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)

        if not hasattr(self, 'plot_performance_2'):
            self.plot_performance_2 = {'X': [], 'Y': [], 'legend': ['PSNR']}
        self.plot_performance_2['X'].append(epoch + counter_ratio)
        self.plot_performance_2['Y'].append([mt_pfm[k] for k in self.plot_performance_2['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_performance_2['X'])] * len(self.plot_performance_2['legend']), 1),
                Y=np.array(self.plot_performance_2['Y']),
                opts={
                    'title': self.name + ' PSNR over time',
                    'legend': self.plot_performance_2['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'PSNR'},
                win=self.display_id + 4)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)

        if not hasattr(self, 'plot_performance_3'):
            self.plot_performance_3 = {'X': [], 'Y': [], 'legend': ['SSIM']}
        self.plot_performance_3['X'].append(epoch + counter_ratio)
        self.plot_performance_3['Y'].append([mt_pfm[k] for k in self.plot_performance_3['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_performance_3['X'])] * len(self.plot_performance_3['legend']), 1),
                Y=np.array(self.plot_performance_3['Y']),
                opts={
                    'title': self.name + ' SSIM over time',
                    'legend': self.plot_performance_3['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'Performance'},
                win=self.display_id + 5)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)

    def plot_current_ssim_val(self, epoch, counter_ratio, mt_pfm):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """

        if not hasattr(self, 'plot_performance_val'):
            self.plot_performance_val = {'X': [], 'Y': [], 'legend': ['MSE']}
        self.plot_performance_val['X'].append(epoch + counter_ratio)
        self.plot_performance_val['Y'].append([mt_pfm[k] for k in self.plot_performance_val['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_performance_val['X'])] * len(self.plot_performance_val['legend']), 1),
                Y=np.array(self.plot_performance_val['Y']),
                opts={
                    'title': self.name + ' Val MSE over time',
                    'legend': self.plot_performance_val['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'Performance'},
                win=self.display_id + 6)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)

        if not hasattr(self, 'plot_performance_val_2'):
            self.plot_performance_val_2 = {'X': [], 'Y': [], 'legend': ['PSNR']}
        self.plot_performance_val_2['X'].append(epoch + counter_ratio)
        self.plot_performance_val_2['Y'].append([mt_pfm[k] for k in self.plot_performance_val_2['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_performance_val_2['X'])] * len(self.plot_performance_val_2['legend']), 1),
                Y=np.array(self.plot_performance_val_2['Y']),
                opts={
                    'title': self.name + ' Val PSNR over time',
                    'legend': self.plot_performance_val_2['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'PSNR'},
                win=self.display_id + 7)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)

        if not hasattr(self, 'plot_performance_val_3'):
            self.plot_performance_val_3 = {'X': [], 'Y': [], 'legend': ['SSIM']}
        self.plot_performance_val_3['X'].append(epoch + counter_ratio)
        self.plot_performance_val_3['Y'].append([mt_pfm[k] for k in self.plot_performance_val_3['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_performance_val_3['X'])] * len(self.plot_performance_val_3['legend']), 1),
                Y=np.array(self.plot_performance_val_3['Y']),
                opts={
                    'title': self.name + ' Val SSIM over time',
                    'legend': self.plot_performance_val_3['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'Performance'},
                win=self.display_id + 8)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)
