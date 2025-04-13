import torch
import itertools
import torchvision
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的命令行参数，并返回修改后的parser。

        Args:
            parser: 原始命令行参数
            is_train: 是否是训练模式

        Returns: 修改后的parser

        对于CycleGAN，除了GAN损失外，我们还引入lambda_A、lambda_B和lambda_identity用于以下损失。
        A（源域），B（目标域）。
        Generator: G_A：A -> B；G_B：B -> A。
        Discriminator: D_A：G_A（A）vs。B；D_B：G_B（B）vs。A。
        Forward cycle loss: lambda_A * ||G_B（G_A（A）） - A||（论文中的方程（2））
        Backward cycle loss: lambda_B * ||G_A（G_B（B）） - B||（论文中的方程（2））
        Identity loss (optional): lambda_identity *（||G_A（B） - B|| * lambda_B + ||G_B（A） - A|| * lambda_A）（第5.2节“从绘画生成照片”中）
        """
        parser.set_defaults(no_dropout=True)  # 默认不使用dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        # 如果需要计算identity loss，我们还需要将idt_A和idt_B添加到visual_names_A和visual_names_B中。
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # 结合A和B的视觉结果

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']  # 在测试时，我们只使用生成器

        # TODO 定义生成器
        self.netG_A = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # TODO 定义判别器
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert (opt.input_nc == opt.output_nc)
            # TODO 图像缓冲区，用于保存之前生成的图像
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # TODO 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.SGD(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr)
            # self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        解包输入数据并执行必要的预处理步骤。

        Args:
            input: 包括数据本身及其元数据信息。

        Returns:

        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        pass

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A), 即A->B, 有烟雾的图像
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)), 即A->B->A, 生成的去烟雾图像转有烟雾
        self.fake_A = self.netG_B(self.real_B)  # G_B(B), 即B->A, 去烟雾图像
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)), 即B->A->B, 生成的有烟雾图像转去烟雾
        pass

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B

        Returns:

        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)  # B->B', 希望B'和B尽可能相似
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            self.idt_B = self.netG_B(self.real_A)  # A->A', 希望A'和A尽可能相似
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """
        优化网络权重

        Returns:

        """

        self.forward()  # 计算网络输出

        # 更新生成器
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # 优化G_A和G_B，不优化D_A和D_B
        self.optimizer_G.zero_grad()  # 设置G_A和G_B的梯度为0
        self.backward_G()  # 计算G_A和G_B的梯度
        self.optimizer_G.step()  # 更新G_A和G_B的权重

        # 更新判别器
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
# import math
# from torch.autograd import Variable
# import torch
# import itertools
# from torch import nn
# import torch.nn.functional as F
# from util.image_pool import ImagePool
# from .base_model import BaseModel
# from .vision_transformer import SwinUnet
# from . import networks
#
#
# class GuidedFilter(nn.Module):
#     def __init__(self, r, eps=1e-8):
#         super(GuidedFilter, self).__init__()
#
#         self.r = r
#         self.eps = eps
#         self.boxfilter = BoxFilter(r)
#
#     def forward(self, x, y):
#         n_x, c_x, h_x, w_x = x.size()
#         n_y, c_y, h_y, w_y = y.size()
#
#         assert n_x == n_y
#         # assert c_x == 1 or c_x == c_y
#         assert h_x == h_y and w_x == w_y
#         assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1
#
#         # N
#         N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
#
#         # mean_x
#         mean_x = self.boxfilter(x) / N
#         # mean_y
#         mean_y = self.boxfilter(y) / N
#         # cov_xy
#         cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
#         # var_x
#         var_x = self.boxfilter(x * x) / N - mean_x * mean_x
#
#         # A
#         A = cov_xy / (var_x + self.eps)
#         # b
#         b = mean_y - A * mean_x
#
#         # mean_A; mean_b
#         mean_A = self.boxfilter(A) / N
#         mean_b = self.boxfilter(b) / N
#
#         return torch.mean(mean_A * x + mean_b, dim=1, keepdim=True)
#
#
# def diff_x(input, r):
#     assert input.dim() == 4
#
#     left = input[:, :, r:2 * r + 1]
#     middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
#     right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]
#
#     output = torch.cat([left, middle, right], dim=2)
#
#     return output
#
#
# def diff_y(input, r):
#     assert input.dim() == 4
#
#     left = input[:, :, :, r:2 * r + 1]
#     middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
#     right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]
#
#     output = torch.cat([left, middle, right], dim=3)
#
#     return output
#
#
# class BoxFilter(nn.Module):
#     def __init__(self, r):
#         super(BoxFilter, self).__init__()
#
#         self.r = r
#
#     def forward(self, x):
#         assert x.dim() == 4
#
#         return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
#
#
# class GaussianFilter(nn.Module):
#     def __init__(self, kernel_size=5, sigma=3):
#         super(GaussianFilter, self).__init__()
#
#         self.gaussian_kernel = self.cal_kernel(kernel_size=kernel_size, sigma=sigma).expand(1, 1, -1, -1).cuda()
#
#     def apply_gaussian_filter(self, x):
#         # cal gaussian filter of N C H W in cuda
#         n, c, h, w = x.shape
#         gaussian = torch.nn.functional.conv2d(x, self.gaussian_kernel.expand(c, 1, -1, -1), padding=self.gaussian_kernel.shape[2] // 2, groups=c)
#
#         return gaussian
#
#     def cal_gaussian_kernel_at_ij(self, i, j, sigma):
#         return (1. / (2 * math.pi * pow(sigma, 2))) * math.exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2)))
#
#     def cal_kernel(self, kernel_size=3, sigma=1.):
#         kernel = torch.ones((kernel_size, kernel_size)).float()
#         for i in range(kernel_size):
#             for j in range(kernel_size):
#                 kernel[i, j] = self.cal_gaussian_kernel_at_ij(-(kernel_size // 2) + j, (kernel_size // 2) - i, sigma=sigma)
#
#         kernel = kernel / torch.sum(kernel)
#         return kernel
#
#
# class TransmissionEstimator(nn.Module):
#     def __init__(self, width=15, ):
#         super(TransmissionEstimator, self).__init__()
#         self.width = width
#         self.t_min = 0.2
#         self.alpha = 2.5
#         self.A_max = 220.0 / 255
#         self.omega = 0.95
#         self.p = 0.001
#         self.max_pool = nn.MaxPool2d(kernel_size=width, stride=1)
#         self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)
#         self.guided_filter = GuidedFilter(r=40, eps=1e-3)
#
#     def get_dark_channel(self, x):
#         x = torch.min(x, dim=1, keepdim=True)[0]
#         x = F.pad(x, (self.width // 2, self.width // 2, self.width // 2, self.width // 2), mode='constant', value=1)
#         x = -(self.max_pool(-x))
#         return x
#
#     def get_atmosphere_light(self, I, dc):
#         n, c, h, w = I.shape
#         flat_I = I.view(n, c, -1)
#         flat_dc = dc.view(n, 1, -1)
#         searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:, :, :int(h * w * self.p)]
#         searchidx = searchidx.expand(-1, 3, -1)
#         searched = torch.gather(flat_I, dim=2, index=searchidx)
#         return torch.max(searched, dim=2, keepdim=True)[0].unsqueeze(3)
#
#     def get_atmosphere_light_new(self, I):
#         I_dark = self.get_dark_channel(I)
#         A = self.get_atmosphere_light(I, I_dark)
#         A[A > self.A_max] = self.A_max
#         return A
#
#     def get_transmission(self, I, A):
#         return 1 - self.omega * self.get_dark_channel(I / A)
#
#     def get_refined_transmission(self, I, rawt):
#         I_max = torch.max(I.contiguous().view(I.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
#         I_min = torch.min(I.contiguous().view(I.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
#         normI = (I - I_min) / (I_max - I_min)
#         refinedT = self.guided_filter(normI, rawt)
#
#         return refinedT
#
#
# class CycleGANModel(BaseModel):
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         parser.set_defaults(no_dropout=True)  # 默认不使用dropout
#         if is_train:
#             parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
#             parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
#             parser.add_argument('--lambda_identity', type=float, default=0.5,
#                                 help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.')
#
#         return parser
#
#     def __init__(self, opt):
#         BaseModel.__init__(self, opt)
#         self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
#         # visual_names_A = ['real_A', 'fake_B', 'rec_A']
#         visual_names_A = ['hazy_image', 'dehaze_image_g', 'haze_image_rec']
#         visual_names_B = ['clear_image', 'haze_image_g', 'dehaze_image_rec']
#         # visual_names_B = ['real_B', 'fake_A', 'rec_B']
#
#         # 如果需要计算identity loss，我们还需要将idt_A和idt_B添加到visual_names_A和visual_names_B中。
#         if self.isTrain and self.opt.lambda_identity > 0.0:
#             visual_names_A.append('idt_B')
#             visual_names_B.append('idt_A')
#
#         self.visual_names = visual_names_A + visual_names_B  # 结合A和B的视觉结果
#
#         if self.isTrain:
#             self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
#         else:
#             self.model_names = ['G_A', 'G_B']  # 在测试时，我们只使用生成器
#
#         # TODO 定义生成器
#         self.netG_A = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.netG_B = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         # self.a_net = networks.init_net(ANet(), opt.init_type, opt.init_gain, self.gpu_ids)
#         # self.a_net = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         # self.c_net = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.map = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.transmission = TransmissionEstimator()
#         # self.netG_A = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         # self.netG_B = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#
#         # TODO 定义判别器
#         if self.isTrain:
#             self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#             self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#
#         if self.isTrain:
#             if opt.lambda_identity > 0.0:
#                 assert (opt.input_nc == opt.output_nc)
#             # TODO 图像缓冲区，用于保存之前生成的图像
#             self.fake_A_pool = ImagePool(opt.pool_size)
#             self.fake_B_pool = ImagePool(opt.pool_size)
#             # TODO 定义损失函数
#             self.criterionEq = torch.nn.L1Loss()
#             self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
#             self.criterionCycle = torch.nn.L1Loss()
#             self.criterionIdt = torch.nn.L1Loss()
#             # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(itertools.chain(self.map.parameters(), self.transmission.parameters(), self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D)
#
#     def set_input(self, input):
#         """
#         解包输入数据并执行必要的预处理步骤。
#
#         Args:
#             input: 包括数据本身及其元数据信息。
#
#         Returns:
#
#         """
#         AtoB = self.opt.direction == 'AtoB'
#         self.hazy_image = input['A' if AtoB else 'B'].to(self.device)
#         # self.dark_hazy_image = input['D'].to(self.device)
#         self.clear_image = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']
#
#     def forward(self):
#         self.dehaze_image_g = self.netG_A(self.hazy_image)  # A->B', 有雾图像转去雾图像，充当J
#         self.haze_image_rec = self.netG_B(self.dehaze_image_g)  # B'->A, 去雾图像转有雾图像
#
#         self.haze_image_g = self.netG_B(self.clear_image)  # B->A', 去雾图像转有雾图像
#         self.dehaze_image_rec = self.netG_A(self.haze_image_g)
#
#         self.feats = self.map(self.hazy_image)
#         self.A = self.transmission.get_atmosphere_light_new(self.feats)
#         self.t_ = self.transmission.get_transmission(self.feats, self.A)
#         self.t = self.transmission.get_refined_transmission(self.feats, self.t_)
#         # self.A = self.a_net(self.hazy_image, self.dark_hazy_image)
#         # self.A = self.a_net(self.hazy_image)
#         # self.t = self.c_net(self.hazy_image)
#
#         # import matplotlib.pyplot as plt
#         # import torchvision
#
#         self.haze_image_eq = self.dehaze_image_g * self.t + self.A * (1 - self.t)
#
#         # plt.imshow(self.clear_image[0].cpu().detach().numpy().transpose(1, 2, 0))
#         # plt.show()
#         # pass
#         # self.fake_B = self.netG_A(self.real_A)  # G_A(A), 即A->B, 有烟雾的图像
#         # self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)), 即A->B->A, 生成的去烟雾图像转有烟雾
#         # self.fake_A = self.netG_B(self.real_B)  # G_B(B), 即B->A, 去烟雾图像
#         # self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)), 即B->A->B, 生成的有烟雾图像转去烟雾
#
#     def backward_D_basic(self, netD, real, fake):
#         """Calculate GAN loss for the discriminator
#
#         Parameters:
#             netD (network)      -- the discriminator D
#             real (tensor array) -- real images
#             fake (tensor array) -- images generated by a generator
#
#         Return the discriminator loss.
#         We also call loss_D.backward() to calculate the gradients.
#         """
#         # Real
#         pred_real = netD(real)
#         loss_D_real = self.criterionGAN(pred_real, True)
#         # Fake
#         pred_fake = netD(fake.detach())
#         loss_D_fake = self.criterionGAN(pred_fake, False)
#         # Combined loss and calculate gradients
#         loss_D = (loss_D_real + loss_D_fake) * 0.5
#         loss_D.backward()
#         return loss_D
#
#     def backward_D_A(self):
#         """Calculate GAN loss for discriminator D_A"""
#         fake_B = self.fake_B_pool.query(self.dehaze_image_g)
#         self.loss_D_A = self.backward_D_basic(self.netD_A, self.clear_image, fake_B)
#
#     def backward_D_B(self):
#         """Calculate GAN loss for discriminator D_B"""
#         fake_A = self.fake_A_pool.query(self.haze_image_g)
#         self.loss_D_B = self.backward_D_basic(self.netD_B, self.hazy_image, fake_A)
#
#     def backward_G(self):
#         """Calculate the loss for generators G_A and G_B
#
#         Returns:
#
#         """
#         lambda_idt = self.opt.lambda_identity
#         lambda_A = self.opt.lambda_A
#         lambda_B = self.opt.lambda_B
#
#         # Identity loss
#         # 自己和自己更相似
#         if lambda_idt > 0:
#             self.idt_A = self.netG_A(self.hazy_image)
#             self.loss_idt_A = self.criterionIdt(self.idt_A, self.hazy_image) * lambda_B * lambda_idt
#             # self.idt_A = self.netG_A(self.real_B)  # B->B', 希望B'和B尽可能相似
#             # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
#
#             self.idt_B = self.netG_B(self.clear_image)
#             self.loss_idt_B = self.criterionIdt(self.idt_B, self.clear_image) * lambda_A * lambda_idt
#             # self.idt_B = self.netG_B(self.real_A)  # A->A', 希望A'和A尽可能相似
#             # self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
#         else:
#             self.loss_idt_A = 0
#             self.loss_idt_B = 0
#
#         # GAN loss D_A(G_A(A))
#         self.loss_G_A = self.criterionGAN(self.netD_A(self.dehaze_image_g), True)
#         # GAN loss D_B(G_B(B))
#         self.loss_G_B = self.criterionGAN(self.netD_B(self.haze_image_g), True)
#         # Forward cycle loss || G_B(G_A(A)) - A||
#         self.loss_cycle_A = self.criterionCycle(self.haze_image_rec, self.hazy_image) * lambda_A
#         # Backward cycle loss || G_A(G_B(B)) - B||
#         self.loss_cycle_B = self.criterionCycle(self.dehaze_image_rec, self.clear_image) * lambda_B
#         # eq
#         self.loss_eq = self.criterionEq(self.haze_image_eq, self.hazy_image) * 10.
#         # combined loss and calculate gradients
#         self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_eq
#         self.loss_G.backward()
#
#     def optimize_parameters(self):
#         """
#         优化网络权重
#
#         Returns:
#
#         """
#
#         self.forward()  # 计算网络输出
#
#         # 更新生成器
#         self.set_requires_grad([self.netD_A, self.netD_B], False)  # 优化G_A和G_B，不优化D_A和D_B
#         self.optimizer_G.zero_grad()  # 设置G_A和G_B的梯度为0
#         self.backward_G()  # 计算G_A和G_B的梯度
#         self.optimizer_G.step()  # 更新G_A和G_B的权重
#
#         # 更新判别器
#         self.set_requires_grad([self.netD_A, self.netD_B], True)
#         self.optimizer_D.zero_grad()
#         self.backward_D_A()
#         self.backward_D_B()
#         self.optimizer_D.step()
