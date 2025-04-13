import torch
import itertools
from torch import nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from .vision_transformer import SwinUnet
from . import networks


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.maxpool = torch.nn.MaxPool2d(kernel_size=8, stride=8)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        # batch_size, seq_length, embed_dim = q.size()
        batch_size, embed_dim, h, w = q.size()
        seq_length = h * w

        # k = self.maxpool(k)
        # v = self.maxpool(v)

        k = k.reshape(1, 3, int(256 * 256))
        v = v.view(1, 3, int(256 * 256))
        q = q.view(1, 3, int(256 * 256))

        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        # Apply linear transformations
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        k_ = self.maxpool(k.permute(0, 2, 1).reshape(1, 3, int(256), int(256))).reshape(1, 3, -1).permute(0, 2, 1).reshape(1, 3, int(256 * 256 / 64))
        v_ = self.maxpool(v.permute(0, 2, 1).reshape(1, 3, int(256), int(256))).reshape(1, 3, -1).permute(0, 2, 1).reshape(1, 3, int(256 * 256 / 64))

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k_ = k_.view(batch_size, int(seq_length / 64), self.num_heads, self.head_dim).transpose(1, 2)
        v_ = v_.view(batch_size, int(seq_length / 64), self.num_heads, self.head_dim).transpose(1, 2)
        # k = k.view(batch_size, int(seq_length), self.num_heads, self.head_dim).transpose(1, 2)
        # v = v.view(batch_size, int(seq_length), self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k_.transpose(-2, -1)) * self.scaling
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_)

        # Concatenate heads and apply final linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out(attn_output)

        return output, attn_weights, v


class ANet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinUnet(num_classes=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=8, stride=8)
        self.attention = MultiHeadSelfAttention(embed_dim=3, num_heads=3)
        self.conv_var = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_mean = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, image, dark_image):
        feat = self.backbone(image)
        dark_feat = self.backbone(dark_image)

        attn_output, attn_weights, v = self.attention(feat, dark_feat, feat)

        attn_output = attn_output.reshape(1, int(224), int(224), 3).permute(0, 3, 1, 2)
        v = v.reshape(1, int(256), int(256), 3).permute(0, 3, 1, 2)
        var = v - attn_output

        var = self.conv_var(var)
        attn_output = self.conv_mean(attn_output)

        return attn_output + var


class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # 默认不使用dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_A = ['hazy_image', 'dehaze_image_g', 'haze_image_rec']
        visual_names_B = ['clear_image', 'haze_image_g', 'dehaze_image_rec']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']

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
        self.netG_B = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.a_net = networks.init_net(ANet(), opt.init_type, opt.init_gain, self.gpu_ids)
        self.a_net = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.c_net = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netG_A = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

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
            self.criterionEq = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.c_net.parameters(), self.a_net.parameters(), self.netG_A.parameters(), self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.hazy_image = input['A' if AtoB else 'B'].to(self.device)
        self.dark_hazy_image = input['D'].to(self.device)
        self.clear_image = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.dehaze_image_g = self.netG_A(self.hazy_image)  # A->B', 有雾图像转去雾图像，充当J
        self.haze_image_rec = self.netG_B(self.dehaze_image_g)  # B'->A, 去雾图像转有雾图像

        self.haze_image_g = self.netG_B(self.clear_image)  # B->A', 去雾图像转有雾图像
        self.dehaze_image_rec = self.netG_A(self.haze_image_g)

        self.A = self.a_net(self.hazy_image, self.dark_hazy_image)
        self.t = self.c_net(self.hazy_image)
        self.haze_image_eq = self.dehaze_image_g * self.t + self.A * (1 - self.t)

        # self.fake_B = self.netG_A(self.real_A)  # G_A(A), 即A->B, 有烟雾的图像
        # self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)), 即A->B->A, 生成的去烟雾图像转有烟雾
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B), 即B->A, 去烟雾图像
        # self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)), 即B->A->B, 生成的有烟雾图像转去烟雾

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
        fake_B = self.fake_B_pool.query(self.dehaze_image_g)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.clear_image, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.haze_image_g)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.hazy_image, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B

        Returns:

        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        # 自己和自己更相似
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.hazy_image)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.hazy_image) * lambda_B * lambda_idt
            # self.idt_A = self.netG_A(self.real_B)  # B->B', 希望B'和B尽可能相似
            # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            self.idt_B = self.netG_B(self.clear_image)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.clear_image) * lambda_A * lambda_idt
            # self.idt_B = self.netG_B(self.real_A)  # A->A', 希望A'和A尽可能相似
            # self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.dehaze_image_g), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.haze_image_g), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.haze_image_rec, self.hazy_image) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.dehaze_image_rec, self.clear_image) * lambda_B
        # eq
        self.loss_eq = self.criterionEq(self.haze_image_eq, self.hazy_image)
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_eq
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
