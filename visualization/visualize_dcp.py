import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from models.atmosphere_pix2pix_model import TransmissionEstimator

def get_dark_channel(x, width):
    max_pool = torch.nn.MaxPool2d(kernel_size=width, stride=1)
    x = torch.min(x, dim=1, keepdim=True)[0]
    x = F.pad(x, (width // 2, width // 2, width // 2, width // 2), mode='constant', value=1)
    x = -(max_pool(-x))
    return x

model = TransmissionEstimator(width=15)

img_path = '/home/ypj/dataset/cholec80_desmoking/cycle_gan/testA/581.bmp'

img = Image.open(img_path).convert('RGB')
img = torchvision.transforms.ToTensor()(img)


# A和dark_channel的shape为[1, 1, H, W]

# A_img = torchvision.transforms.ToPILImage()(A.squeeze(0))
# # plt.imsave('A.png', A_img)
# plt.imshow(A_img)
# plt.show()
# dark_channel = get_dark_channel(img.unsqueeze(0), 2)
# dark_channel_img = torchvision.transforms.ToPILImage()(dark_channel.squeeze(0))

# plt.imshow(dark_channel_img, cmap='gray')
# plt.imsave('dark_channel.png', dark_channel_img, cmap='gray')

A = model.get_atmosphere_light_new(img.unsqueeze(0))
A_img = torchvision.transforms.ToPILImage()(A.squeeze(0))

t_ = model.get_transmission(img.unsqueeze(0), A)
t__img = torchvision.transforms.ToPILImage()(t_.squeeze(0))

t = model.get_refined_transmission(img.unsqueeze(0), t_)

t_img = torchvision.transforms.ToPILImage()(t.squeeze(0))

plt.imshow(t_img, cmap='gray')
plt.show()

plt.imsave('A.png', A_img)
plt.imsave('t_.png', t__img, cmap='gray')
plt.imsave('t.png', t_img, cmap='gray')