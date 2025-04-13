import os
import random
from PIL import Image
from util.util import get_transform
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.hazy_image_dir = os.path.join(opt.dataroot, opt.phase + 'A')
        self.clear_image_dir = os.path.join(opt.dataroot, opt.phase + 'B')

        self.hazy_image_paths = sorted(make_dataset(self.hazy_image_dir, opt.max_dataset_size))
        self.clear_image_paths = sorted(make_dataset(self.clear_image_dir, opt.max_dataset_size))
        self.clear_image_size = len(self.hazy_image_paths)  # 获取数据集A的大小
        self.clear_image_size = len(self.clear_image_paths)  # 获取数据集B的大小
        clear_to_hazy = self.opt.direction == 'BtoA'  # A->B, 模糊->清晰
        input_nc = self.opt.output_nc if clear_to_hazy else self.opt.input_nc
        output_nc = self.opt.input_nc if clear_to_hazy else self.opt.output_nc
        self.hazy_transform = get_transform(self.opt, grayscale=(input_nc == 1), convert=True)
        self.clear_transform = get_transform(self.opt, grayscale=(output_nc == 1), convert=True)

    def __getitem__(self, index):
        hazy_image_path = self.hazy_image_paths[index % self.clear_image_size]
        if self.opt.serial_batches:
            clear_image_index = index % self.clear_image_size
        else:
            clear_image_index = random.randint(0, self.clear_image_size - 1)
        clear_image_path = self.clear_image_paths[clear_image_index]

        hazy_image = Image.open(hazy_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        hazy_image = self.hazy_transform(hazy_image)
        clear_image = self.clear_transform(clear_image)

        return {'A': hazy_image, 'B': clear_image, 'A_paths': hazy_image_path, 'B_paths': clear_image_path}

    def __len__(self):
        return max(self.clear_image_size, self.clear_image_size)
