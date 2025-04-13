import os
from PIL import Image
from util.util import get_transform
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.hazy_image_dir = os.path.join(opt.dataroot, opt.phase + 'A')

        self.hazy_image_path = sorted(make_dataset(self.hazy_image_dir, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1), convert=True)

    def __getitem__(self, index):
        hazy_image_path = self.hazy_image_path[index]
        hazy_image = Image.open(hazy_image_path).convert('RGB')
        hazy_image = self.transform(hazy_image)

        return {'A': hazy_image, 'A_paths': hazy_image_path}

    def __len__(self):
        return len(self.hazy_image_path)
