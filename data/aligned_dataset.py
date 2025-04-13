import os
from PIL import Image
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from util.util import get_transform, get_params


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.combine_image_dir = os.path.join(opt.dataroot, opt.phase)
        self.combine_paths = sorted(make_dataset(self.combine_image_dir, opt.max_dataset_size))
        assert (self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        combine_img_path = self.combine_paths[index]
        combine_image = Image.open(combine_img_path).convert('RGB')

        # TODO 按照中心切割图像
        w, h = combine_image.size
        w2 = int(w / 2)
        hazy_image = combine_image.crop((0, 0, w2, h))
        clear_image = combine_image.crop((w2, 0, w, h))

        transform_params = get_params(self.opt, hazy_image.size)
        hazy_image_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        clear_image_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        hazy_image = hazy_image_transform(hazy_image)
        clear_image = clear_image_transform(clear_image)

        return {'A': hazy_image, 'B': clear_image, 'A_paths': combine_img_path, 'B_paths': combine_img_path}

    def __len__(self):
        return len(self.combine_paths)
