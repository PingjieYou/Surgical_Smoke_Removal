# -*- coding: utf-8 -*-
# @Author  : Pingjie You
# @Time    : 2024/12/30 3:18
# @Email   : youpingjie@stu.cqu.edu.cn
# @Function: PyCharm

import os
from PIL import Image

dataset_path = '/home/ypj/dataset/cholec80_desmoking/psv2rs/heavy'

mode = 'train'

os.makedirs(os.path.join(dataset_path, mode), exist_ok=True)

valA_path = os.path.join(dataset_path, mode + 'A')
valB_path = os.path.join(dataset_path, mode + 'B')

valA_lists = [os.path.join(valA_path, img) for img in os.listdir(valA_path)]
valB_lists = [os.path.join(valB_path, img) for img in os.listdir(valB_path)]

for i in range(len(os.listdir(valA_path))):
    image1 = Image.open(valA_lists[i])
    image2 = Image.open(valB_lists[i])

    width1, height1 = image1.size
    width2, height2 = image2.size

    # 创建一个新的图像，宽度是两张图片宽度之和，高度是最大的高度
    new_width = width1 + width2
    new_height = max(height1, height2)

    # 创建新图像，背景色为白色
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # 将两幅图像粘贴到新图像上
    new_image.paste(image1, (0, 0))  # image1 在左边
    new_image.paste(image2, (width1, 0))  # image2 在右边

    new_image.save(os.path.join(dataset_path, mode, str(i) + '.bmp'))
