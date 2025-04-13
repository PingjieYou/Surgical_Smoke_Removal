# -*- coding: utf-8 -*-
# @Author  : Pingjie You
# @Time    : 2024/5/26 21:27
# @Email   : youpj2000@gmail.com
# @Function:

import os

# 处理Dense_Haze数据集
ori_dataset_path = 'D:\Dataset\Desmoking\Dense_Haze'
dst_dataset_path = 'D:\Dataset\Desmoking\Dense_Haze_CycleGAN'

if not os.path.exists(dst_dataset_path):
    os.makedirs(dst_dataset_path)

train_haze_dir = os.path.join(dst_dataset_path, 'trainA')
train_dehaze_dir = os.path.join(dst_dataset_path, 'trainB')
val_haze_dir = os.path.join(dst_dataset_path, 'valA')
val_dehaze_dir = os.path.join(dst_dataset_path, 'valB')
test_haze_dir = os.path.join(dst_dataset_path, 'testA')
test_dehaze_dir = os.path.join(dst_dataset_path, 'testB')

if not os.path.exists(train_haze_dir):
    os.makedirs(train_haze_dir)
if not os.path.exists(train_dehaze_dir):
    os.makedirs(train_dehaze_dir)
if not os.path.exists(val_haze_dir):
    os.makedirs(val_haze_dir)
if not os.path.exists(val_dehaze_dir):
    os.makedirs(val_dehaze_dir)
if not os.path.exists(test_haze_dir):
    os.makedirs(test_haze_dir)
if not os.path.exists(test_dehaze_dir):
    os.makedirs(test_dehaze_dir)

ori_train_dir = os.path.join(ori_dataset_path, 'train')
ori_val_dir = os.path.join(ori_dataset_path, 'val')
ori_test_dir = os.path.join(ori_dataset_path, 'test')

train_image_name = os.listdir(ori_train_dir)
val_image_name = os.listdir(ori_val_dir)
test_image_name = os.listdir(ori_test_dir)

train_haze_path = [os.path.join(ori_train_dir, image_name) for image_name in train_image_name if 'hazy' in image_name]
train_dehaze_path = [os.path.join(ori_train_dir, image_name) for image_name in train_image_name if 'clear' in image_name]

print(train_haze_path)
