import os
import shutil
import numpy as np

mode = "test"
ori_dataset_path = "/home/ypj/dataset/PSv2rs"
dst_dataset_path = "/home/ypj/dataset/cholec80_desmoking/psv2rs"
ori_clear_path = sorted(os.listdir(os.path.join(ori_dataset_path, mode, "gt")))
ori_smk_path = sorted(os.listdir(os.path.join(ori_dataset_path, mode, "smk")))

samples = list(range(0, len(ori_clear_path)-3, 3))
samples = [i * 3 for i in np.linspace(0, len(samples), 120, dtype=int)]

if not os.path.exists(dst_dataset_path):
    os.makedirs(dst_dataset_path)

for item in ['light', 'middle', 'heavy']:
    if not os.path.exists(os.path.join(dst_dataset_path, item)):
        os.makedirs(os.path.join(dst_dataset_path, item))

        os.makedirs(os.path.join(dst_dataset_path, item, "trainA"))
        os.makedirs(os.path.join(dst_dataset_path, item, "trainB"))
        os.makedirs(os.path.join(dst_dataset_path, item, "testA"))
        os.makedirs(os.path.join(dst_dataset_path, item, "testB"))

for i in samples:
    print(i)
    light_smk_path = os.path.join(ori_dataset_path, mode, 'smk', ori_smk_path[i])
    middle_smk_path = os.path.join(ori_dataset_path, mode, 'smk', ori_smk_path[i + 1])
    heavy_smk_path = os.path.join(ori_dataset_path, mode, 'smk', ori_smk_path[i + 2])

    shutil.copy(light_smk_path, os.path.join(dst_dataset_path, 'light', mode + 'A', ori_clear_path[i]))
    shutil.copy(middle_smk_path, os.path.join(dst_dataset_path, 'middle', mode + 'A', ori_clear_path[i + 1]))
    shutil.copy(heavy_smk_path, os.path.join(dst_dataset_path, 'heavy', mode + 'A', ori_clear_path[i + 2]))
    shutil.copy(os.path.join(ori_dataset_path, mode, "gt", ori_clear_path[i]),
                os.path.join(dst_dataset_path, 'light', mode + 'B', ori_clear_path[i]))
    shutil.copy(os.path.join(ori_dataset_path, mode, "gt", ori_clear_path[i + 1]),
                os.path.join(dst_dataset_path, 'middle', mode + 'B', ori_clear_path[i + 1]))
    shutil.copy(os.path.join(ori_dataset_path, mode, "gt", ori_clear_path[i + 2]),
                os.path.join(dst_dataset_path, 'heavy', mode + 'B', ori_clear_path[i + 2]))
