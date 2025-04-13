import os
import numpy as np
from PIL import Image
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import concurrent.futures
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def rgb_to_lab(rgb):
    """将RGB颜色值转换为Lab颜色值。"""
    rgb_normalized = np.array(rgb) / 255.0
    rgb_color = sRGBColor(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2], is_upscaled=False)
    return convert_color(rgb_color, LabColor)

def calculate_ciede2000_from_rgb(rgb1, rgb2):
    """计算两组RGB颜色值之间的CIEDE2000色差。"""
    return delta_e_cie2000(rgb_to_lab(rgb1), rgb_to_lab(rgb2))

def calculate_ciede2000_from_image(image_path1, image_path2):
    """计算两张图像的CIEDE2000色差矩阵。"""
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)

    if pixels1.shape != pixels2.shape:
        raise ValueError("两张图像的尺寸必须一致")

    height, width, _ = pixels1.shape
    delta_e_matrix = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            delta_e_matrix[y, x] = calculate_ciede2000_from_rgb(tuple(pixels1[y, x]), tuple(pixels2[y, x]))

    return delta_e_matrix

def calculate_ssim_psnr(image_path1, image_path2):
    """计算两张图像的 SSIM 和 PSNR。"""
    image1 = np.array(Image.open(image_path1).convert("L"))  # 转为灰度图
    image2 = np.array(Image.open(image_path2).convert("L"))

    ssim_value = ssim(image1, image2, data_range=255)
    psnr_value = psnr(image1, image2, data_range=255)

    return ssim_value, psnr_value

def calculate_metrics_for_pair(image_pair):
    """
    计算每对图像的CIEDE2000色差、SSIM 和 PSNR，并返回均值。

    返回:
    (float, float, float): (平均CIEDE2000色差, SSIM, PSNR)
    """
    delta_e_matrix = calculate_ciede2000_from_image(image_pair[0], image_pair[1])
    mean_ciede = np.mean(delta_e_matrix)

    ssim_value, psnr_value = calculate_ssim_psnr(image_pair[0], image_pair[1])

    return mean_ciede, ssim_value, psnr_value

if __name__ == "__main__":
    image_root = "/home/ypj/code/research/surgical desmoking/atomosphere cyclegan/results"

    modes = ['light', 'middle', 'heavy']
    models = ["pix2pix", "cycle_gan", "atmosphere_pix2pix", "atmosphere_cycle_gan"]
    netGs = ["PFAN", "unet_256", "unet_128", "resnet_9blocks", "resnet_6blocks"]

    for mode in modes:
        for model in models:
            for net in netGs:
                name = model.capitalize() + net.capitalize() + mode.capitalize()
                image_folder = os.path.join(image_root, name, 'test_latest/images')
                real_paths = []
                fake_paths = []

                if "pix2pix" in model:
                    for image_path in sorted(os.listdir(image_folder)):
                        if "fake_B" in image_path:
                            fake_paths.append(os.path.join(image_folder, image_path))
                        elif "real_B" in image_path:
                            real_paths.append(os.path.join(image_folder, image_path))
                else:
                    for image_path in sorted(os.listdir(image_folder)):
                        if "dehaze_image_g" in image_path:
                            fake_paths.append(os.path.join(image_folder, image_path))
                        elif "clear_image" in image_path:
                            real_paths.append(os.path.join(image_folder, image_path))

                # 配对图像路径
                image_pairs = list(zip(fake_paths, real_paths))

                # 并行计算 CIEDE2000、SSIM 和 PSNR
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    metrics_list = list(executor.map(calculate_metrics_for_pair, image_pairs))

                ciede_values, ssim_values, psnr_values = zip(*metrics_list)

                # 计算均值和标准差
                print(name + f" 平均CIEDE2000色差: {np.mean(ciede_values):.4f}, 标准差: {np.std(ciede_values):.4f}")
                print(name + f" 平均SSIM: {np.mean(ssim_values):.4f}, 标准差: {np.std(ssim_values):.4f}")
                print(name + f" 平均PSNR: {np.mean(psnr_values):.4f}, 标准差: {np.std(psnr_values):.4f}")
