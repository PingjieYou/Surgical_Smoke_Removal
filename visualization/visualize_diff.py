from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像路径
image1_path = r"C:\Users\youpj\Documents\School\CQU\Thesis\手术图像去雾\效果图\AtmospherePix2PixResnet9\test_latest\images\0049_fake_B.png"  # 去雾图像路径
image2_path = r"C:\Users\youpj\Documents\School\CQU\Thesis\手术图像去雾\效果图\AtmospherePix2PixSwinunet\test_latest\images\0049_real_B.png"  # 无烟图像路径

# 使用PIL读取RGB图像
real_image_pil = Image.open(image2_path).convert('RGB')  # 转换为RGB图像
generated_image_pil = Image.open(image1_path).convert('RGB')

# 将PIL图像转换为NumPy数组
real_image = np.array(real_image_pil)
generated_image = np.array(generated_image_pil)

# 检查图像是否成功加载
if real_image is None or generated_image is None:
    raise ValueError("图像加载失败，请检查路径是否正确。")

# 计算两幅图像的差异
diff = cv2.absdiff(real_image, generated_image)

# 计算每个通道的差异（RGB）
diff_r = diff[:, :, 0]
diff_g = diff[:, :, 1]
diff_b = diff[:, :, 2]

# 合并RGB差异图像
diff_rgb = np.stack([diff_r, diff_g, diff_b], axis=-1)

# 对差异图像进行二值化处理
_, diff_thresh = cv2.threshold(diff_rgb, 30, 255, cv2.THRESH_BINARY)

# 使用形态学操作（如膨胀）来增强差异区域
kernel = np.ones((5, 5), np.uint8)
diff_dilated = cv2.dilate(diff_thresh, kernel, iterations=2)

# 找到差异区域的轮廓
# 确保差异图像是单通道
diff_dilated_gray = cv2.cvtColor(diff_dilated, cv2.COLOR_BGR2GRAY)

# 找到差异区域的轮廓
contours, _ = cv2.findContours(diff_dilated_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 按轮廓面积排序，找到最大的 1~3 个差异区域
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# 创建掩码图像，用于显示差异区域
mask_image = np.zeros_like(real_image)  # 创建与原图同样大小的全黑图像
for contour in contours:
    cv2.drawContours(mask_image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)  # 使用红色填充掩码区域

# 在模型生成的图像上绘制边界框
output_image_with_bboxes = generated_image.copy()  # 复制生成图像
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)  # 获取边界框
    cv2.rectangle(output_image_with_bboxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制红色边界框

# 显示结果

# 使用掩码图像（掩码区域用红色标记）
masked_image = cv2.addWeighted(generated_image, 1, mask_image, 0.7, 0)  # 合成掩码和生成图像
plt.imshow(masked_image)

# # 隐藏边框
# plt.axis('off')
#
# plt.show()
im1 = Image.open(image1_path)
im1.save("./generated_image.png")

im2 = Image.open(image2_path)
im2.save("./real_image.png")

im = Image.fromarray(masked_image)
im.save("./masked_image.png")