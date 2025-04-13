import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据
methods = ["CycleGAN", "Ours+CycleGAN", "Pix2pix", "Ours+Pix2pix"]
models = ["ResNet6", "ResNet9", "U-Net128", "U-Net256", "SwinUNet", "PFAN"]
values = np.array([
    [10.234, 7.993, 6.07, 5.151],
    [9.461,9.626,6.25,5.437],
    [10.387,5.984,4.343,4.512],
    [9.233,6.511,4.27,4.043],
    [10.617,5.219,4.818,4.793],
    [7.842,6.821,4.644,4.447]
])

# 颜色方案（浅色调）
palette = sns.color_palette("pastel")

# 创建画布
fig, ax = plt.subplots(figsize=(12, 7))
bar_width = 0.15
x = np.arange(len(models))

# 绘制条形图
for i, method in enumerate(methods):
    bars = ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=method, color=palette[i], edgecolor='gray', linewidth=1.2)
    
    # 添加阴影效果
    for bar in bars:
        bar.set_alpha(0.9)  # 设置透明度
        bar.set_linewidth(1)  # 轮廓线条

# 美化 X 轴和 Y 轴
ax.set_xticks(x + bar_width * (len(methods) / 2 - 0.5))
ax.set_xticklabels(models, fontsize=14, fontweight='bold')
ax.set_yticks(np.arange(0, max(values.flatten()) + 2, 2))
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
ax.set_xlabel("", fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel("", fontsize=16, fontweight='bold', labelpad=10)
ax.set_title("", fontsize=18, fontweight='bold', pad=15)

# 美化边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

# 添加网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.grid(False)

# 添加图例
legend = ax.legend(fontsize=12, loc='upper right', frameon=True, edgecolor="gray")
legend.get_frame().set_alpha(0.8)  # 让图例稍微透明

# 显示图表
plt.show()
