import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 风格
sns.set_theme(style="whitegrid")

# 设置全局字体为 Times New Roman
plt.rc("font", family="Times New Roman")

# 给定的均值数据（除以 100 进行归一化）
mean_values = {
    "CycleGAN": np.array([65.333, 66.006, 78.703, 81.392, 60.297, 87.71]) / 100,
    "CycleGAN+Ours": np.array([70.952, 65.724, 88.567, 87.703, 86.193, 88.802]) / 100,
    "Pix2pix": np.array([73.238, 75.892, 90.476, 90.743, 82.723, 90.471]) / 100,
    "Pix2pix+Ours": np.array([79.64, 78.311, 91.041, 91.159, 85.296, 91.16]) / 100
}

# 计算标准差，假设其为均值的 20%
std_dev = {key: np.std(values) * 0.2 for key, values in mean_values.items()}

# 生成 20 轮训练数据
num_epochs = 20
simulated_data = {
    framework: np.clip(np.random.normal(loc=means, scale=std_dev[framework], size=(num_epochs, len(means))), 0, 1)
    for framework, means in mean_values.items()
}

# 定义模型名称
columns = ["ResNet6", "ResNet9", "U-Net128", "U-Net256", "SwinUNet", "PFAN"]
data_frames = []

# 转换为 DataFrame
for framework, data in simulated_data.items():
    df = pd.DataFrame(data, columns=columns)
    df["Framework"] = framework
    data_frames.append(df)

df_all = pd.concat(data_frames, ignore_index=True)

# 保存为 Excel 文件
excel_filename = "simulated_boxplot_data.xlsx"
df_all.to_excel(excel_filename, index=False)

# 创建箱型图
plt.figure(figsize=(14, 8))

# 颜色方案
palette = sns.color_palette("muted", len(mean_values))  # 选择更柔和的颜色

# 计算位置
positions = np.arange(len(columns))
width = 0.15

# 绘制箱型图
for i, (framework, data) in enumerate(simulated_data.items()):
    bplot = plt.boxplot(data, positions=positions + i * width, widths=width, patch_artist=True,
                        boxprops=dict(facecolor=palette[i], alpha=0.5),  # 轻微透明
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linestyle="--", linewidth=1, color="gray"),
                        capprops=dict(linewidth=1, color="gray"))
    for patch in bplot['boxes']:
        patch.set_facecolor(palette[i])  # 设置颜色

# 添加分隔线
for pos in positions[1:]:
    plt.axvline(x=pos - width / 2, linestyle='--', color='gray', alpha=0.5)

# 设置 X 轴刻度和标签
plt.xticks(ticks=positions + width * (len(mean_values) / 2 - 0.5), labels=columns, fontsize=12, rotation=15)

# 设置 Y 轴范围（0.5 到 1），保证图形不拥挤
plt.ylim(0.5, 1)

# 设置标题和轴标签
plt.xlabel("", fontsize=14, fontweight='bold')
plt.ylabel("SSIM Score", fontsize=14, fontweight='bold')
plt.title("Medium Smoke", fontsize=16, fontweight='bold')

# 添加图例
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=palette[i], alpha=0.5) for i in range(len(mean_values))],
           labels=mean_values.keys(), loc='upper left', fontsize=12)

# 添加网格
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 显示图表
plt.show()

print(f"Data saved to {excel_filename}")
