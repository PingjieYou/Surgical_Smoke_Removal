import matplotlib.pyplot as plt

# 模型数据（GFLOPs 和参数量，单位已调整为 GFLOPs 和百万参数）
models = ["PFAN", "SwinUNet", "U-Net256", "U-Net128", "ResNet9", "ResNet6"]
gflops = [22.63299008, 11.83132877, 6.063034368, 6.037848064, 49.61651917, 35.10842163]
params = [0.629421, 41.393028, 54.413955, 41.828995, 11.383427, 7.841411]

# 颜色和形状的配置
colors = ['#86BBD8', '#E3B0A7', '#C4E1D0', '#F5C2A3', '#D8A4D0', '#B6D7A8']
markers = ['o', 's', '^', 'D', 'v', '*']  # 各种不同形状的标记

# 设置图表背景颜色为白色
plt.figure(figsize=(10, 6), facecolor="white")

# 创建散点图，每个模型使用不同的形状和颜色
for i, model in enumerate(models):
    plt.scatter(gflops[i], params[i], color=colors[i], alpha=0.8, s=150, edgecolors='black', linewidth=1.5,
                marker=markers[i], label=model)

# 标注每个点的模型名称，位置调整为在散点的上方
for i, model in enumerate(models):
    plt.text(gflops[i], params[i] + 1.5, model, fontsize=12, ha="center", va="bottom", color="#3e3e3e", fontweight="bold", fontname="Times New Roman")

# 设置坐标轴和标题
plt.xlabel("GFLOPs", fontsize=14, fontweight="bold", fontname="Times New Roman", color="#3e3e3e")
plt.ylabel("Params (Millions)", fontsize=14, fontweight="bold", fontname="Times New Roman", color="#3e3e3e")
plt.title("GFLOPs vs. Parameters", fontsize=16, fontweight="bold", fontname="Times New Roman", color="#3e3e3e")

# 美化网格线
plt.grid(True, linestyle="--", alpha=0.5, color="#d3d3d3")

# 显示图例
plt.legend(facecolor='white', edgecolor='black', fontsize=12, loc='upper right')

# 设置X轴和Y轴的范围，确保图形不过于拥挤
plt.xlim(0, 55)
plt.ylim(0, 60)

# 显示图表
plt.show()
