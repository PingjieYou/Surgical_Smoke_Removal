import matplotlib.pyplot as plt
import numpy as np

# 数据
phases = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
precision = [0.9796, 0.9707, 0.8811, 0.9609, 0.8353, 0.8974, 0.8848]
recall = [0.8, 0.9734, 0.9196, 0.9829, 0.9375, 0.891, 0.7377]
jaccard = [0.7782, 0.9446, 0.8192, 0.9376, 0.7869, 0.7859, 0.7008]

x = np.arange(len(phases))

# 颜色方案（浅色交替背景）
phase_bg_colors = ["#eef3f9", "#fcf5e5", "#e8f8e0", "#f4e5ff", "#ffe5e5", "#e5f3ff", "#fdf7e3"]  # 柔和蓝、米、绿、紫、粉、青、黄

# 折线颜色（浅色）
line_colors = ["#66b3ff", "#99cc99", "#ff9999"]  # 浅蓝、浅绿、浅红
grid_color = "#d0d7e1"  # 网格线颜色

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"

# 创建图像
fig, ax = plt.subplots(figsize=(12, 7))

# 添加不同 phase 之间的背景颜色
for i in range(len(phases)):
    ax.axvspan(i - 0.5, i + 0.5, color=phase_bg_colors[i], alpha=0.7)

# 绘制折线图和散点
ax.plot(x, precision, marker="o", linestyle="-", linewidth=3, markersize=12, label="Precision", color=line_colors[0])
ax.plot(x, recall, marker="s", linestyle="-", linewidth=3, markersize=12, label="Recall", color=line_colors[1])
ax.plot(x, jaccard, marker="^", linestyle="-", linewidth=3, markersize=12, label="Jaccard", color=line_colors[2])

# 标注散点数值（带虚线指向）
for i in range(len(phases)):
    # Precision (上方)
    ax.plot([x[i], x[i]], [precision[i], precision[i] + 0.03], linestyle="--", color=line_colors[0], alpha=0.7)
    ax.text(x[i], precision[i] + 0.05, f"{precision[i]:.3f}", ha="center", fontsize=12, color=line_colors[0])

    # Recall (右侧)
    ax.plot([x[i], x[i] + 0.1], [recall[i], recall[i] + 0.03], linestyle="--", color=line_colors[1], alpha=0.7)
    ax.text(x[i] + 0.12, recall[i] + 0.04, f"{recall[i]:.3f}", ha="left", fontsize=12, color=line_colors[1])

    # Jaccard (下方)
    ax.plot([x[i], x[i]], [jaccard[i], jaccard[i] - 0.03], linestyle="--", color=line_colors[2], alpha=0.7)
    ax.text(x[i], jaccard[i] - 0.05, f"{jaccard[i]:.3f}", ha="center", fontsize=12, color=line_colors[2])

# 坐标轴优化
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=14, fontweight="bold", rotation=0)  # X轴刻度增大
ax.set_yticks(np.linspace(0.7, 1.0, 7))  # Y轴刻度固定范围，避免贴边
ax.set_ylim(0.7, 1.02)  # Y轴上方多预留一点空间
ax.tick_params(axis="y", labelsize=12)  # Y轴刻度增大
ax.spines["top"].set_visible(False)  # 隐藏上方轴线
ax.spines["right"].set_visible(False)  # 隐藏右侧轴线
ax.spines["left"].set_linewidth(1.5)  # 左轴加粗
ax.spines["bottom"].set_linewidth(1.5)  # 下轴加粗

# 移除 X 轴标签
ax.set_xlabel("")

# 移除 Y 轴标签
ax.set_ylabel("")

# 移除标题
ax.set_title("")

# 网格优化（只显示 Y 轴的水平网格线）
ax.grid(axis="y", linestyle="--", alpha=0.5, color=grid_color)
ax.grid(axis="x", linestyle="--", alpha=0)  # 隐藏 X 轴网格线

# 显示图例
ax.legend(fontsize=12, loc="upper right", frameon=True, shadow=False, facecolor="white", edgecolor=grid_color)

# 显示图表
plt.show()
