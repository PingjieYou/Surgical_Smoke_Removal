import matplotlib.pyplot as plt
import numpy as np

# 假设有6个模型，每个模型有4个不同架构的值
models = ['PFAN', 'SwinUNet', 'UNet256', 'UNet128', 'ResNet9', 'ResNet6']
architecture_1 = [9.444, 14.103, 11.422, 10.873, 11.883, 11.867]  # 架构1的值
architecture_2 = [7.530, 5.362, 6.113, 6.433, 9.321, 9.293]  # 架构2的值
architecture_3 = [4.713, 4.675, 4.353, 4.432, 5.872, 5.300]  # 架构3的值
architecture_4 = [4.753, 5.174, 4.193, 4.737, 5.839, 5.454] # 架构4的值

# 创建一个宽度为0.25的条形图
bar_width = 0.25
index = np.arange(len(models))  # 横坐标位置

# 创建柱状图
fig, ax = plt.subplots(figsize=(5, 4))

# 使用新的优雅的颜色方案
bar1 = ax.bar(index - 1.5 * bar_width, architecture_1, bar_width, label='CycleGAN', color='#98FF98')  # 薄荷绿
bar2 = ax.bar(index - 0.5 * bar_width, architecture_2, bar_width, label='Ours+CycleGAN', color='#A9A9A9')  # 烟灰色
bar3 = ax.bar(index + 0.5 * bar_width, architecture_3, bar_width, label='Pix2pix', color='#4682B4')  # 深海蓝
bar4 = ax.bar(index + 1.5 * bar_width, architecture_4, bar_width, label='Ours+Pix2pix', color='#D2B48C')  # 黄褐色

# 添加数值显示，调整字体大小并避免超过柱宽
for bar in bar1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=5)
for bar in bar2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=5)
for bar in bar3:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=5)
for bar in bar4:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=5)

# 添加标签和标题
ax.set_xlabel('', fontsize=10, fontweight='bold')
ax.set_ylabel('', fontsize=10, fontweight='bold')
ax.set_title('Heavy Smoke', fontsize=12, fontweight='bold')

# 设置 x 坐标轴标签
ax.set_xticks(index)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)

# 移除坐标轴的刻度线和框架
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.xaxis.set_ticks_position('none')  # 移除底部坐标轴
ax.yaxis.set_ticks_position('none')  # 移除左侧坐标轴

# 设置网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# 设置图例位置，移到图下方
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=4, fontsize=8)

# 显示图表
plt.tight_layout()
plt.show()
