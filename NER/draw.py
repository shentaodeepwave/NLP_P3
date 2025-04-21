import matplotlib.pyplot as plt

# 数据
iterations = list(range(1, 50))  # 迭代次数
log_likelihood = [
    -0.69315, -0.08337, -0.07576, -0.06593, -0.05773, -0.05144, -0.04662, -0.04285, -0.03983, -0.03736,
    -0.03531, -0.03357, -0.03207, -0.03078, -0.02964, -0.02863, -0.02773, -0.02692, -0.02618, -0.02551,
    -0.02490, -0.02433, -0.02381, -0.02332, -0.02287, -0.02245, -0.02206, -0.02168, -0.02133, -0.02100,
    -0.02069, -0.02039, -0.02011, -0.01984, -0.01959, -0.01934, -0.01911, -0.01888, -0.01867, -0.01846,
    -0.01827, -0.01807, -0.01789, -0.01771, -0.01754, -0.01738, -0.01722, -0.01707, -0.01692
]
accuracy = [
    0.055, 0.945, 0.945, 0.954, 0.969, 0.976, 0.980, 0.983, 0.985, 0.986,
    0.987, 0.987, 0.988, 0.989, 0.989, 0.989, 0.990, 0.990, 0.990, 0.991,
    0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.992, 0.992, 0.992, 0.992,
    0.992, 0.992, 0.992, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993,
    0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.994, 0.994, 0.994
]

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 创建并列的两个子图

# 绘制对数似然值曲线
ax1.plot(iterations, log_likelihood, color='tab:blue', linestyle='-', linewidth=2)
ax1.set_title('Log Likelihood over Iterations', fontsize=14)
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Log Likelihood', fontsize=12)
ax1.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 绘制准确率曲线
ax2.plot(iterations, accuracy, color='tab:orange', linestyle='--', linewidth=2)
ax2.set_title('Accuracy over Iterations', fontsize=14)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()