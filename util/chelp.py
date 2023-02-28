import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义相机内部参数和图像分辨率
K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
width, height = 1000, 1000

# 定义相机外部参数，即旋转矩阵和平移向量
R = np.array([[0.283, 0.119, 0.952], [-0.958, -0.034, 0.286], [0.035, -0.992, 0.122]])
T = np.array([1, 2, 3])

# 定义相机视锥的六个面的方程
# 设置相机视锥的六个面
left = np.array([-width/2, 0, 0, 1])
right = np.array([width/2, 0, 0, 1])
bottom = np.array([0, -height/2, 0, 1])
top = np.array([0, height/2, 0, 1])
near = np.array([0, 0, 0, 1])
far = np.array([0, 0, 1000, 1])

# 将向量转换到相机坐标系下
left = np.linalg.inv(K) @ left[:3]
right = np.linalg.inv(K) @ right[:3]
bottom = np.linalg.inv(K) @ bottom[:3]
top = np.linalg.inv(K) @ top[:3]
near = np.linalg.inv(K) @ near[:3]
far = np.linalg.inv(K) @ far[:3]

# 将向量转换到世界坐标系下
left = R @ left[:3] + T
right = R @ right[:3] + T
bottom = R @ bottom[:3] + T
top = R @ top[:3] + T
near = R @ near[:3] + T
far = R @ far[:3] + T

# 根据相机视锥的边界绘制相机视锥
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(-left[2]/left[0]*Y, X, -left[2]/left[1]*X, alpha=0.2)
ax.plot_surface(-right[2]/right[0]*Y, X, -right[2]/right[1]*X, alpha=0.2)
ax.plot_surface(-near[2]/near[0]*Y, X, -near[2]/near[1]*X, alpha=0.2)
ax.plot_surface(-far[2]/far[0]*Y, X, -far[2]/far[1]*X, alpha=0.2)
ax.plot_surface(-bottom[2]/bottom[0]*Y, X, -bottom[2]/bottom[1]*X, alpha=0.2)
ax.plot_surface(-top[2]/top[0]*Y, X, -top[2]/top[1]*X, alpha=0.2)

# 绘制相机位置
ax.scatter(T[0], T[1], T[2], c='r')

# 设置坐标轴范围和标签
# ax.set_xlim3d(-5, 5)
# ax.set_ylim3d(-5, 5)
# ax.set_zlim3d(0, 10)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# 显示图形
plt.show()