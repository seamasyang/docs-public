# 文件: codes/ann-basic.py
# 用途: 手动实现一个2层隐藏层的前馈神经网络的前向传播和反向传播（梯度下降一次更新）

import numpy as np

# ---------- 初始化参数 ----------
# 固定随机种子，使结果可复现
np.random.seed(42)

# 输入层→第1隐藏层的权重矩阵，形状 (3,2)，即3个神经元、2个输入特征
W1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]], dtype=float)
# 第1隐藏层的偏置向量，形状 (3,)
b1 = np.array([0.01, 0.02, 0.03], dtype=float)

# 第1隐藏层→第2隐藏层的权重矩阵，形状 (2,3)
W2 = np.array([[0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2]], dtype=float)
# 第2隐藏层的偏置向量，形状 (2,)
b2 = np.array([0.04, 0.05], dtype=float)

# 第2隐藏层→输出层的权重矩阵，形状 (1,2)
W3 = np.array([[1.3, 1.4]], dtype=float)
# 输出层偏置，形状 (1,)
b3 = np.array([0.06], dtype=float)

# ---------- 输入和目标值 ----------
# 单个样本的两个特征
x = np.array([1.0, 2.0])
# 真实的目标值（回归问题）
y = 0.5

# ---------- 前向传播 ----------

# 第1隐藏层的线性计算: z1 = W1 * x + b1
# W1 @ x 做矩阵-向量乘法，结果形状 (3,)，再加上偏置 b1
z1 = W1 @ x + b1                # 形状 (3,)

# 通过ReLU激活函数: a1 = max(0, z1)
a1 = np.maximum(0, z1)          # 形状 (3,)

# 第2隐藏层的线性计算: z2 = W2 * a1 + b2
z2 = W2 @ a1 + b2               # 形状 (2,)

# ReLU激活
a2 = np.maximum(0, z2)          # 形状 (2,)

# 输出层线性计算（无激活函数，用于回归问题）
z3 = W3 @ a2 + b3               # 形状 (1,) 因为 W3 是 (1,2)，a2 是 (2,)

# 取出唯一的预测值
y_pred = z3[0]

# 计算均方误差损失的一半（简化求导用）
loss = 0.5 * (y_pred - y)**2

print(f"前向: y_pred={y_pred:.4f}, loss={loss:.4f}")

# ---------- 反向传播 ----------

# 输出层误差: delta3 = dL/dz3 = (y_pred - y) * 1  (因为损失函数导数为 y_pred - y)
delta3 = y_pred - y               # 标量

# 输出层权重梯度: dL/dW3 = delta3 * a2^T，形状 (1,2)
dW3 = delta3 * a2.reshape(1, -1)  # a2 原本 (2,)，reshape为 (1,2) 后与标量相乘

# 输出层偏置梯度: dL/db3 = delta3 (形状 (1,))
db3 = np.array([delta3])

# ---------- 第2隐藏层反向传播 ----------
# ReLU导数: 当 z2 > 0 时导数为1，否则为0
drelu2 = (z2 > 0).astype(float)   # 形状 (2,)

# 第2隐藏层误差: delta2 = (W3^T * delta3) ⊙ ReLU'(z2)
# 注意: W3.T 形状 (2,1)，delta3 是标量，所以用 * 可以实现逐元素（广播）乘法
# .squeeze() 去掉长度为1的维度，得到 (2,) 向量
delta2 = (W3.T * delta3).squeeze() * drelu2   # 形状 (2,)

# 第2隐藏层权重梯度: dL/dW2 = delta2 * a1^T，形状 (2,3)
dW2 = np.outer(delta2, a1)        # np.outer 计算外积

# 第2隐藏层偏置梯度: dL/db2 = delta2，形状 (2,)
db2 = delta2

# ---------- 第1隐藏层反向传播 ----------
# ReLU导数
drelu1 = (z1 > 0).astype(float)   # 形状 (3,)

# 第1隐藏层误差: delta1 = (W2^T @ delta2) ⊙ ReLU'(z1)
# W2.T 形状 (3,2)，delta2 形状 (2,)，矩阵乘法后得到 (3,)
delta1 = (W2.T @ delta2) * drelu1  # 形状 (3,)

# 第1隐藏层权重梯度: dL/dW1 = delta1 * x^T，形状 (3,2)
dW1 = np.outer(delta1, x)

# 第1隐藏层偏置梯度: dL/db1 = delta1，形状 (3,)
db1 = delta1

# 打印各层权重梯度（便于观察数值）
print("梯度 dW1:\n", dW1)
print("梯度 dW2:\n", dW2)
print("梯度 dW3:\n", dW3)

# ---------- 参数更新（随机梯度下降，学习率 lr = 0.01）----------
lr = 0.01

# 更新第1隐藏层的权重和偏置
W1 -= lr * dW1
W2 -= lr * dW2
# W3 梯度 dW3 形状 (1,2)，需要 reshape 为与 W3 一致的形状
W3 -= lr * dW3.reshape(1, 2)

b1 -= lr * db1
b2 -= lr * db2
b3 -= lr * db3

print("更新后一次的参数。")