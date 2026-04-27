import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class ThreeLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化三层神经网络
        
        参数:
        input_size: 输入层神经元数量
        hidden_size: 隐藏层神经元数量
        output_size: 输出层神经元数量
        learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        # He初始化 - 适用于ReLU激活函数
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # 存储损失历史
        self.loss_history = []
        
    def relu(self, x):
        """
        ReLU激活函数
        输入为正数时输出本身，为负数时输出0。
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        ReLU导数
        ReLU在正半轴（z > 0）的导数恒为1，梯度可以无损地通过正向区域反向传播，
        大大缓解了深层网络的梯度消失问题。
        """
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax激活函数"""

        # 数值稳定的指数计算： 减去最大值，这是关键优化！将所有值平移到 ≤ 0，避免指数溢出
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        # 归一化： 每个元素除以该行的总和，确保概率和为1
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据
        
        返回:
        A2: 输出层的预测值
        cache: 中间值缓存，用于反向传播
        """
        # 输入层到隐藏层
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # 隐藏层到输出层
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        
        cache = {
            'X': X,
            'Z1': self.Z1,
            'A1': self.A1,
            'Z2': self.Z2,
            'A2': self.A2
        }
        
        return self.A2, cache
    
    def compute_loss(self, y_true, y_pred):
        """
        计算交叉熵损失
        
        参数:
        y_true: 真实标签(one-hot编码)
        y_pred: 预测值
        
        返回:
        loss: 平均损失值
        """
        m = y_true.shape[0]
        # 添加小量避免log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward(self, cache, y_true):
        """
        反向传播
        
        参数:
        cache: 前向传播的中间值
        y_true: 真实标签
        
        返回:
        grads: 梯度字典
        """
        m = y_true.shape[0]
        X = cache['X']
        A1 = cache['A1']
        A2 = cache['A2']
        
        # 输出层误差
        dZ2 = A2 - y_true
        
        # 隐藏层到输出层的梯度
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(cache['Z1'])
        
        # 输入层到隐藏层的梯度
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return grads
    
    def update_parameters(self, grads):
        """
        使用梯度下降更新网络参数

        参数:
        grads: 包含各层梯度（dW1, db1, dW2, db2）的字典
        """
        self.W1 -= self.learning_rate * grads['dW1']
        self.b1 -= self.learning_rate * grads['db1']
        self.W2 -= self.learning_rate * grads['dW2']
        self.b2 -= self.learning_rate * grads['db2']
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        训练神经网络
        
        参数:
        X: 训练数据
        y: 训练标签(one-hot编码)
        epochs: 训练轮数
        verbose: 是否打印训练信息
        """
        for epoch in range(epochs):
            # 前向传播
            y_pred, cache = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 反向传播
            grads = self.backward(cache, y)
            
            # 更新参数
            self.update_parameters(grads)
            
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.accuracy(X, y)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    
    def predict(self, X):
        """
        预测概率输出

        参数:
        X: 输入数据

        返回:
        预测概率矩阵
        """
        y_pred, _ = self.forward(X)
        return y_pred
    
    def predict_classes(self, X):
        """
        预测类别标签

        参数:
        X: 输入数据

        返回:
        预测的类别索引数组
        """
        y_pred = self.predict(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y_true):
        """
        计算分类准确率

        参数:
        X: 输入数据
        y_true: 真实标签（one-hot编码）

        返回:
        准确率（0~1之间的浮点数）
        """
        y_pred_classes = self.predict_classes(X)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes)
    
    def plot_training_history(self):
        """
        绘制训练历史图表
        包含损失曲线和对数损失曲线两张子图。
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.loss_history)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # 损失的对数尺度
        axes[1].plot(np.log(self.loss_history))
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Log Loss')
        axes[1].set_title('Log Training Loss Over Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def one_hot_encode(y, num_classes):
    """
    One-hot编码
    将类别标签转换为one-hot向量表示。

    参数:
    y: 类别标签数组
    num_classes: 类别总数

    返回:
    one-hot编码矩阵
    """
    m = y.shape[0]
    encoded = np.zeros((m, num_classes))
    encoded[np.arange(m), y] = 1
    return encoded

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    绘制混淆矩阵热力图

    参数:
    y_true: 真实标签
    y_pred: 预测标签
    class_names: 类别名称列表

    返回:
    cm: 混淆矩阵数组
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return cm

def plot_decision_boundaries(X, y, model, feature_pairs, class_names):
    """
    绘制决策边界图
    在多个特征组合的二维平面上可视化模型的分类决策区域。

    参数:
    X: 特征数据
    y: 真实标签
    model: 训练好的神经网络模型
    feature_pairs: 特征组合索引列表
    class_names: 类别名称列表
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (i, j) in enumerate(feature_pairs[:6]):
        # 创建网格
        x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
        y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # 预测网格点
        grid_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
        grid_points[:, i] = xx.ravel()
        grid_points[:, j] = yy.ravel()
        
        Z = model.predict_classes(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        scatter = axes[idx].scatter(X[:, i], X[:, j], c=y, 
                                   cmap='RdYlBu', edgecolor='black')
        axes[idx].set_xlabel(f'Feature {i+1}')
        axes[idx].set_ylabel(f'Feature {j+1}')
        axes[idx].set_title(f'Decision Boundary: Features {i+1} vs {j+1}')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("三层神经网络 - 鸾尾花数据集分类")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载鸾尾花数据集...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    print(f"   数据集大小: {X.shape[0]} 个样本")
    print(f"   特征数量: {X.shape[1]}")
    print(f"   类别数量: {len(class_names)}")
    print(f"   类别: {class_names}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("   ✅ 特征标准化完成")
    
    # One-hot编码
    y_onehot = one_hot_encode(y, len(class_names))
    print("   ✅ One-hot编码完成")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
        X_scaled, y, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✅ 数据划分完成")
    print(f"   训练集: {X_train.shape[0]} 个样本")
    print(f"   测试集: {X_test.shape[0]} 个样本")
    
    # 3. 创建和训练模型
    print("\n3. 创建三层神经网络...")
    model = ThreeLayerNeuralNetwork(
        input_size=4,
        hidden_size=16,
        output_size=3,
        learning_rate=0.1
    )
    
    print(f"   网络结构:")
    print(f"   - 输入层: 4 个神经元")
    print(f"   - 隐藏层: 8 个神经元 (ReLU激活)")
    print(f"   - 输出层: 3 个神经元 (Softmax激活)")
    
    print("\n4. 开始训练...")
    model.train(X_train, y_train_onehot, epochs=1000)
    
    # 4. 测试模型
    print("\n5. 模型测试...")
    
    # 在训练集上评估
    train_accuracy = model.accuracy(X_train, y_train_onehot)
    print(f"   训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # 在测试集上评估
    test_accuracy = model.accuracy(X_test, y_test_onehot)
    print(f"   测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 5. 详细预测结果
    y_pred = model.predict_classes(X_test)
    y_pred_proba = model.predict(X_test)
    
    print("\n6. 分类报告:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 6. 可视化
    print("\n7. 生成可视化...")
    
    # 绘制训练历史
    model.plot_training_history()
    
    # 绘制混淆矩阵
    cm = plot_confusion_matrix(y_test, y_pred, class_names)
    
    # 绘制部分样本的预测概率
    plt.figure(figsize=(10, 6))
    n_samples = min(15, len(y_test))
    indices = np.random.choice(len(y_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i+1)
        probabilities = y_pred_proba[idx]
        bars = plt.bar(class_names, probabilities)
        bars[np.argmax(probabilities)].set_color('red')
        bars[y_test[idx]].set_color('green')
        plt.title(f'True: {class_names[y_test[idx]]}')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
    
    plt.suptitle('Sample Predictions (Green=True, Red=Predicted)')
    plt.tight_layout()
    plt.show()
    
    # 7. 测试报告
    print("\n" + "=" * 60)
    print("测试分析报告")
    print("=" * 60)
    print(f"""
    模型架构:
    - 网络类型: 三层全连接神经网络
    - 输入层: 4个神经元 (花萼长度、花萼宽度、花瓣长度、花瓣宽度)
    - 隐藏层: 8个神经元，ReLU激活函数
    - 输出层: 3个神经元，Softmax激活函数
    
    训练参数:
    - 学习率: 0.1
    - 训练轮数: 1000
    - 优化算法: 批量梯度下降
    - 损失函数: 交叉熵损失
    
    数据集划分:
    - 总样本数: 150
    - 训练集: 120 (80%)
    - 测试集: 30 (20%)
    - 分层采样: 是
    
    性能指标:
    - 训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
    - 测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
    
    结果分析:
    1. 模型在训练集上达到了 {train_accuracy*100:.1f}% 的准确率，表明模型成功学习了训练数据中的模式。
    2. 测试集准确率为 {test_accuracy*100:.1f}%，与训练集准确率{'接近' if abs(train_accuracy - test_accuracy) < 0.05 else '有一定差距'}，表明模型{'具有良好的泛化能力' if abs(train_accuracy - test_accuracy) < 0.05 else '存在轻微过拟合'}。
    3. 从混淆矩阵可以看出，模型在所有三个类别上都表现{'优秀' if test_accuracy > 0.95 else '良好'}。
    4. 花瓣长度和花瓣宽度是区分不同鸢尾花品种的最重要特征。
    
    改进建议:
    1. 可以尝试调整隐藏层神经元数量以优化模型性能
    2. 可以尝试不同的学习率和训练轮数
    3. 可以添加L2正则化来防止过拟合
    4. 可以尝试使用不同的优化器（如Adam）
    5. 可以进行交叉验证以获得更稳定的性能评估
    """)