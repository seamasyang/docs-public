import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']          
plt.rcParams['axes.unicode_minus'] = False  

# 1. 直观理解 - 什么是"熵"？
def entropy_demonstration():
    """
    直观理解熵的概念
    """
    print("="*70)
    print("                   熵的概念理解")
    print("="*70)
    
    # 场景1：确定的事件（低熵）
    print("\n🎯 场景1：几乎确定的事件")
    print("   一个装满了99个红球和1个蓝球的袋子")
    print("   猜测结果：几乎肯定是红球，不确定性很低")
    prob1 = np.array([0.99, 0.01])  # 红球概率99%，蓝球1%
    entropy1 = -np.sum(prob1 * np.log2(prob1))
    print(f"   熵值: {entropy1:.4f} bits (低熵 = 低不确定性)")
    
    # 场景2：完全不确定（高熵）
    print("\n🎲 场景2：完全不确定的事件")
    print("   抛一枚公平的硬币")
    print("   猜测结果：完全不确定是正面还是反面")
    prob2 = np.array([0.5, 0.5])  # 各50%概率
    entropy2 = -np.sum(prob2 * np.log2(prob2))
    print(f"   熵值: {entropy2:.4f} bits (高熵 = 高不确定性)")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：概率分布对比
    scenarios = ['确定事件\n(99%红球)', '不确定事件\n(公平硬币)']
    entropies = [entropy1, entropy2]
    bars = axes[0].bar(scenarios, entropies, color=['green', 'red'], alpha=0.7)
    axes[0].set_ylabel('熵值 (bits)')
    axes[0].set_title('熵：衡量不确定性')
    axes[0].set_ylim([0, 1.2])
    for bar, entropy in zip(bars, entropies):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{entropy:.3f}', ha='center', fontsize=12)
    
    # 右图：二分类的熵函数
    p = np.linspace(0.001, 0.999, 100)
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
    axes[1].plot(p, entropy, linewidth=2, color='purple')
    axes[1].set_xlabel('某个类别的概率')
    axes[1].set_ylabel('熵值 (bits)')
    axes[1].set_title('二分类的熵函数')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='最大熵')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# entropy_demonstration()

# 2. 从熵到交叉熵
# 信息熵（Entropy）：描述一个真实分布的不确定性
# 交叉熵（Cross-Entropy）：用预测分布 q 来描述真实分布 p 所需的"额外信息量"
def cross_entropy_vs_entropy():
    """
    对比信息熵和交叉熵
    """
    print("\n" + "="*70)
    print("              熵 vs 交叉熵的区别")
    print("="*70)
    
    # 真实分布（one-hot编码）
    p = np.array([1.0, 0.0, 0.0])  # 真实类别是第0类
    
    # 不同质量的预测
    predictions = {
        '完美预测': np.array([0.999, 0.001, 0.000]),
        '良好预测': np.array([0.8, 0.1, 0.1]),
        '一般预测': np.array([0.6, 0.2, 0.2]),
        '较差预测': np.array([0.3, 0.3, 0.4]),
        '错误预测': np.array([0.1, 0.8, 0.1]),
    }
    
    print(f"\n真实分布 p: {p} (类别0)")
    print(f"信息熵 H(p): {-np.sum(p * np.log2(p + 1e-10)):.4f} bits\n")
    
    for name, q in predictions.items():
        # 交叉熵计算（以2为底的对数）
        cross_ent = -np.sum(p * np.log2(q + 1e-10))
        # KL散度（额外信息量）
        kl_div = cross_ent - (-np.sum(p * np.log2(p + 1e-10)))
        
        print(f"{name}:")
        print(f"  预测分布 q: {q}")
        print(f"  交叉熵 H(p,q): {cross_ent:.4f} bits")
        print(f"  额外信息量: {kl_div:.4f} bits\n")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prediction_names = list(predictions.keys())
    cross_entropies = []
    
    for name, q in predictions.items():
        ce = -np.sum(p * np.log2(q + 1e-10))
        cross_entropies.append(ce)
    
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    bars = ax.barh(prediction_names, cross_entropies, color=colors, alpha=0.7)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='理想值 (0)')
    ax.set_xlabel('交叉熵 (bits)')
    ax.set_title('不同预测质量下的交叉熵')
    ax.legend()
    
    for bar, ce in zip(bars, cross_entropies):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2.,
               f'{ce:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

# cross_entropy_vs_entropy()

#3. 在机器学习中的应用
def cross_entropy_in_ml():
    """
    交叉熵在机器学习中的实际应用
    """
    print("\n" + "="*70)
    print("           机器学习中的交叉熵损失")
    print("="*70)
    
    # 模拟MNIST数字识别的场景
    print("\n📝 场景：手写数字识别（类别0-9）")
    
    # 真实标签（数字5）
    y_true = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
    print(f"真实标签（数字5）: {y_true}")
    
    # 不同模型的预测
    models_predictions = {
        '模型A（优秀）': np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.85, 0.02, 0.02, 0.02, 0.02]),
        '模型B（一般）': np.array([0.05, 0.05, 0.05, 0.10, 0.05, 0.40, 0.10, 0.05, 0.10, 0.05]),
        '模型C（混淆）': np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.30, 0.50, 0.05, 0.03, 0.02]),
        '模型D（错误）': np.array([0.02, 0.02, 0.02, 0.70, 0.10, 0.05, 0.03, 0.02, 0.02, 0.02]),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (model_name, y_pred) in enumerate(models_predictions.items()):
        ax = axes[idx]
        
        # 计算交叉熵损失
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        ce_loss = -np.sum(y_true * np.log(y_pred_clipped))
        
        # 绘制预测概率
        digits = range(10)
        bars = ax.bar(digits, y_pred, alpha=0.7, color='lightblue')
        
        # 高亮真实类别
        bars[5].set_color('green')
        bars[5].set_alpha(0.9)
        
        # 高亮错误预测（如果不是5）
        if np.argmax(y_pred) != 5:
            bars[np.argmax(y_pred)].set_color('red')
            bars[np.argmax(y_pred)].set_alpha(0.7)
        
        ax.set_title(f'{model_name}\n交叉熵损失: {ce_loss:.4f}')
        ax.set_xlabel('数字类别')
        ax.set_ylabel('预测概率')
        ax.set_xticks(digits)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加概率标签
        for digit, prob in enumerate(y_pred):
            if prob > 0.1:  # 只显示较大的概率
                ax.text(digit, prob + 0.02, f'{prob:.2f}', 
                       ha='center', fontsize=8)
    
    plt.suptitle('不同模型预测的交叉熵损失', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 详细计算过程
    print("\n" + "-"*40)
    print("详细计算示例（以模型A为例）：")
    print("-"*40)
    
    y_pred = models_predictions['模型A（优秀）']
    y_pred_safe = np.clip(y_pred, 1e-15, 1-1e-15)
    
    print(f"预测概率: {y_pred_safe}")
    print(f"\n计算步骤:")
    print(f"  1. 取对数:")
    log_probs = np.log(y_pred_safe)
    for i, lp in enumerate(log_probs):
        print(f"     log(概率{i}) = {lp:.4f}")
    
    print(f"\n  2. 与真实标签相乘:")
    weighted = -y_true * log_probs
    for i, w in enumerate(weighted):
        print(f"     -真实标签{i} × log(概率{i}) = {w:.4f}")
    
    print(f"\n  3. 求和:")
    ce = np.sum(weighted)
    print(f"     交叉熵 = {ce:.4f}")
    
    print(f"\n  💡 注意：只有真实类别（数字5）的项非零！")
    print(f"     因此交叉熵 = -log(模型对数字5的预测概率)")
    print(f"     = -log({y_pred_safe[5]:.4f}) = {ce:.4f}")

# cross_entropy_in_ml()

# 4. 交叉熵损失的梯度推导
def cross_entropy_gradient_demo():
    """
    演示交叉熵损失 + Softmax的梯度
    """
    print("\n" + "="*70)
    print("           交叉熵损失的梯度推导")
    print("="*70)
    
    def softmax(x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(y_true, y_pred):
        """交叉熵损失"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))
    
    print("\n🎯 证明：Softmax + 交叉熵的梯度 = 预测 - 真实")
    print("-"*50)
    
    # 示例数据
    logits = np.array([[2.0, 1.0, 0.5]])  # 最后一层的输出
    y_true = np.array([[0., 1., 0.]])      # 真实标签（类别1）
    
    print(f"输入 logits: {logits[0]}")
    print(f"真实标签: {y_true[0]}")
    
    # 前向传播
    y_pred = softmax(logits)
    print(f"\n前向传播:")
    print(f"  Softmax输出: {y_pred[0]}")
    
    loss = cross_entropy_loss(y_true, y_pred)
    print(f"  交叉熵损失: {loss:.4f}")
    
    # 梯度计算
    gradient = y_pred - y_true
    print(f"\n反向传播:")
    print(f"  梯度 (∂L/∂logits): {gradient[0]}")
    print(f"  简化公式: 预测概率 - 真实标签")
    
    # 验证梯度
    epsilon = 1e-5
    numerical_grad = np.zeros_like(logits)
    
    for i in range(logits.shape[1]):
        logits_plus = logits.copy()
        logits_plus[0, i] += epsilon
        y_pred_plus = softmax(logits_plus)
        loss_plus = cross_entropy_loss(y_true, y_pred_plus)
        
        logits_minus = logits.copy()
        logits_minus[0, i] -= epsilon
        y_pred_minus = softmax(logits_minus)
        loss_minus = cross_entropy_loss(y_true, y_pred_minus)
        
        numerical_grad[0, i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    print(f"\n数值梯度验证:")
    print(f"  解析梯度: {gradient[0]}")
    print(f"  数值梯度: {numerical_grad[0]}")
    print(f"  差异: {np.abs(gradient - numerical_grad)[0]}")
    
    # 可视化梯度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：预测与真实
    x = np.arange(3)
    width = 0.35
    bars1 = ax1.bar(x - width/2, y_true[0], width, label='真实', alpha=0.8, color='green')
    bars2 = ax1.bar(x + width/2, y_pred[0], width, label='预测', alpha=0.8, color='blue')
    ax1.set_title('真实标签 vs 预测概率')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['类别0', '类别1', '类别2'])
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # 右图：梯度
    colors = ['green' if g > 0 else 'red' for g in gradient[0]]
    ax2.bar(x, gradient[0], color=colors, alpha=0.7)
    ax2.set_title('梯度值 (∂L/∂logits)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['类别0', '类别1', '类别2'])
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    # 添加数值标签
    for i, grad in enumerate(gradient[0]):
        ax2.text(i, grad + 0.05, f'{grad:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return "美丽的是：Softmax + 交叉熵的梯度 = 预测 - 真实，如此简洁！"

# cross_entropy_gradient_demo()

# 5. 完整的训练示例
def cross_entropy_training_demo():
    """
    演示交叉熵损失如何指导模型训练
    """
    print("\n" + "="*70)
    print("       交叉熵损失指导训练的过程")
    print("="*70)
    
    # 模拟一个简单的二分类问题
    np.random.seed(42)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def binary_cross_entropy(y_true, y_pred):
        """二分类交叉熵"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    print("\n🔍 模拟二分类问题训练过程")
    print("-"*50)
    
    # 生成数据
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # 初始化参数
    W = np.random.randn(2, 1) * 0.01
    b = np.zeros((1, 1))
    
    # 训练过程
    learning_rate = 0.1
    losses = []
    accuracies = []
    
    print("训练开始...")
    print("Epoch | Loss   | Accuracy | W")
    print("-"*50)
    
    for epoch in range(100):
        # 前向传播
        logits = X @ W + b
        y_pred = sigmoid(logits)
        
        # 计算损失
        loss = binary_cross_entropy(y, y_pred)
        accuracy = np.mean((y_pred > 0.5) == y)
        
        # 反向传播
        dL_dpred = -(y / y_pred - (1 - y) / (1 - y_pred)) / len(y)
        dpred_dlogits = y_pred * (1 - y_pred)
        dL_dlogits = dL_dpred * dpred_dlogits
        
        dW = X.T @ dL_dlogits
        db = np.sum(dL_dlogits, axis=0, keepdims=True)
        
        # 更新参数
        W -= learning_rate * dW
        b -= learning_rate * db
        
        losses.append(loss)
        accuracies.append(accuracy)
        
        if epoch % 20 == 0:
            print(f"{epoch:3d}   | {loss:.4f} | {accuracy:.4f}   | {W.T[0]}")
    
    # 可视化训练过程
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 损失曲线
    axes[0, 0].plot(losses, linewidth=2, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].set_title('Training Loss Decrease')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    axes[0, 1].plot(accuracies, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy Increase')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # 3. 决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ W + b)
    Z = Z.reshape(xx.shape)
    
    axes[1, 0].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = axes[1, 0].scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                                 cmap='RdYlBu', edgecolor='black')
    axes[1, 0].set_title('Decision Boundary')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # 4. 损失与准确率的关系
    axes[1, 1].scatter(losses, accuracies, alpha=0.5, c=range(len(losses)), 
                      cmap='viridis')
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Loss vs Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Epoch')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ 训练完成！")
    print(f"   最终损失: {losses[-1]:.4f}")
    print(f"   最终准确率: {accuracies[-1]:.4f}")

# cross_entropy_training_demo()

# 6. 关键总结
def cross_entropy_summary():
    """
    交叉熵损失的关键概念总结
    """
    print("\n" + "="*70)
    print("                 📚 交叉熵损失 - 学习总结")
    print("="*70)
    
    summary = """
    
    🎯 核心概念：
    ┌─────────────────────────────────────────────────────────┐
    │ 交叉熵衡量：使用预测分布 q 描述真实分布 p 的"代价"      │
    │ H(p,q) = -Σ pᵢ log(qᵢ)                                │
    └─────────────────────────────────────────────────────────┘
    
    💡 为什么使用交叉熵？
    
    1. 概率解释
       ✓ 输出可以理解为概率
       ✓ 损失有明确的信息论基础
       ✓ 预测越准确，损失越小
    
    2. 梯度性质
       ✓ Softmax + 交叉熵的梯度非常简洁
       ✓ ∂L/∂logits = y_pred - y_true
       ✓ 避免了梯度消失问题
    
    3. 数值稳定性
       ✓ 配合Softmax使用时数值稳定
       ✓ 指数运算被抵消
       ✓ 防止log(0)的出现
    
    ⚖️ 对比其他损失函数：
    
    损失函数        | 适用场景        | 优点              | 缺点
    ─────────────────────────────────────────────────────
    交叉熵          | 分类问题        | 概率输出          | 对异常值敏感
                    |                | 梯度简单          |
    ─────────────────────────────────────────────────────
    均方误差(MSE)   | 回归问题        | 计算简单          | 分类时梯度小
                    |                | 凸函数            | 学习慢
    ─────────────────────────────────────────────────────
    Hinge Loss     | SVM分类        | 稀疏解            | 不输出概率
                    |                | 对异常值鲁棒      |
    
    🔑 关键公式：
    
    1. 标准形式（多分类）：
       L = -Σᵢ yᵢ log(ŷᵢ)
       其中 yᵢ 是真实标签（one-hot），ŷᵢ 是预测概率
    
    2. 二分类特殊情况：
       L = -[y log(ŷ) + (1-y) log(1-ŷ)]
    
    3. 在神经网络中：
       前向: ŷ = softmax(Wx + b)
       损失: L = -Σ y log(ŷ)
       梯度: ∂L/∂x = ŷ - y  （惊人的简洁！）
    
    ⚠️ 注意事项：
    
    1. 必须确保预测值在 (0,1) 之间
    2. 避免 log(0)：添加小的 epsilon
    3. 最好配合 softmax/sigmoid 使用
    4. 真实标签应该是概率分布（one-hot或软标签）
    
    🚀 实践建议：
    
    1. 分类问题默认使用交叉熵
    2. 配合Adam优化器效果更好
    3. 监控训练和验证的交叉熵
    4. 交叉熵下降 ≈ 模型在学习
    5. 训练集交叉熵远低于验证集 ≈ 过拟合
    
    """
    
    print(summary)
    
    # 创建知识图谱
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 中心概念
    ax.text(0.5, 0.5, '交叉熵损失\nCross-Entropy Loss', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 相关概念
    concepts = {
        '信息熵': (0.2, 0.8),
        'KL散度': (0.5, 0.85),
        'Softmax': (0.8, 0.8),
        '梯度下降': (0.8, 0.2),
        '概率分布': (0.2, 0.2),
        '最大似然': (0.5, 0.15),
    }
    
    for concept, (x, y) in concepts.items():
        ax.text(x, y, concept, ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        ax.plot([0.5, x], [0.5, y], 'gray', alpha=0.3)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('交叉熵损失 - 知识地图', fontsize=14, fontweight='bold', pad=20)
    
    plt.show()

cross_entropy_summary()