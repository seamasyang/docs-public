# One-Hot编码的解决方案：

# 每个水果有独立的"开关"
# 苹果 = [1, 0, 0]  # 第1个位置是1，其他是0
# 香蕉 = [0, 1, 0]  # 第2个位置是1，其他是0
# 橙子 = [0, 0, 1]  # 第3个位置是1，其他是0

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

plt.rcParams['font.sans-serif'] = ['SimHei']          
plt.rcParams['axes.unicode_minus'] = False  

def one_hot_basic_concept():
    """
    One-Hot编码的基本概念
    """
    print("="*70)
    print("                One-Hot编码基础概念")
    print("="*70)
    
    # 原始数据
    fruits = ['苹果', '香蕉', '橙子', '苹果', '香蕉', '橙子', '苹果']
    
    print("\n🍎 原始数据：")
    print(f"   水果列表: {fruits}")
    
    # 手动One-Hot编码
    print("\n📊 One-Hot编码转换：")
    
    unique_fruits = list(set(fruits))
    print(f"   唯一类别: {unique_fruits}")
    
    print("\n   编码结果：")
    print("   " + "-"*40)
    print("   | 水果 | 编码向量 |")
    print("   " + "-"*40)
    
    one_hot_encoded = []
    for fruit in fruits:
        encoding = [1 if fruit == uf else 0 for uf in unique_fruits]
        one_hot_encoded.append(encoding)
        print(f"   | {fruit:4s} | {encoding} |")
    
    print("   " + "-"*40)
    
    # 转换为numpy数组
    one_hot_array = np.array(one_hot_encoded)
    print(f"\n   形状: {one_hot_array.shape}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：One-Hot矩阵
    im1 = axes[0].imshow(one_hot_array, cmap='Blues', aspect='auto')
    axes[0].set_xticks(range(len(unique_fruits)))
    axes[0].set_xticklabels(unique_fruits)
    axes[0].set_yticks(range(len(fruits)))
    axes[0].set_yticklabels(fruits)
    axes[0].set_xlabel('类别')
    axes[0].set_ylabel('样本')
    axes[0].set_title('One-Hot编码矩阵')
    
    # 添加数值标注
    for i in range(len(fruits)):
        for j in range(len(unique_fruits)):
            text = axes[0].text(j, i, one_hot_array[i, j],
                              ha="center", va="center", color="black")
    
    # 右图：每列求和（每个类别的样本数）
    category_counts = np.sum(one_hot_array, axis=0)
    bars = axes[1].bar(unique_fruits, category_counts, color=['red', 'yellow', 'orange'])
    axes[1].set_ylabel('样本数量')
    axes[1].set_title('各类别样本数量统计')
    
    for bar, count in zip(bars, category_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    str(count), ha='center')
    
    plt.tight_layout()
    plt.show()

# one_hot_basic_concept()

# 2  One-Hot编码的数学原理

def one_hot_mathematical_principle():
    """
    One-Hot编码的数学原理
    """
    print("\n" + "="*70)
    print("              One-Hot编码数学原理")
    print("="*70)
    
    # 不同的编码方式对比
    print("\n🔢 编码方式对比：")
    
    # 1. 标签编码（Label Encoding）
    print("\n1️⃣ 标签编码 (Label Encoding)：")
    categories = ['猫', '狗', '鸟', '鱼']
    label_encoded = [0, 1, 2, 3]
    
    for cat, le in zip(categories, label_encoded):
        print(f"   {cat} → {le}")
    
    print("\n   ⚠️ 问题：")
    print("   - 计算机认为：0 < 1 < 2 < 3")
    print("   - 但猫(0)并不比狗(1)'小'！")
    print("   - 欧氏距离：猫和狗的距离=1，猫和鱼的距离=3")
    print("   - 这引入了虚假的序关系")
    
    # 2. One-Hot编码
    print("\n2️⃣ One-Hot编码：")
    one_hot = np.eye(4)
    
    for cat, oh in zip(categories, one_hot):
        print(f"   {cat} → {oh}")
    
    print("\n   ✅ 优势：")
    print("   - 各类别之间距离相等")
    print("   - 没有虚假的序关系")
    
    # 计算距离矩阵
    print("\n📏 距离矩阵对比：")
    
    # 标签编码的距离
    label_coords = np.array([[0], [1], [2], [3]])
    label_distances = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            label_distances[i, j] = np.linalg.norm(label_coords[i] - label_coords[j])
    
    # One-Hot编码的距离
    oh_distances = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            oh_distances[i, j] = np.linalg.norm(one_hot[i] - one_hot[j])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左：标签编码的距离矩阵
    im1 = axes[0].imshow(label_distances, cmap='YlOrRd')
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(categories)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(categories)
    axes[0].set_title('标签编码的距离矩阵\n(虚假的距离关系)')
    plt.colorbar(im1, ax=axes[0])
    
    # 右：One-Hot编码的距离矩阵
    im2 = axes[1].imshow(oh_distances, cmap='YlOrRd')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(categories)
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(categories)
    axes[1].set_title('One-Hot编码的距离矩阵\n(所有类别距离相等)')
    plt.colorbar(im2, ax=axes[1])
    
    # 添加数值
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{label_distances[i,j]:.0f}', 
                        ha="center", va="center")
            axes[1].text(j, i, f'{oh_distances[i,j]:.1f}', 
                        ha="center", va="center")
    
    plt.tight_layout()
    plt.show()
    
    # 3. One-Hot编码的线性代数解释
    print("\n📐 线性代数解释：")
    print("   One-Hot向量 = 标准基向量（Standard Basis Vectors）")
    print("   - e₁ = [1,0,0,0] : 第1个类别")
    print("   - e₂ = [0,1,0,0] : 第2个类别")
    print("   - e₃ = [0,0,1,0] : 第3个类别")
    print("   - e₄ = [0,0,0,1] : 第4个类别")
    print("\n   这些向量：")
    print("   • 相互正交（内积为0）")
    print("   • 长度为1（单位向量）")
    print("   • 张成一个4维空间")

# one_hot_mathematical_principle() 

# 3 different approach in Python
def one_hot_implementation_methods():
    """
    不同的One-Hot编码实现方法
    """
    print("\n" + "="*70)
    print("           One-Hot编码实现方法对比")
    print("="*70)
    
    # 示例数据
    labels = ['red', 'green', 'blue', 'red', 'green']
    numeric_labels = [0, 1, 2, 0, 1]
    
    print(f"\n原始标签: {labels}")
    print(f"数值标签: {numeric_labels}")
    
    methods = {}
    
    # 方法1：手动实现
    print("\n1️⃣ 手动实现（基础版）：")
    unique = list(set(labels))
    manual_onehot = np.zeros((len(labels), len(unique)))
    for i, label in enumerate(labels):
        manual_onehot[i, unique.index(label)] = 1
    print(f"   结果:\n{manual_onehot}")
    methods['手动实现'] = manual_onehot
    
    # 方法2：使用numpy的eye函数
    print("\n2️⃣ NumPy eye函数（简洁版）：")
    num_classes = len(set(labels))
    eye_onehot = np.eye(num_classes)[numeric_labels]
    print(f"   结果:\n{eye_onehot}")
    methods['NumPy eye'] = eye_onehot
    
    # 方法3：使用numpy的高级索引
    print("\n3️⃣ NumPy高级索引（高效版）：")
    advanced_onehot = np.zeros((len(numeric_labels), num_classes))
    advanced_onehot[np.arange(len(numeric_labels)), numeric_labels] = 1
    print(f"   结果:\n{advanced_onehot}")
    methods['NumPy高级索引'] = advanced_onehot
    
    # 方法4：使用pandas的get_dummies
    print("\n4️⃣ Pandas get_dummies：")
    import pandas as pd
    df = pd.DataFrame({'color': labels})
    pandas_onehot = pd.get_dummies(df['color']).values
    print(f"   结果:\n{pandas_onehot}")
    methods['Pandas'] = pandas_onehot
    
    # 方法5：使用sklearn
    print("\n5️⃣ Scikit-learn OneHotEncoder：")
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    sklearn_onehot = encoder.fit_transform(np.array(labels).reshape(-1, 1))
    print(f"   结果:\n{sklearn_onehot}")
    methods['Scikit-learn'] = sklearn_onehot
    
    # 验证所有方法结果一致
    print("\n✅ 验证：所有方法的结果应该一致")
    reference = methods['手动实现']
    for name, result in methods.items():
        if name != '手动实现':
            is_equal = np.array_equal(reference, result)
            print(f"   {name:15s}: {'✓ 一致' if is_equal else '✗ 不一致'}")
    
    # 性能对比
    print("\n⏱️ 性能对比（10000个样本）：")
    import time
    
    n_samples = 10000
    large_labels = np.random.randint(0, 10, n_samples)
    n_classes = 10
    
    # 方法2：NumPy eye
    start = time.time()
    _ = np.eye(n_classes)[large_labels]
    eye_time = time.time() - start
    
    # 方法3：NumPy高级索引
    start = time.time()
    result = np.zeros((n_samples, n_classes))
    result[np.arange(n_samples), large_labels] = 1
    adv_time = time.time() - start
    
    print(f"   NumPy eye:      {eye_time:.6f} 秒")
    print(f"   NumPy高级索引:  {adv_time:.6f} 秒")
    
    # 可视化不同方法
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, (method_name, result) in enumerate(methods.items()):
        if idx < 6:
            ax = axes[idx]
            im = ax.imshow(result, cmap='Blues', aspect='auto')
            ax.set_title(f'{method_name}')
            ax.set_xlabel('类别')
            ax.set_ylabel('样本')
            
            # 设置刻度
            ax.set_xticks(range(result.shape[1]))
            ax.set_yticks(range(result.shape[0]))
    
    plt.suptitle('不同方法的One-Hot编码结果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# one_hot_implementation_methods()

# 4. apply in DL
def one_hot_in_deep_learning():
    """
    One-Hot编码在深度学习中的关键作用
    """
    print("\n" + "="*70)
    print("        One-Hot编码在深度学习中的应用")
    print("="*70)
    
    # 模拟MNIST数字识别
    print("\n🎯 场景：手写数字识别 (0-9)")
    
    # 示例：一批4个样本
    true_labels = np.array([5, 0, 9, 2])
    print(f"\n真实标签 (数值): {true_labels}")
    
    # One-Hot编码
    num_classes = 10
    y_true_onehot = np.eye(num_classes)[true_labels]
    print(f"One-Hot编码:")
    for i, (label, onehot) in enumerate(zip(true_labels, y_true_onehot)):
        print(f"   样本{i+1} (数字{label}): {onehot}")
    
    # 模拟模型预测
    np.random.seed(42)
    logits = np.random.randn(4, 10) * 2  # 随机logits
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    predictions = softmax(logits)
    
    print(f"\n模型预测概率:")
    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        print(f"   样本{i+1}: 预测为 {predicted_class}, 概率 {pred[predicted_class]:.3f}")
    
    # 计算交叉熵损失
    def cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    loss = cross_entropy_loss(y_true_onehot, predictions)
    
    print(f"\n📊 损失计算过程:")
    print(f"   L = -1/N Σᵢ Σⱼ y_ij * log(p_ij)")
    print(f"   其中 y_ij 是one-hot编码，p_ij 是预测概率")
    
    # 逐样本展示
    print(f"\n   逐样本损失:")
    for i in range(4):
        true_class = true_labels[i]
        pred_prob = predictions[i, true_class]
        sample_loss = -np.log(max(pred_prob, 1e-15))
        print(f"   样本{i+1}: -log({pred_prob:.4f}) = {sample_loss:.4f}")
    
    print(f"\n   平均损失: {loss:.4f}")
    
    # 可视化One-Hot标签与预测的对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i in range(4):
        ax = axes[i]
        
        # 绘制真实标签（one-hot）
        x = np.arange(10)
        ax.bar(x - 0.15, y_true_onehot[i], 0.3, 
              label='真实标签', color='green', alpha=0.8)
        
        # 绘制预测概率
        ax.bar(x + 0.15, predictions[i], 0.3, 
              label='预测概率', color='blue', alpha=0.6)
        
        # 高亮真实类别
        true_class = true_labels[i]
        ax.axvline(x=true_class, color='green', linestyle='--', alpha=0.5)
        
        ax.set_title(f'样本 {i+1} (真实数字: {true_class})')
        ax.set_xlabel('数字类别')
        ax.set_ylabel('概率')
        ax.set_xticks(range(10))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 显示预测结果
        pred_class = np.argmax(predictions[i])
        color = 'green' if pred_class == true_class else 'red'
        ax.text(0.5, 0.95, f'预测: {pred_class}', 
               transform=ax.transAxes, ha='center',
               fontsize=12, color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('One-Hot编码在损失计算中的作用', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 为什么One-Hot使梯度计算简单
    print(f"\n💡 One-Hot编码的关键优势：")
    print(f"   1. 损失函数简化:")
    print(f"      -L = -log(p_true_class)")
    print(f"      因为只有真实类别位置的y=1，其他都是0")
    print(f"   2. 梯度简化:")
    print(f"      ∂L/∂logits = softmax(logits) - y_onehot")
    print(f"      梯度 = 预测概率 - one_hot标签")
    print(f"   3. 优化目标明确:")
    print(f"      增加真实类别的概率，降低其他类别的概率")

# one_hot_in_deep_learning()

# 5. pron and cons, also alternative
def one_hot_advantages_and_alternatives():
    """
    One-Hot编码的优缺点和替代方案
    """
    print("\n" + "="*70)
    print("       One-Hot编码的优缺点与替代方案")
    print("="*70)
    
    # 1. 优缺点总结
    print("\n📊 One-Hot编码的优缺点：")
    
    advantages = [
        "消除了类别间的序关系",
        "所有类别在向量空间中距离相等",
        "与神经网络输出层完美配合",
        "使交叉熵损失计算变得简单",
        "梯度计算简洁（∂L/∂x = pred - onehot）",
    ]
    
    disadvantages = [
        "维度灾难：n个类别产生n维向量",
        "稀疏表示：大多数值为0，存储效率低",
        "不保留类别间的语义关系",
        "无法处理未见过的类别",
        "对于高基数特征（如城市名）不适用",
    ]
    
    print("\n✅ 优点:")
    for i, adv in enumerate(advantages, 1):
        print(f"   {i}. {adv}")
    
    print("\n❌ 缺点:")
    for i, dis in enumerate(disadvantages, 1):
        print(f"   {i}. {dis}")
    
    # 2. 替代方案对比
    print("\n🔄 常见替代方案：")
    
    # 创建示例数据
    categories = ['红色', '蓝色', '绿色', '黄色', '紫色']
    
    # 不同编码方式
    encodings = {
        'One-Hot编码': np.eye(5),
        '标签编码': np.array([[0], [1], [2], [3], [4]]),
        '二进制编码': np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0]]),
        '哈希编码': np.array([[0.7], [0.3], [-0.5], [0.1], [-0.8]]),  # 简化的哈希
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, encoding) in enumerate(encodings.items()):
        ax = axes[idx]
        
        if encoding.ndim == 1:
            encoding = encoding.reshape(-1, 1)
        
        im = ax.imshow(encoding, cmap='RdYlBu', aspect='auto')
        ax.set_title(f'{name}\n维度: {encoding.shape[1]}')
        ax.set_xlabel('编码维度')
        ax.set_ylabel('类别')
        ax.set_yticks(range(5))
        ax.set_yticklabels(categories)
        
        plt.colorbar(im, ax=ax)
        
        # 添加数值
        for i in range(encoding.shape[0]):
            for j in range(encoding.shape[1]):
                ax.text(j, i, f'{encoding[i,j]:.1f}', 
                       ha='center', va='center', fontsize=8)
    
    plt.suptitle('不同编码方式对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 3. 实际应用建议
    print("\n💡 实际应用建议：")
    
    scenarios = {
        '小规模分类 (<100类)': 'One-Hot编码 - 最合适',
        '中等规模分类 (100-1000类)': 'One-Hot编码 - 仍可使用',
        '大规模分类 (>1000类)': '考虑Embedding或哈希编码',
        '高基数类别特征': 'Embedding、Target Encoding',
        'NLP词表示': 'Word Embeddings (Word2Vec, GloVe)',
        '有序类别': '标签编码或自定义映射',
    }
    
    for scenario, recommendation in scenarios.items():
        print(f"   • {scenario:25s}: {recommendation}")

# one_hot_advantages_and_alternatives()

# 6. best practice
def one_hot_best_practices():
    """
    One-Hot编码的最佳实践
    """
    print("\n" + "="*70)
    print("           One-Hot编码最佳实践")
    print("="*70)
    
    # 1. 处理未见过的类别
    print("\n🔧 实践1：处理未见过的类别")
    
    from sklearn.preprocessing import OneHotEncoder
    
    # 训练时看到的类别
    train_categories = np.array(['A', 'B', 'C']).reshape(-1, 1)
    
    # 测试时的新类别
    test_categories = np.array(['A', 'B', 'D', 'E']).reshape(-1, 1)
    
    print("   训练集类别:", train_categories.ravel())
    print("   测试集类别:", test_categories.ravel())
    
    # 方法1：handle_unknown='ignore'
    encoder_ignore = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_ignore.fit(train_categories)
    
    try:
        result_ignore = encoder_ignore.transform(test_categories)
        print(f"\n   方法1 (ignore): 忽略未知类别")
        print(f"   结果:\n{result_ignore}")
        print(f"   未知类别'D'和'E'被编码为全0")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 方法2：添加'unknown'类别
    all_categories = np.unique(np.vstack([train_categories, test_categories]))
    print(f"\n   方法2: 添加所有可能的类别")
    encoder_all = OneHotEncoder(sparse_output=False)
    encoder_all.fit(np.array(all_categories).reshape(-1, 1))
    result_all = encoder_all.transform(test_categories)
    print(f"   结果:\n{result_all}")
    
    # 2. 内存优化：稀疏矩阵
    print("\n💾 实践2：内存优化（稀疏矩阵）")
    
    n_samples = 10000
    n_categories = 1000
    
    # 创建大规模数据
    large_labels = np.random.randint(0, n_categories, n_samples)
    
    # 密集矩阵
    dense_onehot = np.eye(n_categories)[large_labels]
    dense_memory = dense_onehot.nbytes / 1024 / 1024
    
    # 稀疏矩阵
    from scipy import sparse
    sparse_onehot = sparse.csr_matrix(
        (np.ones(n_samples), (range(n_samples), large_labels)),
        shape=(n_samples, n_categories)
    )
    sparse_memory = sparse_onehot.data.nbytes / 1024 / 1024
    
    print(f"   样本数: {n_samples:,}, 类别数: {n_categories:,}")
    print(f"   密集矩阵内存: {dense_memory:.2f} MB")
    print(f"   稀疏矩阵内存: {sparse_memory:.2f} MB")
    print(f"   内存节省: {(1 - sparse_memory/dense_memory)*100:.1f}%")
    
    # 3. 在神经网络中的最佳实践
    print("\n🧠 实践3：神经网络中的最佳实践")
    
    best_practices = """
    1. 输入层：
       • 使用One-Hot编码作为分类特征的输入
       • 配合Embedding层处理高基数特征
    
    2. 输出层：
       • 使用One-Hot标签与Softmax + 交叉熵配合
       • 标签平滑(Label Smoothing)：不完全使用0/1
         例如：[0.9, 0.033, 0.033, 0.033] 代替 [1, 0, 0, 0]
    
    3. 损失计算：
       • 利用稀疏性：手动实现时使用 gather 操作
       • PyTorch: F.cross_entropy 接受类别索引而非one-hot
       • TensorFlow: SparseCategoricalCrossentropy
    
    4. 性能优化：
       • 小规模类别：直接使用one-hot
       • 大规模类别：使用稀疏表示或Embedding
       • GPU训练：注意内存限制
    """
    
    print(best_practices)
    
    # 可视化：标签平滑的效果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 标准One-Hot
    standard_labels = np.array([1, 0, 0, 0, 0])
    axes[0].bar(range(5), standard_labels, color='blue', alpha=0.7)
    axes[0].set_title('标准One-Hot标签')
    axes[0].set_ylabel('概率')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 标签平滑
    epsilon = 0.1
    smoothed_labels = standard_labels * (1 - epsilon) + epsilon / 5
    axes[1].bar(range(5), smoothed_labels, color='green', alpha=0.7)
    axes[1].set_title(f'标签平滑 (ε={epsilon})')
    axes[1].set_ylabel('概率')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('One-Hot vs 标签平滑', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# one_hot_best_practices()

# 7. summary
def one_hot_learning_summary():
    """
    One-Hot编码学习总结
    """
    print("\n" + "="*70)
    print("              📚 One-Hot编码 - 学习总结")
    print("="*70)
    
    summary = """
    
    🎯 核心概念：
    ┌─────────────────────────────────────────────────────┐
    │ One-Hot编码 = 将N个类别转化为N维向量               │
    │ 每个向量只有一个位置为1，其余为0                     │
    │ 类似于"独热"开关：一次只有一个状态激活              │
    └─────────────────────────────────────────────────────┘
    
    📐 数学形式：
    • 第i个类别 → e_i = [0,0,...,1,...,0] (第i位为1)
    • 向量空间中的标准正交基
    • 所有类别间距离相等：||e_i - e_j|| = √2 (i≠j)
    
    💻 常用实现：
    1. NumPy:  np.eye(n_classes)[labels]
    2. Pandas: pd.get_dummies(data)
    3. Sklearn: OneHotEncoder().fit_transform(data)
    4. 手动:    onehot[np.arange(n), labels] = 1
    
    🧠 在深度学习中的角色：
    ┌─────────────────────────────────────────────────────┐
    │ 输入层 ← One-Hot特征                               │
    │   ↓                                                 │
    │ 隐藏层                                              │
    │   ↓                                                 │
    │ 输出层 → Softmax → 与One-Hot标签计算损失            │
    └─────────────────────────────────────────────────────┘
    
    ⚡ 关键优势：
    1. 消除虚假的序关系
    2. 所有类别平等对待
    3. 与Softmax+交叉熵完美配合
    4. 梯度计算简洁：∂L/∂logits = y_pred - y_true
    
    ⚠️ 注意事项：
    1. 类别太多时维度爆炸
    2. 稀疏表示需要特殊处理
    3. 无法表达类别间的语义关系
    4. 新类别处理需要策略
    
    🔄 替代方案选择指南：
    
    场景                    → 推荐方案
    ─────────────────────────────────────
    小规模分类 (<100类)     → One-Hot
    中等分类 (100-1000)     → One-Hot/Embedding  
    大规模分类 (>1000)      → Embedding
    自然语言处理            → Word Embeddings
    高基数特征              → Embedding/Hashing
    有序类别                → Label Encoding
    
    🎓 学习检查清单：
    □ 理解One-Hot编码的基本原理
    □ 能够手动实现One-Hot编码
    □ 知道何时使用One-Hot编码
    □ 了解One-Hot在损失函数中的作用
    □ 掌握处理未知类别的策略
    □ 知道常见的替代方案
    □ 能在PyTorch/TensorFlow中正确使用
    
    """
    
    print(summary)
    
    # 创建思维导图
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 中心
    center = (5, 5)
    ax.text(center[0], center[1], 'One-Hot\n编码', 
            ha='center', va='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 分支
    branches = [
        ('概念', (2, 8), '#FFB3BA'),
        ('实现', (8, 8), '#BAFFC9'),
        ('应用', (2, 2), '#BAE1FF'),
        ('优化', (8, 2), '#FFFFBA'),
    ]
    
    sub_branches = {
        '概念': ['独热开关', '正交基向量', '等距离表示'],
        '实现': ['NumPy eye', 'Pandas dummies', 'Sklearn Encoder'],
        '应用': ['分类标签', '损失函数', 'Embedding输入'],
        '优化': ['稀疏矩阵', '标签平滑', 'Embedding替代'],
    }
    
    for name, (x, y), color in branches:
        # 主分支
        ax.text(x, y, name, ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        ax.plot([center[0], x], [center[1], y], 'gray', alpha=0.5, linewidth=2)
        
        # 子分支
        for i, sub in enumerate(sub_branches[name]):
            sub_y = y - 0.8 - i * 0.5
            ax.text(x, sub_y, f'• {sub}', ha='center', va='center', 
                   fontsize=10, alpha=0.8)
    
    ax.set_title('One-Hot编码知识地图', fontsize=16, fontweight='bold', pad=20)
    plt.show()

one_hot_learning_summary()