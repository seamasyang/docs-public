# 混淆矩阵是**评估分类模型性能**最直观、最强大的工具之一。

# 1. 什么是混淆矩阵？直观理解
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei']          
plt.rcParams['axes.unicode_minus'] = False  

def confusion_matrix_intuition():
    """
    混淆矩阵的直观理解
    """
    print("="*80)
    print("                    混淆矩阵 - 直观理解")
    print("="*80)
    
    # 医疗诊断场景
    print("\n🏥 场景：医生诊断疾病")
    print("-"*50)
    
    # 真实情况：65人有病，35人没病
    # 医生诊断：判断60人有病，40人没病
    
    actual = ['有病']*65 + ['没病']*35
    predicted = ['有病']*55 + ['没病']*10 + ['有病']*5 + ['没病']*30
    
    # 计算四种结果
    true_positive = 55   # 正确诊断为有病
    false_negative = 10  # 误诊为没病（漏诊）
    false_positive = 5   # 误诊为有病（误诊）
    true_negative = 30   # 正确诊断为没病
    
    # 创建混淆矩阵
    cm = np.array([[true_negative, false_positive],
                   [false_negative, true_positive]])
    
    print("\n📊 诊断结果分析：")
    print("┌─────────────────────────────────────┐")
    print("│              预测：没病  预测：有病  │")
    print("├─────────────────────────────────────┤")
    print(f"│ 真实：没病    {true_negative:3d} (TN)   {false_positive:3d} (FP)  │")
    print(f"│ 真实：有病    {false_negative:3d} (FN)   {true_positive:3d} (TP)  │")
    print("└─────────────────────────────────────┘")
    
    print(f"\n📝 四种结果的含义：")
    print(f"   ✅ 真阳性 (TP={true_positive})：有病被正确诊断出来")
    print(f"   ❌ 假阴性 (FN={false_negative})：有病但被漏诊（危险！）")
    print(f"   ❌ 假阳性 (FP={false_positive})：没病但被误诊（虚惊一场）")
    print(f"   ✅ 真阴性 (TN={true_negative})：没病被正确排除")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1：混淆矩阵热图
    labels = ['阴性', '阳性']
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], cbar_kws={'label': '人数'})
    axes[0].set_xlabel('预测结果', fontsize=12)
    axes[0].set_ylabel('真实情况', fontsize=12)
    axes[0].set_title('混淆矩阵', fontsize=14, fontweight='bold')
    
    # 添加标注
    positions = [(0,0,'TN\n正确排除'), (0,1,'FP\n误诊'),
                 (1,0,'FN\n漏诊'), (1,1,'TP\n正确诊断')]
    for i, j, text in positions:
        color = 'green' if i == j else 'red'
        axes[0].text(j+0.5, i+0.5, text, ha='center', va='center', 
                    color=color, fontsize=10, fontweight='bold')
    
    # 图2：诊断结果分布
    categories = ['真阳性\n(正确诊断)', '假阴性\n(漏诊)', 
                  '假阳性\n(误诊)', '真阴性\n(正确排除)']
    values = [true_positive, false_negative, false_positive, true_negative]
    colors = ['darkgreen', 'red', 'orange', 'green']
    
    wedges, texts, autotexts = axes[1].pie(values, labels=categories, 
                                           colors=colors, autopct='%1.1f%%',
                                           explode=(0.05, 0.05, 0.05, 0.05))
    axes[1].set_title('诊断结果分布', fontsize=14, fontweight='bold')
    
    # 图3：性能指标
    accuracy = (true_positive + true_negative) / len(actual)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    f1 = 2 * precision * recall / (precision + recall)
    
    metrics = ['准确率', '精确率', '召回率\n(灵敏度)', '特异度', 'F1分数']
    scores = [accuracy, precision, recall, specificity, f1]
    bar_colors = ['blue', 'purple', 'orange', 'green', 'red']
    
    bars = axes[2].bar(metrics, scores, color=bar_colors, alpha=0.7)
    axes[2].set_ylabel('分数')
    axes[2].set_title('从混淆矩阵计算的指标', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, scores):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle('混淆矩阵 - 医疗诊断案例', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 从混淆矩阵计算的关键指标：")
    print(f"   准确率 (Accuracy):   {accuracy:.3f} = ({true_positive}+{true_negative})/{len(actual)}")
    print(f"   精确率 (Precision):  {precision:.3f} = {true_positive}/({true_positive}+{false_positive})")
    print(f"   召回率 (Recall):     {recall:.3f} = {true_positive}/({true_positive}+{false_negative})")
    print(f"   特异度 (Specificity):{specificity:.3f} = {true_negative}/({true_negative}+{false_positive})")
    print(f"   F1分数:             {f1:.3f} = 2*{precision:.3f}*{recall:.3f}/({precision:.3f}+{recall:.3f})")

# confusion_matrix_intuition()

# 2. 混淆矩阵的结构与公式
def confusion_matrix_structure():
    """
    混淆矩阵的详细结构和所有衍生指标
    """
    print("\n" + "="*80)
    print("               混淆矩阵结构详解")
    print("="*80)
    
    # 标准混淆矩阵结构
    print("""
    🏗️ 标准混淆矩阵结构：
    
                    预测类别
                 阴性(-)    阳性(+)
    ┌────────────────────────────────┐
    │ 真实阴性(-) │  TN      FP    │
    │ 真实阳性(+) │  FN      TP    │
    └────────────────────────────────┘
    
    📝 记忆技巧：
    • 第一个字母 T/F = True/False (预测正确/错误)
    • 第二个字母 P/N = Positive/Negative (预测为正/负)
    
    • TP (True Positive)： 预测为真，实际为真 ✓
    • FP (False Positive)：预测为真，实际为假 ✗ (误报)
    • FN (False Negative)：预测为假，实际为真 ✗ (漏报)
    • TN (True Negative)： 预测为假，实际为假 ✓
    """)
    
    # 多分类混淆矩阵
    print("\n📊 多分类混淆矩阵（以3类为例）：")
    print("""
                    预测类别
                  A      B      C
    ┌────────────────────────────────┐
    │   真实A  │  TPA    EAB    EAC  │
    │   真实B  │  EBA    TPB    EBC  │
    │   真实C  │  ECA    ECB    TPC  │
    └────────────────────────────────┘
    
    其中：
    • TPx：类别x被正确预测的数量（对角线）
    • Exy：类别x被错误预测为类别y的数量（非对角线）
    """)
    
    # 创建示例
    np.random.seed(42)
    y_true = np.repeat([0, 1, 2], [100, 80, 70])
    y_pred = y_true.copy()
    
    # 添加一些错误
    error_indices = np.random.choice(len(y_true), 30, replace=False)
    for idx in error_indices:
        y_pred[idx] = np.random.choice([i for i in range(3) if i != y_true[idx]])
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的指标
    print("\n🔍 每个类别的详细指标：")
    
    for i in range(3):
        tp = cm[i, i]  # 对角线
        fp = cm[:, i].sum() - tp  # 列和减去TP
        fn = cm[i, :].sum() - tp  # 行和减去TP
        tn = cm.sum() - tp - fp - fn  # 总和减去其他
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n   类别 {i}:")
        print(f"   ├─ TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"   ├─ 精确率 (Precision): {precision:.3f}")
        print(f"   ├─ 召回率 (Recall): {recall:.3f}")
        print(f"   ├─ 特异度 (Specificity): {specificity:.3f}")
        print(f"   └─ F1分数: {f1:.3f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 计数混淆矩阵
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('混淆矩阵 (计数)', fontweight='bold')
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    
    # 高亮对角线
    for i in range(3):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))
    
    # 2. 归一化混淆矩阵（按行）
    ax = axes[0, 1]
    cm_row_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_row_norm, annot=True, fmt='.1%', cmap='YlOrRd',
                xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'],
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    ax.set_title('按行归一化 (召回率)', fontweight='bold')
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    
    # 3. 归一化混淆矩阵（按列）
    ax = axes[0, 2]
    cm_col_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    sns.heatmap(cm_col_norm, annot=True, fmt='.1%', cmap='YlOrRd',
                xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'],
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Precision'})
    ax.set_title('按列归一化 (精确率)', fontweight='bold')
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    
    # 4. 每类指标对比
    ax = axes[1, 0]
    metrics_data = {}
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_data[f'类别{i}'] = [precision, recall, f1]
    
    x = np.arange(3)
    width = 0.25
    metric_names = ['精确率', '召回率', 'F1分数']
    
    for j, (class_name, scores) in enumerate(metrics_data.items()):
        bars = ax.bar(j + x * width, scores, width, label=class_name if j == 0 else "")
    
    ax.set_ylabel('分数')
    ax.set_title('各类别指标对比')
    ax.set_xticks(np.arange(3) + width)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. 错误分布
    ax = axes[1, 1]
    errors = []
    for i in range(3):
        for j in range(3):
            if i != j:
                errors.append(cm[i, j])
    
    error_labels = [f'{i}→{j}' for i in range(3) for j in range(3) if i != j]
    colors = ['orange' if e > 0 else 'lightgray' for e in errors]
    ax.barh(error_labels, errors, color=colors)
    ax.set_xlabel('错误数量')
    ax.set_title('错误分类分布')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. 正确率与错误率
    ax = axes[1, 2]
    correct = np.diag(cm).sum()
    incorrect = cm.sum() - correct
    ax.pie([correct, incorrect], labels=['正确', '错误'], 
           colors=['green', 'red'], autopct='%1.1f%%', explode=(0, 0.1))
    ax.set_title(f'总体正确率: {correct/cm.sum():.1%}')
    
    plt.suptitle('混淆矩阵完全分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# confusion_matrix_structure()

# 3. 从混淆矩阵计算所有指标
def metrics_from_confusion_matrix():
    """
    从混淆矩阵推导所有性能指标
    """
    print("\n" + "="*80)
    print("         从混淆矩阵计算所有性能指标")
    print("="*80)
    
    # 示例混淆矩阵
    cm = np.array([[85, 5, 10],
                   [8, 72, 20],
                   [3, 8, 89]])
    
    print("\n📊 示例混淆矩阵：")
    print(pd.DataFrame(cm, 
                       index=['真实:A', '真实:B', '真实:C'],
                       columns=['预测:A', '预测:B', '预测:C']))
    
    # 总体指标
    total = cm.sum()
    correct = np.diag(cm).sum()
    
    print("\n" + "="*50)
    print("📊 总体性能指标")
    print("="*50)
    
    # 1. 准确率
    accuracy = correct / total
    print(f"\n1. 准确率 (Accuracy): {accuracy:.4f}")
    print(f"   公式: (TP+TN)/Total = {correct}/{total}")
    print(f"   含义: 所有预测中正确的比例")
    
    # 2. 错误率
    error_rate = 1 - accuracy
    print(f"\n2. 错误率 (Error Rate): {error_rate:.4f}")
    print(f"   公式: 1 - Accuracy")
    
    print("\n" + "="*50)
    print("📊 各类别详细指标")
    print("="*50)
    
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        
        print(f"\n--- 类别 {i} ---")
        print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # 精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"\n   精确率 (Precision): {precision:.4f}")
        print(f"   公式: TP/(TP+FP) = {tp}/({tp}+{fp})")
        print(f"   含义: 预测为类别{i}的样本中，有多少是真的")
        
        # 召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\n   召回率 (Recall/Sensitivity): {recall:.4f}")
        print(f"   公式: TP/(TP+FN) = {tp}/({tp}+{fn})")
        print(f"   含义: 真实类别{i}的样本中，有多少被正确识别")
        
        # 特异度
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\n   特异度 (Specificity): {specificity:.4f}")
        print(f"   公式: TN/(TN+FP) = {tn}/({tn}+{fp})")
        print(f"   含义: 非类别{i}的样本中，有多少被正确排除")
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\n   F1分数: {f1:.4f}")
        print(f"   公式: 2*Precision*Recall/(Precision+Recall)")
        print(f"   含义: 精确率和召回率的调和平均")
    
    # 宏平均和微平均
    print("\n" + "="*50)
    print("📊 宏平均 vs 微平均")
    print("="*50)
    
    precisions = []
    recalls = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    
    print(f"\n   宏平均 (Macro-average):")
    print(f"   精确率: {macro_precision:.4f}")
    print(f"   召回率: {macro_recall:.4f}")
    print(f"   F1分数: {macro_f1:.4f}")
    print(f"   (每个类别平等对待)")
    
    # 微平均
    total_tp = np.diag(cm).sum()
    total_fp = np.array([cm[:, i].sum() - cm[i, i] for i in range(cm.shape[0])]).sum()
    total_fn = np.array([cm[i, :].sum() - cm[i, i] for i in range(cm.shape[0])]).sum()
    
    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    
    print(f"\n   微平均 (Micro-average):")
    print(f"   精确率: {micro_precision:.4f}")
    print(f"   召回率: {micro_recall:.4f}")
    print(f"   F1分数: {micro_f1:.4f}")
    print(f"   (按样本数加权)")
    
    # 可视化指标关系
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1：精确率-召回率权衡
    ax = axes[0]
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ax.scatter(recall, precision, s=200, label=f'类别{i}')
        ax.annotate(f'类别{i}', (recall, precision), 
                   xytext=(10, 10), textcoords='offset points')
    
    ax.set_xlabel('召回率 (Recall)')
    ax.set_ylabel('精确率 (Precision)')
    ax.set_title('精确率-召回率空间')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # 图2：指标雷达图
    ax = axes[1]
    categories = ['准确率', '精确率', '召回率', 'F1分数']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 各类别的指标
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        values = [accuracy, precision, recall, f1]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'类别{i}')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 1])
    ax.set_title('各类别性能雷达图')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('从混淆矩阵计算的所有指标', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# metrics_from_confusion_matrix()

# 4. 实际案例：手写数字识别
def mnist_confusion_matrix_demo():
    """
    MNIST手写数字识别的混淆矩阵分析
    """
    print("\n" + "="*80)
    print("         MNIST手写数字识别 - 混淆矩阵实战")
    print("="*80)
    
    # 模拟MNIST识别结果
    np.random.seed(42)
    n_samples = 1000
    true_labels = np.random.randint(0, 10, n_samples)
    
    # 模拟预测（大部分正确，有一些典型错误）
    pred_labels = true_labels.copy()
    
    # 常见混淆模式
    confusion_patterns = {
        (4, 9): 0.15,  # 4容易被误认为9
        (9, 4): 0.15,  # 9容易被误认为4
        (3, 8): 0.10,  # 3容易被误认为8
        (8, 3): 0.10,  # 8容易被误认为3
        (7, 1): 0.10,  # 7容易被误认为1
        (5, 6): 0.08,  # 5容易被误认为6
        (2, 7): 0.05,  # 2容易被误认为7
    }
    
    for (true_digit, pred_digit), prob in confusion_patterns.items():
        mask = true_labels == true_digit
        n_confuse = int(mask.sum() * prob)
        indices = np.where(mask)[0][:n_confuse]
        pred_labels[indices] = pred_digit
    
    # 创建混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    print("\n📊 MNIST手写数字识别混淆矩阵：")
    df_cm = pd.DataFrame(cm, 
                         index=[f'真实:{i}' for i in range(10)],
                         columns=[f'预测:{i}' for i in range(10)])
    print(df_cm)
    
    # 找出最常见的错误
    print("\n🔍 最常见的混淆对：")
    errors = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                errors.append((i, j, cm[i, j]))
    
    errors.sort(key=lambda x: x[2], reverse=True)
    
    for true_digit, pred_digit, count in errors[:10]:
        print(f"   数字 {true_digit} → 数字 {pred_digit}: {count} 次")
    
    # 可视化
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 主混淆矩阵
    ax1 = plt.subplot(2, 3, (1, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=range(10), yticklabels=range(10),
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('MNIST手写数字识别 - 混淆矩阵', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测数字')
    ax1.set_ylabel('真实数字')
    
    # 高亮对角线
    for i in range(10):
        ax1.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=2))
    
    # 2. 归一化混淆矩阵
    ax2 = plt.subplot(2, 3, 4)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='RdYlGn',
                xticklabels=range(10), yticklabels=range(10),
                ax=ax2, vmin=0, vmax=1, cbar_kws={'label': '召回率'})
    ax2.set_title('归一化混淆矩阵 (每行和为100%)', fontweight='bold')
    ax2.set_xlabel('预测数字')
    ax2.set_ylabel('真实数字')
    
    # 3. 每个数字的准确率
    ax3 = plt.subplot(2, 3, 5)
    accuracies = np.diag(cm) / cm.sum(axis=1)
    colors = ['green' if acc > 0.9 else 'orange' if acc > 0.8 else 'red' 
             for acc in accuracies]
    bars = ax3.bar(range(10), accuracies, color=colors, alpha=0.7)
    ax3.set_xlabel('数字')
    ax3.set_ylabel('召回率')
    ax3.set_title('每个数字的识别准确率')
    ax3.set_ylim([0, 1.1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', fontsize=9)
    
    # 4. 错误流向图
    ax4 = plt.subplot(2, 3, 6)
    
    # 创建错误矩阵（排除对角线）
    error_matrix = cm.copy()
    np.fill_diagonal(error_matrix, 0)
    
    # 绘制错误热图
    im = ax4.imshow(error_matrix, cmap='Reds', aspect='auto')
    ax4.set_xticks(range(10))
    ax4.set_yticks(range(10))
    ax4.set_xlabel('预测数字')
    ax4.set_ylabel('真实数字')
    ax4.set_title('错误分布热图 (排除正确预测)')
    
    # 标注非零错误
    for i in range(10):
        for j in range(10):
            if error_matrix[i, j] > 0:
                ax4.text(j, i, error_matrix[i, j], ha='center', va='center',
                        color='black' if error_matrix[i, j] < error_matrix.max()/2 else 'white')
    
    plt.colorbar(im, ax=ax4, label='错误数量')
    
    plt.suptitle('MNIST手写数字识别混淆矩阵分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 生成分析报告
    print("\n" + "="*80)
    print("                   混淆矩阵分析报告")
    print("="*80)
    
    accuracy = np.diag(cm).sum() / cm.sum()
    
    report = f"""
    📊 总体性能:
    • 总体准确率: {accuracy:.2%}
    • 正确预测: {np.diag(cm).sum()} / {cm.sum()}
    • 错误预测: {cm.sum() - np.diag(cm).sum()} / {cm.sum()}
    
    📈 各类别性能:
    """
    
    for i in range(10):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        report += f"""
    数字 {i}:
    • 精确率: {precision:.3f}  • 召回率: {recall:.3f}  • F1: {f1:.3f}
    • TP={tp}, FP={fp}, FN={fn}
    """
    
    report += f"""
    🎯 关键发现:
    1. 最佳识别: 数字 {np.argmax(accuracies)} (准确率: {max(accuracies):.1%})
    2. 最难识别: 数字 {np.argmin(accuracies)} (准确率: {min(accuracies):.1%})
    3. 最常见混淆: {errors[0][0]}→{errors[0][1]} ({errors[0][2]}次)
    
    💡 改进建议:
    • 对于容易混淆的数字对，可以增加训练数据
    • 考虑使用数据增强技术
    • 针对性地优化特征提取
    """
    
    print(report)

# mnist_confusion_matrix_demo()

# 5. 高级技巧与最佳实践
def advanced_confusion_matrix():
    """
    混淆矩阵的高级应用和最佳实践
    """
    print("\n" + "="*80)
    print("          混淆矩阵高级应用与最佳实践")
    print("="*80)
    
    # 1. 阈值调整的影响
    print("\n🎚️ 技巧1：阈值调整对混淆矩阵的影响")
    
    # 模拟二分类问题的不同阈值
    np.random.seed(42)
    n_samples = 1000
    true_labels = np.random.randint(0, 2, n_samples)
    scores = np.random.rand(n_samples)
    
    thresholds = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 绘制分数分布
    axes[0].hist(scores[true_labels == 0], bins=30, alpha=0.5, label='负类', color='blue')
    axes[0].hist(scores[true_labels == 1], bins=30, alpha=0.5, label='正类', color='red')
    axes[0].set_xlabel('预测分数')
    axes[0].set_ylabel('频次')
    axes[0].set_title('预测分数分布')
    axes[0].legend()
    
    for thr_idx, threshold in enumerate(thresholds):
        pred_labels = (scores >= threshold).astype(int)
        cm = confusion_matrix(true_labels, pred_labels)
        
        ax = axes[thr_idx + 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['负', '正'], yticklabels=['负', '正'],
                   ax=ax, cbar=False)
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ax.set_title(f'阈值={threshold}\n精确率={precision:.3f}, 召回率={recall:.3f}')
    
    plt.suptitle('不同阈值下的混淆矩阵变化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 2. ROC曲线
    print("\n📈 技巧2：ROC曲线与混淆矩阵的关系")
    
    from sklearn.metrics import roc_curve, auc
    
    # 生成更真实的预测分数
    scores = np.where(true_labels == 1, 
                     np.random.beta(5, 2, n_samples),
                     np.random.beta(2, 5, n_samples))
    
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC曲线
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='随机分类器')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('假阳性率 (1 - 特异度)')
    axes[0].set_ylabel('真阳性率 (召回率)')
    axes[0].set_title('ROC曲线')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # 标注几个阈值点
    for threshold in [0.3, 0.5, 0.7]:
        pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(true_labels, pred)
        tn, fp, fn, tp = cm.ravel()
        fpr_point = fp / (fp + tn)
        tpr_point = tp / (tp + fn)
        axes[0].plot(fpr_point, tpr_point, 'ro', markersize=10)
        axes[0].annotate(f'阈值={threshold}', (fpr_point, tpr_point),
                        xytext=(10, -10), textcoords='offset points')
    
    # 混淆矩阵在不同工作点
    axes[1].axis('off')
    axes[1].text(0.1, 0.9, 'ROC曲线上的点对应不同的混淆矩阵：', 
                fontsize=12, fontweight='bold')
    
    explanations = [
        '• 左上角：高召回率，低误报率（理想）',
        '• 右上角：高召回率，高误报率（激进）',
        '• 左下角：低召回率，低误报率（保守）',
        '• 对角线：随机猜测',
        '',
        f'当前模型 AUC = {roc_auc:.3f}',
        f' AUC = 1.0 表示完美分类器',
        f' AUC = 0.5 表示随机分类器',
    ]
    
    for i, text in enumerate(explanations):
        axes[1].text(0.1, 0.7 - i*0.08, text, fontsize=11)
    
    plt.suptitle('混淆矩阵与ROC曲线的关系', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 3. 最佳实践总结
    print("\n" + "="*80)
    print("              混淆矩阵最佳实践总结")
    print("="*80)
    
    best_practices = """
    
    📋 使用混淆矩阵的最佳实践：
    
    1. 选择合适的归一化方式：
       • 按行归一化：查看召回率（各类别被正确识别的比例）
       • 按列归一化：查看精确率（预测为该类的样本中正确的比例）
       • 不归一化：查看绝对错误数量
    
    2. 分析错误模式：
       • 找出最常见的混淆对
       • 分析混淆的原因（特征相似？数据不足？）
       • 针对性改进模型或增加数据
    
    3. 结合业务场景：
       • 医疗诊断：关注假阴性（漏诊可能致命）
       • 垃圾邮件：关注假阳性（重要邮件被拦截）
       • 欺诈检测：关注假阴性（漏掉欺诈交易）
    
    4. 使用多个指标：
       • 不要只看准确率
       • 结合精确率、召回率、F1分数
       • 考虑使用宏平均和微平均
    
    5. 可视化技巧：
       • 使用热图展示混淆矩阵
       • 高亮对角线和最大错误
       • 添加数值标注
       • 使用颜色梯度表示大小
    
    6. 常见陷阱：
       • 类别不平衡时准确率可能误导
       • 多分类问题要注意宏/微平均的选择
       • 测试集要代表真实数据分布
    
    """
    
    print(best_practices)
    
    return best_practices

# advanced_confusion_matrix()

# 6 summary
def confusion_matrix_summary():
    """
    混淆矩阵学习总结
    """
    print("\n" + "="*80)
    print("               📚 混淆矩阵 - 学习总结")
    print("="*80)
    
    summary = """
    
    🎯 核心定义：
    混淆矩阵是一个 N×N 的表格，用于可视化分类模型的性能，
    其中行表示真实类别，列表示预测类别。
    
    📊 基本结构（二分类）：
                   预测负类    预测正类
    真实负类    [    TN         FP    ]
    真实正类    [    FN         TP    ]
    
    🔢 关键指标：
    
    准确率 (Accuracy) = (TP+TN)/(TP+TN+FP+FN)
    └─ 所有预测中正确的比例
    
    精确率 (Precision) = TP/(TP+FP)
    └─ 预测为正的样本中真正为正的比例
    
    召回率 (Recall) = TP/(TP+FN)
    └─ 真正为正的样本中被正确识别的比例
    
    特异度 (Specificity) = TN/(TN+FP)
    └─ 真正为负的样本中被正确识别的比例
    
    F1分数 = 2 × Precision × Recall / (Precision + Recall)
    └─ 精确率和召回率的调和平均
    
    🎨 多分类扩展：
    对于K个类别，混淆矩阵是K×K的：
    • 对角线：每类的正确预测 (TP)
    • 非对角线：各类间的混淆
    
    💡 记忆技巧：
    • "混淆"指的是模型在各个类别间的混淆情况
    • 好的模型：对角线数值大，非对角线数值小
    • 差的模型：非对角线数值大，对角线数值小
    
    ⚡ 实战要点：
    1. 始终可视化混淆矩阵
    2. 关注业务相关的错误类型
    3. 结合多个指标综合评估
    4. 分析错误模式指导模型改进
    
    """
    
    print(summary)
    
    # 创建知识地图
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 中心节点
    ax.text(5, 5, '混淆矩阵\nConfusion\nMatrix', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 一级分支
    branches = [
        ('基础概念', (1, 8), '#FFB3BA'),
        ('核心指标', (9, 8), '#BAFFC9'),
        ('可视化', (1, 2), '#BAE1FF'),
        ('高级应用', (9, 2), '#FFFFBA'),
    ]
    
    sub_concepts = {
        '基础概念': ['TP/TN/FP/FN', '二分类矩阵', '多分类矩阵', '归一化方式'],
        '核心指标': ['准确率', '精确率', '召回率', 'F1分数'],
        '可视化': ['热力图', '百分比矩阵', '错误分析', '趋势图'],
        '高级应用': ['ROC曲线', '阈值调整', '代价敏感', '错误分析'],
    }
    
    for name, (x, y), color in branches:
        # 主分支
        ax.text(x, y, name, ha='center', va='center', fontsize=13,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        ax.plot([5, x], [5, y], 'gray', alpha=0.5, linewidth=2)
        
        # 子概念
        for i, sub in enumerate(sub_concepts[name]):
            sub_y = y - 0.7 - i * 0.5
            ax.text(x, sub_y, f'• {sub}', ha='center', va='center', 
                   fontsize=10, alpha=0.8)
    
    ax.set_title('混淆矩阵知识地图', fontsize=16, fontweight='bold', pad=20)
    plt.show()

# confusion_matrix_summary()