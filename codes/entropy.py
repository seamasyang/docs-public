import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']          
plt.rcParams['axes.unicode_minus'] = False  

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

entropy_demonstration()