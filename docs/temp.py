"""
演示：三个角相等（AAA）不能证明三角形全等
=========================================
思路：
  - 构造两个三角形，具有相同的三个角度（例如 30°、60°、90°）。
  - 但边长按不同比例缩放（如 1:2）。
  - 计算内角、边长、面积，验证角度相同但边长不同。
"""

import math

def build_triangle(angles_deg, scale):
    """
    根据角度列表（单位：度）和缩放因子，使用正弦定理计算边长。
    返回边长的三元组 (a, b, c) 分别对应角 A, B, C 的对边。
    """
    # 转换为弧度
    angles_rad = [math.radians(a) for a in angles_deg]
    # 选择第一条边作为基准
    a = scale  # 假设边 a（角 A 对边）长度为 scale
    # 根据正弦定理：b = a * sin(B)/sin(A), c = a * sin(C)/sin(A)
    sinA = math.sin(angles_rad[0])
    b = a * math.sin(angles_rad[1]) / sinA
    c = a * math.sin(angles_rad[2]) / sinA
    return a, b, c

# 定义共同的角度（保证三个角之和为180°）
angles = [30, 60, 90]  # 单位：度

# 构造两个三角形，缩放因子不同
triangle1 = build_triangle(angles, scale=1)
triangle2 = build_triangle(angles, scale=2)  # 边长是三角形1的两倍

print("=== 证明 AAA 不能判定全等 ===")
print(f"两个三角形的三个角均为：{angles[0]}°, {angles[1]}°, {angles[2]}°\n")

print("三角形1（缩放因子=1）：")
print(f"  边长: a={triangle1[0]:.4f}, b={triangle1[1]:.4f}, c={triangle1[2]:.4f}")
print(f"  角度: {angles} (与另一三角形相同)")

print("\n三角形2（缩放因子=2）：")
print(f"  边长: a={triangle2[0]:.4f}, b={triangle2[1]:.4f}, c={triangle2[2]:.4f}")
print(f"  角度: {angles} (相同)")

# 计算面积（使用海伦公式）
def triangle_area(sides):
    a, b, c = sides
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

area1 = triangle_area(triangle1)
area2 = triangle_area(triangle2)

print(f"\n三角形1面积: {area1:.4f}")
print(f"三角形2面积: {area2:.4f}")

# 比较边长
side_diff = [abs(triangle1[i] - triangle2[i]) for i in range(3)]
if any(d > 1e-6 for d in side_diff):
    print("\n结论：尽管三个角完全相同，但边长不全等，面积也不等。")
    print("所以 AAA 不能判定三角形全等，只能判定相似。")
else:
    print("\n（奇怪，边长完全相等，但这只是特例；一般情况下不相等）")