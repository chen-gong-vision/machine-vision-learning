# -*- coding: utf-8 -*-
"""
第 1 个月第 4 周任务（2026-09-04）
实验：原生 Python for 循环 vs numpy 向量化的速度对比

v2 修正说明（对比 v1 的两个 bug）：
    bug1: 原先用 random.random() 和 np.random.rand() 各生成一份数据，
          两个矩阵内容不同，求和差值 306 —— 比的不是同一份数据。
          修正: 用 b.tolist() 保证两边是同一份数据。
    bug2: 原生侧用了内建 sum()（C 实现，本身很快），不能代表
          "Python 循环遍历像素"的真实速度。
          修正: 改为手写双重 for 循环逐元素累加。
"""
import time

import numpy as np

N = 1000  # 模拟一张 1000x1000 的灰度图

# ---------- 用同一份数据，保证对比公平 ----------
b = np.random.rand(N, N)
a = b.tolist()  # numpy 转原生嵌套 list，内容完全相同

# ---------- 原生 Python：手写双重 for 循环（对应"for 循环遍历像素"）----------
t0 = time.perf_counter()
s1 = 0.0
for row in a:
    for v in row:
        s1 += v
t1 = time.perf_counter()

# ---------- numpy：向量化求和 ----------
t2 = time.perf_counter()
s2 = b.sum()
t3 = time.perf_counter()

print(f"矩阵规模 : {N} x {N}（同一份数据）")
print(f"for 循环 : {(t1 - t0) * 1000:8.1f} ms")
print(f"numpy    : {(t3 - t2) * 1000:8.1f} ms")
print(f"加速倍数 : {(t1 - t0) / (t3 - t2):8.0f} 倍")
print(f"求和差值 : {abs(s1 - s2):.9f}  <- 现在才是真正的浮点累计误差")
print()
print("结论：for 循环遍历像素慢几十倍 —— 第 3 个月写 OpenCV 代码时")
print("      一律用 numpy/OpenCV 的向量化操作，禁止逐像素 for 循环。")
