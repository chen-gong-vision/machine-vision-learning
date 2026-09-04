# -*- coding: utf-8 -*-
"""
OpenCV 最小演示（2026-09-04 提前预习，正式学习在 M3 第 3 个月）

目的：先建立"图像 = 数字矩阵"的直观感受，为第 3 个月打底。
输入：data/demo_input.jpg（康工笔记手写截图）
输出：data/output/ 下的灰度图与二值图

核心认知：
    一张 1920x1080 的彩色图，在内存里就是一个 shape=(1080, 1920, 3) 的
    numpy 三维数组 —— 高 x 宽 x 通道(BGR)。
    所谓"图像处理"，本质就是对这个数组做数学运算。
"""
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent  # 项目根目录
SRC = BASE / "data" / "demo_input.jpg"
OUT = BASE / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def imread_zh(path):
    """读图（支持中文路径）

    ⚠️ OpenCV 经典坑：Windows 下 cv2.imread() 遇到中文路径会静默返回 None
    （不报错，只是读不到，新手能卡半天）。
    标准解法：用 np.fromfile 读字节流，再交给 cv2.imdecode 解码。
    工业现场同理——德日系视觉软件多半也不吃中文路径。
    """
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_zh(path, img):
    """存图（支持中文路径），与 imread_zh 配对使用"""
    cv2.imencode(".jpg", img)[1].tofile(str(path))


# ---------- 1. 读图 ----------
img = imread_zh(SRC)
if img is None:
    raise FileNotFoundError(f"读不到图片: {SRC}")

h, w, c = img.shape
print(f"图片尺寸 : {w} x {h} 像素")
print(f"数组形状 : {img.shape}   <- (高, 宽, 通道数) 注意顺序是 高在前")
print(f"数据类型 : {img.dtype}   <- uint8，每个通道 0~255")
print(f"像素总数 : {img.size:,}  <- {w} x {h} x {c} = 这就是第1周说的'二维数组'")
print(f"左上角像素 BGR 值: {img[0, 0]}   <- 注意 OpenCV 是 BGR 不是 RGB")

# ---------- 2. 灰度化 ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imwrite_zh(OUT / "gray.jpg", gray)
print(f"\n灰度图形状: {gray.shape}   <- 通道没了，只剩 (高, 宽)")

# ---------- 3. 二值化（对应康工笔记"预处理 → 阈值分割"）----------
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
imwrite_zh(OUT / "binary.jpg", binary)

white = (binary == 255).sum()
print(f"二值化后 : 白色像素 {white:,} / {binary.size:,} = {white / binary.size:.1%}")
print("-> 手写笔记的白色部分（纸面）被留下，黑字被压成 0")

# ---------- 4. 一句话总结 ----------
print(f"\n结果已保存: {OUT}")
print("整段代码只做了三件事：读成数组 -> 变换 -> 存回去。")
print("图像处理 = 数组运算，没有魔法。")
