# -*- coding: utf-8 -*-
"""
摄像头差异检测 —— 工业 AOI 最基础的形态："标准件比对"

原理（工业上叫 Golden Sample / 标准件比对）：
    1. 先把一件"合格品"放到镜头前，按 s 拍下来当基准（baseline）
    2. 之后每件产品放上来，程序实时把当前画面和基准做差分
    3. 不一样的像素超过阈值 -> 判 NG（不合格），画面红色高亮差异区域 + 报警

这就是 AOI 检测（手机屏划痕、零件漏装、标签错贴）的核心逻辑。

操作：
    s = 拍基准（把合格品放好再按）
    c = 抓拍当前画面存档（存到 data/output/）
    q = 退出

⚠️ 重要认知：不能拿"摄像头实时画面"去比对"你电脑里的一张照片"
    —— 角度、光照、分辨率全不一样，必然 100% 差异，永远判 NG。
    必须"用同一颗摄像头、同一位置、同一光照"先拍一张当基准，再比同类物品。
    这也是为什么工业现场光源要恒定、夹具要定位。

⚠️ OpenCV 的 putText 不支持中文（画出来是乱码），所以画面文字一律用英文。
"""
import cv2
import numpy as np
from pathlib import Path

# ==================== 可调参数 ====================
CAMERA_INDEX = 1      # 摄像头编号：1 = C920（跑一次看列表确认）
WIDTH, HEIGHT = 1280, 720   # 采集分辨率
GAUSS_KSIZE = (5, 5)        # 高斯模糊核：压掉摄像头噪点，避免"假差异"
PIXEL_THRESH = 30           # 单像素灰度差超过 30 才算"变了"(0~255)
NG_RATIO = 0.02             # 差异像素占比 > 2% 判 NG（可调，见文末说明）
# =================================================

BASE = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = BASE / "data" / "baseline.jpg"
OUT_DIR = BASE / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def imread_zh(path):
    """读图（支持中文路径）—— cv2.imread 遇中文路径会静默返回 None"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_zh(path, img):
    """存图（支持中文路径）"""
    cv2.imencode(".jpg", img)[1].tofile(str(path))


def to_gray(img):
    """转灰度 + 高斯模糊去噪

    为什么要模糊：摄像头传感器有噪点，即使画面完全静止，
    逐像素相减也会有小幅波动，不处理会误判成 NG。
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, GAUSS_KSIZE, 0)


def compare(cur_bgr, base_bgr):
    """返回 (差异率, 差异掩码)

    差异掩码：255 表示该位置"和基准不一样"，0 表示一样
    """
    a, b = to_gray(cur_bgr), to_gray(base_bgr)
    # 若基准图和当前帧尺寸不同，先把基准缩放到当前尺寸
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    diff = cv2.absdiff(a, b)                       # 逐像素相减取绝对值
    _, mask = cv2.threshold(diff, PIXEL_THRESH, 255, cv2.THRESH_BINARY)
    ratio = float((mask > 0).mean())               # 不同像素占比
    return ratio, mask


def draw_hud(frame, ratio, has_baseline):
    """在画面上画状态栏（英文，因为 OpenCV 画中文会乱码）"""
    if not has_baseline:
        cv2.putText(frame, "NO BASELINE - press 's' to capture",
                    (20, 45), FONT, 1.0, (0, 165, 255), 2)
    else:
        ok = ratio <= NG_RATIO
        label, color = ("OK", (0, 200, 0)) if ok else ("NG !!!", (0, 0, 255))
        cv2.putText(frame, label, (20, 60), FONT, 2.0, color, 4)
        cv2.putText(frame, f"diff: {ratio * 100:.2f}%  (NG > {NG_RATIO * 100:.0f}%)",
                    (20, 105), FONT, 0.8, (255, 255, 255), 2)
        if not ok:
            cv2.putText(frame, "DIFFERENT FROM BASELINE", (20, 150), FONT, 0.9, (0, 0, 255), 2)

    # 底部操作提示
    cv2.putText(frame, "s=set baseline   c=capture   q=quit",
                (20, frame.shape[0] - 20), FONT, 0.6, (200, 200, 200), 1)


def main():
    # 载入已有基准（如果之前拍过）
    baseline = imread_zh(BASELINE_PATH) if BASELINE_PATH.exists() else None
    if baseline is not None:
        print(f"已载入旧基准: {BASELINE_PATH.name}  {baseline.shape[1]}x{baseline.shape[0]}")
    else:
        print("尚无基准图，先摆好合格品按 s 拍照")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"打不开摄像头 index={CAMERA_INDEX}（关掉微信/腾讯会议，或改编号）")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    print("运行中：s=拍基准  c=抓拍  q=退出")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("取帧失败（摄像头被抢占？）")
            break

        display = frame.copy()
        if baseline is not None:
            ratio, mask = compare(frame, baseline)
            # 差异区域红色高亮：把"不一样"的像素染红再半透明叠加
            overlay = display.copy()
            overlay[mask > 0] = (0, 0, 255)   # BGR，红色
            display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)
            draw_hud(display, ratio, True)
        else:
            draw_hud(display, 0.0, False)

        cv2.imshow("AOI Diff Detect", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):                       # 拍基准
            baseline = frame.copy()
            imwrite_zh(BASELINE_PATH, baseline)
            print(f"✓ 基准已保存 -> {BASELINE_PATH}")
        elif key == ord("c"):                     # 抓拍存档
            from datetime import datetime
            name = datetime.now().strftime("capture_%H%M%S.jpg")
            imwrite_zh(OUT_DIR / name, display)
            print(f"✓ 已抓拍 -> data/output/{name}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
