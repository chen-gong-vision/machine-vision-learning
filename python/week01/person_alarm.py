# -*- coding: utf-8 -*-
"""
人员进出检测报警程序

==================== 原理（工业标准的"两级检测"）====================
第一级  MOG2 背景建模（便宜，4ms/帧）
        自动学习"静止的背景"长什么样，画面里有东西在动就把前景抠出来。
        作用：快速判断"有没有动静"。没动静就直接跳过第二级，省算力。

第二级  HOG 行人检测（贵，35ms/帧）
        只在"有动静"的帧上跑，用方向梯度直方图特征判断"这个动的东西是不是人"。
        作用：把猫、光影晃动、窗帘飘动过滤掉，只对人报警。

为什么要分两级？
        直接在每帧跑 HOG 也能用，但你这台机器 640x480 下只有 6.9 fps。
        分两级后：静止时 0 开销，有人动时才花 35ms —— 这就是工业软件
        常用的"先便宜算子筛、再上贵算子"的性能套路。

==================== 操作 ====================
    q = 退出
    a = 布防 / 撤防切换（撤防时只显示不报警）
    s = 手动抓拍
    r = 进出计数清零

==================== 局限（必须知道）====================
    HOG 是 2005 年的老算法：对正面站立、全身可见的人效果最好；
    侧身、坐姿、只露半身、人很小的时候会漏检。
    OpenCV 4.x 自带的 HOG 只能"检测人"，不能"认出是谁"。
    工业现场现在主流是 YOLO 等深度学习方案（你 M5/M6 会学到），
    但 HOG 的好处是：零依赖、老机器能跑、原理看得见，适合打基础。
"""
import time
import winsound
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==================== 可调参数 ====================
CAMERA_INDEX = 1          # 1 = C920（跑一次看开头列表确认）
FRAME_W, FRAME_H = 640, 480      # 采集分辨率
DETECT_W, DETECT_H = 480, 360    # HOG 检测分辨率（降低分辨率是提速关键）

HOG_WIN_STRIDE = (8, 8)   # 滑窗步长，改 (16,16) 更快但可能漏检
HOG_PADDING = (8, 8)
HOG_SCALE = 1.05          # 多尺度检测的缩放系数
HOG_HIT_THRESH = 0.3      # HOG 置信度阈值，调高更严格（漏检多），调低更灵敏（误报多）
NMS_THRESH = 0.4          # 非极大值抑制：去掉重叠的检测框

MOTION_MIN_AREA = 1500    # 运动轮廓最小面积：小于这个不算"有动静"（过滤噪点/小物体）
MOG2_HISTORY = 500        # 背景模型学习的帧数
MOG2_VAR_THRESH = 50      # MOG2 灵敏度，调小更灵敏
WARMUP_FRAMES = 30        # 预热帧数：前 30 帧背景模型还没学好，不报警

ALARM_INTERVAL = 3.0      # 报警节流（秒）：避免一秒钟存几十张图、响几十次
BEEP_FREQ, BEEP_DUR = 1000, 180   # 报警声频率/时长

LINE_RATIO = 0.5          # 进出检测线位置（画面高度的 50% 处）
TRACK_MAX_DIST = 100      # 质心跟踪：两帧之间位移小于此值认为是同一个人
TRACK_MAX_MISSED = 8      # 连续丢失多少帧后删除该跟踪目标
# =================================================

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, WHITE, GRAY = (0, 200, 0), (0, 0, 255), (0, 200, 255), (255, 255, 255), (160, 160, 160)


def imwrite_zh(path, img):
    """存图（支持中文路径）—— cv2.imwrite 遇中文路径会失败"""
    cv2.imencode(".jpg", img)[1].tofile(str(path))


def beep():
    """声音报警（Windows）"""
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR)
    except Exception:
        pass  # 无蜂鸣器/非 Windows 就静默跳过，不影响主流程


class CentroidTracker:
    """极简质心跟踪器：给每个检测到的人一个临时 ID，用于判断"跨线进出"

    原理：两帧之间同一个人位移不会太大，所以"最近的质心"就是同一个人。
    工业上 ByteTrack / DeepSORT 就是这套思路的加强版（加了特征匹配）。
    局限：人多互相遮挡时会跟丢或 ID 跳变——本项目单人/少人场景够用。
    """

    def __init__(self, max_missed=TRACK_MAX_MISSED, max_dist=TRACK_MAX_DIST):
        self.tracks = {}      # id -> {'c': (x,y), 'side': -1/1, 'missed': n}
        self.next_id = 1
        self.max_missed = max_missed
        self.max_dist = max_dist

    def update(self, centroids, line_y):
        """返回本次发生的跨线事件列表：[('in' 或 'out', 跟踪ID), ...]"""
        events, used = [], set()

        # 1) 优先匹配已存在的跟踪目标
        for tid, t in list(self.tracks.items()):
            best, best_d = None, self.max_dist
            for i, c in enumerate(centroids):
                if i in used:
                    continue
                d = ((c[0] - t["c"][0]) ** 2 + (c[1] - t["c"][1]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best = d, i
            if best is None:
                t["missed"] += 1
                if t["missed"] > self.max_missed:
                    del self.tracks[tid]
                continue

            c = centroids[best]
            used.add(best)
            # 注意图像坐标系：y 向下增大。side=-1 表示人在检测线上方，+1 表示在下方
            side = -1 if c[1] < line_y else 1
            if t["side"] is not None and side != t["side"]:
                # 从上方跨到下方 = 向下走 = 记 IN；反向 = 向上走 = 记 OUT
                events.append(("in" if side == 1 else "out", tid))
            t.update({"c": c, "side": side, "missed": 0})

        # 2) 剩下的质心建立新跟踪目标
        for i, c in enumerate(centroids):
            if i in used:
                continue
            self.tracks[self.next_id] = {
                "c": c,
                "side": -1 if c[1] < line_y else 1,
                "missed": 0,
            }
            self.next_id += 1

        return events


def build_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_motion(mog, frame):
    """第一级：MOG2 背景建模，判断有没有"够大的东西"在动

    返回 (是否有动静, 前景掩码)
    """
    fg = mog.apply(frame)                       # 前景掩码：255=前景(在动)，127=阴影，0=背景
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # 去掉灰色阴影，只留实心前景
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # 开运算：去小白点
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # 闭运算：填小黑洞
    fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=2)            # 膨胀：把人的碎片连成整块

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has = any(cv2.contourArea(c) > MOTION_MIN_AREA for c in contours)
    return has, fg


def detect_people(hog, frame):
    """第二级：HOG 行人检测

    注意：先缩小到 DETECT_W x DETECT_H 再检测（提速 4 倍的关键），
    检测完再把坐标放大回原尺寸画框。
    返回 [(x, y, w, h), ...]
    """
    small = cv2.resize(frame, (DETECT_W, DETECT_H))
    rects, weights = hog.detectMultiScale(
        small, winStride=HOG_WIN_STRIDE, padding=HOG_PADDING,
        scale=HOG_SCALE, hitThreshold=HOG_HIT_THRESH,
    )
    if len(rects) == 0:
        return []

    rects = list(rects)
    scores = [float(w) for w in weights]
    # NMS 去掉重叠框（HOG 常对同一个人给出多个重叠框）
    idxs = cv2.dnn.NMSBoxes(rects, scores, HOG_HIT_THRESH, NMS_THRESH)

    sx, sy = FRAME_W / DETECT_W, FRAME_H / DETECT_H
    out = []
    for i in np.array(idxs).flatten():
        x, y, w, h = rects[i]
        out.append((int(x * sx), int(y * sy), int(w * sx), int(h * sy)))
    return out


def draw_hud(frame, state):
    """画状态栏。⚠️ OpenCV 的 putText 不支持中文，画面文字一律用英文"""
    h = frame.shape[0]
    armed = state["armed"]

    if state["warming"]:
        cv2.putText(frame, f"WARMING UP {state['warm_left']}", (12, 32), FONT, 0.8, YELLOW, 2)
    elif not armed:
        cv2.putText(frame, "DISARMED (press 'a' to arm)", (12, 32), FONT, 0.8, GRAY, 2)
    elif state["n_people"] > 0:
        cv2.putText(frame, "ALARM! PERSON DETECTED", (12, 38), FONT, 1.0, RED, 3)
    else:
        cv2.putText(frame, "ARMED - monitoring", (12, 32), FONT, 0.8, GREEN, 2)

    cv2.putText(frame, f"people: {state['n_people']}   IN: {state['count_in']}   OUT: {state['count_out']}",
                (12, h - 42), FONT, 0.6, WHITE, 1)
    cv2.putText(frame, f"fps: {state['fps']:.0f}", (12, h - 20), FONT, 0.6, WHITE, 1)
    cv2.putText(frame, "q=quit  a=arm/disarm  s=snapshot  r=reset count",
                (150, h - 20), FONT, 0.45, GRAY, 1)

    if state["last_alarm"] and time.time() - state["last_alarm"] < 2.0:
        cv2.putText(frame, "SAVED", (FRAME_W - 110, 32), FONT, 0.7, RED, 2)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"打不开摄像头 index={CAMERA_INDEX}。关掉占用摄像头的软件，或改编号")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hog = build_hog()
    mog = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESH, detectShadows=True
    )
    tracker = CentroidTracker()

    state = {
        "armed": True, "warming": True, "warm_left": WARMUP_FRAMES,
        "n_people": 0, "count_in": 0, "count_out": 0,
        "fps": 0.0, "last_alarm": 0.0,
    }
    frame_no, last_alarm_t, last_beep_t = 0, 0.0, 0.0

    print("运行中：q=退出  a=布防/撤防  s=抓拍  r=计数清零")
    print(f"（前 {WARMUP_FRAMES} 帧是背景学习期，不报警）")

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("取帧失败")
            break
        frame_no += 1
        display = frame.copy()
        line_y = int(FRAME_H * LINE_RATIO)

        # ---------- 第一级：有没有动静 ----------
        has_motion, fgmask = detect_motion(mog, frame)
        state["warming"] = frame_no <= WARMUP_FRAMES
        state["warm_left"] = max(0, WARMUP_FRAMES - frame_no)

        # ---------- 第二级：有动静才检测是不是人 ----------
        boxes = []
        if has_motion and not state["warming"]:
            boxes = detect_people(hog, frame)

        state["n_people"] = len(boxes)

        # 画检测框
        for (x, y, w, h) in boxes:
            cv2.rectangle(display, (x, y), (x + w, y + h), GREEN, 2)
            cv2.putText(display, "person", (x, max(12, y - 6)), FONT, 0.5, GREEN, 1)

        # ---------- 进出计数（跨线判定）----------
        centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
        for direction, tid in tracker.update(centroids, line_y):
            if direction == "in":
                state["count_in"] += 1
                print(f"  ↓ IN  向下穿过检测线 (ID {tid})  累计 IN {state['count_in']} / OUT {state['count_out']}")
            else:
                state["count_out"] += 1
                print(f"  ↑ OUT 向上穿过检测线 (ID {tid})  累计 IN {state['count_in']} / OUT {state['count_out']}")
        # 画检测线和跟踪点
        cv2.line(display, (0, line_y), (FRAME_W, line_y), YELLOW, 1)
        cv2.putText(display, "counting line", (6, line_y - 6), FONT, 0.4, YELLOW, 1)
        for c in centroids:
            cv2.circle(display, c, 4, RED, -1)

        # ---------- 报警：截图留证 + 声音（都做节流）----------
        now = time.time()
        if (state["armed"] and not state["warming"] and len(boxes) > 0
                and now - last_alarm_t > ALARM_INTERVAL):
            name = datetime.now().strftime("alarm_%m%d_%H%M%S.jpg")
            imwrite_zh(OUT_DIR / name, display)
            print(f"  🔴 报警！已存证 data/output/{name}（检出 {len(boxes)} 人）")
            last_alarm_t = state["last_alarm"] = now
            if now - last_beep_t > ALARM_INTERVAL:
                beep()
                last_beep_t = now

        state["fps"] = 1.0 / max(1e-6, time.time() - t0)
        draw_hud(display, state)
        cv2.imshow("Person Alarm", display)

        # 前景掩码单独开个小窗，方便理解 MOG2 在干什么（可按 ESC 关掉这个小窗）
        cv2.imshow("MOG2 foreground (black=bg, white=moving)", fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            state["armed"] = not state["armed"]
            print(f"  布防状态: {'ARMED 布防中' if state['armed'] else 'DISARMED 已撤防'}")
        elif key == ord("s"):
            name = datetime.now().strftime("snap_%m%d_%H%M%S.jpg")
            imwrite_zh(OUT_DIR / name, display)
            print(f"  ✓ 手动抓拍 -> data/output/{name}")
        elif key == ord("r"):
            state["count_in"] = state["count_out"] = 0
            tracker.tracks.clear()
            print("  计数已清零")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n结束。最终统计：进入 {state['count_in']} 人 / 离开 {state['count_out']} 人")


if __name__ == "__main__":
    main()
