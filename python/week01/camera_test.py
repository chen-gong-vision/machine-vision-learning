# -*- coding: utf-8 -*-
"""
USB 摄像头实时预览（验证用）

用法：
    1. 把 USB 摄像头插好（笔记本内置摄像头 index=0 也能用）
    2. PyCharm 里右键本文件 -> 运行
    3. 弹出窗口实时显示画面
    4. 按 q 退出（务必用 q 退出，否则摄像头会被程序一直占用）

踩坑提示：
    - 打不开画面：把下面 VIDEO_SOURCE 从 0 改成 1 / 2 试试，
      多摄像头时系统给的编号不一定是 0
    - 黑屏/卡顿：先把其他占用摄像头的软件（微信、腾讯会议）关掉
    - 工业相机（海康/大华/Basler 等 GigE/USB3 视觉相机）不走这套，
      要用厂家 SDK 或 GenICam，普通 USB 摄像头才用 VideoCapture
"""
import cv2

VIDEO_SOURCE = 0  # 0 = 默认摄像头；多摄像头时试 1、2…


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"✗ 打不开摄像头 index={VIDEO_SOURCE}")
        print("  试试把 VIDEO_SOURCE 改成 1、2；并关闭占用摄像头的软件")
        return

    print("✓ 摄像头已打开，按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("✗ 读取帧失败")
            break

        cv2.imshow("USB Camera (按 q 退出)", frame)

        # 等待 1ms；收到 q 键（ASCII 113）就退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()          # 释放摄像头（必须）
    cv2.destroyAllWindows()  # 关闭窗口


if __name__ == "__main__":
    main()
