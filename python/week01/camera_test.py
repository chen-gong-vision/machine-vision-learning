# -*- coding: utf-8 -*-
"""
USB 摄像头实时预览（验证用）

本机实际有两颗摄像头：
    0 = Integrated Camera（内置）
    1 = HD Pro Webcam C920（你插的罗技）  <- 默认用这颗

用法：
    1. 插好 C920（注意拉开镜头上的物理隐私快门小拨片，否则画面全黑）
    2. PyCharm 里右键本文件 -> 运行
    3. 程序会先列出所有能打开的摄像头 index，再打开 VIDEO_SOURCE
    4. 弹出窗口实时显示 -> 按 q 退出（务必用 q，否则摄像头被一直占用）

打不开的常见原因（按概率排序）：
    ① 别的软件占着摄像头：关掉 微信 / 腾讯会议 / Windows 相机 / 含摄像头的网页
    ② 隐私快门没拉开：C920 镜头上的小拨片拨到侧面
    ③ 多摄像头编号不对：看运行开头的列表，把 VIDEO_SOURCE 改成对应数字
    ④ 工业相机（海康/大华/Basler）不走这套，需厂家 SDK / GenICam
"""
import cv2

VIDEO_SOURCE = 1  # 默认 1 = C920；若列表显示不同，改成对应数字


def list_cameras(max_try=6):
    """扫描并列出所有能打开的摄像头 index"""
    print("正在扫描可用摄像头……")
    found = []
    for i in range(max_try):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    if not found:
        print("  ✗ 一个都没扫到。检查：USB 是否插好 / 是否被其他软件占用")
    else:
        print("  ✓ 能打开的 index:", found, "（VIDEO_SOURCE 选其中一个）")
    return found


def main():
    list_cameras()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"\n✗ 打不开摄像头 index={VIDEO_SOURCE}")
        print("  先关掉占用摄像头的软件（微信/腾讯会议/相机），再改 VIDEO_SOURCE 试别的号")
        return

    # 把分辨率拉到 1280x720（C920 默认只开 480x640 的 VGA，太糊）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n✓ C920 已打开  {w}x{h}  (报告帧率 {fps:.0f}fps，实际看流畅度)")
    print("  按 q 退出")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("✗ 取帧失败（可能被其他程序抢占了，关掉它再试）")
            break

        # 隐私快门没拉开时整帧接近纯黑，给个提示
        if frame.mean() < 5:
            cv2.putText(frame, "画面全黑? 检查C920镜头隐私快门是否拉开",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("C920 (按 q 退出)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
