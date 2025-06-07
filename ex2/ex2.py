import cv2
from ultralytics import YOLO

# YOLOv8 Pose モデルの読み込み
model = YOLO('yolov8n-pose.pt')

# 画像読み込み
image_path = 'ex2/ex1.jpg'
image = cv2.imread(image_path)

results = model(image)

# キーポイントインデックスの定義
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# 骨格とキーポイントを描画
for result in results:
    keypoints = result.keypoints.xy[0]  # (17, 2) shape の Tensor

    # 顔のキーポイントインデックス（COCO準拠: 0〜4）
    face_keypoints = {0, 1, 2, 3, 4}

    # キーポイントを黄色で描画（顔以外）
    for idx, point in enumerate(keypoints):
        if idx in face_keypoints:
            continue
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # 黄色

    # 骨格ライン描画
    skeleton = [
        (10, 8), (8, 6), (6, 5), (5, 7),
        (7, 9), (6, 12), (5, 11),
        (11, 12), (12, 14), (11, 13),
        (14, 16), (13, 15)
    ]
    for joint in skeleton:
        if joint[0] < len(keypoints) and joint[1] < len(keypoints):
            p1 = keypoints[joint[0]]
            p2 = keypoints[joint[1]]
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青線

    # ✅ 肩と腰の中心座標を直接計算
    ls = keypoints[LEFT_SHOULDER]
    rs = keypoints[RIGHT_SHOULDER]
    lh = keypoints[LEFT_HIP]
    rh = keypoints[RIGHT_HIP]

    center_x = float(ls[0] + rs[0] + lh[0] + rh[0]) / 4
    center_y = float(ls[1] + rs[1] + lh[1] + rh[1]) / 4

    # 赤丸で描画
    cv2.circle(image, (int(center_x), int(center_y)), 6, (0, 0, 255), -1)

# 画像保存
cv2.imwrite('ex2/out_ex2.jpg', image)
