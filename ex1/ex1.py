import cv2
from ultralytics import YOLO

# YOLOv8 Pose モデルの読み込み
model = YOLO('yolov8n-pose.pt')

# 画像読み込み
image_path = 'ex1.jpg'
image = cv2.imread(image_path)

# 推論実行
results = model(image)

# 骨格とキーポイントを描画（顔は無視）
for result in results:
    keypoints = result.keypoints.xy[0]  # 1人目の人物のキーポイント

    # 顔のキーポイントインデックス（COCO準拠: 0〜4）
    face_keypoints = {0, 1, 2, 3, 4}

    # キーポイントを塗りつぶし黄色で描画（顔以外）
    for idx, point in enumerate(keypoints):
        if idx in face_keypoints:
            continue  # 顔は描画しない
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # 黄色で塗りつぶし

    # 骨格ライン描画（顔と無関係なライン）
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
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青い線で描画

# 結果画像の保存
cv2.imwrite('out_ex1.jpg', image)
