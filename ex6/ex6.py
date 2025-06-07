from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 画像パス
path = "ex6/ex6.jpg"
results = model(path, save=False)
boxes = results[0].boxes
class_names = results[0].names
img = cv2.imread(path)

# HSVの色範囲定義
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_cyan = np.array([85, 50, 100])
upper_cyan = np.array([95, 255, 255])

for box in boxes:
    data = box.data.cpu().numpy()[0]
    x1, y1, x2, y2, conf, cls = map(int, data[:6])
    class_id = int(cls)

    if class_names[class_id] == 'person':
        person_crop = img[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        h, w = person_crop.shape[:2]

        # 肩付近クロップ（高さの10%〜40%、幅は中央50%）
        crop_top = int(h * 0.1)
        crop_bottom = int(h * 0.4)
        crop_left = int(w * 0.25)
        crop_right = int(w * 0.75)
        shoulder_crop = person_crop[crop_top:crop_bottom, crop_left:crop_right]

        hsv = cv2.cvtColor(shoulder_crop, cv2.COLOR_BGR2HSV)
        total_pixels = shoulder_crop.shape[0] * shoulder_crop.shape[1]

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = cv2.countNonZero(mask_yellow) / total_pixels

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        red_ratio = cv2.countNonZero(red_mask) / total_pixels

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = cv2.countNonZero(mask_blue) / total_pixels

        mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
        cyan_ratio = cv2.countNonZero(mask_cyan) / total_pixels

        # チーム判定
        if yellow_ratio > 0.1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # ドルトムント（黄）
        elif (red_ratio + blue_ratio > 0.07) and (cyan_ratio < 0.06):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)    # バルセロナ（赤）

cv2.imwrite("ex6/outex6.jpg", img)
