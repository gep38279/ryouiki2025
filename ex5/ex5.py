from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 画像のパス
path = "ex5/ex5.jpg"

# モデルを使用して画像から物体を検出
results = model(path, save=False)

# 検出結果からボックスを抽出
boxes = results[0].boxes
class_names = results[0].names

# 画像の読み込み
img = cv2.imread(path)

# より狭い青の範囲（黒が入らないように調整）
lower_blue = np.array([90, 40, 60])   # 彩度・明度を上げて黒除外
upper_blue = np.array([150, 255, 255])


for box in boxes:
    data = box.data.cpu().numpy()[0]
    x1, y1, x2, y2, conf, cls = data
    class_id = int(cls)

    if class_names[class_id] == 'person':
        # 座標をintにして切り抜き
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        person_crop = img[y1:y2, x1:x2]

        # BGR→HSVに変換して青色ピクセルの割合を算出
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = cv2.countNonZero(mask) / (person_crop.shape[0] * person_crop.shape[1])

        # 青色の割合が一定以上なら赤枠で囲む
        if blue_ratio > 0.1:  # 15%以上が青なら青選手とみなす
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 結果の保存
cv2.imwrite("ex5/outex5.jpg", img)
