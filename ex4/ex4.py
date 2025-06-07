from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 画像のパス
path = "ex04/ex4.jpg"

# モデルを使用して画像から物体を検出
results = model(path, save=False)  # save=Trueだと元画像に書き込まれるのでFalseに変更

# 検出結果からボックスを抽出
boxes = results[0].boxes
class_names = results[0].names

# 画像の読み込み
img = cv2.imread(path)

# 一番大きなpersonのバウンディングボックスを探す
max_area = 0
max_box = None

for box in boxes:
    data = box.data.cpu().numpy()[0]
    x1, y1, x2, y2, conf, cls = data
    class_id = int(cls)
    if class_names[class_id] == 'person':
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            max_box = (int(x1), int(y1), int(x2), int(y2))

# 一番大きなボックスのみ描画
if max_box:
    x1, y1, x2, y2 = max_box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 赤色の枠

# 結果の保存
cv2.imwrite("ex04/outex4.jpg", img)
