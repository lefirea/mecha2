import numpy as np
import cv2
import sys, os


# 検証用の画像を用意する
# img = np.full((500, 500, 3), 128, dtype=np.uint8)  # 500x500ピクセルの画像を用意し、グレー単色で塗りつぶす
img = cv2.imread("data/sample/1_left.jpg")  # 指定の画像を読み込む

# 用意した画像に矢印を行場する
# img = cv2.arrowedLine(img, (100, 250), (400, 250), (255, 0, 127), thickness=80, tipLength=0.5)
img = cv2.arrowedLine(img, (200, 300), (300, 300), (255, 0, 255), thickness=50, tipLength=0.5)
# cv2.imshow("", img)
# cv2.waitKey(0)

# HSVに変換する
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 色フィルター設定
upper = np.array([160, 255, 255])
lower = np.array([140, 100, 100])

# マスク処理
mask = cv2.inRange(hsv, lower, upper)
ret = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow("", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# sys.exit()

# 矢印の外接矩形を検出
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(cnts[0])
print(x, y, w, h)
arrow = mask[y:y+h, x:x+w]  # 矢印の領域だけ切り出す

# モーメント(重心)から左右を検出
m = cv2.moments(cnts[0])
cx = int(m["m10"]/m["m00"])
cy = int(m["m01"]/m["m00"])
print(cx, cy)  # 重心座標を表示
if (x + w // 2) - cx < 0:  # 外接矩形の中心より右側なら（中心よりｘ方向に寄っていれば）右向き
    print("right")
else:
    print("left")

# 矢印の重心と中心を描画
# img = cv2.circle(img, (cx, cy), 10, (255, 0, 0), thickness=-1)  # 重心
# img = cv2.circle(img, (x + w // 2, y + h // 2), 10, (0, 255, 255), thickness=3)  # 中心

cv2.imshow("", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
