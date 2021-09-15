import cv2
import numpy as np
import sys, os


def filterMasking(img, hsv, lower, upper):
    """
    入力画像に色フィルターを適用し、マスク画像を生成する関数
    :param img: 入力画像(BGR)。これにマスク画像を適用する。
    :param hsv: 入力画像(HSV)。これに色フィルターを適用する。
    :param lower: 色フィルターの下限値。
    :param upper: 色フィルターの上限値。

    :return result: 色フィルターによって、指定の色だけが残された画像(BGR)。
    :return mask: 生成したマスク画像。
    """
    mask = cv2.inRange(hsv, lower, upper)  # HSV画像に、HSVで指定した色フィルターを適用してマスク画像を生成する。
    result = cv2.bitwise_and(img, img, mask=mask)  # 生成したマスク画像をBGR画像に適用し、特定の色のピクセルだけを残す。
    return result, mask


"""
色の条件指定(フィルター設定)
Upper: 上限値
Lower: 下限値
H,S,V の順で指定する
# 今回使用する標識の色の理論値は以下の通り
# (  B,   G,   R) | (  H,   S,   V)
# (255,   0, 255) | (150, 255, 255)  # right
# (255,   0, 127) | (135, 255, 255)  # left
# (  0, 127, 255) | ( 15, 255, 255)  # stop
# (255,   0,   0) | (120, 255, 255) # 速度100
# (  0, 255,   0) | (110, 255, 255) # 速度50
# (  0,   0, 255) | (  0, 255, 255) # 速度10
"""

# 「右に曲がれ」標識
rightFilterUpper = np.array([155, 255, 255])
rightFilterLower = np.array([145, 100, 100])

# 「左に曲がれ」標識
leftFilterUpper = np.array([140, 255, 255])
leftFilterLower = np.array([130, 100, 100])

# 「止まれ」標識
stopFilterUpper = np.array([20, 255, 255])
stopFilterLower = np.array([10, 100, 100])

# 「速度を100%にしろ」標識
speed100FilterUpper = np.array([125, 255, 255])
speed100FilterLower = np.array([90, 100, 100])

# 「速度を50%にしろ」標識
speed50FilterUpper = np.array([115, 255, 255])
speed50FilterLower = np.array([55, 100, 100])

# 「速度を10%にしろ」標識
speed10FilterUpper = np.array([1, 255, 255])
speed10FilterLower = np.array([0, 100, 100])

# 動作確認用の画像を読み込む
img = cv2.imread("data/sample/11_left.jpg", 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGRからHSVに変換

"""
各標識のフィルターを適用して、マスク画像と処理結果を得る
"""
speed100Result, speed100Mask = filterMasking(img, hsv, speed100FilterLower, speed100FilterUpper)
speed50Result, speed50Mask = filterMasking(img, hsv, speed50FilterLower, speed50FilterUpper)
speed10Result, speed10Mask = filterMasking(img, hsv, speed10FilterLower, speed10FilterUpper)
rightResult, rightMask = filterMasking(img, hsv, rightFilterLower, rightFilterUpper)
leftResult, leftMask = filterMasking(img, hsv, leftFilterLower, leftFilterUpper)
stopResult, stopMask = filterMasking(img, hsv, stopFilterLower, stopFilterUpper)

"""
各マスク画像に写った白色の領域の面積を求め、しきい値を超えていれば、その標識が写っている判定を出す
(面積と言うが、正確にはマスク画像に写った白色のピクセルの個数で判定している)
"""
if speed100Mask.sum() > 205000:
    print("speed100Mask", speed100Mask.sum())
    cv2.imshow("", speed100Result)
    cv2.waitKey(0)

if speed50Mask.sum() > 205000:
    print("speed50Mask", speed50Mask.sum())
    cv2.imshow("", speed50Result)
    cv2.waitKey(0)

if speed10Mask.sum() > 205000:
    print("speed10Mask", speed10Mask.sum())
    cv2.imshow("", speed10Result)
    cv2.waitKey(0)

if rightMask.sum() > 205000:
    print("rightMask", rightMask.sum())
    cv2.imshow("", rightResult)
    cv2.waitKey(0)

if leftMask.sum() > 205000:
    print("leftMask", leftMask.sum())
    cv2.imshow("", leftResult)
    cv2.waitKey(0)

if stopMask.sum() > 205000:
    print("stopMask", stopMask.sum())
    cv2.imshow("", stopResult)
    cv2.waitKey(0)

# cv2.imshow("", leftResult)
# cv2.waitKey(0)
cv2.destroyAllWindows()
