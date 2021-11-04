import cv2
import numpy as np


class ImageMaskingClass:
    def __init__(self):
        self.upper = np.array([179, 255, 255])
        self.lower = np.array([0, 0, 0])

    def filterMasking(self, img, hsv):
        self.mask = cv2.inRange(hsv, self.lower, self.upper)
        self.area = self.mask.sum() // 255
        result = cv2.bitwise_and(img, img, mask=self.mask)
        return result, self.mask


class StopSign(ImageMaskingClass):
    def __init__(self):
        super(StopSign).__init__()
        self.upper = np.array([30, 255, 255])
        self.lower = np.array([20, 100, 100])


class Speed10Sign(ImageMaskingClass):
    def __init__(self):
        super(Speed10Sign).__init__()
        self.upper = np.array([5, 255, 255])
        self.lower = np.array([0, 100, 100])


class Speed50Sign(ImageMaskingClass):
    def __init__(self):
        super(Speed50Sign).__init__()
        self.upper = np.array([115, 255, 255])
        self.lower = np.array([105, 100, 100])


class Speed100Sign(ImageMaskingClass):
    def __init__(self):
        super(Speed100Sign).__init__()
        self.upper = np.array([50, 255, 255])
        self.lower = np.array([40, 100, 100])


class RightSign(ImageMaskingClass):
    def __init__(self):
        super(RightSign).__init__()
        self.upper = np.array([85, 255, 255])
        self.lower = np.array([75, 100, 100])


class LeftSign(ImageMaskingClass):
    def __init__(self):
        super(LeftSign).__init__()
        self.upper = np.array([140, 255, 255])
        self.lower = np.array([130, 100, 100])


class ArrowSign(ImageMaskingClass):
    def __init__(self):
        super(ArrowSign).__init__()
        # 右向きと同じ色（向き検出が可能なので色分けする必要がない）
        self.upper = np.array([85, 255, 255])
        self.lower = np.array([75, 100, 100])

    def getArrowDirection(self):
        # 矢印の外接矩形を検出
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = max(cnts, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(cnts)

        # 重心から左右を検出
        m = cv2.moments(cnts)
        cx = int(m["m10"] / m["m00"])
        if (x + w // 2) - cx < 0:
            return "right"
        else:
            return "left"
