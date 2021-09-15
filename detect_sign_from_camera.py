import cv2
import numpy as np
import sys, os


def filterMasking(img, hsv, lower, upper):
	mask = cv2.inRange(hsv, lower, upper)
	result = cv2.bitwise_and(img, img, mask=mask)
	return result, mask


rightFilterUpper = np.array([155, 255, 255])
rightFilterLower = np.array([145, 100, 100])

leftFilterUpper = np.array([140, 255, 255])
leftFilterLower = np.array([130, 100, 100])

stopFilterUpper = np.array([20, 255, 255])
stopFilterLower = np.array([10, 100, 100])

speed100FilterUpper = np.array([125, 255, 255])
speed100FilterLower = np.array([90, 100, 100])

speed50FilterUpper = np.array([115, 255, 255])
speed50FilterLower = np.array([55, 100, 100])

speed10FilterUpper = np.array([1, 255, 255])
speed10FilterLower = np.array([0, 100, 100])

cv2.namedWindow("camera")
cv2.namedWindow("result")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
	_, frame = cap.read()
	cv2.imshow("read", frame)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	speed100Result, speed100Mask = filterMasking(frame, hsv, speed100FilterLower, speed100FilterUpper)
	speed50Result, speed50Mask = filterMasking(frame, hsv, speed50FilterLower, speed50FilterUpper)
	speed10Result, speed10Mask = filterMasking(frame, hsv, speed10FilterLower, speed10FilterUpper)
	rightResult, rightMask = filterMasking(frame, hsv, rightFilterLower, rightFilterUpper)
	leftResult, leftMask = filterMasking(frame, hsv, leftFilterLower, leftFilterUpper)
	stopResult, stopMask = filterMasking(frame, hsv, stopFilterLower, stopFilterUpper)

	if speed100Mask.sum() > 205000:
		print("speed100", speed100Mask.sum())
		cv2.imshow("result", speed100Result)
	if speed50Mask.sum() > 205000:
		print("speed50", speed50Mask.sum())
		cv2.imshow("result", speed50Result)
	if speed10Mask.sum() > 205000:
		print("speed10", speed10Mask.sum())
		cv2.imshow("result", speed10Result)
	
	if rightMask.sum() > 205000:
		print("right", rightMask.sum())
		cv2.imshow("result", rightResult)
	if leftMask.sum() > 205000:
		print("speed100", leftMask.sum())
		cv2.imshow("result", leftResult)
	if stopMask.sum() > 205000:
		print("speed100", stopMask.sum())
		cv2.imshow("result", stopResult)
	
    # "q"キーが入力されたらプログラムを終了する
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
