
import imutils
import cv2
image = cv2.imread("sample_1.jpeg")

def resized_roi(image):
    resized = imutils.resize(image, width=1024)
    roi = resized[100:700, 50:950]
    return roi
