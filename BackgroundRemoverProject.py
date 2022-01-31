import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()

fpsReader = cvzone.FPS()

#imgBG = cv2.imread("BackgroundRemoverImages/image5.jpg")

listImg = os.listdir("BackgroundRemoverImages")
print(listImg)


imgList = []

for imgPath in listImg:
    img = cv2.imread(f'BackgroundRemoverImages/{imgPath}')
    imgList.append(img)
print(len(listImg))
#print(imgList)

indexImg = 1
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgList[indexImg],threshold=0.8)

    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    fps, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))
    print(indexImg)
    cv2.imshow("Image", imgStacked)

    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg > 1:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg <len(imgList)-1:
            indexImg += 1
    elif key == ord('q'):
        break

