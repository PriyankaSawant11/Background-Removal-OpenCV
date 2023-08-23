# NAME @: Priyanka Sawant
# TOPIC @: Background Removal of Model Deployment
# DATE @:  23/08/2023


# IMPORT THE DEPENDENCIES
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)

# #if you want to set the size of image internally while passing single image (height,width)
# cap.set(3, 640)
# cap.set(4, 480)

cap.set(cv2.CAP_PROP_FPS, 60)   #increase the frame rate
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()  #frame  per seconds
# imgBg = cv2.imread("Images\ Imgresizers-tree-736885_640.jpg")

#take multiple image through list using os
listImg = os.listdir("Images")
print(listImg)
imglist=[]
for imgpath in listImg:
    img= cv2.imread(f"Images/{imgpath}")   #read image from folder
    img = cv2.resize(img, (640, 480))  #resize the image
    imglist.append(img)
print(len(imglist))

indexImg=0   #to run in loop provide index initial value as 0

while True:

    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imglist[indexImg], threshold=0.8)

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))   #will print fps on screen with color(RED)
    print(indexImg)
    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'): #if we press 'a' key than if will go backward i.e print last image
        if indexImg>0:  #should be greater than 0
            indexImg -=1
    elif key == ord('d'):    #if we press 'd' key than if will go forward i.e print next image
        if indexImg < len(imglist)-1:
            indexImg += 1
    elif key == ord('q'):  #if we press 'q' key than it will close the window
        break