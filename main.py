import cv2
from tracker import *
import time
import numpy as np
# Create tracker object
tracker = EuclideanDistTracker()
import joblib
# from datetime import datetime 
pts = [] # for storing points
  
 # :mouse callback function
def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN: # Left click, select point
         pts.append((x, y))  
 
    if event == cv2.EVENT_RBUTTONDOWN: # Right click to cancel the last selected point
        pts.pop()  
 
    if event == cv2.EVENT_MBUTTONDOWN: # 
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
                 # 
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop
 
        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
 
        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)
 
        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)
        
    if len(pts) > 0:
                 # Draw the last point in pts
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
 
    if len(pts) > 1:
                 # 
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1) # x ,y is the coordinates of the mouse click place
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
    # img2=cv2.resize(img2,(800,600))
    cv2.imshow('image', img2)
cap = cv2.VideoCapture("ada.m4v")

_,img=cap.read()
# img = cv2.imread("3.png")
# img = imutils.resize(img, width=1200)
img = cv2.resize(img,(800,600))
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        saved_data = {
            "ROI": pts
        }

        break
cv2.destroyAllWindows()
# Object detection from Stable camera
subtracao = cv2.createBackgroundSubtractorKNN()
dizi=[]
def draw_line(cx,cy,id,frame):
    dizi.append([])
    center=(int(cx),int(cy))
    dizi[id].insert(len(dizi[id]),center)
    if len(dizi[id])>1:
        for i in range(len(dizi[id])-1):
                        
            cv2.line(frame,dizi[id][i],dizi[id][i+1],(0,0,255),1)

maps=np.zeros(img.shape, np.uint8)
id_dizi=[]
while True:
    _,img=cap.read()
    img = cv2.resize(img,(800,600))
    frame=img.copy()
    mask = np.zeros(img.shape, np.uint8)
    points = np.array(pts, np.int32)
    points = points.reshape((-1, 1, 2))
                 # 
    mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
    mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop
 
    show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

    # cv2.imshow("mask", mask2)
    
 
    roi = cv2.bitwise_and(mask2, img)
    
    # 1. Object Detection
    grey = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur=cv2.medianBlur(grey, ksize=7)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contours,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask = object_detector.apply(roi)
    # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 300:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])
    
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        id_dizi.append(id)
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        draw_line(cx,cy,id,maps)
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    count=max(id_dizi)
    cv2.putText(show_image, f"counter : {count}", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("show_img", show_image)
    # cv2.imshow("roi", roi)
    result = np.hstack((show_image,roi))
    # cv2.imshow("result", result)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("masp", maps)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()