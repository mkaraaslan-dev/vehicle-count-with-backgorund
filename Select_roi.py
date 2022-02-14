# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:59:37 2022

@author: MAHMUT KARAASLAN
"""

import cv2
import imutils
import numpy as np
import pickle

 
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
    # cv2.imshow('image', img2)
 
 
 #Create images and windows and bind windows to callback functions
cap = cv2.VideoCapture("ada.m4v")
_,img= cap.read()
img = cv2.resize(img,(800,600))
# img = imutils.resize(img, width=1200)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)
 # Print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
 # Print("[INFO] Press ‘S’ to determine the selection area and save it)
 # Print("[INFO] Press ESC to quit")

file =  open('roi.txt', 'w')

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        
        file =  open('roi.txt', 'w')
        # print(type(pts))
        file.write(str(pts))
                 # Print("[INFO] ROI coordinates have been saved to local.")
        break
file.close()
cv2.destroyAllWindows()
