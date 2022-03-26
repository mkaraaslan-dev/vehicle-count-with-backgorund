import cv2
from tracker import *
import time
import numpy as np
# Create tracker object
tracker = EuclideanDistTracker()



"""
    Getting roi value
"""

roi_file = open('roi.txt', 'r')
pts=roi_file.read()

pts=eval(pts) 

"""
applying roi values on frame 

"""

def roi_frame(img,pts):
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop
 
        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("show_img", show_image)
        return (ROI,show_image)

cap = cv2.VideoCapture("ada.m4v")




"""
Object detection with background method
"""

subtracao = cv2.createBackgroundSubtractorKNN()
dizi=[]



# maps=np.zeros(img.shape, np.uint8)
id_dizi=[]
while True:
    _,img=cap.read()
    img = cv2.resize(img,(800,600))
    frame=img.copy()

    " Generate roi "
    
    roi,show_image=roi_frame(img,pts)


    
    " Object Detection"
   
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
        if area > 400:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])
    
    
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        id_dizi.append(id)


        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    " Get vehicle count "
    
    count=max(id_dizi)
    
    " Show image "
    
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
