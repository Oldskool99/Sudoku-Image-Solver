import cv2
import numpy as np
import imutils

webcam = cv2.VideoCapture(0)

    

def binary_Image (a):
    
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("blur", blur)
    
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    #cv2.imshow("thresh", thresh)
    
    return thresh



while True:
    
    
    key = cv2.waitKey(1)
    
    ret,frame = webcam.read()
    #cv2.imshow ("Live", frame)
    
    
    
    cnts = cv2.findContours(binary_Image(frame), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    
    for c in cnts:
    	
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    	if len(approx) == 4:
    		screenCnt = approx
    		break
    
    
    if screenCnt is not None:    
        
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")
        
        #  TL has smallest Sum, BR has largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # TR has smallest difference, BL has largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        
        
        (tl, tr, br, bl) = rect
        # Width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        
        # Height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        
        # Final Dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
       
        # Destination points
        dst = np.array([
        	[0, 0],
        	[maxWidth - 1, 0],
        	[maxWidth - 1, maxHeight - 1],
        	[0, maxHeight - 1]], dtype = "float32")
        
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Warp transform
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    
        cv2.imshow('Initial Frame', frame)  
        
        
        cv2.imshow('Transformed Capture', warp) 
        
       
        lines = cv2.HoughLinesP(binary_Image(warp),1,np.pi/180,100,minLineLength=100,maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(warp,(x1,y1),(x2,y2),(0,255,0),2)
            
            cv2.imshow('HoughP Transform', warp)
        
    
    if key == 27:
        break
    
webcam.release()
cv2.destroyAllWindows()



    