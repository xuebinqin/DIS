import cv2 
from motionHelper import MotionDetection
from Config import *

import time 

motionDetector = MotionDetection()

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
motionDetector.motionUpdate(frame)
liveCount = 0
prevFrame = frame 


if calibration:
    while True:
        ret, frame = cap.read()
        motionDetector.motionCalibration(frame)
        motionValue = motionDetector.motionValue
        live_frame = cv2.putText(live_frame, f"Motion Calibration in Progress : {motionValue}", org, font,  fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Calibration Frame', live_frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break 
    cv2.destroyAllWindows()
    liveCount = 0

motionStatus = False 
while True:
    liveCount += 1 
    ret,frame = cap.read()
    live_frame = frame.copy()

    if liveCount % queueAddInterval == 0:
        motionDetector.motionUpdate(frame)
        motionStatus = motionDetector.motion
    live_frame = cv2.putText(live_frame, f"Motion : {motionStatus}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Live Stream for Motion', live_frame)
    prevFrame = frame 

    

    
