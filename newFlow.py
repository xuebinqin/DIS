import cv2 
from motionHelper import MotionDetection
from Config import *
from inf import * 
import time 
from maskHelper import maskImage

maskClass = maskImage()
motionDetector = MotionDetection()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
motionDetector.frameQueue.append(frame)
liveCount = 0
prevFrame = frame 


if calibration:
    while True:
        ret, frame = cap.read()
        motionDetector.motionCalibration(frame)
        motionValue = motionDetector.motionValue
        frame = cv2.putText(frame, f"Motion Calibration in Progress : {motionValue}\n Press E to exit", org, font,  fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Calibration Frame', frame)
        if cv2.waitKey(1) & 0xff == ord('e'):
            break 
    cv2.destroyAllWindows()
    liveCount = 0

motionStatus = False 
prevStatus = False 
currentStatus = False 
startFrame = frame
beforeMotionFrame = None 

while True:
    liveCount += 1 
    ret,frame = cap.read()
    live_frame = frame.copy()

    if liveCount % queueAddInterval == 0:
        motionDetector.motionUpdate(frame)
        currentStatus = motionDetector.motion
        motionVal = motionDetector.motionValue
        threshold = motionDetector.threshold
        print(f"Motion Threshold : {threshold} x {motionVal}")

    if prevStatus is False and currentStatus is True: #### That means the motion started, we need to capture the previous frame
        beforeMotionFrame = prevFrame

    if prevStatus == True and currentStatus == False: #### After motion ended 
        currentMask = inference(frame)
        cv2.imwrite('mask.jpg', currentMask)
        cv2.imshow('Live Mask', currentMask)
        previousMask = inference(beforeMotionFrame)
        croppedImage = maskClass.maskedImage(frame, currentMask)
        cv2.imshow('Cropped Image', croppedImage)





    live_frame = cv2.putText(live_frame, f"Motion : {currentStatus}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Live Stream for Motion', live_frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break 
    prevFrame = frame 
    prevStatus = currentStatus
cv2.destroyAllWindows()

    

    
