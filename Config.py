import cv2 

MOTION_CALIBRATION_THRESHOLD = 100
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 125, 100) 
thickness = 2
queueAddInterval = 5
calibration = True 
calibrationTime = 10
fps = 30 
calibrationFrameCount = calibrationTime * fps