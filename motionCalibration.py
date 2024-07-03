
import cv2 
from skimage.metrics import structural_similarity as ssim
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging 
import time 

global CHANGE 
global MOTION_VAL 

MOTION_VAL = 1000
CHANGE = False 
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 125, 100) 
thickness = 2
   

frameQueue = deque(maxlen=10)
executor = ThreadPoolExecutor(max_workers=5)

def check_significant_change(frame1, frame2, threshold=0.93):
    try:
        global CHANGE, MOTION_VAL 
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between two frames
        ssim_index, _ = ssim(gray_frame1, gray_frame2, full=True)
        print(ssim_index)
        # Check if the SSIM index is below the threshold
        if ssim_index < threshold:
            CHANGE = True 
        else:
            CHANGE = False 
        if ssim_index < MOTION_VAL:
            MOTION_VAL = ssim_index 
        return 
    except Exception as e:
        logging.info(f"Code failed due to:\n {e}")
        raise e 
    
def calibration(camera_id, queueAddInterval):
    global MOTION_VAL
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    frameQueue.append(frame)
    if ret is None:
        raise BufferError("There are no frames captured")
    s = time.time()
    runtimeCounter = 0
    while True:
        print(runtimeCounter)
        runtimeCounter += 1 
        ret, frame = cap.read()
        if runtimeCounter % queueAddInterval == 0:
            frameQueue.append(frame)
            executor.submit(check_significant_change, frameQueue[-1], frameQueue[-2])
        ## GET CONSTANT VALUE ##
        e = time.time()
        frame = cv2.putText(frame, f'{MOTION_VAL}' , org, font,  fontScale, color, thickness, cv2.LINE_AA) 
        cv2.imshow('Calibration Stream', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
        if (e-s) > 15:
            print('######### MOTION CALIBRATED ##############')
            print(f" MOTION VAL CONSTANT: {MOTION_VAL} ")
            cap.release()
            cv2.destroyAllWindows()
            return 
        
 
if __name__ == "__main__":
    calibration(camera_id=1, queueAddInterval=5)
    runtimeCounter = 0
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    frameQueue.append(frame)
    queueAddInterval = 5
    while True:
        runtimeCounter += 1 
        ret, frame = cap.read()
        if runtimeCounter % queueAddInterval == 0:
            frameQueue.append(frame)
            executor.submit(check_significant_change, frameQueue[-1], frameQueue[-2])
        frame = cv2.putText(frame, f'{CHANGE}' , org, font,  fontScale, color, thickness, cv2.LINE_AA) 
        cv2.imshow('Motion Detection Stream', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

        
        


# import cv2 
  
  
# # define a video capture object 
# vid = cv2.VideoCapture(1) 
  
# while(True): 
      
#     # Capture the video frame 
#     # by frame 
#     ret, frame = vid.read() 
  
#     # Display the resulting frame 
#     cv2.imshow('frame', frame) 
      
#     # the 'q' button is set as the 
#     # quitting button you may use any 
#     # desired button of your choice 
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break
  
# # After the loop release the cap object 
# vid.release() 
# # Destroy all the windows 
# cv2.destroyAllWindows() 















