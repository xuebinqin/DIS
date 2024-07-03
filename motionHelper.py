
import cv2 
from skimage.metrics import structural_similarity as ssim
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging 
import time 


class MotionDetection:
    def __init__(self, queueLen = 10, max_workers = 5):
        self.motionValue = 1000 
        self.motion = False 
        self.frameQueue = deque(queueLen)
        self.executor = ThreadPoolExecutor(max_workers)
        self.threshold = 0.93
    

    def motionDetect(self,frame1, frame2, log = False):
        try:
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            ssim_index, _ = ssim(gray_frame1, gray_frame2, full=True)
            self.motionValue = ssim_index
            if log:
                logging.info(f"Similarity Index : {ssim_index}")
            if ssim_index < self.threshold:
                self.motion = True 
            else:
                self.motion = False 
        except Exception as e:
            logging.info(f"Code failed due to:\n {e}")
            raise e 
        
    def motionUpdate(self, frame):
        self.frameQueue.append(frame)
        self.executor.submit(self.motionDetect, self.frameQueue[-1], self.frameQueue[-2])
        
        
    def motionCalibration(self,frame):
        print('########### Don\'t perform any motion in the rack ###################')
        self.motionUpdate(frame)
        if self.motionValue < self.threshold:
            self.threshold = self.motionValue 
    
    


        



   
