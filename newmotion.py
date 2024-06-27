from inf import *
import numpy as np
import cv2
import datetime
import uuid
# from ultralytics import YOLO
from mask import maskImage
# from img_similarity import check_significant_change
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
import logging 
# Initialize video capture



cap = cv2.VideoCapture(0)  # Change the index (0, 1, 2, etc.) based on your camera
frameQueue = deque(maxlen=10)
executor = ThreadPoolExecutor(max_workers=5)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
statusd = "Invalid Motion"
frameQueue = deque(maxlen=10)
executor = ThreadPoolExecutor(max_workers=5)
global CHANGE
CHANGE = False
# # from Config import CHANGE
# rack_background_polygon_right = [[400, 469], [379,178], [422,479], [640,93], [640,266]]
# rack_background_polygon_left = [[42,454], [0,412], [0,80], [163,40], [180,170]]
main_fov_polygon = [[24,419], [176,81], [390,173], [415,458]]
def check_significant_change(frame1, frame2, threshold=0.97):
    try:
        global CHANGE 
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between two frames
        ssim_index, _ = ssim(gray_frame1, gray_frame2, full=True)
        print(ssim_index)
        # if ssim_index < 0.99:
            # obj = BackgroundSubtraction()
        # Check if the SSIM index is below the threshold
        if ssim_index < threshold:
            CHANGE = True 
        else:
            CHANGE = False 
    except Exception as e:
        logging.info(f"Code failed due to:\n {e}")
        raise e 


# Initialize VideoWriter


masker = maskImage()
out = cv2.VideoWriter('/home/onm/Downloads/onm_rnd_ocr/output_poc_opticalflow/stack_5.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

# Function to generate random points inside a polygon
def generate_points_inside_polygon(polygon_coords, num_points):
    min_x, min_y = np.min(polygon_coords, axis=0)
    max_x, max_y = np.max(polygon_coords, axis=0)

    random_points = []

    while len(random_points) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        if is_point_inside_polygon(x, y, polygon_coords):
            random_points.append((x, y))

    return random_points

# Function to check if a point is inside a polygon
def is_point_inside_polygon(x, y, polygon_coords):
    n = len(polygon_coords)
    inside = False

    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# def check_all_points(points_list, polygon):
#     check = False 
#     for point in points_list:
#         if is_point_inside_polygon(point[0], point[1], polygon):
#             continue
#         else:
#             return False 
#     return True  


# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(100, 100),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if old_frame is None:
    print("Error reading frame")
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite('live_frame.png', old_frame)
s = time.time()
pil_mask = inference(old_frame)
e = time.time()
print(f'{e-s} inferencing on one frame')
cv2.imwrite('live_mask.png', pil_mask)
print(pil_mask.shape)

p0 = None
polygon_coords, area = masker.maskProcessor(pil_mask, 20)
# print(polygon_coords)
if polygon_coords is None:
    p0 = None
else:
    polygon_coords = np.array(polygon_coords[0])
    num_points = 30
    random_points = generate_points_inside_polygon(polygon_coords, num_points)
    p0 = np.array([[list(x)] for x in random_points]).astype(np.float32)


mask = np.zeros_like(old_frame)


# Background subtraction using MOG2
# bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Function to find convex hull center
def find_convex_hull_center(contour):
    hull = cv2.convexHull(contour)
    M = cv2.moments(hull)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    else:
        return None

# Initialize variables for status tracking
prev_status, current_status = False, False
prev_keys = "None"
reset_key_pressed = False
num_points = 30
runtimeCounter=0
queueAddInterval = 5
font = cv2.FONT_HERSHEY_SIMPLEX
# emptyFill_current = False 
# emptyFill_prev = False
frameQueue.append(old_frame)
emptyFill = False 
initCheck = 0
while True:
    prev_status = current_status

    prevmask = pil_mask
    runtimeCounter +=1
    ret, frame = cap.read()
    if frame is None:
        print("Error in opening the camera")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raw_frame = frame.copy()
    


    
    
    motion_plot = frame.copy()
    frame = raw_frame.copy()
    print("Status Check : ",  prev_status, current_status)

    if runtimeCounter % queueAddInterval == 0:
        frameQueue.append(frame)
        executor.submit(check_significant_change, frameQueue[-1], frameQueue[-2])
        current_status = CHANGE

    emptyFill = False 
    if reset_key_pressed or (prev_status == True and current_status == False):
        print("======================== Inside reset ==============================")
        s = time.time()
        pil_mask = inference(frame)
        e = time.time()
        print(f'{e-s} for inferencing one frame')
        polygon_coords, curr_area = masker.maskProcessor(pil_mask, 20)
        old_coords, prev_area = masker.maskProcessor(prevmask, 20)
        
        if polygon_coords is None:
            p0 = None
        else:
            polygon_coords = np.array(polygon_coords[0])
            print(polygon_coords)
            random_points = generate_points_inside_polygon(polygon_coords, num_points)
            # if check_all_points(polygon_coords, rack_background_polygon_left) is False and \
            # check_all_points(polygon_coords, rack_background_polygon_right) is False:
            #     statusd = "Return"
            #     emptyFill_current = True
            # cv2.imwrite('after_motion.png', frame)
            temp_save_frame = frame.copy()
            for point in polygon_coords:
                temp_save_frame = cv2.circle(temp_save_frame, (point[0], point[1]), 2, (0,255,120), 2)
            cv2.imwrite('after_motion.png', temp_save_frame)
            if is_point_inside_polygon(polygon_coords[0][0], polygon_coords[0][1], main_fov_polygon) is True:
                emptyFill = True 
                print(f'EMPTY FILL : {emptyFill}')
                # statusd = "Return"
                initCheck +=1
            p0 = np.array([[list(x)] for x in random_points]).astype(np.float32)
            reset_key_pressed = False
            mask = np.zeros_like(frame)
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_keys = "None"


    # cv2.imshow('Live Stream', frame)

    if p0 is None:
        # cv2.imshow('Tracker Frame', img)
        cv2.imshow("Motion Frame", motion_plot)
        cv2.imshow('Live Mask', pil_mask)
        k = cv2.waitKey(25)
        if k == 27:
            break
        elif k == ord('r') or k == ord('R'):
            # Set reset_key_pressed when 'r' or 'R' is pressed
            reset_key_pressed = True

        continue

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    try:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    except Exception as e:
        print(f'Error at good_new 239 {e}')
        # continue
    if prev_keys=="None":
        prev_keys = [np.mean([x[0] for x in good_new]), np.mean([x[1] for x in good_new])]
    curr_keys = [np.mean([x[0] for x in good_new]), np.mean([x[1] for x in good_new])]    
    print("New : ", curr_keys)
    print("Old : ", prev_keys)
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = round(a), round(b), round(c), round(d)
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

         
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    print(frame.shape)
    print(mask.shape)
    img = cv2.add(frame, mask)
    try:
        img = cv2.line(img, [round(curr_keys[0]), round(curr_keys[1])], [round(prev_keys[0]), round(prev_keys[1])], (255, 125, 255), 19)
        cv2.putText(img, f'Current Keys: {curr_keys}', (50, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'Previous Keys: {prev_keys}', (50, 150), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except ValueError as e:
        print(f"Error in drawing line: {e}")
        # cv2.imshow('Tracker Frame', img)
        # cv2.imshow("Motion Frame", motion_plot)
        # cv2.imshow('Live Mask', pil_mask)
        # continue
    # img = cv2.line(img, [round(curr_keys[0]), round(curr_keys[1])], [round(prev_keys[0]), round(prev_keys[1])], (255, 125, 255), 19)
    
    # -------- hanging -------------------------
    # if prev_keys[1] > curr_keys[1]:
    #     if abs(prev_keys[1] - curr_keys[1]) > 3:
    #         statusd = "Return"
    #     else:
    #         statusd = "Invalid Motion"

    # if prev_keys[1] < curr_keys[1]:
    #     if abs(prev_keys[1] - curr_keys[1]) > 3:
    #         statusd = "Grab"
    #     else:
    #         statusd = "Invalid Motion"
    # -------------- stacked ---------------------
    # print('REACHING HERE')
    if emptyFill is True or initCheck >0:
        print(f"{emptyFill=} status")
        statusd = "Return"
        cv2.imwrite(f'{statusd}.jpg', frame)
        initCheck +=1
        if initCheck == 10:
            initCheck = 0
    else:
        if prev_keys[1] > curr_keys[1]:
            if abs(prev_keys[1] - curr_keys[1]) > 15 :
                statusd = "Return"
                # print(f"{emptyFill=} status")
            else:
                statusd = "Invalid Motion"

        if prev_keys[1] < curr_keys[1]:
            if abs(prev_keys[1] - curr_keys[1]) > 15:
                statusd = "Grab"
            else:
                statusd = "Invalid Motion"
    img = cv2.putText(img, statusd, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    out.write(img)
    cv2.imshow('Tracker Frame', img)
    cv2.imshow("Motion Frame", motion_plot)
    cv2.imshow('Live Mask', pil_mask)

    print("=" * 10)
    print("Previous Status : ", prev_status)
    print("Current Status : ", current_status)

    if prev_status == True and current_status == False and statusd != "Invalid Motion":
        final_status = "Moved In" if statusd == "Return" else "Moved Out"
        payload = {
            "Final_Motion_Status": final_status,
            "Crossed_In_Status": True,
            "Crossed_Out_Status": True,
            "Compartment_Interacted": 1,
            "Final_Intent_Status": final_status,
            "Item_Name": "Merit MAK Guide Wire",
            "compartmentID": "NA",
            "ReferenceID": "REF MAK001N",
            "session_start_time": str(datetime.datetime.now()),
            "session_end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=5)),
        }
        camera_id = "ad7026eda1074d33ae17b4b621ac6af8(usb-0000:00:14.0-4)"
        activityId = str(uuid.uuid4())
        # create_UI_Intent(payload, raw_frame, camera_id, activityId)
    k = cv2.waitKey(25)
    if k == 27:
        break
    elif k == ord('r') or k == ord('R'):
        # Set reset_key_pressed when 'r' or 'R' is pressed
        reset_key_pressed = True

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release VideoWriter and cleanup
out.release()
cv2.destroyAllWindows()
cap.release()

