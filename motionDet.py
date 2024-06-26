from image_inference import *
import numpy as np
import cv2
import datetime
import uuid
# from ultralytics import YOLO
from maskHelper import maskImage
# Initialize video capture
cap = cv2.VideoCapture(0)  # Change the index (0, 1, 2, etc.) based on your camera

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
statusd = "Invalid Motion"

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

# Load YOLO model
# model_obj = YOLO('/home/onm/Downloads/onm_rnd_ocr/best_34.pt')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(100, 100),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if ret is False:
    print("Error reading frame")
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

cv2.imwrite('live_frame.png', old_frame)
s = time.time()
pil_mask = inference(old_frame)
e = time.time()
print(f'{e-s} inferencing on one frame')
cv2.imwrite('live_mask.png', pil_mask)
try:
    polygon_coords = masker.maskProcessor(pil_mask, 20)
    polygon_coords = np.array(polygon_coords[0])
    num_points = 30
    random_points = generate_points_inside_polygon(polygon_coords, num_points)
    p0 = np.array([[list(x)] for x in random_points]).astype(np.float32)

except Exception as e:
    p0 = np.array([[100,200], [200,200]]).astype(np.float32) # Declaring dummy points 
    print(e)

mask = np.zeros_like(old_frame)





# Background subtraction using MOG2
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

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
# p0 = None 
num_points = 30
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error in opening the camera")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raw_frame = frame.copy()

    # Background subtraction
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_detected_points = []

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            all_detected_points.extend(contour.reshape(-1, 2))

    if all_detected_points:
        all_detected_points = np.array(all_detected_points)
        hull = cv2.convexHull(all_detected_points)
        coords = find_convex_hull_center(hull)
        if coords is not None:
            prev_status = current_status
            current_status = True
            cx, cy = coords
            cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), radius=15, color=(0, 0, 255), thickness=-1)
    else:
        if current_status == True:
            prev_status = current_status
            current_status = False
        else:
            prev_status = current_status

    motion_plot = frame.copy()
    frame = raw_frame.copy()

    if reset_key_pressed or (prev_status == False and current_status == True) or statusd == "Grab" or statusd == "Return":
        s = time.time()
        pil_mask = inference(frame)
        # cv2.imshow('MASK AFTER RESET', pil_mask)
        e = time.time()
        print(f'{e-s} for inferencing one frame')
        try:
            polygon_coords = masker.maskProcessor(pil_mask, 20)
            polygon_coords = np.array(polygon_coords[0])
            print(polygon_coords)
            random_points = generate_points_inside_polygon(polygon_coords, num_points)
            p0 = np.array([[list(x)] for x in random_points]).astype(np.float32)
        except Exception as e:
            p0 = None 
            print(f'Exception due to no points {e}')

        reset_key_pressed = False
        mask = np.zeros_like(frame)
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_keys = "None"

    # Calculate optical flow

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    try:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    except:
        # cv2.imshow('frame', img)
        continue

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

    img = cv2.add(frame, mask)
    try:
        img = cv2.line(img, [round(curr_keys[0]), round(curr_keys[1])], [round(prev_keys[0]), round(prev_keys[1])], (255, 125, 255), 19)
    except ValueError as e:
        print(f"Error in drawing line: {e}")
        continue
    # img = cv2.line(img, [round(curr_keys[0]), round(curr_keys[1])], [round(prev_keys[0]), round(prev_keys[1])], (255, 125, 255), 19)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Current Keys: {curr_keys}', (50, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f'Previous Keys: {prev_keys}', (50, 150), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
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
    # if prev_keys[1] > curr_keys[1]:
    #     if abs(prev_keys[1] - curr_keys[1]) > 15:
    #         statusd = "Return"
    #     else:
    #         statusd = "Invalid Motion"

    # if prev_keys[1] < curr_keys[1]:
    #     if abs(prev_keys[1] - curr_keys[1]) > 15:
    #         statusd = "Grab"
    #     else:
            # statusd = "Invalid Motion"
    img = cv2.putText(img, statusd, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    out.write(img)
    cv2.imshow('Tracker Frame', img)
    cv2.imshow("Motion Frame", motion_plot)
    cv2.imshow('Live Mask', pil_mask)

    print("=" * 10)
    print("Previous Status : ", prev_status)
    print("Current Status : ", current_status)

    # Perform action based on motion status change
    # if prev_status == True and current_status == False and statusd != "Invalid Motion":
    #     final_status = "Moved In" if statusd == "Return" else "Moved Out"
    #     payload = {
    #         "Final_Motion_Status": final_status,
    #         "Crossed_In_Status": True,
    #         "Crossed_Out_Status": True,
    #         "Compartment_Interacted": 1,
    #         "Final_Intent_Status": final_status,
    #         "Item_Name": "Merit MAK Guide Wire",
    #         "compartmentID": "NA",
    #         "ReferenceID": "REF MAK001N",
    #         "session_start_time": str(datetime.datetime.now()),
    #         "session_end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=5)),
    #     }
    #     camera_id = "ad7026eda1074d33ae17b4b621ac6af8(usb-0000:00:14.0-4)"
    #     activityId = str(uuid.uuid4())
    #     # create_UI_Intent(payload, raw_frame, camera_id, activityId)

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