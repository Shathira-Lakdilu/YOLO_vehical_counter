import cv2
import cvzone
import torch
from ultralytics import YOLO
import math
from tracker import *

# Path to the video file
video_path = 'D:/Image_processing/Vehical_counting/images/traffic.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# cap.set(3, 1280) #3 for width
# cap.set(4, 720)  #4 for height

model = YOLO('yolo_weights/yolov8l.pt')
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
mask1 = cv2.imread("D:/Image_processing/Vehical_counting/mask1.png")
mask2 = cv2.imread("D:/Image_processing/Vehical_counting/mask2.png")
mask1 = cv2.resize(mask1, (1280, 720))
mask2 = cv2.resize(mask2, (1280, 720))
total_mask = cv2.bitwise_or(mask1, mask2, mask2)
mask2 = cv2.imread("D:/Image_processing/Vehical_counting/mask2.png")
# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

glimits = [30, 525, 450, 600]  # limits for the line green
blimits = [500, 525, 1200, 600]  # limits for the line blue
btotalCount = []
gtotalCount = []

mask1_gray = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
total_mask_gray = cv2.bitwise_or(mask1_gray, mask2_gray, mask2_gray)
# Define the color to fill (BGR format)
mask1_color = (0, 255, 0)  # Green
mask2_color = (255, 0, 0)  # Blue
# Alpha value for transparency (0.0 fully transparent to 1.0 fully opaque)
alpha = 0.25


def to_numpy(tensor_or_scalar):
    if isinstance(tensor_or_scalar, torch.Tensor):
        return tensor_or_scalar.cpu().numpy()
    else:
        return tensor_or_scalar


# Ensure the mask is a 3-channel image
# mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
while True:
    # print("loop started ........")
    success, frame = cap.read()
    imgRegion1 = cv2.bitwise_and(frame, mask1)
    imgRegion2 = cv2.bitwise_and(frame, mask2)
    imgRegion = cv2.bitwise_and(frame, total_mask)

    # Create an image filled with the desired color
    colored_region1 = np.full_like(frame, mask1_color)
    colored_region2 = np.full_like(frame, mask2_color)

    # Apply the mask to the colored region
    colored_mask1 = cv2.bitwise_and(colored_region1, mask1)
    colored_mask2 = cv2.bitwise_and(colored_region2, mask2)
    colored_mask = cv2.bitwise_or(colored_mask1, colored_mask2)
    # colored_mask1 = cv2.cvtColor(colored_mask1, cv2.COLOR_BGR2BGR)

    # Invert the binary mask
    mask1_inv = cv2.bitwise_not(mask1_gray)
    mask2_inv = cv2.bitwise_not(mask2_gray)
    bg_mask_inv = cv2.bitwise_not(total_mask_gray)
    # Apply the inverse mask to the original image to get the background
    background1 = cv2.bitwise_and(frame, frame, mask=mask1_inv)
    background2 = cv2.bitwise_and(frame, frame, mask=mask2_inv)
    background = cv2.bitwise_and(frame, frame, mask=bg_mask_inv)

    # Blend the colored mask with the original image using alpha blending
    # blended_region1 = cv2.addWeighted(colored_mask1, alpha, frame, 1 - alpha, 0)
    blended_region2 = cv2.addWeighted(colored_mask2, alpha, frame, 1 - alpha, 0)
    blended_region1 = cv2.addWeighted(colored_mask, alpha, frame, 1 - alpha, 0)

    # Combine the blended region with the background
    new_frame1 = cv2.add(blended_region1, background)
    # new_frame1 = cv2.add(new_frame1, blended_region1)

    mask1_inv_3ch = cv2.cvtColor(mask1_inv, cv2.COLOR_GRAY2BGR)
    mask2_inv_3ch = cv2.cvtColor(mask2_inv, cv2.COLOR_GRAY2BGR)
    mask_inv_3ch = cv2.cvtColor(bg_mask_inv, cv2.COLOR_GRAY2BGR)

    new_frame2 = cv2.bitwise_and(frame, mask2_inv_3ch) + cv2.bitwise_and(blended_region2, mask2)
    new_frame1 = cv2.bitwise_and(frame, mask1_inv_3ch) + cv2.bitwise_and(blended_region1, mask1)
    new_frame = cv2.bitwise_and(frame, mask_inv_3ch) + cv2.bitwise_and(blended_region1, total_mask)

    # Blend the colored masks with the original image using alpha blending
    # If frame read was unsuccessful, break the loop
    if not success:
        print("End of video file or error reading frame.")
        break

    # results1 = model(imgRegion1, stream=True)
    results1 = model(imgRegion1, stream=True)
    results2 = model(imgRegion2, stream=True)

    g_detections = np.empty((0, 5))
    # b_detections = np.empty((0, 5))

    for r1 in results1:

        gboxes = r1.boxes
        for box in gboxes:
            gx1, gy1, gx2, gy2 = box.xyxy[0]
            gx1, gy1, gx2, gy2 = int(gx1), int(gy1), int(gx2), int(gy2)
            gw, gh = gx2 - gx1, gy2 - gy1

            gw, gh = int(gw), int(gh)
            # print(gx1, gy1, gx2, gy2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentcls = class_names[cls]

            if currentcls == "car" or currentcls == "bus" or currentcls == "truck" or currentcls == "motorcycle" and conf > 0.3:
                cvzone.putTextRect(new_frame, f'{currentcls}', (gx1 + 2, gy1 - 4), scale=1, thickness=2, offset=3)
                cvzone.putTextRect(new_frame, f'{currentcls}', (gx1 + 2, gy1 - 4), scale=1, thickness=2, offset=3)
                cvzone.cornerRect(new_frame, (gx1, gy1, gw, gh), l=10, t=2)
                cvzone.cornerRect(new_frame, (gx1, gy1, gw, gh), l=10, t=2)
                # print(x1, y1, w, h)

                # currentArray = np.array((x1, y1, x2, y2, conf))
                # Move the tensor to CPU and then convert to NumPy array
                g_currentArray = np.array([to_numpy(gx1), to_numpy(gy1), to_numpy(gx2), to_numpy(gy2), to_numpy(conf)])
                g_detections = np.vstack((g_detections, g_currentArray))
                # print(g_detections)
                # print("..............")

    # print("g_detections")
    # print(g_detections)
    g_results_tracker = tracker.update(g_detections)
    # print("g_results_tracker")
    # print(g_results_tracker)
    cv2.line(new_frame, (glimits[0], glimits[1]), (glimits[2], glimits[3]), (0, 0, 255), 5)

    for g_result in g_results_tracker:
        # print("looop")
        gx1, gy1, gx2, gy2, gid = g_result
        gx1, gy1, gx2, gy2 = int(gx1), int(gy1), int(gx2), int(gy2)
        # print(gx1, gy1, gx2, gy2)
        gw, gh = gx2 - gx1, gy2 - gy1
        # cvzone.cornerRect(new_frame, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(new_frame, f'{int(gid)}', (gx1 + 40, gy1 - 4), scale=1, thickness=2, colorT=(255, 0, 0),
                           colorR=(255, 255, 255), offset=3)

        gcx, gcy = gx1 + gw // 2, gy1 + gh // 2
        cv2.circle(new_frame, (int(gcx), int(gcy)), 2, (255, 0, 0), cv2.FILLED)

        if (glimits[0] < gcx < glimits[2]) and (glimits[1] - 10 < gcy < glimits[1] + 10):
            if gtotalCount.count(gid) == 0:
                gtotalCount.append(gid)
                cv2.line(new_frame, (glimits[0], glimits[1]), (glimits[2], glimits[3]), (0, 255, 0), 5)

    for r2 in results2:
        bboxes = r2.boxes
        for box in bboxes:
            bx1, by1, bx2, by2 = box.xyxy[0]
            # x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            bw, bh = bx2 - bx1, by2 - by1
            # print(x1,y1,x2,y2)
            bx1, by1, bw, bh = int(bx1), int(by1), int(bw), int(bh)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentcls = class_names[cls]

            if currentcls == "car" or currentcls == "bus" or currentcls == "truck" or currentcls == "motorcycle" and conf > 0.3:
                cvzone.putTextRect(new_frame, f'{currentcls}', (bx1 + 2, by1 - 4), scale=1, thickness=2, offset=3)
                cvzone.putTextRect(new_frame, f'{currentcls}', (bx1 + 2, by1 - 4), scale=1, thickness=2, offset=3)
                cvzone.cornerRect(new_frame, (bx1, by1, bw, bh), l=10, t=2)
                cvzone.cornerRect(new_frame, (bx1, by1, bw, bh), l=10, t=2)

    cvzone.putTextRect(new_frame, f'count: {len(gtotalCount)}', (50, 50))
    # cvzone.putTextRect(new_frame, f'count: {len(btotalCount)}', (300, 50))
    cv2.imshow("video", new_frame)
    # cv2.imshow("video1", imgRegion2)
    # cv2.imshow("video2", mask2)
    # cv2.imshow("video2", frame)
    # cv2.waitKey(0)
    # print("end .........")

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
