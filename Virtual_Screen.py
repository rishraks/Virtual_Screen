import cv2
import mediapipe as mp
import numpy as np
import math

# Web Cam Initialization
cap = cv2.VideoCapture(0)

# Hand Tracking
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(max_num_hands = 1, static_image_mode = False, min_detection_confidence = 0.5, min_tracking_confidence = 0.8)

# Index Finger Position
index_new_pos = (0,0)
smooth_pos = (0,0)

# Smoothening Weight
alpha = 0.6

# Initializing Canvas For Paint
canvas = None
w,h = (800,600)
brush_color = (255,255,255)
brush_thickness = 3

# Paint Area Boundary
b_start = (50,50)
b_end = (750,550)

# Function to detect finger extension
def is_extended(landmarks,tip_idx,pip_idx):
    return landmarks[tip_idx].y<landmarks[pip_idx].y

# Function to detect finger close
def is_bent(landmarks,tip_idx,pip_idx):
    return landmarks[tip_idx].y>landmarks[pip_idx].y

# Function to detect Thumb Close
def is_thumb_bent(landmarks, handedness):
    if handedness == "Left":
        return landmarks[4].x < landmarks[3].x and landmarks[4].x < landmarks[2].x
    else:
        return (
            landmarks[4].x > landmarks[3].x
            and landmarks[4].x > landmarks[2].x
            or landmarks[4].y > landmarks[3].y
        )

# Function to detect Thumb extension
def is_thumb_extended(landmarks, handedness):
    if handedness == "Left":
        return landmarks[4].x > landmarks[3].x
    else:
        return landmarks[4].x < landmarks[3].x

selection_mode = False

# Color Coordinates
color_bbox=[{"color_centre":(200,25),"radius":22,"color":(0,0,255)},
            {"color_centre":(280,25),"radius":22,"color":(0,255,0)},
            {"color_centre":(360,25),"radius":22,"color":(255,0,0)},
            {"color_centre":(440,25),"radius":22,"color":(0,255,255)},
            {"color_centre":(520,25),"radius":22,"color":(255,165,0)},
            {"color_centre":(600,25),"radius":22,"color":(255,0,255)},  
            {"color_centre":(680,25),"radius":22,"color":(255,165,0)}]


brush_thickness_box=[{"coord_center":(775,150),"radius":22,"thickness":3,"text":"3","text_coord":(766,159)},
                     {"coord_center":(775,230),"radius":22,"thickness":5,"text":"5","text_coord":(766,239)},
                     {"coord_center":(775,310),"radius":22,"thickness":8,"text":"8","text_coord":(766,319)},
                     {"coord_center":(775,390),"radius":22,"thickness":10,"text":"10","text_coord":(755,399)}]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        frame = cv2.resize(frame,(w,h))
        frame = cv2.flip(frame,1)
        h,w,c = frame.shape
        
        if canvas is None:
            canvas = np.zeros((h,w,3),dtype=np.uint8)
        
        results = hands.process(frame)
        
        # Blur Effect
        area = frame[b_start[1]:b_end[1],b_start[0]:b_end[0]]
        blurred_area = cv2.GaussianBlur(area,(19,19),15)
        frame[b_start[1]:b_end[1], b_start[0]:b_end[0]] = blurred_area        

        cv2.rectangle(frame,b_start,b_end,(128, 128, 128),2)
        
        # Drawing Color Pallet
        for box in color_bbox:
            cv2.circle(frame, box["color_centre"], box["radius"], box["color"],-1)
            cv2.circle(frame,box["color_centre"],box["radius"]+2,(255,255,255),1)

        # Drawing Brush Thickness Pallet 
        for box in brush_thickness_box:
            cv2.circle(frame,box["coord_center"],box["radius"],(0,0,0),-1)
            cv2.circle(frame,box["coord_center"],box["radius"]+2,(255,255,255),1)
            cv2.putText(frame,box["text"],box["text_coord"],cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        
        # Hand Landmarks Detection
        if results.multi_hand_landmarks:
            for handlms,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                
                index_tip = handlms.landmark[8]
                index_pos = int(index_tip.x*w),int(index_tip.y*h)
                    
                middle_tip = handlms.landmark[12]
                middle_pos = int(middle_tip.x*w),int(middle_tip.y*h)
                
                # Detects hand type
                handtype = handedness.classification[0].label
                
                # Condition to check index,middle finger extension Selection Mode
                if is_extended(handlms.landmark,8,7) and is_extended(handlms.landmark,12,11) and is_bent(handlms.landmark,16,15) and is_bent(handlms.landmark,20,19) and is_thumb_bent(handlms.landmark,handtype):
                    cv2.putText(frame,"Select Color",(10,35),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
                    cv2.rectangle(frame,index_pos,middle_pos,(255,255,0),-1)
                    index_new_pos = (0,0)
                    smooth_pos = (0,0)
                    selection_mode = True
                
                # Condition to check Palm for Erase
                elif is_extended(handlms.landmark,8,7) and is_extended(handlms.landmark,12,11) and is_extended(handlms.landmark,16,15) and is_extended(handlms.landmark,20,19) and is_thumb_extended(handlms.landmark,handtype):
                    cv2.putText(frame, "Erasing", (10, 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    palm_center = handlms.landmark[9]
                    palm_pos = int(palm_center.x * w), int(palm_center.y * h)
                    cv2.circle(frame, palm_pos, 50, (0, 0, 255), 2)
                    cv2.circle(canvas, palm_pos, 50, (0, 0, 0), -1)

                # Condition to check index finger extension for paint
                elif is_extended(handlms.landmark,8,7) and is_bent(handlms.landmark,12,11) and is_bent(handlms.landmark,16,15) and is_bent(handlms.landmark,20,19) and is_thumb_bent(handlms.landmark,handtype):
                    if selection_mode:
                        smooth_pos = index_pos
                        index_new_pos = index_pos
                        selection_mode = False
                    
                    smooth_pos = (int(alpha*smooth_pos[0]+(1-alpha)*index_pos[0]),int(alpha*smooth_pos[1]+(1-alpha)*index_pos[1]))
                    cv2.circle(frame, smooth_pos, 8, brush_color, -1)
                    
                    # Condition to check if index and middle fingers are inside the circles
                    for box in color_bbox:
                        region = box["color_centre"]
                        radius = box["radius"]
                        
                        finger_pos = smooth_pos  # Position of the index finger tip
                        distance = math.sqrt((finger_pos[0] - region[0]) ** 2 + (finger_pos[1] - region[1]) ** 2)
                            
                        if distance<= radius:
                            brush_color = box["color"]
                            cv2.circle(frame, smooth_pos, 8, brush_color, -1)
                    
                    
                    for box in brush_thickness_box:
                        region = box["coord_center"]
                        radius = box["radius"]
                        
                        finger_pos = smooth_pos
                        distance = math.sqrt((finger_pos[0]-region[0])**2+ (finger_pos[1]-region[1])**2)
                        
                        if distance <=radius:
                            brush_thickness = box["thickness"]
                    
                    # Condition to check if index finger is inside the painting region boundary
                    if b_start[0]<=smooth_pos[0]<=b_end[0] and b_start[1]<=smooth_pos[1]<=b_end[1]:
                        if index_new_pos != (0, 0) and abs(smooth_pos[0] - index_new_pos[0]) < 30 and abs(smooth_pos[1] - index_new_pos[1]) < 30:
                            cv2.line(canvas,index_new_pos,smooth_pos,brush_color,brush_thickness)
                        
                        index_new_pos = smooth_pos
        
        # Combining frame and canvas     
        combined_frame = cv2.addWeighted(frame,0.5,canvas,0.9,3)
        cv2.imshow("Virtual Screen",combined_frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            canvas = np.zeros((h,w,3),dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()