import cv2
import mediapipe as mp
import numpy as np


# Web Cam Initialization
cap = cv2.VideoCapture(0)

# Hand Tracking
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(max_num_hands = 1, static_image_mode = False, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)

# Index Finger Position
index_new_pos = (0,0)
smooth_pos = (0,0)

# Smoothening Weight
alpha = 0.6

# Initializing Canvas For Paint
canvas = None
w,h = (800,600)
brush_color = (255,255,255)

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
        
        
        if results.multi_hand_landmarks:
            for handlms,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                
                # Detects hand type
                handtype = handedness.classification[0].label
                
                # Condition to check index,middle finger extension Selection Mode
                if is_extended(handlms.landmark,8,7) and is_extended(handlms.landmark,12,11) and is_bent(handlms.landmark,16,15) and is_bent(handlms.landmark,20,19) and is_thumb_bent(handlms.landmark,handtype):
                    cv2.putText(frame,"Selection mode",(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    index_new_pos = (0,0)
                    smooth_pos = (0,0)
                    selection_mode = True
                
                # Condition to check Palm for Erase
                elif is_extended(handlms.landmark,8,7) and is_extended(handlms.landmark,12,11) and is_extended(handlms.landmark,16,15) and is_extended(handlms.landmark,20,19) and is_thumb_extended(handlms.landmark,handtype):
                    cv2.putText(frame, "Erasing", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    palm_center = handlms.landmark[9]
                    palm_pos = int(palm_center.x * w), int(palm_center.y * h)
                    cv2.circle(frame, palm_pos, 50, (0, 0, 255), 2)
                    cv2.circle(canvas, palm_pos, 50, (0, 0, 0), -1)

                # Condition to check index finger extension for paint
                elif is_extended(handlms.landmark,8,7) and is_bent(handlms.landmark,12,11) and is_bent(handlms.landmark,16,15) and is_bent(handlms.landmark,20,19) and is_thumb_bent(handlms.landmark,handtype):
                    index_tip = handlms.landmark[8]
                    index_pos = int(index_tip.x*w),int(index_tip.y*h)
                    
                    if selection_mode:
                        smooth_pos = index_pos
                        index_new_pos = index_pos
                        selection_mode = False
                    
                    smooth_pos = (int(alpha*smooth_pos[0]+(1-alpha)*index_pos[0]),int(alpha*smooth_pos[1]+(1-alpha)*index_pos[1]))
                    
                    cv2.circle(frame, smooth_pos, 8, (255,0,255), -1)
                    
                    if b_start[0]<=smooth_pos[0]<=b_end[0] and b_start[1]<=smooth_pos[1]<=b_end[1]:
                        if index_new_pos != (0, 0) and abs(smooth_pos[0] - index_new_pos[0]) < 30 and abs(smooth_pos[1] - index_new_pos[1]) < 30:
                            cv2.line(canvas,index_new_pos,smooth_pos,brush_color,3)
                        
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