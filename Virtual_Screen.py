import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hand.Hands(max_num_hands = 1, static_image_mode = False, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)

index_new_pos = (0,0)
smooth_pos = (0,0)

alpha = 0.6

canvas = None
w,h = (800,600)


b_start = (50,50)
b_end = (750,550)


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
        
        
        area = frame[b_start[1]:b_end[1],b_start[0]:b_end[0]]
        blurred_area = cv2.GaussianBlur(area,(19,19),15)
        frame[b_start[1]:b_end[1], b_start[0]:b_end[0]] = blurred_area        

        cv2.rectangle(frame,b_start,b_end,(128, 128, 128),2)
        
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                
                
                if handlms.landmark[8]:
                    index_tip = handlms.landmark[8]
                    index_pos = int(index_tip.x*w),int(index_tip.y*h)
                    
                    smooth_pos = (int(alpha*smooth_pos[0]+(1-alpha)*index_pos[0]),int(alpha*smooth_pos[1]+(1-alpha)*index_pos[1]))
                    
                    cv2.circle(frame, smooth_pos, 8, (255,0,255), -1)
                    
                    if b_start[0]<=smooth_pos[0]<=b_end[0] and b_start[1]<=smooth_pos[1]<=b_end[1]:
                        if index_new_pos != (0, 0) and abs(smooth_pos[0] - index_new_pos[0]) < 50 and abs(smooth_pos[1] - index_new_pos[1]) < 50:
                            cv2.line(canvas,index_new_pos,smooth_pos,(0,255,255),3)
                        
                        index_new_pos = smooth_pos
                
        combined_frame = cv2.addWeighted(frame,0.5,canvas,0.9,3)
        cv2.imshow("Virtual Screen",combined_frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            canvas = np.zeros((h,w,3),dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()