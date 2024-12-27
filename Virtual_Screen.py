import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hand.Hands(max_num_hands = 1, static_image_mode = False, min_detection_confidence = 0.6, min_tracking_confidence = 0.5)

index_new_pos = (0,0)

canvas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        
        frame = cv2.flip(frame,1)
        h,w,c = frame.shape
        
        if canvas is None:
            canvas = np.zeros((h,w,3),dtype=np.uint8)
        
        results = hands.process(frame)
        
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                
                if handlms.landmark[8]:
                    index_tip = handlms.landmark[8]
                    index_pos = int(index_tip.x*w),int(index_tip.y*h)
                    cv2.circle(frame, index_pos,8,(255,0,255),-1)
                    
                    if index_new_pos!=(0,0):
                        cv2.line(canvas,index_new_pos,index_pos,(0,255,255),3)
                    
                    index_new_pos = index_pos
                
        combined_frame = cv2.addWeighted(frame,0.5,canvas,0.9,3)
        cv2.imshow("Virtual Screen",combined_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()