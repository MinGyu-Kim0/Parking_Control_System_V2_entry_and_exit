from ultralytics import YOLO
import cv2
import numpy as np

import easyocr
import threading
import time

# import RPi.GPIO as GPIO
switching = False
detect = False

# def servo():
#     global switching, detect
# # GPIO and pin number setting
#     GPIO.setmode(GPIO.BCM)
#     servo_pin = 18
#     GPIO.setup(servo_pin, GPIO.OUT)

#     pwm = GPIO.PWM(servo_pin, 50)
#     pwm.start(0)
#     try:
#         while True:
#             if switching == False and detect == True:
#                 pwm.ChangeDutyCycle(7.5)
#             else:
#                 pwm.ChangeDutyCycle(2.5)
#             time.sleep(0.1)
#     finally:
#         pwm.stop()
#         GPIO.cleanup()

# videocapture and easyocr setting
capture = cv2.VideoCapture(0)
reader = easyocr.Reader(['ko'], gpu=True)

# 객체 인식 모델 설정
model = YOLO("Parking_Control_System_V2/best.pt")

# ROI 좌표 설정
R_X1 = 200
R_Y1 = 200
R_X2 = 600
R_Y2 = 400

# 중심점 좌표 설정
C_X = (R_X1+R_X2) // 2
C_Y = (R_Y1+R_Y2) // 2

# main 반복문 실행
def main(): 
    try:
        global detect
   
        if capture.isOpened():
            while True:
            
            # 프레임 가져오기
                ret, frame = capture.read()
                roi = frame[R_Y1:R_Y2, R_X1:R_X2]

            # 모델에 프레임 넘겨주기
                results = model(roi)

            # 이미지 범위를 가시적으로 출력
                cv2.rectangle(frame, (R_X1, R_Y1), (R_X2, R_Y2), (0, 255, 0), 2)
                cv2.circle(frame, (C_X, C_Y), 2, (0, 0, 255), 1)

            # 모델이 추론한 값 가져오기
                boxes = results[0].boxes
                detect = False
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                # label = results[0].names.get(cls_id, str(cls_id))
                    o_x, o_y, o_x2, o_y2 = map(int, xyxy)

                # webcam 화면 크기 설정            
                    X1, Y1 = o_x + R_X1, o_y + R_Y1
                    X2, Y2 = o_x2 + R_X1, o_y2 + R_Y1

                    if conf >= 0.5:
                        if X1 <= C_X <= X2 and Y1 <= C_Y <= Y2:
                            detect = True
                            ocr = reader.readtext(frame[Y1:Y2, X1:X2])
                            for (bbox, text, prob) in ocr:
                                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                                cv2.putText(frame, text, (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                print(detect)
                        else:
                            detect = False
                            print(detect)
# 이미지 띄우기
                cv2.imshow("YOLOv8 Detection", frame)
# 반복문 종료
                if cv2.waitKey(1) & 0xFF ==  ord('q'):
                    break
# 화면 닫기                    
    finally:
        capture.release()
        cv2.destroyAllWindows()


# thread setting
t1 = threading.Thread(target=main)
# t2 = threading.Thread(target=servo)

t1.start()
# t2.start()

t1.join()
# t2.join()




