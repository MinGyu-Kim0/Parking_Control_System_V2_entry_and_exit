from ultralytics import YOLO
import cv2
# import RPi.GPIO as GPIO

# GPIO and pin number setting
# GPIO.setmode(GPIO.BCM)

# servo_pin = 18
# GPIO.setup(servo_pin, GPIO.OUT)

# 객체 인식 모델 설정
model = YOLO("best.pt")

# videocapture
capture = cv2.VideoCapture(0)

# ROI 좌표 설정
R_X1 = 200
R_Y1 = 200
R_X2 = 600
R_Y2 = 400

# 중심점 좌표 설정
C_X = (R_X1+R_X2) // 2
C_Y = (R_Y1+R_Y2) // 2

# main 반복문 실행
if capture.isOpened():
    while True:

# 프레임 가져오기
        ret, frame = capture.read() 
        roi = frame[R_Y1:R_Y2,R_X1:R_X2] 
        
# 모델에 프레임 넘겨주기
        results = model(roi)

# 이미지 범위를 가시적으로 출력
        cv2.rectangle(frame, (R_X1, R_Y1), (R_X2, R_Y2), (0, 255, 0), 2)
        cv2.circle(frame, (C_X,C_Y), 3, (0, 0, 255), 3)
        
# 모델이 추론한 값 가져오기
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results[0].names.get(cls_id, str(cls_id))
            o_x, o_y, o_x2, o_y2 = map(int, xyxy)
            
# webcam 화면 크기 설정
            X1, Y1 = o_x + R_X1, o_y + R_Y1
            X2, Y2 = o_x2 + R_X1, o_y2 + R_Y1
            
# 신뢰도를 우선 판단하고 인식한 객체 박스안에 중심점이 들어오면 가시적으로 출력
            if conf >= 0.5:    
                if X1 <= C_X <= X2 and Y1 <= C_Y <= Y2:
                    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                    cv2.putText(frame, f"{label} {conf:.2f}", (X1+10, Y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
# 이미지 띄우기
        cv2.imshow("YOLOv8 Detection", frame)

# break 설정
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 화면 닫기
capture.release()
cv2.destroyAllWindows()

