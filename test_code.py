# ==== 라이브러리 임포트 ====
from ultralytics import YOLO
import cv2, numpy as np, easyocr, threading, time
# import Jetson.GPIO as GPIO X
import socketio # Flask-SocketIO가 아닌, 클라이언트용 socketio 라이브러리

# ==== JetPack 6 PWM 서보 제어 (Jetson.GPIO 대체) ====
PWM_CHIP = "/sys/class/pwm/pwmchip3"
PWM_CH = "0"

def pwm_force_release():
    try:
        with open(f"{PWM_CHIP}/unexport", "w") as f:
            f.write(PWM_CH)
        time.sleep(0.1)
    except:
        pass
def pwm_write(path, value):
    try:
        with open(path, "w") as f:
            f.write(str(value))
    except Exception as e:
        print(f"[PWM ERROR] {e}")

def pwm_init():
    try:
        pwm_write(f"{PWM_CHIP}/export", PWM_CH)
    except:
        pass  # 이미 export 된 경우 무시
    time.sleep(0.1)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "0")
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/period", "20000000")  # 20ms(50Hz)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/duty_cycle", "0")
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "1")

def set_servo_angle(angle):
    # 0~180도 → 1ms~2ms 맵핑
    duty_ns = 1000000 + (angle / 180) * 1000000
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/duty_cycle", int(duty_ns))
    time.sleep(0.15)

def pwm_cleanup():
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "0")

# ==== Socket.IO 클라이언트 설정 ====
sio = socketio.Client()

# ==== 전역 상태 ====
detect = False
running = True

# ==== 특수문자 지정 ====
special_chars = "!{()},'`.^ "
translation_table = str.maketrans('', '', special_chars)


# ==== 서보 각도 -> 듀티 변환 (기존 코드와 동일) ====
def angle_to_duty(angle: float) -> float:
    return 2.5 + (angle * 10.0 / 180.0)

OPEN_ANGLE = 90
CLOSE_ANGLE = 0

# ==== 서보 스레드 (기존 코드와 동일) ====
def servo():
    pwm_force_release()
    pwm_init()
    global detect, running
    pwm_init()
    last_angle = None
    try:
        while running:
            angle = OPEN_ANGLE if detect else CLOSE_ANGLE
            if angle != last_angle:
                set_servo_angle(angle)
                last_angle = angle
            time.sleep(0.05)
    finally:
        pwm_cleanup()
# ==== 모델/캠 (기존 코드와 동일) ====
model = YOLO("best.pt")
capture = cv2.VideoCapture(0)
reader = easyocr.Reader(['ko'], gpu=True)

# ROI / 중심
R_X1, R_Y1, R_X2, R_Y2 = 200, 200, 600, 400
C_X, C_Y = (R_X1 + R_X2) // 2, (R_Y1 + R_Y2) // 2


# ==== 메인 로직 (YOLO, OCR) ====
# rpi_client.py 파일의 main_logic 함수 내부만 수정

# rpi_client.py 파일의 main_logic 함수 내부만 수정

def main_logic():
    global detect, running

    if not capture.isOpened():
        print("카메라를 열 수 없습니다.")
        running = False
        return

    open_count, close_count = 0, 0
    OPEN_TH, CLOSE_TH = 3, 5
    last_sent_number = None

    while running:
        ok, frame = capture.read()
        if not ok:
            time.sleep(0.01)
            continue

        # ... (ROI, 모델 추론 등은 동일)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, R_X1), max(0, R_Y1), min(w, R_X2), min(h, R_Y2)
        roi = frame[y1:y2, x1:x2]
        results = model(roi)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (C_X, C_Y), 3, (0, 0, 255), 1)

        center_detected = False
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)

        if boxes is not None:
            # [디버깅 1] 객체가 하나라도 감지되었는지 확인
            # print(f"감지된 객체 수: {len(boxes)}") 
            
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                X1, Y1, X2, Y2 = map(int, [xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1])
                
                # [디버깅 2] 신뢰도가 0.5 이상인지 확인
                if conf < 0.5:
                    continue
                # print(f"신뢰도 통과: {conf:.2f}")

                # [디버깅 3] 중심점이 박스 안에 들어왔는지 확인
                if not ((X1 <= C_X <= X2) and (Y1 <= C_Y <= Y2)):
                    continue
                # print("중심점 통과!")

                center_detected = True
                ocr_results = reader.readtext(frame[Y1:Y2, X1:X2])
                
                # [디버깅 4] OCR이 텍스트를 인식했는지 확인
                if not ocr_results:
                    # print("OCR 결과 없음")
                    continue
                
                for (_, text, _prob) in ocr_results:
                    text = text.translate(translation_table)
                    if text and len(text) == 4:
                        if text != last_sent_number:

                            print(f"✅ [전송 시도] 인식된 번호: {text}")
                            # sio.emit('message', {'data': text})
                            cv2.imwrite(f"{text}.jpg", frame)
                            sio.emit('plate_number', text)
                            with open(f"{text}.jpg", 'rb') as f:
                                sio.emit("plate_image", {"number": text, "image": f.read()}) 
                            last_sent_number = text
                        
                        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                        cv2.putText(frame, text, (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    
                    # [디버깅 5] 최종 전송 조건 확인
                    
                    # else:
                        # print(f"중복된 번호 감지: {text}")

        # ... (히스테리시스 및 화면 표시는 동일) ...
        # ...
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
            
    capture.release()
    cv2.destroyAllWindows()


# ==== Socket.IO 클라이언트 이벤트 핸들러 ====
@sio.event
def connect():
    """서버에 성공적으로 연결되었을 때 실행됩니다."""
    print('✅ 서버에 성공적으로 연결되었습니다!')

@sio.event
def disconnect():
    """서버와의 연결이 끊겼을 때 실행됩니다."""
    print('🔌 서버와의 연결이 끊겼습니다.')


# ==== 메인 실행 블록 ====
if __name__ == '__main__':
    try:
        # 1. 데이터를 받을 서버의 주소를 입력합니다.
        # !!!! 반드시 실제 서버의 IP 주소와 포트로 변경해주세요 !!!!
        server_address = 'http://localhost:5002' 
        sio.connect(server_address, transports=['websocket'])
       
        # 2. 서보 제어 스레드를 데몬 스레드로 시작합니다.
        servo_thread = threading.Thread(target=servo, daemon=True)
        servo_thread.start()

        # 3. 메인 스레드에서 YOLO/OCR 로직을 실행합니다.
        main_logic()

    except socketio.exceptions.ConnectionError as e:
        print(f"❌ 서버 연결 실패: {e}")
    except KeyboardInterrupt:
        print("\n프로그램 중단 요청 (Ctrl+C)")
    finally:
        # 4. 프로그램 종료 시 전역 플래그를 설정하고 연결을 확실히 해제합니다.
        running = False
        if sio.connected:
            sio.disconnect()
        # servo 스레드가 정리될 시간을 잠시 줍니다.
        time.sleep(0.5) 
        print("프로그램을 종료합니다.")
