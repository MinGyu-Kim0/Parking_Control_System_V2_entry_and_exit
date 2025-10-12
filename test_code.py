# test_code.py
# ==== 라이브러리 임포트 ====
from ultralytics import YOLO
import cv2, numpy as np, easyocr, threading, time
import socketio  # 클라이언트용

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
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/duty_cycle", "1500000")  # 중앙값
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "1")

def set_servo_angle(angle):
    # 0~180도 → 1ms~2ms 맵핑 (MG996R)
    duty_ns = int(1000000 + (angle / 180.0) * 1000000)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/duty_cycle", duty_ns)
    time.sleep(0.2)

def pwm_cleanup():
    try:
        pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "0")
    except:
        pass

# ==== Socket.IO 클라이언트 설정 ====
sio = socketio.Client()

# ==== 전역 상태 ====
detect = False          # 서보 열림 여부 (서보 스레드에서 사용)
running = True

# ==== OCR 특수문자 제거 테이블 ====
special_chars = "!{()},'`.^ "
translation_table = str.maketrans('', '', special_chars)

# ==== 서보 각도 정의 ====
OPEN_ANGLE = 90
CLOSE_ANGLE = 0

# ==== 서보 스레드 ====
def servo():
    global detect, running
    print("✅ SERVO THREAD START")
    # 강제 해제 후 초기화 — 안전하게 1번만 수행
    pwm_force_release()
    pwm_init()

    last_angle = None
    try:
        while running:
            angle = OPEN_ANGLE if detect else CLOSE_ANGLE
            if angle != last_angle:
                print(f"[SERVO] set angle -> {angle}")
                set_servo_angle(angle)
                last_angle = angle
            time.sleep(0.05)
    finally:
        pwm_cleanup()
        print("🔚 SERVO CLEANUP DONE")

# ==== 모델 / 카메라 / OCR 초기화 ====
model = YOLO("best.pt")
capture = cv2.VideoCapture(0)
reader = easyocr.Reader(['ko'], gpu=True)

# ROI / 중심 좌표
R_X1, R_Y1, R_X2, R_Y2 = 200, 200, 600, 400
C_X, C_Y = (R_X1 + R_X2) // 2, (R_Y1 + R_Y2) // 2

# ==== 메인 로직 ====
def main_logic():
    global detect, running

    if not capture.isOpened():
        print("카메라를 열 수 없습니다.")
        running = False
        return

    last_seen_time = 0.0   # 마지막으로 OCR 인식(정상)된 시각
    HOLD_SECONDS = 3.0     # 유지 시간 (사용자 요구대로 3초)
    last_sent_number = None

    while running:
        ok, frame = capture.read()
        if not ok:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, R_X1), max(0, R_Y1), min(w, R_X2), min(h, R_Y2)
        roi = frame[y1:y2, x1:x2]

        # 모델 추론 (YOLO)
        results = model(roi)

        # 시각화: ROI 박스 및 중심점
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (C_X, C_Y), 3, (0, 0, 255), 1)

        center_detected = False
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)

        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                X1, Y1, X2, Y2 = map(int, [xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1])

                # 신뢰도 조건
                if conf < 0.5:
                    continue

                # 중심점이 박스 안에 있는지
                if not ((X1 <= C_X <= X2) and (Y1 <= C_Y <= Y2)):
                    continue

                # 중심에 들어온 경우 OCR 시도
                ocr_results = reader.readtext(frame[Y1:Y2, X1:X2])
                if not ocr_results:
                    continue

                for (_bbox, text, prob) in ocr_results:
                    text = text.translate(translation_table)
                    # 숫자판이 4글자인지 확인
                    if text and len(text) == 4:
                        center_detected = True
                        last_seen_time = time.time()   # ★ 마지막 인식 시각 갱신
                        # 전송 처리 (기존 동작 유지)
                        if text != last_sent_number:
                            print(f"✅ [전송 시도] 인식된 번호: {text}")
                            cv2.imwrite(f"{text}.jpg", frame)
                            try:
                                sio.emit('plate_number', text)
                                with open(f"{text}.jpg", 'rb') as f:
                                    sio.emit("plate_image", {"number": text, "image": f.read()})
                            except Exception as e:
                                print(f"[SOCKET EMIT ERROR] {e}")
                            last_sent_number = text

                        # 사각형 / 텍스트 표시
                        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                        cv2.putText(frame, text, (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # 한 프레임에서 여러 OCR 결과가 있어도 첫 인식으로 처리
                        break
                # 하나의 박스로 충분하므로 다음 박스로 넘어가지 않음
                if center_detected:
                    break

        # ---- detect 유지/해제 로직 (요구사항: 보이면 3초 유지, 보이면 다시 연장) ----
        now = time.time()
        if center_detected:
            # 이미 last_seen_time은 위에서 갱신됨; detect는 열림 상태로 유지
            detect = True
        else:
            # center_detected False : 마지막 인식 후 경과 시간이 3초보다 크면 닫음
            if now - last_seen_time > HOLD_SECONDS:
                detect = False
            else:
                # 아직 3초 유지 기간 안이면 열림 유지
                detect = True if last_seen_time > 0 else False

        # 디버그 출력 (원치 않으면 주석 처리)
        print(f"[DEBUG] center_detected={center_detected}, detect={detect}, last_seen_age={now-last_seen_time:.2f}")

        # 화면 표시 (Jetson에서 GUI가 설치되어 있을 때)
        try:
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
        except Exception as e:
            # OpenCV GUI가 없거나 에러가 나도 계속 실행 가능
            pass

    # 루프 종료 시 정리
    capture.release()
    cv2.destroyAllWindows()

# ==== Socket.IO 이벤트 핸들러 ====
@sio.event
def connect():
    print('✅ 서버에 성공적으로 연결되었습니다!')

@sio.event
def disconnect():
    print('🔌 서버와의 연결이 끊겼습니다.')

# ==== 메인 실행 블록 ====
if __name__ == '__main__':
    try:
        server_address = 'http://localhost:5002'
        sio.connect(server_address, transports=['websocket'])
    except Exception as e:
        print(f"[SOCKET CONNECT] {e}  (계속 로컬 동작은 진행됩니다)")

    # 서보 스레드 시작
    servo_thread = threading.Thread(target=servo, daemon=True)
    servo_thread.start()

    # 메인 루프 실행
    try:
        main_logic()
    except KeyboardInterrupt:
        print("프로그램 중단 요청 (Ctrl+C)")
    finally:
        running = False
        # 연결 해제
        try:
            if sio.connected:
                sio.disconnect()
        except:
            pass
        time.sleep(0.5)
        print("프로그램 종료")
