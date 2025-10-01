from ultralytics import YOLO
import cv2, numpy as np, easyocr, threading, time
import RPi.GPIO as GPIO


# ==== 전역 상태 ====
detect = False
running = True

# ==== 서보 각도 -> 듀티 변환 ====
def angle_to_duty(angle: float) -> float:
    # 0도=2.5%, 180도=12.5% (대부분의 서보 범용)
    return 2.5 + (angle * 10.0 / 180.0)

OPEN_ANGLE = 90  # 열기 각도(필요시 조정)
CLOSE_ANGLE = 0  # 닫기 각도(필요시 조정)

# ==== 서보 스레드 ====
def servo():
    global detect, running

    GPIO.setmode(GPIO.BOARD)
    SERVO_PIN = 32  # BOARD 32 = BCM 12 (PWM0 가능핀)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
    pwm.start(0)

# 상태 변화시에만 듀티 갱신 (중복 호출로 인한 떨림 방지)
    last_target = None
    try:
        while running:
        # detect만 보고 목표 각도 결정
            target_angle = OPEN_ANGLE if detect else CLOSE_ANGLE

            if target_angle != last_target:
                duty = angle_to_duty(target_angle)
                pwm.ChangeDutyCycle(duty)
            # 짧게 안정화 대기 (너무 짧으면 미세 떨림, 너무 길면 반응 느림)
                time.sleep(0.18)
            # 같은 듀티를 계속 때리면 떨리는 케이스가 있어 한박자 쉬어줌
                pwm.ChangeDutyCycle(0)  # pulse 유지 대신, 잠깐 0으로 내려 타이밍 흔들림 감소(토크 유지 필요하면 이 줄 빼도 됨)
                last_target = target_angle

        # 폴링 주기 (너무 짧으면 CPU 낭비/떨림, 너무 길면 반응 느림)
            time.sleep(0.02)
    finally:
        pwm.stop()
        GPIO.cleanup()

# ==== 모델/캠 ====
model = YOLO("best.pt")
capture = cv2.VideoCapture(0)
reader = easyocr.Reader(['ko'], gpu=False)

# ROI / 중심
R_X1, R_Y1, R_X2, R_Y2 = 200, 200, 600, 400
C_X, C_Y = (R_X1 + R_X2) // 2, (R_Y1 + R_Y2) // 2


# ==== 메인 스레드 ====
def main():
    global detect, running

    if not capture.isOpened():
        print("카메라를 열 수 없습니다.")
        running = False
        return False

# 히스테리시스: 연속 프레임 기준으로 열고/닫기 결정
    open_count = 0
    close_count = 0
    OPEN_TH = 3  # 중심 안에서 3프레임 연속 감지되면 '열기'
    CLOSE_TH = 5  # 5프레임 연속 미검지면 '닫기'

    try:
        while running:
            ok, frame = capture.read()
            if not ok or frame is None:
            # 프레임 읽기 실패 시 잠깐 대기 후 계속
                time.sleep(0.01)
                continue

        # ROI 슬라이싱 안전성 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(R_X1, w - 1))
            x2 = max(0, min(R_X2, w))
            y1 = max(0, min(R_Y1, h - 1))
            y2 = max(0, min(R_Y2, h))

            if x2 <= x1 or y2 <= y1:
            # 잘못된 ROI면 전체 프레임 사용
                roi = frame
            else:
                roi = frame[y1:y2, x1:x2]

        # 추론
            results = model(roi)

        # 가시화
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (C_X, C_Y), 3, (0, 0, 255), 3)

        # 디텍션 판정 (중심점이 감지 박스 내부에 들어오면 '감지')
            center_detected = False
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                # names 딕셔너리 접근 안전화
                    names = getattr(r0, "names", {})
                    label = names.get(cls_id, str(cls_id))

                    o_x1, o_y1, o_x2, o_y2 = map(int, xyxy)
                # ROI 좌표를 원본 프레임 좌표로 보정
                    X1, Y1 = o_x1 + x1, o_y1 + y1
                    X2, Y2 = o_x2 + x1, o_y2 + y1

                    if conf >= 0.5:
                    # 중심점이 박스 내부?
                        if (X1 <= C_X <= X2) and (Y1 <= C_Y <= Y2):
                            center_detected = True
                        # OCR은 비용 큰 편: 중심안에서만 수행
                            ocr = reader.readtext(frame[Y1:Y2, X1:X2])
                            for (_, text, _prob) in ocr:
                                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                                cv2.putText(frame, text, (X1, Y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
        # ---- 히스테리시스 적용 ----
            if center_detected:
                open_count += 1
                close_count = 0
            else:
                close_count += 1
                open_count = 0

        # 상태 갱신은 문턱을 넘겼을 때만 (서보 떨림 방지)
            if open_count >= OPEN_TH and not detect:
                detect = True
            elif close_count >= CLOSE_TH and detect:
                detect = False

            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return detect

# ==== 스레드 실행 ====
t1 = threading.Thread(target=main, daemon=True)
t2 = threading.Thread(target=servo, daemon=True)

t1.start(); t2.start()

t1.join(); t2.join()