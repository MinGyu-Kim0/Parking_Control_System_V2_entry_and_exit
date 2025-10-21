# =================================================================
# YOLO + CRNN OCR과 서보 제어 및 Socket.IO 연동 시스템
# CAM_TEST.py의 OCR 안정화 로직과 test_code2.py의 시스템 통합 버전
# =================================================================

# ==== 라이브러리 임포트 ====
import cv2
import numpy as np
import threading
import time
import socketio  # 클라이언트용
import re
from collections import deque, Counter
import requests

# ==== 딥러닝 라이브러리 ====
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# ==== 사용자 정의 CRNN 모델 구성 요소 ====
# 'train_and_infer_crnn_ctc.py' 파일이 같은 폴더에 있어야 합니다.
from train_and_infer_crnn_ctc import CRNN, CHARS, converter

# =========================
# 설정 및 전역 변수
# =========================
# --- 모델 및 장치 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W, IMG_H = 128, 32
YOLO_MODEL_PATH = "./best.pt"
CRNN_MODEL_PATH = "./checkpoints/crnn_best.pth"

# --- 관심 영역(ROI) 및 중심점 ---
R_X1, R_Y1, R_X2, R_Y2 = 200, 200, 600, 400
C_X1, C_Y1, C_X2, C_Y2 = 250, 200, 500, 400

# --- 서보 제어 (Jetson SYSFS PWM) ---
OPEN_ANGLE = 90
CLOSE_ANGLE = 0
PWM_CHIP = "/sys/class/pwm/pwmchip3"  # Jetson 모델에 따라 PWM 칩 번호가 다를 수 있습니다.
PWM_CH = "0"

# --- 애플리케이션 상태 ---
detect = False
running = True
history = deque(maxlen=10) # OCR 결과 안정화를 위한 Deque

# --- Socket.IO 클라이언트 ---
sio = socketio.Client()

# =========================
# CRNN OCR 관련 함수
# =========================
def load_crnn_model(ckpt_path):
    """체크포인트 파일에서 CRNN 모델을 불러옵니다."""
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        chars = ckpt.get("chars", CHARS)
        model = CRNN(num_classes=1 + len(chars)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print("✅ CRNN OCR 모델을 성공적으로 불러왔습니다.")
        return model, chars
    except FileNotFoundError:
        print(f"❌ 오류: CRNN 모델 체크포인트를 찾을 수 없습니다. 경로: '{ckpt_path}'.")
        exit()
    except Exception as e:
        print(f"❌ 오류: CRNN 모델 로딩에 실패했습니다. {e}")
        exit()

# 이미지 전처리를 위한 변환기
infer_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

@torch.no_grad()
def predict_ocr(img_pil: Image.Image):
    """PIL 이미지를 입력받아 OCR을 수행합니다."""
    x = infer_tf(img_pil).unsqueeze(0).to(DEVICE)
    logits, _ = ocr_model(x)
    log_probs = logits.log_softmax(2)
    pred = converter.decode_greedy(log_probs.cpu())[0]
    return pred

def stabilize_text(new_text: str):
    """
    OCR 결과를 안정화합니다.
    - 숫자 이외의 문자를 모두 제거합니다.
    - 4자리 숫자가 아니면 무효 처리합니다.
    - 최근 10개의 유효한 결과를 저장하고, 그 중 가장 빈번하게 나타난 값을 반환합니다.
    """
    # 숫자만 남기고 모든 문자 제거
    new_text = re.sub(r'[^0-9]', '', new_text)

    # 4자리 숫자가 아니면 유효하지 않은 것으로 간주하고 None 반환
    if len(new_text) != 4:
        return None

    # 유효한 4자리 숫자를 history에 추가
    history.append(new_text)
    # history에서 가장 많이 나온 숫자 찾기
    most_common = Counter(history).most_common(1)[0][0]
    return most_common

# =========================
# 서보 모터 제어 함수 (Jetson SYSFS)
# =========================
def pwm_write(path, value):
    try:
        with open(path, "w") as f:
            f.write(str(value))
    except Exception as e:
        print(f"[PWM 오류] {e}")

def pwm_init():
    try:
        pwm_write(f"{PWM_CHIP}/export", PWM_CH)
    except:
        pass  # 이미 export 된 경우 무시
    time.sleep(0.1)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "0")
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/period", "20000000")  # 20ms (50Hz)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/duty_cycle", "1500000") # 1.5ms (중앙값)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "1")

def set_servo_angle(angle):
    duty_ns = int(1000000 + (angle / 180.0) * 1000000)
    pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/duty_cycle", duty_ns)

def pwm_cleanup():
    try:
        pwm_write(f"{PWM_CHIP}/pwm{PWM_CH}/enable", "0")
        time.sleep(0.1)
        pwm_write(f"{PWM_CHIP}/unexport", PWM_CH)
    except:
        pass

# ==== 서보 제어 스레드 ====
def servo_thread_func():
    global detect, running
    print("✅ 서보 제어 스레드 시작")
    pwm_init()
    last_angle = None
    try:
        while running:
            angle = OPEN_ANGLE if detect else CLOSE_ANGLE
            if angle != last_angle:
                print(f"[서보] 각도 설정 -> {angle}도")
                set_servo_angle(angle)
                last_angle = angle
            time.sleep(0.05)
    finally:
        set_servo_angle(CLOSE_ANGLE) # 종료 시 차단기 닫기
        pwm_cleanup()
        print("🔚 서보 제어 스레드 정리 완료")

# =========================
# 모델 및 카메라 로딩
# =========================
print("딥러닝 모델을 로딩합니다...")
model = YOLO(YOLO_MODEL_PATH)
ocr_model, _ = load_crnn_model(CRNN_MODEL_PATH)
capture = cv2.VideoCapture(0)

# =========================
# 메인 로직
# =========================
def main_logic():
    global detect, running

    if not capture.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        running = False
        return

    last_seen_time = 0.0
    HOLD_SECONDS = 3.0
    last_sent_number = None

    while running:
        ok, frame = capture.read()
        if not ok:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        x1_roi, y1_roi, x2_roi, y2_roi = max(0, R_X1), max(0, R_Y1), min(w, R_X2), min(h, R_Y2)
        roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        results = model(roi)
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)

        center_detected = False

        # --- 탐지된 객체 중 가장 신뢰도 높은 것 하나만 선택 ---
        best_box = None
        best_conf = 0.0
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is not None and best_conf > 0.5:
            # 전체 프레임 기준 좌표로 변환
            xyxy = best_box.xyxy[0].tolist()
            X1, Y1, X2, Y2 = map(int, [xyxy[0] + x1_roi, xyxy[1] + y1_roi, xyxy[2] + x1_roi, xyxy[3] + y1_roi])

            # 중심점이 박스 안에 있는지 확인
            if (X1 >= C_X1) and (Y1 >= C_Y1) and (X2 <= C_X2) and (Y2 <= C_Y2):
                # --- OCR 실행 및 안정화 로직 ---
                crop = frame[Y1:Y2, X1:X2]

                if crop.size > 0:
                    # OpenCV 이미지를 PIL 이미지로 변환
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                    # OCR 예측
                    raw_text = predict_ocr(pil_img)
                    # 결과 안정화
                    text = stabilize_text(raw_text)

                    if text: # 안정화된 텍스트가 유효할 경우
                        print(f"✅ 인식된 번호: {text} (Raw: {raw_text})")
                        last_seen_time = time.time()

                        if text != last_sent_number:
                            try:
                                response = requests.post(server_address, params={"car_number": text})
                                center_detected = response.json()["parking_available"]
                                # 이미지 전송(사용 시 주석 해제)
                                # img_filename = f"{text}.jpg"
                                # cv2.imwrite(img_filename, frame)
                                # with open(img_filename, 'rb') as f:
                                #     sio.emit("plate_image", {"number": text, "image": f.read()})
                            except Exception as e:
                                print(f"[Socket Emit 오류] {e}")
                            last_sent_number = text

                        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                        cv2.putText(frame, text, (X1, Y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ==== 'detect' 상태 유지 및 해제 로직 ====
        now = time.time()
        if center_detected:
            detect = True
            last_seen_time = now
        else:
            if now - last_seen_time > HOLD_SECONDS:
                detect = False

        # --- 시각화 ---
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
        cv2.rectangle(frame, (C_X1, C_Y1), (C_X2, C_Y2), (0, 0, 255), 1)

        # 화면 표시
        try:
            cv2.imshow("YOLOv8 + CRNN OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
        except:
            pass

    capture.release()
    cv2.destroyAllWindows()

# =========================
# Socket.IO 이벤트 핸들러
# =========================
@sio.event
def connect():
    print('✅ 서버에 성공적으로 연결되었습니다!')

@sio.event
def disconnect():
    print('🔌 서버와의 연결이 끊겼습니다.')

# =========================
# 메인 실행 블록
# =========================
if __name__ == '__main__':
    server_address = 'http://localhost:5005' # 서버 주소
    response = requests.get(f"{server_address}/health")

    print(response.json()["message"])    

    # 서보 스레드 시작
    servo_thread = threading.Thread(target=servo_thread_func, daemon=True)
    servo_thread.start()

    # 메인 로직 실행
    try:
        main_logic()
    except KeyboardInterrupt:
        print("\n🛑 프로그램 중단 요청 (Ctrl+C)")
    finally:
        running = False
        time.sleep(0.5) # 스레드가 정리될 시간을 줍니다.
        print("프로그램을 종료합니다.")