# =================================================================
# YOLO + CRNN OCR과 서보 제어 및 Socket.IO 연동 시스템
# (★ 네트워크 비동기 스레드 처리 및 안정화 로직 개선 버전)
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
import base64 

# ==== 딥러닝 라이브러리 ====
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# ==== 사용자 정의 CRNN 모델 구성 요소 ====
# 'train_and_infer_crnn_ctc.py' 파일이 같은 폴더에 있어야 합니다.
try:
    from train_and_infer_crnn_ctc import CRNN, CHARS, converter
except ImportError:
    print("❌ 오류: 'train_and_infer_crnn_ctc.py' 파일을 찾을 수 없습니다.")
    exit()

# =========================
# 설정 및 전역 변수
# =========================
# --- 모델 및 장치 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W, IMG_H = 128, 32
YOLO_MODEL_PATH = "./best.pt"
CRNN_MODEL_PATH = "./checkpoints/crnn_best.pth"

# --- 네트워크 ---
HTTP_ADDRESS = 'http://192.168.0.48:5005' # 서버 주소 (전역으로 이동)
SIO_SERVER = 'http://localhost:5003'  # 이미지 전송 서버 (전역으로 이동)

# --- 관심 영역(ROI) 및 중심점 ---
R_X1, R_Y1, R_X2, R_Y2 = 200, 200, 600, 400
C_X1, C_Y1, C_X2, C_Y2 = 250, 200, 500, 400

# --- 서보 제어 (Jetson SYSFS PWM) ---
OPEN_ANGLE = 90
CLOSE_ANGLE = 0
PWM_CHIP = "/sys/class/pwm/pwmchip3"  # Jetson 모델에 따라 PWM 칩 번호가 다를 수 있습니다.
PWM_CH = "0"
HOLD_SECONDS = 3.0 # 서보가 열린 상태를 유지하는 시간

# --- 애플리케이션 상태 (★ 전역 관리) ---
detect = False # 서보 열림/닫힘 플래그
running = True # 메인 루프 실행 플래그
last_seen_time = 0.0 # 마지막으로 유효한 응답을 받은 시간
last_sent_number = None # 마지막으로 서버에 전송한 번호 (중복 전송 방지)

# --- OCR 안정화 (★ 로직 변경) ---
history = deque(maxlen=7) # 반응성을 위해 10 -> 7로 변경
STABLE_COUNT_REQ = 3      # 7개 프레임 중 3개 이상 같아야 안정화

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
    OCR 결과를 안정화합니다. (★ 개선된 로직)
    - 4자리 숫자가 아니면 None을 history에 추가
    - 4자리 숫자면 해당 숫자를 history에 추가
    - history에서 가장 빈번한 값을 찾되, 그 횟수가 STABLE_COUNT_REQ 이상일 때만 반환
    """
    cleaned_text = re.sub(r'[^0-9]', '', new_text)
    
    if len(cleaned_text) != 4:
        history.append(None) # 유효하지 않은 값 (연속성 깨기)
        return None
    
    # 유효한 4자리 숫자 추가
    history.append(cleaned_text)

    # history에 데이터가 충분하지 않으면 None 반환
    if len(history) < STABLE_COUNT_REQ:
        return None
        
    # Counter를 사용해 가장 빈번한 값과 횟수 찾기
    counts = Counter(history)
    most_common, count = counts.most_common(1)[0]
    
    # 가장 빈번한 값이 'None'이 아니고, 최소 요구 횟수를 넘었을 때만 반환
    if most_common is not None and count >= STABLE_COUNT_REQ:
        return most_common
        
    return None # 아직 안정화되지 않음

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
# 네트워크 스레드 함수 (★ 견고하게 수정된 버전)
# =========================
def send_to_server(car_number, frame_to_send):
    """
    [별도 스레드] 서버로 데이터를 전송하고, 응답에 따라 'detect' 플래그를 설정합니다.
    (★ 서버가 JSON이 아닌 빈 값/HTML 오류를 보내도 절대 죽지 않도록 수정됨)
    """
    global detect, last_seen_time
    
    server_response_ok = False # ★ 서버가 "주차 가능"이라고 응답했는지 여부

    try:
        print(f"🚀 [네트워크 스레드] 서버 전송 시도: {car_number}")
        
        # 1. HTTP 요청 (주차 가능 여부 확인)
        response = requests.post(HTTP_ADDRESS, params={"car_number": car_number}, timeout=5) # 5초 타임아웃
        response.raise_for_status() # 4xx, 5xx 에러가 나면 예외 발생

        # ★★★ 핵심 수정 ★★★
        # 서버가 JSON이 아닌 (크래시로 인한) 빈 값이나 HTML을 보낼 경우
        try:
            response_data = response.json()
            parking_available = response_data.get("parking_available", False)
        except requests.exceptions.JSONDecodeError:
            # "Expecting value..." 오류가 여기서 잡힘
            print("❌ [네트워크 스레드] 오류: 서버가 JSON이 아닌 응답 (빈 값 또는 HTML)을 반환함.")
            parking_available = False # 주차 불가로 간주

        if parking_available:
            print(f"✅ [네트워크 스레드] 서버 응답: {car_number} 주차 가능. 서보 개방.")
            server_response_ok = True
        else:
            print(f"❌ [네트워크 스레드] 서버 응답: {car_number} 주차 불가.")
            
        # 2. Socket.IO로 이미지 전송... (이전과 동일)
        # if sio.connected:
        #     print("📸 [네트워크 스레드] Socket.IO로 이미지 전송 시도...")
        #     _, buffer = cv2.imencode('.jpg', frame_to_send)
        #     img_b64 = base64.b64encode(buffer).decode('utf-8')
        #     sio.emit("entry_photo", {"node": {"car_number": car_number, "entry_photo": img_b64}})
        #     print("... 이미지 전송 완료.")

    except requests.exceptions.ConnectionError as e:
        print(f"❌ [네트워크 스레드] HTTP 서버 연결 실패 (서버가 꺼져있음): {e}")
    except requests.exceptions.Timeout as e:
        print(f"❌ [네트워크 스레드] HTTP 서버 응답 시간 초과: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"❌ [네트워크 스레드] HTTP 오류 (예: 500 Internal Server Error): {e}")
    except Exception as e:
        print(f"❌ [네트워크 스레드] 알 수 없는 오류 발생: {e}")

    # 통신이 성공하고 + 서버가 "주차 가능"이라고 했을 때만
    # detect 플래그와 시간을 갱신합니다.
    if server_response_ok:
        detect = True
        last_seen_time = time.time()
            


# =========================
# 모델 및 카메라 로딩
# =========================
print("딥러닝 모델을 로딩합니다...")
model = YOLO(YOLO_MODEL_PATH)
ocr_model, _ = load_crnn_model(CRNN_MODEL_PATH)
capture = cv2.VideoCapture(0)

# =========================
# 메인 로직 (★ 수정됨)
# =========================
def main_logic():
    global detect, running, last_sent_number, last_seen_time

    if not capture.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        running = False
        return

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

        stable_text = None # ★ 루프 시작 시 안정화 텍스트 초기화

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
            xyxy = best_box.xyxy[0].tolist()
            X1, Y1, X2, Y2 = map(int, [xyxy[0] + x1_roi, xyxy[1] + y1_roi, xyxy[2] + x1_roi, xyxy[3] + y1_roi])

            # 중심점이 박스 안에 있는지 확인
            if (X1 >= C_X1) and (Y1 >= C_Y1) and (X2 <= C_X2) and (Y2 <= C_Y2):
                crop = frame[Y1:Y2, X1:X2]
                if crop.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                    raw_text = predict_ocr(pil_img)
                    stable_text = stabilize_text(raw_text) # ★ 개선된 안정화 함수 호출

                    if stable_text:
                        print(f"👀 [메인 루프] 안정화된 번호: {stable_text} (Raw: {raw_text})")
                        
                        # ★★★ 핵심 수정 ★★★
                        # 안정화된 번호가 '새로운' 번호일 때만 서버 전송 스레드 시작
                        if stable_text != last_sent_number:
                            last_sent_number = stable_text # ★ 상태를 즉시 업데이트 (중복 전송 방지)
                            # ★ 네트워크 작업을 별도 스레드로 실행 (non-blocking)
                            threading.Thread(
                                target=send_to_server, 
                                args=(stable_text, frame.copy()), # 프레임 복사본 전달
                                daemon=True
                            ).start()

                        # 시각화 (서버 응답과 관계없이 즉시 표시)
                        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
                        cv2.putText(frame, stable_text, (X1, Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ==== 'detect' 상태 유지 및 해제 로직 (★ 수정됨) ====
        # 'detect'는 네트워크 스레드가 True로 설정함.
        # 여기서는 False로 설정하는 것만 담당.
        now = time.time()
        if now - last_seen_time > HOLD_SECONDS:
            if detect: # 상태가 True였다면
                print("🔒 [메인 루프] 시간 초과. 서보 닫힘.")
            detect = False
            
            # 화면에 안정화된 텍스트가 보이지 않으면 (차가 떠났으면)
            # last_sent_number를 리셋하여 다시 감지할 수 있도록 함
            if stable_text is None:
                last_sent_number = None 

        # --- 시각화 ---
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
        cv2.rectangle(frame, (C_X1, C_Y1), (C_X2, C_Y2), (0, 0, 255), 1)

        try:
            cv2.imshow("YOLOv8 + CRNN OCR (Non-Blocking)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
        except Exception as e:
            # (예: 도커 -e DISPLAY=$DISPLAY 옵션 없이 실행 시)
            print(f"⚠️ GUI 표시 실패 (오류: {e}). GUI 없이 계속 실행합니다.")
            pass 

    capture.release()
    cv2.destroyAllWindows()

# =========================
# Socket.IO 이벤트 핸들러
# =========================
@sio.event
def connection():
    print('✅ Socket.IO 서버에 성공적으로 연결되었습니다!')
    sio.emit("register", {"id" : "pi7"})
@sio.event
def disconnect():
    print('🔌 Socket.IO 서버와의 연결이 끊겼습니다.')

# =========================
# 메인 실행 블록
# =========================
if __name__ == '__main__':
    try:
        response = requests.get(f"{HTTP_ADDRESS}/health")
        print(f"✅ HTTP 서버 연결 성공: {response.json()['message']}")
    except requests.exceptions.ConnectionError:
        print(f"❌ HTTP 서버({HTTP_ADDRESS})에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        exit()
    
    # 이미지 전송할 서버 (필요시 주석 해제)
    # try:
    #    sio.connect(SIO_SERVER, transports=['websocket'])
    # except socketio.exceptions.ConnectionError:
    #    print(f"❌ Socket.IO 서버({SIO_SERVER})에 연결할 수 없습니다.")
    #    pass # Socket.IO 없이 계속 진행

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
        if sio.connected:
            sio.disconnect()
        time.sleep(0.5) # 스레드가 정리될 시간을 줍니다.
        print("프로그램을 종료합니다.")