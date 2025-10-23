import socketio
import lgpio
import time

sio = socketio.Client(reconnection=True)

# ---- SERVO 설정 ----
SERVO_PIN = 18  # BCM 기준 PWM 가능 핀
CHIP = 0        # gpiochip 번호 (기본 0)

h = lgpio.gpiochip_open(CHIP)

# PWM 설정 (50Hz = 서보모터 기본 PWM)
PWM_FREQ = 50  # 50Hz (주기 20ms)

OPEN_ANGLE = 138
CLOSE_ANGLE = 48

def set_angle(angle):
    # angle(0~180) → pulse width (1~2ms)
    pulse_width = 1000 + (angle / 180) * 1000  # µs
    duty_cycle = pulse_width / 20000 * 100     # 20ms 기준 %

    lgpio.tx_pwm(h, SERVO_PIN, PWM_FREQ, duty_cycle)
    time.sleep(0.5)
    lgpio.tx_pwm(h, SERVO_PIN, 0, 0)  # PWM 멈춤 (서보 지터 방지)

# ---- SOCKET.IO 이벤트 ----
@sio.event
def connect():
    print("✅ 서버 연결됨")

@sio.on("gate_open")
def gate_open(data=None):
    print("🔓 게이트 열기 이벤트 수신!")
    set_angle(OPEN_ANGLE)

@sio.on("gate_close")
def gate_close(data=None):
    print("🔒 게이트 닫기 이벤트 수신!")
    set_angle(CLOSE_ANGLE)

@sio.event
def disconnect():
    print("❌ 서버 연결 끊김")

# ---- 메인 ----
SERVER_ADDRESS = "http://127.0.0.1:5002"

if __name__ == "__main__":
    try:
        # 서보 핀 OUTPUT 할당
        lgpio.gpio_claim_output(h, SERVO_PIN)

        sio.connect(SERVER_ADDRESS, transports=["websocket"])
        sio.wait()
    except Exception as e:
        print(f"🚨 실행 중 오류 발생: {e}")
    finally:
        lgpio.gpiochip_close(h)
