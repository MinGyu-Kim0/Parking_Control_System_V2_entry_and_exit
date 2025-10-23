import socketio
import gpiozero as GPIO
import time

sio = socketio.Client(reconnection=True)

SERVO_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

OPEN_ANGLE = 138
CLOSE_ANGLE = 48

def set_angle(angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)


@sio.event
def connect():
    print("✅ 서버 연결됨")

@sio.on("gate_open")
def gate_open(data=None):
    set_angle(OPEN_ANGLE)
    print("🔓 게이트 열기 이벤트 수신!")

@sio.on("gate_close")
def gate_close(data=None):
    set_angle(CLOSE_ANGLE)
    print("🔒 게이트 닫기 이벤트 수신!")

@sio.event
def disconnect():
    print("❌ 서버 연결 끊김")

# 🚨 핵심: WebSocket으로 강제 연결
SERVER_ADDRESS = "http://127.0.0.1:5002"

if __name__ == "__main__":
    try:
        sio.connect(SERVER_ADDRESS, transports=["websocket"])
        sio.wait()
    except:
        pass