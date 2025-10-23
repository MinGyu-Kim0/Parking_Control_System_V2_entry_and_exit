from flask import Flask
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")  # ✅ 중요

def console_input():
    while True:
        cmd = input("명령 입력 (open / close): ").strip()
        if cmd == "open":
            socketio.emit("gate_open")
            print("✅ gate_open 전송 완료")
        elif cmd == "close":
            socketio.emit("gate_close")
            print("✅ gate_close 전송 완료")
        else:
            print("❌ 알 수 없는 명령")

@socketio.on("connect")
def connect():
    print("✅ 클라이언트 연결됨")

@socketio.on("disconnect")
def disconnect():
    print("❌ 클라이언트 연결 끊김")

if __name__ == "__main__":
    threading.Thread(target=console_input, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5002)

