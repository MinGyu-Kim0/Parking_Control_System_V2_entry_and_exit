import time
import platform
import cv2, numpy as np
import socketio as socketio_client

from flask import Flask, render_template, Response, stream_with_context, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# 2. Flask 앱을 생성합니다.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
flask_sio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})

# 접속한 라즈베리파이 관리 ( IP + Session ID )
clients = {}

@app.route('/')
def home():
    return "✅ Flask-SocketIO 서버 실행 중 (5002)"

# 클라이언트에서 연결되었을 때 처리하는 이벤트
@flask_sio.on('connect')
def handle_connect():
	ip = request.remote_addr
	sid = request.sid
	clients[ip] = sid
	print(f"Client connected: {ip} -> {sid}")
	emit("message", {'data': 'Connected to server'})

# 클라이언트에서 보내는 기본 메세지 처리
@flask_sio.on('plate_number')
def handle_message(number):
    print(f"Received message:  {number}")
    # 받은 데이터를 처리하거나, 필요에 따라 클라이언트에게 다시 전송할 수 있습니다.
    emit('data', {'data': number}, broadcast=True)   # broadcast=True로 설정하면 연결된 모든 클라이언트에게 전송
    # express_sio.emit("vehicle_data", data) # Flask가 Express 서버로 데이터 전송

@flask_sio.on('plate_image')
def handle_plate_image(data):
    number = data["number"]
    image_bytes = data["image"]

    # ✅ 바이트 데이터를 numpy 배열로 변환
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # ✅ 저장 경로 설정
    filename = f"received_{number}.jpg"
    cv2.imwrite(filename, img)

    print(f"✅ 저장 완료: {filename}")

@flask_sio.on('rpi_data')
def handle_rpi_data(data):
    for target_ip, payload in data.items():
        if target_ip in clients:
            flask_sio.emit("rpi_data", payload, room=clients[target_ip])
            print(f"Sent to {target_ip} : {payload}")
        else:
            print(f"{target_ip} not connected")

# 클라이언트가 연결 해제되었을 때 처리하는 이벤트
@flask_sio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Flask-SocketIO는 일반 Flask와 다르게 socketio.run()을 사용해 서버를 실행합니다.
    flask_sio.run(app, host='0.0.0.0', port=5002, debug=True)
    