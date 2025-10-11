# 1. 필요한 라이브러리를 가져옵니다. (Flask-SocketIO로 변경)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# 2. Flask 앱을 생성합니다.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


# 3. Flask-SocketIO를 사용하여 Flask 앱과 Socket.IO를 초기화합니다.
# 이 한 줄이 복잡한 연동을 모두 처리해 줍니다.
# socketio = SocketIO(app, async_mode='eventlet')
socketio = SocketIO(app, cors_allowed_origins="*")

# 4. Flask 라우트를 정의합니다.
@app.route('/')
def index():
    return render_template('index.html')

# --- Socket.IO 이벤트 핸들러 (함수 시그니처가 약간 다름) ---

# 5. 클라이언트 연결 이벤트 핸들러
# Flask-SocketIO 에서는 @socketio.on() 데코레이터를 사용합니다.
@socketio.on('connect')
def handle_connect():
    print('✅ Client connected!')

# 6. 클라이언트 연결 종료 이벤트 핸들러
@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected!')

# 7. 사용자 정의 채팅 메시지 이벤트 핸들러
@socketio.on('chat_message')
def handle_message(data):
# def handle_ocr_result(data):
    print(f'📥 Received message: {data["message"]}')
    # 모든 클라이언트에게 메시지를 보낼 때는 broadcast=True 옵션을 사용합니다.
    emit('server_message', data, broadcast=True)

# 8. 서버를 실행합니다.
if __name__ == '__main__':
    # uvicorn 대신 socketio.run()을 사용하여 서버를 실행합니다.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)