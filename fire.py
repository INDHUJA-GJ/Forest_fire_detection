from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO
import cv2
import cvzone
import math
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Load the YOLO model
model = YOLO('fire.pt')
classnames = ['fire']
# Variable to store uploaded video path
video_path = None
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5,
                                       thickness=2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path
    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        return redirect(url_for('video_feed'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
