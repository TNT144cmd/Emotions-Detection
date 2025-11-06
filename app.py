from flask import Flask, render_template, Response, request, jsonify
import cv2
import json
import re
from collections import defaultdict
from datetime import datetime
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time
import sys
import io
import threading
import os
from emotion_analyzer import StudentEmotionAnalyzer
import threading
from time import sleep

analyzer = StudentEmotionAnalyzer(HF_TOKEN="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")


app = Flask(__name__)

# Camera and emotion detection variables
camera = None
camera_lock = threading.Lock()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
mtcnn = MTCNN(keep_all=True, device=device)
face_recog_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
known_embeddings, known_names = torch.load(r"Model\Final_Model\new_face_db.pth", map_location=device)

# Emotion model
emotion_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
emotion_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = emotion_model.fc.in_features
emotion_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)
)
checkpoint = torch.load(r"Model\Final_Model\4_Model_checkpoint.pth", map_location=device, weights_only=True)
emotion_model.load_state_dict(checkpoint['model_state_dict'])
emotion_model = emotion_model.to(device)
emotion_model.eval()

FER2013_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion tracking variables
start_time = time.time()
last_time = start_time
person_emotion_stats = defaultdict(lambda: {
    emotion: {"score_sum": 0.0, "duration": 0.0, "count": 0} for emotion in FER2013_LABELS
})

# Global variables
file_path = r'emotion_per_person.json'  # Đảm bảo bạn thay thế đường dẫn file thực tế của bạn
current_model_output = None
last_update_time = None
cached_report = None
cached_parsed_data = None
data_lock = threading.Lock()

import json
json_file_path = "emotion_per_person.json"

def reset_data():
    global current_model_output, last_update_time, cached_report, cached_parsed_data
    with data_lock:
        current_model_output = None
        last_update_time = None
        cached_report = None
        cached_parsed_data = None
        
        # Clear the emotion_per_person.json file
        try:
            with open("emotion_per_person.json", "w") as f:
                # Write an empty JSON object to the file
                json.dump({}, f)
            # print("emotion_per_person.json đã được reset.")
        except Exception as e:
            print(f"Error resetting emotion_per_person.json: {e}")

reset_data()


def get_model_output():
    """Lấy output từ file JSON."""
    # Đọc dữ liệu từ file JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            students_data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None  # Trả về None nếu có lỗi đọc file
    
    # Xây dựng chuỗi output từ dữ liệu JSON
    output = ""
    for student in students_data:
        student_name = student['name']
        emotions = student['emotions']
        emotion_details = []
        
        # Tạo các cảm xúc cho từng học sinh
        for emotion, details in emotions.items():
            score = details['score']
            duration = details['duration']
            emotion_details.append(f"{emotion.lower()}={score} ({duration}min)")
        
        # Ghép các thông tin của học sinh lại thành chuỗi
        output += f"{student_name}: " + "; ".join(emotion_details) + " | "
    
    return output.strip(" | ")  # Loại bỏ dấu '|' thừa ở cuối chuỗi

def update_data_continuously():
    while True:
        # Kiểm tra và cập nhật input_text tự động
        with data_lock:
            global current_model_output
            # Cập nhật `current_model_output` (dữ liệu input_text)
            current_model_output = get_model_output()  # Cập nhật từ nguồn dữ liệu (có thể từ file JSON, DB...)
        
        sleep(5)  # Cập nhật mỗi 5 giây, có thể thay đổi tùy nhu cầu

# Chạy background task
thread = threading.Thread(target=update_data_continuously)
thread.daemon = True
thread.start()

def parse_emotion_data(input_text):
    EMOTIONS = ['happy', 'sad', 'anxious', 'excited', 'bored', 'confident', 'tired']
    students_data = []
    student_entries = input_text.strip().split('|')
    for entry in student_entries:
        entry = entry.strip()
        if not entry:
            continue
        name_match = re.match(r'^([^:]+):', entry)
        if not name_match:
            continue
        student_name = name_match.group(1).strip()
        emotion_pattern = r'(\w+)=([0-9.]+) \(([0-9.]+)min\)'
        emotions = re.findall(emotion_pattern, entry)
        student_emotions = {em: {"score": 0.0, "duration": 0.0} for em in EMOTIONS}
        for emotion, score, duration in emotions:
            if emotion in EMOTIONS:
                student_emotions[emotion] = {
                    'score': float(score),
                    'duration': float(duration)
                }
        students_data.append({
            'name': student_name,
            'emotions': student_emotions
        })
    return students_data

def check_data_freshness():
    """Kiểm tra và cập nhật cache nếu có dữ liệu mới."""
    global current_model_output, last_update_time, cached_report, cached_parsed_data
    with data_lock:
        new_output = get_model_output()
        if new_output != current_model_output:
            print("Detected new model output, updating cache...")
            current_model_output = new_output
            last_update_time = datetime.now()
            if new_output:
                try:
                    cached_parsed_data = parse_emotion_data(new_output)
                    cached_report = analyzer.analyze_and_generate_report(json_file_path, use_ai=True)
                except Exception as e:
                    print(f"Error processing new data: {e}")
                    cached_parsed_data = []
                    cached_report = "Error processing data"
            return True
        return False

cap = cv2.VideoCapture(0)  # 0 là ID của camera mặc định (có thể thay đổi nếu sử dụng camera khác)


def init_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            # video_path = r"D:\Code\y4_semester2\AI_Project\TestCase\IMG_8530.MOV" # Kha Ngan
            # video_path = r"D:\Code\y4_semester2\AI_Project\TestCase\BaoAn_My_Hoc.MOV" # Bao An va My
            # camera = cv2.VideoCapture(video_path)
            if not camera.isOpened():
                # print("Không thể kết nối đến camera.")
                camera = None

def save_emotion_data():
    """Ghi dữ liệu cảm xúc vào file JSON"""
    output_json = []
    for person_name, emotions in person_emotion_stats.items():
        emotion_info = {}
        for emotion, stats in emotions.items():
            if stats["duration"] > 0:
                avg_score = stats["score_sum"] / stats["duration"]
                emotion_info[emotion] = {
                    "score": round(avg_score, 2),
                    "duration": round(stats["duration"], 2)
                }
        output_json.append({
            "name": person_name,
            "emotions": emotion_info
        })

    with open(file_path, "w") as f:
        json.dump(output_json, f, indent=4)
def preprocess_emotion_face(face_gray):
    face_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    face_tensor = torch.from_numpy(face_gray).unsqueeze(0).unsqueeze(0).float()
    face_tensor = (face_tensor - 127.5) / 127.5
    return face_tensor.to(device)

def generate_frames():
    global last_time, person_emotion_stats
    init_camera()
    
    if camera is None:
        return
    
    while True:
        with camera_lock:
            if camera is None:
                break
            ret, frame = camera.read()
        
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        boxes, _ = mtcnn.detect(img)
        faces = mtcnn.extract(img, boxes, save_path=None) if boxes is not None else []

        for i, face_tensor in enumerate(faces):
            name = "Unknown"
            min_dist = 1.0

            # Nhận diện tên
            with torch.no_grad():
                emb = face_recog_model(face_tensor.unsqueeze(0).to(device))
            for db_emb, db_name in zip(known_embeddings, known_names):
                dist = (emb - db_emb).norm().item()
                if dist < min_dist and dist < 0.9:
                    min_dist = dist
                    name = db_name

            # Cắt khuôn mặt
            x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

            try:
                input_face = preprocess_emotion_face(gray_face)
                with torch.no_grad():
                    output = emotion_model(input_face)
                    prob = F.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)
                    label = FER2013_LABELS[pred.item()]
                    confidence = conf.item()
            except:
                continue

            # Cập nhật thống kê theo người
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            person_stats = person_emotion_stats[name]
            person_stats[label]["score_sum"] += confidence * delta_time
            person_stats[label]["duration"] += delta_time
            person_stats[label]["count"] += 1

            # Hiển thị
            display_text = f"{name} - {label} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # Save emotion data periodically
        if int(time.time()) % 10 == 0:  # Save every 10 seconds
            save_emotion_data()

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_input_text", methods=["GET"])
def get_input_text():
    with data_lock:
        # Trả về giá trị mới của input_text
        return jsonify({"input_text": current_model_output})


@app.route("/", methods=["GET", "POST"])
def index():
    global cached_report, cached_parsed_data, last_update_time
    check_data_freshness()
    with data_lock:
        report = None
        parsed_data = cached_parsed_data if cached_parsed_data else []
        input_text = current_model_output if current_model_output else ""
        update_time = last_update_time

    # Nếu muốn cho phép user nhập mới qua form
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        use_ai = request.form.get("use_ai") == "on"
        if input_text:
            try:
                report = analyzer.analyze_and_generate_report(json_file_path, use_ai)
                parsed_data = parse_emotion_data(input_text)
                update_time = datetime.now()
            except Exception as e:
                print(f"Error processing form data: {e}")
                report = f"Error: {str(e)}"
                parsed_data = []
    return render_template("index.html",
                          report=report,
                          input_text=input_text,
                          students_data=parsed_data,
                          last_update=update_time,
                          auto_refresh=True)

@app.route("/api/emotions")
def api_emotions():
    check_data_freshness()
    with data_lock:
        parsed_data = cached_parsed_data if cached_parsed_data else []
        update_time = last_update_time
    return jsonify({
        'data': parsed_data,
        'last_update': update_time.isoformat() if update_time else None,
        'total_students': len(parsed_data)
    })

@app.route("/api/update", methods=["POST"])
def api_update():
    try:
        data = request.json
        new_output = data.get('output_model_1', '')
        if new_output:
            global output_model_1
            output_model_1 = new_output
            return jsonify({'status': 'success', 'message': 'Data updated successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
