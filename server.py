from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from flask_cors import CORS
from backend.db_helper import *
from main import *
import os
import sys
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import threading
from audio_detection import audio_detection  # <-- import audio detection

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'  # Needed if you want to use sessions

audio_logs = []

@app.route('/get_audio_logs')
def get_audio_logs():
    return jsonify(audio_logs)


# 1. Signup Route
@app.route('/signup_data', methods=['POST'])
def signup_data():
    data = request.get_json()
    if insert_signup(data['signupEmail'], data['username'], data['signupPassword']) == 1:
        return jsonify({'message': 'Data inserted successfully!'})
    else:
        return jsonify({'message': 'Error in inserting the Data!'})

# 2. Login Route
@app.route('/login_data', methods=['POST'])
def login_data():
    data = request.get_json()
    print(data)
    response_data = search_login_credentials(data['email'], data['password'])
    if response_data:
        return jsonify(response_data)
    return jsonify({'message': 'Data not found!'})

# 3. Index
@app.route('/')
def index_page():
    return render_template('index.html')

# 4. Verification Page
@app.route('/verify')
def verify():
    return render_template('verify.html')
@app.route('/verify_face', methods=['POST'])
@app.route('/verify_face', methods=['POST'])
def verify_face():
    # List of multiple reference images
    reference_image_paths = [
        "static/reference.jpg",
        "static/reference_b1.jpg",
        # Add more paths as needed
    ]

    reference_encodings = []
    for path in reference_image_paths:
        if not os.path.exists(path):
            return render_template('verify.html', error=f"Reference image not found: {path}")
        
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) != 1:
            return render_template('verify.html', error=f"{path} must contain exactly one face.")
        
        reference_encodings.append(encodings[0])

    # Get uploaded image from user
    file = request.files.get('photo')
    if not file or file.filename == '':
        return render_template('verify.html', error="No image uploaded.")

    uploaded_image = face_recognition.load_image_file(file)
    uploaded_encodings = face_recognition.face_encodings(uploaded_image)
    if len(uploaded_encodings) != 1:
        return render_template('verify.html', error="Uploaded image must contain exactly one face.")

    uploaded_encoding = uploaded_encodings[0]

    # Compare uploaded face with all reference encodings
    face_distances = face_recognition.face_distance(reference_encodings, uploaded_encoding)
    min_distance = min(face_distances)
    threshold = 0.45  # You can increase slightly if it's too strict

    print(f"[INFO] Face distances: {face_distances}, Minimum: {min_distance}")

    if min_distance <= threshold:
        return redirect(url_for('quiz_page'))
    else:
        return render_template('verify.html', error=f"Face verification failed. Distance: {min_distance:.2f}")



@app.route('/quiz_html')
def quiz_page():
    # Start audio detection in background thread
    audio_thread = threading.Thread(target=audio_detection, args=([],), daemon=True)
    audio_thread.start()
    
    return render_template('quiz.html')

# 7. Video Feed
@app.route('/video_feed')
def video_feed():
    return Response(proctoringAlgo(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 8. Stop Camera
@app.route('/stop_camera')
def stop_camera():
    global running
    running = False
    main_app()
    print('Camera and Server stopping.....')
    os._exit(0)

# 9. Main
if __name__ == "__main__":
    print("Starting the Python Flask Server.....")
    app.run(port=5000, debug=False, use_reloader=False)
