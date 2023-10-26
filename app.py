# app.py
import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from yolov4 import detect_objects

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            image.save(image_path)
            img = cv2.imread(image_path)
            results = detect_objects(image_path)  # yolov4.py에서 정의한 객체 검출 함수
            return render_template('index.html', image_path=image_path, results=results)


if __name__ == '__main__':
    app.run(debug=True)
