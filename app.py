from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('pneumonia_model.h5')

def prepare_image(img_data):
    # Decode base64 image from JS
    encoded_data = img_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess
    img = cv2.resize(img, (150, 150))
    img = img.reshape(-1, 150, 150, 1) / 255.0
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = prepare_image(data['image'])
    
    prediction = model.predict(image)[0][0]
    
    # 0 = Pneumonia, 1 = Normal
    normal_prob = float(prediction * 100)
    pneumonia_prob = float((1 - prediction) * 100)
    
    diagnosis = "NORMAL" if prediction > 0.5 else "PNEUMONIA"
    
    return jsonify({
        "diagnosis": diagnosis,
        "normal": round(normal_prob, 2),
        "pneumonia": round(pneumonia_prob, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)