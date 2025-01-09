from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from werkzeug.utils import secure_filename
import logging
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(4)  # Example: Use 4 threads
tf.config.threading.set_inter_op_parallelism_threads(2)  # Example: Use 2 threads for inter-op parallelism



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
try:
    print("Loading model...")
    model = tf.keras.models.load_model('flowers_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FLOWER_CLASSES = ['Rose', 'Daisy', 'Dandelion', 'Sunflower', 'Tulip']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_flower(image_path):
    print(f"Processing image: {image_path}")
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        img = cv2.resize(img, (180, 180))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        score = tf.nn.softmax(prediction[0])
        predicted_class = np.argmax(score)
        confidence = float(score[predicted_class] * 100)
        
        return FLOWER_CLASSES[predicted_class], confidence
    except Exception as e:
        print(f"Error in predict_flower: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            
            flower_name, confidence = predict_flower(filepath)
            return jsonify({
                'success': True,
                'flower': flower_name,
                'confidence': f"{confidence:.2f}%",
                'image_path': f"/static/uploads/{filename}"
            })
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no PORT variable is set
    app.run(host='0.0.0.0', port=port)