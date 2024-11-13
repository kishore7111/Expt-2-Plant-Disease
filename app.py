import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
from skimage.feature import hog
from skimage.io import imread
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Create the 'uploads' folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract features from an image
def extract_features(image):
    fixed_size = (128, 128)
    image = cv2.resize(image, fixed_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, _ = hog(gray, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    hist_features = []
    for channel in range(image.shape[2]):
        hist = cv2.calcHist([image], [channel], None, [32], [0, 256])
        hist_features.extend(hist.flatten())
    return np.hstack((hog_features, hist_features))

# Load pre-trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Route to display the upload form
@app.route('/')
def index():
    return render_template('home.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the image to the uploads folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Load and preprocess the image for prediction
        image = imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        features = extract_features(image)
        features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features)[0]
        label = 'Healthy' if prediction == 0 else 'Diseased'
        
        # Render the result template with the image and label
        return render_template('analysis.html', label=label, image_url=file.filename)

# Route to serve uploaded menaja
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
