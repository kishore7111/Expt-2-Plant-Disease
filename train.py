import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.io import imread 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Function to extract HOG and color histogram features from an image
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

# Function to load dataset and extract features
def load_dataset(healthy_dir, diseased_dir):
    features = []
    labels = []
    
    # Load healthy plant images
    for img_name in os.listdir(healthy_dir):
        img_path = os.path.join(healthy_dir, img_name)
        image = imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        features.append(extract_features(image))
        labels.append(0)  # Healthy label

    # Load diseased plant images
    for img_name in os.listdir(diseased_dir):
        img_path = os.path.join(diseased_dir, img_name)
        image = imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        features.append(extract_features(image))
        labels.append(1)  # Diseased label
    
    return np.array(features), np.array(labels)

# Directories for healthy and diseased plants
healthy_dir = 'images/healthy_plants'
diseased_dir = 'images/diseased_plants'

# Load the dataset
X, y = load_dataset(healthy_dir, diseased_dir)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Save the trained model and the scaler
joblib.dump(svm_classifier, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions and evaluate the model (optional)
y_pred = svm_classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
 