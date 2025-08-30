import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Parameters
IMG_SIZE = (64, 64)   
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

# Feature extractor

def extract_hog_features(img):
    img = cv2.resize(img, IMG_SIZE)
    features, _ = hog(img, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, visualize=True)
    return features

def extract_hog_from_path(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    return extract_hog_features(img)


# Load dataset
DATASET_PATH = "dataset/" 

X, y = [], []
for label, folder in enumerate(["cats", "dogs"]):
    folder_path = os.path.join(DATASET_PATH, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            features = extract_hog_from_path(os.path.join(folder_path, file))
            X.append(features)
            y.append(label)

X, y = np.array(X), np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["cat","dog"]))

# Save the model
joblib.dump(model, "cat_dog_svm.pkl")

# Prediction function
def predict_image(img_path):
    features = extract_hog_from_path(img_path).reshape(1, -1)
    model = joblib.load("cat_dog_svm.pkl") 
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return ("cat" if pred == 0 else "dog", proba)

# Test prediction
label, proba = predict_image("elephand.jpg")
print(f"Prediction: {label} (confidence: {max(proba)*100:.2f}%)")
