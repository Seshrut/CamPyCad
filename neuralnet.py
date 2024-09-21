import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# Load images and extract features
def load_images_and_extract_features(image_dir):
    images = []
    labels = []
    for class_label in range(1, 11):  # Assuming 10 classes
        class_dir = f"{image_dir}/class_{class_label}"
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = cv2.imread(image_path)
            # Extract features (e.g., using HOG)
            features = extract_hog_features(img)
            images.append(features)
            labels.append(class_label)
    return np.array(images), np.array(labels)

# Extract HOG features (replace with your own feature extraction method)
def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten()

# Load images and extract features
image_dir = "path/to/your/image/directory"
X, y = load_images_and_extract_features(image_dir)

# Reshape data if necessary
if len(X.shape) == 4:
    X = X.reshape(X.shape[0], -1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create MLP classifier with Adam optimizer
clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(100, 50), random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

joblib.dump(clf,'neuralnetmodel.pkl')

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

