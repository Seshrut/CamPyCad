import cv2
import numpy as np
import os

# Define data directory and class names
data_dir = "path/to/your/dataset"
class_names = ["class0", "class1", ..., "class7"]

# Load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(class_names.index(class_name))
    return np.array(images), np.array(labels)

images, labels = load_data(data_dir)

# Preprocess images (e.g., resize, normalize)
images = [cv2.resize(image, (224, 224)) for image in images]  # Adjust size as needed
images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])  # Convert to grayscale if needed
images = images.reshape(len(images), -1)  # Flatten images

svm = cv2.ml.SVM.cr
svm.setType(cv2.ml.SVM_C_SVC)  # Set SVM type
svm.setKernel(cv2.ml.SVM_LINEAR)  # Set kernel type (adjust as needed)
svm.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3))  # Set termination criteria

svm.train(images, cv2.ml.ROW_SAMPLE, labels)

svm.save("svm_model.xml")