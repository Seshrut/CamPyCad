import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2


data_dir = "photos/dataset_final"
class_names = [labels for labels in os.listdir(data_dir)]

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
        print(f'{class_name} defined')

    return np.array(images), np.array(labels)

print('loading data')
X, y = load_data(data_dir)
print('data loaded, now spliting')
X = X.reshape(X.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('settings for model')
clf = SVC(kernel='linear')  # Experiment with different kernels (e.g., 'rbf', 'poly')
print('started training')
clf.fit(X_train, y_train)
print('testing')
y_pred = clf.predict(X_test)
print('calculating acuracy')
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(clf,"svm_model_scikit.pkl")