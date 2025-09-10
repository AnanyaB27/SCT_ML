import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Paths - set to your extracted folder
cat_dir = 'data/Cat'
dog_dir = 'data/Dog'
img_size = (64, 64) # SVM works best with low-res/fewer features

def load_images(folder, label):
    features, labels = [], []
    for file in os.listdir(folder):
        if file.endswith('.jpg'):
            img = imread(os.path.join(folder, file), as_gray=True)
            img = resize(img, img_size, anti_aliasing=True)
            features.append(img.flatten())
            labels.append(label)
    return features, labels

# Load cats and dogs
X_cat, y_cat = load_images(cat_dir, 0)
X_dog, y_dog = load_images(dog_dir, 1)

# Combine
X = np.array(X_cat + X_dog)
y = np.array(y_cat + y_dog)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
