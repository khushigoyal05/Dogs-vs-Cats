import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64

# Load trained model
model = load_model("../model/cat_dog_mlp.h5")

# Load test image
img_path = "../data/test/my_dog.jpg"   # Change name if needed

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = img.flatten().reshape(1, -1)

prediction = model.predict(img)

print("Raw Prediction Value:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Prediction: DOG 🐶")
else:
    print("Prediction: CAT 🐱")
