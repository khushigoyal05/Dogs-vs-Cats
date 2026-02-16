import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 64
DATA_PATH = "../data/train"


# -----------------------------
# LOAD DATA
# -----------------------------
def load_images(folder, label):
    data = []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            data.append((img.flatten(), label))
        except:
            continue

    return data


print("Loading images...")

cats = load_images(os.path.join(DATA_PATH, "cats"), 0)
dogs = load_images(os.path.join(DATA_PATH, "dogs"), 1)

data = cats + dogs
np.random.shuffle(data)

X = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

print("Total samples:", len(X))


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# BUILD DYNAMIC MODEL
# -----------------------------
def build_model(input_size):

    model = Sequential()

    num_hidden_layers = int(input("Enter number of hidden layers: "))

    for i in range(num_hidden_layers):
        neurons = int(input(f"Enter number of neurons for hidden layer {i+1}: "))

        if i == 0:
            model.add(Dense(neurons, activation='relu', input_shape=(input_size,)))
        else:
            model.add(Dense(neurons, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model


model = build_model(IMG_SIZE * IMG_SIZE)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining model...\n")
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("\nEvaluating model...\n")
loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)
model.save("../model/cat_dog_mlp.h5")
print("Model saved successfully!")
