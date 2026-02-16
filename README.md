# 🐶🐱 Cat vs Dog Classification using Multi-Layer Perceptron (MLP)

## 📌 Project Overview

This project implements **Binary Image Classification** (Cat vs Dog) using:

* Multi-Layer Perceptron (MLP)
* Backpropagation
* TensorFlow/Keras
* OpenCV for image preprocessing

Unlike CNN-based approaches, this model uses a fully connected neural network (MLP) trained on flattened image pixels.

---

## 🎯 Objective

1. Download Cat vs Dog dataset from Kaggle
2. Preprocess images
3. Build a dynamic MLP model
4. Train using Backpropagation
5. Save trained model
6. Test on a manually downloaded dog image

---

## 📂 Project Structure

```
Dogs-vs-Cats/
│
├── data/
│   ├── raw/
│   │   └── train/           # Original Kaggle dataset
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       └── my_dog.jpg
│
├── model/
│   └── cat_dog_mlp.h5
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd Dogs-vs-Cats
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Dataset used: Kaggle Dogs vs Cats dataset.

Original structure:

```
train/
   cat.0.jpg
   dog.0.jpg
```

We reorganize into:

```
train/
   cats/
   dogs/
```

---

## 🔄 Data Preprocessing

* Convert image to grayscale
* Resize to 64×64
* Normalize pixel values (0–1)
* Flatten image to 4096 features

---

## 🧠 Model Architecture

Dynamic MLP Model:

* Input Layer: 4096 neurons (64×64)
* Hidden Layers: User-defined
* Output Layer: 1 neuron (Sigmoid activation)

Example configuration:

```
Hidden Layers: 3
Neurons: 64 → 32 → 16
```

---

## 🔥 Training

Run:

```bash
cd src
python train.py
```

The program will ask:

```
Enter number of hidden layers:
Enter number of neurons for each layer:
```

Training uses:

* Adam Optimizer
* Binary Crossentropy Loss
* Backpropagation

---

## 💾 Saving Model

After training:

```
model/cat_dog_mlp.h5
```

---

## 🐕 Testing on New Image

1. Place image inside:

```
data/test/
```

2. Run:

```bash
python predict.py
```

Example Output:

```
Raw Prediction Value: 0.489
Prediction: CAT
```

If value > 0.5 → DOG
If value ≤ 0.5 → CAT

---

## 📈 Results

* Training Accuracy: ~63%
* Test Accuracy: ~60%

---

## 👩‍💻 Author
Khushi Goyal