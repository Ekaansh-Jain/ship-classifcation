
---

# 🚢 Ship Classification using VGG16 (Flask + Deep Learning)

This project classifies ship images using a deep learning model based on **VGG16**. It consists of two main components:

* 🧠 A Google Colab to train the model
* 🌐 A Flask Web App to classify ship images using the trained model

---

## 📁 Project Structure

```
├── 1. Project Initialization and Planning Phase/
├── 2. Data Collection and Preprocessing Phase_/
├── 3. Model Development Phase/
├── 4. Model Optimization and Tuning Phase/
├── models/
│   └── best_vgg16_model.h5 (to be added manually)
├── notebook/
│   └── Ship_Classification.ipynb
├── screenshots/
│   └── Output images
├── static/
├── template/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Overview

* **Base Model**: VGG16 (pretrained on ImageNet, frozen)
* **Custom Layers**: Flatten → Dense → Dropout → Dense (Softmax)
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Accuracy**: *\~XX% on validation set* (replace with actual value)
* **Training Enhancements**:

  * Data Augmentation via `ImageDataGenerator`
  * Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`

---

## ⚙️ Running the Flask Web App

> **Pre-requisite:** Python 3.8+ and the trained `.h5` model file

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ship-classification.git
cd ship-classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Model File

Download `best_vgg16_model.h5` from the link below and place it in the `models/` folder:


### Step 4: Run the Flask App

```bash
python app.py
```

### Step 5: Access in Browser

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ⚠️ Note on Model File

The trained model (`best_vgg16_model.h5`) is approximately **2GB** and cannot be pushed to GitHub.
Please download it manually using the link provided above and place it in the `models/` directory.

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* Flask
* HTML + CSS (used in templates)
* Jupyter Notebook
* VS Code

---

## ✅ Project Status

* [x] Model Trained
* [x] Flask App Implemented
* [x] Project Repository Structured
* [x] Demo Video Uploaded

---

