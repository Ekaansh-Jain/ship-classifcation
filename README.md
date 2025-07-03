# 🚢 Ship Classification using VGG16 (Flask + Deep Learning)

This project classifies ship images using a convolutional neural network based on **VGG16**. It is developed in two parts:

- 🧠 A Jupyter Notebook to train the deep learning model  
- 🌐 A Flask Web App to load the model and classify images

---

## 📁 Project Structure

```
├── Ship_Classification.ipynb     ← Training notebook
├── app.py                        ← Flask app (frontend + backend)
├── requirements.txt              ← Required Python packages
├── models/
│   └── README.txt                ← Info about the large model file
└── README.md                     ← This file
```

---

## 🧠 Model Overview

- **Base Model**: VGG16 (pretrained on ImageNet, frozen)
- **Custom Layers**: Flatten → Dense → Dropout → Output
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Accuracy**: *around XX% on validation set* (fill in)
- **Techniques Used**:
  - Image Augmentation (`ImageDataGenerator`)
  - Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`

---

## ⚙️ How to Run the Flask App

> Prerequisite: Python 3.8+ and the `.h5` model file (downloaded separately)

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ship-classification.git
   cd ship-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model file `best_vgg16_model.h5` from the link below and place it inside the `models/` folder:
   📁 [Insert Google Drive link here]

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ⚠️ Model File Note

The trained model (`best_vgg16_model.h5`) is about **2GB** and cannot be uploaded to GitHub.  
Please download it from the provided Google Drive link and place it in the `models/` folder manually.

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Flask
- HTML + CSS (inline in `app.py`)
- VS Code

---

## ✅ Status

- [x] Model Trained
- [x] Flask App Built
- [x] GitHub Setup Complete
- [ ] Model Uploaded to Google Drive (TBD)
