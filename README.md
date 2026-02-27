# 🖐️ Hand Gesture Classification Using MediaPipe Landmarks

A machine learning project that classifies **18 hand gestures** in real time using MediaPipe hand landmarks, geometric feature engineering, and classic ML models.

---

## 📋 Project Overview

| Item | Details |
|---|---|
| **Dataset** | HaGRID — 25,675 samples, 18 gesture classes |
| **Input** | 63 raw landmark features (x, y, z × 21 landmarks) |
| **Engineered Features** | 231 geometric features (pairwise distances + direction signs) |
| **Best Model** | SVM (Linear kernel) — ~99% accuracy |
| **Inference** | Live webcam + recorded video |

---

## 🤌 Gesture Classes

| | | | |
|---|---|---|---|
| call | dislike | fist | four |
| like | mute | ok | one |
| palm | peace | peace_inverted | rock |
| stop | stop_inverted | three | three2 |
| two_up | two_up_inverted | | |

---

## 🗂️ Project Structure

```
📦 hand-gesture-classification
 ├── Hand_Gesture_Classification.ipynb   ← Main Colab notebook
 ├── README.md                           ← This file
 └── hagrid_model/
     ├── best_model.pkl                  ← Trained SVM model
     ├── scaler.pkl                      ← Fitted StandardScaler
     ├── pca.pkl                         ← Fitted PCA (35 components)
     └── label_encoder.pkl               ← Label encoder (18 classes)
```

---

## ⚙️ Workflow

```
Raw CSV (63 features)
       ↓
Feature Engineering → 231 features
  • 210 pairwise Euclidean distances (XY plane)
  •  21 Y-direction signs relative to wrist
       ↓
StandardScaler → zero mean, unit variance
       ↓
PCA → 35 components
       ↓
Train & Compare 4 Models
  • Random Forest
  • SVM ⭐
  • KNN
  • XGBoost
       ↓
Evaluate → Confusion Matrix + Classification Report
       ↓
Live Webcam / Video Inference
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **SVM** ⭐ | ~99% | ~99% | ~99% | ~99% |
| Random Forest | ~98% | ~98% | ~98% | ~98% |
| XGBoost | ~97% | ~97% | ~97% | ~97% |
| KNN | ~96% | ~96% | ~96% | ~96% |

---

## 🔧 Feature Engineering

Instead of using raw x/y/z coordinates (which vary with hand position and distance from camera), we extract **position and scale invariant** features:

**1. Pairwise Euclidean Distances (210 features)**
```
All C(21,2) = 210 distances between every pair of the 21 landmarks on the XY plane
→ Captures hand shape regardless of position or scale
```

**2. Y-Direction Signs (21 features)**
```
sign(landmark_y - wrist_y) for each of the 21 landmarks
→ Captures whether each landmark is above (+1) or below (-1) the wrist
→ Helps distinguish gestures with similar shapes but different finger directions
```

---

## 🚀 How to Run

### On Google Colab
1. Upload `hagrid_landmarks.csv` to your Google Drive
2. Open `Hand_Gesture_Classification.ipynb` in Colab
3. Update `CSV_PATH` and `SAVE_DIR` to your Drive paths
4. Run all cells top to bottom

### Webcam (Local)
```python
# Requires: opencv-python, mediapipe, joblib, scipy
# Run Step 9 of the notebook locally
# Press Q to quit
```

### Video File
```python
VIDEO_INPUT  = r'path\to\sample_video.mp4'
VIDEO_OUTPUT = r'path\to\gesture_output.mp4'
process_video(VIDEO_INPUT, VIDEO_OUTPUT, window_size=15)
```

---

## 📦 Dependencies

```bash
pip install mediapipe xgboost scikit-learn scipy numpy pandas matplotlib seaborn joblib opencv-python
```

---

## 🎥 Demo Video

> 📎 [Watch the output video on Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)

---

## 📌 Key Takeaways

- Geometric features (distances + direction signs) significantly outperform raw coordinates
- PCA reduces 231 → 35 features with minimal accuracy loss and faster training
- The exact same preprocessing must be applied at inference time
- Mode smoothing over a 15-frame window removes prediction flickering in live demos
- SVM with a linear kernel is the best fit for this structured geometric feature space

---

## 👤 Author

**Omar**  
Hand Gesture Classification Project — HaGRID Dataset
