# 🧠 Parkinson’s Disease Detection API (Flask + Gemini AI)

> *The backend service for the Parkinson's Detection Android App, handling ML inference and Generative AI analysis.*

## 🚀 Overview
This repository contains the Python-based REST API built using **Flask**. It serves as the processing engine for the mobile application, handling two main tasks:
1.  **Audio Analysis:** Processes voice recordings using a custom ML model to detect vocal tremors.
2.  **Video Analysis:** Processes gait/walking videos using **Google Gemini 1.5 Pro API** for generative analysis.

---

## 🛠️ Tech Stack
- **Framework:** Flask (Python)
- **AI/LLM:** Google Gemini 1.5 Pro API
- **Machine Learning:** Custom Model (Pickle/TensorFlow)
- **Deployment:** Render / Hugging Face Spaces
- **Dependencies:** `flask`, `google-generativeai`, `numpy`, `librosa`

---

## 🔌 API Endpoints

### 1️⃣ Audio Analysis
**Endpoint:** `POST /predict-audio`
- **Description:** Accepts an audio file, extracts MFCC features, and runs it through the ML model.
- **Input:** Form-data (`file`: audio.wav)
- **Response:**
  ```json
  {
    "status": "success",
    "prediction": "Parkinson's Detected",
    "confidence": 0.89
  }
