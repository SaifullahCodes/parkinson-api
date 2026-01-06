from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Parkinson's Voice Detection API")

# --- 1. CONFIGURATION (CLOUD COMPATIBLE) ---
# This finds the directory where api.py is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This looks for the model file in the SAME folder as api.py
MODEL_PATH = os.path.join(BASE_DIR, "parkinsons_mfcc_model.h5")

# --- 2. LOAD MODEL ---
print(f"Loading model from: {MODEL_PATH}")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # We don't exit here on cloud, just print error to logs
    model = None

# --- 3. EXACT SCALER VALUES (High Precision) ---
# ⚠️ CRITICAL: Matches training data exactly.
HARDCODED_MEAN = np.array([
    -233.23172052589382, 208.9925267215066, -69.96216482119941, -17.912778577080843, 0.9710564632231081, 
    -39.03255755380845, 13.329869740841552, 9.818470685097866, -26.96551459534487, 4.256489320346613, 
    3.4151535812943536, -15.043113048586278, 2.5658866759726315, -6.564097938704974, -15.370565869618085, 
    -0.03985351861996725, -7.598832500858031, -9.147705943441727, 1.3806584800501736, -7.7013746950285675, 
    -6.066502381758076, 1.106044023566357, -7.277421055647133, -3.6192629746611757, 0.3223162010889238, 
    -6.4588717895222745, -1.1920983581667102, 0.3750919050809232, -4.3454433630806495, 1.92989537198842, 
    1.527214350487639, -2.041173422119379, 3.4903673890293816, 1.2663787664091892, -0.5768835478103741, 
    4.842929320660215, 1.4795529401118914, -0.07482238009816884, 3.32468245362758, -0.48629802894918467
])

HARDCODED_SCALE = np.array([
    43.79166776107933, 25.098320242328658, 27.47236977717347, 15.41126449698062, 15.299606277764694, 
    14.548644937262765, 14.348055886988748, 11.423489137250344, 9.417242242201539, 11.034930039304104, 
    7.485101877508501, 9.069648775752773, 8.541214502558583, 8.54362532566373, 6.630800975143475, 
    7.903991772197426, 6.610877977767781, 6.178388938940979, 7.3237717680231675, 6.243125204829299, 
    5.442014671860252, 6.401086915835411, 6.690410390984811, 6.897311014088648, 8.425743231792376, 
    8.799747829109947, 8.792460199388715, 9.057045191365566, 10.163408456892999, 11.949173669053183, 
    12.049259701192712, 11.471112944674381, 12.211991570661157, 12.815409320577174, 12.542136505839315, 
    12.36693117298452, 12.260388661722384, 12.100840964436408, 11.785625572043537, 11.262541953940874
])

# --- 4. FEATURE EXTRACTION ---
def extract_mfcc(file_path, n_mfcc=40, duration=5, offset=0.5):
    # Load audio (sr=22050 to match model logic)
    y, sr = librosa.load(file_path, sr=22050, duration=duration, offset=offset)
    
    # Extract MFCCs (Using 40 to match model, n_fft=2048, hop=512)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    
    # Mean across time
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# --- 5. PREDICTION ---
def predict_file(file_path):
    if model is None:
        return {"error": "Model not loaded"}

    # 1. Extract
    features = extract_mfcc(file_path)
    
    # 2. Normalize (Manual Calculation)
    safe_scale = np.where(HARDCODED_SCALE == 0, 1, HARDCODED_SCALE)
    features_normalized = (features - HARDCODED_MEAN) / safe_scale
    
    # 3. Reshape for Model (1, 40)
    features_normalized = features_normalized.reshape(1, -1)
    
    # 4. Predict
    prob = model.predict(features_normalized)[0][0]
    label = "Parkinson's" if prob > 0.5 else "Healthy"
    
    return {
        "probability": float(prob),
        "prediction": label,
        "raw_features": features.tolist()[:5] # Debugging info
    }

# --- 6. API ENDPOINT ---
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    tmp_path = f"temp_{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        
        result = predict_file(tmp_path)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 7. RUN SERVER ---
if __name__ == "__main__":
    # Get port from environment variable (Required for Cloud)
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
