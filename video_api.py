import os
import json
import time
from flask import Flask, request, jsonify
import google.generativeai as genai
import typing_extensions as typing

# ======================= CONFIGURATION =======================
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# TARGET MODEL
TARGET_MODEL = "models/gemini-2.5-flash"

# READ KEYS FROM RENDER SETTINGS (Environment Variables)
API_KEYS = [
    os.environ.get("API_KEY_1"),
    os.environ.get("API_KEY_2"),
    os.environ.get("API_KEY_3"),
    os.environ.get("API_KEY_4"),
    os.environ.get("API_KEY_5")
]

# Filter out empty keys (in case you only add 3 instead of 5)
API_KEYS = [key for key in API_KEYS if key]

if not API_KEYS:
    # Fallback for local testing if env vars are missing
    print("⚠️ No Environment Variables found. Using hardcoded backup.")
    API_KEYS = ["PASTE_YOUR_BACKUP_KEY_HERE"] 
# =============================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_key_index = 0
genai.configure(api_key=API_KEYS[0])

class ParkinsonAnalysis(typing.TypedDict):
    parkinson_probability: int
    freezing_percentage: float
    bradykinesia_score: int
    freezing_score: int
    variability_score: int
    reasoning: str
    clinical_interpretation: str
    recommendation: str

def switch_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    new_key = API_KEYS[current_key_index]
    genai.configure(api_key=new_key)
    print(f"🔑 Limit hit. Switched to Key #{current_key_index + 1}")

def analyze_video_logic(video_path):
    global TARGET_MODEL 
    print(f"🎬 Processing: {video_path}")
    
    # 1. Upload
    try:
        print("🚀 Uploading to Gemini...")
        # Force MP4 Mime Type
        video_file = genai.upload_file(path=video_path, mime_type="video/mp4")
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

    # 2. Wait
    print("⏳ Waiting for processing...")
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        return {"error": "Video processing failed on Google servers."}

    # 3. Analyze Loop
    prompt = """
    You are an expert Neurologist. Analyze gait for Parkinson's.
    Evaluate: Arm Swing, Stride Length, Turning Hesitation.
    Return JSON: parkinson_probability (int), freezing_percentage (float), 
    bradykinesia_score (0-3), freezing_score (0-3), variability_score (0-3), 
    reasoning (str), clinical_interpretation (str), recommendation (str).
    """
    
    for attempt in range(10): 
        try:
            model = genai.GenerativeModel(model_name=TARGET_MODEL)
            result = model.generate_content(
                [video_file, prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", 
                    response_schema=ParkinsonAnalysis,
                    temperature=0.0 
                ),
            )
            data = json.loads(result.text)
            genai.delete_file(video_file.name)
            return data

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota" in error_msg:
                switch_key()
                time.sleep(1)
            elif "404" in error_msg:
                print(f"⚠️ Model {TARGET_MODEL} not found. Trying 1.5-flash...")
                TARGET_MODEL = "models/gemini-1.5-flash"
            else:
                return {"error": f"Analysis error: {error_msg}"}

    return {"error": "All API keys exhausted."}

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_video_logic(filepath)
        
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except: pass
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)