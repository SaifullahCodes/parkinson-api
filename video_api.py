import os
import json
import time
from flask import Flask, request, jsonify
import google.generativeai as genai
import typing_extensions as typing

# ======================= CONFIGURATION =======================
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. NEW MODEL STRATEGY
# Primary: Gemini 2.0 Flash (Newest, Fast)
# Backup: Gemini 1.5 Pro (Slower but very smart, if 2.0 fails)
MODELS = ["models/gemini-2.0-flash-exp", "models/gemini-1.5-pro"]
current_model_index = 0

# READ KEYS
API_KEYS = [
    os.environ.get("API_KEY_1"),
    os.environ.get("API_KEY_2"),
    os.environ.get("API_KEY_3"),
    os.environ.get("API_KEY_4"),
    os.environ.get("API_KEY_5")
]
API_KEYS = [key for key in API_KEYS if key]

if not API_KEYS:
    print("⚠️ No Environment Variables found. Using hardcoded backup.")
    # You can leave this empty or put a backup key for local testing
    API_KEYS = ["PASTE_YOUR_BACKUP_KEY_HERE"] 

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
    global current_model_index
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
    
    # Try up to 20 times (Keys * Models)
    for attempt in range(20): 
        try:
            target_model = MODELS[current_model_index]
            print(f"🤖 Attempt {attempt+1}: Using {target_model}")
            
            model = genai.GenerativeModel(model_name=target_model)
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
            print(f"⚠️ Error: {error_msg}")
            
            # CASE 1: LIMIT HIT -> Switch Key
            if "429" in error_msg or "Quota" in error_msg or "503" in error_msg:
                switch_key()
                # If cycled all keys, switch MODEL
                if attempt % len(API_KEYS) == (len(API_KEYS) - 1):
                    current_model_index = (current_model_index + 1) % len(MODELS)
                    print(f"🔄 Switching Model to: {MODELS[current_model_index]}")
                time.sleep(1)

            # CASE 2: MODEL NOT FOUND -> Switch Model Immediately
            elif "404" in error_msg or "not found" in error_msg.lower():
                print(f"❌ Model {target_model} not found. Switching...")
                current_model_index = (current_model_index + 1) % len(MODELS)
            
            else:
                return {"error": f"Analysis error: {error_msg}"}

    return {"error": "Server is busy. Please try again in 1 minute."}

# =======================================================
# 👇 ROUTES
# =======================================================
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # 1. Health Check
    if request.method == 'GET':
        return jsonify({
            "status": "Live", 
            "service": "Parkinson Video API",
            "endpoints": ["/models"],
            "message": "Send a POST request with a 'file' to analyze."
        }), 200

    # 2. Upload Logic
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

# ✅ NEW DEBUG ROUTE: Shows exactly which models work!
@app.route('/models', methods=['GET'])
def list_models():
    try:
        model_list = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_list.append(m.name)
        return jsonify({"available_models": model_list})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
