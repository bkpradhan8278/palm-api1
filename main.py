from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = FastAPI()

# --- OpenAI API key (env var for security) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in Render dashboard!

# --- Firebase Admin setup (service account JSON file path) ---
# Use the Secret File path you set in Render
FIREBASE_SECRET_PATH = "/etc/secrets/serviceAccount"
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_SECRET_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

class PalmRequest(BaseModel):
    image_base64: str      # Palm image (base64 string)
    user_id: str = None    # Optional, for user-specific records

def generate_gpt4o_report(base64_img, openai_api_key):
    openai.api_key = openai_api_key
    prompt = (
        "You are a world-class palmistry expert. "
        "Analyze the attached palm image and provide a detailed palmistry reading, "
        "including personality, health, career, relationships, and unique features."
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a palmistry expert AI."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
                    ]
                }
            ],
            max_tokens=900
        )
        return completion['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenAI API Error: {e}"

@app.post("/predict_palm")
async def predict_palm(request: PalmRequest):
    print("Incoming request keys:", request.dict().keys())
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key not set.")
    # 1. Get GPT-4o prediction
    report = generate_gpt4o_report(request.image_base64, OPENAI_API_KEY)
    # 2. Save to Firestore
    palm_doc = {
        "user_id": request.user_id or "anonymous",
        "image_base64": request.image_base64[:30] + "...",  # Don't store full image to save space!
        "prediction": report,
        "timestamp": datetime.utcnow(),
    }
    db.collection("palm_readings").add(palm_doc)
    # 3. Return response
    return {"prediction": report}
