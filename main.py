from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import mediapipe as mp
import openai
import os

app = FastAPI()
mp_hands = mp.solutions.hands

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")  # Fallback for local/dev

class PalmRequest(BaseModel):
    image: str  # base64-encoded palm image

def extract_hand_landmarks(base64_img):
    image_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img_np = np.array(img)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(img_np)
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
        return coords

def generate_gpt4o_report(base64_img, landmarks, openai_api_key):
    openai.api_key = openai_api_key
    prompt = f"""You are a world-class palmistry expert.
Analyze the following hand:
- Landmark coordinates: {landmarks}
Based on the palm image and the detected hand geometry, provide a detailed palmistry reading including personality, health, career, relationships, and unique features."""
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

@app.post("/predict_palm")
async def predict_palm(request: PalmRequest):
    landmarks = extract_hand_landmarks(request.image)
    if not landmarks:
        return {"prediction": "No hand detected. Please upload a clear palm image."}
    try:
        report = generate_gpt4o_report(request.image, landmarks, OPENAI_API_KEY)
        return {"prediction": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
