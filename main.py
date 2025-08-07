from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import openai
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import base64

app = FastAPI()

# --- OpenAI API key (env var for security) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in Render dashboard!

# --- Firebase Admin setup (service account JSON file path) ---
FIREBASE_SECRET_PATH = "/etc/secrets/serviceAccount"
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_SECRET_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

class PalmRequest(BaseModel):
    image_base64: str
    user_id: str = None

def generate_gpt4o_report(base64_img, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = (
        "You are a world-class palmistry expert. "
        "Analyze the attached palm image and provide a detailed palmistry reading, "
        "including personality, health, career, relationships, and unique features."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a palmistry expert AI."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]
                }
            ],
            max_tokens=900
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {e}"

@app.post("/predict_palm")
async def predict_palm(request: PalmRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key not set.")
    report = generate_gpt4o_report(request.image_base64, OPENAI_API_KEY)
    palm_doc = {
        "user_id": request.user_id or "anonymous",
        "image_base64": request.image_base64[:30] + "...",
        "prediction": report,
        "timestamp": datetime.utcnow(),
    }
    db.collection("palm_readings").add(palm_doc)
    return {"prediction": report}

# ------------ Kundali Feature Below ------------

def generate_gpt4o_kundali(prompt_txt, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = (
        "You are a Vedic astrologer and kundali analysis expert. "
        "Analyze the following details or PDF image and give a detailed Vedic astrology report, including personality, career, marriage, wealth, health, and predictions.\n"
        "Details/PDF: " + prompt_txt
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Vedic astrologer and Kundali analysis AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=950
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {e}"

@app.post("/predict_kundli")
async def predict_kundli(
    name: str = Form(None),
    dob: str = Form(None),
    tob: str = Form(None),
    place: str = Form(None),
    user_id: str = Form(None),
    file: UploadFile = File(None)
):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key not set.")

    if file is not None:
        # Handle PDF or Image
        file_content = await file.read()
        try:
            # If image, encode as base64 for GPT
            if file.content_type.startswith("image/"):
                b64_img = base64.b64encode(file_content).decode()
                prompt_txt = f"User: {name}\nUploaded Kundali image below."
                # You can modify to send as image_url in OpenAI if using vision model
                # For now, we just describe it.
                result = generate_gpt4o_kundali(prompt_txt, OPENAI_API_KEY)
            elif file.content_type == "application/pdf":
                prompt_txt = f"User: {name}\nUploaded Kundali PDF. (Cannot process PDF image directly here, but mention file upload.)"
                result = generate_gpt4o_kundali(prompt_txt, OPENAI_API_KEY)
            else:
                raise HTTPException(status_code=400, detail="Only PDF or Image supported.")
        except Exception as e:
            result = f"Kundali analysis failed: {e}"

        # Save to Firestore
        kundali_doc = {
            "user_id": user_id or "anonymous",
            "name": name or "",
            "input_type": "file",
            "filename": file.filename,
            "prediction": result,
            "timestamp": datetime.utcnow(),
        }
        db.collection("kundali_readings").add(kundali_doc)
        return {"analysis": result}

    elif name and dob and tob and place:
        # Handle birth details
        prompt_txt = f"Name: {name}\nDOB: {dob}\nTOB: {tob}\nPlace: {place}"
        result = generate_gpt4o_kundali(prompt_txt, OPENAI_API_KEY)
        kundali_doc = {
            "user_id": user_id or "anonymous",
            "name": name,
            "dob": dob,
            "tob": tob,
            "place": place,
            "input_type": "details",
            "prediction": result,
            "timestamp": datetime.utcnow(),
        }
        db.collection("kundali_readings").add(kundali_doc)
        return {"analysis": result}

    else:
        raise HTTPException(status_code=400, detail="Provide either all fields (name, dob, tob, place) or a file.")
