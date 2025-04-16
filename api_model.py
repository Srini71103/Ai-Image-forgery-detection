from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import google.generativeai as genai
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
import numpy as np
import shutil
import os
import datetime
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Image Forgery Detection API",
    description="API for detecting forged images using deep learning",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://image-ftd5oka17-srinivas46s-projects.vercel.app/",  # Your frontend Vercel domain
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Load the VGG model
try:
    # Download model from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id="Srinivas7113/IFD",
        filename="best_model.h5",
        token=os.getenv("HF_TOKEN")  # Add HF_TOKEN to your .env file
    )
    
    # Load the downloaded model
    full_model = load_model(model_path)
    vgg16_base = full_model.get_layer("vgg16")
    feature_extractor = Model(inputs=vgg16_base.input, outputs=vgg16_base.output)
except Exception as e:
    print(f"Error loading model from Hugging Face: {e}")
    raise

def analyze_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = {}
        
        if hasattr(image, '_getexif') and image._getexif():
            for tag_id in image._getexif():
                tag = TAGS.get(tag_id, tag_id)
                data = image._getexif().get(tag_id)
                
                # Convert bytes to string if necessary
                if isinstance(data, bytes):
                    try:
                        data = data.decode()
                    except:
                        data = str(data)
                        
                exif_data[tag] = data

            # Analyze suspicious patterns in EXIF
            suspicious_patterns = []
            
            # Check creation date inconsistencies
            if 'DateTime' in exif_data:
                try:
                    date = datetime.datetime.strptime(exif_data['DateTime'], '%Y:%m:%d %H:%M:%S')
                    if date > datetime.datetime.now():
                        suspicious_patterns.append("Future creation date detected")
                except:
                    suspicious_patterns.append("Invalid date format")

            # Check for software modification traces
            if 'Software' in exif_data:
                editing_software = ['photoshop', 'lightroom', 'gimp', 'illustrator']
                if any(sw.lower() in str(exif_data['Software']).lower() for sw in editing_software):
                    suspicious_patterns.append(f"Image edited with {exif_data['Software']}")

            return {
                "exif_data": exif_data,
                "suspicious_patterns": suspicious_patterns,
                "has_metadata": bool(exif_data)
            }
        
        return {
            "exif_data": {},
            "suspicious_patterns": ["No EXIF data found - possible metadata removal"],
            "has_metadata": False
        }
        
    except Exception as e:
        print(f"EXIF analysis error: {str(e)}")
        return {
            "exif_data": {},
            "suspicious_patterns": [f"Error analyzing EXIF: {str(e)}"],
            "has_metadata": False
        }

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Get both features and prediction
        features = feature_extractor.predict(x)
        prediction = full_model.predict(x)[0][0]  # Get binary prediction
        return features, prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

def get_gemini_analysis(image_path):
    try:
        img = Image.open(image_path)
        
        prompt = """Analyze this image forensically for signs of manipulation or AI generation. Focus on:
        1. Pixel-level inconsistencies or artifacts
        2. Lighting and shadow anomalies
        3. Edge irregularities or unnatural transitions
        4. Noise patterns and texture inconsistencies
        5. Color distribution abnormalities
        
        Provide a detailed technical analysis explaining why this image might be real or manipulated."""

        response = model.generate_content([prompt, img])
        print("Gemini analysis response:", response)
        return response.text
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")

def generate_explanation(features, prediction, gemini_analysis):
    try:
        # Convert numpy types to Python native types
        avg_activation = float(np.mean(features))
        max_activation = float(np.max(features))
        std_activation = float(np.std(features))
        feature_entropy = float(np.sum(-features * np.log2(features + 1e-10)))
        
        # Use binary classification (1=real, 0=fake)
        label = 1 if prediction >= 0.5 else 0
        classification = "AUTHENTIC" if label == 1 else "FAKE"
        base_explanation = ("✅ Model predicts this image is authentic (confidence: {:.1f}%)".format(prediction * 100) 
                          if label == 1 
                          else "🔍 Model detects this image as manipulated (confidence: {:.1f}%)".format((1-prediction) * 100))
        
        # Calculate confidence based on prediction probability
        confidence_score = float(abs(prediction - 0.5) * 200)  # Scale to 0-100
        confidence = "HIGH" if confidence_score > 70 else "MODERATE"
        
        # Enhanced explanation structure
        explanation = {
            "classification": classification,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "base_explanation": base_explanation,
            "feature_analysis": {
                "activation_patterns": {
                    "mean": float(avg_activation),
                    "max": float(max_activation),
                    "std": float(std_activation),
                    "entropy": float(feature_entropy)
                }
            },
            "vgg_analysis": {
                "summary": base_explanation,
                "metrics": {
                    "prediction_label": label,
                    "raw_prediction": float(prediction),
                    "probability_real": float(prediction),
                    "probability_fake": float(1 - prediction)
                }
            },
            "detailed_analysis": gemini_analysis,
            "visualization_data": {
                "scores": {
                    "authenticity": float(prediction * 100),
                    "manipulation": float((1 - prediction) * 100),
                    "confidence": confidence_score
                },
                "thresholds": {
                    "decision_boundary": 0.5,
                    "prediction": float(prediction)
                }
            }
        }
        
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_file_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get features, prediction and analysis
        features, prediction = extract_features(temp_file_path)
        gemini_analysis = get_gemini_analysis(temp_file_path)
        exif_analysis = analyze_exif_data(temp_file_path)
        
        explanation = generate_explanation(
            features, 
            prediction, 
            gemini_analysis
        )

        return JSONResponse(
            content={"success": True, "result": explanation},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
