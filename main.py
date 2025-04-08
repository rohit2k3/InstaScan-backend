import os
from fastapi import Body, FastAPI,  HTTPException
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import uvicorn
import requests

app = FastAPI()

# Load Models Separately
MODEL_PATH_SKIN = "models/skin_cancer_model.tflite"
MODEL_PATH_PNEUMONIA = "models/Pneumonia_Detection.tflite"

def load_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Model loaded successfully: {model_path}")
        return interpreter, input_details, output_details
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Load models separately
skin_interpreter, skin_input_details, skin_output_details = load_model(MODEL_PATH_SKIN)
pneumonia_interpreter, pneumonia_input_details, pneumonia_output_details = load_model(MODEL_PATH_PNEUMONIA)


def fetch_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or open image: {str(e)}")


# Preprocessing function
def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "InstaScan API is running!"}

# Pneumonia Detection Route (Unchanged)
@app.post("/predictPneumonia")
async def predict2(data: dict = Body(...)):
    try:
        url = data.get("image_url")
        if not url:
            raise HTTPException(status_code=400, detail="Missing image_url in request body")

        img = fetch_image_from_url(url)

        image = preprocess_image(img)
        pneumonia_interpreter.set_tensor(pneumonia_input_details[0]['index'], image)
        pneumonia_interpreter.invoke()
        prediction = pneumonia_interpreter.get_tensor(pneumonia_output_details[0]['index'])[0][0] * 100

        return {"disease-confidence": f"{prediction:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Skin Cancer Prediction Route (Unchanged)
@app.post("/predictSkinCancer")
async def predict(data: dict = Body(...)):
    try:
        url = data.get("image_url")
        if not url:
            raise HTTPException(status_code=400, detail="Missing image_url in request body")

        img = fetch_image_from_url(url)
        input_data = preprocess_image(img)

        if input_data.shape != tuple(skin_input_details[0]['shape']):
            raise HTTPException(status_code=400, detail="Invalid input shape")

        skin_interpreter.set_tensor(skin_input_details[0]['index'], input_data)
        skin_interpreter.invoke()
        predictions = skin_interpreter.get_tensor(skin_output_details[0]['index'])[0]
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        pred_dict = dict(zip(class_names, predictions.tolist()))
        disease_confidence = (1 - pred_dict["nv"]) * 100

        return {"disease_confidence": f"{disease_confidence:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)