from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os

from predict import predict as pr

app = FastAPI(title="Dog Breed Classifier API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Dog grtyy API is healthy 🐶"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an uploaded image file and returns the predicted dog breed.
    """
    # Save the uploaded file temporarily
    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4().hex}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict
    results = pr(UPLOAD_DIR, checkpoint_path='checkpoints/saved_model.pth')

    # Remove temp file
    os.remove(temp_path)

    return {"prediction": results}
