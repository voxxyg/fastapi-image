from fastapi import FastAPI
from fastapi import UploadFile, File
from starlette.responses import RedirectResponse
from model.prediction import predict, read_imagefile, upload_to_gcs, db, bucket
import uvicorn
import uuid
from firebase_admin import credentials, firestore, storage
import firebase_admin
from google.cloud import storage as gcs

app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.get('/')
def helloWorld(name: str):
    return f"Hello {name}"

@app.post("/predict/image")
async def predict_api(user_id: str, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    # Upload the image to GCS
    image_data = await file.read()
    image = read_imagefile(image_data)
    prediction = predict(image)

    filename = f"images/{user_id}/{file.filename}"
    url = upload_to_gcs(image_data, filename)
    
    # Save the result to Firestore
    doc_ref = db.collection('users').document(user_id).collection('foods-history').document()
    doc_ref.set({
        'imageUrl': url,
        'prediction': prediction
    })

    return {"status": "success", "url": url, "prediction": prediction}

if __name__ == '__main__':
    uvicorn.run(app, port= 8080, host="0.0.0.0")