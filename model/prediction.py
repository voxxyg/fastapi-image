from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions

import uuid
from firebase_admin import credentials, firestore, storage
import firebase_admin
from google.cloud import storage as gcs

firebase_admin.initialize_app(options={
    'storageBucket': 'coba-upload-foto'
})
db = firestore.client()
bucket = storage.bucket()

model = None

def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model

def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    result = decode_predictions(model.predict(image), 2)[0]

    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"

        response.append(resp)

    return response

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def upload_to_gcs(image_data, filename):
    blob = bucket.blob(filename)
    blob.upload_from_string(
        image_data,
        content_type='image/jpeg'
    )
    blob.make_public()
    return blob.public_url
