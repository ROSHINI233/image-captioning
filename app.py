import gradio as gr
import numpy as np
import cv2
import pickle
import tensorflow as tf
from PIL import Image
import torch

from transformers import ViTImageProcessor, ViTModel
from ultralytics import YOLO
from tensorflow.keras.preprocessing.sequence import pad_sequences


print("Loading models...")

# load caption model
model = tf.keras.models.load_model("caption_model.keras")

# load tokenizer
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

# YOLO
yolo = YOLO("yolov8n.pt")

# ViT
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model.to(device)

max_length = 39


def extract_features(img):

    results = yolo(img)

    objects = []

    for cls in results[0].boxes.cls:
        label = results[0].names[int(cls)]
        objects.append(label)

    object_text = " ".join(list(set(objects)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = vit_processor(images=img_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = vit_model(**inputs)

    feature = outputs.last_hidden_state[:,0,:].cpu().numpy()[0]

    return feature, object_text


def generate_caption(image):

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    photo, objects = extract_features(img)

    in_text = "startseq " + objects

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo.reshape(1,-1), sequence], verbose=0)

        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    caption = in_text.replace("startseq","").replace("endseq","")

    return caption


interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Caption Generator",
    description="Upload an image to generate a caption."
)

interface.launch()