import numpy as np
import streamlit as st
import keras
import io
from PIL import Image

class_names = ['Airplane','Horse','Truck','Automobile','Ship','Dog','Bird','Frog','Cat','Deer']
model = keras.models.load_model("model/CIFAR-10.keras")

st.set_page_config(page_title="CIFAR_10_Classification", layout="centered")
st.title("CIFAR_10_Classification")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    image_data = uploaded_file.read()
    image_dis = Image.open(io.BytesIO(image_data))
    image_dis = np.array([image_dis.resize((500, 500))])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image_dis, caption="Uploaded Image")
    pil_image = Image.open(io.BytesIO(image_data))
    image_resized = np.array([pil_image.resize((32, 32))])
    image_resized = image_resized/255
    result = model.predict(image_resized)
    pred = np.argmax(result, axis=1)
    st.write("<center>" + class_names[pred[0]] + "</center>", unsafe_allow_html=True)