import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
st.title("Leaf Disease Classification")

file = st.file_uploader("Upload Image", type=["jpg", "png"])

model = tf.keras.models.load_model('myModel.hdf5')

CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    nimg = np.array(image)
    img_array = tf.expand_dims(nimg,0)
    st.image(image, use_column_width=True)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    st.write(predicted_class)
    st.write(confidence)
