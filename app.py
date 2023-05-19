from __future__ import division
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from io import BytesIO


#Loading the Model
model = load_model('classify.h5', compile=False)
st.write("""
# Image Classifier"""
)
#Uploading Image
upload_image = st.file_uploader("Upload an image from your computer", type=['png','jpg','jpeg','webp'])
submit = st.button('Classify Image')
if submit:

    if upload_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)
        opencv_image = cv2.resize(opencv_image, (224,224))
        opencv_image.shape = (1,224,224,3)
        pred = model.predict(opencv_image)
        pred_classes = resnet50.decode_predictions(pred, top=4)
        # Displaying the image
        st.markdown("""
        Results: """
        )
        for imagenet_id, name, likelihood in pred_classes[0]:
            st.text('• {} with {:.0%} probability'.format(name,likelihood))
