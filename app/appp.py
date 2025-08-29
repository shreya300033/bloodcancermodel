import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = load_model('final1_model.h5')

# Define function to make predictions
def predict_blood_cancer(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize to match input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    return prediction

# Streamlit UI
st.title('Blood Cancer Detection Web App')

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict_blood_cancer(image)

    # Display prediction result
    if prediction[0][0] > 0.5:
        st.write('Prediction: Blood cancer detected')
    else:
        st.write('Prediction: No blood cancer detected')
