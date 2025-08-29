import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('app/final1_model.h5')


# Function to preprocess the uploaded image and make predictions
def predict_blood_cancer(image):
    # Preprocess the image
    img = image.resize((224, 224))  # Resize to match input size of the model
    img_arr = np.array(img) / 255.0  # Normalize pixel values
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_arr)
    return prediction

# Set background image
st.markdown(
    """
    <style>
    body {
        background-image: url("https://example.com/background_image.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title('Blood Cancer Detection')

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
