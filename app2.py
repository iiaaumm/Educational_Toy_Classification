import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.models import load_model

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_saved_model(model_url):
    response = requests.get(model_url)
    model = load_model(BytesIO(response.content))
    return model

# Streamlit app
st.title("Toy Classification")

# URL of the model file in your GitHub repository
model_url = 'https://github.com/iiaaumm/Educational_Toy_Classification/blob/main/model_chunks/Toy_classification_10class.h5?raw=true'
model = load_saved_model(model_url)

# List of class labels
class_labels = ['Activity_Cube', 'Ball', 'Puzzle', 'Rubik', 'Tricycle', 'baby_walker', 'lego', 'poppet', 'rattle', 'stacking']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(img)

        # Perform inference to obtain predictions
        predictions = model.predict(img_array)
        
        # Get the predicted class label
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        # Display the predicted class label
        st.write(f"Predicted Class: {predicted_class_label}")

        # Display the probabilities for each class
        st.write("Class Probabilities:")
        for i, class_label in enumerate(class_labels):
            st.write(f"{class_label}: {predictions[0][i]}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
