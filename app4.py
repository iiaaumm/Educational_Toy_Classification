import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import datetime

# Function to download the model file
def download_model_file(model_url, model_path):
    with requests.get(model_url, stream=True) as response:
        response.raise_for_status()
        with open(model_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)

# Function to load the model
@st.cache_resource
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Custom CSS for headers with Google Fonts (Noto Sans Lao Looped)
header_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao+Looped&display=swap');
        .header-text {
            font-family: 'Noto Sans Lao Looped', sans-serif;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 20px;
        }
        .prediction-box {
            font-family: 'Noto Sans Lao Looped', sans-serif;
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """
st.markdown(header_style, unsafe_allow_html=True)
st.markdown("<p class='header-text'>ລະບົບການຈໍາແນກເຄື່ອງຫຼິ້ນເສີມທັກສະຂອງເດັກນ້ອຍດ້ວຍເຕັກນິກ CNN</p>", unsafe_allow_html=True)
st.markdown("<p class='header-text'>Classification of Children Toys Using CNN</p>", unsafe_allow_html=True)

# URL of the model file in your GitHub repository
model_url = 'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/Toy_classification_10class.h5'
model_path = './Toy_classification_10class.h5'

# Download the model file
try:
    download_model_file(model_url, model_path)
    st.write("Model downloaded successfully.")
except Exception as e:
    st.error(f"Unable to download the model file: {e}")

# Loading the model
model = load_saved_model(model_path)

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

# Sidebar with various components
with st.sidebar:
    st.header("Settings")
    selected_label = st.selectbox("Choose a class label", class_labels)
    selected_date = st.date_input("Select a date")
    selected_date_range = st.date_input("Select date range", [datetime.date.today(), datetime.date.today() + datetime.timedelta(days=1)], key="date_range", disabled=False)

# Tabs
tab1, tab2 = st.tabs(["Upload & Predict", "Random Images"])

with tab1:
    st.header("Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            if model:
                # Preprocess the image
                img_array = preprocess_image(img)

                # Perform inference to obtain predictions
                predictions = model.predict(img_array)

                # Get the predicted class label
                predicted_class_index = np.argmax(predictions)
                predicted_class_label = class_labels[predicted_class_index]

                # Display the predicted class label
                st.markdown(f"<div class='prediction-box'>Prediction: {predicted_class_label}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab2:
    st.header("Display Random Images from Dataset")
    if st.button('Display Random Images'):
        # Assuming you have the logic to display random images from your dataset here
        st.write("Displaying random images...")

# Additional Components
st.header("Additional Components")
text_input_value = st.text_input("Enter some text")
slider_value = st.slider("Select a range", 0, 100, (25, 75))
button_clicked = st.button("Click Me")

# Display collected values
st.write("Text Input Value:", text_input_value)
st.write("Slider Value:", slider_value)
st.write("Button Clicked:", button_clicked)

# Table
st.header("Data Table")
sample_data = {
    "Column 1": [1, 2, 3],
    "Column 2": [4, 5, 6]
}
st.table(sample_data)

# Alert Dialog (simulated using st.info, st.warning, etc.)
st.info("This is an info alert")
st.warning("This is a warning alert")
st.error("This is an error alert")

# Badge (simulated using markdown)
st.markdown("<span style='background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 5px;'>Badge Example</span>", unsafe_allow_html=True)

# Link Button
if st.button("Go to GitHub"):
    st.write("[GitHub Repository](https://github.com/iiaaumm/Educational_Toy_Classification)")

# Avatar (simulated using image)
st.image("https://avatars.githubusercontent.com/u/9919?s=280&v=4", width=50, caption="Avatar Example")
