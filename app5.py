import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import pandas as pd  # Import pandas for CSV file operations
from collections import Counter  # Import Counter for counting occurrences
from toy_info import toy_info  
# Page configuration


st.set_page_config(page_title='Educational Toy Classification', page_icon='🎲', layout='wide', initial_sidebar_state='expanded')
# Function to download the model file
def download_model_file(model_url, model_path):
    # Download the model file from the URL
    with requests.get(model_url, stream=True) as response:
        response.raise_for_status()
        with open(model_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Apply the custom style with Google Fonts
st.markdown(header_style, unsafe_allow_html=True)


# URL of the model file in your GitHub repository
model_url = 'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/Toy_classification_10class.h5'

# Path to save the assembled model
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


# Read and apply custom CSS from the external file
with open('styles.css', 'r') as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    
    


# Streamlit app
st.markdown("<p class='header-text'>ລະບົບການຈໍາແນກເຄື່ອງຫຼິ້ນເສີມທັກສະຂອງເດັກນ້ອຍດ້ວຍເຕັກນິກ CNN</p>", unsafe_allow_html=True)
st.markdown("<p class='header-text'>Classification of Children Toys Using CNN</p>", unsafe_allow_html=True)


# Initialize session state for prediction counts and history
if 'prediction_counts' not in st.session_state:
    st.session_state.prediction_counts = Counter()
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load existing prediction history from CSV if it exists
csv_file_path = 'prediction_history.csv'
try:
    st.session_state.prediction_history_df = pd.read_csv(csv_file_path)
    # Check if the 'Count' column exists, else create it
    if 'Count' not in st.session_state.prediction_history_df.columns:
        st.session_state.prediction_history_df['Count'] = 1
        st.session_state.prediction_history_df.to_csv(csv_file_path, index=False)

    st.session_state.prediction_history = st.session_state.prediction_history_df.to_dict('records')
    # Update the Counter object with existing data
    for record in st.session_state.prediction_history:
        # Ensure the 'Count' key exists and is an integer
        if 'Count' in record and isinstance(record['Count'], (int, float)):
            st.session_state.prediction_counts[record['Class Label']] = int(record['Count'])
except FileNotFoundError:
    st.session_state.prediction_history_df = pd.DataFrame(columns=['Class Label', 'Count'])

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main():
    # File uploader for image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.sidebar.image(img, caption='Uploaded Image', use_column_width=True)

            if model:   
                # Preprocess the image
                img_array = preprocess_image(img)

                # Perform inference to obtain predictions
                predictions = model.predict(img_array)
                
                # Get the predicted class label
                predicted_class_index = np.argmax(predictions)
                predicted_class_label = class_labels[predicted_class_index]

                # Update the Counter object with the new prediction
                st.session_state.prediction_counts[predicted_class_label] += 1

                # Check if the prediction already exists in the history
                found = False
                for record in st.session_state.prediction_history:
                    if record['Class Label'] == predicted_class_label:
                        record['Count'] += 1
                        found = True
                        break
                if not found:
                    st.session_state.prediction_history.append({'Class Label': predicted_class_label, 'Count': 1})

                # Convert to DataFrame and save to CSV
                st.session_state.prediction_history_df = pd.DataFrame(st.session_state.prediction_history)
                st.session_state.prediction_history_df.to_csv(csv_file_path, index=False)

                # Display the predicted class label and probabilities
                st.subheader('Prediction')
                st.markdown(f"<p class='prediction-text'>Predicted Class: {predicted_class_label}</p>", unsafe_allow_html=True)    



                # Sample description text
                description_text = toy_info[predicted_class_label]['description']
                image_path = "D:/STUDY/Year4/Development_Toy-A-10class/validation/Puzzle/Image_35.jpg"  # Absolute path to the image

                # Displaying description and image side by side using Streamlit columns
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("<p class='sub-header'>ລາຍລະອຽດກ່ຽວກັບເຄື່ອງຫຼິ້ນ</p>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="description-text">
                            {description_text}
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                
                    st.markdown("<p class='sub-header'>ປະໂຫຍດທີ່ໄດ້ຮັບຈາກການຫຼິ້ນ</p>", unsafe_allow_html=True)
                    benefits = toy_info[predicted_class_label]['benefit']
                    benefits_html = "<div class='description-text'>" + "<ul>" + "".join([f"<li>{benefit}</li>" for benefit in benefits]) + "</ul>" + "</div>"
                    st.markdown(benefits_html, unsafe_allow_html=True)
                
                

                # st.subheader("Recommended Age:")
                #st.markdown(f"<p class='description-text'>{toy_info[predicted_class_label]['recommended_age']}</p>", unsafe_allow_html=True)

                #st.info(toy_info[predicted_class_label]['description'])
                #st.write(f"Recommended Age:")
                #st.success(toy_info[predicted_class_label]['recommended_age'])


                # Display the top 3 predicted classes
                st.subheader('Top 3 Ranking')

                # Prepare the HTML table content
                table_html = "<table class='ranking-table'><thead><tr><th>Rank</th><th>Class</th><th>Count</th></tr></thead><tbody>"
                sorted_predictions = st.session_state.prediction_counts.most_common(3)
                for i, (label, count) in enumerate(sorted_predictions, 1):
                    table_html += f"<tr><td>{i}</td><td>{label}</td><td>{count}</td></tr>"
                table_html += "</tbody></table>"

                # Display the HTML table
                st.markdown(table_html, unsafe_allow_html=True)


                # Display the updated prediction history
                st.subheader('Prediction History')
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")



if __name__ == '__main__':
    main()
