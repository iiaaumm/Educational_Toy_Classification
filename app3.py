import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main():
    # Page title
    st.title("Educational Toy Classification")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Path to the model file
    model_path = './Toy_classification_10class.h5'

    # Loading the model
    model = load_saved_model(model_path)

    # List of class labels
    class_labels = ['Activity_Cube', 'Ball', 'Puzzle', 'Rubik', 'Tricycle', 'baby_walker', 'lego', 'poppet', 'rattle', 'stacking']

    # Toy descriptions and recommended age groups
    toy_info = {
        'Activity_Cube': {
            'description': 'An activity cube is a multi-functional toy designed to stimulate and entertain infants and toddlers with various activities.',
            'recommended_age': 'Early Childhood (Ages 1-3)'
        },
        'Ball': {
            'description': 'A ball is a round object used in games and sports. It is one of the simplest toys.',
            'recommended_age': 'Early Childhood (Ages 1-6)'
        },
        'Puzzle': {
            'description': 'A puzzle is a game or problem designed to test ingenuity or knowledge.',
            'recommended_age': 'Early Childhood (Ages 3-10)'
        },
        'Rubik': {
            'description': 'A Rubik\'s Cube is a 3D combination puzzle invented in 1974 by Hungarian sculptor and professor of architecture Ernő Rubik.',
            'recommended_age': 'Early Childhood (Ages 6-10)'
        },
        'Tricycle': {
            'description': 'A tricycle is a three-wheeled vehicle that is propelled by the rider pushing pedals.',
            'recommended_age': 'Early Childhood (Ages 2-5)'
        },
        'baby_walker': {
            'description': 'A baby walker is a device designed for young children to sit in and move themselves with their feet while walking.',
            'recommended_age': 'Early Childhood (Ages 6-12 months)'
        },
        'lego': {
            'description': 'LEGO toys are construction toys that consist of interlocking plastic bricks and an accompanying array of gears, minifigures, and various other parts.',
            'recommended_age': 'Early Childhood (Ages 3-16)'
        },
        'poppet': {
            'description': 'A poppet is a small figure of a human being used in sorcery and witchcraft.',
            'recommended_age': 'Early Childhood (Ages 5-10)'
        },
        'rattle': {
            'description': 'A rattle is a baby toy that makes a noise when shaken.',
            'recommended_age': 'Early Childhood (Ages 0-2)'
        },
        'stacking': {
            'description': 'Stacking toys are a set of rings or blocks of various sizes that can be fitted one on top of the other.',
            'recommended_age': 'Early Childhood (Ages 1-5)'
        }
    }

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

                # Display the predicted class label and probabilities
                st.subheader('Prediction')
                st.write(f"Predicted Class: {predicted_class_label}")
                st.write(f"Description:")
                st.info(toy_info[predicted_class_label]['description'])
                st.write(f"Recommended Age:")
                st.success(toy_info[predicted_class_label]['recommended_age'])

                # Display probabilities for each class
                st.subheader('Class Probabilities')
                prob_chart = {
                    'Class Label': class_labels,
                    'Probability': predictions[0]
                }
                prob_chart_data = np.array(prob_chart['Probability'])
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size
                ax.bar(class_labels, prob_chart_data)
                ax.set_xlabel('Class Labels')
                ax.set_ylabel('Probability')
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Confusion Matrix (Placeholder)
                st.subheader('Confusion Matrix')
                cm = np.random.randint(0, 50, (10, 10))  # Replace with actual confusion matrix data
                plt.figure(figsize=(8, 6))  # Adjusted size
                sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(plt)

                # Class Distribution (Placeholder)
                st.subheader('Class Distribution')
                class_counts = np.random.randint(10, 50, 10)  # Replace with actual class distribution data
                plt.figure(figsize=(8, 6))  # Adjusted size
                sns.barplot(x=class_labels, y=class_counts)
                plt.xlabel('Class Labels')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(plt)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
