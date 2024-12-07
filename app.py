import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Class labels
CLASS_LABELS = ["Pituitary", "No Tumor", "Meningioma", "Glioma"]

# Tumor descriptions
TUMOR_DESCRIPTIONS = {
    "Pituitary": "A noncancerous tumor in the pituitary gland, often causing hormonal imbalance.",
    "No Tumor": "No sign of a tumor detected.",
    "Meningioma": "A tumor that forms on membranes covering the brain and spinal cord inside the skull.",
    "Glioma": "A type of tumor that occurs in the brain and spinal cord."
}

def load_model():
    try:
        model = tf.keras.models.load_model('models/keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_tumor(model, image):
    try:
        # Process the image
        image = image.convert("RGB")  # Ensure image is in RGB format
        image = image.resize((224, 224))  # Resize to match model input
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(image_array)[0]  # Get prediction probabilities
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_class_index]

        # Calculate prediction percentages
        prediction_percentages = [
            {"class": CLASS_LABELS[i], "percentage": round(prob * 100, 2)}
            for i, prob in enumerate(predictions)
        ]

        return predicted_class, prediction_percentages
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Brain Tumor Detection", page_icon=":medical_symbol:")

    # Title
    st.title("🧠 Brain Tumor Detection")
    st.write("Upload a brain MRI image for tumor classification")

    # Load the model
    model = load_model()
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a medical MRI scan image"
    )

    # Display upload and prediction section
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Prediction button
        if st.button("Detect Tumor"):
            with st.spinner('Analyzing image...'):
                # Perform prediction
                predicted_class, prediction_percentages = predict_tumor(model, image)

                if predicted_class:
                    # Display results
                    st.success(f"Detected Tumor Type: {predicted_class}")
                    
                    # Description
                    st.info(f"Description: {TUMOR_DESCRIPTIONS[predicted_class]}")
                    
                    # Prediction Percentages
                    st.subheader("Prediction Breakdown")
                    for pred in prediction_percentages:
                        st.progress(
                            pred['percentage'] / 100, 
                            text=f"{pred['class']}: {pred['percentage']}%"
                        )

    # Sidebar information
    st.sidebar.header("About the App")
    st.sidebar.info(
        "This app uses a deep learning model to classify brain tumors "
        "from MRI images into four categories: Pituitary, No Tumor, "
        "Meningioma, and Glioma."
    )

# Run the app
if __name__ == "__main__":
    main()
