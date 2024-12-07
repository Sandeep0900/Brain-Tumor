import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D

def custom_load_model(model_path):
    try:
        # Custom objects dictionary to handle potential compatibility issues
        custom_objects = {
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
            'tf': tf
        }
        
        # Attempt to load with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        
        # Attempt to recreate the model if loading fails
        try:
            # Load the model weights
            original_model = load_model(model_path, compile=False)
            
            # Create a new model with similar architecture
            new_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                # Recreate the model layers here
                # You'll need to manually add layers based on your original model architecture
            ])
            
            # Copy weights
            new_model.set_weights(original_model.get_weights())
            
            return new_model
        
        except Exception as reconstruction_error:
            st.error(f"Model reconstruction failed: {reconstruction_error}")
            return None

def preprocess_image(image):
    # Standardized image preprocessing
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title("Brain Tumor Detection Model Diagnostic")
    
    # Model loading
    st.write("Attempting to load the model...")
    model = custom_load_model('models/keras_model.h5')
    
    if model is None:
        st.error("Could not load the model. Please check the model file.")
        return
    
    # Model summary
    st.write("Model Summary:")
    model.summary(print_fn=st.write)
    
    # Test prediction capability
    st.write("\nTesting Model Prediction Capability:")
    test_image_path = st.file_uploader("Upload a test image", type=['jpg', 'png', 'jpeg'])
    
    if test_image_path is not None:
        from PIL import Image
        
        # Open and display the image
        image = Image.open(test_image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Predict
        try:
            predictions = model.predict(processed_image)
            st.write("Prediction Probabilities:")
            st.write(predictions)
        except Exception as pred_error:
            st.error(f"Prediction failed: {pred_error}")

if __name__ == "__main__":
    main()
