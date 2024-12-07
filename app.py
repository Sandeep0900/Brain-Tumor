from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/keras_model.h5')

# Class labels
CLASS_LABELS = ["Pituitary", "No Tumor", "Meningioma", "Glioma"]

# Tumor descriptions
TUMOR_DESCRIPTIONS = {
    "Pituitary": "A noncancerous tumor in the pituitary gland, often causing hormonal imbalance.",
    "No Tumor": "No sign of a tumor detected.",
    "Meningioma": "A tumor that forms on membranes covering the brain and spinal cord inside the skull.",
    "Glioma": "A type of tumor that occurs in the brain and spinal cord."
}

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image_url = None
    if request.method == "POST":
        # Check if the file is uploaded
        if 'image' not in request.files:
            return render_template("index.html", result="No file uploaded.", description="Please upload an image.")

        file = request.files['image']
        if file.filename == '':
            return render_template("index.html", result="No file selected.", description="Please upload an image.")

        try:
            # Save and display the uploaded image
            file_path = f"static/uploads/{file.filename}"
            file.save(file_path)
            uploaded_image_url = url_for('static', filename=f"uploads/{file.filename}")

            # Process the image
            image = Image.open(file).convert("RGB")  # Ensure image is in RGB format
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

            # Pass result to template
            return render_template(
                "index.html",
                result=predicted_class,
                description=TUMOR_DESCRIPTIONS[predicted_class],
                prediction_percentages=prediction_percentages,
                uploaded_image_url=uploaded_image_url
            )

        except Exception as e:
            return render_template("index.html", result="Error processing image.", description=str(e))

    # Default landing page
    return render_template("index.html", result=None, description=None)

if __name__ == "__main__":
    app.run(debug=True)
