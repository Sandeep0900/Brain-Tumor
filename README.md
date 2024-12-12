## Brain Tumor Detection Model Diagnostic 🧠

### Project Description
This project uses a machine learning model to identify brain tumors based on MRI images. It supports classification into four categories:
1. **Pituitary Tumor**
2. **No Tumor**
3. **Meningioma**
4. **Glioma**

The model is deployed on Streamlit and provides an intuitive interface for users to upload MRI images, view predictions, and analyze probabilities for each class.

### Features
- Upload and test MRI images.
- View predictions with confidence probabilities.
- Intuitive UI hosted on Streamlit.
- Real-time processing and results display.

### Hosted Link
You can access the project [here](https://xngfzntr7ywjztjyiwj8pi.streamlit.app/).

---

### How It Works
1. Upload an MRI image (JPG, PNG, JPEG).
2. The image is preprocessed and passed through a pre-trained model.
3. The model predicts the tumor type or indicates if no tumor is present.
4. Results and probabilities are displayed instantly.

---

### How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Sandeep0900/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the `keras_model.h5` file in the `models` directory.
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. Open the app in your browser at `http://localhost:8501`.

---

### Future Enhancements
- Improve model accuracy with additional data.
- Add more tumor classifications.
- Deploy on additional platforms for broader accessibility.

Feel free to explore, contribute, and give feedback. Let's make diagnostics smarter together! 🚀
